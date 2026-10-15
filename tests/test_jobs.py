"""Job queue (aerosplat.jobs): claiming, the GPU slot, cancel, orphans, and real workers."""

import threading
import time

import pytest
from pydantic import ValidationError

from aerosplat.jobs import (DiagnosticSpec, EventType, JobKind, JobQueue, JobStatus, PipelineSpec,
                            Resource)
from aerosplat.jobs.worker import Worker
from aerosplat.store import RunStatus


@pytest.fixture
def q(catalog):
    return JobQueue(catalog.engine)


def _diag(q, resource=Resource.cpu, priority=0, **spec):
    return q.enqueue(JobKind.diagnostic, DiagnosticSpec(**spec).model_dump(),
                     resource=resource, priority=priority)


def test_enqueue_claim_finish(q):
    job = _diag(q)
    assert job.status == JobStatus.queued and q.counts()["queued"] == 1
    got = q.claim("w1", Resource.cpu)
    assert got is not None and got.id == job.id
    assert got.status == JobStatus.running and got.worker_id == "w1" and got.started_at
    assert q.claim("w2", Resource.cpu) is None          # nothing else queued
    done = q.finish(job.id, JobStatus.succeeded)
    assert done.status == JobStatus.succeeded and done.finished_at
    states = [e.data["status"] for e in q.events(job.id) if e.type == EventType.status]
    assert states == ["queued", "running", "succeeded"]
    seqs = [e.seq for e in q.events(job.id)]
    assert seqs == sorted(seqs) and q.events(job.id, after=seqs[-1]) == []


def test_finish_needs_terminal_status(q):
    job = _diag(q)
    with pytest.raises(ValueError):
        q.finish(job.id, JobStatus.running)


def test_gpu_slot_is_exclusive(q):
    a, b = _diag(q, Resource.gpu), _diag(q, Resource.gpu)
    first = q.claim("gpu-1", Resource.gpu)
    assert first is not None and first.id == a.id
    assert q.claim("gpu-2", Resource.gpu) is None        # second GPU worker must wait
    q.finish(a.id, JobStatus.failed, error="x")
    second = q.claim("gpu-2", Resource.gpu)
    assert second is not None and second.id == b.id


def test_cpu_jobs_run_in_parallel_and_resources_are_separate(q):
    _diag(q, Resource.cpu), _diag(q, Resource.cpu)
    g = _diag(q, Resource.gpu)
    assert q.claim("c1", Resource.cpu) and q.claim("c2", Resource.cpu)
    assert q.claim("c3", Resource.cpu) is None
    assert q.claim("g1", Resource.gpu).id == g.id


def test_priority_then_fifo(q):
    low = _diag(q, priority=0)
    high = _diag(q, priority=5)
    low2 = _diag(q, priority=0)
    order = [q.claim("w", Resource.cpu).id for _ in range(3)]
    assert order == [high.id, low.id, low2.id]


def test_cancel_queued_and_running(q):
    queued = _diag(q)
    assert q.request_cancel(queued.id).status == JobStatus.cancelled
    assert q.claim("w", Resource.cpu) is None            # cancelled jobs are never claimed
    running = _diag(q)
    q.claim("w", Resource.cpu)
    flagged = q.request_cancel(running.id)
    assert flagged.status == JobStatus.running and flagged.cancel_requested
    assert q.cancel_requested(running.id)
    assert q.finish(running.id, JobStatus.cancelled).status == JobStatus.cancelled
    # cancelling a finished job is a no-op
    assert q.request_cancel(running.id).status == JobStatus.cancelled


def test_result_merge_emits_event(q):
    job = _diag(q)
    q.merge_result(job.id, {"run_id": "r1"})
    q.merge_result(job.id, {"site": "demo/s"})
    assert q.get(job.id).result == {"run_id": "r1", "site": "demo/s"}
    assert [e.data for e in q.events(job.id) if e.type == EventType.result][-1] == {"site": "demo/s"}


def test_workers_heartbeat_and_orphans(q):
    q.register_worker("h:gpu:1", Resource.gpu, "h", 1)
    assert [w.online for w in q.workers()] == [True]
    job = _diag(q, Resource.gpu)
    q.claim("h:gpu:1", Resource.gpu)
    assert q.recover_orphans() == []                     # worker alive and fresh
    q.stop_worker("h:gpu:1")
    assert q.workers() == [] and q.workers(include_stopped=True)[0].stopped_at
    assert q.recover_orphans() == [job.id]
    lost = q.get(job.id)
    assert lost.status == JobStatus.failed and "worker lost" in lost.error


def test_pipeline_spec_validation_and_argv():
    spec = PipelineSpec(site="demo/es", capture="c1", from_stage="sfm", to_stage="train",
                        overrides={"iterations": 100, "init_scale": None})
    argv = spec.argv("/p")
    assert argv[:4] == ["--site", "demo/es", "--epoch", "baseline"]
    assert ["--from", "sfm"] == argv[argv.index("--from"):argv.index("--from") + 2]
    assert "--projects-root" in argv and "iterations=100" in argv and "init_scale=none" in argv
    for bad in ({"stages": ["nope"]}, {"stages": []}, {"from_stage": "train", "to_stage": "sfm"},
                {"stages": ["geo"], "from_stage": "geo"}, {"overrides": {"nope": 1}},
                {"overrides": {"iterations": 0}}, {"overrides": {"dataset": "x"}}):
        with pytest.raises(ValidationError):
            PipelineSpec(site="demo/es", **bad)


# ── real workers (child processes) ───────────────────────────────────────
def _worker(catalog, resource=Resource.cpu):
    return Worker(catalog.paths.root, resource, poll_s=0.05)


def test_worker_runs_diagnostic_job(catalog, q):
    job = _diag(q, stages=2, seconds_per_stage=0, lines_per_stage=2)
    _worker(catalog).run(once=True)
    done = q.get(job.id)
    assert done.status == JobStatus.succeeded, done.error
    events = q.events(job.id)
    stages = [(e.data["stage"], e.data["state"]) for e in events if e.type == EventType.stage]
    assert stages == [("step1", "started"), ("step1", "done"), ("step2", "started"),
                      ("step2", "done")]
    logs = [ln for e in events if e.type == EventType.log for ln in e.data["lines"]]
    assert "   step2: line 2/2" in logs
    assert all(w.stopped_at for w in q.workers(include_stopped=True))


def test_worker_records_failure_with_output_tail(catalog, q):
    job = _diag(q, stages=2, seconds_per_stage=0, lines_per_stage=1, fail_at=2)
    _worker(catalog).run(once=True)
    failed = q.get(job.id)
    assert failed.status == JobStatus.failed
    assert "exit code 1" in failed.error and "requested failure" in failed.error


def test_worker_cancels_running_job_tree(catalog, q):
    job = _diag(q, stages=1, seconds_per_stage=60, lines_per_stage=60)
    worker = _worker(catalog)
    t = threading.Thread(target=worker.run, kwargs={"once": True}, daemon=True)
    t.start()
    deadline = time.time() + 30
    while time.time() < deadline and not any(e.type == EventType.log for e in q.events(job.id)):
        time.sleep(0.1)
    assert q.get(job.id).status == JobStatus.running
    q.request_cancel(job.id)
    t.join(timeout=30)
    assert not t.is_alive()
    assert q.get(job.id).status == JobStatus.cancelled


def test_worker_runs_pipeline_job_end_to_end(es_site):
    cat, site, cap, root = es_site
    q = JobQueue(cat.engine)
    spec = PipelineSpec(site=site.key, capture=cap.id, stages=["geo"], run_id="r-job")
    job = q.enqueue(JobKind.pipeline, spec.model_dump(), resource=Resource.gpu, site_key=site.key)
    Worker(root, Resource.gpu, poll_s=0.05).run(once=True)
    done = q.get(job.id)
    assert done.status == JobStatus.succeeded, done.error
    assert done.result == {"site": "demo/es", "epoch": "baseline", "run_id": "r-job"}
    run = cat.get_run(site, "baseline", "r-job")
    assert run.status == RunStatus.ok and "known_poses" in run.products
    stage_events = [e.data for e in q.events(job.id) if e.type == EventType.stage]
    assert [(d["stage"], d["state"]) for d in stage_events] == [("geo", "started"), ("geo", "done")]
    assert stage_events[-1]["total"] == 1 and stage_events[-1]["seconds"] >= 0


def test_failed_pipeline_job_closes_its_run(es_site, monkeypatch):
    cat, site, cap, root = es_site
    q = JobQueue(cat.engine)
    # sfm needs COLMAP; a bogus binary makes the stage fail after the run is created.
    spec = PipelineSpec(site=site.key, capture=cap.id, stages=["geo", "sfm"], run_id="r-bad",
                        overrides={"colmap_bin": "definitely-not-colmap"})
    job = q.enqueue(JobKind.pipeline, spec.model_dump(), resource=Resource.gpu, site_key=site.key)
    Worker(root, Resource.gpu, poll_s=0.05).run(once=True)
    assert q.get(job.id).status == JobStatus.failed
    assert cat.get_run(site, "baseline", "r-bad").status == RunStatus.failed
