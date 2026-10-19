"""Uploaded captures ingested by a CPU job (aerosplat.ingest.upload): crash, cancel, re-run.

The invariant under test: a capture is listed only when it is complete, and a job killed
at any point can simply be run again.
"""

import io
import json
import time

import pytest
from PIL import Image as PILImage

pytest.importorskip("httpx")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from aerosplat.ingest import upload as up  # noqa: E402
from aerosplat.ingest.images import sha256_file  # noqa: E402
from aerosplat.jobs import IngestSpec, JobKind, JobQueue, JobStatus, Resource  # noqa: E402
from aerosplat.jobs.worker import Worker  # noqa: E402
from aerosplat.server import api_v2  # noqa: E402

ORIGIN = {"lat": 40.8942, "lon": -74.2173, "h": 233.0}


class Killed(BaseException):
    """Stands in for the process being killed: no handler in the job sees it."""


@pytest.fixture
def root(tmp_path, monkeypatch):
    r = tmp_path / "projects"
    monkeypatch.setenv("AEROSPLAT_PROJECTS", str(r))
    yield r
    for cat in api_v2._CATALOGS.values():
        cat.close()
    api_v2._CATALOGS.clear()


@pytest.fixture
def client(root):
    app = FastAPI()
    app.include_router(api_v2.router)
    with TestClient(app) as c:
        assert c.post("/api/v2/projects", json={"id": "demo"}).status_code == 201
        assert c.post("/api/v2/projects/demo/sites",
                      json={"id": "pier", "origin": ORIGIN}).status_code == 201
        yield c


def _jpeg(shade):
    buf = io.BytesIO()
    PILImage.new("RGB", (40, 30), (shade, 90, 20)).save(buf, "JPEG")
    return buf.getvalue()


def _upload(client, capture_id="cap", n=3):
    files = [("files", (f"f{i}.jpg", _jpeg(30 * i), "image/jpeg")) for i in range(n)]
    r = client.post("/api/v2/sites/demo/pier/captures",
                    data={"capture_id": capture_id, "source": "drone"}, files=files)
    assert r.status_code == 202, r.text
    return r.json()


def _ctx():
    cat = api_v2.get_catalog()
    return cat, JobQueue(cat.engine), cat.get_site("demo", "pier")


def _spec(job):
    return IngestSpec(**job["payload"])


def _listed(client):
    return [c["capture"]["id"] for c in client.get("/api/v2/sites/demo/pier").json()["captures"]]


def _events(q, job_id):
    return [(e.data["stage"], e.data["state"]) for e in q.events(job_id) if e.type.value == "stage"]


def test_kill_while_hashing_then_rerun(client):
    job = _upload(client)
    cat, q, site = _ctx()
    staging = up.staging_dir(cat, site, _spec(job).staging)
    originals = {p.name: sha256_file(p) for p in (staging / up.FILES).iterdir()}

    def dying(t, d):
        q.add_event(job["id"], t, d)
        if d.get("progress", {}).get("done") == 1:
            raise Killed()

    with pytest.raises(Killed):
        up.run_ingest(cat, _spec(job), job["id"], dying)
    assert _listed(client) == []                                  # never half visible
    assert not cat.site_dirs(site).capture("cap").root.exists()   # not even in place
    assert (staging / up.BUILD).exists()                          # half-built, inside staging

    res = up.run_ingest(cat, _spec(job), job["id"], lambda t, d: q.add_event(job["id"], t, d))
    assert res["images"] == 3 and _listed(client) == ["cap"]
    imgs = {i.name: i.sha256 for i in cat.list_images(site, "cap")}
    assert imgs == originals                                       # every file, intact
    assert not staging.exists()


def test_kill_between_rename_and_catalog_commit_then_rerun(client, monkeypatch):
    job = _upload(client)
    cat, q, site = _ctx()
    real = cat.add_capture

    def die_once(*a, **kw):
        monkeypatch.setattr(cat, "add_capture", real)
        raise Killed()

    monkeypatch.setattr(cat, "add_capture", die_once)
    with pytest.raises(Killed):
        up.run_ingest(cat, _spec(job), job["id"], lambda t, d: None)
    final = cat.site_dirs(site).capture("cap").root
    assert final.is_dir() and (final / up.MARKER).is_file()
    assert _listed(client) == []                                   # in place but not a capture
    blocked = client.post("/api/v2/sites/demo/pier/captures", data={"capture_id": "cap",
                          "source": "drone"}, files=[("files", ("a.jpg", _jpeg(1), "image/jpeg"))])
    assert blocked.status_code == 409

    res = up.run_ingest(cat, _spec(job), job["id"], lambda t, d: q.add_event(job["id"], t, d))
    assert res["images"] == 3 and _listed(client) == ["cap"]
    assert ("hash", "skipped") in _events(q, job["id"])
    assert json.loads((final / up.MARKER).read_text()) == {"job": job["id"], "registered": True}


def test_rerun_of_a_finished_job_is_a_no_op(client):
    job = _upload(client)
    cat, q, site = _ctx()
    first = up.run_ingest(cat, _spec(job), job["id"], lambda t, d: None)
    again = up.run_ingest(cat, _spec(job), job["id"], lambda t, d: None)
    assert first == again and _listed(client) == ["cap"]
    # ... but another job for the same id is refused, not merged
    other = _spec(job).model_copy(update={"staging": up.staging_name("f" * 32)})
    with pytest.raises(up.IngestError, match="already exists"):
        up.run_ingest(cat, other, "f" * 32, lambda t, d: None)


def test_missing_or_unsealed_staging_fails_without_a_capture(client):
    job = _upload(client)
    cat, q, site = _ctx()
    staging = up.staging_dir(cat, site, _spec(job).staging)
    (staging / up.SEAL).unlink()
    with pytest.raises(up.IngestError, match="incomplete"):
        up.run_ingest(cat, _spec(job), job["id"], lambda t, d: q.add_event(job["id"], t, d))
    assert ("check", "failed") in _events(q, job["id"])
    import shutil

    shutil.rmtree(staging)
    with pytest.raises(up.IngestError, match="missing"):
        up.run_ingest(cat, _spec(job), job["id"], lambda t, d: None)
    assert _listed(client) == []


def test_a_foreign_capture_directory_is_left_alone(client):
    job = _upload(client)
    cat, q, site = _ctx()
    final = cat.site_dirs(site).capture("cap").root
    final.mkdir(parents=True)
    (final / "keep.txt").write_text("someone else's")
    with pytest.raises(up.IngestError, match="not created by this upload"):
        up.run_ingest(cat, _spec(job), job["id"], lambda t, d: None)
    assert (final / "keep.txt").read_text() == "someone else's"
    assert sorted(p.name for p in final.iterdir()) == ["keep.txt"]
    assert up.staging_dir(cat, site, _spec(job).staging).is_dir()   # the upload is not lost


def test_cancelled_upload_is_swept_later_and_active_ones_are_kept(client):
    cancelled = _upload(client, "c1")
    active = _upload(client, "c2")
    cat, q, site = _ctx()
    assert client.post(f"/api/v2/jobs/{cancelled['id']}/cancel").json()["status"] == "cancelled"
    captures = cat.site_dirs(site).captures
    orphan_new = captures / up.staging_name("a" * 32)
    orphan_old = captures / up.staging_name("b" * 32)
    for d in (orphan_new, orphan_old):
        (d / up.FILES).mkdir(parents=True)
        (d / up.FILES / "x.jpg").write_bytes(b"x")

    now = time.time()
    assert up.sweep_stale_uploads(cat, site, q, now=now) == []     # all too fresh
    later = now + up.KEEP_ORPHAN_S + 10
    import os

    os.utime(orphan_new / up.FILES / "x.jpg", (later, later))       # still being written
    removed = up.sweep_stale_uploads(cat, site, q, now=later)
    assert sorted(removed) == sorted([up.staging_name(cancelled["id"]), orphan_old.name])
    assert up.staging_dir(cat, site, _spec(active).staging).is_dir()
    assert orphan_new.is_dir()


def test_spec_and_staging_names_cannot_escape_the_captures_directory(client):
    cat, q, site = _ctx()
    for bad in ("../../x", ".upload-../x", "captures", ".upload-" + "g" * 32):
        with pytest.raises(ValidationError):
            IngestSpec(site="demo/pier", capture_id="c", source="drone", staging=bad)
    for bad in ("../x", ".upload-a/../../b", "x"):
        with pytest.raises(up.IngestError):
            up.staging_dir(cat, site, bad)


def test_real_cpu_worker_runs_the_ingest_job(client, root):
    job = _upload(client, n=2)
    worker = Worker(root, Resource.cpu, poll_s=0.05)
    assert worker.run(once=True) == 0
    _, q, _ = _ctx()
    done = q.get(job["id"])
    assert done.kind == JobKind.ingest and done.status == JobStatus.succeeded, done.error
    assert done.result["images"] == 2 and _listed(client) == ["cap"]
    log = [ln for e in q.events(job["id"]) if e.type.value == "log" for ln in e.data["lines"]]
    assert any("2/2" in ln for ln in log)
