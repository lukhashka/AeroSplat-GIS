"""Job process: `python -m aerosplat.jobs.child --projects-root R --job ID`.

Started by a worker (aerosplat.jobs.worker) for exactly one job. Stage progress
goes straight to the job's event log; stdout/stderr are read by the worker and
stored as log events. Exit code 0 = succeeded, anything else = failed.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from aerosplat.jobs.models import DiagnosticSpec, EventType, Job, JobKind, PipelineSpec
from aerosplat.jobs.queue import JobQueue
from aerosplat.logging_utils import fix_console_encoding
from aerosplat.store.catalog import Catalog


def run_pipeline(q: JobQueue, job: Job, projects_root: Path) -> int:
    from aerosplat import starter

    spec = PipelineSpec(**job.payload)

    def on_run(site_key: str, run_id: str) -> None:
        q.merge_result(job.id, {"site": site_key, "epoch": spec.epoch, "run_id": run_id})

    def on_event(ev: dict[str, Any]) -> None:
        q.add_event(job.id, EventType.stage, ev)

    try:
        return starter.main(spec.argv(str(projects_root)), on_event=on_event, on_run=on_run)
    except SystemExit as e:
        if e.code in (None, 0):
            return 0
        if not isinstance(e.code, int):
            print(e.code, file=sys.stderr, flush=True)
        return 1


def run_diagnostic(q: JobQueue, job: Job) -> int:
    spec = DiagnosticSpec(**job.payload)
    for i in range(1, spec.stages + 1):
        name = f"step{i}"
        base = {"stage": name, "label": f"Diagnostic step {i}/{spec.stages}",
                "index": i, "total": spec.stages}
        q.add_event(job.id, EventType.stage, {**base, "state": "started"})
        print(f"▶  {base['label']}", flush=True)
        t0 = time.perf_counter()
        for n in range(spec.lines_per_stage):
            print(f"   {name}: line {n + 1}/{spec.lines_per_stage}", flush=True)
            time.sleep(spec.seconds_per_stage / max(1, spec.lines_per_stage))
        if spec.lines_per_stage == 0:
            time.sleep(spec.seconds_per_stage)
        dt = round(time.perf_counter() - t0, 2)
        if spec.fail_at == i:
            q.add_event(job.id, EventType.stage,
                        {**base, "state": "failed", "seconds": dt, "error": "requested failure"})
            print(f"❌ {name}: requested failure", file=sys.stderr, flush=True)
            return 1
        q.add_event(job.id, EventType.stage, {**base, "state": "done", "seconds": dt})
        print(f"✅ {name} — {dt}s", flush=True)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    fix_console_encoding()
    ap = argparse.ArgumentParser(prog="aerosplat.jobs.child")
    ap.add_argument("--projects-root", type=Path, required=True)
    ap.add_argument("--job", required=True)
    args = ap.parse_args(argv)

    catalog = Catalog.open(args.projects_root)
    q = JobQueue(catalog.engine)
    try:
        job = q.get(args.job)
        if job.kind == JobKind.pipeline:
            return run_pipeline(q, job, args.projects_root)
        if job.kind == JobKind.diagnostic:
            return run_diagnostic(q, job)
        print(f"unknown job kind {job.kind}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        catalog.close()


if __name__ == "__main__":
    sys.exit(main())
