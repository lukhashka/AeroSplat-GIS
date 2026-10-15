"""`aerosplat worker`: claim jobs of one resource class and run each in a child process.

    aerosplat worker                 # GPU slot (pipeline runs)
    aerosplat worker --resource cpu  # CPU jobs
    aerosplat worker --once          # drain the queue, then exit (CI, scripts)

The child (aerosplat.jobs.child) owns the job; the worker only streams its output
into the event log, watches for cancellation, keeps a heartbeat and records the
outcome. Killing the child (cancel, OOM, crash) never takes the worker down.
"""

from __future__ import annotations

import argparse
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import IO, Optional

from aerosplat.jobs.models import EventType, Job, JobKind, JobStatus, Resource
from aerosplat.jobs.proctree import group_kwargs, kill_tree, pid_alive
from aerosplat.jobs.queue import JobQueue
from aerosplat.logging_utils import fix_console_encoding, get_logger
from aerosplat.store.catalog import Catalog, CatalogError
from aerosplat.store.models import RunStatus
from aerosplat.store.paths import default_projects_root

log = get_logger("aerosplat.worker")

LOG_FLUSH_S = 0.5
LOG_BATCH = 500
CANCEL_POLL_S = 1.0
HEARTBEAT_S = 5.0
ERROR_TAIL = 20


def _pump(stream: IO[str], out: "queue.Queue[Optional[str]]") -> None:
    for line in iter(stream.readline, ""):
        out.put(line.rstrip("\r\n"))
    out.put(None)


class Worker:
    def __init__(self, projects_root: Path, resource: Resource, *, poll_s: float = 1.0,
                 python: str = sys.executable):
        self.projects_root = Path(projects_root)
        self.resource = resource
        self.poll_s = poll_s
        self.python = python
        self.host = socket.gethostname()
        self.id = f"{self.host}:{resource.value}:{os.getpid()}"
        self.catalog = Catalog.open(self.projects_root)
        self.q = JobQueue(self.catalog.engine)
        self._last_beat = 0.0

    # ── lifecycle ────────────────────────────────────────────────────────
    def _beat(self, current: Optional[str], force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self._last_beat >= HEARTBEAT_S:
            self.q.heartbeat(self.id, current)
            self._last_beat = now

    def _reap_local(self) -> None:
        """Mark this host's dead workers stopped, then fail the jobs they left running."""
        for w in self.q.workers():
            if w.host == self.host and w.id != self.id and not pid_alive(w.pid):
                self.q.stop_worker(w.id)
        for job_id in self.q.recover_orphans():
            log.warning(f"⚠️  job {job_id}: worker lost, marked failed")
            self._close_run(self.q.get(job_id))

    def run(self, *, once: bool = False, max_jobs: Optional[int] = None) -> int:
        self.q.register_worker(self.id, self.resource, self.host, os.getpid())
        self._beat(None, force=True)
        log.info(f"👷 worker {self.id} — {self.projects_root}")
        done = 0
        try:
            while max_jobs is None or done < max_jobs:
                self._reap_local()
                job = self.q.claim(self.id, self.resource)
                if job is None:
                    if once:
                        break
                    self._beat(None)
                    time.sleep(self.poll_s)
                    continue
                self.execute(job)
                done += 1
        except KeyboardInterrupt:
            log.info("⏹  worker interrupted")
        finally:
            self.q.stop_worker(self.id)
            self.catalog.close()
        return 0

    # ── one job ──────────────────────────────────────────────────────────
    def _spawn(self, job: Job) -> subprocess.Popen:
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
               "PYTHONUTF8": "1", "AEROSPLAT_JOB_ID": job.id}
        cmd = [self.python, "-m", "aerosplat.jobs.child",
               "--projects-root", str(self.projects_root), "--job", job.id]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, text=True, encoding="utf-8",
                                errors="replace", bufsize=1, env=env, **group_kwargs())

    def execute(self, job: Job) -> JobStatus:
        log.info(f"▶  job {job.id} [{job.kind.value}] {job.title}")
        self._beat(job.id, force=True)
        try:
            proc = self._spawn(job)
        except OSError as e:
            return self.q.finish(job.id, JobStatus.failed, error=f"cannot start job process: {e}").status

        lines: "queue.Queue[Optional[str]]" = queue.Queue()
        assert proc.stdout is not None
        reader = threading.Thread(target=_pump, args=(proc.stdout, lines), daemon=True)
        reader.start()

        tail: deque[str] = deque(maxlen=ERROR_TAIL)
        pending: list[str] = []
        cancelled = False
        eof = False
        last_flush = last_cancel_check = time.monotonic()

        def flush() -> None:
            nonlocal last_flush
            while pending:
                chunk = pending[:LOG_BATCH]
                del pending[:LOG_BATCH]
                self.q.add_event(job.id, EventType.log, {"lines": chunk})
            last_flush = time.monotonic()

        while not eof:
            try:
                line = lines.get(timeout=0.2)
            except queue.Empty:
                line = ""
            else:
                if line is None:
                    eof = True
                else:
                    pending.append(line)
                    if line.strip():
                        tail.append(line)
            now = time.monotonic()
            if pending and (now - last_flush >= LOG_FLUSH_S or len(pending) >= LOG_BATCH):
                flush()
            if not cancelled and now - last_cancel_check >= CANCEL_POLL_S:
                last_cancel_check = now
                if self.q.cancel_requested(job.id):
                    cancelled = True
                    pending.append("⏹  cancel requested — stopping the job process tree")
                    kill_tree(proc)
            self._beat(job.id)

        rc = proc.wait()
        reader.join(timeout=5)
        flush()

        if cancelled:
            status, error = JobStatus.cancelled, None
        elif rc == 0:
            status, error = JobStatus.succeeded, None
        else:
            status = JobStatus.failed
            error = f"exit code {rc}" + ("\n" + "\n".join(tail) if tail else "")
        final = self.q.finish(job.id, status, error=error)
        if status != JobStatus.succeeded:
            self._close_run(final)
        self._beat(None, force=True)
        log.info(f"{'✅' if status == JobStatus.succeeded else '❌'} job {job.id} → {status.value}")
        return status

    def _close_run(self, job: Job) -> None:
        """A killed job process cannot close its catalog run; do it here."""
        if job.kind != JobKind.pipeline or not job.result.get("run_id"):
            return
        try:
            site = self.catalog.find_site(job.result["site"])
            epoch, run_id = job.result.get("epoch", "baseline"), job.result["run_id"]
            if self.catalog.get_run(site, epoch, run_id).status == RunStatus.running:
                self.catalog.finish_run(site, epoch, run_id, status=RunStatus.failed)
        except (CatalogError, KeyError) as e:
            log.warning(f"⚠️  job {job.id}: could not close run: {e}")


def main(argv: Optional[list[str]] = None) -> int:
    fix_console_encoding()
    ap = argparse.ArgumentParser(prog="aerosplat worker",
                                 description="Виконавець задач з черги (ADR-0010)")
    ap.add_argument("--resource", choices=[r.value for r in Resource], default="gpu")
    ap.add_argument("--projects-root", type=Path, default=None,
                    help="за замовчуванням $AEROSPLAT_PROJECTS або <repo>/projects")
    ap.add_argument("--once", action="store_true", help="виконати чергу й завершитись")
    ap.add_argument("--poll", type=float, default=1.0, help="інтервал опитування черги, с")
    args = ap.parse_args(argv)
    worker = Worker(args.projects_root or default_projects_root(), Resource(args.resource),
                    poll_s=args.poll)
    return worker.run(once=args.once)


if __name__ == "__main__":
    sys.exit(main())
