"""SQLite-backed job queue in the catalog DB (TARGET_ARCHITECTURE §8).

    q = JobQueue(catalog.engine)
    job = q.enqueue(JobKind.pipeline, spec.model_dump(), resource=Resource.gpu, site_key="demo/s1")
    job = q.claim("host:1234", Resource.gpu)       # worker side; None when nothing to do

Claiming is a single conditional UPDATE, so two workers can never take the same
job, and the GPU check sits inside that statement: at most one `gpu` job is
`running` at any moment, however many GPU workers are started.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from datetime import timedelta
from typing import Any, Optional

from sqlalchemy import Engine, and_, exists, func, select, update
from sqlalchemy.orm import Session, aliased, sessionmaker

from aerosplat.jobs.models import (TERMINAL, EventType, Job, JobEvent, JobKind, JobStatus,
                                   Resource, WorkerInfo)
from aerosplat.store.db import JobEventRow, JobRow, WorkerRow
from aerosplat.store.models import utcnow

WORKER_ONLINE_S = 15.0     # heartbeat younger than this -> worker shown as online
ORPHAN_AFTER_S = 45.0      # running job whose worker is silent this long -> failed


class JobNotFoundError(KeyError):
    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


def new_job_id() -> str:
    return uuid.uuid4().hex


def _job(r: JobRow) -> Job:
    return Job(id=r.id, kind=JobKind(r.kind), resource=Resource(r.resource),
               status=JobStatus(r.status), priority=r.priority, site_key=r.site_key,
               title=r.title, payload=dict(r.payload), result=dict(r.result), error=r.error,
               cancel_requested=r.cancel_requested, worker_id=r.worker_id,
               created_at=r.created_at, started_at=r.started_at, finished_at=r.finished_at)


def _event(r: JobEventRow) -> JobEvent:
    return JobEvent(seq=r.pk, job_id=r.job_id, ts=r.ts, type=EventType(r.type), data=dict(r.data))


class JobQueue:
    def __init__(self, engine: Engine):
        self.engine = engine
        self._sessions = sessionmaker(engine, expire_on_commit=False)

    def _tx(self) -> Session:
        return self._sessions()

    @staticmethod
    def _row(s: Session, job_id: str) -> JobRow:
        row = s.scalar(select(JobRow).where(JobRow.id == job_id))
        if row is None:
            raise JobNotFoundError(f"job '{job_id}' not found")
        return row

    @staticmethod
    def _add_event(s: Session, job_id: str, type_: EventType, data: dict[str, Any]) -> None:
        s.add(JobEventRow(job_id=job_id, ts=utcnow(), type=type_.value, data=data))

    # ── submit / inspect ─────────────────────────────────────────────────
    def enqueue(self, kind: JobKind, payload: dict[str, Any], *, resource: Resource,
                site_key: Optional[str] = None, title: str = "", priority: int = 0) -> Job:
        with self._tx() as s, s.begin():
            row = JobRow(id=new_job_id(), kind=kind.value, resource=resource.value,
                         status=JobStatus.queued.value, priority=priority, site_key=site_key,
                         title=title[:200], payload=payload, result={}, error=None,
                         cancel_requested=False, created_at=utcnow())
            s.add(row)
            s.flush()
            self._add_event(s, row.id, EventType.status, {"status": JobStatus.queued.value})
            return _job(row)

    def get(self, job_id: str) -> Job:
        with self._tx() as s:
            return _job(self._row(s, job_id))

    def list_jobs(self, *, statuses: Optional[Sequence[JobStatus]] = None,
             site_key: Optional[str] = None, limit: int = 100) -> list[Job]:
        q = select(JobRow).order_by(JobRow.pk.desc()).limit(limit)
        if statuses:
            q = q.where(JobRow.status.in_([st.value for st in statuses]))
        if site_key is not None:
            q = q.where(JobRow.site_key == site_key)
        with self._tx() as s:
            return [_job(r) for r in s.scalars(q)]

    def counts(self) -> dict[str, int]:
        with self._tx() as s:
            rows = s.execute(select(JobRow.status, func.count()).group_by(JobRow.status))
            out = {st.value: 0 for st in JobStatus}
            out.update({status: n for status, n in rows})
            return out

    # ── worker side ──────────────────────────────────────────────────────
    def claim(self, worker_id: str, resource: Resource) -> Optional[Job]:
        """Atomically move the next queued job of `resource` to running, or return None."""
        with self._tx() as s:
            candidates = list(s.scalars(
                select(JobRow.id)
                .where(JobRow.status == JobStatus.queued.value, JobRow.resource == resource.value)
                .order_by(JobRow.priority.desc(), JobRow.pk).limit(8)))
        for job_id in candidates:
            with self._tx() as s, s.begin():
                cond = and_(JobRow.id == job_id, JobRow.status == JobStatus.queued.value)
                if resource == Resource.gpu:
                    other = aliased(JobRow)
                    cond = and_(cond, ~exists().where(
                        other.resource == Resource.gpu.value,
                        other.status == JobStatus.running.value))
                res = s.execute(update(JobRow).where(cond).values(
                    status=JobStatus.running.value, worker_id=worker_id, started_at=utcnow())
                    .execution_options(synchronize_session=False))
                if res.rowcount != 1:  # type: ignore[attr-defined]
                    if resource == Resource.gpu:
                        return None    # the GPU slot is taken; nothing else may start
                    continue           # another worker won this one
                self._add_event(s, job_id, EventType.status,
                                {"status": JobStatus.running.value, "worker": worker_id})
                return _job(self._row(s, job_id))
        return None

    def add_events(self, job_id: str, events: Iterable[tuple[EventType, dict[str, Any]]]) -> None:
        with self._tx() as s, s.begin():
            for type_, data in events:
                self._add_event(s, job_id, type_, data)

    def add_event(self, job_id: str, type_: EventType, data: dict[str, Any]) -> None:
        self.add_events(job_id, [(type_, data)])

    def events(self, job_id: str, *, after: int = 0, limit: int = 2000) -> list[JobEvent]:
        q = (select(JobEventRow).where(JobEventRow.job_id == job_id, JobEventRow.pk > after)
             .order_by(JobEventRow.pk).limit(limit))
        with self._tx() as s:
            return [_event(r) for r in s.scalars(q)]

    def merge_result(self, job_id: str, patch: dict[str, Any]) -> Job:
        with self._tx() as s, s.begin():
            row = self._row(s, job_id)
            row.result = {**row.result, **patch}
            self._add_event(s, job_id, EventType.result, patch)
            return _job(row)

    def finish(self, job_id: str, status: JobStatus, *, error: Optional[str] = None) -> Job:
        if not status.terminal:
            raise ValueError(f"finish() needs a terminal status, got {status.value}")
        with self._tx() as s, s.begin():
            row = self._row(s, job_id)
            if row.status in TERMINAL:
                return _job(row)       # already closed (e.g. cancelled while queued)
            row.status = status.value
            row.error = error
            row.finished_at = utcnow()
            data: dict[str, Any] = {"status": status.value}
            if error:
                data["error"] = error
            self._add_event(s, job_id, EventType.status, data)
            return _job(row)

    def request_cancel(self, job_id: str) -> Job:
        """Queued jobs are cancelled at once; running ones are flagged for their worker."""
        with self._tx() as s, s.begin():
            row = self._row(s, job_id)
            if row.status == JobStatus.queued.value:
                row.status = JobStatus.cancelled.value
                row.finished_at = utcnow()
                self._add_event(s, job_id, EventType.status, {"status": row.status})
            elif row.status == JobStatus.running.value and not row.cancel_requested:
                row.cancel_requested = True
                self._add_event(s, job_id, EventType.status,
                                {"status": row.status, "cancel_requested": True})
            return _job(row)

    def cancel_requested(self, job_id: str) -> bool:
        with self._tx() as s:
            return bool(s.scalar(select(JobRow.cancel_requested).where(JobRow.id == job_id)))

    # ── workers ──────────────────────────────────────────────────────────
    def register_worker(self, worker_id: str, resource: Resource, host: str, pid: int) -> None:
        now = utcnow()
        with self._tx() as s, s.begin():
            row = s.get(WorkerRow, worker_id)
            if row is None:
                s.add(WorkerRow(id=worker_id, resource=resource.value, host=host, pid=pid,
                                started_at=now, heartbeat_at=now))
            else:
                row.resource, row.host, row.pid = resource.value, host, pid
                row.started_at = row.heartbeat_at = now
                row.current_job, row.stopped_at = None, None

    def heartbeat(self, worker_id: str, current_job: Optional[str]) -> None:
        with self._tx() as s, s.begin():
            s.execute(update(WorkerRow).where(WorkerRow.id == worker_id)
                      .values(heartbeat_at=utcnow(), current_job=current_job))

    def stop_worker(self, worker_id: str) -> None:
        with self._tx() as s, s.begin():
            s.execute(update(WorkerRow).where(WorkerRow.id == worker_id)
                      .values(stopped_at=utcnow(), current_job=None))

    def workers(self, *, include_stopped: bool = False) -> list[WorkerInfo]:
        now = utcnow()
        q = select(WorkerRow).order_by(WorkerRow.resource, WorkerRow.id)
        if not include_stopped:
            q = q.where(WorkerRow.stopped_at.is_(None))
        with self._tx() as s:
            out = []
            for r in s.scalars(q):
                online = (r.stopped_at is None
                          and (now - r.heartbeat_at).total_seconds() < WORKER_ONLINE_S)
                out.append(WorkerInfo(id=r.id, resource=Resource(r.resource), host=r.host,
                                      pid=r.pid, started_at=r.started_at,
                                      heartbeat_at=r.heartbeat_at, current_job=r.current_job,
                                      stopped_at=r.stopped_at, online=online))
            return out

    def recover_orphans(self, *, stale_after_s: float = ORPHAN_AFTER_S) -> list[str]:
        """Fail running jobs whose worker stopped or went silent (killed, machine reboot)."""
        cutoff = utcnow() - timedelta(seconds=stale_after_s)
        with self._tx() as s:
            running = list(s.scalars(select(JobRow).where(JobRow.status == JobStatus.running.value)))
            orphans = []
            for job in running:
                w = s.get(WorkerRow, job.worker_id) if job.worker_id else None
                if w is None or w.stopped_at is not None or w.heartbeat_at < cutoff:
                    orphans.append(job.id)
        for job_id in orphans:
            self.finish(job_id, JobStatus.failed,
                        error="worker lost (stopped or no heartbeat) while the job was running")
        return orphans
