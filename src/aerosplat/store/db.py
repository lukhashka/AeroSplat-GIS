"""SQLite catalog schema (SQLAlchemy 2.x). Schema changes go through Alembic
(`aerosplat/store/migrations`); `tests/test_store.py` checks the two agree.

Written against SQLAlchemy Core types only, so moving to PostgreSQL/PostGIS
(TARGET_ARCHITECTURE §7.2) does not touch the domain model.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import (JSON, Boolean, Column, DateTime, Engine, Float, ForeignKey, Integer,
                        MetaData, String, Table, Text, TypeDecorator, UniqueConstraint,
                        create_engine, event)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, validates

NAMING = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class OriginImmutableError(ValueError):
    """A Site's ENU origin cannot change once the Site exists (ADR-0006)."""


class UtcDateTime(TypeDecorator[datetime]):
    """Timezone-aware UTC in Python, naive UTC in SQLite (which has no tz type)."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: Optional[datetime], dialect: Any) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime given to the catalog; use timezone-aware UTC")
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def process_result_value(self, value: Optional[datetime], dialect: Any) -> Optional[datetime]:
        return None if value is None else value.replace(tzinfo=timezone.utc)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING)


class ProjectRow(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime)

    sites: Mapped[list["SiteRow"]] = relationship(back_populates="project")


class SiteRow(Base):
    __tablename__ = "sites"
    __table_args__ = (UniqueConstraint("project_id", "slug", name="uq_sites_project_slug"),)

    pk: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    slug: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(200), default="")
    kind: Mapped[str] = mapped_column(String(16))
    # Fixed at creation; also guarded by a SQLite trigger (migration 0001).
    origin_lat: Mapped[float] = mapped_column(Float)
    origin_lon: Mapped[float] = mapped_column(Float)
    origin_h: Mapped[float] = mapped_column(Float)
    bbox: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    taxonomy_packs: Mapped[list[str]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime)

    project: Mapped[ProjectRow] = relationship(back_populates="sites")
    epochs: Mapped[list["EpochRow"]] = relationship(back_populates="site")
    captures: Mapped[list["CaptureRow"]] = relationship(back_populates="site")
    runs: Mapped[list["RunRow"]] = relationship(back_populates="site")

    @validates("origin_lat", "origin_lon", "origin_h")
    def _origin_is_fixed(self, key: str, value: float) -> float:
        current = self.__dict__.get(key)
        if current is not None and current != value:
            raise OriginImmutableError(
                f"site origin is fixed at creation ({key}: {current} -> {value} refused)")
        return value


class EpochRow(Base):
    __tablename__ = "epochs"
    __table_args__ = (UniqueConstraint("site_pk", "slug", name="uq_epochs_site_slug"),)

    pk: Mapped[int] = mapped_column(Integer, primary_key=True)
    site_pk: Mapped[int] = mapped_column(ForeignKey("sites.pk"), index=True)
    slug: Mapped[str] = mapped_column(String(64))
    label: Mapped[str] = mapped_column(String(200), default="")
    valid_from: Mapped[Optional[datetime]] = mapped_column(UtcDateTime, nullable=True)
    valid_to: Mapped[Optional[datetime]] = mapped_column(UtcDateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime)

    site: Mapped[SiteRow] = relationship(back_populates="epochs")


class CaptureRow(Base):
    __tablename__ = "captures"
    __table_args__ = (UniqueConstraint("site_pk", "slug", name="uq_captures_site_slug"),)

    pk: Mapped[int] = mapped_column(Integer, primary_key=True)
    site_pk: Mapped[int] = mapped_column(ForeignKey("sites.pk"), index=True)
    slug: Mapped[str] = mapped_column(String(64))
    source: Mapped[str] = mapped_column(String(16))
    license: Mapped[str] = mapped_column(Text, default="")
    author: Mapped[str] = mapped_column(String(200), default="")
    attribution: Mapped[str] = mapped_column(Text, default="")
    captured_at: Mapped[Optional[datetime]] = mapped_column(UtcDateTime, nullable=True)
    pose_prior_sigma_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    metadata_file: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    frame_prefix: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime)

    site: Mapped[SiteRow] = relationship(back_populates="captures")
    images: Mapped[list["ImageRow"]] = relationship(back_populates="capture")


class ImageRow(Base):
    __tablename__ = "images"
    __table_args__ = (UniqueConstraint("capture_pk", "name", name="uq_images_capture_name"),)

    pk: Mapped[int] = mapped_column(Integer, primary_key=True)
    capture_pk: Mapped[int] = mapped_column(ForeignKey("captures.pk"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    bytes: Mapped[int] = mapped_column(Integer)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    prior_lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prior_lon: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prior_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prior_sigma_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pose: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    registration_status: Mapped[str] = mapped_column(String(16))
    privacy_flags: Mapped[list[str]] = mapped_column(JSON)

    capture: Mapped[CaptureRow] = relationship(back_populates="images")


run_captures = Table(
    "run_captures", Base.metadata,
    Column("run_pk", ForeignKey("runs.pk"), primary_key=True),
    Column("capture_pk", ForeignKey("captures.pk"), primary_key=True),
)


class RunRow(Base):
    __tablename__ = "runs"
    __table_args__ = (UniqueConstraint("epoch_pk", "slug", name="uq_runs_epoch_slug"),)

    pk: Mapped[int] = mapped_column(Integer, primary_key=True)
    site_pk: Mapped[int] = mapped_column(ForeignKey("sites.pk"), index=True)
    epoch_pk: Mapped[int] = mapped_column(ForeignKey("epochs.pk"), index=True)
    slug: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))
    earth_model: Mapped[str] = mapped_column(String(32))
    georeferenced: Mapped[bool] = mapped_column(Boolean)
    config: Mapped[dict[str, Any]] = mapped_column(JSON)
    git_commit: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime)
    finished_at: Mapped[Optional[datetime]] = mapped_column(UtcDateTime, nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")

    site: Mapped[SiteRow] = relationship(back_populates="runs")
    epoch: Mapped[EpochRow] = relationship()
    captures: Mapped[list[CaptureRow]] = relationship(secondary=run_captures)
    products: Mapped[list["ProductRow"]] = relationship(back_populates="run",
                                                        cascade="all, delete-orphan")


class ProductRow(Base):
    __tablename__ = "products"
    __table_args__ = (UniqueConstraint("run_pk", "kind", name="uq_products_run_kind"),)

    pk: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_pk: Mapped[int] = mapped_column(ForeignKey("runs.pk"), index=True)
    kind: Mapped[str] = mapped_column(String(32))
    path: Mapped[str] = mapped_column(String(512))   # relative to the run directory

    run: Mapped[RunRow] = relationship(back_populates="products")


class JobRow(Base):
    """A unit of background work (aerosplat.jobs); the queue lives in the catalog DB."""

    __tablename__ = "jobs"

    # Insertion order: FIFO among equal priorities (created_at is per-second).
    pk: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(32), unique=True)
    kind: Mapped[str] = mapped_column(String(32))
    resource: Mapped[str] = mapped_column(String(8), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    site_key: Mapped[Optional[str]] = mapped_column(String(130), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(200), default="")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    result: Mapped[dict[str, Any]] = mapped_column(JSON)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    worker_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime)
    started_at: Mapped[Optional[datetime]] = mapped_column(UtcDateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(UtcDateTime, nullable=True)


class JobEventRow(Base):
    """Append-only event log of a job; `pk` doubles as the clients' stream cursor."""

    __tablename__ = "job_events"

    pk: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    ts: Mapped[datetime] = mapped_column(UtcDateTime)
    type: Mapped[str] = mapped_column(String(16))
    data: Mapped[dict[str, Any]] = mapped_column(JSON)


class WorkerRow(Base):
    """A worker process; `heartbeat_at` tells the UI (and orphan recovery) it is alive."""

    __tablename__ = "workers"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    resource: Mapped[str] = mapped_column(String(8))
    host: Mapped[str] = mapped_column(String(128))
    pid: Mapped[int] = mapped_column(Integer)
    started_at: Mapped[datetime] = mapped_column(UtcDateTime)
    heartbeat_at: Mapped[datetime] = mapped_column(UtcDateTime)
    current_job: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    stopped_at: Mapped[Optional[datetime]] = mapped_column(UtcDateTime, nullable=True)


def make_engine(db: Path | str) -> Engine:
    """Engine for a catalog file (or any SQLAlchemy URL), with SQLite FKs enforced.

    The API, the workers and the job processes they spawn all write the same
    SQLite file, so it runs in WAL mode (readers never block the writer) and
    waits for the write lock instead of failing at once.
    """
    url = db if isinstance(db, str) and "://" in db else f"sqlite:///{Path(db).as_posix()}"
    engine = create_engine(url)
    if engine.dialect.name == "sqlite":
        in_memory = engine.url.database in (None, "", ":memory:")

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA busy_timeout=10000")
            if not in_memory:
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()
    return engine
