"""The catalog: SQLite index of the project tree plus its YAML sidecar files.

    cat = Catalog.open()                      # <repo>/projects, or $AEROSPLAT_PROJECTS
    site = cat.create_site(Site(id="pier-7", project_id="demo",
                                origin=GeoPoint(lat=..., lon=..., h=...)))
    cat.add_capture(site, Capture(...), images)
    run = cat.create_run(site, "baseline", new_run_id(), capture_ids=[...], config=cfg)

Every public method returns detached pydantic models (aerosplat.store.models);
SQLAlchemy rows never leave this module.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
from alembic import command
from alembic.config import Config as AlembicConfig
from pydantic import BaseModel
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker

from aerosplat.geo.crs import GeoPoint
from aerosplat.store.db import (CaptureRow, EpochRow, ImageRow, ProductRow, ProjectRow, RunRow,
                                SiteRow, make_engine)
from aerosplat.store.models import (BBox, Capture, CaptureSource, Epoch, Image, ImagePose,
                                    Project, RegistrationStatus, Run, RunStatus, Site, SiteKind,
                                    utcnow)
from aerosplat.store.paths import RunDirs, SiteDirs, StorePaths, default_projects_root

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
DEFAULT_EPOCH = "baseline"


class CatalogError(RuntimeError):
    pass


class NotFoundError(CatalogError, KeyError):
    def __str__(self) -> str:  # KeyError repr()s its message otherwise
        return str(self.args[0]) if self.args else ""


def alembic_config(engine: Engine | None = None) -> AlembicConfig:
    cfg = AlembicConfig()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    if engine is not None:
        cfg.set_main_option("sqlalchemy.url", engine.url.render_as_string(hide_password=False))
    return cfg


def upgrade(engine: Engine, revision: str = "head") -> None:
    """Bring the catalog schema to `revision` (idempotent)."""
    cfg = alembic_config()
    with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, revision)


def new_run_id(now: Optional[datetime] = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")


def _dump_yaml(path: Path, model: BaseModel, **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = model.model_dump(mode="json")
    data.update(extra)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


# ── row -> model ─────────────────────────────────────────────────────────
def _project(r: ProjectRow) -> Project:
    return Project(id=r.id, name=r.name, description=r.description, created_at=r.created_at)


def _site(r: SiteRow) -> Site:
    return Site(id=r.slug, project_id=r.project_id, name=r.name, kind=SiteKind(r.kind),
                origin=GeoPoint(lat=r.origin_lat, lon=r.origin_lon, h=r.origin_h),
                bbox=BBox(**r.bbox) if r.bbox else None,
                taxonomy_packs=list(r.taxonomy_packs), created_at=r.created_at)


def _epoch(r: EpochRow) -> Epoch:
    return Epoch(id=r.slug, label=r.label, valid_from=r.valid_from, valid_to=r.valid_to,
                 created_at=r.created_at)


def _capture(r: CaptureRow) -> Capture:
    return Capture(id=r.slug, source=CaptureSource(r.source), license=r.license,
                   author=r.author, attribution=r.attribution, captured_at=r.captured_at,
                   pose_prior_sigma_m=r.pose_prior_sigma_m, metadata_file=r.metadata_file,
                   frame_prefix=r.frame_prefix, notes=r.notes, created_at=r.created_at)


def _image(r: ImageRow) -> Image:
    prior = None
    if r.prior_lat is not None and r.prior_lon is not None:
        prior = GeoPoint(lat=r.prior_lat, lon=r.prior_lon, h=r.prior_h or 0.0)
    return Image(name=r.name, sha256=r.sha256, bytes=r.bytes, width=r.width, height=r.height,
                 prior=prior, prior_sigma_m=r.prior_sigma_m,
                 pose=ImagePose(**r.pose) if r.pose else None,
                 registration_status=RegistrationStatus(r.registration_status),
                 privacy_flags=list(r.privacy_flags))


def _run(r: RunRow, site_key: str) -> Run:
    return Run(id=r.slug, site_key=site_key, epoch_id=r.epoch.slug,
               capture_ids=sorted(c.slug for c in r.captures), status=RunStatus(r.status),
               earth_model=r.earth_model, georeferenced=r.georeferenced, config=dict(r.config),
               git_commit=r.git_commit, products={p.kind: p.path for p in r.products},
               created_at=r.created_at, finished_at=r.finished_at, notes=r.notes)


class Catalog:
    def __init__(self, root: Path, engine: Engine):
        self.paths = StorePaths(root)
        self.engine = engine
        self._sessions = sessionmaker(engine, expire_on_commit=False)

    @classmethod
    def open(cls, root: Path | str | None = None, *, db: Path | str | None = None) -> "Catalog":
        """Open (creating if needed) the catalog under `root` and migrate it to head."""
        root_path = Path(root) if root is not None else default_projects_root()
        root_path.mkdir(parents=True, exist_ok=True)
        engine = make_engine(db if db is not None else StorePaths(root_path).catalog_db)
        upgrade(engine)
        return cls(root_path, engine)

    def close(self) -> None:
        self.engine.dispose()

    @contextmanager
    def _tx(self) -> Iterator[Session]:
        with self._sessions() as s, s.begin():
            yield s

    # ── lookups ──────────────────────────────────────────────────────────
    @staticmethod
    def _site_row(s: Session, project_id: str, site_id: str) -> SiteRow:
        row = s.scalar(select(SiteRow).where(SiteRow.project_id == project_id,
                                             SiteRow.slug == site_id))
        if row is None:
            raise NotFoundError(f"site '{project_id}/{site_id}' not in catalog")
        return row

    @staticmethod
    def _epoch_row(s: Session, site_row: SiteRow, epoch_id: str) -> EpochRow:
        row = s.scalar(select(EpochRow).where(EpochRow.site_pk == site_row.pk,
                                              EpochRow.slug == epoch_id))
        if row is None:
            raise NotFoundError(f"epoch '{epoch_id}' not in site "
                                f"'{site_row.project_id}/{site_row.slug}'")
        return row

    @staticmethod
    def _capture_row(s: Session, site_row: SiteRow, capture_id: str) -> CaptureRow:
        row = s.scalar(select(CaptureRow).where(CaptureRow.site_pk == site_row.pk,
                                                CaptureRow.slug == capture_id))
        if row is None:
            raise NotFoundError(f"capture '{capture_id}' not in site "
                                f"'{site_row.project_id}/{site_row.slug}'")
        return row

    def _run_row(self, s: Session, site: Site, epoch_id: str, run_id: str) -> RunRow:
        ep = self._epoch_row(s, self._site_row(s, site.project_id, site.id), epoch_id)
        row = s.scalar(select(RunRow).where(RunRow.epoch_pk == ep.pk, RunRow.slug == run_id))
        if row is None:
            raise NotFoundError(f"run '{run_id}' not in {site.key}@{epoch_id}")
        return row

    # ── projects ─────────────────────────────────────────────────────────
    def create_project(self, project: Project) -> Project:
        with self._tx() as s:
            if s.get(ProjectRow, project.id) is not None:
                raise CatalogError(f"project '{project.id}' already exists")
            s.add(ProjectRow(id=project.id, name=project.name, description=project.description,
                             created_at=project.created_at))
        _dump_yaml(self.paths.project_yaml(project.id), project)
        return project

    def get_project(self, project_id: str) -> Project:
        with self._tx() as s:
            row = s.get(ProjectRow, project_id)
            if row is None:
                raise NotFoundError(f"project '{project_id}' not in catalog")
            return _project(row)

    def ensure_project(self, project_id: str, name: str = "") -> Project:
        try:
            return self.get_project(project_id)
        except NotFoundError:
            return self.create_project(Project(id=project_id, name=name or project_id))

    def list_projects(self) -> list[Project]:
        with self._tx() as s:
            return [_project(r) for r in s.scalars(select(ProjectRow).order_by(ProjectRow.id))]

    # ── sites ────────────────────────────────────────────────────────────
    def site_dirs(self, site: Site) -> SiteDirs:
        return self.paths.site(site.project_id, site.id)

    def create_site(self, site: Site) -> Site:
        """Register a Site. Its origin is fixed from here on (ADR-0006)."""
        dirs = self.site_dirs(site)
        if dirs.site_yaml.is_file():
            on_disk = yaml.safe_load(dirs.site_yaml.read_text(encoding="utf-8")) or {}
            if GeoPoint(**on_disk.get("origin", {})) != site.origin:
                raise CatalogError(f"{dirs.site_yaml} exists with a different origin; "
                                   f"a Site's origin cannot change")
        with self._tx() as s:
            if s.get(ProjectRow, site.project_id) is None:
                raise NotFoundError(f"project '{site.project_id}' not in catalog")
            exists = s.scalar(select(SiteRow.pk).where(SiteRow.project_id == site.project_id,
                                                       SiteRow.slug == site.id))
            if exists is not None:
                raise CatalogError(f"site '{site.key}' already exists")
            s.add(SiteRow(project_id=site.project_id, slug=site.id, name=site.name,
                          kind=site.kind.value, origin_lat=site.origin.lat,
                          origin_lon=site.origin.lon, origin_h=site.origin.h,
                          bbox=site.bbox.model_dump() if site.bbox else None,
                          taxonomy_packs=list(site.taxonomy_packs), created_at=site.created_at))
        for d in (dirs.captures, dirs.epochs, dirs.authored, dirs.missions):
            d.mkdir(parents=True, exist_ok=True)
        _dump_yaml(dirs.site_yaml, site)
        return site

    def get_site(self, project_id: str, site_id: str) -> Site:
        with self._tx() as s:
            return _site(self._site_row(s, project_id, site_id))

    def find_site(self, ref: str) -> Site:
        """'project/site', or a bare site id when it is unique across projects."""
        if "/" in ref:
            project_id, site_id = ref.split("/", 1)
            return self.get_site(project_id, site_id)
        matches = [s for s in self.list_sites() if s.id == ref]
        if not matches:
            raise NotFoundError(f"site '{ref}' not in catalog")
        if len(matches) > 1:
            raise CatalogError(f"site id '{ref}' is ambiguous: "
                               f"{', '.join(m.key for m in matches)}")
        return matches[0]

    def list_sites(self, project_id: Optional[str] = None) -> list[Site]:
        with self._tx() as s:
            q = select(SiteRow).order_by(SiteRow.project_id, SiteRow.slug)
            if project_id is not None:
                q = q.where(SiteRow.project_id == project_id)
            return [_site(r) for r in s.scalars(q)]

    # ── epochs ───────────────────────────────────────────────────────────
    def create_epoch(self, site: Site, epoch: Epoch) -> Epoch:
        with self._tx() as s:
            site_row = self._site_row(s, site.project_id, site.id)
            if s.scalar(select(EpochRow.pk).where(EpochRow.site_pk == site_row.pk,
                                                  EpochRow.slug == epoch.id)) is not None:
                raise CatalogError(f"epoch '{epoch.id}' already exists in {site.key}")
            s.add(EpochRow(site_pk=site_row.pk, slug=epoch.id, label=epoch.label,
                           valid_from=epoch.valid_from, valid_to=epoch.valid_to,
                           created_at=epoch.created_at))
        dirs = self.site_dirs(site)
        (dirs.epoch(epoch.id) / "runs").mkdir(parents=True, exist_ok=True)
        _dump_yaml(dirs.epoch_yaml(epoch.id), epoch)
        return epoch

    def ensure_epoch(self, site: Site, epoch_id: str = DEFAULT_EPOCH, label: str = "") -> Epoch:
        for e in self.list_epochs(site):
            if e.id == epoch_id:
                return e
        return self.create_epoch(site, Epoch(id=epoch_id, label=label or epoch_id))

    def list_epochs(self, site: Site) -> list[Epoch]:
        with self._tx() as s:
            site_row = self._site_row(s, site.project_id, site.id)
            q = select(EpochRow).where(EpochRow.site_pk == site_row.pk).order_by(EpochRow.slug)
            return [_epoch(r) for r in s.scalars(q)]

    # ── captures ─────────────────────────────────────────────────────────
    def add_capture(self, site: Site, capture: Capture, images: Sequence[Image]) -> Capture:
        """Register a capture whose files already sit in its capture directory.

        Never touches the Site (its origin in particular): a new capture is
        expressed in the existing Site frame, it does not define one.
        """
        with self._tx() as s:
            site_row = self._site_row(s, site.project_id, site.id)
            if s.scalar(select(CaptureRow.pk).where(CaptureRow.site_pk == site_row.pk,
                                                    CaptureRow.slug == capture.id)) is not None:
                raise CatalogError(f"capture '{capture.id}' already exists in {site.key}")
            row = CaptureRow(site_pk=site_row.pk, slug=capture.id, source=capture.source.value,
                             license=capture.license, author=capture.author,
                             attribution=capture.attribution, captured_at=capture.captured_at,
                             pose_prior_sigma_m=capture.pose_prior_sigma_m,
                             metadata_file=capture.metadata_file,
                             frame_prefix=capture.frame_prefix, notes=capture.notes,
                             created_at=capture.created_at)
            s.add(row)
            s.flush()
            for im in images:
                s.add(ImageRow(
                    capture_pk=row.pk, name=im.name, sha256=im.sha256, bytes=im.bytes,
                    width=im.width, height=im.height,
                    prior_lat=im.prior.lat if im.prior else None,
                    prior_lon=im.prior.lon if im.prior else None,
                    prior_h=im.prior.h if im.prior else None,
                    prior_sigma_m=im.prior_sigma_m,
                    pose=im.pose.model_dump() if im.pose else None,
                    registration_status=im.registration_status.value,
                    privacy_flags=list(im.privacy_flags)))
        cdirs = self.site_dirs(site).capture(capture.id)
        _dump_yaml(cdirs.capture_yaml, capture, num_images=len(images))
        with cdirs.images_jsonl.open("w", encoding="utf-8") as f:
            for im in images:
                f.write(json.dumps(im.model_dump(mode="json"), ensure_ascii=False) + "\n")
        return capture

    def get_capture(self, site: Site, capture_id: str) -> Capture:
        with self._tx() as s:
            site_row = self._site_row(s, site.project_id, site.id)
            return _capture(self._capture_row(s, site_row, capture_id))

    def list_captures(self, site: Site) -> list[Capture]:
        with self._tx() as s:
            site_row = self._site_row(s, site.project_id, site.id)
            q = select(CaptureRow).where(CaptureRow.site_pk == site_row.pk).order_by(CaptureRow.slug)
            return [_capture(r) for r in s.scalars(q)]

    def list_images(self, site: Site, capture_id: str) -> list[Image]:
        with self._tx() as s:
            cap = self._capture_row(s, self._site_row(s, site.project_id, site.id), capture_id)
            q = select(ImageRow).where(ImageRow.capture_pk == cap.pk).order_by(ImageRow.name)
            return [_image(r) for r in s.scalars(q)]

    # ── runs ─────────────────────────────────────────────────────────────
    def run_dirs(self, site: Site, epoch_id: str, run_id: str) -> RunDirs:
        return self.site_dirs(site).run(epoch_id, run_id)

    def create_run(self, site: Site, epoch_id: str, run_id: str, *,
                   capture_ids: Sequence[str] = (), config: Optional[Mapping[str, Any]] = None,
                   earth_model: str = "wgs84", georeferenced: bool = False,
                   git_commit: Optional[str] = None, notes: str = "") -> Run:
        with self._tx() as s:
            site_row = self._site_row(s, site.project_id, site.id)
            ep = self._epoch_row(s, site_row, epoch_id)
            if s.scalar(select(RunRow.pk).where(RunRow.epoch_pk == ep.pk,
                                                RunRow.slug == run_id)) is not None:
                raise CatalogError(f"run '{run_id}' already exists in {site.key}@{epoch_id}")
            row = RunRow(site_pk=site_row.pk, epoch_pk=ep.pk, slug=run_id,
                         status=RunStatus.running.value, earth_model=earth_model,
                         georeferenced=georeferenced, config=dict(config or {}),
                         git_commit=git_commit, created_at=utcnow(), notes=notes)
            row.captures = [self._capture_row(s, site_row, c) for c in capture_ids]
            s.add(row)
            s.flush()
            run = _run(row, site.key)
        self.run_dirs(site, epoch_id, run_id).root.mkdir(parents=True, exist_ok=True)
        return run

    def finish_run(self, site: Site, epoch_id: str, run_id: str, *, status: RunStatus,
                   products: Optional[Mapping[str, Path | str]] = None,
                   georeferenced: Optional[bool] = None) -> Run:
        """Close a run: final status, and the products that exist (paths made run-relative)."""
        root = self.run_dirs(site, epoch_id, run_id).root
        with self._tx() as s:
            row = self._run_row(s, site, epoch_id, run_id)
            row.status = status.value
            row.finished_at = utcnow()
            if georeferenced is not None:
                row.georeferenced = georeferenced
            for kind, path in (products or {}).items():
                p = Path(path)
                if not p.is_absolute():
                    p = root / p
                if not p.exists():
                    continue
                rel = p.resolve().relative_to(root.resolve()).as_posix()
                existing = next((x for x in row.products if x.kind == kind), None)
                if existing is None:
                    row.products.append(ProductRow(kind=kind, path=rel))
                else:
                    existing.path = rel
            s.flush()
            return _run(row, site.key)

    def get_run(self, site: Site, epoch_id: str, run_id: str) -> Run:
        with self._tx() as s:
            return _run(self._run_row(s, site, epoch_id, run_id), site.key)

    def list_runs(self, site: Site, epoch_id: Optional[str] = None) -> list[Run]:
        with self._tx() as s:
            site_row = self._site_row(s, site.project_id, site.id)
            q = select(RunRow).where(RunRow.site_pk == site_row.pk)
            if epoch_id is not None:
                q = q.where(RunRow.epoch_pk == self._epoch_row(s, site_row, epoch_id).pk)
            q = q.order_by(RunRow.created_at, RunRow.slug)
            return [_run(r, site.key) for r in s.scalars(q)]

    def latest_run(self, site: Site, *, status: RunStatus = RunStatus.ok,
                   epoch_id: Optional[str] = None,
                   with_products: Sequence[str] = ()) -> Optional[Run]:
        """Most recently finished run in `status` that has at least one of `with_products`."""
        runs = [r for r in self.list_runs(site, epoch_id) if r.status == status]
        if with_products:
            runs = [r for r in runs if any(k in r.products for k in with_products)]
        runs.sort(key=lambda r: (r.finished_at or r.created_at, r.id))
        return runs[-1] if runs else None

    def product_path(self, site: Site, run: Run, kind: str) -> Optional[Path]:
        rel = run.products.get(kind)
        return None if rel is None else self.run_dirs(site, run.epoch_id, run.id).root / rel
