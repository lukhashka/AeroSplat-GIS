"""API v2: projects, sites, captures, runs and the job queue (TARGET_ARCHITECTURE §8.1).

Mounted by `aerosplat.server.main` next to the v1 scene endpoints. Deliberately
torch-free: everything heavy happens in workers (aerosplat.jobs), the API only
reads the catalog, enqueues jobs and streams their events.
"""

from __future__ import annotations

import asyncio
import io
import json
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import (APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile,
                     WebSocket, WebSocketDisconnect)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aerosplat import __version__
from aerosplat.config import RunConfig
from aerosplat.geo.crs import GeoPoint
from aerosplat.jobs.models import (PIPELINE_STAGES, DiagnosticSpec, Job, JobEvent, JobKind,
                                   JobStatus, PipelineSpec, Resource, WorkerInfo)
from aerosplat.jobs.queue import JobNotFoundError, JobQueue
from aerosplat.paths import PROJECT_DIR
from aerosplat.pipeline.manifest import git_commit
from aerosplat.store.catalog import Catalog, CatalogError, NotFoundError
from aerosplat.store.models import (BBox, Capture, CaptureSource, Epoch, Image, Project, Run,
                                    RunStatus, Site, SiteKind)
from aerosplat.store.paths import default_projects_root
from aerosplat.store.scenes import CLOUD, SPLAT

router = APIRouter(prefix="/api/v2", tags=["v2"])

SLUG = r"^[a-z0-9][a-z0-9_.-]{0,63}$"
THUMB_MAX = 1024


# ── catalog / queue handles ──────────────────────────────────────────────
_CATALOGS: dict[str, Catalog] = {}
_CATALOG_LOCK = threading.Lock()


def get_catalog() -> Catalog:
    """One open Catalog per projects root (resolved per call, so tests can switch roots)."""
    root = default_projects_root()
    key = str(root.resolve())
    with _CATALOG_LOCK:
        cat = _CATALOGS.get(key)
        if cat is None:
            cat = _CATALOGS[key] = Catalog.open(root)
        return cat


def get_queue(cat: Catalog = Depends(get_catalog)) -> JobQueue:
    return JobQueue(cat.engine)


def _site(cat: Catalog, project_id: str, site_id: str) -> Site:
    try:
        return cat.get_site(project_id, site_id)
    except NotFoundError as e:
        raise HTTPException(404, str(e)) from None


# ── schemas ──────────────────────────────────────────────────────────────
class _In(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectCreate(_In):
    id: str = Field(pattern=SLUG)
    name: str = ""
    description: str = ""


class SiteCreate(_In):
    id: str = Field(pattern=SLUG)
    name: str = ""
    kind: SiteKind = SiteKind.urban
    origin: GeoPoint
    bbox: Optional[BBox] = None


class RunBrief(BaseModel):
    id: str
    epoch_id: str
    status: RunStatus
    earth_model: str
    georeferenced: bool
    created_at: Any
    finished_at: Any = None
    products: list[str] = Field(default_factory=list)


class SiteSummary(BaseModel):
    site: Site
    captures: int
    images: int
    runs: int
    latest_run: Optional[RunBrief] = None
    scene: Optional[str] = Field(None, description="?dataset= for the viewer, when viewable")
    active_jobs: int = 0


class CaptureOut(BaseModel):
    capture: Capture
    images: int
    bytes: int
    with_prior: int


class SiteDetail(BaseModel):
    site: Site
    epochs: list[Epoch]
    captures: list[CaptureOut]
    runs: list[Run]
    jobs: list[Job]
    scene: Optional[str] = None


class RunRequest(_In):
    epoch: str = Field("baseline", pattern=SLUG)
    capture: Optional[str] = None
    run_id: Optional[str] = Field(None, pattern=SLUG)
    stages: Optional[list[str]] = None
    from_stage: Optional[str] = None
    to_stage: Optional[str] = None
    resume: bool = False
    overrides: dict[str, Any] = Field(default_factory=dict)
    priority: int = 0


class DiagnosticRequest(DiagnosticSpec):
    resource: Resource = Resource.cpu


class StageInfo(BaseModel):
    stage: str
    status: str
    seconds: Optional[float] = None
    error: Optional[str] = None


class RunDetail(BaseModel):
    run: Run
    stages: list[StageInfo]
    metrics: Optional[dict[str, Any]] = None
    has_report: bool = False
    scene: Optional[str] = None


class SystemInfo(BaseModel):
    version: str
    git_commit: Optional[str]
    projects_root: str
    workers: list[WorkerInfo]
    queue: dict[str, int]
    gpu: Optional[list[dict[str, Any]]] = None


def _brief(run: Run) -> RunBrief:
    return RunBrief(id=run.id, epoch_id=run.epoch_id, status=run.status,
                    earth_model=run.earth_model, georeferenced=run.georeferenced,
                    created_at=run.created_at, finished_at=run.finished_at,
                    products=sorted(run.products))


def _scene_name(cat: Catalog, site: Site) -> Optional[str]:
    run = cat.latest_run(site, with_products=(SPLAT, CLOUD))
    return site.key if run is not None else None


# ── projects & sites ─────────────────────────────────────────────────────
@router.get("/projects", response_model=list[Project])
def list_projects(cat: Catalog = Depends(get_catalog)) -> list[Project]:
    return cat.list_projects()


@router.post("/projects", response_model=Project, status_code=201)
def create_project(body: ProjectCreate, cat: Catalog = Depends(get_catalog)) -> Project:
    try:
        return cat.create_project(Project(**body.model_dump()))
    except CatalogError as e:
        raise HTTPException(409, str(e)) from None


@router.get("/sites", response_model=list[SiteSummary])
def list_sites(project: Optional[str] = None, cat: Catalog = Depends(get_catalog),
               q: JobQueue = Depends(get_queue)) -> list[SiteSummary]:
    active = q.list_jobs(statuses=[JobStatus.queued, JobStatus.running], limit=1000)
    out = []
    for site in cat.list_sites(project):
        captures = cat.list_captures(site)
        runs = cat.list_runs(site)
        latest = runs[-1] if runs else None
        out.append(SiteSummary(
            site=site, captures=len(captures),
            images=sum(len(cat.list_images(site, c.id)) for c in captures),
            runs=len(runs), latest_run=_brief(latest) if latest else None,
            scene=_scene_name(cat, site),
            active_jobs=sum(1 for j in active if j.site_key == site.key)))
    return out


@router.post("/projects/{project_id}/sites", response_model=Site, status_code=201)
def create_site(project_id: str, body: SiteCreate, cat: Catalog = Depends(get_catalog)) -> Site:
    try:
        cat.get_project(project_id)
        site = cat.create_site(Site(project_id=project_id, **body.model_dump()))
        cat.ensure_epoch(site)
        return site
    except NotFoundError as e:
        raise HTTPException(404, str(e)) from None
    except CatalogError as e:
        raise HTTPException(409, str(e)) from None


@router.get("/sites/{project_id}/{site_id}", response_model=SiteDetail)
def site_detail(project_id: str, site_id: str, cat: Catalog = Depends(get_catalog),
                q: JobQueue = Depends(get_queue)) -> SiteDetail:
    site = _site(cat, project_id, site_id)
    caps = []
    for c in cat.list_captures(site):
        imgs = cat.list_images(site, c.id)
        caps.append(CaptureOut(capture=c, images=len(imgs), bytes=sum(i.bytes for i in imgs),
                               with_prior=sum(1 for i in imgs if i.prior is not None)))
    return SiteDetail(site=site, epochs=cat.list_epochs(site), captures=caps,
                      runs=list(reversed(cat.list_runs(site))),
                      jobs=q.list_jobs(site_key=site.key, limit=20), scene=_scene_name(cat, site))


# ── captures ─────────────────────────────────────────────────────────────
@router.get("/sites/{project_id}/{site_id}/captures/{capture_id}/images",
            response_model=list[Image])
def capture_images(project_id: str, site_id: str, capture_id: str,
                   cat: Catalog = Depends(get_catalog)) -> list[Image]:
    site = _site(cat, project_id, site_id)
    try:
        return cat.list_images(site, capture_id)
    except NotFoundError as e:
        raise HTTPException(404, str(e)) from None


@router.get("/sites/{project_id}/{site_id}/captures/{capture_id}/images/{name}/thumb")
def capture_thumb(project_id: str, site_id: str, capture_id: str, name: str,
                  size: int = Query(256, ge=32, le=THUMB_MAX),
                  cat: Catalog = Depends(get_catalog)) -> Response:
    site = _site(cat, project_id, site_id)
    try:
        known = {i.name for i in cat.list_images(site, capture_id)}
    except NotFoundError as e:
        raise HTTPException(404, str(e)) from None
    if name not in known:   # only catalogued names: no path traversal via {name}
        raise HTTPException(404, f"image '{name}' not in capture '{capture_id}'")
    path = cat.site_dirs(site).capture(capture_id).images / name
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=82)
    return Response(buf.getvalue(), media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=86400, immutable"})


def _safe_name(filename: str) -> str:
    name = Path(filename.replace("\\", "/")).name
    if not name or name in (".", "..") or not re.fullmatch(r"[\w .()+-]+", name):
        raise HTTPException(422, f"unsupported file name '{filename}'")
    return name


@router.post("/sites/{project_id}/{site_id}/captures", response_model=CaptureOut, status_code=201)
def upload_capture(project_id: str, site_id: str,
                   capture_id: str = Form(..., pattern=SLUG),
                   source: CaptureSource = Form(...),
                   license: str = Form(""), author: str = Form(""), notes: str = Form(""),
                   files: list[UploadFile] = File(...),
                   cat: Catalog = Depends(get_catalog)) -> CaptureOut:
    """Upload images (and, for Earth Studio, the export JSON) as a new capture.

    Files are staged next to the captures on the same disk, then moved in by the
    regular ingest adapters, so the catalog sees exactly what the CLI would.
    """
    site = _site(cat, project_id, site_id)
    if capture_id in {c.id for c in cat.list_captures(site)}:
        raise HTTPException(409, f"capture '{capture_id}' already exists in {site.key}")
    staging = cat.site_dirs(site).captures / f".upload-{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=True)
    try:
        names = []
        for f in files:
            name = _safe_name(f.filename or "")
            with (staging / name).open("wb") as out:
                shutil.copyfileobj(f.file, out)
            names.append(name)
        if source == CaptureSource.earthstudio:
            from aerosplat.ingest.earthstudio import ATTRIBUTION_FILE, import_earthstudio

            jsons = [n for n in names if n.lower().endswith(".json")]
            if len(jsons) != 1:
                raise HTTPException(422, "Earth Studio upload needs exactly one export .json "
                                         f"(got {len(jsons)})")
            extras = [staging / n for n in names if n == ATTRIBUTION_FILE or n.endswith(".esp")]
            capture = import_earthstudio(cat, site, staging / jsons[0], staging,
                                         capture_id=capture_id, extra_files=extras,
                                         author=author, move=True)
        else:
            from aerosplat.ingest.folder import import_folder

            capture = import_folder(cat, site, staging, capture_id=capture_id, source=source,
                                    license=license, author=author, notes=notes, move=True)
    except (FileNotFoundError, FileExistsError, ValueError, CatalogError) as e:
        raise HTTPException(422, f"{type(e).__name__}: {e}") from None
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    imgs = cat.list_images(site, capture.id)
    return CaptureOut(capture=capture, images=len(imgs), bytes=sum(i.bytes for i in imgs),
                      with_prior=sum(1 for i in imgs if i.prior is not None))


# ── runs ─────────────────────────────────────────────────────────────────
@router.post("/sites/{project_id}/{site_id}/runs", response_model=Job, status_code=202)
def start_run(project_id: str, site_id: str, body: RunRequest,
              cat: Catalog = Depends(get_catalog), q: JobQueue = Depends(get_queue)) -> Job:
    """Queue a pipeline run of the Site on the GPU worker."""
    site = _site(cat, project_id, site_id)
    captures = [c.id for c in cat.list_captures(site)]
    if not captures:
        raise HTTPException(409, f"{site.key} has no captures yet — upload one first")
    capture = body.capture
    if capture is None:
        if len(captures) > 1:
            raise HTTPException(422, f"{site.key} has several captures; choose one of {captures}")
        capture = captures[0]
    elif capture not in captures:
        raise HTTPException(404, f"capture '{capture}' not in {site.key}")
    try:
        spec = PipelineSpec(site=site.key, capture=capture,
                            **body.model_dump(exclude={"capture", "priority"}))
    except ValidationError as e:
        raise HTTPException(422, json.loads(e.json(include_url=False))) from None
    title = f"{site.key} · {spec.epoch} · {capture}"
    return q.enqueue(JobKind.pipeline, spec.model_dump(), resource=Resource.gpu,
                     site_key=site.key, title=title, priority=body.priority)


def _manifest_stages(path: Path) -> list[StageInfo]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    stages = data.get("stages") or {}
    return [StageInfo(stage=name, status=str(st.get("status", "?")), seconds=st.get("seconds"),
                      error=st.get("error"))
            for name, st in stages.items() if isinstance(st, dict)]


@router.get("/sites/{project_id}/{site_id}/runs/{epoch_id}/{run_id}", response_model=RunDetail)
def run_detail(project_id: str, site_id: str, epoch_id: str, run_id: str,
               cat: Catalog = Depends(get_catalog)) -> RunDetail:
    site = _site(cat, project_id, site_id)
    try:
        run = cat.get_run(site, epoch_id, run_id)
    except NotFoundError as e:
        raise HTTPException(404, str(e)) from None
    dirs = cat.run_dirs(site, epoch_id, run_id)
    metrics = None
    mpath = cat.product_path(site, run, "metrics")
    if mpath is not None and mpath.is_file():
        try:
            metrics = json.loads(mpath.read_text(encoding="utf-8"))
        except ValueError:
            metrics = None
    report = cat.product_path(site, run, "report")
    latest = cat.latest_run(site, with_products=(SPLAT, CLOUD))
    return RunDetail(run=run, stages=_manifest_stages(dirs.manifest), metrics=metrics,
                     has_report=bool(report and report.is_file()),
                     scene=site.key if latest is not None and latest.id == run.id else None)


@router.get("/sites/{project_id}/{site_id}/runs/{epoch_id}/{run_id}/report")
def run_report(project_id: str, site_id: str, epoch_id: str, run_id: str,
               cat: Catalog = Depends(get_catalog)) -> Response:
    site = _site(cat, project_id, site_id)
    try:
        run = cat.get_run(site, epoch_id, run_id)
    except NotFoundError as e:
        raise HTTPException(404, str(e)) from None
    path = cat.product_path(site, run, "report")
    if path is None or not path.is_file():
        raise HTTPException(404, "run has no report")
    return Response(path.read_bytes(), media_type="text/html; charset=utf-8")


# ── jobs ─────────────────────────────────────────────────────────────────
def _job_or_404(q: JobQueue, job_id: str) -> Job:
    try:
        return q.get(job_id)
    except JobNotFoundError as e:
        raise HTTPException(404, str(e)) from None


@router.get("/jobs", response_model=list[Job])
def list_jobs(status: Optional[list[JobStatus]] = Query(None), site: Optional[str] = None,
              limit: int = Query(50, ge=1, le=500), q: JobQueue = Depends(get_queue)) -> list[Job]:
    return q.list_jobs(statuses=status, site_key=site, limit=limit)


@router.post("/jobs/diagnostic", response_model=Job, status_code=202)
def start_diagnostic(body: DiagnosticRequest, q: JobQueue = Depends(get_queue)) -> Job:
    spec = DiagnosticSpec(**body.model_dump(exclude={"resource"}))
    return q.enqueue(JobKind.diagnostic, spec.model_dump(), resource=body.resource,
                     title=f"diagnostic · {body.resource.value} · {spec.stages} steps")


@router.get("/jobs/{job_id}", response_model=Job)
def get_job(job_id: str, q: JobQueue = Depends(get_queue)) -> Job:
    return _job_or_404(q, job_id)


@router.post("/jobs/{job_id}/cancel", response_model=Job)
def cancel_job(job_id: str, q: JobQueue = Depends(get_queue)) -> Job:
    _job_or_404(q, job_id)
    return q.request_cancel(job_id)


@router.get("/jobs/{job_id}/events", response_model=list[JobEvent])
def job_events(job_id: str, after: int = Query(0, ge=0), limit: int = Query(2000, ge=1, le=10000),
               q: JobQueue = Depends(get_queue)) -> list[JobEvent]:
    _job_or_404(q, job_id)
    return q.events(job_id, after=after, limit=limit)


STREAM_POLL_S = 0.4


@router.websocket("/jobs/{job_id}/stream")
async def job_stream(ws: WebSocket, job_id: str, after: int = 0) -> None:
    """Push the job (on every status change) and its events from cursor `after` until it ends.

    Messages: {"type": "job", "job": Job} and {"type": "events", "events": [JobEvent, ...]}.
    """
    await ws.accept()
    q = JobQueue(get_catalog().engine)
    try:
        job = await run_in_threadpool(q.get, job_id)
    except JobNotFoundError as e:
        await ws.send_json({"type": "error", "detail": str(e)})
        await ws.close(code=4404)
        return
    cursor = after
    last_sent: Optional[str] = None
    try:
        while True:
            events = await run_in_threadpool(q.events, job_id, after=cursor, limit=2000)
            if events:
                cursor = events[-1].seq
                await ws.send_json({"type": "events",
                                    "events": [e.model_dump(mode="json") for e in events]})
            job = await run_in_threadpool(q.get, job_id)
            snapshot = job.model_dump_json()
            if snapshot != last_sent:
                last_sent = snapshot
                await ws.send_json({"type": "job", "job": job.model_dump(mode="json")})
            if job.status.terminal and not events:
                break
            await asyncio.sleep(STREAM_POLL_S)
        await ws.close()
    except WebSocketDisconnect:
        return


# ── system & pipeline metadata ───────────────────────────────────────────
_GPU_CACHE: dict[str, Any] = {"t": 0.0, "v": None}
_COMMIT: dict[str, Optional[str]] = {}


def _gpu_info() -> Optional[list[dict[str, Any]]]:
    """nvidia-smi snapshot, cached for a few seconds; None when there is no NVIDIA GPU."""
    now = time.monotonic()
    if now - _GPU_CACHE["t"] < 5.0:
        return _GPU_CACHE["v"]
    value: Optional[list[dict[str, Any]]] = None
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            out = subprocess.run(
                [exe, "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3, check=True).stdout
            value = []
            for line in out.strip().splitlines():
                name, used, total, util, temp = (x.strip() for x in line.split(","))
                value.append({"name": name, "memory_used_mb": float(used),
                              "memory_total_mb": float(total), "utilization": float(util),
                              "temperature_c": float(temp)})
        except (OSError, subprocess.SubprocessError, ValueError):
            value = None
    _GPU_CACHE.update(t=now, v=value)
    return value


@router.get("/system", response_model=SystemInfo)
def system(cat: Catalog = Depends(get_catalog), q: JobQueue = Depends(get_queue)) -> SystemInfo:
    if "commit" not in _COMMIT:
        _COMMIT["commit"] = git_commit(PROJECT_DIR)
    return SystemInfo(version=__version__, git_commit=_COMMIT["commit"],
                      projects_root=str(cat.paths.root), workers=q.workers(),
                      queue=q.counts(), gpu=_gpu_info())


@router.get("/pipeline")
def pipeline_info() -> dict[str, Any]:
    """Stage list and the RunConfig schema/defaults, for the run dialog."""
    from aerosplat.pipeline.stages import STAGES

    return {"stages": [{"name": n, "label": STAGES[n].label} for n in PIPELINE_STAGES],
            "config_schema": RunConfig.model_json_schema(),
            "defaults": RunConfig().to_dict()}


# Set by aerosplat.server.main: re-reads scenes so finished runs become viewable.
SceneRefresher = Callable[[], dict[str, Any]]


@router.post("/scenes/refresh")
def refresh_scenes(request: Request) -> dict[str, Any]:
    refresher: Optional[SceneRefresher] = getattr(request.app.state, "refresh_scenes", None)
    if refresher is None:
        raise HTTPException(501, "scene refresh is not wired in this app")
    return refresher()
