"""API v2, GIS export (Phase 3.2): queue an export of a run, list and download its files.

Mounted next to `api_v2`, sharing its catalog handle. Torch-free: the export itself runs
as a job (`JobKind.export`, aerosplat.jobs.child -> `aerosplat export`), on the GPU
worker when an orthophoto is asked for (it renders the splat) and on the CPU worker
otherwise. A run without georeference is refused here, before anything is queued, with
the same explanation the CLI gives.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aerosplat.export.georef import ExportRefused, SiteProjection
from aerosplat.export.run import (EXPORTS_DIR, KINDS, MANIFEST, PRODUCT_PREFIX, ExportTarget,
                                  read_manifest)
from aerosplat.jobs.models import ExportSpec, Job, JobKind
from aerosplat.jobs.queue import JobQueue
from aerosplat.server import api_v2
from aerosplat.store.catalog import Catalog, NotFoundError
from aerosplat.store.models import Run, Site

router = APIRouter(prefix="/api/v2", tags=["v2", "export"])

MEDIA = {".tif": "image/tiff", ".laz": "application/octet-stream",
         ".las": "application/octet-stream", ".ply": "application/octet-stream"}


class ExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kinds: list[str] = Field(default_factory=lambda: list(KINDS))
    crs: str = "utm"
    gsd_m: float = 0.25
    dem_px_m: Optional[float] = None
    las_source: Literal["cloud", "splat"] = "cloud"
    laz: bool = True
    cog: bool = True
    priority: int = 0


class ExportFile(BaseModel):
    kind: str
    file: str
    bytes: int
    url: str
    crs: Optional[dict[str, Any]] = None
    created: Optional[str] = None


class ExportListing(BaseModel):
    run: str
    epoch: str
    exportable: bool
    refusal: Optional[str] = None
    files: list[ExportFile] = Field(default_factory=list)
    georeference: Optional[dict[str, Any]] = None


def _run(cat: Catalog, site: Site, epoch_id: str, run_id: str) -> Run:
    try:
        return cat.get_run(site, epoch_id, run_id)
    except NotFoundError as e:
        raise HTTPException(404, str(e)) from None


def _base(site: Site, run: Run) -> str:
    return f"{api_v2.router.prefix}/sites/{site.project_id}/{site.id}/runs/{run.epoch_id}/{run.id}/exports"


@router.post("/sites/{project_id}/{site_id}/runs/{epoch_id}/{run_id}/exports",
             response_model=Job, status_code=202)
def start_export(project_id: str, site_id: str, epoch_id: str, run_id: str, body: ExportRequest,
                 cat: Catalog = Depends(api_v2.get_catalog),
                 q: JobQueue = Depends(api_v2.get_queue)) -> Job:
    """Queue GIS exports of a finished, georeferenced run (409 with the reason otherwise)."""
    site = api_v2._site(cat, project_id, site_id)
    run = _run(cat, site, epoch_id, run_id)
    try:
        spec = ExportSpec(site=site.key, epoch=run.epoch_id, run_id=run.id,
                          **body.model_dump(exclude={"priority"}))
    except ValidationError as e:
        raise HTTPException(422, json.loads(e.json(include_url=False))) from None
    try:
        ExportTarget.for_run(cat, site, run)
    except ExportRefused as e:
        raise HTTPException(409, str(e)) from None
    try:
        SiteProjection(site.origin, spec.crs)
    except ExportRefused as e:
        raise HTTPException(422, str(e)) from None
    title = f"{site.key} · {run.id} · експорт {', '.join(spec.kinds)}"
    return q.enqueue(JobKind.export, spec.model_dump(), resource=spec.resource,
                     site_key=site.key, title=title, priority=body.priority)


@router.get("/sites/{project_id}/{site_id}/runs/{epoch_id}/{run_id}/exports",
            response_model=ExportListing)
def list_exports(project_id: str, site_id: str, epoch_id: str, run_id: str,
                 cat: Catalog = Depends(api_v2.get_catalog)) -> ExportListing:
    """The run's exported files (those on disk) and the georeference budget they carry."""
    site = api_v2._site(cat, project_id, site_id)
    run = _run(cat, site, epoch_id, run_id)
    refusal = None
    try:
        ExportTarget.for_run(cat, site, run)
    except ExportRefused as e:
        refusal = str(e)
    out_dir = cat.run_dirs(site, run.epoch_id, run.id).root / EXPORTS_DIR
    manifest = read_manifest(out_dir)
    files = []
    for kind in KINDS:
        entry = (manifest.get("files") or {}).get(kind)
        path = cat.product_path(site, run, PRODUCT_PREFIX + kind)
        if not entry or path is None or not path.is_file():
            continue
        files.append(ExportFile(kind=kind, file=path.name, bytes=path.stat().st_size,
                                url=f"{_base(site, run)}/{kind}", crs=entry.get("crs"),
                                created=entry.get("created")))
    return ExportListing(run=run.id, epoch=run.epoch_id, exportable=refusal is None,
                         refusal=refusal, files=files, georeference=manifest.get("georeference"))


@router.get("/sites/{project_id}/{site_id}/runs/{epoch_id}/{run_id}/exports/{kind}")
def download_export(project_id: str, site_id: str, epoch_id: str, run_id: str, kind: str,
                    cat: Catalog = Depends(api_v2.get_catalog)) -> FileResponse:
    if kind not in KINDS and kind != "manifest":
        raise HTTPException(404, f"unknown export '{kind}'; known: {list(KINDS)}")
    site = api_v2._site(cat, project_id, site_id)
    run = _run(cat, site, epoch_id, run_id)
    root = cat.run_dirs(site, run.epoch_id, run.id).root
    if kind == "manifest":
        path = root / EXPORTS_DIR / MANIFEST
    else:
        path = cat.product_path(site, run, PRODUCT_PREFIX + kind)
    if path is None or not path.is_file():
        raise HTTPException(404, f"run {run.id} has no '{kind}' export yet")
    resolved = path.resolve()
    if root.resolve() not in resolved.parents:
        raise HTTPException(404, f"export '{kind}' is outside the run directory")
    name = f"{site.id}_{run.id}_{path.name}"
    return FileResponse(resolved, media_type=MEDIA.get(path.suffix.lower(), "application/json"),
                        filename=name)
