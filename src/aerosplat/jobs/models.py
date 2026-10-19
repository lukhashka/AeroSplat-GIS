"""Job queue domain: what a job is, what it asks for, what it reports (ADR-0010).

A job is queued by the API, claimed by a worker of the matching resource class
and executed in a child process (`aerosplat.jobs.child`), so a CUDA OOM or a
crashed COLMAP binary takes down the child, never the API or the worker.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from aerosplat.config import RunConfig


class JobKind(str, Enum):
    pipeline = "pipeline"        # `aerosplat run --site ...` for one Site x Epoch
    diagnostic = "diagnostic"    # fake stages with sleeps: checks a worker end to end
    export = "export"            # `aerosplat export`: GIS files of one finished run
    ingest = "ingest"            # an uploaded capture: hash, lay out, register (CPU worker)


class Resource(str, Enum):
    gpu = "gpu"   # exactly one at a time (8 GB VRAM), enforced in JobQueue.claim
    cpu = "cpu"


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled)


class EventType(str, Enum):
    status = "status"    # {"status": ..., "error"?: ...}
    stage = "stage"      # {"stage", "label", "state": started|skipped|done|failed, "index", "total", "seconds"?}
    log = "log"          # {"lines": [...], "stream": "stdout"}
    result = "result"    # {"run_id": ...} etc., merged into Job.result


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Job(_Model):
    id: str
    kind: JobKind
    resource: Resource
    status: JobStatus
    priority: int = 0
    site_key: Optional[str] = None
    title: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    cancel_requested: bool = False
    worker_id: Optional[str] = None
    created_at: AwareDatetime
    started_at: Optional[AwareDatetime] = None
    finished_at: Optional[AwareDatetime] = None


class JobEvent(_Model):
    seq: int                      # stream cursor: strictly increasing per catalog
    job_id: str
    ts: AwareDatetime
    type: EventType
    data: dict[str, Any] = Field(default_factory=dict)


class WorkerInfo(_Model):
    id: str
    resource: Resource
    host: str
    pid: int
    started_at: AwareDatetime
    heartbeat_at: AwareDatetime
    current_job: Optional[str] = None
    stopped_at: Optional[AwareDatetime] = None
    online: bool = False


# ── payloads ─────────────────────────────────────────────────────────────
PIPELINE_STAGES = ("geo", "exif", "sfm", "dense", "parse", "train", "eval", "georef",
                   "semantics", "occupancy")


class PipelineSpec(_Model):
    """What `POST /sites/{p}/{s}/runs` accepts; becomes `aerosplat run` argv in the job process."""

    site: str = Field(description="project/site")
    epoch: str = "baseline"
    capture: Optional[str] = None
    run_id: Optional[str] = Field(None, pattern=r"^[a-z0-9][a-z0-9_.-]{0,63}$")
    stages: Optional[list[str]] = Field(None, description="explicit stage list")
    from_stage: Optional[str] = None
    to_stage: Optional[str] = None
    resume: bool = False
    overrides: dict[str, Any] = Field(default_factory=dict,
                                      description="RunConfig keys (validated on submit)")

    @field_validator("stages")
    @classmethod
    def _known_stages(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is not None:
            bad = [s for s in v if s not in PIPELINE_STAGES]
            if bad:
                raise ValueError(f"unknown stages {bad}; known: {list(PIPELINE_STAGES)}")
            if not v:
                raise ValueError("stages must not be empty")
        return v

    @field_validator("from_stage", "to_stage")
    @classmethod
    def _known_stage(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in PIPELINE_STAGES:
            raise ValueError(f"unknown stage '{v}'; known: {list(PIPELINE_STAGES)}")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> "PipelineSpec":
        if self.stages and (self.from_stage or self.to_stage):
            raise ValueError("give either stages or from_stage/to_stage, not both")
        if (self.from_stage and self.to_stage
                and PIPELINE_STAGES.index(self.from_stage) > PIPELINE_STAGES.index(self.to_stage)):
            raise ValueError("from_stage comes after to_stage")
        bad = sorted(set(self.overrides) - set(RunConfig.model_fields))
        if bad:
            raise ValueError(f"unknown config keys {bad}")
        # Site/dataset selection is the spec's job, not the overrides'.
        clash = sorted({"dataset", "footage_dir", "json_path"} & set(self.overrides))
        if clash:
            raise ValueError(f"{clash} are derived from the Site/capture and cannot be overridden")
        RunConfig(**self.overrides)          # full validation (ranges, enums)
        return self

    def argv(self, projects_root: Optional[str] = None) -> list[str]:
        a = ["--site", self.site, "--epoch", self.epoch]
        if self.capture:
            a += ["--capture", self.capture]
        if self.run_id:
            a += ["--run-id", self.run_id]
        if projects_root:
            a += ["--projects-root", projects_root]
        if self.stages:
            a += ["--stages", ",".join(self.stages)]
        if self.from_stage:
            a += ["--from", self.from_stage]
        if self.to_stage:
            a += ["--to", self.to_stage]
        if self.resume:
            a.append("--resume")
        for k, v in self.overrides.items():
            a += ["--set", f"{k}={'none' if v is None else v}"]
        return a


EXPORT_KINDS = ("ortho", "dsm", "dtm", "ndsm", "las", "ply")   # = aerosplat.export.run.KINDS


class ExportSpec(_Model):
    """What `POST /sites/{p}/{s}/runs/{e}/{r}/exports` accepts; becomes `aerosplat export` argv.

    Only `ortho` needs the GPU (it renders the splat), so the job's resource follows the kinds.
    """

    site: str = Field(description="project/site")
    epoch: str
    run_id: str
    kinds: list[str] = Field(default_factory=lambda: list(EXPORT_KINDS))
    crs: str = Field("utm", max_length=64, pattern=r"^(utm|[A-Za-z]+:[0-9]+)$",
                     description="'utm' (the Site origin's WGS84 UTM zone) or e.g. EPSG:32618")
    gsd_m: float = Field(0.25, ge=0.01, le=10.0, description="orthophoto pixel, metres")
    dem_px_m: Optional[float] = Field(None, ge=0.1, le=50.0)
    las_source: Literal["cloud", "splat"] = "cloud"
    laz: bool = True
    cog: bool = True

    @field_validator("kinds")
    @classmethod
    def _known_kinds(cls, v: list[str]) -> list[str]:
        bad = [k for k in v if k not in EXPORT_KINDS]
        if bad or not v:
            raise ValueError(f"unknown export kinds {bad}; known: {list(EXPORT_KINDS)}")
        return list(dict.fromkeys(v))

    @property
    def resource(self) -> "Resource":
        return Resource.gpu if "ortho" in self.kinds else Resource.cpu

    def argv(self, projects_root: Optional[str] = None) -> list[str]:
        a = ["--site", self.site, "--epoch", self.epoch, "--run", self.run_id,
             "--kinds", ",".join(self.kinds), "--crs", self.crs, "--gsd", str(self.gsd_m),
             "--las-source", self.las_source]
        if self.dem_px_m is not None:
            a += ["--dem-px", str(self.dem_px_m)]
        if not self.laz:
            a.append("--las")
        if not self.cog:
            a.append("--no-cog")
        if projects_root:
            a += ["--projects-root", projects_root]
        return a


STAGING_RE = r"^\.upload-[0-9a-f]{32}$"


class IngestSpec(_Model):
    """An uploaded capture waiting in `captures/<staging>/` of its Site (aerosplat.ingest.upload)."""

    site: str = Field(description="project/site")
    capture_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{0,63}$")
    source: str
    license: str = ""
    author: str = ""
    notes: str = ""
    # A directory *name*, never a path: the job resolves it inside the Site's captures/.
    staging: str = Field(pattern=STAGING_RE)
    files: int = Field(0, ge=0)
    bytes: int = Field(0, ge=0)


class DiagnosticSpec(_Model):
    stages: int = Field(3, ge=1, le=20)
    seconds_per_stage: float = Field(1.0, ge=0, le=60)
    fail_at: Optional[int] = Field(None, ge=1, description="1-based stage that fails")
    lines_per_stage: int = Field(3, ge=0, le=1000)


TERMINAL: tuple[str, ...] = tuple(s.value for s in JobStatus if s.terminal)
StageState = Literal["started", "skipped", "done", "failed"]
