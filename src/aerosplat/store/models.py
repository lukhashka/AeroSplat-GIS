"""Domain model: Project -> Site -> Epoch / Capture -> Image, and Run (ADR-0006).

These pydantic models are what the rest of the code passes around. The
SQLite catalog (`aerosplat.store.db`) is an index of them; the YAML files in
the project tree (`site.yaml`, `capture.yaml`, ...) are their human-readable
copies. Geometry lives in the Site ENU frame (`aerosplat.geo.crs.EnuFrame`)
whose origin is fixed when the Site is created.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Optional

from pydantic import (AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints,
                      model_validator)

from aerosplat.geo.crs import GeoPoint

# Directory-safe identifier used for every entity that owns a directory.
Slug = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_.-]{0,63}$")]


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


class SiteKind(str, Enum):
    urban = "urban"
    depot = "depot"
    critical = "critical"


class CaptureSource(str, Enum):
    earthstudio = "earthstudio"
    drone = "drone"
    handheld = "handheld"
    internet = "internet"


class RegistrationStatus(str, Enum):
    pending = "pending"          # ingested, not yet placed by any run
    registered = "registered"
    weak = "weak"
    rejected = "rejected"


class RunStatus(str, Enum):
    running = "running"
    ok = "ok"
    failed = "failed"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class BBox(_Model):
    """WGS84 bounding box in degrees."""

    west: float = Field(ge=-180, le=180)
    south: float = Field(ge=-90, le=90)
    east: float = Field(ge=-180, le=180)
    north: float = Field(ge=-90, le=90)

    @model_validator(mode="after")
    def _ordered(self) -> "BBox":
        if self.south >= self.north or self.west >= self.east:
            raise ValueError("bbox must satisfy west < east and south < north")
        return self


class Project(_Model):
    id: Slug
    name: str = ""
    description: str = ""
    created_at: AwareDatetime = Field(default_factory=utcnow)


class Site(_Model):
    """A patch of ground (<= ~1 km^2) with its own fixed metric frame."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Slug
    project_id: Slug
    name: str = ""
    kind: SiteKind = SiteKind.urban
    # Site ENU origin, WGS84 (EPSG:4979). Fixed for the life of the Site:
    # every capture, run and authored feature is expressed relative to it.
    origin: GeoPoint
    bbox: Optional[BBox] = None
    taxonomy_packs: list[str] = Field(default_factory=lambda: ["core"])
    created_at: AwareDatetime = Field(default_factory=utcnow)

    @property
    def key(self) -> str:
        return f"{self.project_id}/{self.id}"


class Epoch(_Model):
    """A state of the Site in time, e.g. 'baseline' or 'post-flood-2026-11-03'."""

    id: Slug
    label: str = ""
    valid_from: Optional[AwareDatetime] = None
    valid_to: Optional[AwareDatetime] = None
    created_at: AwareDatetime = Field(default_factory=utcnow)


class Capture(_Model):
    """One batch of images from one source under one licence and set of conditions."""

    id: Slug
    source: CaptureSource
    license: str = ""
    author: str = ""
    attribution: str = ""
    captured_at: Optional[AwareDatetime] = None
    # 1-sigma of the pose prior, metres (None = no prior, e.g. internet photos).
    pose_prior_sigma_m: Optional[float] = Field(None, ge=0)
    # Adapter input relative to the capture dir, e.g. the Earth Studio JSON.
    metadata_file: Optional[str] = None
    # Frame file prefix the adapter matches ('<prefix>_<n>.<ext>' for Earth Studio).
    frame_prefix: Optional[str] = None
    notes: str = ""
    created_at: AwareDatetime = Field(default_factory=utcnow)


class ImagePose(_Model):
    """Camera pose in Site ENU (COLMAP world-to-camera) with its uncertainty."""

    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    sigma_m: Optional[float] = None


class Image(_Model):
    name: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)
    width: Optional[int] = None
    height: Optional[int] = None
    # Pose prior from the source (WGS84) and its sigma, before registration.
    prior: Optional[GeoPoint] = None
    prior_sigma_m: Optional[float] = Field(None, ge=0)
    pose: Optional[ImagePose] = None
    registration_status: RegistrationStatus = RegistrationStatus.pending
    privacy_flags: list[str] = Field(default_factory=list)


class Run(_Model):
    """One immutable pipeline run for Site x Epoch; its directory holds only derived data."""

    id: Slug
    site_key: str
    epoch_id: Slug
    capture_ids: list[str] = Field(default_factory=list)
    status: RunStatus = RunStatus.running
    # Earth model the run's geometry was produced in: "wgs84", or
    # "sphere-legacy" for runs migrated from before ADR-0006.
    earth_model: str = "wgs84"
    # True when the reconstruction is in metric Site ENU (known georeferenced
    # poses); False for runs whose scale is arbitrary (synthetic GPS / SfM only).
    georeferenced: bool = False
    config: dict[str, Any] = Field(default_factory=dict)
    git_commit: Optional[str] = None
    products: dict[str, str] = Field(default_factory=dict)   # kind -> path relative to run dir
    created_at: AwareDatetime = Field(default_factory=utcnow)
    finished_at: Optional[AwareDatetime] = None
    notes: str = ""
