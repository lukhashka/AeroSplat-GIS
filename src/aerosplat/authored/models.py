"""Authored access points: what a person drew, as it is stored (ACCESS_POINTS §3.1).

Authored data is not derived data. It lives in `sites/<site>/authored/`, never in a
run directory, because a run may be deleted and recomputed (ADR-0006). That is also
why a feature does not reference a run for its geometry: it carries the facade frame
it was drawn in (`host`), so it can be re-anchored when a new run moves the facade
(§5.9) instead of silently pointing somewhere else.

The drawing itself is `geometry.polygon_uv` — facade (u, v) metres, with `u` growing
to the right of an observer standing outside facing the wall. The ENU fields beside
it are derived from that polygon and the host frame, so a consumer that only wants a
point in the world needs no facade lookup.
"""

from __future__ import annotations

import os
import secrets
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

SCHEMA = 1
# Crockford base32, as ULID uses: no I, L, O or U, so a read-aloud id has no ambiguity.
CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
AP_PREFIX = "ap_"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _base32(value: int, length: int) -> str:
    out = []
    for _ in range(length):
        out.append(CROCKFORD[value & 31])
        value >>= 5
    return "".join(reversed(out))


def new_ap_id(now_ms: Optional[int] = None) -> str:
    """`ap_` + a ULID: 48 bits of millisecond timestamp then 80 random bits (§3.1).

    Sortable by creation time, unique without a central counter, and stable for the
    life of the feature across runs.
    """
    ms = now_ms if now_ms is not None else int(time.time() * 1000)
    return AP_PREFIX + _base32(ms, 10) + _base32(secrets.randbits(80), 16)


class Lifecycle(str, Enum):
    """§3.4. A person drawing a feature by hand lands straight in `verified`.

    `rejected` is never deleted: it is a hard negative for ML and the memory that stops
    the pipeline proposing the same wrong candidate again.
    """

    candidate = "candidate"
    verified = "verified"
    rejected = "rejected"
    stale = "stale"


class Origin(str, Enum):
    pipeline = "pipeline"
    manual = "manual"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class HostFacade(_Model):
    """The facade frame `polygon_uv` was drawn in, copied onto the feature.

    Keeping the frame rather than only the id is what lets a feature survive a
    recomputed run: a new run may name or shift the facade differently, and the stored
    plane is what a re-anchoring pass compares against (§5.9).
    """

    facade_id: str                              # "<building>/<octant><ordinal>", e.g. "B01/N1"
    building_id: str
    origin: tuple[float, float, float]          # ENU point at (u, v) = (0, v_ground)
    normal: tuple[float, float, float]          # unit, outward
    u_axis: tuple[float, float, float]          # unit, horizontal, right when facing the wall
    v_ground: float                             # ENU z of the ground at the wall foot
    azimuth_deg: float


class Geometry(_Model):
    """Where the feature is. `polygon_uv` is authoritative; the rest is derived from it."""

    frame: str = "site_enu"                     # of center / normal / u_axis (ADR-0006)
    polygon_uv: list[tuple[float, float]] = Field(min_length=3)
    center: tuple[float, float, float]
    normal: tuple[float, float, float]
    u_axis: tuple[float, float, float]
    width: float = Field(gt=0)                  # along u, metres
    height: float = Field(gt=0)                 # along v, metres
    u_center: float
    v_center: float

    @property
    def u_bounds(self) -> tuple[float, float]:
        us = [p[0] for p in self.polygon_uv]
        return min(us), max(us)

    @property
    def v_bounds(self) -> tuple[float, float]:
        vs = [p[1] for p in self.polygon_uv]
        return min(vs), max(vs)


class Address(_Model):
    """§4. A wrong floor or column sends the drone to someone else's window, so
    `floor_ambiguous` blocks confirming a target without an explicit human check (§6.3)."""

    floor_index: Optional[int] = None           # 0 = the row at ground level
    floor_label: Optional[str] = None           # by local convention (UA: ground floor = "1")
    column: Optional[int] = None                # 1-based, left to right facing the wall
    ordinal_on_floor: Optional[int] = None      # n-th opening that actually exists on the row
    floor_ambiguous: bool = False
    canonical: Optional[str] = None             # "<site>/<building>/<facade>/<floor>/<column>"


class Edit(_Model):
    by: str
    at: AwareDatetime
    what: str


class Provenance(_Model):
    origin: Origin = Origin.manual
    run_id: Optional[str] = None                # the run whose ortho was drawn on, for reference
    epoch_id: Optional[str] = None
    model: Optional[str] = None                 # set when a detector proposed the feature
    support_views: Optional[int] = None
    evidence: list[str] = Field(default_factory=list)      # e.g. ["ortho:B01/N1@5cm"]
    created_by: str = "op:unknown"
    created_at: AwareDatetime = Field(default_factory=utcnow)
    edits: list[Edit] = Field(default_factory=list)


class AccessPoint(_Model):
    """One access point as stored in `authored/access_points.json` (§3.1).

    `class` is the field name in the file and on the wire, as the design doc has it;
    in Python it is `class_`, since `class` is a keyword.
    """

    id: str = Field(pattern=r"^ap_[0-9A-HJKMNP-TV-Z]{26}$")
    site_key: str
    class_: str = Field(alias="class")
    kind: str                                   # the class's kind, denormalised for filtering
    taxonomy: str                               # "core@1.0+residential@1.0" (§2.5)
    host: HostFacade
    geometry: Geometry
    attributes: dict[str, str] = Field(default_factory=dict)
    address: Optional[Address] = None
    confidence: Optional[float] = Field(None, ge=0, le=1)
    lifecycle: Lifecycle = Lifecycle.verified
    provenance: Provenance
    notes: str = ""

    @model_validator(mode="after")
    def _kind_matches_class(self) -> "AccessPoint":
        head = self.class_.split(".", 1)[0]
        if self.kind != head:
            raise ValueError(f"kind '{self.kind}' does not match class '{self.class_}'")
        return self

    @property
    def facade_id(self) -> str:
        return self.host.facade_id


class AuthoredFile(_Model):
    """`authored/access_points.json`: the current state of a Site's authored features."""

    schema_version: int = Field(SCHEMA, alias="schema")
    site_key: str
    updated_at: AwareDatetime = Field(default_factory=utcnow)
    features: list[AccessPoint] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> "AuthoredFile":
        ids = [f.id for f in self.features]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate feature ids")
        return self


class AuditEntry(_Model):
    """One line of `authored/audit.jsonl` (§3.5): who changed what, and what it was before."""

    at: AwareDatetime = Field(default_factory=utcnow)
    by: str
    op: str                                     # create | update | delete | address
    feature_id: str
    site_key: str
    what: str = ""                              # short human summary of the change
    before: Optional[dict] = None
    after: Optional[dict] = None


def actor(name: Optional[str] = None) -> str:
    """Who is making a change. No accounts yet: the OS user, as `op:<login>` (§13 q. 5)."""
    if name:
        return name if ":" in name else f"op:{name}"
    login = os.environ.get("AEROSPLAT_OPERATOR") or os.environ.get("USERNAME") \
        or os.environ.get("USER") or "unknown"
    return f"op:{login}"
