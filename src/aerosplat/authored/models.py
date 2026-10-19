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

A pad (a flat roof or a patch of ground, §3.2) has no facade: it is drawn as a
horizontal outline in Site ENU (`pad`), and then carries neither `host` nor `geometry`.
"""

from __future__ import annotations

import os
import secrets
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from aerosplat.auth.context import current_principal

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


def new_ulid(prefix: str, now_ms: Optional[int] = None) -> str:
    """`prefix` + a ULID: 48 bits of millisecond timestamp then 80 random bits."""
    ms = now_ms if now_ms is not None else int(time.time() * 1000)
    return prefix + _base32(ms, 10) + _base32(secrets.randbits(80), 16)


def new_ap_id(now_ms: Optional[int] = None) -> str:
    """`ap_` + a ULID (§3.1).

    Sortable by creation time, unique without a central counter, and stable for the
    life of the feature across runs.
    """
    return new_ulid(AP_PREFIX, now_ms)


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


class ApFlag(str, Enum):
    """Conditions that block designation without changing the lifecycle (§5.9, §6.3)."""

    # A new run did not find this verified feature again: maybe the building changed,
    # maybe the detector failed. Somebody has to look before it becomes a target.
    not_redetected = "not_redetected"


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


class PadGeometry(_Model):
    """A horizontal pad (§3.2): an outline in Site ENU on a flat surface.

    v0 keeps only what approach planning needs. `center` and `area_m2` are derived from
    the outline (aerosplat.authored.geometry.pad_geometry), like the ENU block of an opening.
    """

    frame: str = "site_enu"
    polygon_enu: list[tuple[float, float]] = Field(min_length=3)   # (east, north), metres
    z: float                                    # ENU height of the flat surface
    building_id: Optional[str] = None           # the roof's building; None for a ground pad
    center: tuple[float, float, float]          # area centroid, at z
    area_m2: float = Field(gt=0)


class ApUncertainty(_Model):
    """§3.1: how well the feature's centre is known in Site ENU (1σ, the model's frame).

    Georeferencing (model -> WGS84) is a separate term, added where the target is built.
    """

    center_cov: tuple[tuple[float, float, float], tuple[float, float, float],
                      tuple[float, float, float]]                   # m²
    sigma_normal_deg: Optional[float] = Field(None, ge=0)

    @model_validator(mode="after")
    def _covariance(self) -> "ApUncertainty":
        c = self.center_cov
        for a in range(3):
            if not c[a][a] >= 0:
                raise ValueError("center_cov must have a non-negative diagonal")
            for b in range(3):
                if abs(c[a][b] - c[b][a]) > 1e-9:
                    raise ValueError("center_cov must be symmetric")
        return self


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
    # Drawn on a facade: `host` + `geometry`. A pad: `pad` alone.
    host: Optional[HostFacade] = None
    geometry: Optional[Geometry] = None
    pad: Optional[PadGeometry] = None
    uncertainty: Optional[ApUncertainty] = None
    attributes: dict[str, str] = Field(default_factory=dict)
    address: Optional[Address] = None
    confidence: Optional[float] = Field(None, ge=0, le=1)
    lifecycle: Lifecycle = Lifecycle.verified
    flags: list[ApFlag] = Field(default_factory=list)
    provenance: Provenance
    notes: str = ""

    @model_validator(mode="after")
    def _kind_matches_class(self) -> "AccessPoint":
        head = self.class_.split(".", 1)[0]
        if self.kind != head:
            raise ValueError(f"kind '{self.kind}' does not match class '{self.class_}'")
        if (self.host is None) != (self.geometry is None):
            raise ValueError("host and geometry go together: a facade frame and the shape in it")
        if (self.host is not None) == (self.pad is not None):
            raise ValueError("an access point is drawn either on a facade (host + geometry) "
                             "or as a pad (pad), exactly one of the two")
        if self.pad is not None and self.kind != "pad":
            raise ValueError(f"a pad outline needs a pad class, not '{self.class_}'")
        return self

    @property
    def facade_id(self) -> Optional[str]:
        return self.host.facade_id if self.host is not None else None

    @property
    def where(self) -> str:
        """Where it is, for humans and audit lines: the facade, or the pad's host."""
        if self.host is not None:
            return self.host.facade_id
        assert self.pad is not None
        return f"roof of {self.pad.building_id}" if self.pad.building_id else "ground"


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
    """Who is making a change.

    A request authenticated with an API token is recorded as `token:<token name>`, and
    a name the client claims (`by`) is then ignored: the token is what was verified.
    Otherwise (this machine, CLI) as before — the claimed name or the OS user, as
    `op:<login>` (§13 q. 5)."""
    p = current_principal()
    if p is not None and p.actor:
        return p.actor
    if name:
        return name if ":" in name else f"op:{name}"
    login = os.environ.get("AEROSPLAT_OPERATOR") or os.environ.get("USERNAME") \
        or os.environ.get("USER") or "unknown"
    return f"op:{login}"
