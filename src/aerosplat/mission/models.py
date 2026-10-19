"""What a mission is made of: targets, planning numbers, legs, findings, the plan itself.

Geometry is Site ENU metres (ADR-0006), like everything the plan is computed from. WGS84 and
altitudes appear only at export (aerosplat.mission.export), where the altitude frame is
chosen explicitly and checked.

Two kinds of "no" are kept apart, because they mean different things to an operator:

* a **refusal** (`MissionRefused`) — the target may not be planned at all: it is not
  confirmed, its access point is no longer `verified`, its σ is unknown. No plan exists;
* a **finding** — a plan was computed and something about it is wrong (`blocking`) or
  needs a person to accept it (`risk`). The verdict follows from the findings.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from aerosplat.authored.models import AccessPoint, Address, utcnow
from aerosplat.taxonomy.models import HandoverMode

SCHEMA = 1
TARGET_PREFIX = "tgt_"
MISSION_PREFIX = "msn_"
TARGETS_JSON = "targets.json"           # in sites/<site>/authored/
MISSION_JSON = "mission.json"           # in sites/<site>/missions/<id>/

Vec3 = tuple[float, float, float]


class Code(str, Enum):
    """Named reasons. Each is an expected outcome of some negative test (SIMULATION §4.6)."""

    # designation: the target may not exist or may not be planned
    not_verified = "not_verified"               # the access point is a candidate or rejected
    stale = "stale"
    floor_ambiguous = "floor_ambiguous"
    not_redetected = "not_redetected"
    sigma_unknown = "sigma_unknown"             # nobody can say how well the target is known
    sigma_too_large = "sigma_too_large"
    no_handover = "no_handover"                 # the class has no approach numbers (a constraint)
    mode_not_allowed = "mode_not_allowed"
    no_threshold = "no_threshold"               # the taxonomy has no σ gate for the mode
    no_entry_distance = "no_entry_distance"
    unsupported_kind = "unsupported_kind"
    override_relaxes_taxonomy = "override_relaxes_taxonomy"
    ap_missing = "ap_missing"
    outside_grid = "outside_grid"               # the mission leaves the occupancy grid: no check
    no_takeoff_spot = "no_takeoff_spot"         # home search found no spot for this profile
    # the takeoff search, run again, no longer proposes the spot the operator chose from it
    takeoff_spot_changed = "takeoff_spot_changed"
    target_not_confirmed = "target_not_confirmed"
    target_revoked = "target_revoked"
    target_changed = "target_changed"           # the access point was edited after confirmation
    # feasibility: a plan exists, and this is what is wrong with it
    unknown_space = "unknown_space"
    obstacle_in_corridor = "obstacle_in_corridor"
    overhang_above = "overhang_above"
    wire_risk = "wire_risk"
    assumed_free_air = "assumed_free_air"
    standoff_too_small = "standoff_too_small"
    transit_too_high = "transit_too_high"
    home_unknown = "home_unknown"
    home_blocked = "home_blocked"
    home_not_on_surface = "home_not_on_surface"


class Severity(str, Enum):
    blocking = "blocking"           # the mission must not fly
    risk = "risk"                   # it may, if a person accepts this


class Verdict(str, Enum):
    feasible = "feasible"
    feasible_with_risks = "feasible_with_risks"
    infeasible = "infeasible"


class TargetState(str, Enum):
    """ACCESS_POINTS §6.4, the part v0 needs. `planned / dispatched / outcome` come with dispatch."""

    draft = "draft"                 # proposed (by a person or a resolver), not yet confirmed
    confirmed = "confirmed"         # a named operator confirmed it; recorded in the audit log
    revoked = "revoked"             # withdrawn; kept, like a rejected access point


class AltitudeFrame(str, Enum):
    """How exported altitudes are measured. Always explicit; there is no implicit default.

    `relative_home` is MAV_FRAME_GLOBAL_RELATIVE_ALT: metres above the point the vehicle
    arms at. `amsl` is MAV_FRAME_GLOBAL and is refused in v0: Earth Studio heights are
    unverified (ROADMAP 6.1) and there is no geoid model in the environment.
    """

    relative_home = "relative_home"
    amsl = "amsl"


class AltitudeFrameError(ValueError):
    """An altitude in the wrong frame, or a frame v0 cannot vouch for (SIMULATION §4.6)."""


def require_relative(frame: AltitudeFrame) -> None:
    if frame is not AltitudeFrame.relative_home:
        raise AltitudeFrameError(
            f"altitude frame '{frame.value}' is refused: the twin's absolute heights are "
            f"unverified (Earth Studio, ROADMAP 6.1) and no geoid model is available, so v0 "
            f"exports altitudes relative to home only")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Finding(_Model):
    code: Code
    severity: Severity
    leg: Optional[str] = None               # which leg, for feasibility findings
    detail: str = ""
    at: Optional[Vec3] = None               # where, in Site ENU


class MissionRefused(Exception):
    """No plan: the target cannot be planned. `findings` say why (all blocking)."""

    def __init__(self, findings: list[Finding]):
        self.findings = findings
        super().__init__("; ".join(f"{f.code.value}: {f.detail}" if f.detail else f.code.value
                                   for f in findings))


# ── targets ──────────────────────────────────────────────────────────────
class SigmaTerm(_Model):
    """One term of the σ budget (1σ, metres) and who or what says so."""

    value_m: float = Field(ge=0)
    source: str


class SigmaBudget(_Model):
    """Where the target is, in the world: the access point in the model, and the model in
    WGS84 (ACCESS_POINTS §7.2). GNSS is the vehicle's and joins at planning."""

    ap: SigmaTerm
    georef: SigmaTerm

    @property
    def target_m(self) -> float:
        return math.hypot(self.ap.value_m, self.georef.value_m)


class HandoverOverride(_Model):
    """Per-target changes to the class's numbers (§7.1). Only towards caution."""

    standoff_m: Optional[float] = Field(None, gt=0, le=100)
    min_clearance_m: Optional[float] = Field(None, gt=0, le=100)
    approach_cone_deg: Optional[float] = Field(None, gt=0, le=180)
    entry_distance_m: Optional[float] = Field(None, gt=0, le=200)


class HandoverSpec(_Model):
    """The numbers a target is approached with, resolved from the taxonomy (+ overrides)."""

    mode: HandoverMode
    standoff_m: float
    min_clearance_m: float
    approach_cone_deg: float
    entry_distance_m: Optional[float] = None
    hover_dh_m: float = 0.0
    max_sigma_m: float
    source: str                             # "core@1.1 opening.window"
    overridden: list[str] = Field(default_factory=list)


class MissionTarget(_Model):
    """A target (ACCESS_POINTS §6): a person's decision to deliver to one verified access point."""

    schema_version: int = Field(SCHEMA, alias="schema")
    id: str = Field(pattern=r"^tgt_[0-9A-HJKMNP-TV-Z]{26}$")
    site_key: str
    ap_id: str
    ap_class: str
    ap_kind: str
    # sha256 of the access point's geometry, class and lifecycle at confirmation: a plan
    # against an access point that has moved since is refused (`target_changed`).
    ap_fingerprint: str
    taxonomy: str
    handover: HandoverSpec
    sigma: SigmaBudget
    address: Optional[Address] = None
    state: TargetState = TargetState.draft
    created_by: str
    created_at: AwareDatetime = Field(default_factory=utcnow)
    confirmed_by: Optional[str] = None
    confirmed_at: Optional[AwareDatetime] = None
    revoked_by: Optional[str] = None
    revoked_at: Optional[AwareDatetime] = None
    notes: str = ""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class TargetsFile(_Model):
    """`authored/targets.json`: a Site's targets. Authored data, like the access points."""

    schema_version: int = Field(SCHEMA, alias="schema")
    site_key: str
    updated_at: AwareDatetime = Field(default_factory=utcnow)
    targets: list[MissionTarget] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ── navigation (ADR-0015) ────────────────────────────────────────────────
class GnssUse(str, Enum):
    """What the mission lets GNSS do (ADR-0015 §2.2)."""

    primary = "primary"     # EKF3 on GNSS: the SIM-1 package, unchanged
    monitor = "monitor"     # receiver on, never an EKF3 source; compared with VPS (spoofing)
    none = "none"           # no GNSS at all, from power-on: GPS1_TYPE 0


class ExtNavSpec(_Model):
    """The external navigation source the plan relies on: VPS against the twin (ADR-0015 layer A),
    fed to EKF3 as ExternalNav (MAVLink ODOMETRY). What is DECLARED for planning and for the
    package; what the source really does in a simulation is the scenario's error model."""

    sigma_m: float = Field(0.5, gt=0, le=10)            # 1σ horizontal position, per axis
    sigma_z_m: float = Field(1.0, gt=0, le=20)          # 1σ vertical (EKF3 height stays on baro)
    yaw_sigma_deg: float = Field(2.0, gt=0, le=45)
    vel_sigma_mps: float = Field(0.1, ge=0.05, le=5)    # VISO_VEL_M_NSE's documented range
    rate_hz: float = Field(10.0, ge=1, le=50)
    latency_ms: float = Field(50.0, ge=0, le=250)       # VISO_DELAY_MS's documented range
    camera: str = "imx290:4mm"                          # configs/sim/cameras/<name>.yaml : lens


class NavProfile(_Model):
    """How the vehicle knows where it is. `primary` keeps the SIM-1 package byte for byte;
    `none` and `monitor` need `extnav` and make EKF3 run on it (set 1) with optical flow as the
    degraded source (set 2, when `flow`)."""

    gnss: GnssUse = GnssUse.primary
    extnav: Optional[ExtNavSpec] = None
    flow: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> "NavProfile":
        if self.gnss is GnssUse.primary and (self.extnav is not None or self.flow):
            raise ValueError("nav: gnss primary takes no extnav / flow (SIM-2n stage 1)")
        if self.gnss is not GnssUse.primary and self.extnav is None:
            raise ValueError(f"nav: gnss {self.gnss.value} needs an extnav source")
        return self

    @property
    def gnss_free(self) -> bool:
        """EKF3 never takes position, velocity or yaw from GNSS."""
        return self.gnss is not GnssUse.primary


# ── planning ─────────────────────────────────────────────────────────────
class PlanningProfile(_Model):
    """The vehicle and the planning margins. Placeholders until the vehicle is chosen
    (SIMULATION §7 q. 4); every plan records the profile it was computed with.

    The approach numbers per class (standoff, clearance, cone) are not here: they are
    taxonomy data (ADR-0007) and arrive through the target's `HandoverSpec`.
    """

    r_drone_m: float = Field(0.5, gt=0, le=10)          # radius of the vehicle incl. propellers
    sigma_gnss_m: float = Field(1.5, ge=0, le=50)       # 1σ horizontal GNSS of the vehicle
    k_sigma: float = Field(2.0, ge=0, le=10)            # margin in σ: r = r_drone + m + k·σ
    transit_margin_m: float = Field(2.0, ge=0, le=50)   # `m` for transit, takeoff and landing
    max_alt_rel_m: float = Field(120.0, gt=0, le=1000)  # above home: regulatory ceiling
    transit_speed_mps: float = Field(8.0, gt=0, le=30)
    approach_speed_mps: float = Field(1.0, gt=0, le=10)
    climb_speed_mps: float = Field(2.5, gt=0, le=10)
    descent_speed_mps: float = Field(1.5, gt=0, le=10)
    hold_s: float = Field(20.0, ge=0, le=600)           # at the hover point
    # Stop at E before the approach. Copter plans the leg after a pass-through waypoint in
    # advance, with the speed it has then, so the approach speed set at E would only apply
    # late: in SITL (Copter-4.7.2) the 1 m/s approach was flown at 2.4 m/s without this stop.
    entry_hold_s: float = Field(2.0, gt=0, le=60)
    fence_margin_m: float = Field(5.0, ge=0, le=100)    # beyond the corridor, horizontally
    fence_alt_margin_m: float = Field(10.0, ge=0, le=100)
    # Wires are not reconstructed (ACCESS_POINTS §5.8): a corridor this low over the surface
    # below it carries `wire_risk`.
    wire_height_m: float = Field(15.0, ge=0, le=100)
    # Relief of the target surface (sills, cornices, roof gravel): voxels this close to the
    # target plane are the surface itself, which onboard guidance keeps its distance from.
    surface_relief_m: float = Field(0.3, ge=0, le=2)
    cone_tilt_steps: int = Field(3, ge=0, le=12)        # tilts tried inside the cone
    cone_azimuths: int = Field(8, ge=1, le=36)          # directions per tilt
    # Navigation (ADR-0015). Default: GNSS, as before; the package is then unchanged.
    nav: NavProfile = NavProfile()

    @property
    def sigma_nav_m(self) -> float:
        """1σ horizontal of the vehicle's position source: GNSS, or the declared ExternalNav
        when GNSS is not used (the corridor radius takes it where it took σ_GNSS)."""
        if self.nav.gnss_free:
            assert self.nav.extnav is not None
            return self.nav.extnav.sigma_m
        return self.sigma_gnss_m

    @property
    def nav_source(self) -> str:
        return "extnav" if self.nav.gnss_free else "gnss"


class SigmaUsed(_Model):
    ap_m: float
    georef_m: float
    # σ of the vehicle's position source. Named for GNSS (SIM-1 plans); with a GNSS-free
    # navigation profile it is the declared ExternalNav σ, and `nav_source` says so.
    gnss_m: float
    nav_source: str = "gnss"
    target_m: float                         # hypot(ap, georef)
    approach_m: float                       # hypot(ap, georef, gnss): σ_total of §7.2
    transit_m: float                        # hypot(georef, gnss): the map against the vehicle


class HomePoint(_Model):
    """Where the vehicle arms. `relative_home` altitudes are measured from `enu[2]`."""

    enu: Vec3
    lat: float
    lon: float
    h_ellipsoid: float                      # WGS84
    source: str
    # AMSL needs a verified absolute height and a geoid model; v0 has neither.
    amsl_m: Optional[float] = None
    amsl_verified: bool = False


class TakeoffCandidate(_Model):
    """A place to arm and take off from (aerosplat.mission.takeoff), checked like a plan's home."""

    rank: int = Field(ge=1)
    enu: Vec3                               # on the measured ground (column centre)
    distance_m: float                       # horizontally, to the target
    open_radius_m: float                    # ≈ open ground around it (±0.1 m; capped)
    # ENU height of a transit from here to the target, and how far above home that is
    # (None for `TakeoffSearch.best_open`, which was never checked).
    transit_z: Optional[float] = None
    climb_m: Optional[float] = None
    above_dtm_m: float                      # surface minus the semantics DTM
    notes: list[str] = Field(default_factory=list)


class TakeoffSearch(_Model):
    """Every candidate found, or an honest account of why there is none."""

    target: Vec3                            # where the transit ends: E, or the pad
    # How the planner's approach put `target` there ("" : the direct approach, or a given point).
    approach_note: str = ""
    radius_m: float                         # takeoff corridor r = r_drone + m + k·σ_transit
    relief_m: float                         # ground bumps the takeoff leg ignores
    sigma_transit_m: float                  # hypot(σ_georef, σ_GNSS)
    max_distance_m: float
    spacing_m: float                        # candidates are at least this far apart
    ground_tol_m: float
    grid_limited: bool                      # the search disk reaches beyond the grid
    searched_columns: int
    rejected: dict[str, int]                # first failed test -> columns
    checked: int                            # candidates that went through the exact checks
    candidates: list[TakeoffCandidate]
    best_open: Optional[TakeoffCandidate] = None    # the most open ground, when none qualifies
    sigma_gnss_would_fit_m: Optional[float] = None  # ... and the σ_GNSS that would let it
    message: str


class OccupancyRef(_Model):
    """Which twin the plan was checked against (DOMAIN §3 p. 6: reproducible)."""

    run_id: str
    epoch_id: str
    format: str
    res_m: float
    unknown_mode: Optional[str] = None
    carved: bool = False


class Approach(_Model):
    kind: str                               # "opening" | "pad"
    center: Vec3                            # c
    normal: Vec3                            # n: outward for an opening, up for a pad
    direction: Vec3                         # a: from the target towards E (n, or tilted in the cone)
    tilt_deg: float                         # angle between a and n
    hover: Vec3                             # H
    entry: Vec3                             # E
    heading_deg: Optional[float] = None     # compass yaw at H (towards the opening); None: free
    candidates: int = 1                     # directions tried
    candidate_clearance_m: list[tuple[float, float, float]] = Field(default_factory=list)
    # Directions of the cone not tried: E or H closer to the grid's edge than r + 2 voxels.
    candidates_outside_grid: int = 0


class Leg(_Model):
    """A piece of the path and how it was checked."""

    name: str                       # takeoff transit descent approach terminal rtl_climb rtl_return
    points: list[Vec3]
    radius_m: float                         # required clearance: r of §7.2 for this leg
    sigma_m: float
    excluded_surface: Optional[str] = None  # "target" / "home": the surface the leg ends on
    # Lower bound over the whole leg. Where a surface is excluded the exact search stops just
    # beyond the radius, so there a value above radius_m means "at least that".
    min_clearance_m: Optional[float] = None
    blocked_state: Optional[str] = None     # the path itself enters occupied / unknown
    assumed_free: bool = False
    risk_only: bool = False                 # a contingency (RTL climb): findings are risks


class MissionItem(_Model):
    """One MAVLink mission command. Positional ones carry their Site ENU point; the export
    turns it into lat / lon / altitude in the plan's frame and checks the round trip."""

    command: int                            # MAV_CMD
    role: str
    enu: Optional[Vec3] = None
    param1: float = 0.0
    param2: float = 0.0
    param3: float = 0.0
    param4: float = 0.0


class Fence(_Model):
    polygon_enu: list[tuple[float, float]]  # inclusion polygon, counter-clockwise
    alt_max_rel_m: float                    # FENCE_ALT_MAX: above home
    radius_m: float                         # FENCE_RADIUS: circle about home holding the polygon
    margin_m: float


class MissionPlan(_Model):
    """`missions/<id>/mission.json`: everything needed to rebuild, review and export a mission."""

    schema_version: int = Field(SCHEMA, alias="schema")
    id: str = Field(pattern=r"^msn_[0-9A-HJKMNP-TV-Z]{26}$")
    site_key: str
    created_by: str
    created_at: AwareDatetime = Field(default_factory=utcnow)
    target: MissionTarget
    access_point: AccessPoint
    occupancy: OccupancyRef
    profile: PlanningProfile
    sigma: SigmaUsed
    altitude_frame: AltitudeFrame
    home: HomePoint
    transit_z: float                        # ENU height of transit
    approach: Approach
    legs: list[Leg]
    items: list[MissionItem]
    fence: Fence
    rally_enu: list[Vec3]
    verdict: Verdict
    findings: list[Finding]

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.blocking]

    @property
    def risks(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.risk]


def verdict_of(findings: list[Finding]) -> Verdict:
    if any(f.severity is Severity.blocking for f in findings):
        return Verdict.infeasible
    if findings:
        return Verdict.feasible_with_risks
    return Verdict.feasible
