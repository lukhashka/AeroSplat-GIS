"""Scenario files for the L2 runner (SIMULATION_AND_DELIVERY §4.5): what to fly, what goes
wrong on the way, and what must come out.

    scenario: street-window-gps-loss
    site: demo/street
    scene: street                       # a synthetic scene (tests/mission_scenes.py), or omitted
    target: window
    mission: missions/msn_...           # export folder, relative to this file (or msn_ id + site)
    occupancy: occupancy/               # the twin to judge against (default: the plan's run)
    vehicle: sitl-quad                  # configs/sim/vehicles/<name>.yaml
    ardupilot: {tag: Copter-4.7.2}      # must be the pinned build
    environment:
      wind: {speed_mps: 3, dir_deg: 240, turbulence: light}
    faults:
      - {fault: gps_loss, after: hover, at_s: 3}
    expect:
      outcome: abort                    # complete / abort / refused / violation
      reason: gps_loss
      modes: [LAND]                     # optional: must agree with what the .parm makes of it
    seeds: 1

Faults are abstract; `aerosplat.sim.faults` turns them into SIM_* parameters of the pinned
build. Times are simulation seconds (the autopilot's clock), counted from mission start or
from the moment the vehicle starts flying to the mission item named in `after`.

A run that ends in the expected refusal or abort is a pass (§4.6): the outcome and its
reason are part of the expectation, not a side effect of it. Anything the L2 level cannot
honour — actors, sky-view GNSS — is refused when the file is read, never silently ignored.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aerosplat.sim.vehicle import COPTER_MODES


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FaultKind(str, Enum):
    gps_loss = "gps_loss"
    rc_loss = "rc_loss"
    gcs_loss = "gcs_loss"
    battery_sag = "battery_sag"
    wind_gust = "wind_gust"
    motor_degraded = "motor_degraded"


class Outcome(str, Enum):
    complete = "complete"       # flew the mission, reached H, landed at home
    abort = "abort"             # a failsafe took over as the .parm says, and the vehicle came down
    refused = "refused"         # the runner would not fly it: preflight caught the problem
    violation = "violation"     # flew, and the named oracle caught what was wrong


class Invariant(str, Enum):
    clearance = "clearance"             # distance of the TRUE position to blocked space
    fence = "fence"                     # inside the inclusion polygon, under FENCE_ALT_MAX
    hover = "hover"                     # reached H and held it
    heading = "heading"                 # faced the target at H (CONDITION_YAW)
    altitude_frame = "altitude_frame"   # flown heights are the planned heights above home
    failsafe = "failsafe"               # the mode sequence the .parm prescribes, in time
    completion = "completion"           # mission completed, or aborted as expected; came down


# Why a run ends the way it does: the reasons each outcome may name.
ABORT_REASONS = {"fence_breach", "gps_loss", "rc_loss", "gcs_loss", "battery_low",
                 "battery_critical"}
REFUSAL_REASONS = {"unknown_parameter", "parameter_value", "altitude_frame", "mission_mismatch",
                   "mission_infeasible", "firmware_mismatch", "package_incomplete",
                   "upload_mismatch", "fault_unavailable", "home_mismatch"}
FAULT_REASONS = {FaultKind.gps_loss: "gps_loss", FaultKind.rc_loss: "rc_loss",
                 FaultKind.gcs_loss: "gcs_loss"}
TURBULENCE_MPS = {"none": 0.0, "light": 1.0, "moderate": 2.0, "severe": 4.0}
DEFAULT_INVARIANTS = {
    Outcome.complete: (Invariant.clearance, Invariant.fence, Invariant.hover, Invariant.heading,
                       Invariant.altitude_frame, Invariant.failsafe, Invariant.completion),
    Outcome.abort: (Invariant.clearance, Invariant.fence, Invariant.altitude_frame,
                    Invariant.failsafe, Invariant.completion),
    Outcome.violation: (Invariant.clearance, Invariant.fence, Invariant.altitude_frame,
                        Invariant.failsafe, Invariant.completion),
    Outcome.refused: (),
}


class Fault(_Model):
    """One abstract fault. Only the fields of its kind may be given."""

    fault: FaultKind
    at_s: float = Field(ge=0, le=3600)
    # "start" (MISSION_START accepted), the role of a mission waypoint (mission.json
    # `items[].role`, e.g. "over_entry", "hover": the clock starts when the vehicle starts flying
    # to it), or "<role>:reached" (when the autopilot reports it reached).
    after: str = "start"
    duration_s: Optional[float] = Field(None, gt=0, le=3600)   # None: to the end of the run
    level: Optional[str] = None                                # battery_sag: low / critical
    voltage_v: Optional[float] = Field(None, gt=0, le=60)      # battery_sag: explicit voltage
    speed_mps: Optional[float] = Field(None, ge=0, le=40)      # wind_gust
    dir_deg: Optional[float] = Field(None, ge=0, lt=360)       # wind_gust: from, true north
    turbulence_mps: Optional[float] = Field(None, ge=0, le=20)  # wind_gust
    motor: Optional[int] = Field(None, ge=1, le=8)             # motor_degraded: 1-based output
    thrust: Optional[float] = Field(None, ge=0, le=1)          # motor_degraded: multiplier

    @model_validator(mode="after")
    def _fields_of_kind(self) -> "Fault":
        allowed = {FaultKind.battery_sag: {"level", "voltage_v"},
                   FaultKind.wind_gust: {"speed_mps", "dir_deg", "turbulence_mps"},
                   FaultKind.motor_degraded: {"motor", "thrust"}}.get(self.fault, set())
        given = {k for k in ("level", "voltage_v", "speed_mps", "dir_deg", "turbulence_mps",
                             "motor", "thrust") if getattr(self, k) is not None}
        extra = given - allowed
        if extra:
            raise ValueError(f"{self.fault.value} does not take {', '.join(sorted(extra))}")
        if self.fault is FaultKind.battery_sag:
            if (self.level is None) == (self.voltage_v is None):
                raise ValueError("battery_sag needs exactly one of level (low / critical) or voltage_v")
            if self.level is not None and self.level not in ("low", "critical"):
                raise ValueError(f"battery_sag level '{self.level}' is not low / critical")
        if self.fault is FaultKind.wind_gust and self.speed_mps is None:
            raise ValueError("wind_gust needs speed_mps")
        if self.fault is FaultKind.motor_degraded and (self.motor is None or self.thrust is None):
            raise ValueError("motor_degraded needs motor and thrust")
        role, _, when = self.after.partition(":")
        if not role or any(c.isspace() for c in self.after) or when not in ("", "reached")                 or (role == "start" and when):
            raise ValueError(f"after '{self.after}' must be 'start', a waypoint role, or "
                             f"'<role>:reached'")
        return self

    @property
    def reason(self) -> Optional[str]:
        """The abort reason this fault would cause, if any."""
        if self.fault is FaultKind.battery_sag:
            return "battery_critical" if self.level == "critical" else "battery_low"
        return FAULT_REASONS.get(self.fault)


class Wind(_Model):
    speed_mps: float = Field(0.0, ge=0, le=40)
    dir_deg: float = Field(0.0, ge=0, lt=360)        # where the wind comes from, true north
    turbulence: Union[float, str] = 0.0              # m/s, or none / light / moderate / severe

    @field_validator("turbulence")
    @classmethod
    def _turbulence(cls, v: Union[float, str]) -> Union[float, str]:
        if isinstance(v, str) and v not in TURBULENCE_MPS:
            raise ValueError(f"turbulence '{v}' is not one of {', '.join(TURBULENCE_MPS)} or m/s")
        if not isinstance(v, str) and not 0 <= v <= 20:
            raise ValueError("turbulence must be within 0..20 m/s")
        return v

    @property
    def turbulence_mps(self) -> float:
        return TURBULENCE_MPS[self.turbulence] if isinstance(self.turbulence, str) else float(self.turbulence)


class Environment(_Model):
    wind: Wind = Wind()
    # Sky-view GNSS modulation is an L3 feature (SIM-2): refused here, not ignored.
    gnss: Optional[dict] = None

    @field_validator("gnss")
    @classmethod
    def _no_gnss_model(cls, v: Optional[dict]) -> Optional[dict]:
        if v:
            raise ValueError("environment.gnss (sky-view GNSS model) is not available at L2 (SIM-2)")
        return v


class ArduPilotSpec(_Model):
    tag: str
    params: list[str] = Field(default_factory=list)   # extra .parm files, relative to the YAML


class Tolerances(_Model):
    hover_m: float = Field(1.0, gt=0, le=10)          # horizontal distance to H while holding
    hover_alt_m: float = Field(0.5, gt=0, le=10)      # vertical distance to H while holding
    hover_hold_s: float = Field(3.0, ge=0, le=600)    # how long within tolerance, at least
    approach_speed_ratio: float = Field(1.3, ge=1, le=10)   # approach to H vs planned speed
    heading_deg: float = Field(10.0, gt=0, le=180)
    altitude_m: float = Field(1.0, gt=0, le=20)       # flown vs planned height above home
    react_s: float = Field(5.0, gt=0, le=120)         # failsafe mode after its trigger delay
    land_m: float = Field(1.5, gt=0, le=50)           # final position vs home (RTL / complete)
    fence_overshoot_m: float = Field(10.0, ge=0, le=100)   # a breach scenario: how far out


class Expect(_Model):
    outcome: Outcome
    reason: str
    modes: Optional[list[str]] = None          # the failsafe mode(s) the .parm should give
    invariants: Optional[list[Invariant]] = None
    # Required clearance of the TRUE position: r_drone + this (default: the smaller of the
    # class's minimum clearance and the profile's transit margin, from mission.json).
    min_clearance_m: Optional[float] = Field(None, ge=0, le=50)
    tolerances: Tolerances = Tolerances()

    @model_validator(mode="after")
    def _consistent(self) -> "Expect":
        o, r = self.outcome, self.reason
        if o is Outcome.complete and r != "delivered":
            raise ValueError("outcome complete has reason 'delivered'")
        if o is Outcome.abort and r not in ABORT_REASONS:
            raise ValueError(f"abort reason '{r}' is not one of {sorted(ABORT_REASONS)}")
        if o is Outcome.refused and r not in REFUSAL_REASONS:
            raise ValueError(f"refusal reason '{r}' is not one of {sorted(REFUSAL_REASONS)}")
        if o is Outcome.violation:
            names = {i.value for i in Invariant}
            if r not in names:
                raise ValueError(f"violation reason '{r}' must name an invariant: {sorted(names)}")
            if self.invariants is not None and Invariant(r) not in self.invariants:
                raise ValueError(f"violation of '{r}' is expected, but '{r}' is not checked")
        if self.modes is not None:
            if o not in (Outcome.abort,):
                raise ValueError("modes are the failsafe modes of an abort")
            bad = [m for m in self.modes if m not in COPTER_MODES]
            if bad or not self.modes:
                raise ValueError(f"unknown Copter modes {bad}" if bad else "modes is empty")
        return self

    @property
    def checked(self) -> tuple[Invariant, ...]:
        return tuple(self.invariants) if self.invariants is not None else DEFAULT_INVARIANTS[self.outcome]


class Scenario(_Model):
    scenario: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,80}$")
    description: str = ""
    site: Optional[str] = None
    epoch: Optional[str] = None
    scene: Optional[str] = None                 # synthetic scene name (label; built by the caller)
    variant: Optional[str] = None               # how the caller altered the scene / mission
    target: Optional[str] = None                # tgt_ id, or a scene's target name
    handover: Optional[str] = None              # checked against the mission's target
    mission: Optional[str] = None               # export folder (relative to the YAML) or msn_ id
    occupancy: Optional[str] = None             # occupancy folder (relative to the YAML)
    origin: Optional[dict[str, float]] = None   # {lat, lon, h}: site origin, if not in the .plan
    vehicle: str = "sitl-quad"
    ardupilot: Optional[ArduPilotSpec] = None
    environment: Environment = Environment()
    # People, animals and cars need the L3 renderer and dynamic occupancy (SIM-4).
    actors: list[dict] = Field(default_factory=list)
    faults: list[Fault] = Field(default_factory=list)
    expect: Expect
    seeds: int = Field(1, ge=1, le=1000)
    speedup: Optional[float] = Field(None, gt=0, le=10)
    timeout_s: float = Field(600.0, gt=0, le=7200)  # simulation seconds after mission start

    @field_validator("actors")
    @classmethod
    def _no_actors(cls, v: list) -> list:
        if v:
            raise ValueError("actors need the L3 pipeline (SIM-4); an L2 scenario has none")
        return v

    @field_validator("origin")
    @classmethod
    def _origin(cls, v: Optional[dict[str, float]]) -> Optional[dict[str, float]]:
        if v is not None and set(v) != {"lat", "lon", "h"}:
            raise ValueError("origin is {lat, lon, h}")
        return v

    @model_validator(mode="after")
    def _faults_vs_expectation(self) -> "Scenario":
        by_kind: dict[tuple[FaultKind, str], list[Fault]] = {}
        for f in self.faults:
            by_kind.setdefault((f.fault, f.after), []).append(f)
        for (kind, after), fs in by_kind.items():
            fs = sorted(fs, key=lambda f: f.at_s)
            for a, b in zip(fs, fs[1:]):
                if a.duration_s is None or a.at_s + a.duration_s > b.at_s:
                    raise ValueError(f"two {kind.value} faults after '{after}' overlap")
        e = self.expect
        # Whether a fault leads to an abort at all is the .parm's business (FS_OPTIONS can make
        # a vehicle carry on); the runner checks that against the expectation before flying.
        if e.outcome is Outcome.abort and e.reason != "fence_breach":
            if not any(f.reason == e.reason for f in self.faults):
                raise ValueError(f"abort '{e.reason}' expected, but no fault causes it")
        return self


def load_scenario(path: Path) -> Scenario:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: not a scenario mapping")
    return Scenario.model_validate(data)


def resolve(base: Path, ref: Optional[str]) -> Optional[Path]:
    """A path of the scenario file, relative to the file's folder."""
    if ref is None:
        return None
    p = Path(ref).expanduser()
    return p if p.is_absolute() else (base / p)
