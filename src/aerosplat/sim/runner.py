"""L2 scenario runner (SIMULATION_AND_DELIVERY §4.2 level L2, step SIM-1).

    python -m aerosplat.sim.runner scenario.yaml [--mission DIR] [--occupancy DIR] [--out DIR]

A mission exported by `aerosplat.mission` is flown in the pinned ArduCopter SITL (built-in
physics) and judged on the simulator's TRUE state against the twin (`aerosplat.sim.oracles`).

1. **Preflight, offline.** The export folder must be complete and self-consistent: the frames
   of every item, `.waypoints` = `.plan` = `mission.json` (positions within 1 cm), the fence of
   the `.plan` = the plan's fence, every `.parm` name and value valid for the pinned build
   (`aerosplat.sim.params`). Each fault must be injectable, and the scenario's expected abort
   must be what the `.parm` prescribes. Any problem is a **refusal** with a named reason.
2. **SITL.** A fresh instance (wiped parameters) with home = the mission's home row; or a
   given endpoint, flown only if its vehicle stands at that home.
3. **Upload.** The running build must be the pin (AUTOPILOT_VERSION). Vehicle profile, runner
   and environment parameters, then the `.parm` — each set confirmed by the vehicle, and a name
   the live table lacks is a refusal. Fence, rally points and mission over the mission
   protocol, each downloaded again and compared.
4. **Flight.** GUIDED, arm, MISSION_START; faults injected on schedule in simulation time;
   truth (SIM_STATE), estimate, modes, mission progress, STATUSTEXT, fence and EKF status
   recorded until the vehicle has landed and disarmed, or the timeout.
5. **Judgement and report.** Oracles on the truth, the expected outcome, and
   `report.html` + `report.json` (+ `trajectory.json`) next to the export: `<mission>/sim/<scenario>/`.

A run that ends in the expected refusal or abort is a pass (§4.6).

**GNSS-free packages (SIM-2n, ADR-0015).** The package's navigation profile must be implemented
by its `.parm` (else `mission_mismatch`), and the scenario must fly the same profile. SITL boots
with the vehicle's sensor drivers, the package's navigation section and a world without GNSS
(an extra defaults file; every value read back, else `nav_config`). The companion emulator
(`aerosplat.sim.extnav`) attaches at once — EKF3 keeps the first origin it gets — and the run
waits for the origin to read back as the Site origin. Arming is then tried even if pre-arm
never passed: an autopilot that refuses is the expected outcome `arming_refused` of negative
scenarios. Parameters are checked against the extnav snapshot of the pinned build.

**VPS in the loop (SIM-3).** With `nav.vps` the companion is `aerosplat.sim.vpsloop.VpsCompanion`:
the package must name its atlas (plan `vps`, file present) and take no velocity from ExternalNav,
the vehicle must carry the camera the atlas was rendered for, and the run flies at speedup 1. A
companion that refuses its atlas streams nothing; arming is still tried, and the autopilot's
refusal makes the outcome `refused: atlas_mismatch` (arming anyway is a run error).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from aerosplat.geo.crs import EnuFrame, GeoPoint
from aerosplat.mission.export import (BOOT_PARAMS, POSITIONAL, WireItem, enu_from_geodetic,
                                      nav_parameters, nav_problems, read_parm, read_plan,
                                      read_waypoints, wire_to_enu)
from aerosplat.mission.models import (AltitudeFrameError, ExtNavSpec, GnssUse, MissionPlan,
                                      Verdict as PlanVerdict)
from aerosplat.paths import PROJECT_DIR
from aerosplat.sim import mavmission
from aerosplat.sim.airframe import (Airframe, CameraMount, Companion as CompanionBoard, Flow, Frame,
                                    NavHardware, Rangefinder)
from aerosplat.sim.extnav import Companion
from aerosplat.sim.faults import (FLAT_GROUND, RUNNER_PARAMS, RUNNER_SYSID, FaultUnavailable,
                                  Injection, Response, failsafe_trigger_problems, injection,
                                  response, environment_params)
from aerosplat.sim.oracles import (FAIL, PASS, ClearanceSeries, Context, NavContext, Verdict,
                                   actual_outcome, clearance_series, expected_status, judge)
from aerosplat.sim.params import FirmwareParams, ParameterError, load_snapshot
from aerosplat.sim.record import Recording
from aerosplat.sim.scenario import Fault, NavSim, Outcome, Scenario, load_scenario, resolve
from aerosplat.sim.sitl import Home, Sitl, SitlConfig, SitlError, load_config
from aerosplat.sim.vehicle import (Vehicle, VehicleError, VehicleTimeout, horizontal_distance_m,
                                   mode_name, truth_from_sim_state)
from aerosplat.surface.occupancy import OccupancyGrid, read_occupancy

VEHICLES_DIR = PROJECT_DIR / "configs" / "sim" / "vehicles"
PLAN_TOL_M = 0.01            # .waypoints / .plan / mission.json agree within this
FENCE_TOL_M = 0.05           # the .plan fence vs the plan's polygon (9-decimal lat/lon)
HOME_TOL_M = 1.0             # the vehicle stands this close to the mission's home
TAIL_S = 2.0                 # simulation seconds recorded after the disarm
MSG_SYSTEM_TIME, MSG_MISSION_CURRENT, MSG_FENCE_STATUS = 2, 42, 162
MSG_EKF_STATUS_REPORT, MSG_HOME_POSITION = 193, 242
MSG_GPS_RAW_INT, MSG_GPS_GLOBAL_ORIGIN = 24, 49
MSG_ATTITUDE, MSG_DISTANCE_SENSOR, MSG_EXTENDED_SYS_STATE = 30, 132, 245
# GNSS-free runs (SIM-2n), wall-clock seconds unless named otherwise.
ORIGIN_WAIT_S = 60.0         # for the EKF origin to read back after the companion sends it
NAV_PREARM_S = 45.0          # for pre-arm; then arming is tried anyway (negative scenarios)
NAV_ARM_S = 20.0             # arming attempts before "the autopilot refused"
GNSS_LOCK_S = 30.0           # gnss monitor: SITL receiver's first fix, simulation seconds
REACHED_TEXT = re.compile(r"^Reached command #(\d+)")


class Refusal(Exception):
    """The runner will not fly this: `reason` is one of scenario.REFUSAL_REASONS."""

    def __init__(self, reason: str, detail: str):
        self.reason, self.detail = reason, detail
        super().__init__(f"{reason}: {detail}")


class ScenarioError(ValueError):
    """The scenario contradicts itself or the package (a broken test, never a pass)."""


class RunError(RuntimeError):
    """The run could not be carried out (SITL down, link lost, arming impossible)."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VehicleProfile(_Model):
    vehicle: str
    description: str = ""
    params: dict[str, float] = {}
    # Must hold from power-on (sensor drivers): SITL boots with them, the runner reads them back.
    boot_params: dict[str, float] = {}
    camera: Optional[str] = None            # configs/sim/cameras: "<camera>:<lens>" (data)
    # The real airframe the profile stands for (aerosplat.sim.airframe): data, never sent to SITL.
    frame: Optional[Frame] = None
    cameras: tuple[CameraMount, ...] = ()
    rangefinders: tuple[Rangefinder, ...] = ()
    flow: Optional[Flow] = None
    companion: Optional[CompanionBoard] = None
    nav: Optional[NavHardware] = None
    # Autopilot parameters of the real vehicle only (the SITL drivers stay in boot_params).
    hardware_params: dict[str, float] = {}

    @property
    def airframe(self) -> Airframe:
        return Airframe(frame=self.frame, cameras=self.cameras, rangefinders=self.rangefinders,
                        flow=self.flow, companion=self.companion, nav=self.nav)

    @model_validator(mode="after")
    def _consistent(self) -> "VehicleProfile":
        bad = self.airframe.check(self.camera)
        if bad:
            raise ValueError("; ".join(bad))
        return self


def load_vehicle(name: str, folder: Path = VEHICLES_DIR) -> VehicleProfile:
    path = folder / f"{name}.yaml"
    if not path.is_file():
        raise ScenarioError(f"no vehicle profile '{name}' ({path})")
    prof = VehicleProfile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if prof.vehicle != name:
        raise ScenarioError(f"{path} describes vehicle '{prof.vehicle}', not '{name}'")
    return prof


# ── the package ──────────────────────────────────────────────────────────
@dataclass
class MissionPackage:
    """An export folder, read and cross-checked (step 1 of the module docstring)."""

    dir: Path
    plan: MissionPlan
    origin: GeoPoint
    items: list[WireItem]                       # .waypoints, seq 0 = home
    parm: dict[str, float]
    fence_latlon: list[tuple[float, float]]
    rally: list[tuple[float, float, float]]
    planned_enu: dict[int, tuple[float, float, float]]
    fence_enu: list[tuple[float, float]]

    @property
    def home(self) -> WireItem:
        return self.items[0]

    def role_seq(self, role: str) -> Optional[int]:
        return next((k for k, it in enumerate(self.plan.items, start=1) if it.role == role), None)

    @classmethod
    def load(cls, folder: Path, origin: Optional[GeoPoint] = None) -> "MissionPackage":
        folder = Path(folder)
        files = {n: folder / n for n in ("mission.json", "mission.waypoints", "mission.plan",
                                         "mission.parm")}
        missing = [n for n, p in files.items() if not p.is_file()]
        if missing:
            raise Refusal("package_incomplete", f"{folder}: missing {', '.join(missing)}")
        try:
            plan = MissionPlan(**json.loads(files["mission.json"].read_text(encoding="utf-8")))
            plan_doc = json.loads(files["mission.plan"].read_text(encoding="utf-8"))
            parm = read_parm(files["mission.parm"].read_text(encoding="utf-8"))
        except (ValueError, KeyError) as e:
            raise Refusal("package_incomplete", f"{folder}: unreadable: {e}") from None
        if origin is None:
            o = (plan_doc.get("aerosplat") or {}).get("origin")
            if not o:
                raise Refusal("package_incomplete", "the .plan has no site origin (exported before "
                                                    "SIM-1); give the scenario an `origin`")
            origin = GeoPoint(**o)
        try:
            items = read_waypoints(files["mission.waypoints"].read_text(encoding="utf-8"))
            plan_items = read_plan(files["mission.plan"].read_text(encoding="utf-8"))
        except AltitudeFrameError as e:
            raise Refusal("altitude_frame", str(e)) from None
        except ValueError as e:
            raise Refusal("package_incomplete", f"unreadable mission file: {e}") from None
        planned = _compare_with_plan(plan, origin, items)
        _items_on_legs(plan)
        diffs = mavmission.compare([mavmission.Item.from_wire(i) for i in items],
                                   [mavmission.Item.from_wire(i) for i in plan_items], ".plan vs .waypoints")
        if diffs:
            raise Refusal("mission_mismatch", "; ".join(diffs[:5]))
        polys = (plan_doc.get("geoFence") or {}).get("polygons") or []
        if len(polys) != 1 or not polys[0].get("inclusion"):
            raise Refusal("mission_mismatch", f"the .plan holds {len(polys)} fence polygons; v0 "
                                              f"exports exactly one inclusion polygon")
        fence_latlon = [(float(a), float(b)) for a, b in polys[0]["polygon"]]
        frame = EnuFrame(origin)
        ll = np.asarray(fence_latlon, float)
        fence_enu_a = enu_from_geodetic(frame, ll[:, 0], ll[:, 1], np.full(len(ll), plan.home.h_ellipsoid))
        want = np.asarray(plan.fence.polygon_enu, float)
        if want.shape != fence_enu_a[:, :2].shape or np.abs(fence_enu_a[:, :2] - want).max() > FENCE_TOL_M:
            raise Refusal("mission_mismatch", "the .plan fence is not the plan's fence polygon")
        rally = [(float(a), float(b), float(c)) for a, b, c in
                 ((plan_doc.get("rallyPoints") or {}).get("points") or [])]
        return cls(dir=folder, plan=plan, origin=origin, items=items, parm=parm,
                   fence_latlon=fence_latlon, rally=rally, planned_enu=planned,
                   fence_enu=[(float(x), float(y)) for x, y in fence_enu_a[:, :2]])


def _compare_with_plan(plan: MissionPlan, origin: GeoPoint,
                       items: list[WireItem]) -> dict[int, tuple[float, float, float]]:
    """The .waypoints must be the plan: same commands, every position within PLAN_TOL_M. A
    vertical-only mismatch is an altitude frame error (an AMSL height in a relative item)."""
    if len(items) != len(plan.items) + 1:
        raise Refusal("mission_mismatch", f"{len(items) - 1} items in .waypoints, "
                                          f"{len(plan.items)} in mission.json")
    back = wire_to_enu(items, origin)
    for seq, it in enumerate(plan.items, start=1):
        w = items[seq]
        if w.command != it.command:
            raise Refusal("mission_mismatch", f"item {seq}: command {w.command} in .waypoints, "
                                              f"{it.command} ({it.role}) in mission.json")
        if it.enu is None:
            continue
        d = np.asarray(back[seq]) - np.asarray(it.enu, float)
        dxy, dz = float(np.linalg.norm(d[:2])), float(d[2])
        if dxy > PLAN_TOL_M:
            raise Refusal("mission_mismatch", f"item {seq} ({it.role}) is {dxy:.2f} m from its "
                                              f"planned position")
        if abs(dz) > PLAN_TOL_M:
            raise Refusal("altitude_frame", f"item {seq} ({it.role}) is at {w.alt:.2f} m above home, "
                                            f"planned {it.enu[2] - plan.home.enu[2]:.2f} m "
                                            f"({dz:+.2f} m): wrong altitude frame?")
    return back


def _items_on_legs(plan: MissionPlan) -> None:
    """Every waypoint of mission.json lies on a leg the planner checked (the LAND item: under the
    start of the take-off leg). A waypoint edited after planning is refused here; a plan whose
    legs were computed wrongly passes, and is the flight oracles' business."""
    from aerosplat.sim.oracles import _dist_to_segments

    legs = [np.asarray(leg.points, float) for leg in plan.legs if len(leg.points) >= 2]
    home = np.asarray(plan.home.enu, float)
    for seq, it in enumerate(plan.items, start=1):
        if it.enu is None:
            continue
        p = np.asarray(it.enu, float)
        if it.command == 21 and np.linalg.norm(p - home) <= PLAN_TOL_M:
            continue
        d = min(float(_dist_to_segments(p[None], pts)[0]) for pts in legs) if legs else math.inf
        if d > PLAN_TOL_M:
            raise Refusal("mission_mismatch", f"item {seq} ({it.role}) at {tuple(round(v, 2) for v in p)}"
                                              f" is {d:.2f} m off every checked leg of mission.json")


# ── SITL ─────────────────────────────────────────────────────────────────
class SitlProvider:
    """Where runs fly.

    Default: one SITL instance this provider starts afresh for every run, with the run's home,
    and stops when closed — wiped parameters, vehicle at home, nothing left from the last run.
    It never stops an instance it did not start. With `endpoint`, a vehicle someone else runs
    is flown as it is, and only when it stands at the run's home."""

    def __init__(self, cfg: Optional[SitlConfig] = None, instance: int = 1,
                 endpoint: Optional[str] = None):
        self.cfg = cfg or load_config()
        self.sitl = Sitl(self.cfg, instance)
        self.endpoint = endpoint
        self.started = False
        self.home: Optional[Home] = None

    def prepare(self, home: Home, speedup: float, boot_defaults: tuple[Path, ...] = ()) -> str:
        """`boot_defaults`: parameter files SITL boots with (GNSS-free runs). A given endpoint
        cannot be rebooted: the runner then checks what the vehicle holds instead."""
        if self.endpoint:
            return self.endpoint
        if self.started:
            self.sitl.stop()
            self.started = False
        elif self.sitl.status() is not None:
            raise SitlError(f"SITL instance {self.sitl.instance} is running and was not started by "
                            f"this runner: stop it (python -m aerosplat.sim.sitl -I "
                            f"{self.sitl.instance} stop) or use another instance")
        self.sitl.start(home, speedup, boot_defaults=boot_defaults)
        self.started, self.home = True, home
        return self.sitl.endpoint()

    def close(self) -> None:
        if self.started:
            self.sitl.stop()
            self.started = False

    def __enter__(self) -> "SitlProvider":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ── results ──────────────────────────────────────────────────────────────
@dataclass
class RunResult:
    scenario: Scenario
    seed: int
    outcome: str                              # complete / abort / refused / incomplete / error
    reason: Optional[str] = None
    detail: str = ""
    verdicts: list[Verdict] = field(default_factory=list)
    expected: dict[str, str] = field(default_factory=dict)
    passed: bool = False
    summary: str = ""
    preflight: dict[str, Any] = field(default_factory=dict)
    recording: Optional[Recording] = None
    clearance: Optional[ClearanceSeries] = None
    package: Optional[MissionPackage] = None
    grid: Optional[OccupancyGrid] = None
    injected: list[dict] = field(default_factory=list)
    applied: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    firmware: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    response: Optional[Response] = None
    out_dir: Optional[Path] = None

    def verdict(self, name: str) -> Optional[Verdict]:
        return next((v for v in self.verdicts if v.invariant == name), None)


def decide(result: RunResult) -> None:
    """passed / summary from the outcome and the verdicts, against the expectation."""
    e = result.scenario.expect
    if e.outcome is Outcome.refused:
        result.passed = result.outcome == "refused" and result.reason == e.reason
        result.summary = (f"refused as expected: {result.reason}" if result.passed else
                          f"expected refusal '{e.reason}', got {result.outcome}"
                          + (f" ({result.reason}: {result.detail})" if result.reason else ""))
        return
    if result.outcome in ("refused", "error"):
        result.passed = False
        result.summary = f"{result.outcome}: {result.reason}: {result.detail}"
        return
    want_outcome = Outcome.abort.value if e.outcome is Outcome.abort else Outcome.complete.value
    result.expected = {v.invariant: expected_status(v.invariant, e) for v in result.verdicts}
    wrong = [v for v in result.verdicts if v.status != result.expected[v.invariant]
             and not (v.status == "n/a" and result.expected[v.invariant] == PASS)]
    problems = [f"{v.invariant}: expected {result.expected[v.invariant]}, got {v.status} ({v.detail})"
                for v in wrong]
    if result.outcome != want_outcome:
        problems.insert(0, f"outcome {result.outcome}, expected {want_outcome}")
    result.passed = not problems
    if result.passed:
        flagged = [v.invariant for v in result.verdicts if v.status == FAIL]
        result.summary = {Outcome.complete: "delivered: all invariants hold",
                          Outcome.abort: f"aborted as the package prescribes ({e.reason})",
                          Outcome.violation: f"the {e.reason} oracle caught it"}[e.outcome] \
            + (f"; flagged as expected: {', '.join(flagged)}" if flagged else "")
    else:
        result.summary = "; ".join(problems)


# ── the runner ───────────────────────────────────────────────────────────
@dataclass
class _Raw:
    """Truth rows as received (converted to ENU in one go at the end)."""

    t: list[float] = field(default_factory=list)
    lla: list[tuple[float, float, float]] = field(default_factory=list)
    vel: list[tuple[float, float, float]] = field(default_factory=list)
    yaw: list[float] = field(default_factory=list)
    mode: list[str] = field(default_factory=list)
    armed: list[bool] = field(default_factory=list)
    seq: list[int] = field(default_factory=list)
    est: list[Optional[tuple[float, float, float]]] = field(default_factory=list)


class Runner:
    def __init__(self, scenario: Scenario, package: MissionPackage, grid: OccupancyGrid, *,
                 provider: Optional[SitlProvider] = None, fw: Optional[FirmwareParams] = None,
                 speedup: Optional[float] = None, out_dir: Optional[Path] = None,
                 log=print, world: Any = None, base: Optional[Path] = None):
        """`world`: the renderer of the camera's world for VPS in the loop (default: the
        scenario's `nav.vps.run_dir`, loaded when the run starts); `base`: the folder relative
        scenario paths are resolved against (the scenario file's)."""
        self.sc = scenario
        self.pkg = package
        self.grid = grid
        self.provider = provider
        self.cfg = provider.cfg if provider is not None else load_config()
        self.profile = load_vehicle(scenario.vehicle)
        # Navigation (SIM-2n): the package's profile; the run's VPS model (scenario `nav`, or the
        # package's declared numbers).
        self.navp = package.plan.profile.nav
        self.navsim: Optional[NavSim] = scenario.nav
        if self.navsim is None and self.navp.gnss_free:
            self.navsim = NavSim(gnss=self.navp.gnss)
        # GNSS-free packages and sensor-carrying vehicles name parameters that exist only with
        # their subsystem enabled: checked against the extnav snapshot of the same build.
        self.nav_run = self.navp.gnss_free or self.navsim is not None or bool(self.profile.boot_params)
        self.fw = fw or load_snapshot(self.cfg.ardupilot.tag, self.cfg.ardupilot.commit,
                                      variant="extnav" if self.nav_run else None)
        self.speedup = float(speedup or scenario.speedup or self.cfg.vehicle.speedup)
        self.out_dir = out_dir
        self.log = log
        self.companion: Optional[Companion] = None
        # VPS in the loop (SIM-3)
        self.vps = self.navsim.vps if self.navsim is not None else None
        self.base = Path(base) if base is not None else Path.cwd()
        self.world = world
        if self.vps is not None and self.speedup != 1.0:
            raise ScenarioError(f"VPS in the loop flies at speedup 1, not {self.speedup:g} (the "
                                f"localizer's compute time is real time)")

    # -- step 1 -----------------------------------------------------------
    def preflight(self, seed: int = 0) -> dict[str, Any]:
        """Offline checks; raises Refusal (expected outcomes) or ScenarioError (broken scenario)."""
        sc, pkg, fw, plan = self.sc, self.pkg, self.fw, self.pkg.plan
        if sc.ardupilot is not None and sc.ardupilot.tag != self.cfg.ardupilot.tag:
            raise Refusal("firmware_mismatch", f"the scenario is for {sc.ardupilot.tag}; the pinned "
                                               f"build is {self.cfg.ardupilot.tag}")
        if sc.ardupilot is not None and sc.ardupilot.params:
            raise ScenarioError("ardupilot.params (extra .parm files) are not supported in L2 yet")
        if plan.verdict is PlanVerdict.infeasible:
            raise Refusal("mission_infeasible", "; ".join(f.code.value for f in plan.blocking))
        if sc.handover is not None and sc.handover != plan.target.handover.mode.value:
            raise Refusal("mission_mismatch", f"the scenario is for handover {sc.handover}; the "
                                              f"mission's target uses {plan.target.handover.mode.value}")
        unknown = fw.unknown(pkg.parm)
        if unknown:
            raise Refusal("unknown_parameter", fw.problems({n: pkg.parm[n] for n in unknown})[0])
        bad = fw.problems(pkg.parm)
        if bad:
            raise Refusal("parameter_value", "; ".join(bad))
        try:
            fw.require(self.profile.params, f"vehicle profile {self.profile.vehicle}")
            fw.require(self.profile.boot_params, f"vehicle profile {self.profile.vehicle} (boot)")
        except ParameterError as e:
            raise ScenarioError(str(e)) from None
        self._nav_preflight()
        nav_roles = {it.role for it in plan.items if it.command in POSITIONAL}
        for f in sc.faults:
            if f.after != "start" and f.after.partition(":")[0] not in nav_roles:
                raise ScenarioError(f"fault {f.fault.value} after '{f.after}': the mission has no such "
                                    f"waypoint (roles: {', '.join(sorted(nav_roles))})")
        values = self.vehicle_values({n: i.default for n, i in fw.params.items()})
        injections = self.injections(values, seed)
        resp = self.expected_response(values)
        out = {"firmware": fw.tag, "parm": len(pkg.parm), "faults": [i.note for i in injections],
               "response": None if resp is None else {"trigger": resp.trigger, "modes": list(resp.modes),
                                                       "by": list(resp.by), "delay_s": resp.delay_s,
                                                       "note": resp.note},
               "warnings": failsafe_trigger_problems(pkg.parm, values)}
        if self.nav_run:
            out["nav"] = {"gnss": self.navp.gnss.value, "flow": self.navp.flow,
                          "boot": self.boot_params(), "snapshot": "extnav"}
        return out

    def _nav_preflight(self) -> None:
        """A GNSS-free run needs a package that implements its navigation profile, a scenario
        that flies the same profile, and a vehicle that carries the sensors (SIM-2n)."""
        navp, sim = self.navp, self.navsim
        bad = nav_problems(navp, self.pkg.parm)
        if bad:
            raise Refusal("mission_mismatch", "the package does not implement its navigation "
                                              "profile: " + "; ".join(bad))
        if sim is not None and sim.gnss is not navp.gnss:
            raise Refusal("mission_mismatch", f"the scenario flies gnss {sim.gnss.value}; the "
                                              f"package was planned and exported for gnss "
                                              f"{navp.gnss.value}")
        if navp.gnss_free:
            veh = {**self.profile.boot_params, **self.profile.params}
            if navp.flow and not (veh.get("FLOW_TYPE", 0) and veh.get("RNGFND1_TYPE", 0)):
                raise Refusal("nav_config", f"the package falls back to optical flow; vehicle "
                                            f"{self.profile.vehicle} has no flow sensor and "
                                            f"downward rangefinder (FLOW_TYPE, RNGFND1_TYPE)")
        if self.vps is not None:
            self._vps_preflight()

    def _vps_preflight(self) -> None:
        """VPS in the loop: the package names its atlas and flies on poses only; the vehicle
        carries the camera the atlas was rendered for; the world to render exists."""
        b, ext = self.pkg.plan.vps, self.navp.extnav
        if b is None:
            raise Refusal("mission_mismatch", "VPS in the loop, but the package names no VPS atlas "
                                              "(plan without aerosplat.vps.route)")
        if ext is not None and ext.velocity:
            raise Refusal("mission_mismatch", "the package takes horizontal velocity from "
                                              "ExternalNav (EK3_SRC1_VELXY 6); a VPS fix is a pose only")
        if not (self.pkg.dir / b.atlas.file).is_file():
            raise Refusal("package_incomplete", f"the package names atlas {b.atlas.file}, which is "
                                                f"not in {self.pkg.dir}")
        mount = self.profile.airframe.vps_mount()
        if mount is None or mount.vps is None:
            raise ScenarioError(f"vehicle {self.profile.vehicle} has no VPS camera (cameras[].vps)")
        if (mount.camera, float(mount.pitch_deg), float(mount.yaw_deg)) != \
                (b.atlas.camera, b.atlas.mount_pitch_deg, b.atlas.mount_yaw_deg) \
                or tuple(mount.vps.resolution) != tuple(b.atlas.resolution) \
                or mount.vps.extractor != b.atlas.extractor:
            raise Refusal("mission_mismatch", f"the atlas was rendered for {b.atlas.camera} at "
                                              f"{b.atlas.mount_pitch_deg:g}° {b.atlas.resolution} "
                                              f"{b.atlas.extractor}; vehicle {self.profile.vehicle} "
                                              f"carries {mount.camera} at {mount.pitch_deg:g}° "
                                              f"{mount.vps.resolution} {mount.vps.extractor}")
        assert self.vps is not None
        if self.world is None and not (self._resolve(self.vps.run_dir) / "gs").is_dir():
            raise ScenarioError(f"nav.vps.run_dir {self._resolve(self.vps.run_dir)} is not a site "
                                f"run with a splat")

    def _resolve(self, ref: str) -> Path:
        p = Path(ref).expanduser()
        return p if p.is_absolute() else self.base / p

    def boot_params(self) -> dict[str, float]:
        """What SITL must boot with: the vehicle's sensor drivers, the package's reboot-only
        parameters, and the simulated world (no sky for GNSS; flat ground for flow / range)."""
        if not self.nav_run:
            return {}
        out = dict(self.profile.boot_params)
        # The package's whole navigation section, not only what needs a reboot: EKF3 must start
        # on its sources (switched after boot, the cores had initialised on GPS / compass and
        # SITL refused to arm: "EKF3 Yaw inconsistent 99 deg", "Need Position Estimate").
        nav_names = {n for n, _, _ in nav_parameters(self.navp)} | BOOT_PARAMS
        out.update({n: float(v) for n, v in self.pkg.parm.items() if n in nav_names})
        out.update(FLAT_GROUND)
        if self.navp.gnss is GnssUse.none:
            out["SIM_GPS1_ENABLE"] = 0.0          # the world has no GNSS signal at all
        elif self.navp.gnss is GnssUse.monitor:
            # SITL's receiver has a fix ~1 s after boot; a real cold start takes tens of
            # seconds, which is what lets the companion set the EKF origin before GNSS can.
            out["SIM_GPS1_LCKTIME"] = GNSS_LOCK_S
        return out

    def vehicle_values(self, base: dict[str, float]) -> dict[str, float]:
        """What the vehicle will hold once everything is applied, in that order."""
        out = dict(base)
        out.update(self.boot_params())
        out.update(self.profile.params)
        out.update(RUNNER_PARAMS)
        out.update(environment_params(self.sc.environment, self.fw))
        out.update(self.pkg.parm)
        return out

    def injections(self, values: dict[str, float], seed: int) -> list[Injection]:
        out = []
        for f in self.jittered_faults(seed):
            try:
                out.append(injection(f, self.fw, values, self.sc.environment.wind,
                                     lat_deg=self.pkg.origin.lat))
            except FaultUnavailable as e:
                raise Refusal("fault_unavailable", str(e)) from None
        return out

    def jittered_faults(self, seed: int) -> list[Fault]:
        """Seed 0: the faults as written. Seed k > 0: each time moved by up to ±1 s."""
        if seed == 0:
            return list(self.sc.faults)
        rng = random.Random(seed)
        return [f.model_copy(update={"at_s": max(0.0, f.at_s + rng.uniform(-1.0, 1.0))})
                for f in self.sc.faults]

    def expected_response(self, values: dict[str, float]) -> Optional[Response]:
        e = self.sc.expect
        if e.outcome is not Outcome.abort:
            return None
        try:
            resp = response(e.reason, self.pkg.parm, values,
                            reaction=None if self.navsim is None else self.navsim.reaction)
        except FaultUnavailable as err:
            raise Refusal("fault_unavailable", str(err)) from None
        if resp.continues:
            raise ScenarioError(f"abort on {e.reason} expected, but the package makes the vehicle "
                                f"carry on ({resp.note or ', '.join(resp.by)})")
        if e.modes and not set(e.modes) <= set(resp.modes):
            raise ScenarioError(f"the scenario expects {e.modes} on {e.reason}; the package "
                                f"prescribes {list(resp.modes)} ({', '.join(resp.by)})")
        return resp

    # -- steps 2-5 ----------------------------------------------------------
    def run(self, seed: int = 0) -> RunResult:
        result = RunResult(self.sc, seed, outcome="error", package=self.pkg, grid=self.grid)
        t0 = time.monotonic()
        try:
            result.preflight = self.preflight(seed)
            result.warnings += result.preflight["warnings"]
            if self.provider is None:
                raise RunError("no SITL provider: preflight only")
            self._fly(result, seed)
        except Refusal as r:
            result.outcome, result.reason, result.detail = "refused", r.reason, r.detail
        except (RunError, SitlError, VehicleError, OSError) as e:
            result.outcome, result.reason, result.detail = "error", type(e).__name__, str(e)
        result.timing["wall_s"] = time.monotonic() - t0
        decide(result)
        if self.out_dir is not None:
            from aerosplat.sim.report import write_report

            write_report(result, self.out_dir)
            result.out_dir = self.out_dir
        return result

    def _fly(self, result: RunResult, seed: int) -> None:
        assert self.provider is not None
        pkg = self.pkg
        home = Home(lat=pkg.home.lat, lon=pkg.home.lon, alt=pkg.home.alt, heading=0.0,
                    source=f"mission {pkg.plan.id} home row")
        boot = self.boot_params()
        files: tuple[Path, ...] = ()
        if boot:
            folder = Path(tempfile.mkdtemp(prefix="aerosplat-boot-"))
            path = folder / "boot.parm"
            path.write_text("".join(f"{n} {v:g}\n" for n, v in sorted(boot.items())),
                            encoding="utf-8")
            files = (path,)
        try:
            self._fly_booted(result, seed, home, files, boot)
        finally:
            # Kept until the flight is over: SITL may read it after "up" (its values are in
            # the preflight record and were read back from the vehicle).
            for f in files:
                shutil.rmtree(f.parent, ignore_errors=True)

    def _fly_booted(self, result: RunResult, seed: int, home: Home, files: tuple[Path, ...],
                    boot: dict[str, float]) -> None:
        assert self.provider is not None
        pkg, sc = self.pkg, self.sc
        t_start = time.monotonic()
        endpoint = self.provider.prepare(home, self.speedup, boot_defaults=files)
        result.timing["sitl_start_s"] = time.monotonic() - t_start
        self.log(f"[{sc.scenario}] SITL at {home.arg()} ({endpoint}), speedup {self.speedup:g}"
                 + (f", boot {', '.join(f'{n} {v:g}' for n, v in sorted(boot.items()))}" if boot else ""))
        with Vehicle.connect(endpoint, timeout=60, source_system=RUNNER_SYSID) as v:
            v.heartbeat_period = min(1.0, 0.25 / self.speedup)
            rec = Recording()
            if self.navsim is not None:
                # At once: EKF3 keeps the first origin it gets, and must get the Site's.
                kw: dict[str, Any] = dict(
                    origin=pkg.origin, sim=self.navsim, spec=self._extnav_spec(),
                    flow=self.navp.flow, grid=self.grid, speedup=self.speedup,
                    event=rec.event, log=lambda s: self.log(f"[{sc.scenario}] {s}"))
                self.companion = self._vps_companion(v, kw) if self.vps is not None \
                    else Companion(v, **kw)
                self.companion.attach()
            try:
                self._fly_vehicle(v, result, seed, rec)
            finally:
                if self.companion is not None:
                    self.companion.detach()
                    result.preflight.setdefault("nav", {})["companion"] = self.companion.summary()

    def _vps_companion(self, v: Vehicle, kw: dict[str, Any]) -> Companion:
        """The in-the-loop companion: the world renderer (loaded once per runner), the camera of
        the vehicle profile, the atlas the companion holds (the package's, unless the scenario
        gives another one)."""
        from aerosplat.sim.camera import load_camera
        from aerosplat.sim.vpsloop import VpsCompanion, make_world
        from aerosplat.vps.route import CameraModel

        assert self.vps is not None and self.pkg.plan.vps is not None
        mount = self.profile.airframe.vps_mount()
        assert mount is not None and mount.vps is not None
        camera = load_camera(mount.camera)
        if self.world is None:
            t0 = time.monotonic()
            self.world = make_world(self._resolve(self.vps.run_dir))
            self.log(f"[{self.sc.scenario}] VPS world {self.vps.run_dir} loaded in "
                     f"{time.monotonic() - t0:.1f} s")
        atlas = self._resolve(self.vps.atlas) if self.vps.atlas else self.pkg.dir / self.pkg.plan.vps.atlas.file
        readout = camera.readout_ms(self.vps.fps) / 1000.0 if self.vps.rolling_shutter else 0.0
        land_seq = next((it.seq for it in self.pkg.items if it.command == 21), None)
        res = mount.vps.resolution
        home = self.pkg.plan.home.enu
        return VpsCompanion(v, binding=self.pkg.plan.vps, atlas_path=atlas, world=self.world,
                            cam=CameraModel.of(camera, mount, (int(res[0]), int(res[1]))),
                            readout_s=readout, rate_hz=self.vps.rate_hz or mount.vps.rate_hz,
                            home_enu=(float(home[0]), float(home[1]), float(home[2])),
                            land_seq=land_seq, **kw)

    def _extnav_spec(self) -> ExtNavSpec:
        return self.navp.extnav if self.navp.extnav is not None else ExtNavSpec()

    def _streams(self, v: Vehicle) -> None:
        v.stream_defaults(hz=10.0)
        for msg_id, hz in ((MSG_MISSION_CURRENT, 4.0), (MSG_FENCE_STATUS, 2.0),
                           (MSG_EKF_STATUS_REPORT, 2.0), (MSG_HOME_POSITION, 1.0),
                           (MSG_SYSTEM_TIME, 4.0)):
            v.set_message_interval(msg_id, hz)
        if self.nav_run:
            for msg_id, hz in ((MSG_GPS_GLOBAL_ORIGIN, 1.0), (MSG_GPS_RAW_INT, 5.0)):
                v.set_message_interval(msg_id, hz)
        if self.vps is not None:
            # the localizer's prior (attitude) and the anchor's lift-off (landed state)
            v.set_message_interval(MSG_ATTITUDE, 10.0)
            v.set_message_interval(MSG_EXTENDED_SYS_STATE, 10.0)
            # the downward rangefinder: the companion's wall guard ends near the ground
            v.set_message_interval(MSG_DISTANCE_SENSOR, 10.0)

    def _fly_vehicle(self, v: Vehicle, result: RunResult, seed: int, rec: Recording) -> None:
        pkg, sc, fw, cfg = self.pkg, self.sc, self.fw, self.cfg
        self._streams(v)
        firmware = v.firmware()
        result.firmware = {"version": ".".join(map(str, firmware.version)), "git": firmware.git_hash,
                           "pin": cfg.ardupilot.commit[:8], "tag": cfg.ardupilot.tag}
        if firmware.git_hash != cfg.ardupilot.commit[:8]:
            raise Refusal("firmware_mismatch", f"the vehicle runs {firmware.git_hash}, the pin is "
                                               f"{cfg.ardupilot.commit[:8]}")
        t = time.monotonic()
        live = v.fetch_params()
        result.timing["param_fetch_s"] = time.monotonic() - t
        drift = sorted(set(live) ^ set(fw.params))
        if self.nav_run and (live.get("GPS1_TYPE") == 0 or live.get("SIM_GPS1_ENABLE") == 0):
            # GPS1_TYPE 0 hides the receiver's other GPS1_* parameters, SIM_GPS1_ENABLE 0 the
            # simulated receiver's SIM_GPS1_* (the extnav snapshot was booted with both on):
            # absent for that reason, not a different build.
            hidden = [n for n in drift if n.startswith(("GPS1_", "SIM_GPS1_")) and n not in live]
            drift = [n for n in drift if n not in hidden]
            result.preflight.setdefault("nav", {})["hidden_with_gps_off"] = len(hidden)
        if drift:
            result.warnings.append(f"live parameter table differs from the {fw.tag} snapshot in "
                                   f"{len(drift)} names: {', '.join(drift[:8])}")
        unknown = [n for n in pkg.parm if n not in live]
        if unknown:
            raise Refusal("unknown_parameter", f"the vehicle has no {', '.join(unknown)}")
        truth = v.wait_truth(20)
        off = horizontal_distance_m(truth.lat, truth.lon, pkg.home.lat, pkg.home.lon)
        if off > HOME_TOL_M or v.armed:
            raise Refusal("home_mismatch", f"the vehicle stands {off:.1f} m from the mission's home"
                                           f"{' and is armed' if v.armed else ''}")
        # Parameters: vehicle, runner (GCS role), environment, then the mission package.
        layers = [("vehicle", self.profile.params), ("runner", RUNNER_PARAMS),
                  ("environment", environment_params(sc.environment, fw)), ("package", pkg.parm)]
        for source, values in layers:
            for name, value in values.items():
                before = live.get(name)
                got = v.set_param(name, float(value))
                result.applied.append({"name": name, "before": before, "value": got, "source": source})
        values = dict(live)
        values.update({a["name"]: a["value"] for a in result.applied})
        if self.nav_run:
            self._check_nav_vehicle(live, values, result)
        result.warnings += [w for w in failsafe_trigger_problems(pkg.parm, values)
                            if w not in result.warnings]
        injections = self.injections(values, seed)
        resp = self.expected_response(values)
        result.response = resp
        # Fence, rally, mission: uploaded, read back, compared.
        try:
            mavmission.upload_verified(v, mavmission.fence_items(pkg.fence_latlon),
                                       mavmission.MISSION_TYPE_FENCE)
            mavmission.upload_verified(v, mavmission.rally_items(pkg.rally),
                                       mavmission.MISSION_TYPE_RALLY)
            held = mavmission.upload_verified(v, [mavmission.Item.from_wire(i) for i in pkg.items],
                                              mavmission.MISSION_TYPE_MISSION)
        except mavmission.MissionProtocolError as e:
            raise Refusal("upload_mismatch", str(e)) from None
        result.preflight["upload_verified"] = True
        result.preflight["frames"] = sorted({i.frame for i in held[1:]})
        t = time.monotonic()
        if self.companion is None:
            v.wait_prearm_ok(timeout=180)
        else:
            self._wait_nav_ready(v, result)
        result.timing["prearm_s"] = time.monotonic() - t
        self._mission(v, result, injections, resp, values, rec)

    # -- GNSS-free runs (SIM-2n) ------------------------------------------
    def _check_nav_vehicle(self, live: dict[str, float], values: dict[str, float],
                           result: RunResult) -> None:
        """Every boot-time parameter is what SITL was booted with (a PARAM_SET would not act
        before a reboot), and the vehicle as configured takes nothing from GNSS."""
        boot = self.boot_params()
        wrong = [f"{n} = {live.get(n)} (needs {v:g})" for n, v in boot.items()
                 if n not in live or not math.isclose(float(live[n]), v, abs_tol=1e-6)]
        if wrong:
            raise Refusal("nav_config", "the vehicle was not booted as the run needs (a reused "
                                        "endpoint?): " + ", ".join(wrong))
        bad = nav_problems(self.navp, values)
        if bad:
            raise Refusal("nav_config", "; ".join(bad))
        result.preflight.setdefault("nav", {})["boot_verified"] = sorted(boot)

    def _wait_nav_ready(self, v: Vehicle, result: RunResult) -> None:
        """The EKF origin must read back as the Site origin before anything else (unless the
        scenario withholds it on purpose); then pre-arm. A vehicle that will not get ready is
        tried anyway: whether the AUTOPILOT refuses to arm is what negative scenarios test."""
        comp = self.companion
        assert comp is not None and self.navsim is not None
        nav = result.preflight.setdefault("nav", {})
        if self.navsim.origin:
            try:
                v.wait("EKF origin to read back", lambda: comp.origin_ok
                       or comp.origin_mismatch() is not None, ORIGIN_WAIT_S)
            except VehicleTimeout:
                raise Refusal("nav_config", f"the EKF did not report an origin within "
                                            f"{ORIGIN_WAIT_S:g} s of the companion sending it "
                                            f"({comp.nav.origin_sent} times)") from None
            bad = comp.origin_mismatch()
            if bad:
                raise Refusal("nav_config", bad + " (GNSS set it first?)")
            nav["origin_verified"] = comp.nav.origin_readback
        try:
            v.wait_prearm_ok(timeout=NAV_PREARM_S)
            nav["prearm"] = "ok"
        except VehicleTimeout:
            nav["prearm"] = f"not ok after {NAV_PREARM_S:g} s: {v.statustext[-4:]}"

    def _arm(self, v: Vehicle, result: RunResult) -> None:
        if self.companion is None:
            v.set_mode("GUIDED")
            v.arm(timeout=60)
            return
        try:
            v.set_mode("GUIDED")
        except VehicleError as e:
            # GUIDED needs a position estimate; arming is then tried in the mode it is in
            # (ARMING_NEED_LOC 1: a position estimate is required in every mode).
            result.preflight.setdefault("nav", {})["guided_refused"] = str(e)
        refusal = getattr(self.companion, "refusal", None)
        try:
            v.arm(timeout=NAV_ARM_S)
        except VehicleTimeout:
            texts = [s for s in v.statustext[-12:] if "Arm" in s or "PreArm" in s]
            said = "; ".join(dict.fromkeys(texts or v.statustext[-4:]))
            if refusal:
                # the companion refused first (another atlas than the package's): that is the
                # cause; the autopilot's own refusal is the proof that it cannot fly without it
                raise Refusal("atlas_mismatch", f"{refusal}; the autopilot did not arm: {said}") from None
            raise Refusal("arming_refused", "the autopilot refused to arm: " + said) from None
        if refusal:
            v.disarm(force=True)
            raise RunError(f"the vehicle ARMED although the companion refused ({refusal}): "
                           f"arming without ExternalNav")

    def _mission(self, v: Vehicle, result: RunResult, injections: list[Injection],
                 resp: Optional[Response], values: dict[str, float], rec: Recording) -> None:
        pkg, sc = self.pkg, self.sc
        sp = self.speedup
        raw = _Raw()
        roles = {k: it.role for k, it in enumerate(pkg.plan.items, start=1)}
        state: dict[str, Any] = {"mode": v.mode, "armed": v.armed, "seq": 0, "fence": None,
                                 "landed": None, "ekf": None, "home": None, "volt": None}
        nav = {it.seq for it in pkg.items[1:] if it.command in POSITIONAL}
        land_seq = next((it.seq for it in pkg.items if it.command == 21), None)
        reached: set[int] = set()

        def mark_reached(t: float, seq: int, via: str) -> None:
            # Copter keeps ONE pending MISSION_ITEM_REACHED: a waypoint followed by DO items is
            # overwritten by them before it is sent. So "reached" also comes from the
            # "Reached command #N" STATUSTEXT and from AUTO moving past a waypoint.
            if seq in nav and seq not in reached:
                reached.add(seq)
                rec.event(t, "reached", roles.get(seq, ""), seq=seq, via=via)

        def now() -> float:
            return v.sim_now(sp)

        def on(msg: Any) -> None:
            kind = msg.get_type()
            t = now()
            if kind == "SIM_STATE":
                tr = truth_from_sim_state(msg, t)
                est = None
                gp = v.fresh("GLOBAL_POSITION_INT", max_age=0.3 / sp + 0.05)
                if gp is not None and gp.lat != 0:
                    est = (gp.lat * 1e-7, gp.lon * 1e-7, gp.alt / 1000.0)
                raw.t.append(t)
                raw.lla.append((tr.lat, tr.lon, tr.alt_amsl))
                raw.vel.append((tr.vn, tr.ve, tr.vd))
                raw.yaw.append(math.degrees(float(msg.yaw)) % 360.0)
                raw.mode.append(state["mode"] or "?")
                raw.armed.append(bool(state["armed"]))
                raw.seq.append(int(state["seq"]))
                raw.est.append(est)
            elif kind == "HEARTBEAT":
                mode = mode_name(msg.custom_mode)
                armed = bool(msg.base_mode & 128)
                if mode != state["mode"]:
                    rec.event(t, "mode", mode, previous=state["mode"])
                    state["mode"] = mode
                if armed != state["armed"]:
                    if not armed and state["mode"] == "AUTO" and state["seq"] == land_seq                             and land_seq is not None:
                        mark_reached(t, land_seq, "disarmed on the LAND item")
                    rec.event(t, "armed" if armed else "disarmed")
                    state["armed"] = armed
            elif kind == "MISSION_CURRENT":
                if int(msg.seq) != state["seq"]:
                    old, state["seq"] = state["seq"], int(msg.seq)
                    if state["mode"] == "AUTO" and state["armed"] and 0 < old < int(msg.seq):
                        mark_reached(t, old, "mission advanced")
                    rec.event(t, "current", roles.get(int(msg.seq), ""), seq=int(msg.seq))
            elif kind == "MISSION_ITEM_REACHED":
                mark_reached(t, int(msg.seq), "MISSION_ITEM_REACHED")
            elif kind == "STATUSTEXT":
                text = str(msg.text)
                rec.event(t, "statustext", text, severity=int(msg.severity))
                m = REACHED_TEXT.match(text)
                if m:
                    mark_reached(t, int(m.group(1)), "STATUSTEXT")
            elif kind == "FENCE_STATUS":
                key = (int(msg.breach_status), int(msg.breach_count))
                if key != state["fence"]:
                    state["fence"] = key
                    rec.event(t, "fence", "breach" if key[0] else "inside", breach_status=key[0],
                              breach_count=key[1], breach_type=int(msg.breach_type))
            elif kind == "EXTENDED_SYS_STATE":
                if int(msg.landed_state) != state["landed"]:
                    state["landed"] = int(msg.landed_state)
                    rec.event(t, "landed_state", ["undefined", "on_ground", "in_air", "takeoff",
                                                  "landing"][min(int(msg.landed_state), 4)])
            elif kind == "EKF_STATUS_REPORT":
                if int(msg.flags) != state["ekf"]:
                    state["ekf"] = int(msg.flags)
                    rec.event(t, "ekf", f"flags 0x{int(msg.flags):04x}", flags=int(msg.flags))
            elif kind == "HOME_POSITION":
                h = (int(msg.latitude), int(msg.longitude), int(msg.altitude))
                if h != state["home"]:
                    state["home"] = h
                    rec.event(t, "home", "", lat=h[0] * 1e-7, lon=h[1] * 1e-7, alt=h[2] / 1000.0)
            elif kind == "SYS_STATUS":
                volt = msg.voltage_battery / 1000.0 if msg.voltage_battery not in (0, 65535) else None
                if volt is not None and (state["volt"] is None or abs(volt - state["volt"]) >= 0.2):
                    state["volt"] = volt
                    rec.event(t, "battery", f"{volt:.2f} V", volts=volt)

        v.listeners.append(on)
        try:
            v.pump(0.2)
            self._arm(v, result)
            v.start_mission()
            t_start = now()
            rec.event(t_start, "mission_start")
            result.timing["mission_start_sim_s"] = t_start
            self.log(f"[{sc.scenario}] mission started at t={t_start:.1f} s")
            self._loop(v, rec, state, injections, t_start, result)
        finally:
            v.listeners.remove(on)
            v.send_heartbeats = True
            for inj in injections:          # leave the simulator as found (reused endpoints)
                if any(e["fault"] == inj.fault.fault.value and e["on"] is not None and e["off"] is None
                       for e in result.injected):
                    try:
                        self._clear(v, inj, rec, now(), result, final=True)
                    except VehicleError as err:
                        result.warnings.append(f"could not clear {inj.fault.fault.value}: {err}")
        frame = EnuFrame(pkg.origin)
        if raw.t:
            lla = np.asarray(raw.lla, float)
            enu = enu_from_geodetic(frame, lla[:, 0], lla[:, 1], lla[:, 2])
            est_rows = [k for k, e in enumerate(raw.est) if e is not None]
            est_enu: list[Optional[tuple[float, float, float]]] = [None] * len(raw.t)
            if est_rows:
                ea = np.asarray([raw.est[k] for k in est_rows], float)
                ee = enu_from_geodetic(frame, ea[:, 0], ea[:, 1], ea[:, 2])
                for k, row in zip(est_rows, ee):
                    est_enu[k] = (float(row[0]), float(row[1]), float(row[2]))
            for k in range(len(raw.t)):
                rec.add(t=raw.t[k], enu=enu[k], vel_ned=raw.vel[k], yaw_deg=raw.yaw[k],
                        mode=raw.mode[k], armed=raw.armed[k], seq=raw.seq[k], est_enu=est_enu[k])
        rec.meta = {"scenario": sc.scenario, "speedup": self.speedup, "origin": pkg.origin.model_dump(),
                    "home_enu": list(pkg.plan.home.enu), "mission": pkg.plan.id}
        result.recording = rec
        trigger_t = next((e["on"] for e in result.injected if resp is not None
                          and e["reason"] == resp.trigger and e["on"] is not None), None)
        nav_ctx = None
        if self.companion is not None:
            comp = self.companion
            assert self.navsim is not None          # a companion is only attached with one
            if trigger_t is None and resp is not None and resp.trigger == "vps_loss":
                # No injected fault: a no-fix zone took VPS away (capture time of the first miss).
                trigger_t = next((t for t, fix in comp.nav.fix if not fix and t >= t_start), None)
            nav_ctx = NavContext(gnss=self.navp.gnss, flow=self.navp.flow, fix=list(comp.nav.fix),
                                 gnss_fixes=comp.nav.gnss_fixes, spoof_t=comp.nav.spoof_t,
                                 origin_ok=comp.origin_ok, origin_required=self.navsim.origin,
                                 # From the first PARAM_SET: the glitch may act before the echo.
                                 spoof_injected_t=next((e.get("sent", e["on"]) for e in result.injected
                                                        if e["fault"] == "gps_spoof"), None),
                                 summary=comp.summary())
            records = getattr(comp, "records", None)
            if records is not None:
                mount = self.profile.airframe.vps_mount()
                nav_ctx.vps_records = list(records)
                nav_ctx.vps_summary = comp.vps_summary(sc.expect.tolerances.false_fix_m)  # type: ignore[attr-defined]
                nav_ctx.vps_min_rate_hz = mount.vps.min_rate_hz if mount is not None and mount.vps else None
        ctx = Context(plan=pkg.plan, items=pkg.items, planned_enu=pkg.planned_enu,
                      fence_enu=pkg.fence_enu,
                      params=values, grid=self.grid, expect=sc.expect, response=resp, trigger_t=trigger_t,
                      mission_start_t=t_start, nav=nav_ctx)
        result.verdicts = judge(rec, ctx, result.preflight)
        result.clearance = clearance_series(rec, ctx)
        result.outcome = actual_outcome(rec, ctx)
        if result.outcome == Outcome.abort.value and resp is not None:
            result.reason = resp.trigger

    def _loop(self, v: Vehicle, rec: Recording, state: dict, injections: list[Injection],
              t_start: float, result: RunResult) -> None:
        sc, sp = self.sc, self.speedup
        anchors: dict[str, float] = {"start": t_start}
        seq_role = {k: it.role for k, it in enumerate(self.pkg.plan.items, start=1)}
        pending: list[tuple[Injection, Optional[float]]] = [(inj, None) for inj in injections]
        for inj in injections:
            result.injected.append({"fault": inj.fault.fault.value, "reason": inj.fault.reason,
                                    "after": inj.fault.after, "at_s": inj.fault.at_s,
                                    "duration_s": inj.fault.duration_s, "note": inj.note,
                                    "on": None, "off": None,
                                    "params": [[c.name, c.value] for c in inj.inject]})
        wall_deadline = time.monotonic() + sc.timeout_s / sp + 120.0
        end_at: Optional[float] = None
        was_armed = v.armed
        while True:
            v.pump(0.02)
            if self.companion is not None:
                self.companion.tick()
            t = v.sim_now(sp)
            for e in rec.events_of("current", "reached"):
                role = seq_role.get(int(e.data.get("seq", -1)))
                if not role:
                    continue
                key = role if e.kind == "current" else f"{role}:reached"
                if key not in anchors:
                    anchors[key] = e.t
            for k, (inj, t_on) in enumerate(pending):
                entry = result.injected[k]
                anchor = anchors.get(inj.fault.after)
                if t_on is None and anchor is not None and t >= anchor + inj.fault.at_s \
                        and entry["on"] is None:
                    self._inject(v, inj, rec, t, result, k)
                    pending[k] = (inj, t)
                elif t_on is not None and entry["off"] is None and inj.fault.duration_s is not None \
                        and t >= t_on + inj.fault.duration_s:
                    self._clear(v, inj, rec, t, result)
            was_armed = was_armed or v.armed
            if end_at is None and was_armed and not v.armed:
                end_at = t + TAIL_S
            if end_at is not None and t >= end_at:
                rec.event(t, "end", "landed")
                break
            if t > t_start + sc.timeout_s:
                rec.event(t, "end", "timeout")
                break
            if time.monotonic() > wall_deadline:
                rec.event(t, "end", "timeout", wall=True)
                break
        result.timing["mission_sim_s"] = v.sim_now(sp) - t_start

    def _inject(self, v: Vehicle, inj: Injection, rec: Recording, t: float, result: RunResult,
                k: int) -> None:
        # The first PARAM_SET leaves now; "on" is when the vehicle confirmed the last one (a
        # lost echo costs a resend: up to seconds of simulation between the two).
        result.injected[k]["sent"] = v.sim_now(self.speedup)
        for c in inj.inject:
            v.set_param(c.name, c.value)
        if inj.heartbeats is False:
            v.send_heartbeats = False
        if inj.vps is False and self.companion is not None:
            self.companion.vps_on = False
        if inj.vps_effect is not None:
            set_effect = getattr(self.companion, "set_effect", None)
            if set_effect is None:
                raise RunError(f"{inj.fault.fault.value} needs the in-the-loop companion (nav.vps)")
            set_effect(inj.vps_effect)
        t_on = v.sim_now(self.speedup)
        result.injected[k]["on"] = t_on
        rec.event(t_on, "fault_on", inj.fault.fault.value, note=inj.note)
        self.log(f"[{self.sc.scenario}] t={t_on:.1f} s {inj.fault.fault.value} on: {inj.note}")

    def _clear(self, v: Vehicle, inj: Injection, rec: Recording, t: float, result: RunResult,
               final: bool = False) -> None:
        if inj.heartbeats is False:
            v.send_heartbeats = True
        if inj.vps is False and self.companion is not None:
            self.companion.vps_on = True
        if inj.vps_effect is not None and hasattr(self.companion, "clear_effect"):
            self.companion.clear_effect(inj.vps_effect)        # type: ignore[union-attr]
        for c in inj.clear:
            v.set_param(c.name, c.value)
        entry = next(e for e in result.injected if e["fault"] == inj.fault.fault.value
                     and e["on"] is not None and e["off"] is None)
        t_off = v.sim_now(self.speedup)
        entry["off"] = t_off
        if not final:
            rec.event(t_off, "fault_off", inj.fault.fault.value)


# ── entry points ─────────────────────────────────────────────────────────
def resolve_inputs(sc: Scenario, base: Path, mission: Optional[Path] = None,
                   occupancy: Optional[Path] = None, projects_root: Optional[Path] = None
                   ) -> tuple[Path, Path]:
    """The export folder and the occupancy folder of a scenario."""
    mdir = mission or resolve(base, sc.mission)
    if mdir is not None and not mdir.exists() and sc.mission and sc.mission.startswith("msn_") and sc.site:
        from aerosplat.store.paths import StorePaths, default_projects_root

        project, site = sc.site.split("/")
        mdir = StorePaths(projects_root or default_projects_root()).site(project, site).missions / sc.mission
    if mdir is None:
        raise ScenarioError("the scenario names no mission (mission: <export folder> or msn_ id)")
    odir = occupancy or resolve(base, sc.occupancy)
    if odir is None:
        plan = MissionPlan(**json.loads((mdir / "mission.json").read_text(encoding="utf-8")))
        if not sc.site:
            raise ScenarioError("no occupancy folder and no site to look the plan's run up in")
        from aerosplat.store.paths import StorePaths, default_projects_root

        project, site = sc.site.split("/")
        run = StorePaths(projects_root or default_projects_root()).site(project, site).run(
            plan.occupancy.epoch_id, plan.occupancy.run_id)
        odir = run.root / "occupancy"
    return mdir, odir


def run_scenario(sc: Scenario, mission_dir: Path, grid: OccupancyGrid, *,
                 provider: Optional[SitlProvider], out_dir: Optional[Path] = None,
                 origin: Optional[GeoPoint] = None, seeds: Optional[int] = None,
                 speedup: Optional[float] = None, log=print, world: Any = None,
                 base: Optional[Path] = None) -> list[RunResult]:
    """Every seed of a scenario; a package that cannot even be read is one refused run.
    `world`: the VPS world renderer to share between runs (else each run loads its own)."""
    if origin is None and sc.origin is not None:
        origin = GeoPoint(**sc.origin)
    out = out_dir if out_dir is not None else Path(mission_dir) / "sim" / sc.scenario
    try:
        pkg = MissionPackage.load(Path(mission_dir), origin)
    except Refusal as r:
        res = RunResult(sc, 0, outcome="refused", reason=r.reason, detail=r.detail, grid=grid)
        decide(res)
        from aerosplat.sim.report import write_report

        write_report(res, out)
        res.out_dir = out
        return [res]
    results = []
    n = seeds or sc.seeds
    owned = []
    try:
        for seed in range(n):
            runner = Runner(sc, pkg, grid, provider=provider, speedup=speedup,
                            out_dir=out if n == 1 else out / f"seed-{seed}", log=log, world=world,
                            base=base)
            results.append(runner.run(seed))
            if world is None and runner.world is not None:
                world = runner.world            # loaded by the first run: the others share it
                owned.append(world)
    finally:
        for w in owned:
            w.close()
    return results


def main(argv: Optional[list[str]] = None) -> int:
    from aerosplat.logging_utils import fix_console_encoding

    fix_console_encoding()
    ap = argparse.ArgumentParser(prog="python -m aerosplat.sim.runner",
                                 description="fly an L2 scenario in the pinned ArduCopter SITL")
    ap.add_argument("scenario", type=Path)
    ap.add_argument("--mission", type=Path, help="export folder (overrides the scenario's)")
    ap.add_argument("--occupancy", type=Path, help="occupancy folder (overrides the scenario's)")
    ap.add_argument("--out", type=Path, help="report folder (default <mission>/sim/<scenario>)")
    ap.add_argument("--instance", "-I", type=int, default=1, help="SITL instance to start (1)")
    ap.add_argument("--endpoint", help="fly an already running SITL instead (pymavlink string)")
    ap.add_argument("--speedup", type=float, default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--preflight", action="store_true", help="offline checks only, no SITL")
    ap.add_argument("--projects-root", type=Path, default=None)
    args = ap.parse_args(argv)

    sc = load_scenario(args.scenario)
    mdir, odir = resolve_inputs(sc, args.scenario.parent, args.mission, args.occupancy,
                                args.projects_root)
    grid, _ = read_occupancy(odir)
    provider = None if args.preflight else SitlProvider(instance=args.instance, endpoint=args.endpoint)
    try:
        results = run_scenario(sc, mdir, grid, provider=provider, out_dir=args.out,
                               seeds=args.seeds, speedup=args.speedup, base=args.scenario.parent)
    finally:
        if provider is not None:
            provider.close()
    ok = True
    for r in results:
        ok &= r.passed
        print(f"{'PASS' if r.passed else 'FAIL'} {sc.scenario} seed {r.seed}: {r.outcome} — {r.summary}")
        for v in r.verdicts:
            print(f"    {v.invariant:15} {v.status:5} {v.detail}")
        for w in r.warnings:
            print(f"    warning: {w}")
        if r.out_dir is not None:
            print(f"    report {r.out_dir / 'report.html'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
