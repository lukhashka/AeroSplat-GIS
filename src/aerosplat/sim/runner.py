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
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict

from aerosplat.geo.crs import EnuFrame, GeoPoint
from aerosplat.mission.export import (POSITIONAL, WireItem, enu_from_geodetic, read_parm, read_plan,
                                      read_waypoints, wire_to_enu)
from aerosplat.mission.models import AltitudeFrameError, MissionPlan, Verdict as PlanVerdict
from aerosplat.paths import PROJECT_DIR
from aerosplat.sim import mavmission
from aerosplat.sim.faults import (RUNNER_PARAMS, RUNNER_SYSID, FaultUnavailable, Injection,
                                  Response, failsafe_trigger_problems, injection, response,
                                  environment_params)
from aerosplat.sim.oracles import (FAIL, PASS, ClearanceSeries, Context, Verdict, actual_outcome,
                                   clearance_series, expected_status, judge)
from aerosplat.sim.params import FirmwareParams, ParameterError, load_snapshot
from aerosplat.sim.record import Recording
from aerosplat.sim.scenario import Fault, Outcome, Scenario, load_scenario, resolve
from aerosplat.sim.sitl import Home, Sitl, SitlConfig, SitlError, load_config
from aerosplat.sim.vehicle import (Vehicle, VehicleError, horizontal_distance_m,
                                   mode_name, truth_from_sim_state)
from aerosplat.surface.occupancy import OccupancyGrid, read_occupancy

VEHICLES_DIR = PROJECT_DIR / "configs" / "sim" / "vehicles"
PLAN_TOL_M = 0.01            # .waypoints / .plan / mission.json agree within this
FENCE_TOL_M = 0.05           # the .plan fence vs the plan's polygon (9-decimal lat/lon)
HOME_TOL_M = 1.0             # the vehicle stands this close to the mission's home
TAIL_S = 2.0                 # simulation seconds recorded after the disarm
MSG_SYSTEM_TIME, MSG_MISSION_CURRENT, MSG_FENCE_STATUS = 2, 42, 162
MSG_EKF_STATUS_REPORT, MSG_HOME_POSITION = 193, 242
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

    def prepare(self, home: Home, speedup: float) -> str:
        if self.endpoint:
            return self.endpoint
        if self.started:
            self.sitl.stop()
            self.started = False
        elif self.sitl.status() is not None:
            raise SitlError(f"SITL instance {self.sitl.instance} is running and was not started by "
                            f"this runner: stop it (python -m aerosplat.sim.sitl -I "
                            f"{self.sitl.instance} stop) or use another instance")
        self.sitl.start(home, speedup)
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
                 log=print):
        self.sc = scenario
        self.pkg = package
        self.grid = grid
        self.provider = provider
        self.cfg = provider.cfg if provider is not None else load_config()
        self.fw = fw or load_snapshot(self.cfg.ardupilot.tag, self.cfg.ardupilot.commit)
        self.speedup = float(speedup or scenario.speedup or self.cfg.vehicle.speedup)
        self.out_dir = out_dir
        self.log = log
        self.profile = load_vehicle(scenario.vehicle)

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
        except ParameterError as e:
            raise ScenarioError(str(e)) from None
        nav_roles = {it.role for it in plan.items if it.command in POSITIONAL}
        for f in sc.faults:
            if f.after != "start" and f.after.partition(":")[0] not in nav_roles:
                raise ScenarioError(f"fault {f.fault.value} after '{f.after}': the mission has no such "
                                    f"waypoint (roles: {', '.join(sorted(nav_roles))})")
        values = self.vehicle_values({n: i.default for n, i in fw.params.items()})
        injections = self.injections(values, seed)
        resp = self.expected_response(values)
        return {"firmware": fw.tag, "parm": len(pkg.parm), "faults": [i.note for i in injections],
                "response": None if resp is None else {"trigger": resp.trigger, "modes": list(resp.modes),
                                                        "by": list(resp.by), "delay_s": resp.delay_s},
                "warnings": failsafe_trigger_problems(pkg.parm, values)}

    def vehicle_values(self, base: dict[str, float]) -> dict[str, float]:
        """What the vehicle will hold once everything is applied, in that order."""
        out = dict(base)
        out.update(self.profile.params)
        out.update(RUNNER_PARAMS)
        out.update(environment_params(self.sc.environment, self.fw))
        out.update(self.pkg.parm)
        return out

    def injections(self, values: dict[str, float], seed: int) -> list[Injection]:
        out = []
        for f in self.jittered_faults(seed):
            try:
                out.append(injection(f, self.fw, values, self.sc.environment.wind))
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
            resp = response(e.reason, self.pkg.parm, values)
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
        pkg, sc = self.pkg, self.sc
        home = Home(lat=pkg.home.lat, lon=pkg.home.lon, alt=pkg.home.alt, heading=0.0,
                    source=f"mission {pkg.plan.id} home row")
        t_start = time.monotonic()
        endpoint = self.provider.prepare(home, self.speedup)
        result.timing["sitl_start_s"] = time.monotonic() - t_start
        self.log(f"[{sc.scenario}] SITL at {home.arg()} ({endpoint}), speedup {self.speedup:g}")
        with Vehicle.connect(endpoint, timeout=60, source_system=RUNNER_SYSID) as v:
            v.heartbeat_period = min(1.0, 0.25 / self.speedup)
            self._fly_vehicle(v, result, seed)

    def _streams(self, v: Vehicle) -> None:
        v.stream_defaults(hz=10.0)
        for msg_id, hz in ((MSG_MISSION_CURRENT, 4.0), (MSG_FENCE_STATUS, 2.0),
                           (MSG_EKF_STATUS_REPORT, 2.0), (MSG_HOME_POSITION, 1.0),
                           (MSG_SYSTEM_TIME, 4.0)):
            v.set_message_interval(msg_id, hz)

    def _fly_vehicle(self, v: Vehicle, result: RunResult, seed: int) -> None:
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
        v.wait_prearm_ok(timeout=180)
        result.timing["prearm_s"] = time.monotonic() - t
        self._mission(v, result, injections, resp, values)

    def _mission(self, v: Vehicle, result: RunResult, injections: list[Injection],
                 resp: Optional[Response], values: dict[str, float]) -> None:
        pkg, sc = self.pkg, self.sc
        sp = self.speedup
        rec = Recording()
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
            v.set_mode("GUIDED")
            v.arm(timeout=60)
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
        ctx = Context(plan=pkg.plan, items=pkg.items, planned_enu=pkg.planned_enu,
                      fence_enu=pkg.fence_enu,
                      params=values, grid=self.grid, expect=sc.expect, response=resp, trigger_t=trigger_t,
                      mission_start_t=t_start)
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
        for c in inj.inject:
            v.set_param(c.name, c.value)
        if inj.heartbeats is False:
            v.send_heartbeats = False
        t_on = v.sim_now(self.speedup)
        result.injected[k]["on"] = t_on
        rec.event(t_on, "fault_on", inj.fault.fault.value, note=inj.note)
        self.log(f"[{self.sc.scenario}] t={t_on:.1f} s {inj.fault.fault.value} on: {inj.note}")

    def _clear(self, v: Vehicle, inj: Injection, rec: Recording, t: float, result: RunResult,
               final: bool = False) -> None:
        if inj.heartbeats is False:
            v.send_heartbeats = True
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
                 speedup: Optional[float] = None, log=print) -> list[RunResult]:
    """Every seed of a scenario; a package that cannot even be read is one refused run."""
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
    for seed in range(n):
        runner = Runner(sc, pkg, grid, provider=provider, speedup=speedup,
                        out_dir=out if n == 1 else out / f"seed-{seed}", log=log)
        results.append(runner.run(seed))
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
                               seeds=args.seeds, speedup=args.speedup)
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
