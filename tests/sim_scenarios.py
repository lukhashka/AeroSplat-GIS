"""Synthetic missions for the L2 runner tests, built on tests/mission_scenes.py (step SIM-1).

The street scene: one box building north of home, a window on its south wall. The window
mission takes off from home (0, 0, 0) to 9 m, flies to E (0, 3, 9), approaches H (0, 12, 9)
facing north (heading 0), holds, and comes back the same way. Every number in an assertion
below follows from that geometry.

Variants are built the way the bug they stand for would build them:

* `through_wall`     the plan itself is wrong — H moved into the building, the approach leg with
                     it, files consistent — so only the flight oracle can catch it;
* `outside_fence`    the same kind of error, a waypoint east of the fence polygon;
* `legacy_parm`      a package with the Copter 4.5/4.6 names of the 8.1b export;
* `amsl_heights`     relative items carrying AMSL heights (the wrong altitude frame).
"""

from __future__ import annotations

import json
from pathlib import Path

import mission_scenes as ms
from aerosplat.mission.export import format_waypoints, read_waypoints, render
from aerosplat.mission.models import MissionPlan, PlanningProfile
from aerosplat.mission.planner import plan_mission
from aerosplat.sim.scenario import Scenario

# Short hold so a run takes about a minute of simulation; wires are not the subject here.
PROFILE = PlanningProfile(wire_height_m=5.0, hold_s=10.0)
WALL_H = (0.0, 17.0, 9.0)            # 2 m behind the south wall (y = 15), inside the building
EAST_OF_FENCE = (25.0, 3.0, 9.0)     # the fence polygon ends at x = 10.6


def window_plan(tmp_path: Path, grid=None) -> MissionPlan:
    grid = grid if grid is not None else ms.street_grid()
    ap = ms.window_ap()
    target, _, _ = ms.confirmed(tmp_path / "authored-window", ap)
    return plan_mission(target=target, ap=ap, grid=grid, occupancy=ms.occupancy_ref(),
                        origin=ms.ORIGIN, by="op:sim-test", profile=PROFILE)


def nav_plan(tmp_path: Path, gnss: str = "none", flow: bool = True, grid=None) -> MissionPlan:
    """The window mission planned for a GNSS-free vehicle (SIM-2n): ExternalNav (VPS, declared
    σ 0.5 m) + optical flow. The corridor takes σ_nav where it took σ_GNSS (1.5 m)."""
    from aerosplat.mission.models import ExtNavSpec, NavProfile

    grid = grid if grid is not None else ms.street_grid()
    ap = ms.window_ap()
    target, _, _ = ms.confirmed(tmp_path / "authored-window", ap)
    prof = PROFILE.model_copy(update={"nav": NavProfile(gnss=gnss, extnav=ExtNavSpec(), flow=flow)})
    return plan_mission(target=target, ap=ap, grid=grid, occupancy=ms.occupancy_ref(),
                        origin=ms.ORIGIN, by="op:sim-test", profile=prof)


def export(plan: MissionPlan, folder: Path, origin=ms.ORIGIN) -> Path:
    """mission.json + the three flyable formats, as MissionStore.export writes them."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "mission.json").write_text(plan.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
    for fmt in ("waypoints", "plan", "parm"):
        name, _, text = render(plan, origin, fmt, accept_risks=True)
        (folder / name).write_text(text, encoding="utf-8")
    return folder


def _move_item(plan: MissionPlan, role: str, to, *, add_leg: bool = False) -> MissionPlan:
    """The plan with the waypoint of `role` moved to `to`, as if the planner had computed it so:
    leg vertices at the old point move with it, or (`add_leg`, for a point several legs share)
    a leg from the waypoint before to the one after is added through the new point."""
    import numpy as np

    k = next(i for i, it in enumerate(plan.items) if it.role == role)
    old = plan.items[k].enu
    to = tuple(float(v) for v in to)
    items = list(plan.items)
    items[k] = items[k].model_copy(update={"enu": to})
    if add_leg:
        prev = next(it.enu for it in reversed(plan.items[:k]) if it.enu is not None)
        nxt = next(it.enu for it in plan.items[k + 1:] if it.enu is not None)
        transit = next(leg for leg in plan.legs if leg.name == "transit")
        legs = list(plan.legs) + [transit.model_copy(update={"name": "return", "points": [prev, to, nxt]})]
    else:
        legs = [leg.model_copy(update={"points": [to if np.allclose(p, old) else p for p in leg.points]})
                for leg in plan.legs]
    return plan.model_copy(update={"items": items, "legs": legs})


def through_wall(plan: MissionPlan) -> MissionPlan:
    p = _move_item(plan, "hover", WALL_H)
    return p.model_copy(update={"approach": p.approach.model_copy(update={"hover": WALL_H})})


def outside_fence(plan: MissionPlan) -> MissionPlan:
    return _move_item(plan, "over_exit", EAST_OF_FENCE, add_leg=True)


LEGACY_NAMES = {"RTL_ALT_M": ("RTL_ALT", 100.0), "RTL_ALT_FINAL_M": ("RTL_ALT_FINAL", 100.0),
                "RTL_CLIMB_MIN_M": ("RTL_CLIMB_MIN", 100.0), "RTL_SPEED_MS": ("RTL_SPEED", 100.0),
                "WP_SPD": ("WPNAV_SPEED", 100.0), "WP_SPD_UP": ("WPNAV_SPEED_UP", 100.0),
                "WP_SPD_DN": ("WPNAV_SPEED_DN", 100.0)}


def legacy_parm(folder: Path) -> None:
    """Rewrite mission.parm with the 4.5/4.6 names and centimetre units of the 8.1b export."""
    lines = []
    for line in (folder / "mission.parm").read_text(encoding="utf-8").splitlines():
        name = line.split()[0] if line and not line.startswith("#") else ""
        if name in LEGACY_NAMES:
            new, k = LEGACY_NAMES[name]
            value = float(line.split()[1]) * k
            line = f"{new} {value:g}  # as exported by 8.1b"
        lines.append(line)
    (folder / "mission.parm").write_text("\n".join(lines) + "\n", encoding="utf-8")


def amsl_heights(folder: Path) -> None:
    """Every relative item gets home's absolute height added: AMSL numbers in frame 3."""
    items = read_waypoints((folder / "mission.waypoints").read_text(encoding="utf-8"))
    home = items[0]
    out = [items[0]] + [it if it.frame != 3 else type(it)(**{**it.__dict__, "alt": it.alt + home.alt})
                        for it in items[1:]]
    (folder / "mission.waypoints").write_text(format_waypoints(out), encoding="utf-8")


def scenario(name: str, expect: dict, **kw) -> Scenario:
    data = {"scenario": name, "site": ms.SITE, "scene": "street", "target": "window",
            "vehicle": "sitl-quad", "ardupilot": {"tag": "Copter-4.7.2"}, "expect": expect,
            "timeout_s": 240}
    data.update(kw)
    return Scenario.model_validate(json.loads(json.dumps(data)))


def yaw_to(plan: MissionPlan, heading_deg: float) -> MissionPlan:
    """The CONDITION_YAW before H commands another heading than the direction of the approach,
    so that facing it at H proves the command acted on the approach leg."""
    items = [it.model_copy(update={"param1": float(heading_deg)}) if it.role == "face_target" else it
             for it in plan.items]
    return plan.model_copy(update={"items": items, "approach": plan.approach.model_copy(
        update={"heading_deg": float(heading_deg)})})


# ── synthetic flights for the oracle tests (no SITL) ────────────────────
def context(plan: MissionPlan, expect: dict, grid=None, **kw):
    """An oracle Context for `plan` as exported: wire items, planned ENU, the plan's fence and
    the parameters the package sets on top of the snapshot defaults."""
    from aerosplat.mission.export import parameters, to_wire, wire_to_enu
    from aerosplat.sim.oracles import Context
    from aerosplat.sim.params import load_snapshot
    from aerosplat.sim.scenario import Expect

    items = to_wire(plan, ms.ORIGIN)
    params = {n: i.default for n, i in load_snapshot().params.items()}
    params.update({n: float(v) for n, v, _ in parameters(plan)})
    return Context(plan=plan, items=items, planned_enu=wire_to_enu(items, ms.ORIGIN),
                   fence_enu=[tuple(p) for p in plan.fence.polygon_enu], params=params,
                   grid=grid if grid is not None else ms.street_grid(),
                   expect=Expect.model_validate(expect), mission_start_t=10.0, **kw)


def fly(plan: MissionPlan, *, dt: float = 0.1, t0: float = 10.0, takeoff_rate: float = 2.5,
        land_rate: float = 1.0, until: str | None = None):
    """A Recording of a vehicle that flies the plan's items exactly: straight lines at the
    speeds of the DO_CHANGE_SPEED items, holds where param1 says, yaw along the leg unless a
    CONDITION_YAW is active (until the next waypoint). `until`: stop after the item of that
    role (no landing, still armed)."""
    import numpy as np
    from aerosplat.sim.record import Recording

    rec = Recording()
    home = np.asarray(plan.home.enu, float)
    pos = home.copy()
    t = t0
    speed = 1.0
    fixed_yaw = None
    yaw = 0.0
    rec.event(t0 - 0.5, "mode", "GUIDED")
    rec.event(t0 - 0.2, "armed")
    rec.event(t0, "mission_start")
    rec.event(t0, "mode", "AUTO", previous="GUIDED")

    def sample(seq: int) -> None:
        rec.add(t=t, enu=pos, vel_ned=(0.0, 0.0, 0.0), yaw_deg=yaw, mode="AUTO", armed=True, seq=seq)

    def move(to, v: float, seq: int) -> None:
        nonlocal pos, t, yaw
        to = np.asarray(to, float)
        d = to - pos
        dist = float(np.linalg.norm(d))
        if dist > 1e-9 and np.linalg.norm(d[:2]) > 1e-6:
            yaw = fixed_yaw if fixed_yaw is not None else float(np.degrees(np.arctan2(d[0], d[1])) % 360)
        n = max(1, int(np.ceil(dist / (v * dt))))
        vel = d / max(dist, 1e-9) * v
        for k in range(1, n + 1):
            pos = pos + d / n
            t += dt
            rec.add(t=t, enu=pos, vel_ned=(vel[1], vel[0], -vel[2]), yaw_deg=yaw, mode="AUTO",
                    armed=True, seq=seq)

    sample(0)
    for seq, it in enumerate(plan.items, start=1):
        if it.command == 178:
            speed = it.param2
            continue
        if it.command == 115:
            fixed_yaw = it.param1
            continue
        rec.event(t, "current", it.role, seq=seq)
        if it.command == 22:
            move(it.enu, takeoff_rate, seq)
        elif it.command == 21:
            move((it.enu[0], it.enu[1], pos[2]), speed, seq)
            move(it.enu, land_rate, seq)
        else:
            move(it.enu, speed, seq)
            if fixed_yaw is not None:
                yaw = fixed_yaw
            for _ in range(int(round(it.param1 / dt))):
                t += dt
                sample(seq)
            fixed_yaw = None
        rec.event(t, "reached", it.role, seq=seq)
        if until == it.role:
            rec.event(t + 1.0, "end", "timeout")
            return rec
    t += 0.5
    rec.event(t, "disarmed")
    rec.add(t=t, enu=pos, vel_ned=(0.0, 0.0, 0.0), yaw_deg=yaw, mode="AUTO", armed=False, seq=0)
    rec.event(t + 1.0, "end", "landed")
    return rec


def with_mode(rec, t_switch: float, mode: str, *, path=None, end_at=None):
    """`rec` up to t_switch, then a failsafe in `mode` flying `path` (list of ENU points, 2 m/s)
    and landing, disarmed. Default path: straight down where it was."""
    import numpy as np
    from aerosplat.sim.record import Recording

    out = Recording()
    keep = [k for k, tt in enumerate(rec.t) if tt <= t_switch]
    for k in keep:
        out.add(t=rec.t[k], enu=rec.enu[k], vel_ned=rec.vel_ned[k], yaw_deg=rec.yaw_deg[k],
                mode=rec.mode[k], armed=rec.armed[k], seq=rec.seq[k])
    out.events = [e for e in rec.events if e.t <= t_switch and e.kind not in ("disarmed", "end")]
    out.event(t_switch, "mode", mode, previous="AUTO")
    pos = np.asarray(rec.enu[keep[-1]], float)
    t = t_switch
    seq = rec.seq[keep[-1]]
    pts = list(path or []) + [None]
    for to in pts:
        if to is None:
            sz = 0.0
            to = (pos[0], pos[1], sz)
        to = np.asarray(to, float)
        dist = float(np.linalg.norm(to - pos))
        n = max(1, int(np.ceil(dist / 0.2)))
        for k in range(1, n + 1):
            t += 0.1
            out.add(t=t, enu=pos + (to - pos) * k / n, vel_ned=(0.0, 0.0, 0.0), yaw_deg=0.0,
                    mode=mode, armed=True, seq=seq)
        pos = to
    t += 0.5
    out.event(t, "disarmed")
    out.add(t=t, enu=pos, vel_ned=(0.0, 0.0, 0.0), yaw_deg=0.0, mode=mode, armed=False, seq=seq)
    out.event(t + 1.0, "end", "landed")
    return out
