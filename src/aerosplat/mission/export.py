"""Exporting a mission for ArduPilot: QGC WPL 110, QGC `.plan`, and a `.parm` package.

    mission.waypoints   QGC WPL 110 — read by Mission Planner and MAVProxy (`wp load`)
    mission.plan        QGC plan: mission + geoFence (inclusion polygon) + rallyPoints
    mission.parm        fence / failsafe / speed parameters, "NAME VALUE" lines (MAVProxy
                        `param load`, SITL `--defaults`, Mission Planner)

Altitudes are **relative to home** (MAV_FRAME_GLOBAL_RELATIVE_ALT), always and explicitly:
the absolute height of Earth Studio sites is unverified and there is no geoid model here,
so an AMSL export is refused rather than guessed (SIMULATION §4.8). The home row of the WPL
file and `plannedHomePosition` must still hold *some* absolute altitude; they get the WGS84
ellipsoidal height, which ArduPilot replaces at arming, and every file says it is not AMSL.

Every export is read back before it is handed out: each positional item is converted to
Site ENU again and must land within 1 cm of the plan (MISSION_ITEM_INT keeps 1e-7°, ≈1.1 cm
of latitude, so the files carry 9 decimals). A frame mix-up — an absolute altitude in a
relative item, a relative item in an absolute file — fails that check or the reader's.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
import numpy as np

from aerosplat.geo.crs import EnuFrame, GeoPoint
from aerosplat.mission.models import (AltitudeFrameError, MissionPlan, Vec3, Verdict,
                                      require_relative)

WPL_HEADER = "QGC WPL 110"
MAV_FRAME_GLOBAL = 0                    # altitude above mean sea level
MAV_FRAME_MISSION = 2                   # no position: DO / CONDITION commands
MAV_FRAME_GLOBAL_RELATIVE_ALT = 3       # altitude above home
QGC_ALT_RELATIVE = 1                    # QGroundControl AltitudeMode: relative to launch
MAV_AUTOPILOT_ARDUPILOTMEGA = 3
MAV_TYPE_QUADROTOR = 2
# Commands whose lat / lon / alt are a position (common.xml).
POSITIONAL = {16, 17, 18, 19, 21, 22}
ROUND_TRIP_TOL_M = 0.01
FORMATS = {"waypoints": ("mission.waypoints", "text/plain"),
           "plan": ("mission.plan", "application/json"),
           "parm": ("mission.parm", "text/plain")}
HOME_ALT_NOTE = ("home altitude is the WGS84 ellipsoidal height, NOT AMSL: the site's absolute "
                 "height is unverified and no geoid model is applied; all mission altitudes are "
                 "relative to home")


class ExportRefused(ValueError):
    """The mission may not leave the system in a flyable format."""


def geodetic(frame: EnuFrame, enu) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`frame.to_geodetic` for (N, 3) points. A single point is padded to two: pyproj treats a
    size-1 array as a scalar, which NumPy deprecates."""
    a = np.asarray(enu, float).reshape(-1, 3)
    n = len(a)
    lat, lon, h = frame.to_geodetic(np.vstack([a, a]) if n == 1 else a)
    return lat[:n], lon[:n], h[:n]


def enu_from_geodetic(frame: EnuFrame, lat, lon, h) -> np.ndarray:
    """`frame.from_geodetic` for N points, padded like `geodetic`."""
    lat, lon, h = (np.atleast_1d(np.asarray(v, float)) for v in (lat, lon, h))
    n = len(lat)
    if n == 1:
        lat, lon, h = (np.r_[v, v] for v in (lat, lon, h))
    return frame.from_geodetic(lat, lon, h)[:n]


@dataclass(frozen=True)
class WireItem:
    """One row of a WPL file / one MISSION_ITEM: what actually reaches the autopilot."""

    seq: int
    current: int
    frame: int
    command: int
    params: tuple[float, float, float, float]
    lat: float
    lon: float
    alt: float
    autocontinue: int = 1


# ── plan -> wire ─────────────────────────────────────────────────────────
def to_wire(plan: MissionPlan, origin: GeoPoint) -> list[WireItem]:
    """Home row + the plan's items, with lat / lon from Site ENU and altitudes above home."""
    require_relative(plan.altitude_frame)
    frame = EnuFrame(origin)
    hlat, hlon, hh = (float(v[0]) for v in geodetic(frame, plan.home.enu))
    out = [WireItem(seq=0, current=1, frame=MAV_FRAME_GLOBAL, command=16, params=(0, 0, 0, 0),
                    lat=hlat, lon=hlon, alt=hh)]
    pos = [it.enu for it in plan.items if it.enu is not None]
    geo = geodetic(frame, pos) if pos else (np.zeros(0),) * 3
    g = 0
    for seq, it in enumerate(plan.items, start=1):
        params = (it.param1, it.param2, it.param3, it.param4)
        if it.enu is None:
            out.append(WireItem(seq=seq, current=0, frame=MAV_FRAME_MISSION, command=it.command,
                                params=params, lat=0.0, lon=0.0, alt=0.0))
            continue
        lat, lon, h = float(geo[0][g]), float(geo[1][g]), float(geo[2][g])
        g += 1
        out.append(WireItem(seq=seq, current=0, frame=MAV_FRAME_GLOBAL_RELATIVE_ALT,
                            command=it.command, params=params, lat=lat, lon=lon, alt=h - hh))
    check_wire(plan, origin, out)
    return out


def check_frames(items: list[WireItem]) -> None:
    """Refuse anything but the v0 layout: home in MAV_FRAME_GLOBAL, every positional item in
    MAV_FRAME_GLOBAL_RELATIVE_ALT, every other item in MAV_FRAME_MISSION."""
    if not items or items[0].seq != 0:
        raise AltitudeFrameError("no home row (seq 0): relative altitudes have no reference")
    home = items[0]
    if home.frame != MAV_FRAME_GLOBAL:
        raise AltitudeFrameError(f"home row is in frame {home.frame}; it must be absolute "
                                 f"(MAV_FRAME_GLOBAL = 0)")
    for it in items[1:]:
        if it.command in POSITIONAL and it.frame != MAV_FRAME_GLOBAL_RELATIVE_ALT:
            raise AltitudeFrameError(
                f"item {it.seq} (command {it.command}) is in frame {it.frame}; every position of "
                f"this mission must be relative to home (frame {MAV_FRAME_GLOBAL_RELATIVE_ALT})")
        if it.command not in POSITIONAL and it.frame not in (MAV_FRAME_MISSION,
                                                             MAV_FRAME_GLOBAL_RELATIVE_ALT):
            raise AltitudeFrameError(f"item {it.seq} (command {it.command}) is in frame {it.frame}")


def wire_to_enu(items: list[WireItem], origin: GeoPoint) -> dict[int, Vec3]:
    """Positional items back to Site ENU, using the home row as the altitude reference."""
    check_frames(items)
    home = items[0]
    frame = EnuFrame(origin)
    pos = [it for it in items[1:] if it.command in POSITIONAL]
    if not pos:
        return {}
    enu = enu_from_geodetic(frame, [it.lat for it in pos], [it.lon for it in pos],
                            [home.alt + it.alt for it in pos])
    return {it.seq: (float(p[0]), float(p[1]), float(p[2])) for it, p in zip(pos, enu)}


def check_wire(plan: MissionPlan, origin: GeoPoint, items: list[WireItem],
               tol_m: float = ROUND_TRIP_TOL_M) -> None:
    """Every positional item must come back to its planned Site ENU point within `tol_m`."""
    if len(items) != len(plan.items) + 1:
        raise AltitudeFrameError(f"{len(items) - 1} items on the wire, {len(plan.items)} planned")
    back = wire_to_enu(items, origin)
    hz = plan.home.enu[2]
    for seq, it in enumerate(plan.items, start=1):
        w = items[seq]
        if w.command != it.command:
            raise AltitudeFrameError(f"item {seq}: command {w.command}, planned {it.command}")
        if it.enu is None:
            continue
        err = float(np.linalg.norm(np.asarray(back[seq]) - np.asarray(it.enu, float)))
        if not err <= tol_m:
            raise AltitudeFrameError(
                f"item {seq} ({it.role}) lands {err:.3f} m from its planned point after the round "
                f"trip (altitude {w.alt:.3f} m vs {it.enu[2] - hz:.3f} m above home): wrong "
                f"altitude frame?")


# ── QGC WPL 110 ──────────────────────────────────────────────────────────
def _num(v: float, digits: int) -> str:
    s = f"{v:.{digits}f}"
    return "0" if s.strip("-0.") == "" else s


def format_waypoints(items: list[WireItem]) -> str:
    lines = [WPL_HEADER]
    for it in items:
        cells = [str(it.seq), str(it.current), str(it.frame), str(it.command),
                 *(_num(p, 6) for p in it.params), _num(it.lat, 9), _num(it.lon, 9),
                 _num(it.alt, 3), str(it.autocontinue)]
        lines.append("\t".join(cells))
    return "\n".join(lines) + "\n"


def read_waypoints(text: str) -> list[WireItem]:
    """Parse a WPL 110 file strictly: header, 12 columns, consecutive seq, known frames."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines or lines[0].strip() != WPL_HEADER:
        raise ValueError(f"not a '{WPL_HEADER}' file")
    out: list[WireItem] = []
    for n, line in enumerate(lines[1:], start=2):
        cells = line.split("\t") if "\t" in line else line.split()
        if len(cells) != 12:
            raise ValueError(f"line {n}: {len(cells)} columns, expected 12")
        try:
            seq, cur, frame, cmd = (int(c) for c in cells[:4])
            p = tuple(float(c) for c in cells[4:8])
            lat, lon, alt = (float(c) for c in cells[8:11])
            auto = int(cells[11])
        except ValueError:
            raise ValueError(f"line {n}: not numbers: {line!r}") from None
        if seq != len(out):
            raise ValueError(f"line {n}: seq {seq}, expected {len(out)}")
        if not all(math.isfinite(v) for v in (*p, lat, lon, alt)):
            raise ValueError(f"line {n}: NaN / inf")
        out.append(WireItem(seq=seq, current=cur, frame=frame, command=cmd,
                            params=(p[0], p[1], p[2], p[3]), lat=lat, lon=lon, alt=alt,
                            autocontinue=auto))
    check_frames(out)
    return out


# ── QGC .plan ────────────────────────────────────────────────────────────
def format_plan(plan: MissionPlan, origin: GeoPoint, items: list[WireItem]) -> str:
    frame = EnuFrame(origin)
    home = items[0]
    mission_items = []
    for it in items[1:]:
        entry = {"type": "SimpleItem", "autoContinue": True, "command": it.command,
                 "doJumpId": it.seq, "frame": it.frame,
                 "params": [*it.params, it.lat, it.lon, it.alt]}
        if it.command in POSITIONAL:
            entry.update({"Altitude": it.alt, "AltitudeMode": QGC_ALT_RELATIVE,
                          "AMSLAltAboveTerrain": None})
        mission_items.append(entry)
    poly = np.asarray(plan.fence.polygon_enu, float)
    lat, lon, _ = geodetic(frame, np.column_stack([poly, np.full(len(poly), plan.home.enu[2])]))
    rally = []
    if plan.rally_enu:
        r = np.asarray(plan.rally_enu, float)
        rlat, rlon, rh = geodetic(frame, r)
        # ArduPilot keeps a rally point's altitude in whole metres (AP_Rally RallyLocation.alt,
        # int16): 14.89 m would be held as 14 m, below the height that was checked. Round up.
        rally = [[float(a), float(b), float(math.ceil(h - home.alt - 1e-6))]
                 for a, b, h in zip(rlat, rlon, rh)]
    doc = {
        "fileType": "Plan",
        "groundStation": "AeroSplat-GIS",
        "version": 1,
        "mission": {
            "version": 2,
            "firmwareType": MAV_AUTOPILOT_ARDUPILOTMEGA,
            "vehicleType": MAV_TYPE_QUADROTOR,
            "cruiseSpeed": plan.profile.transit_speed_mps,
            "hoverSpeed": plan.profile.approach_speed_mps,
            "globalPlanAltitudeMode": QGC_ALT_RELATIVE,
            "plannedHomePosition": [home.lat, home.lon, home.alt],
            "items": mission_items,
        },
        "geoFence": {
            "version": 2,
            "circles": [],
            "polygons": [{"inclusion": True, "version": 1,
                          "polygon": [[float(a), float(b)] for a, b in zip(lat, lon)]}],
        },
        "rallyPoints": {"version": 2, "points": rally},
        # Not read by QGroundControl; for people and for the SITL runner.
        "aerosplat": _manifest(plan, origin),
    }
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def read_plan(text: str) -> list[WireItem]:
    """A `.plan` back to wire items (home first), refusing anything not relative to home."""
    doc = json.loads(text)
    if doc.get("fileType") != "Plan":
        raise ValueError("not a QGC plan")
    m = doc["mission"]
    if m.get("globalPlanAltitudeMode", QGC_ALT_RELATIVE) != QGC_ALT_RELATIVE:
        raise AltitudeFrameError(f"globalPlanAltitudeMode {m.get('globalPlanAltitudeMode')} is not "
                                 f"relative to launch ({QGC_ALT_RELATIVE})")
    hlat, hlon, halt = (float(v) for v in m["plannedHomePosition"])
    out = [WireItem(seq=0, current=1, frame=MAV_FRAME_GLOBAL, command=16, params=(0, 0, 0, 0),
                    lat=hlat, lon=hlon, alt=halt)]
    for n, it in enumerate(m["items"], start=1):
        if it.get("type") != "SimpleItem":
            raise ValueError(f"item {n}: only SimpleItem is supported, got {it.get('type')}")
        p = [0.0 if v is None else float(v) for v in it["params"]]
        if it["command"] in POSITIONAL and it.get("AltitudeMode") != QGC_ALT_RELATIVE:
            raise AltitudeFrameError(f"item {n}: AltitudeMode {it.get('AltitudeMode')} is not "
                                     f"relative to launch")
        out.append(WireItem(seq=n, current=0, frame=int(it["frame"]), command=int(it["command"]),
                            params=(p[0], p[1], p[2], p[3]), lat=p[4], lon=p[5], alt=p[6]))
    check_frames(out)
    return out


# ── .parm ────────────────────────────────────────────────────────────────
# The firmware the parameter names, units and values below are written for. Copter 4.7 renamed
# and re-unitised several (RTL_ALT [cm] -> RTL_ALT_M [m], WPNAV_SPEED [cm/s] -> WP_SPD [m/s]);
# tests check every name and value against the snapshot of this build's own parameter table
# (configs/ardupilot/<tag>.params.json), and that this is the build the SITL suite pins.
PARAMS_FIRMWARE = "Copter-4.7.2"
AVOID_MARGIN_RANGE_M = (1.0, 10.0)      # AVOID_MARGIN's documented range in PARAMS_FIRMWARE
# FS_OPTIONS bit 3: a vehicle already landing keeps landing whatever failsafe follows. Every
# other bit off, so no failsafe is ignored in AUTO ("continue mission" bits 0 and 1).
FS_OPTIONS_CONTINUE_IF_LANDING = 1 << 3


def parameters(plan: MissionPlan) -> list[tuple[str, float, str]]:
    """(name, value, why) for PARAMS_FIRMWARE (SIMULATION §4.5, idea 15).

    Every failsafe that can fire during the mission has its action here, so what a fault does
    follows from the package and not from whatever the vehicle was configured with. Trigger
    thresholds (battery voltages, timeouts) belong to the vehicle, not to the mission."""
    p = plan.profile
    z_rel = plan.transit_z - plan.home.enu[2]
    lo, hi = AVOID_MARGIN_RANGE_M
    margin = min(max(plan.target.handover.min_clearance_m, lo), hi)
    return [
        ("FENCE_ENABLE", 1, "geofence on"),
        ("FENCE_TYPE", 7, "max altitude + circle about home + polygon (the polygon is in mission.plan)"),
        ("FENCE_ACTION", 3, "breach: SmartRTL (retraces the checked path), else RTL, else land"),
        ("FENCE_ALT_MAX", plan.fence.alt_max_rel_m, "m"),
        ("FENCE_ALT_MAX_TP", 1, "FENCE_ALT_MAX is above home, like every altitude of the mission"),
        ("FENCE_RADIUS", plan.fence.radius_m, "m about home; holds the whole polygon"),
        ("FENCE_MARGIN", 2.0, "m"),
        ("RTL_ALT_TYPE", 0, "RTL altitude relative to home"),
        ("RTL_ALT_M", math.ceil(z_rel * 100.0) / 100.0, "m above home: the transit height that was checked"),
        ("RTL_ALT_FINAL_M", 0, "m: land at home"),
        ("RTL_CLIMB_MIN_M", 0, "m"),
        ("RTL_SPEED_MS", 0, "m/s: 0 = WP_SPD"),
        ("WP_SPD", p.transit_speed_mps, "m/s"),
        ("WP_SPD_UP", p.climb_speed_mps, "m/s"),
        ("WP_SPD_DN", p.descent_speed_mps, "m/s"),
        ("FS_THR_ENABLE", 4, "RC loss: SmartRTL or RTL"),
        ("FS_GCS_ENABLE", 3, "GCS loss: SmartRTL or RTL"),
        ("FS_OPTIONS", FS_OPTIONS_CONTINUE_IF_LANDING,
         "no failsafe is ignored in AUTO; one that comes while landing lets the landing finish"),
        ("BATT_FS_LOW_ACT", 3, "low battery: SmartRTL or RTL"),
        ("BATT_FS_CRT_ACT", 1, "critical battery: land"),
        ("FS_EKF_ACTION", 1, "EKF failsafe: land"),
        ("FS_DR_ENABLE", 1, "dead reckoning (no position source, e.g. GPS lost): land, like the EKF "
                            "failsafe; the firmware default (2) flies RTL without a position fix"),
        ("RALLY_INCL_HOME", 1, "home is a rally point (the only safe landing site v0 knows)"),
        ("AVOID_ENABLE", 3, "avoid fence + proximity sensors"),
        ("AVOID_MARGIN", margin, "m: the class's minimum clearance, held on board "
                                 f"(clamped to the documented {lo:g}..{hi:g} m)"),
    ]


def format_parm(plan: MissionPlan) -> str:
    lines = [f"# AeroSplat-GIS mission {plan.id}: fence and failsafe package",
             f"# site {plan.site_key}, target {plan.target.id} ({plan.target.ap_class}, "
             f"{plan.target.handover.mode.value}), verdict {plan.verdict.value}",
             "# altitudes relative to home; " + HOME_ALT_NOTE.split(";")[0],
             f"# names, units and values for {PARAMS_FIRMWARE} (checked against its parameter table)"]
    for name, value, why in parameters(plan):
        v = f"{value:g}" if isinstance(value, float) else str(value)
        lines.append(f"{name} {v}  # {why}")
    return "\n".join(lines) + "\n"


def read_parm(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip().replace(",", " ")
        if not line:
            continue
        name, value = line.split()
        out[name] = float(value)
    return out


# ── the whole export ─────────────────────────────────────────────────────
def _manifest(plan: MissionPlan, origin: GeoPoint) -> dict:
    # `origin`: the Site ENU frame the plan is in, so that a simulator can judge the flight
    # against the twin without the catalog.
    return {"mission_id": plan.id, "site": plan.site_key, "target_id": plan.target.id,
            "origin": {"lat": origin.lat, "lon": origin.lon, "h": origin.h},
            "access_point": plan.target.ap_id, "verdict": plan.verdict.value,
            "risks": [f.code.value for f in plan.risks],
            "altitude_frame": plan.altitude_frame.value, "home_enu": list(plan.home.enu),
            "home_amsl_verified": plan.home.amsl_verified, "home_altitude_note": HOME_ALT_NOTE,
            "occupancy_run": plan.occupancy.run_id, "taxonomy": plan.target.taxonomy}


def render(plan: MissionPlan, origin: GeoPoint, fmt: str,
           accept_risks: bool = False) -> tuple[str, str, str]:
    """(file name, media type, text) of one format, after the export gates."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown format '{fmt}' (use {', '.join(FORMATS)})")
    if plan.verdict is Verdict.infeasible:
        raise ExportRefused("the mission is infeasible: " + "; ".join(
            f"{f.code.value} ({f.leg})" if f.leg else f.code.value for f in plan.blocking))
    if plan.verdict is Verdict.feasible_with_risks and not accept_risks:
        raise ExportRefused("the mission has risks a person must accept first: " +
                            ", ".join(sorted({f.code.value for f in plan.risks})))
    items = to_wire(plan, origin)
    name, media = FORMATS[fmt]
    if fmt == "waypoints":
        text = format_waypoints(items)
        check_wire(plan, origin, read_waypoints(text))
    elif fmt == "plan":
        text = format_plan(plan, origin, items)
        check_wire(plan, origin, read_plan(text))
    else:
        text = format_parm(plan)
    return name, media, text


def enu_of_waypoints(text: str, origin: GeoPoint) -> dict[int, Vec3]:
    """Convenience for tests and the SITL runner: a WPL file's positions in Site ENU."""
    return wire_to_enu(read_waypoints(text), origin)


def point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    for k in range(n):
        x0, y0 = poly[k]
        x1, y1 = poly[(k + 1) % n]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def distance_to_boundary(x: float, y: float, poly: list[tuple[float, float]]) -> float:
    best = math.inf
    n = len(poly)
    for k in range(n):
        ax, ay = poly[k]
        bx, by = poly[(k + 1) % n]
        dx, dy = bx - ax, by - ay
        t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / max(dx * dx + dy * dy, 1e-12)))
        best = min(best, math.hypot(x - (ax + t * dx), y - (ay + t * dy)))
    return best
