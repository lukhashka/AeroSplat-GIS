"""API v2, flights (SIMULATION §2 idea 12): a mission's geometry and its simulated runs, read only.

    GET /sites/{p}/{s}/sims                                         every run of every mission
    GET /sites/{p}/{s}/missions/{mission_id}/geometry               what to draw, Site ENU + WGS84
    GET /sites/{p}/{s}/missions/{mission_id}/sims                   runs of one mission
    GET /sites/{p}/{s}/missions/{mission_id}/sims/{scenario}/report       report.json as written
    GET /sites/{p}/{s}/missions/{mission_id}/sims/{scenario}/trajectory   truth, phases, clearance

A run is `missions/<id>/sim/<scenario>/` (aerosplat.sim.runner), or `.../<scenario>/seed-<n>/`
for a scenario flown with several seeds (`?seed=n`). Nothing here writes or recomputes: the
clearance comes from the run's own `clearance.json` (the oracle's series), or is null.

Frames: every point is given in Site ENU (x east, y north, z up, metres from the Site origin)
and, for the 2D map, as WGS84 lon/lat through the same `EnuFrame` the export used — so a
waypoint here is exactly the row of `mission.waypoints`. The 3D viewer adds the scene's own
transform (scene-meta: centre, up-axis rotation, scale) on top.
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import AwareDatetime, BaseModel

from aerosplat.authored.store import AuthoredError, NotFound
from aerosplat.geo.crs import EnuFrame, GeoPoint
from aerosplat.mission.export import enu_from_geodetic, geodetic
from aerosplat.mission.models import Finding, MissionPlan, Vec3, Verdict
from aerosplat.mission.store import MissionStore
from aerosplat.server import api_v2
from aerosplat.sim.record import Recording
from aerosplat.sim.replay import PHASES, failsafe_mask, mission_start, phases, segments
from aerosplat.sim.report import CLEARANCE_JSON, REPORT_JSON, TRAJECTORY_JSON
from aerosplat.store.catalog import Catalog
from aerosplat.store.models import Site

router = APIRouter(prefix="/api/v2", tags=["v2", "sim"])

SIM_DIR = "sim"
SCENARIO = re.compile(r"^[a-z0-9][a-z0-9._-]{0,80}$")     # aerosplat.sim.scenario.Scenario
SEED_DIR = re.compile(r"^seed-(\d{1,4})$")
LEGACY_RUN = re.compile(r"^legacy:(?P<ds>.+)_(?:dense|cloud|splat)(?:_wgs84)?\.pt$")

LonLat = tuple[float, float]


def get_catalog() -> Catalog:
    return api_v2.get_catalog()


def _missions(cat: Catalog, site: Site) -> MissionStore:
    return MissionStore(cat.site_dirs(site).missions, site.key)


def _plan(cat: Catalog, project_id: str, site_id: str, mission_id: str) -> tuple[Site, MissionStore, MissionPlan]:
    site = api_v2._site(cat, project_id, site_id)
    store = _missions(cat, site)
    try:
        return site, store, store.load(mission_id)
    except NotFound as e:
        raise HTTPException(404, str(e)) from None
    except AuthoredError as e:
        raise HTTPException(500, str(e)) from None


# ── schemas ──────────────────────────────────────────────────────────────
class GeoVertex(BaseModel):
    enu: Vec3
    lon: float
    lat: float
    h_ellipsoid: float


class GeoWaypoint(GeoVertex):
    seq: int                                # as uploaded: seq 0 is home
    role: str
    command: int
    alt_rel_m: float                        # above home, as in mission.waypoints (frame 3)


class GeoLeg(BaseModel):
    name: str
    points_enu: list[Vec3]
    lonlat: list[LonLat]
    radius_m: float                         # the clearance the plan promised along it
    sigma_m: float
    min_clearance_m: Optional[float]
    excluded_surface: Optional[str]
    risk_only: bool


class GeoFence(BaseModel):
    polygon_enu: list[tuple[float, float]]
    lonlat: list[LonLat]
    floor_z: float                          # home height: FENCE_ALT_MAX is above home
    ceiling_z: float
    alt_max_rel_m: float
    radius_m: float


class GeoTarget(BaseModel):
    kind: str                               # opening | pad
    ap_id: str
    ap_class: str
    facade_id: Optional[str]
    polygon_enu: list[Vec3]                 # the access point's outline, 3D
    lonlat: list[LonLat]
    center: GeoVertex
    normal: Vec3
    direction: Vec3                         # from the target towards E
    hover: GeoVertex                        # H
    entry: GeoVertex                        # E
    heading_deg: Optional[float]            # compass yaw at H; None: free


class TwinRef(BaseModel):
    """The twin the plan was checked against, and the 3D scene that shows it."""

    run_id: str
    epoch_id: str
    format: str
    res_m: float
    # ?dataset= of the viewer: the legacy dataset the occupancy came from, or the Site's scene.
    scene: str
    legacy: bool


class MissionGeometry(BaseModel):
    id: str
    site_key: str
    verdict: Verdict
    findings: list[Finding]
    frame: str = "site_enu"
    origin: GeoPoint                        # of Site ENU
    home: GeoVertex
    transit_z: float
    waypoints: list[GeoWaypoint]            # positional items only
    legs: list[GeoLeg]
    fence: GeoFence
    target: GeoTarget
    rally: list[GeoVertex]
    r_drone_m: float
    bbox_enu: tuple[Vec3, Vec3]             # everything above, for framing a view
    twin: TwinRef


class SimVerdict(BaseModel):
    invariant: str
    status: str
    expected: Optional[str] = None
    detail: str = ""


class SimRunSummary(BaseModel):
    mission_id: str
    scenario: str
    seed: Optional[int]                     # None: the scenario's only run
    description: str
    passed: bool
    outcome: str
    reason: Optional[str]
    summary: str
    verdicts: list[SimVerdict]
    samples: int
    mission_sim_s: Optional[float]
    firmware: Optional[str]
    has_trajectory: bool
    has_clearance: bool
    written_at: AwareDatetime


class SimEvent(BaseModel):
    t: float
    kind: str
    detail: str
    data: dict[str, Any]


class SegmentOut(BaseModel):
    phase: str
    failsafe: bool
    mode: str
    i0: int
    i1: int


class ClearanceOut(BaseModel):
    clearance_m: list[Optional[float]]      # aligned with `t`; null where the oracle has none
    target_distance_m: list[Optional[float]]
    required_m: float
    cap_m: Optional[float]                  # values at the cap mean "at least that"


class FlightTrajectory(BaseModel):
    mission_id: str
    scenario: str
    seed: Optional[int]
    origin: GeoPoint                        # what the recording was converted with
    origin_offset_m: float                  # of that origin from the Site's, in Site ENU
    home_enu: Vec3
    mission_start_t: Optional[float]
    t: list[float]                          # simulation seconds since boot
    enu: list[Vec3]                         # truth (SIM_STATE), Site ENU
    lonlat: list[LonLat]
    alt_rel_m: list[float]                  # above home
    speed_mps: list[float]
    yaw_deg: list[float]
    mode: list[str]
    armed: list[bool]
    seq: list[int]
    phase: list[str]
    failsafe: list[bool]
    segments: list[SegmentOut]
    phases: list[str]                       # every phase name, in flight order
    clearance: Optional[ClearanceOut]
    events: list[SimEvent]


# ── geometry ─────────────────────────────────────────────────────────────
def _vertices(frame: EnuFrame, pts) -> list[GeoVertex]:
    a = np.asarray(pts, float).reshape(-1, 3)
    if not len(a):
        return []
    lat, lon, h = geodetic(frame, a)
    return [GeoVertex(enu=(float(p[0]), float(p[1]), float(p[2])), lon=float(lo), lat=float(la),
                      h_ellipsoid=float(hh)) for p, la, lo, hh in zip(a, lat, lon, h)]


def _lonlat(frame: EnuFrame, pts) -> list[LonLat]:
    return [(v.lon, v.lat) for v in _vertices(frame, pts)]


def target_outline(plan: MissionPlan) -> list[Vec3]:
    """The access point's outline in Site ENU: a pad's polygon at its height, or an opening's
    (u, v) polygon on the host facade (v is the ENU height, aerosplat.semantics Facade.point)."""
    ap = plan.access_point
    if ap.pad is not None:
        return [(float(x), float(y), float(ap.pad.z)) for x, y in ap.pad.polygon_enu]
    if ap.geometry is None or ap.host is None:
        return []
    o = np.asarray(ap.host.origin, float)
    u_axis = np.asarray(ap.host.u_axis, float)
    out = []
    for u, v in ap.geometry.polygon_uv:
        p = o + float(u) * u_axis
        out.append((float(p[0]), float(p[1]), float(v)))
    return out


def twin_scene(plan: MissionPlan, site: Site) -> TwinRef:
    occ = plan.occupancy
    m = LEGACY_RUN.match(occ.run_id)
    return TwinRef(run_id=occ.run_id, epoch_id=occ.epoch_id, format=occ.format, res_m=occ.res_m,
                   scene=m.group("ds") if m else site.key, legacy=m is not None)


def mission_geometry(plan: MissionPlan, site: Site) -> MissionGeometry:
    origin = site.origin
    frame = EnuFrame(origin)
    home = _vertices(frame, [plan.home.enu])[0]
    pos = [(seq, it) for seq, it in enumerate(plan.items, start=1) if it.enu is not None]
    wps = []
    for (seq, it), v in zip(pos, _vertices(frame, [it.enu for _, it in pos])):
        wps.append(GeoWaypoint(**v.model_dump(), seq=seq, role=it.role, command=it.command,
                               alt_rel_m=v.h_ellipsoid - home.h_ellipsoid))
    legs = [GeoLeg(name=lg.name, points_enu=list(lg.points), lonlat=_lonlat(frame, lg.points),
                   radius_m=lg.radius_m, sigma_m=lg.sigma_m, min_clearance_m=lg.min_clearance_m,
                   excluded_surface=lg.excluded_surface, risk_only=lg.risk_only)
            for lg in plan.legs]
    hz = float(plan.home.enu[2])
    f = plan.fence
    fence = GeoFence(polygon_enu=list(f.polygon_enu),
                     lonlat=_lonlat(frame, [(x, y, hz) for x, y in f.polygon_enu]),
                     floor_z=hz, ceiling_z=hz + f.alt_max_rel_m, alt_max_rel_m=f.alt_max_rel_m,
                     radius_m=f.radius_m)
    ap, a = plan.access_point, plan.approach
    outline = target_outline(plan)
    c, hv, en = _vertices(frame, [a.center, a.hover, a.entry])
    target = GeoTarget(kind=a.kind, ap_id=ap.id, ap_class=ap.class_,
                       facade_id=ap.host.facade_id if ap.host is not None else None,
                       polygon_enu=outline, lonlat=_lonlat(frame, outline), center=c,
                       normal=a.normal, direction=a.direction, hover=hv, entry=en,
                       heading_deg=a.heading_deg)
    every = np.asarray([plan.home.enu, *[it.enu for _, it in pos], *outline,
                        *[(x, y, hz) for x, y in f.polygon_enu]], float)
    lo, hi = every.min(axis=0), every.max(axis=0)
    return MissionGeometry(
        id=plan.id, site_key=plan.site_key, verdict=plan.verdict, findings=plan.findings,
        origin=origin, home=home, transit_z=plan.transit_z, waypoints=wps, legs=legs, fence=fence,
        target=target, rally=_vertices(frame, plan.rally_enu), r_drone_m=plan.profile.r_drone_m,
        bbox_enu=((float(lo[0]), float(lo[1]), float(lo[2])), (float(hi[0]), float(hi[1]), float(hi[2]))),
        twin=twin_scene(plan, site))


# ── runs ─────────────────────────────────────────────────────────────────
def run_dirs(mission_dir: Path) -> list[tuple[str, Optional[int], Path]]:
    """(scenario, seed, folder) of every run with a report, in name order."""
    root = mission_dir / SIM_DIR
    if not root.is_dir():
        return []
    out: list[tuple[str, Optional[int], Path]] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not SCENARIO.match(d.name):
            continue
        if (d / REPORT_JSON).is_file():
            out.append((d.name, None, d))
        seeds = [(int(m.group(1)), s) for s in d.iterdir()
                 if s.is_dir() and (m := SEED_DIR.match(s.name)) and (s / REPORT_JSON).is_file()]
        out.extend((d.name, n, s) for n, s in sorted(seeds))
    return out


def run_dir(mission_dir: Path, scenario: str, seed: Optional[int]) -> Path:
    if not SCENARIO.match(scenario):                  # also keeps names from leaving the folder
        raise HTTPException(404, f"'{scenario}' is not a scenario name")
    d = mission_dir / SIM_DIR / scenario
    if seed is not None:
        d = d / f"seed-{seed}"
    if not (d / REPORT_JSON).is_file():
        raise HTTPException(404, f"no run '{scenario}'" + (f" seed {seed}" if seed is not None else "")
                            + f" of mission {mission_dir.name}")
    return d


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise HTTPException(500, f"{path.name}: {e}") from None
    if not isinstance(data, dict):
        raise HTTPException(500, f"{path.name}: not a JSON object")
    return data


def summarize(mission_id: str, scenario: str, seed: Optional[int], folder: Path) -> SimRunSummary:
    r = _read_json(folder / REPORT_JSON)
    expected = (r.get("expected") or {}).get("invariants") or {}
    return SimRunSummary(
        mission_id=mission_id, scenario=scenario, seed=seed,
        description=str((r.get("scenario") or {}).get("description", "")),
        passed=bool(r.get("passed")), outcome=str(r.get("outcome", "")), reason=r.get("reason"),
        summary=str(r.get("summary", "")),
        verdicts=[SimVerdict(invariant=v["invariant"], status=v["status"],
                             expected=v.get("expected") or expected.get(v["invariant"]),
                             detail=v.get("detail", "")) for v in r.get("verdicts", [])],
        samples=int(r.get("samples") or 0),
        mission_sim_s=(r.get("timing") or {}).get("mission_sim_s"),
        firmware=(r.get("firmware") or {}).get("tag"),
        has_trajectory=(folder / TRAJECTORY_JSON).is_file(),
        has_clearance=(folder / CLEARANCE_JSON).is_file(),
        written_at=datetime.fromtimestamp(os.path.getmtime(folder / REPORT_JSON), timezone.utc))


def _aligned(t: np.ndarray, ct: list[float], values: list[Optional[float]]) -> list[Optional[float]]:
    """Values on the clearance clock (the airborne samples, rounded to the ms by
    aerosplat.sim.report) -> one per recording sample, None where it has none."""
    index = {round(float(x), 3): k for k, x in enumerate(t)}
    out: list[Optional[float]] = [None] * len(t)
    for tc, v in zip(ct, values):
        k = index.get(round(float(tc), 3))
        if k is not None:
            out[k] = v
    return out


def _clearance(folder: Path, t: np.ndarray) -> Optional[ClearanceOut]:
    path = folder / CLEARANCE_JSON
    if not path.is_file():
        return None
    c = _read_json(path)
    ct = c.get("t") or []
    return ClearanceOut(clearance_m=_aligned(t, ct, c.get("clearance_m") or []),
                        target_distance_m=_aligned(t, ct, c.get("target_distance_m") or []),
                        required_m=float(c.get("required_m", math.nan)), cap_m=c.get("cap_m"))


def flight(plan: MissionPlan, site: Site, scenario: str, seed: Optional[int],
           folder: Path) -> FlightTrajectory:
    path = folder / TRAJECTORY_JSON
    if not path.is_file():
        raise HTTPException(404, f"run '{scenario}' has no trajectory (refused before flying?)")
    try:
        rec = Recording.load(path)
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise HTTPException(500, f"{path.name}: {e}") from None
    report = _read_json(folder / REPORT_JSON)
    tol = ((report.get("scenario") or {}).get("expect") or {}).get("tolerances") or {}
    origin = GeoPoint(**rec.meta["origin"]) if rec.meta.get("origin") else site.origin
    t = rec.times
    p = rec.positions
    lat, lon, _ = geodetic(EnuFrame(origin), p) if len(p) else (np.zeros(0),) * 3
    offset = 0.0
    if origin != site.origin:
        offset = float(np.linalg.norm(enu_from_geodetic(EnuFrame(site.origin), origin.lat,
                                                        origin.lon, origin.h)[0]))
    home = rec.meta.get("home_enu") or list(plan.home.enu)
    start = mission_start(rec)
    fs = failsafe_mask(rec, start)
    ph = phases(rec, plan, hover_m=float(tol.get("hover_m", 1.0)),
                hover_alt_m=float(tol.get("hover_alt_m", 0.5)), failsafe=fs)
    v = np.asarray(rec.vel_ned, float).reshape(-1, 3)
    seen = set(ph)
    return FlightTrajectory(
        mission_id=plan.id, scenario=scenario, seed=seed, origin=origin, origin_offset_m=offset,
        home_enu=(float(home[0]), float(home[1]), float(home[2])), mission_start_t=start,
        t=[float(x) for x in t], enu=[(float(a), float(b), float(c)) for a, b, c in p],
        lonlat=[(float(a), float(b)) for a, b in zip(lon, lat)],
        alt_rel_m=[float(z - home[2]) for z in p[:, 2]] if len(p) else [],
        speed_mps=[float(s) for s in np.linalg.norm(v, axis=1)] if len(v) else [],
        yaw_deg=list(rec.yaw_deg), mode=list(rec.mode), armed=list(rec.armed), seq=list(rec.seq),
        phase=ph, failsafe=[bool(x) for x in fs],
        segments=[SegmentOut(**s.__dict__) for s in segments(ph, fs, list(rec.mode))],
        phases=[x for x in PHASES if x in seen],
        clearance=_clearance(folder, t),
        events=[SimEvent(t=e.t, kind=e.kind, detail=e.detail, data=e.data) for e in rec.events])


# ── endpoints ────────────────────────────────────────────────────────────
@router.get("/sites/{project_id}/{site_id}/sims", response_model=list[SimRunSummary])
def list_site_sims(project_id: str, site_id: str,
                   cat: Catalog = Depends(get_catalog)) -> list[SimRunSummary]:
    """Every simulated run of every mission of the Site, mission by mission."""
    site = api_v2._site(cat, project_id, site_id)
    store = _missions(cat, site)
    return [summarize(mid, sc, seed, d) for mid in store.ids()
            for sc, seed, d in run_dirs(store.folder(mid))]


@router.get("/sites/{project_id}/{site_id}/missions/{mission_id}/geometry",
            response_model=MissionGeometry)
def get_mission_geometry(project_id: str, site_id: str, mission_id: str,
                         cat: Catalog = Depends(get_catalog)) -> MissionGeometry:
    site, _, plan = _plan(cat, project_id, site_id, mission_id)
    return mission_geometry(plan, site)


@router.get("/sites/{project_id}/{site_id}/missions/{mission_id}/sims",
            response_model=list[SimRunSummary])
def list_mission_sims(project_id: str, site_id: str, mission_id: str,
                      cat: Catalog = Depends(get_catalog)) -> list[SimRunSummary]:
    _, store, plan = _plan(cat, project_id, site_id, mission_id)
    return [summarize(plan.id, sc, seed, d) for sc, seed, d in run_dirs(store.folder(plan.id))]


@router.get("/sites/{project_id}/{site_id}/missions/{mission_id}/sims/{scenario}/report")
def get_sim_report(project_id: str, site_id: str, mission_id: str, scenario: str,
                   seed: Optional[int] = Query(None, ge=0),
                   cat: Catalog = Depends(get_catalog)) -> dict[str, Any]:
    """report.json of the run, as the runner wrote it (verdicts, events, timing, parameters)."""
    _, store, plan = _plan(cat, project_id, site_id, mission_id)
    return _read_json(run_dir(store.folder(plan.id), scenario, seed) / REPORT_JSON)


@router.get("/sites/{project_id}/{site_id}/missions/{mission_id}/sims/{scenario}/trajectory",
            response_model=FlightTrajectory)
def get_sim_trajectory(project_id: str, site_id: str, mission_id: str, scenario: str,
                       seed: Optional[int] = Query(None, ge=0),
                       cat: Catalog = Depends(get_catalog)) -> FlightTrajectory:
    site, store, plan = _plan(cat, project_id, site_id, mission_id)
    return flight(plan, site, scenario, seed, run_dir(store.folder(plan.id), scenario, seed))
