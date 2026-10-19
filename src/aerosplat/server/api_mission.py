"""API v2, missions (step 8.1b): targets, mission planning, export for ArduPilot.

    POST /sites/{p}/{s}/targets                       a target from a verified access point
    GET  /sites/{p}/{s}/targets[/{target_id}]
    POST /sites/{p}/{s}/targets/{target_id}/confirm   an operator confirms a draft
    POST /sites/{p}/{s}/targets/{target_id}/takeoff   ranked takeoff spots near the target
    POST /sites/{p}/{s}/missions                      plan a confirmed target against the twin
    GET  /sites/{p}/{s}/missions[/{mission_id}]
    POST /sites/{p}/{s}/missions/{mission_id}/export/{fmt}   waypoints | plan | parm

Refusals come back as 409 with the named reasons (`detail.reasons`), all of them at once.
A mission is computed and stored whatever its verdict — the record of why it may not fly is
worth keeping — but only a feasible one is exported, and one with risks only after somebody
accepts them by name (ADR-0009, SIMULATION §3.1 PREFLIGHT).
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from aerosplat.authored.models import actor
from aerosplat.authored.store import AuthoredError, NotFound
from aerosplat.mission.export import ExportRefused
from aerosplat.mission.models import (AltitudeFrame, AltitudeFrameError, Code, Finding,
                                      HandoverOverride, MissionPlan, MissionRefused, MissionTarget,
                                      OccupancyRef, PlanningProfile, Severity, TakeoffSearch, Vec3,
                                      Verdict)
from aerosplat.mission.planner import plan_mission
from aerosplat.mission.store import MissionStore
from aerosplat.mission.takeoff import find_takeoff_spots, load_ground, transit_point
from aerosplat.mission.targets import TargetStore, planning_findings
from aerosplat.semantics.products import find_semantics
from aerosplat.server import api_v2
from aerosplat.server.api_semantics import _store, _taxonomy
from aerosplat.store.catalog import Catalog
from aerosplat.store.models import Site
from aerosplat.surface.products import OccupancyMissing, OccupancyProduct, require_occupancy
from aerosplat.taxonomy import HandoverMode

router = APIRouter(prefix="/api/v2", tags=["v2", "mission"])


def get_catalog() -> Catalog:
    return api_v2.get_catalog()


def _site(cat: Catalog, project_id: str, site_id: str) -> Site:
    return api_v2._site(cat, project_id, site_id)


def _targets(cat: Catalog, site: Site) -> TargetStore:
    return TargetStore(_store(cat, site))


def _missions(cat: Catalog, site: Site) -> MissionStore:
    return MissionStore(cat.site_dirs(site).missions, site.key)


def _refused(e: MissionRefused, status: int = 409) -> HTTPException:
    return HTTPException(status, {"message": str(e),
                                  "reasons": [f.model_dump(mode="json") for f in e.findings]})


# ── schemas ──────────────────────────────────────────────────────────────
class _In(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TargetCreate(_In):
    ap_id: str
    handover_mode: Optional[HandoverMode] = None        # default: the class's
    override: Optional[HandoverOverride] = None         # only towards caution
    # σ the stored access point does not carry, stated by the operator (recorded as such).
    sigma_ap_m: Optional[float] = Field(None, ge=0, le=100)
    sigma_georef_m: Optional[float] = Field(None, ge=0, le=100)
    confirm: bool = False
    by: Optional[str] = None
    notes: str = ""


class TargetConfirm(_In):
    by: Optional[str] = None


class MissionCreate(_In):
    target_id: str
    run_id: Optional[str] = None                        # default: the newest run with occupancy
    home_enu: Optional[Vec3] = None                     # default: the Site origin, on the surface
    # Instead of home_enu: the nearest spot the takeoff search proposes (409 when there is none).
    home_search: bool = False
    profile: Optional[PlanningProfile] = None
    altitude_frame: AltitudeFrame = AltitudeFrame.relative_home
    by: Optional[str] = None


class TakeoffQuery(_In):
    run_id: Optional[str] = None                        # default: the newest run with occupancy
    profile: Optional[PlanningProfile] = None           # σ_GNSS, r_drone, margins
    max_distance_m: float = Field(150.0, gt=0, le=1000)
    n: int = Field(5, ge=1, le=20)
    ground_tol_m: float = Field(1.0, ge=0, le=5)


class MissionExport(_In):
    accept_risks: bool = False                          # required for feasible_with_risks
    by: Optional[str] = None


class MissionSummary(BaseModel):
    id: str
    target_id: str
    ap_id: str
    verdict: Verdict
    findings: list[Finding]
    created_at: AwareDatetime
    created_by: str


# ── targets ──────────────────────────────────────────────────────────────
@router.post("/sites/{project_id}/{site_id}/targets", response_model=MissionTarget,
             status_code=201)
def create_target(project_id: str, site_id: str, body: TargetCreate,
                  cat: Catalog = Depends(get_catalog)) -> MissionTarget:
    """Propose (or, with `confirm`, confirm) a target. Refused with every reason at once."""
    site = _site(cat, project_id, site_id)
    try:
        return _targets(cat, site).create(
            _taxonomy(site), body.ap_id, by=body.by, mode=body.handover_mode,
            override=body.override, sigma_ap_m=body.sigma_ap_m,
            sigma_georef_m=body.sigma_georef_m, confirm=body.confirm, notes=body.notes)
    except MissionRefused as e:
        raise _refused(e, 404 if [f.code.value for f in e.findings] == ["ap_missing"]
                       else 409) from None


@router.get("/sites/{project_id}/{site_id}/targets", response_model=list[MissionTarget])
def list_targets(project_id: str, site_id: str,
                 cat: Catalog = Depends(get_catalog)) -> list[MissionTarget]:
    return _targets(cat, _site(cat, project_id, site_id)).targets()


@router.get("/sites/{project_id}/{site_id}/targets/{target_id}", response_model=MissionTarget)
def get_target(project_id: str, site_id: str, target_id: str,
               cat: Catalog = Depends(get_catalog)) -> MissionTarget:
    try:
        return _targets(cat, _site(cat, project_id, site_id)).get(target_id)
    except NotFound as e:
        raise HTTPException(404, str(e)) from None


@router.post("/sites/{project_id}/{site_id}/targets/{target_id}/confirm",
             response_model=MissionTarget)
def confirm_target(project_id: str, site_id: str, target_id: str, body: TargetConfirm,
                   cat: Catalog = Depends(get_catalog)) -> MissionTarget:
    """An operator confirms a draft; the gates run again on the access point as it is now."""
    site = _site(cat, project_id, site_id)
    try:
        return _targets(cat, site).confirm(_taxonomy(site), target_id, by=body.by)
    except NotFound as e:
        raise HTTPException(404, str(e)) from None
    except MissionRefused as e:
        raise _refused(e) from None


# ── takeoff spots ────────────────────────────────────────────────────────
def _search_takeoff(cat: Catalog, site: Site, product: OccupancyProduct, target: MissionTarget,
                    ap, profile: Optional[PlanningProfile], *, max_distance_m: float = 150.0,
                    n: int = 5, ground_tol_m: float = 1.0) -> TakeoffSearch:
    """The search on the occupancy of one run and the semantics terrain of the same run."""
    sem = find_semantics(cat, site, run_id=product.run.id)
    if sem is None:
        raise HTTPException(404, f"run {product.run.id} has occupancy but no semantics: the "
                                 f"takeoff search needs its DTM and building map")
    return find_takeoff_spots(product.load_grid(), load_ground(sem.dir), transit_point(target, ap),
                              sigma_georef_m=target.sigma.georef.value_m, profile=profile,
                              max_distance_m=max_distance_m, n=n, ground_tol_m=ground_tol_m)


@router.post("/sites/{project_id}/{site_id}/targets/{target_id}/takeoff",
             response_model=TakeoffSearch)
def takeoff_spots(project_id: str, site_id: str, target_id: str,
                  body: TakeoffQuery = TakeoffQuery(),
                  cat: Catalog = Depends(get_catalog)) -> TakeoffSearch:
    """Ranked takeoff spots near a target, checked as a plan's home and takeoff would be.

    An empty `candidates` with a `message` that says why is an answer, not an error."""
    site = _site(cat, project_id, site_id)
    targets = _targets(cat, site)
    try:
        target = targets.get(target_id)
        ap = targets.authored.get(target.ap_id)
    except NotFound as e:
        raise HTTPException(404, str(e)) from None
    try:
        product = require_occupancy(cat, site, run_id=body.run_id)
    except OccupancyMissing as e:
        raise HTTPException(404, str(e)) from None
    return _search_takeoff(cat, site, product, target, ap, body.profile,
                           max_distance_m=body.max_distance_m, n=body.n,
                           ground_tol_m=body.ground_tol_m)


# ── missions ─────────────────────────────────────────────────────────────
@router.post("/sites/{project_id}/{site_id}/missions", response_model=MissionPlan,
             status_code=201)
def create_mission(project_id: str, site_id: str, body: MissionCreate,
                   cat: Catalog = Depends(get_catalog)) -> MissionPlan:
    """Plan a confirmed target against the newest occupancy of the Site, and store the plan.

    201 whatever the verdict: `verdict` and `findings` say whether it may fly.
    """
    if body.home_search and body.home_enu is not None:
        raise HTTPException(422, "home_enu and home_search exclude each other")
    site = _site(cat, project_id, site_id)
    targets = _targets(cat, site)
    try:
        target = targets.get(body.target_id)
    except NotFound as e:
        raise HTTPException(404, str(e)) from None
    try:
        ap = targets.authored.get(target.ap_id)
    except NotFound:
        ap = None
    try:
        product = require_occupancy(cat, site, run_id=body.run_id)
    except OccupancyMissing as e:
        raise HTTPException(404, str(e)) from None
    report = product.report
    ref = OccupancyRef(run_id=product.run.id, epoch_id=product.run.epoch_id, format=report.format,
                       res_m=report.grid.res_m, unknown_mode=report.params.get("unknown"),
                       carved=report.stats.carving is not None)
    try:
        if ap is None:
            raise MissionRefused(planning_findings(target, None))
        home_enu, home_source = body.home_enu, None
        if body.home_search:
            search = _search_takeoff(cat, site, product, target, ap, body.profile)
            if not search.candidates:
                raise MissionRefused([Finding(code=Code.no_takeoff_spot,
                                              severity=Severity.blocking, detail=search.message)])
            best = search.candidates[0]
            home_enu = best.enu
            home_source = (f"takeoff search: nearest of {len(search.candidates)} spots, "
                           f"{best.distance_m:.0f} m from the target, ≈{best.open_radius_m:.1f} m "
                           f"of open ground (r {search.radius_m:.2f} m)")
        plan = plan_mission(target=target, ap=ap, grid=product.load_grid(), occupancy=ref,
                            origin=site.origin, by=actor(body.by), profile=body.profile,
                            home_enu=home_enu, altitude_frame=body.altitude_frame,
                            home_source=home_source)
    except MissionRefused as e:
        raise _refused(e) from None
    except AltitudeFrameError as e:
        raise HTTPException(422, str(e)) from None
    _missions(cat, site).save(plan)
    return plan


@router.get("/sites/{project_id}/{site_id}/missions", response_model=list[MissionSummary])
def list_missions(project_id: str, site_id: str,
                  cat: Catalog = Depends(get_catalog)) -> list[MissionSummary]:
    store = _missions(cat, _site(cat, project_id, site_id))
    out = []
    for mid in store.ids():
        try:
            p = store.load(mid)
        except AuthoredError:
            continue
        out.append(MissionSummary(id=p.id, target_id=p.target.id, ap_id=p.target.ap_id,
                                  verdict=p.verdict, findings=p.findings,
                                  created_at=p.created_at, created_by=p.created_by))
    return out


@router.get("/sites/{project_id}/{site_id}/missions/{mission_id}", response_model=MissionPlan)
def get_mission(project_id: str, site_id: str, mission_id: str,
                cat: Catalog = Depends(get_catalog)) -> MissionPlan:
    try:
        return _missions(cat, _site(cat, project_id, site_id)).load(mission_id)
    except NotFound as e:
        raise HTTPException(404, str(e)) from None


@router.post("/sites/{project_id}/{site_id}/missions/{mission_id}/export/{fmt}",
             responses={200: {"content": {"text/plain": {}, "application/json": {}},
                              "description": "the file, also written to missions/<id>/"}})
def export_mission(project_id: str, site_id: str, mission_id: str, fmt: str,
                   body: MissionExport = MissionExport(),
                   cat: Catalog = Depends(get_catalog)) -> Response:
    """One flyable format. Refused (409) for an infeasible mission, for one with risks nobody
    accepted, and when the target was revoked or its access point changed since planning."""
    site = _site(cat, project_id, site_id)
    store = _missions(cat, site)
    try:
        plan = store.load(mission_id)
    except NotFound as e:
        raise HTTPException(404, str(e)) from None
    try:
        path, media, text = store.export(plan, fmt, origin=site.origin, targets=_targets(cat, site),
                                         by=body.by, accept_risks=body.accept_risks)
    except ExportRefused as e:
        raise HTTPException(409, str(e)) from None
    except AltitudeFrameError as e:
        raise HTTPException(422, str(e)) from None
    except ValueError as e:
        raise HTTPException(404, str(e)) from None
    return Response(content=text, media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="{mission_id}-{path.name}"'})
