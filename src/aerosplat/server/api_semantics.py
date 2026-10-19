"""API v2, semantic layer: facades, facade orthophotos, authored access points.

Mounted next to `api_v2` and sharing its catalog handle and render proxy. Torch-free
like the rest of the API: the orthophoto pose is plain arithmetic on the stored facade
(`aerosplat.semantics.ortho`), and the GPU work stays in the render service, which
knows nothing about facades.

Two kinds of data meet here and must not be confused:

* **derived** — facades and coverage maps, read from the newest run that produced them
  and thrown away when that run is recomputed;
* **authored** — the access points a person drew, in `sites/<site>/authored/`, which
  outlive every run (ADR-0006) and are never written by the pipeline.

Nothing is ever auto-confirmed (ADR-0009): a feature's `lifecycle` only changes
because somebody said so, and every change is appended to `authored/audit.jsonl`.
"""

from __future__ import annotations

import dataclasses
import io
import json
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from aerosplat.authored.candidates import Proposal, make_candidates, screen
from aerosplat.authored.carry import (PlacementStatus, foreign_on, is_native, native_features,
                                     reanchor)
from aerosplat.authored.geometry import geometry_from_uv, make_access_point
from aerosplat.authored.models import AccessPoint, Address, AuditEntry, Lifecycle, Origin, actor
from aerosplat.authored.store import AuthoredError, AuthoredStore, NotFound as ApNotFound
from aerosplat.semantics.addressing import (SNAP_TOL_M, Opening, SnapBox, address_facade,
                                           canonical_address, floor_label, snap_to_grid)
from aerosplat.semantics.gridfit import CoverageRaster, GridfitParams, GridfitResult, gridfit
from aerosplat.semantics.matching import (VERSION as MATCH_VERSION, MatchStatus, Relation,
                                          match_facades)
from aerosplat.semantics.models import Building, Coverage, Facade, TerrainModel
from aerosplat.semantics.ortho import (FacadeGrid, FacadeOrthoError, FacadeOrthoSpec,
                                       facade_ortho)
from aerosplat.semantics.products import (SemanticsMissing, SemanticsProduct, require_semantics,
                                          semantics_runs)
from aerosplat.render.models import RenderRequest
from aerosplat.server import api_v2
from aerosplat.store.catalog import Catalog
from aerosplat.store.models import Site
from aerosplat.store.scenes import SceneLookupError, run_scene_source
from aerosplat.taxonomy import ApKind, Taxonomy, TaxonomyError, cached_taxonomy
from aerosplat.taxonomy.models import Attribute, TaxonomyClass

router = APIRouter(prefix="/api/v2", tags=["v2", "semantics"])

RENDER_META_HEADER = api_v2.RENDER_META_HEADER
MAX_AUDIT = 1000


# ── shared plumbing ──────────────────────────────────────────────────────
def get_catalog() -> Catalog:
    return api_v2.get_catalog()


def _site(cat: Catalog, project_id: str, site_id: str) -> Site:
    return api_v2._site(cat, project_id, site_id)


def _semantics(cat: Catalog, site: Site, run_id: Optional[str] = None) -> SemanticsProduct:
    try:
        return require_semantics(cat, site, run_id=run_id)
    except SemanticsMissing as e:
        raise HTTPException(404, str(e)) from None


def _facade(product: SemanticsProduct, facade_id: str) -> Facade:
    try:
        return product.facade(facade_id)
    except KeyError:
        raise HTTPException(404, f"run {product.run.id} has no facade '{facade_id}' "
                                 f"(it has {len(product.result.facades)})") from None


def _taxonomy(site: Site) -> Taxonomy:
    try:
        return cached_taxonomy(site.taxonomy_packs)
    except TaxonomyError as e:
        raise HTTPException(500, f"taxonomy packs {site.taxonomy_packs}: {e}") from None


def _store(cat: Catalog, site: Site) -> AuthoredStore:
    return AuthoredStore.for_site(cat.site_dirs(site), site.key)


def _run_has_scene(cat: Catalog, site: Site, product: SemanticsProduct) -> bool:
    try:
        run_scene_source(cat, site, product.run)
        return True
    except SceneLookupError:
        return False


def _run_scene(cat: Catalog, site: Site, product: SemanticsProduct) -> str:
    """The scene name to render this run's facades from, pinned to that run.

    A facade belongs to the run that fitted it; rendering it from another run's splat
    would put this run's frame over a different reconstruction (session 2026-10-18…19,
    error 20). So a run without its own splat is a 404, not a fallback to the newest.
    """
    try:
        run_scene_source(cat, site, product.run)
    except SceneLookupError as e:
        raise HTTPException(404, f"{e}; the facades of this run cannot be rendered from "
                                 f"another run's scene") from None
    return site.key


def _pinned_render(scene: str, request: RenderRequest, run_id: str) -> tuple[bytes, str]:
    """Render through the service pinned to `run_id`, and check that it was that run."""
    png, meta_json = api_v2._render_call(
        scene, request.model_copy(update={"run": run_id}).model_dump_json(exclude_none=True))
    try:
        got = json.loads(meta_json).get("run")
    except (ValueError, AttributeError):
        got = None
    if got != run_id:
        raise HTTPException(502, f"render service rendered run {got!r} instead of {run_id!r} "
                                 f"(is it older than the API? restart `aerosplat up`)")
    return png, meta_json


# ── schemas ──────────────────────────────────────────────────────────────
class _In(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CoverageGrid(BaseModel):
    """Per-cell view count of one facade: where the orthophoto is evidence (§5.3).

    `views[j][i]` is cell (column i, row j) with **row 0 at the bottom** of the wall,
    matching the stage's own layout. Cell (i, j) covers u from `u_min + i*cell_m` and
    v from `v_min + j*cell_m`. A zero means no capture resolved that patch finely
    enough, so whatever the splat drew there is invention and must be shaded.
    """

    cell_m: float
    nu: int
    nv: int
    u_min: float
    v_min: float
    max_gsd_cm: float
    max_views: int
    views: list[list[int]]


class FacadeOut(BaseModel):
    facade: Facade
    building: Optional[Building] = None
    coverage_grid: Optional[CoverageGrid] = None
    run_id: str
    epoch_id: str
    features: int = 0                       # authored access points on this facade


class SemanticsRunOut(BaseModel):
    """A run with semantics, for the facade list's run picker."""

    id: str
    epoch_id: str
    finished_at: Any = None
    scene: bool                             # it has its own splat to render facades from


class SemanticsOut(BaseModel):
    """Everything the editor needs to pick a facade, minus the rasters."""

    run_id: str
    epoch_id: str
    finished_at: Any = None
    frame: str
    terrain: TerrainModel
    buildings: list[Building]
    facades: list[Facade]
    warnings: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    features_by_facade: dict[str, int] = Field(default_factory=dict)
    # The scene this run's facades render from (pinned to the run), None without a splat.
    scene: Optional[str] = None
    # Every run with semantics, newest first; facade ids differ between them.
    runs: list[SemanticsRunOut] = Field(default_factory=list)


class TaxonomyClassOut(BaseModel):
    """One class, with everything it inherits already resolved, for a picker."""

    id: str
    parent: Optional[str]
    kind: ApKind
    pack: str
    labels: dict[str, str]
    orientation: str
    size_prior_m: Optional[dict[str, tuple[float, float]]] = None
    attributes: list[Attribute] = Field(default_factory=list)
    handover: Optional[dict[str, Any]] = None
    citygml: Optional[str] = None


class TaxonomyOut(BaseModel):
    version: str
    packs: list[dict[str, str]]
    classes: list[TaxonomyClassOut]


class OrthoRequest(FacadeOrthoSpec):
    """`FacadeOrthoSpec` plus which run's facade to use (default: the newest)."""

    model_config = ConfigDict(extra="forbid")

    run_id: Optional[str] = None


class AccessPointCreate(_In):
    """What the editor sends when the operator finishes a shape."""

    facade_id: str
    class_id: str = Field(alias="class")
    polygon_uv: list[tuple[float, float]] = Field(min_length=3)
    attributes: dict[str, str] = Field(default_factory=dict)
    # Drawn by hand means verified (§3.4); a detector's proposal comes in as a candidate.
    lifecycle: Lifecycle = Lifecycle.verified
    notes: str = ""
    by: Optional[str] = None
    run_id: Optional[str] = None


class AccessPointPatch(_In):
    """A change to one feature. Omitted fields are left alone."""

    class_id: Optional[str] = Field(None, alias="class")
    polygon_uv: Optional[list[tuple[float, float]]] = Field(None, min_length=3)
    attributes: Optional[dict[str, str]] = None
    lifecycle: Optional[Lifecycle] = None
    notes: Optional[str] = None
    by: Optional[str] = None


class AddressRequest(_In):
    floor_offset: int = Field(0, ge=-5, le=5)
    run_id: Optional[str] = None                # default: the newest run with semantics
    row_tol_m: float = Field(0.6, gt=0.01, le=5.0)
    column_tol_m: float = Field(0.6, gt=0.01, le=5.0)
    by: Optional[str] = None


class RegularizeRequest(_In):
    """Snap the openings' edges to the facade's rows and columns (§5.6 item 3)."""

    run_id: Optional[str] = None
    tol_m: float = Field(SNAP_TOL_M, gt=0.0, le=0.5)
    row_tol_m: float = Field(0.6, gt=0.01, le=5.0)
    column_tol_m: float = Field(0.6, gt=0.01, le=5.0)
    # Compute and return what would move, store nothing.
    dry_run: bool = False
    by: Optional[str] = None


class EdgeMoveOut(BaseModel):
    id: str
    edge: str
    old: float
    new: float
    delta: float
    line: str


class EdgeKeptOut(BaseModel):
    id: str
    edge: str
    off_m: float
    line: str
    reason: str


class RegularizeOut(BaseModel):
    facade_id: str
    run_id: str
    tol_m: float
    dry_run: bool
    moves: list[EdgeMoveOut] = Field(default_factory=list)
    kept: list[EdgeKeptOut] = Field(default_factory=list)
    # Openings not considered at all, with the reason (e.g. not an axis-aligned rectangle).
    skipped: dict[str, str] = Field(default_factory=dict)
    # The changed features as stored (or, on a dry run, as they would be).
    features: list[AccessPoint] = Field(default_factory=list)


class RowOut(BaseModel):
    floor_index: int
    floor_label: str
    v_center: float
    v_bottom: float
    members: list[str]


class AddressOut(BaseModel):
    facade_id: str
    rows: list[RowOut]
    column_centres: list[float]
    floor_height_m: Optional[float]
    ambiguous: bool
    reasons: list[str] = Field(default_factory=list)
    features: list[AccessPoint] = Field(default_factory=list)


# What a grid cell is proposed as. The fit finds window-sized openings in a repeating
# grid; whether one is a door or a loggia is the reviewer's call (they can reclassify).
GRIDFIT_CLASS = "opening.window"
# The ortho gridfit renders for itself. Metric, so any scale works; this is the one it
# was measured at on dataset_1910 (≈ 13 cm/px captures), and it keeps a 95 m wall
# well under the render size limit.
GRIDFIT_GSD_CM = 8.0


class GridfitRequest(_In):
    gsd_cm: float = Field(GRIDFIT_GSD_CM, ge=2.0, le=20.0)
    run_id: Optional[str] = None
    # Also render depth, so a cell behind a tree, a pole or a van -- anything well in
    # front of the wall plane -- is dropped even when it is not green (§5.5).
    depth: bool = True
    # Fit and screen, but store nothing: what pressing the button would propose.
    dry_run: bool = False
    by: Optional[str] = None


class GridfitOut(BaseModel):
    facade_id: str
    model: str
    run_id: str
    gsd_cm: float
    dry_run: bool
    refused: bool
    result: GridfitResult
    proposed: int                           # cells the fit would propose
    # Stored as `candidate` (or, on a dry run, what would have been).
    candidates: list[AccessPoint] = Field(default_factory=list)
    # Proposals dropped because they land on an existing feature, by its lifecycle.
    skipped: dict[str, int] = Field(default_factory=dict)


class PairScoreOut(BaseModel):
    b_id: str
    angle_deg: float
    offset_m: float
    offset_tol_m: float
    overlap_m: float
    overlap_a: float
    overlap_b: float
    overlap_v: float
    span_a: tuple[float, float]
    score: float


class FacadeMatchOut(BaseModel):
    """Where one facade of `from_run` is in `to_run` (ACCESS_POINTS §5.9)."""

    a_id: str
    status: MatchStatus
    b_ids: list[str] = Field(default_factory=list)
    relation: Optional[Relation] = None
    confidence: float = 0.0
    reason: str = ""
    shared_with: list[str] = Field(default_factory=list)
    candidates: list[PairScoreOut] = Field(default_factory=list)


class FacadeMatchesOut(BaseModel):
    from_run: str
    to_run: str
    model: str
    counts: dict[str, int]
    matches: list[FacadeMatchOut]


class FacadeFitOut(BaseModel):
    facade_id: str
    angle_deg: float
    offset_m: float
    offset_tol_m: float
    inside: float
    polygon_uv: list[tuple[float, float]]


class PlacementOut(BaseModel):
    feature_id: str
    status: PlacementStatus
    facade_id: Optional[str] = None
    fit: Optional[FacadeFitOut] = None
    candidates: list[FacadeFitOut] = Field(default_factory=list)
    reason: str = ""


class CarriedOut(BaseModel):
    """A feature drawn in another run that lands on this facade: shown, never moved by itself."""

    feature: AccessPoint
    placement: PlacementOut
    # This facade's fit, even when the placement is ambiguous (so the editor can draw it).
    here: FacadeFitOut


class ReanchorRequest(_In):
    run_id: Optional[str] = None                # the run of `facade_id`; default: the newest
    facade_id: str
    by: Optional[str] = None


class ReanchorOut(BaseModel):
    feature: AccessPoint
    placement: PlacementOut
    from_facade: str


def _plain(obj: Any) -> dict[str, Any]:
    return dataclasses.asdict(obj)


# ── taxonomy ─────────────────────────────────────────────────────────────
def _class_out(tax: Taxonomy, cls: TaxonomyClass) -> TaxonomyClassOut:
    handover = tax.handover_of(cls.id)
    return TaxonomyClassOut(
        id=cls.id, parent=cls.parent, kind=cls.kind, pack=cls.pack,
        labels={"uk": cls.labels.uk, "en": cls.labels.en}, orientation=cls.orientation.value,
        size_prior_m=({"width": cls.size_prior_m.width, "height": cls.size_prior_m.height}
                      if cls.size_prior_m else None),
        attributes=tax.attributes_of(cls.id),
        handover=handover.model_dump(mode="json") if handover else None, citygml=cls.citygml)


@router.get("/sites/{project_id}/{site_id}/taxonomy", response_model=TaxonomyOut)
def site_taxonomy(project_id: str, site_id: str,
                  cat: Catalog = Depends(get_catalog)) -> TaxonomyOut:
    """The classes this Site's `taxonomy_packs` resolve to (ADR-0007)."""
    site = _site(cat, project_id, site_id)
    tax = _taxonomy(site)
    return TaxonomyOut(
        version=tax.version_string,
        packs=[{"pack": p.pack, "version": p.version} for p in tax.packs],
        classes=[_class_out(tax, tax.classes[cid]) for cid in sorted(tax.classes)])


# ── facades ──────────────────────────────────────────────────────────────
def _counts(store: AuthoredStore, facades: list[Facade]) -> dict[str, int]:
    """Non-rejected features native to each of these facades (drawn in exactly its frame).

    By frame, not by id: facade ids are per run, and `B04/SW1` of one run may be another
    building in the next (ACCESS_POINTS §5.9). Features of other runs on the same wall are
    `…/carried`, not counted here.
    """
    by_id = {f.id: f for f in facades}
    out: dict[str, int] = {}
    for f in store.load().features:
        fc = by_id.get(f.facade_id) if f.facade_id is not None else None
        if f.lifecycle is not Lifecycle.rejected and fc is not None and is_native(f, fc):
            out[fc.id] = out.get(fc.id, 0) + 1
    return out


@router.get("/sites/{project_id}/{site_id}/semantics", response_model=SemanticsOut)
def site_semantics(project_id: str, site_id: str, run_id: Optional[str] = None,
                   cat: Catalog = Depends(get_catalog)) -> SemanticsOut:
    """Terrain header, buildings and facades of the newest run that produced them (or `run_id`)."""
    site = _site(cat, project_id, site_id)
    product = _semantics(cat, site, run_id)
    r = product.result
    runs = []
    for run in semantics_runs(cat, site):
        try:
            run_scene_source(cat, site, run)
            has_scene = True
        except SceneLookupError:
            has_scene = False
        runs.append(SemanticsRunOut(id=run.id, epoch_id=run.epoch_id,
                                    finished_at=run.finished_at, scene=has_scene))
    return SemanticsOut(
        run_id=product.run.id, epoch_id=product.run.epoch_id,
        finished_at=product.run.finished_at, frame=r.frame, terrain=r.terrain,
        buildings=r.buildings, facades=r.facades, warnings=r.warnings, params=r.params,
        features_by_facade=_counts(_store(cat, site), r.facades),
        scene=site.key if _run_has_scene(cat, site, product) else None, runs=runs)


def _coverage_out(product: SemanticsProduct, facade: Facade) -> Optional[CoverageGrid]:
    grid = product.coverage_grid(facade.id)
    if grid is None:
        return None
    cov: Optional[Coverage] = facade.coverage
    nv, nu = grid.shape
    # The stage lays its cells out over the fitted extent, so the cell size follows
    # from the counts rather than from the nominal `cell_m`.
    cell_m = cov.cell_m if cov else facade.length / max(nu, 1)
    return CoverageGrid(
        cell_m=cell_m, nu=int(nu), nv=int(nv), u_min=facade.u_min, v_min=facade.v_ground,
        max_gsd_cm=cov.max_gsd_cm if cov else 0.0, max_views=int(grid.max()) if grid.size else 0,
        views=grid.astype(int).tolist())


@router.get("/sites/{project_id}/{site_id}/facades", response_model=list[Facade])
def list_facades(project_id: str, site_id: str, building: Optional[str] = None,
                 run_id: Optional[str] = None,
                 cat: Catalog = Depends(get_catalog)) -> list[Facade]:
    site = _site(cat, project_id, site_id)
    facades = _semantics(cat, site, run_id).result.facades
    return [f for f in facades if building is None or f.building_id == building]


# Declared before the bare `/facades/{facade_id:path}` below: a facade id contains a
# slash ("B01/N1"), so the path converter is greedy and the first matching route wins.
@router.post("/sites/{project_id}/{site_id}/facades/{facade_id:path}/ortho",
             responses={200: {"content": {"image/png": {}}, "description":
                              "PNG of the wall seen head-on; the camera and the (u, v) "
                              f"pixel grid are in `{RENDER_META_HEADER}` (JSON)"}})
async def facade_ortho_render(project_id: str, site_id: str, facade_id: str,
                              body: OrthoRequest = OrthoRequest(),
                              cat: Catalog = Depends(get_catalog)) -> Response:
    """Render one facade head-on, framed to its (u, v) extent at the requested cm/px.

    An orthographic render along the facade's −normal: openings come out rectangular
    and all at one scale, so a pixel maps to a point on the wall by a constant
    (`facade_grid` in the metadata — the facade's answer to `ground_grid`).

    Where the coverage map is empty the picture is the splat's invention rather than
    evidence (§5.3), which is why the editor shades those cells.
    """
    site = _site(cat, project_id, site_id)
    product = _semantics(cat, site, body.run_id)
    facade = _facade(product, facade_id)
    scene = _run_scene(cat, site, product)
    spec = FacadeOrthoSpec(**body.model_dump(exclude={"run_id"}))
    try:
        request, grid = facade_ortho(facade, spec)
    except FacadeOrthoError as e:
        raise HTTPException(422, str(e)) from None

    png, meta_json = await run_in_threadpool(_pinned_render, scene, request, product.run.id)
    try:
        meta = json.loads(meta_json)
    except ValueError:
        meta = {}
    meta["facade_grid"] = grid.model_dump(mode="json")
    meta["facade"] = {"id": facade.id, "building_id": facade.building_id,
                      "azimuth_deg": facade.azimuth_deg, "length_m": facade.length,
                      "height_m": facade.height, "v_ground": facade.v_ground,
                      "u_min": facade.u_min, "u_max": facade.u_max, "v_max": facade.v_max,
                      "plane_rms_m": facade.plane_rms_m}
    meta["gsd_cm"] = spec.gsd_cm
    meta["run"] = product.run.id
    return Response(content=png, media_type="image/png",
                    headers={RENDER_META_HEADER: json.dumps(meta, separators=(",", ":"))})


@router.post("/sites/{project_id}/{site_id}/facades/{facade_id:path}/address",
             response_model=AddressOut)
def facade_address(project_id: str, site_id: str, facade_id: str,
                   body: AddressRequest = AddressRequest(),
                   cat: Catalog = Depends(get_catalog)) -> AddressOut:
    """Recompute floors and columns for this facade's openings and store the addresses (§5.6).

    Clusters the v- and u-centres of the openings the operator drew; it never moves an
    opening and never invents one. A facade whose ground row was not drawn, or whose
    storey intervals are irregular, comes back `ambiguous`, which §6.3 treats as a
    reason to refuse the target rather than a detail.
    """
    site = _site(cat, project_id, site_id)
    product = _semantics(cat, site, body.run_id)
    facade = _facade(product, facade_id)
    store = _store(cat, site)
    # Native only: an opening with this facade id from another run is another wall.
    features = [f for f in native_features(store.features(facade_id=facade_id), facade)
                if f.kind == ApKind.opening.value and f.lifecycle is not Lifecycle.rejected]
    shapes = [(f.id, f.geometry) for f in features if f.geometry is not None]
    result = address_facade(
        [Opening(id=fid, u_center=g.u_center, v_center=g.v_center, height=g.height,
                 width=g.width) for fid, g in shapes],
        v_ground=facade.v_ground, row_tol_m=body.row_tol_m, column_tol_m=body.column_tol_m,
        floor_offset=body.floor_offset)

    changed: list[AccessPoint] = []
    for f in features:
        got = result.of.get(f.id)
        if got is None:
            continue
        floor, column, ordinal = got
        address = Address(
            floor_index=floor, floor_label=floor_label(floor, body.floor_offset), column=column,
            ordinal_on_floor=ordinal, floor_ambiguous=result.ambiguous,
            canonical=canonical_address(site.id, facade.building_id, facade.name, floor, column))
        if f.address != address:
            changed.append(f.model_copy(update={"address": address}))
    if changed:
        store.replace_many(changed, by=body.by, what=f"addressed on {facade_id}", op="address")

    rows = [RowOut(floor_index=r.index, floor_label=floor_label(r.index, body.floor_offset),
                   v_center=r.v_center, v_bottom=r.v_bottom, members=r.members)
            for r in result.rows]
    return AddressOut(facade_id=facade_id, rows=rows, column_centres=result.column_centres,
                      floor_height_m=result.floor_height_m, ambiguous=result.ambiguous,
                      reasons=result.reasons,
                      features=native_features(store.features(facade_id=facade_id), facade))


EDGE_UK = {"u0": "лівий край", "u1": "правий край", "v0": "низ", "v1": "верх"}


def _axis_box(polygon_uv: list[tuple[float, float]]) -> Optional[tuple[float, float, float, float]]:
    """(u0, u1, v0, v1) of an axis-aligned rectangle; None for any other shape."""
    if len(polygon_uv) != 4:
        return None
    us = [p[0] for p in polygon_uv]
    vs = [p[1] for p in polygon_uv]
    u0, u1, v0, v1 = min(us), max(us), min(vs), max(vs)
    eps = 1e-6
    if not all((abs(u - u0) < eps or abs(u - u1) < eps) and (abs(v - v0) < eps or abs(v - v1) < eps)
               for u, v in polygon_uv):
        return None
    if len({(round(u, 6), round(v, 6)) for u, v in polygon_uv}) != 4:
        return None
    return u0, u1, v0, v1


@router.post("/sites/{project_id}/{site_id}/facades/{facade_id:path}/regularize",
             response_model=RegularizeOut)
def facade_regularize(project_id: str, site_id: str, facade_id: str,
                      body: RegularizeRequest = RegularizeRequest(),
                      cat: Catalog = Depends(get_catalog)) -> RegularizeOut:
    """Snap opening edges to the facade's row and column lines, within `tol_m` (§5.6 item 3).

    Only when the operator asks. Lines come from the openings themselves (the clusters the
    addressing uses, per class); an edge moves only if it is within `tol_m` of its line, so
    nothing moves further than that, and every move is written to the feature's history and
    the audit log (`op: regularize`). Edges further off are reported in `kept` and left
    alone. Only this run's own openings (same frame), not rejected ones, and only
    axis-aligned rectangles. A confirmed target on a moved opening sees `target_changed`
    until it is confirmed again.
    """
    site = _site(cat, project_id, site_id)
    product = _semantics(cat, site, body.run_id)
    facade = _facade(product, facade_id)
    store = _store(cat, site)
    with store.lock:
        features = [f for f in native_features(store.features(facade_id=facade_id), facade)
                    if f.kind == ApKind.opening.value and f.lifecycle is not Lifecycle.rejected]
        boxes: list[SnapBox] = []
        skipped: dict[str, str] = {}
        for f in features:
            assert f.geometry is not None
            box = _axis_box(f.geometry.polygon_uv)
            if box is None:
                skipped[f.id] = "не прямокутник по осях (u, v): не вирівнюється"
                continue
            boxes.append(SnapBox(id=f.id, u0=box[0], u1=box[1], v0=box[2], v1=box[3], group=f.class_))
        result = snap_to_grid(boxes, tol_m=body.tol_m, row_tol_m=body.row_tol_m,
                              column_tol_m=body.column_tol_m)
        changed: list[AccessPoint] = []
        whats: list[str] = []
        for f in features:
            box = result.boxes.get(f.id)
            if box is None:
                continue
            u0, u1, v0, v1 = box
            try:
                geometry = geometry_from_uv(facade, [(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
            except ValueError as e:
                skipped[f.id] = str(e)
                continue
            changed.append(f.model_copy(update={"geometry": geometry}, deep=True))
            whats.append(f"прив'язка до сітки фасаду (τ {body.tol_m:g} м): " + ", ".join(
                f"{EDGE_UK[m.edge]} {m.delta:+.3f} м ({m.line})" for m in result.moves_of(f.id)))
        if changed and not body.dry_run:
            changed = store.replace_many(changed, by=body.by, op="regularize", whats=whats)
    moved_ids = {f.id for f in changed}
    return RegularizeOut(
        facade_id=facade.id, run_id=product.run.id, tol_m=body.tol_m, dry_run=body.dry_run,
        moves=[EdgeMoveOut(id=m.id, edge=m.edge, old=m.old, new=m.new, delta=m.delta, line=m.line)
               for m in result.moves if m.id in moved_ids],
        kept=[EdgeKeptOut(**dataclasses.asdict(k)) for k in result.kept],
        skipped=skipped, features=changed)


def _decode(png: bytes, what: str, shape: tuple[int, int]) -> np.ndarray:
    try:
        arr = np.asarray(Image.open(io.BytesIO(png)))
    except (OSError, ValueError):
        raise HTTPException(502, f"render service returned an unreadable {what} image") from None
    if arr.shape[:2] != shape:
        raise HTTPException(502, f"render service returned a {what} image of "
                                 f"{arr.shape[1]} × {arr.shape[0]} px, asked for "
                                 f"{shape[1]} × {shape[0]}")
    return arr


def _run_gridfit(cat: Catalog, site: Site, product: SemanticsProduct, facade: Facade,
                 grid: FacadeGrid, spec: FacadeOrthoSpec, shape: tuple[int, int],
                 layers: dict[str, tuple[bytes, str]], tax: Taxonomy,
                 body: GridfitRequest) -> GridfitOut:
    """The CPU half of the gridfit endpoint: decode, fit, screen, store."""
    rgb = _decode(layers["rgb"][0], "rgb", shape)[..., :3]
    alpha = _decode(layers["alpha"][0], "alpha", shape).astype(float)
    alpha = (alpha[..., 0] if alpha.ndim == 3 else alpha) / 255.0
    front = None
    if "depth" in layers:
        depth = _decode(layers["depth"][0], "depth", shape).astype(float)
        try:
            max_m = float(json.loads(layers["depth"][1])["depth"]["max_m"])
        except (ValueError, KeyError, TypeError):
            raise HTTPException(502, "render service sent depth without its scale") from None
        # Depth is the distance from the camera plane, which stands `standoff_m` in
        # front of the wall; 0 means nothing was hit.
        front = np.where(depth > 0, spec.standoff_m - depth / 65535.0 * max_m, np.nan)

    coverage = None
    cov = _coverage_out(product, facade)
    if cov is not None:
        coverage = CoverageRaster(views=np.asarray(cov.views, np.int32), u_min=cov.u_min,
                                  v_min=cov.v_min, cell_m=cov.cell_m)
    prior = tax.get(GRIDFIT_CLASS).size_prior_m
    params = (GridfitParams(opening_w_m=prior.width, opening_h_m=prior.height) if prior
              else GridfitParams())
    result = gridfit(rgb, alpha, grid, coverage=coverage, front_m=front,
                     bounds_uv=(facade.u_min, facade.u_max, facade.v_ground, facade.v_max),
                     params=params)

    proposals = [Proposal(rect=(c.u0, c.u1, c.v0, c.v1), confidence=c.confidence,
                          notes=result.note(c)) for c in result.proposals]
    store = _store(cat, site)
    # Screen and store under the authored lock, so two presses of the button cannot
    # both decide that a cell is still free.
    with store.lock:
        keep, skipped = screen(proposals, facade, store.features())
        candidates = make_candidates(
            keep, site_key=site.key, facade=facade, class_id=GRIDFIT_CLASS,
            taxonomy=tax.version_string, model=result.model, created_by="pipeline:gridfit",
            run_id=product.run.id, epoch_id=product.run.epoch_id,
            evidence=[f"ortho:{facade.id}@{spec.gsd_cm:g}cm"])
        if candidates and not body.dry_run:
            store.create_many(candidates, by=body.by, what=f"{result.model} запропонував")
    return GridfitOut(facade_id=facade.id, model=result.model, run_id=product.run.id,
                      gsd_cm=spec.gsd_cm, dry_run=body.dry_run, refused=result.refused,
                      result=result, proposed=len(proposals), candidates=candidates,
                      skipped=skipped)


@router.post("/sites/{project_id}/{site_id}/facades/{facade_id:path}/candidates/gridfit",
             response_model=GridfitOut)
async def facade_gridfit(project_id: str, site_id: str, facade_id: str,
                         body: GridfitRequest = GridfitRequest(),
                         cat: Catalog = Depends(get_catalog)) -> GridfitOut:
    """Propose window candidates from the facade's periodic grid (ROADMAP 7.3, step 1).

    Renders the wall head-on (rgb, alpha and, by default, depth), fits the storey and
    bay grid (`aerosplat.semantics.gridfit`) and stores every cell that looks like an
    opening as a `candidate` with `origin: pipeline` -- never `verified` (ADR-0009).
    Cells that land on a feature already there are skipped, a rejected one included,
    so a wrong candidate is not proposed twice (§3.4) and a second press adds nothing.

    A facade the grid does not describe comes back `refused` with the reasons, and
    nothing is stored: that is an answer, not an error.
    """
    site = _site(cat, project_id, site_id)
    product = _semantics(cat, site, body.run_id)
    facade = _facade(product, facade_id)
    scene = _run_scene(cat, site, product)
    tax = _taxonomy(site)
    _check_class(tax, GRIDFIT_CLASS, {})
    spec = FacadeOrthoSpec(gsd_cm=body.gsd_cm)
    try:
        request, grid = facade_ortho(facade, spec)
    except FacadeOrthoError as e:
        raise HTTPException(422, str(e)) from None

    layers: dict[str, tuple[bytes, str]] = {}
    for output in ("rgb", "alpha", "depth") if body.depth else ("rgb", "alpha"):
        one = request.model_copy(update={"output": output})
        layers[output] = await run_in_threadpool(_pinned_render, scene, one, product.run.id)
    return await run_in_threadpool(_run_gridfit, cat, site, product, facade, grid, spec,
                                   (request.height, request.width), layers, tax, body)


@router.get("/sites/{project_id}/{site_id}/facade-matches", response_model=FacadeMatchesOut)
def facade_matches(project_id: str, site_id: str, from_run: str, to_run: Optional[str] = None,
                   facade: Optional[str] = None, status: Optional[list[MatchStatus]] = Query(None),
                   cat: Catalog = Depends(get_catalog)) -> FacadeMatchesOut:
    """Which facade of `to_run` (default: the newest) each facade of `from_run` is (§5.9).

    Geometry only, in Site ENU: normals, plane offset within the planes' scatter, shared
    stretch and height. `ambiguous` and `none` are answers, not errors: nothing is guessed.
    Read-only; carrying a feature over is `POST …/access-points/{id}/reanchor`.
    """
    site = _site(cat, project_id, site_id)
    a = _semantics(cat, site, from_run)
    b = _semantics(cat, site, to_run)
    facades = a.result.facades if facade is None else [_facade(a, facade)]
    matches = match_facades(facades, b.result.facades)
    counts: dict[str, int] = {}
    for m in matches:
        counts[m.status] = counts.get(m.status, 0) + 1
    if status:
        matches = [m for m in matches if m.status in status]
    return FacadeMatchesOut(from_run=a.run.id, to_run=b.run.id, model=f"facade-match@{MATCH_VERSION}",
                            counts=counts,
                            matches=[FacadeMatchOut.model_validate(_plain(m)) for m in matches])


@router.get("/sites/{project_id}/{site_id}/facades/{facade_id:path}/carried",
            response_model=list[CarriedOut])
def facade_carried(project_id: str, site_id: str, facade_id: str, run_id: Optional[str] = None,
                   cat: Catalog = Depends(get_catalog)) -> list[CarriedOut]:
    """Features drawn in other runs that land on this facade of this run (§5.9).

    For the editor to show (dashed) and offer to carry over. `placement.status` says
    whether a feature can be carried here: `placed` on this facade, or `ambiguous` between
    this one and another (then it cannot, and the reason says between which).
    """
    site = _site(cat, project_id, site_id)
    product = _semantics(cat, site, run_id)
    facade = _facade(product, facade_id)
    features = _store(cat, site).features()
    out: list[CarriedOut] = []
    for p in foreign_on(facade, product.result.facades, features):
        f = next(x for x in features if x.id == p.feature_id)
        here = next(c for c in p.candidates if c.facade_id == facade.id) \
            if p.candidates else None
        if here is None:                         # placed elsewhere: not this facade's business
            continue
        out.append(CarriedOut(feature=f, placement=PlacementOut.model_validate(_plain(p)),
                              here=FacadeFitOut.model_validate(_plain(here))))
    return out


@router.get("/sites/{project_id}/{site_id}/facades/{facade_id:path}", response_model=FacadeOut)
def facade_detail(project_id: str, site_id: str, facade_id: str, run_id: Optional[str] = None,
                  cat: Catalog = Depends(get_catalog)) -> FacadeOut:
    """One facade with its coverage map, the building it belongs to, and how many features it has."""
    site = _site(cat, project_id, site_id)
    product = _semantics(cat, site, run_id)
    facade = _facade(product, facade_id)
    building = next((b for b in product.result.buildings if b.id == facade.building_id), None)
    return FacadeOut(facade=facade, building=building,
                     coverage_grid=_coverage_out(product, facade), run_id=product.run.id,
                     epoch_id=product.run.epoch_id,
                     features=_counts(_store(cat, site), [facade]).get(facade_id, 0))


# ── authored access points ───────────────────────────────────────────────
@router.get("/sites/{project_id}/{site_id}/access-points", response_model=list[AccessPoint])
def list_access_points(project_id: str, site_id: str, facade: Optional[str] = None,
                       lifecycle: Optional[list[Lifecycle]] = Query(None),
                       run_id: Optional[str] = None,
                       cat: Catalog = Depends(get_catalog)) -> list[AccessPoint]:
    """Authored features, optionally one facade's.

    With `facade` alone, every feature whose host carries that id, whatever run it was drawn
    in. With `facade` and `run_id`, only the ones native to that run's facade (same frame):
    facade ids are per run, so the id alone can name a different wall in another run.
    """
    site = _site(cat, project_id, site_id)
    features = _store(cat, site).features(facade_id=facade, lifecycle=lifecycle)
    if facade is not None and run_id is not None:
        fc = _facade(_semantics(cat, site, run_id), facade)
        features = native_features(features, fc)
    return features


def _check_class(tax: Taxonomy, class_id: str, attributes: dict[str, str]) -> None:
    try:
        tax.get(class_id)
        tax.validate_attributes(class_id, attributes)
    except TaxonomyError as e:
        raise HTTPException(422, str(e)) from None


@router.post("/sites/{project_id}/{site_id}/access-points", response_model=AccessPoint,
             status_code=201)
def create_access_point(project_id: str, site_id: str, body: AccessPointCreate,
                        cat: Catalog = Depends(get_catalog)) -> AccessPoint:
    """Store a shape the operator drew on a facade orthophoto."""
    site = _site(cat, project_id, site_id)
    tax = _taxonomy(site)
    _check_class(tax, body.class_id, body.attributes)
    product = _semantics(cat, site, body.run_id)
    facade = _facade(product, body.facade_id)
    try:
        feature = make_access_point(
            site_key=site.key, facade=facade, class_id=body.class_id,
            taxonomy=tax.version_string, polygon_uv=body.polygon_uv,
            attributes=body.attributes, lifecycle=body.lifecycle, origin=Origin.manual,
            created_by=actor(body.by), run_id=product.run.id, epoch_id=product.run.epoch_id,
            evidence=[f"ortho:{facade.id}"], notes=body.notes)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    try:
        return _store(cat, site).create(feature, by=body.by)
    except AuthoredError as e:
        raise HTTPException(409, str(e)) from None


@router.patch("/sites/{project_id}/{site_id}/access-points/{feature_id}",
              response_model=AccessPoint)
def update_access_point(project_id: str, site_id: str, feature_id: str, body: AccessPointPatch,
                        cat: Catalog = Depends(get_catalog)) -> AccessPoint:
    """Change a feature's class, shape, attributes or lifecycle. Nothing changes on its own."""
    site = _site(cat, project_id, site_id)
    store = _store(cat, site)
    try:
        old = store.get(feature_id)
    except ApNotFound as e:
        raise HTTPException(404, str(e)) from None

    tax = _taxonomy(site)
    class_id = body.class_id or old.class_
    attributes = old.attributes if body.attributes is None else body.attributes
    _check_class(tax, class_id, attributes)

    update: dict[str, Any] = {"class_": class_id, "kind": class_id.split(".", 1)[0],
                              "attributes": attributes, "taxonomy": tax.version_string}
    if body.lifecycle is not None:
        update["lifecycle"] = body.lifecycle
    if body.notes is not None:
        update["notes"] = body.notes
    if body.polygon_uv is not None:
        # Re-derive the ENU block from the facade the feature was drawn on, not from
        # whatever the newest run calls that facade now.
        facade = _host_facade(cat, site, old)
        try:
            update["geometry"] = geometry_from_uv(facade, body.polygon_uv)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None

    touched = [name for name, value in (("class", body.class_id),
                                        ("polygon_uv", body.polygon_uv),
                                        ("attributes", body.attributes),
                                        ("lifecycle", body.lifecycle),
                                        ("notes", body.notes)) if value is not None]
    # deep=True so appending to the new record's edit history cannot reach into the old one.
    new = old.model_copy(update=update, deep=True)
    try:
        return store.replace(new, by=body.by, what=", ".join(touched) or "touched")
    except (AuthoredError, ValueError) as e:
        raise HTTPException(422, str(e)) from None


@router.post("/sites/{project_id}/{site_id}/access-points/{feature_id}/reanchor",
             response_model=ReanchorOut)
def reanchor_access_point(project_id: str, site_id: str, feature_id: str, body: ReanchorRequest,
                          cat: Catalog = Depends(get_catalog)) -> ReanchorOut:
    """Carry a feature of another run onto a facade of this run — only when a person asks.

    Allowed only when the feature lands on exactly that facade (`placed`, §5.9); an
    `ambiguous` or `none` placement is a 409 with the reason. The shape keeps its height and
    is projected onto the new plane (it moves by the plane offset, reported); lifecycle and
    history stay, the address is dropped (it named the old facade), and the change goes to
    the audit log with the facade it came from. A target confirmed on this feature sees
    `target_changed` at planning until it is confirmed again.
    """
    site = _site(cat, project_id, site_id)
    product = _semantics(cat, site, body.run_id)
    facade = _facade(product, body.facade_id)
    store = _store(cat, site)
    with store.lock:
        try:
            old = store.get(feature_id)
        except ApNotFound as e:
            raise HTTPException(404, str(e)) from None
        if is_native(old, facade):
            raise HTTPException(409, f"{feature_id} is already on {facade.id} of run {product.run.id}")
        try:
            new, placement = reanchor(old, facade, product.result.facades)
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        assert old.host is not None and placement.fit is not None
        what = (f"перенесено з {old.host.facade_id} на {facade.id} (run {product.run.id}): "
                f"зсув площини {placement.fit.offset_m:.2f} м, кут {placement.fit.angle_deg:.1f}°"
                + ("; адресу скинуто" if old.address is not None else ""))
        stored = store.replace(new, by=body.by, what=what, op="reanchor")
    return ReanchorOut(feature=stored, placement=PlacementOut.model_validate(_plain(placement)),
                       from_facade=old.host.facade_id)


def _host_facade(cat: Catalog, site: Site, feature: AccessPoint) -> Facade:
    """The facade frame a feature was drawn in, rebuilt from the feature itself.

    Not looked up in the current run on purpose: the frame on the record is the one the
    polygon means, and a recomputed run may have moved or renamed the facade (§5.9).
    """
    h, g = feature.host, feature.geometry
    if h is None or g is None:
        raise HTTPException(422, f"{feature.id} is a pad, not a shape on a facade: "
                                 f"it has no polygon_uv to redraw")
    return Facade(id=h.facade_id, building_id=h.building_id, origin=h.origin, normal=h.normal,
                  u_axis=h.u_axis, azimuth_deg=h.azimuth_deg,
                  u_min=min(p[0] for p in g.polygon_uv),
                  u_max=max(p[0] for p in g.polygon_uv),
                  v_ground=h.v_ground,
                  v_max=max(p[1] for p in g.polygon_uv),
                  plane_rms_m=0.0, n_points=0)


@router.delete("/sites/{project_id}/{site_id}/access-points/{feature_id}",
               response_model=AccessPoint)
def delete_access_point(project_id: str, site_id: str, feature_id: str,
                        by: Optional[str] = None,
                        cat: Catalog = Depends(get_catalog)) -> AccessPoint:
    """Remove a feature. To mark a wrong *candidate*, set `lifecycle: rejected` instead:
    a rejected feature is kept as a hard negative for training (§3.4)."""
    site = _site(cat, project_id, site_id)
    try:
        return _store(cat, site).delete(feature_id, by=by)
    except ApNotFound as e:
        raise HTTPException(404, str(e)) from None


@router.get("/sites/{project_id}/{site_id}/audit", response_model=list[AuditEntry])
def site_audit(project_id: str, site_id: str, limit: int = Query(100, ge=1, le=MAX_AUDIT),
               cat: Catalog = Depends(get_catalog)) -> list[AuditEntry]:
    """The tail of `authored/audit.jsonl`: who changed what, oldest first (§3.5)."""
    site = _site(cat, project_id, site_id)
    return _store(cat, site).audit(limit=limit)
