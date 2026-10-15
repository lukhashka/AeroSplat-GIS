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

import json
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from aerosplat.authored.geometry import geometry_from_uv, make_access_point
from aerosplat.authored.models import AccessPoint, Address, AuditEntry, Lifecycle, Origin, actor
from aerosplat.authored.store import AuthoredError, AuthoredStore, NotFound as ApNotFound
from aerosplat.semantics.addressing import Opening, address_facade, canonical_address, floor_label
from aerosplat.semantics.models import Building, Coverage, Facade, TerrainModel
from aerosplat.semantics.ortho import FacadeOrthoError, FacadeOrthoSpec, facade_ortho
from aerosplat.semantics.products import SemanticsMissing, SemanticsProduct, require_semantics
from aerosplat.server import api_v2
from aerosplat.store.catalog import Catalog
from aerosplat.store.models import Site
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
    scene: Optional[str] = None             # None when there is nothing to render from


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
    row_tol_m: float = Field(0.6, gt=0.01, le=5.0)
    column_tol_m: float = Field(0.6, gt=0.01, le=5.0)
    by: Optional[str] = None


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
def _counts(store: AuthoredStore) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in store.load().features:
        if f.lifecycle is not Lifecycle.rejected:
            out[f.host.facade_id] = out.get(f.host.facade_id, 0) + 1
    return out


@router.get("/sites/{project_id}/{site_id}/semantics", response_model=SemanticsOut)
def site_semantics(project_id: str, site_id: str, run_id: Optional[str] = None,
                   cat: Catalog = Depends(get_catalog)) -> SemanticsOut:
    """Terrain header, buildings and facades of the newest run that produced them."""
    site = _site(cat, project_id, site_id)
    product = _semantics(cat, site, run_id)
    r = product.result
    return SemanticsOut(
        run_id=product.run.id, epoch_id=product.run.epoch_id,
        finished_at=product.run.finished_at, frame=r.frame, terrain=r.terrain,
        buildings=r.buildings, facades=r.facades, warnings=r.warnings, params=r.params,
        features_by_facade=_counts(_store(cat, site)),
        scene=api_v2._scene_name(cat, site))


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
    scene = api_v2._scene_name(cat, site)
    if scene is None:
        raise HTTPException(404, f"site {site.key} has no trained scene yet")
    spec = FacadeOrthoSpec(**body.model_dump(exclude={"run_id"}))
    try:
        request, grid = facade_ortho(facade, spec)
    except FacadeOrthoError as e:
        raise HTTPException(422, str(e)) from None

    png, meta_json = await run_in_threadpool(
        api_v2._render_call, scene, request.model_dump_json(exclude_none=True))
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
    product = _semantics(cat, site)
    facade = _facade(product, facade_id)
    store = _store(cat, site)
    features = [f for f in store.features(facade_id=facade_id)
                if f.kind == ApKind.opening.value and f.lifecycle is not Lifecycle.rejected]
    result = address_facade(
        [Opening(id=f.id, u_center=f.geometry.u_center, v_center=f.geometry.v_center,
                 height=f.geometry.height, width=f.geometry.width) for f in features],
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
                      reasons=result.reasons, features=store.features(facade_id=facade_id))


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
                     features=_counts(_store(cat, site)).get(facade_id, 0))


# ── authored access points ───────────────────────────────────────────────
@router.get("/sites/{project_id}/{site_id}/access-points", response_model=list[AccessPoint])
def list_access_points(project_id: str, site_id: str, facade: Optional[str] = None,
                       lifecycle: Optional[list[Lifecycle]] = Query(None),
                       cat: Catalog = Depends(get_catalog)) -> list[AccessPoint]:
    """Authored features, optionally one facade's. Keyed to the facade, not to a run."""
    site = _site(cat, project_id, site_id)
    return _store(cat, site).features(facade_id=facade, lifecycle=lifecycle)


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


def _host_facade(cat: Catalog, site: Site, feature: AccessPoint) -> Facade:
    """The facade frame a feature was drawn in, rebuilt from the feature itself.

    Not looked up in the current run on purpose: the frame on the record is the one the
    polygon means, and a recomputed run may have moved or renamed the facade (§5.9).
    """
    h = feature.host
    return Facade(id=h.facade_id, building_id=h.building_id, origin=h.origin, normal=h.normal,
                  u_axis=h.u_axis, azimuth_deg=h.azimuth_deg,
                  u_min=min(p[0] for p in feature.geometry.polygon_uv),
                  u_max=max(p[0] for p in feature.geometry.polygon_uv),
                  v_ground=h.v_ground,
                  v_max=max(p[1] for p in feature.geometry.polygon_uv),
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
