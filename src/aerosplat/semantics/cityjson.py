"""CityJSON 2.0 export of one run's buildings with the Site's access points (ACCESS_POINTS §2.6).

What goes out:

* every building of the run as `Building` with an **LoD1** `Solid`: its footprint extruded
  from `ground_z` to `roof_z` (semantics: GroundSurface, RoofSurface, WallSurface);
* for buildings that hold exported access points, an **LoD3** `MultiSurface` with the
  facades those openings are on (`WallSurface`, the openings cut out as holes) and the
  openings themselves as `Window` / `Door` surfaces (parent ↔ children linked), plus
  roof pads as `RoofSurface`; every such surface carries `aerosplat_id`, `aerosplat_class`,
  the lifecycle and, where known, the canonical address;
* access points with no CityGML equivalent (constraints, ground pads, or openings whose
  wall this run does not have unambiguously) as `GenericCityObject` with `aerosplat_class`
  — the CityJSON 2.0 counterpart of CityGML 3.0 `GenericOccupiedSpace`.

Coordinates are in the Site's projected CRS (WGS84 / UTM of the Site origin,
`aerosplat.geo.crs.SiteProjection`), **heights WGS84 ellipsoidal** (no geoid in the system),
stored as integers with a millimetre `transform`. Openings drawn in another run are put on
this run's walls only when they land on exactly one (`aerosplat.authored.carry`); the rest
are exported where they were drawn, as generic objects, and counted in the report.

No LoD2: the stage fits no roof planes, so there is nothing cheaper than a guess to give.

`validate` checks what can be checked offline without cjio/cjval: the schema's structure,
vertex indices, ring rules (≥ 3 distinct vertices, not closed, no repeats, simple, planar),
holes inside and opposite, semantic surface types, parent/children links, solid shells
closed and consistently oriented (every edge used once each way), no unused or duplicate
vertices, and the geographical extent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from aerosplat.authored.carry import corners_enu, is_native, place_feature
from aerosplat.authored.models import AccessPoint, Lifecycle
from aerosplat.geo.crs import GeoPoint, SiteProjection
from aerosplat.semantics.models import Building, Facade, SemanticsResult
from aerosplat.semantics.pads import point_in_polygon

VERSION = "2.0"
SCALE_M = 0.001
MODEL = "cityjson-export@1.0.0"

BUILDING_SURFACES = {"RoofSurface", "GroundSurface", "WallSurface", "ClosureSurface",
                     "OuterCeilingSurface", "OuterFloorSurface", "Window", "Door",
                     "InteriorWallSurface", "CeilingSurface", "FloorSurface"}
OPENING_SURFACES = {"Window", "Door"}
CITY_OBJECT_TYPES = {
    "Bridge", "BridgePart", "BridgeInstallation", "BridgeConstructiveElement", "BridgeRoom",
    "BridgeFurniture", "Building", "BuildingPart", "BuildingInstallation",
    "BuildingConstructiveElement", "BuildingFurniture", "BuildingStorey", "BuildingRoom",
    "BuildingUnit", "CityFurniture", "CityObjectGroup", "GenericCityObject", "LandUse",
    "OtherConstruction", "PlantCover", "SolitaryVegetationObject", "TINRelief",
    "TransportSquare", "Railway", "Road", "Tunnel", "TunnelPart", "TunnelInstallation",
    "TunnelConstructiveElement", "TunnelHollowSpace", "TunnelFurniture", "WaterBody",
    "Waterway"}
GEOMETRY_DEPTH = {"MultiPoint": 1, "MultiLineString": 2, "MultiSurface": 3,
                  "CompositeSurface": 3, "Solid": 4, "MultiSolid": 5, "CompositeSolid": 5}
PLANARITY_M = 0.01


@dataclass
class ExportReport:
    buildings: int = 0
    solids: int = 0
    lod3: int = 0
    openings: int = 0                     # on a wall of this run (as Window / Door)
    openings_carried: int = 0             # ... of which drawn in another run
    roof_pads: int = 0
    generic: int = 0                      # GenericCityObject access points
    skipped: list[str] = field(default_factory=list)
    repaired: list[str] = field(default_factory=list)   # exported, but changed to be valid

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class _Vertices:
    """ENU points in, indices out; projected and deduplicated at the end."""

    def __init__(self) -> None:
        self.enu: list[tuple[float, float, float]] = []

    def add(self, p) -> int:
        self.enu.append((float(p[0]), float(p[1]), float(p[2])))
        return len(self.enu) - 1

    def ring(self, points) -> list[int]:
        return [self.add(p) for p in points]


def _shoelace(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _segments_cross(p1, p2, p3, p4) -> bool:
    """Proper or touching intersection of segments p1p2 and p3p4 (2-D)."""
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)

    def on(a, b, c):
        return min(a[0], b[0]) - 1e-12 <= c[0] <= max(a[0], b[0]) + 1e-12 and \
            min(a[1], b[1]) - 1e-12 <= c[1] <= max(a[1], b[1]) + 1e-12
    o1, o2, o3, o4 = orient(p1, p2, p3), orient(p1, p2, p4), orient(p3, p4, p1), orient(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    return (o1 == 0 and on(p1, p2, p3)) or (o2 == 0 and on(p1, p2, p4)) or \
        (o3 == 0 and on(p3, p4, p1)) or (o4 == 0 and on(p3, p4, p2))


def is_simple(xy: np.ndarray) -> bool:
    """True when a closed 2-D ring does not touch or cross itself."""
    n = len(xy)
    for i in range(n):
        a, b = xy[i], xy[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or j == (i + 1) % n:
                continue
            if _segments_cross(a, b, xy[j], xy[(j + 1) % n]):
                return False
    return True


def split_pinches(p: np.ndarray) -> list[np.ndarray]:
    """Split a ring at every vertex it visits twice into the simple loops it is made of.

    A footprint traced along raster cell edges touches itself where two cells meet only at
    a corner; such a ring is not a valid polygon. The loops are returned as they are.
    """
    pts = [tuple(x) for x in p]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if pts[i] == pts[j]:
                a = np.asarray(pts[i:j], float)
                b = np.asarray(pts[j:] + pts[:i], float)
                return [loop for part in (a, b) if len(part) >= 3 for loop in split_pinches(part)]
    return [p]


# ── export ───────────────────────────────────────────────────────────────
def _footprint(b: Building) -> tuple[Optional[np.ndarray], str]:
    """A valid counter-clockwise footprint ring, and what had to be done to get it.

    Pinched outlines keep their largest loop (the rest are pieces joined to it only at a
    corner, reported). A ring that still crosses itself (simplification can do that) is
    refused rather than exported broken.
    """
    p = np.asarray(b.footprint, float)
    if len(p) >= 2 and np.allclose(p[0], p[-1]):
        p = p[:-1]
    note = ""
    loops = split_pinches(p)
    if len(loops) > 1:
        loops.sort(key=lambda q: -abs(_shoelace(q)))
        lost = sum(abs(_shoelace(q)) for q in loops[1:])
        p = loops[0]
        note = f"контур торкався сам себе: взято найбільшу частину, відкинуто {lost:.0f} м²"
    if len(p) < 3 or abs(_shoelace(p)) < 1e-6:
        return None, "контур вироджений"
    if _shoelace(p) < 0:
        p = p[::-1]
    if not is_simple(p):
        return None, "контур перетинає сам себе (після спрощення)"
    return p, note


def _solid(b: Building, fp: np.ndarray, verts: _Vertices) -> dict[str, Any]:
    z0, z1 = b.ground_z, b.roof_z
    n = len(fp)
    bottom = verts.ring([(x, y, z0) for x, y in fp])
    top = verts.ring([(x, y, z1) for x, y in fp])
    shell = [[bottom[::-1]], [top]]                       # ground faces down, roof up
    for i in range(n):
        j = (i + 1) % n
        shell.append([[bottom[i], bottom[j], top[j], top[i]]])
    return {"type": "Solid", "lod": "1", "boundaries": [shell],
            "semantics": {"surfaces": [{"type": "GroundSurface"}, {"type": "RoofSurface"},
                                       {"type": "WallSurface"}],
                          "values": [[0, 1] + [2] * n]}}


def _axis_rect(poly: Sequence[tuple[float, float]]) -> Optional[tuple[float, float, float, float]]:
    us = [p[0] for p in poly]
    vs = [p[1] for p in poly]
    u0, u1, v0, v1 = min(us), max(us), min(vs), max(vs)
    if len(poly) != 4 or not all((abs(u - u0) < 1e-6 or abs(u - u1) < 1e-6)
                                 and (abs(v - v0) < 1e-6 or abs(v - v1) < 1e-6) for u, v in poly):
        return None
    return u0, u1, v0, v1


def _ap_attrs(f: AccessPoint) -> dict[str, Any]:
    a: dict[str, Any] = {"aerosplat_id": f.id, "aerosplat_class": f.class_,
                         "aerosplat_lifecycle": f.lifecycle.value, "aerosplat_taxonomy": f.taxonomy}
    if f.confidence is not None:
        a["aerosplat_confidence"] = round(f.confidence, 3)
    if f.address is not None:
        if f.address.canonical:
            a["aerosplat_address"] = f.address.canonical
        if f.address.floor_label is not None:
            a["aerosplat_floor_label"] = f.address.floor_label
        if f.address.floor_ambiguous:
            a["aerosplat_floor_ambiguous"] = True
    for k, v in sorted(f.attributes.items()):
        a[f"aerosplat_attr_{k}"] = v
    return a


def _ccw_uv(poly: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """A (u, v) ring counter-clockwise: in a facade frame that is the outward normal."""
    p = np.asarray(poly, float)
    return [tuple(x) for x in (p if _shoelace(p) > 0 else p[::-1])]


def export_cityjson(site_key: str, origin: GeoPoint, run_id: str, epoch_id: str,
                    result: SemanticsResult, features: Iterable[AccessPoint], *,
                    lifecycles: Sequence[Lifecycle] = (Lifecycle.verified,),
                    citygml_of: Optional[dict[str, Optional[str]]] = None,
                    title: Optional[str] = None,
                    reference_date: Optional[datetime] = None) -> tuple[dict[str, Any], ExportReport]:
    """The CityJSON document and what went into it.

    `citygml_of` maps a class id to its taxonomy `citygml` name (Window, Door, RoofSurface,
    …); a class mapped to nothing is exported as a generic object.
    """
    proj = SiteProjection(origin)
    citygml_of = citygml_of or {}
    keep = set(lifecycles)
    feats = [f for f in features if f.lifecycle in keep]
    report = ExportReport()
    verts = _Vertices()
    objects: dict[str, dict[str, Any]] = {}
    facades = {f.id: f for f in result.facades}

    # Openings: on which wall of this run, if exactly one.
    on_wall: dict[str, list[tuple[AccessPoint, Facade, list[tuple[float, float]]]]] = {}
    generic: list[tuple[AccessPoint, str]] = []
    roof_pads: dict[str, list[AccessPoint]] = {}
    for f in feats:
        if f.pad is not None:
            owner = None
            if f.pad.building_id is not None:
                cx, cy = f.pad.center[0], f.pad.center[1]
                owner = next((b.id for b in result.buildings
                              if point_in_polygon(np.array([cx]), np.array([cy]), b.footprint)[0]), None)
            if owner is not None and citygml_of.get(f.class_) == "RoofSurface":
                roof_pads.setdefault(owner, []).append(f)
            else:
                generic.append((f, "ground" if f.pad.building_id is None else "roof_unmatched"))
            continue
        assert f.host is not None and f.geometry is not None
        kind = citygml_of.get(f.class_)
        if kind not in OPENING_SURFACES:
            generic.append((f, "no_citygml_equivalent"))
            continue
        native = facades.get(f.host.facade_id)
        if native is not None and is_native(f, native):
            on_wall.setdefault(native.id, []).append((f, native, list(f.geometry.polygon_uv)))
            continue
        placed = place_feature(f, result.facades)
        if placed.status == "placed" and placed.fit is not None:
            fc = facades[placed.fit.facade_id]
            on_wall.setdefault(fc.id, []).append((f, fc, placed.fit.polygon_uv))
            report.openings_carried += 1
        else:
            generic.append((f, f"host_{placed.status}"))

    walls_of: dict[str, list[str]] = {}
    for fid in on_wall:
        walls_of.setdefault(facades[fid].building_id, []).append(fid)

    for b in result.buildings:
        fp, fp_note = _footprint(b)
        geoms: list[dict[str, Any]] = []
        if fp is None:
            report.skipped.append(f"{b.id}: {fp_note} — без LoD1")
        elif b.roof_z - b.ground_z < 0.5:
            report.skipped.append(f"{b.id}: висота {b.roof_z - b.ground_z:.2f} м — без LoD1")
        else:
            geoms.append(_solid(b, fp, verts))
            report.solids += 1
            if fp_note:
                report.repaired.append(f"{b.id}: {fp_note}")
        lod3 = _lod3(b, [facades[x] for x in sorted(walls_of.get(b.id, []))], on_wall,
                     roof_pads.get(b.id, []), citygml_of, verts, report)
        if lod3 is not None:
            geoms.append(lod3)
            report.lod3 += 1
        objects[b.id] = {"type": "Building", "geometry": geoms, "attributes": {
            "measuredHeight": round(b.height_m, 3), "aerosplat_building": b.id,
            "aerosplat_run": run_id, "aerosplat_epoch": epoch_id,
            "aerosplat_area_m2": round(b.area_m2, 1),
            "aerosplat_facades": list(b.facade_ids)}}
        if fp_note:
            objects[b.id]["attributes"]["aerosplat_footprint_note"] = fp_note
        report.buildings += 1

    for f, why in generic:
        g = _generic_geometry(f, verts)
        if g is None:
            report.skipped.append(f"{f.id}: немає геометрії для експорту")
            continue
        attrs = _ap_attrs(f)
        attrs["aerosplat_export_reason"] = why
        if f.host is not None:
            attrs["aerosplat_drawn_on"] = f.host.facade_id
        objects[f.id] = {"type": "GenericCityObject", "geometry": [g], "attributes": attrs}
        report.generic += 1

    doc = _assemble(objects, verts, proj, title or f"AeroSplat-GIS {site_key}, run {run_id} "
                    f"(EPSG:{proj.epsg}; heights: WGS84 ellipsoidal, m)",
                    f"{site_key}/{run_id}", reference_date)
    return doc, report


def _lod3(b: Building, walls: list[Facade], on_wall, pads: list[AccessPoint],
          citygml_of: dict[str, Optional[str]], verts: _Vertices,
          report: ExportReport) -> Optional[dict[str, Any]]:
    if not walls and not pads:
        return None
    boundaries: list[Any] = []
    surfaces: list[dict[str, Any]] = []
    values: list[int] = []
    for fc in walls:
        wall_rect = (fc.u_min, fc.u_max, fc.v_ground, fc.v_max)
        items = on_wall[fc.id]
        rects = [_axis_rect(poly) for _, _, poly in items]
        holes: list[int] = []
        for k, r in enumerate(rects):
            if r is None:
                continue
            inside = (r[0] > wall_rect[0] + 0.01 and r[1] < wall_rect[1] - 0.01
                      and r[2] > wall_rect[2] + 0.01 and r[3] < wall_rect[3] - 0.01)
            apart = all(o is None or o is r or r[1] <= o[0] or o[1] <= r[0] or r[3] <= o[2] or o[3] <= r[2]
                        for o in rects)
            if inside and apart:
                holes.append(k)
        wall_idx = len(surfaces)
        u0, u1, v0, v1 = wall_rect
        outer = verts.ring([fc.point(u, v) for u, v in [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]])
        rings = [outer]
        for k in holes:
            ring_uv = _ccw_uv(items[k][2])[::-1]               # a hole runs the other way
            rings.append(verts.ring([fc.point(u, v) for u, v in ring_uv]))
        boundaries.append(rings)
        surfaces.append({"type": "WallSurface", "aerosplat_facade": fc.id, "children": []})
        values.append(wall_idx)
        for k, (f, _, poly) in enumerate(items):
            idx = len(surfaces)
            boundaries.append([verts.ring([fc.point(u, v) for u, v in _ccw_uv(poly)])])
            attrs = _ap_attrs(f)
            if k not in holes:
                attrs["aerosplat_not_cut"] = True
            surfaces.append({"type": citygml_of[f.class_], "parent": wall_idx, **attrs})
            surfaces[wall_idx]["children"].append(idx)
            values.append(idx)
            report.openings += 1
        if not surfaces[wall_idx]["children"]:
            del surfaces[wall_idx]["children"]
    for f in pads:
        assert f.pad is not None
        ring = np.asarray(f.pad.polygon_enu, float)
        if _shoelace(ring) < 0:
            ring = ring[::-1]
        idx = len(surfaces)
        boundaries.append([verts.ring([(x, y, f.pad.z) for x, y in ring])])
        surfaces.append({"type": "RoofSurface", **_ap_attrs(f)})
        values.append(idx)
        report.roof_pads += 1
    return {"type": "MultiSurface", "lod": "3", "boundaries": boundaries,
            "semantics": {"surfaces": surfaces, "values": values}}


def _generic_geometry(f: AccessPoint, verts: _Vertices) -> Optional[dict[str, Any]]:
    if f.pad is not None:
        ring = np.asarray(f.pad.polygon_enu, float)
        if _shoelace(ring) < 0:
            ring = ring[::-1]
        pts = [(x, y, f.pad.z) for x, y in ring]
    else:
        corners = corners_enu(f)
        if corners is None:
            return None
        assert f.host is not None and f.geometry is not None
        # Counter-clockwise in the frame it was drawn in = facing out of that wall.
        p = np.asarray(f.geometry.polygon_uv, float)
        pts = [tuple(c) for c in (corners if _shoelace(p) > 0 else corners[::-1])]
    return {"type": "MultiSurface", "lod": "3", "boundaries": [[verts.ring(pts)]]}


def _assemble(objects: dict[str, dict[str, Any]], verts: _Vertices, proj: SiteProjection,
              title: str, identifier: str, reference_date: Optional[datetime]) -> dict[str, Any]:
    if verts.enu:
        xyz = proj.from_enu(np.asarray(verts.enu, float))
        translate = np.floor(xyz.min(axis=0) / SCALE_M) * SCALE_M
        ints = np.rint((xyz - translate) / SCALE_M).astype(np.int64)
    else:
        translate = np.zeros(3)
        ints = np.zeros((0, 3), np.int64)
    # Deduplicate after quantisation; remap every boundary.
    uniq, remap = np.unique(ints, axis=0, return_inverse=True)
    remap = np.asarray(remap).reshape(-1)

    def fix(node):
        if isinstance(node, list):
            if node and isinstance(node[0], (int, np.integer)):
                out = [int(remap[i]) for i in node]
                # Consecutive repeats can only come from points closer than a millimetre.
                dedup = [v for k, v in enumerate(out) if v != out[k - 1]] if len(out) > 1 else out
                return dedup
            return [fix(x) for x in node]
        return node

    for obj in objects.values():
        for g in obj.get("geometry", []):
            g["boundaries"] = fix(g["boundaries"])
    real = uniq * SCALE_M + translate if len(uniq) else np.zeros((0, 3))
    extent = (list(map(float, real.min(axis=0))) + list(map(float, real.max(axis=0)))
              if len(real) else [0.0] * 6)
    meta: dict[str, Any] = {"referenceSystem": proj.crs_url, "title": title, "identifier": identifier,
                            "geographicalExtent": [round(v, 3) for v in extent]}
    if reference_date is not None:
        meta["referenceDate"] = reference_date.date().isoformat()
    return {"type": "CityJSON", "version": VERSION,
            "transform": {"scale": [SCALE_M] * 3, "translate": [float(round(t, 3)) for t in translate]},
            "metadata": meta, "CityObjects": objects, "vertices": uniq.tolist()}


def citygml_map(tax) -> dict[str, Optional[str]]:
    """class id -> its `citygml` name, inherited from the nearest parent that has one."""
    out: dict[str, Optional[str]] = {}
    for cid in tax.classes:
        c = tax.classes.get(cid)
        while c is not None and not c.citygml:
            c = tax.classes.get(c.parent) if c.parent else None
        out[cid] = c.citygml if c is not None else None
    return out


def export_site(catalog, site, *, run_id: Optional[str] = None,
                lifecycles: Sequence[Lifecycle] = (Lifecycle.verified,)
                ) -> tuple[dict[str, Any], ExportReport, list[str], list[str]]:
    """Export a Site's run (default: the newest with semantics) and validate it.

    Returns (document, report, errors, warnings). Raises `SemanticsMissing` when there is
    nothing to export. Shared by the CLI and the API so both refuse the same things.
    """
    from aerosplat.authored.store import AuthoredStore
    from aerosplat.semantics.products import require_semantics
    from aerosplat.taxonomy import cached_taxonomy

    product = require_semantics(catalog, site, run_id=run_id)
    tax = cached_taxonomy(site.taxonomy_packs)
    store = AuthoredStore.for_site(catalog.site_dirs(site), site.key)
    doc, report = export_cityjson(site.key, site.origin, product.run.id, product.run.epoch_id,
                                  product.result, store.features(), lifecycles=lifecycles,
                                  citygml_of=citygml_map(tax),
                                  reference_date=product.run.finished_at or product.run.created_at)
    errors, warnings = validate(doc)
    return doc, report, errors, warnings


# ── offline validation ───────────────────────────────────────────────────
def _depth(node) -> int:
    d = 0
    while isinstance(node, list):
        if not node:
            return d + 1
        node = node[0]
        d += 1
    return d


def _rings_of(geom_type: str, boundaries) -> list[tuple[tuple[int, ...], list[list[int]]]]:
    """(path to the surface, its rings) for every surface of a geometry."""
    out: list[tuple[tuple[int, ...], list[list[int]]]] = []
    if geom_type in ("MultiSurface", "CompositeSurface"):
        for s, rings in enumerate(boundaries):
            out.append(((s,), rings))
    elif geom_type == "Solid":
        for sh, shell in enumerate(boundaries):
            for s, rings in enumerate(shell):
                out.append(((sh, s), rings))
    elif geom_type in ("MultiSolid", "CompositeSolid"):
        for so, solid in enumerate(boundaries):
            for sh, shell in enumerate(solid):
                for s, rings in enumerate(shell):
                    out.append(((so, sh, s), rings))
    return out


def _value_at(values, path: tuple[int, ...]):
    """The semantic value of the surface at `path` (values mirror boundaries down to surfaces)."""
    node = values
    for k in path:
        if not isinstance(node, list) or k >= len(node):
            raise IndexError(path)
        node = node[k]
    return node


def _plane_check(pts: np.ndarray) -> tuple[float, np.ndarray]:
    c = pts.mean(axis=0)
    _, s, vt = np.linalg.svd(pts - c)
    normal = vt[-1]
    return float(np.max(np.abs((pts - c) @ normal))), normal


def validate(doc: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(errors, warnings) for a CityJSON 2.0 document; an empty error list means valid."""
    errors: list[str] = []
    warnings: list[str] = []
    if doc.get("type") != "CityJSON":
        errors.append("type must be 'CityJSON'")
    if doc.get("version") != VERSION:
        errors.append(f"version must be '{VERSION}'")
    allowed_root = {"type", "version", "transform", "metadata", "CityObjects", "vertices",
                    "extensions", "appearance", "geometry-templates"}
    for k in doc:
        if k not in allowed_root and not k.startswith("+"):
            errors.append(f"unknown root property '{k}'")
    tr = doc.get("transform")
    if not (isinstance(tr, dict) and len(tr.get("scale", [])) == 3 and len(tr.get("translate", [])) == 3):
        errors.append("transform needs scale[3] and translate[3]")
        return errors, warnings
    scale = np.asarray(tr["scale"], float)
    translate = np.asarray(tr["translate"], float)
    vertices = doc.get("vertices")
    if not isinstance(vertices, list) or not all(
            isinstance(v, list) and len(v) == 3 and all(isinstance(c, int) for c in v) for v in vertices):
        errors.append("vertices must be a list of [int, int, int]")
        return errors, warnings
    V = np.asarray(vertices, np.int64).reshape(-1, 3)
    real = V * scale + translate
    if len({tuple(v) for v in vertices}) != len(vertices):
        warnings.append("duplicate vertices")
    meta = doc.get("metadata") or {}
    rs = meta.get("referenceSystem", "")
    if not isinstance(rs, str) or not rs.startswith("https://www.opengis.net/def/crs/"):
        errors.append("metadata.referenceSystem must be an OGC CRS URL")
    ext = meta.get("geographicalExtent")
    if ext is not None and len(V):
        lo, hi = real.min(axis=0), real.max(axis=0)
        if len(ext) != 6 or np.any(np.abs(np.asarray(ext[:3]) - lo) > scale) or \
                np.any(np.abs(np.asarray(ext[3:]) - hi) > scale):
            errors.append(f"metadata.geographicalExtent {ext} does not match the vertices")
    objects = doc.get("CityObjects")
    if not isinstance(objects, dict):
        errors.append("CityObjects must be an object")
        return errors, warnings
    used = np.zeros(len(V), bool)
    for oid, obj in objects.items():
        where = f"CityObjects['{oid}']"
        t = obj.get("type")
        if t not in CITY_OBJECT_TYPES and not str(t).startswith("+"):
            errors.append(f"{where}: unknown type '{t}'")
        for rel, back in (("children", "parents"), ("parents", "children")):
            for other in obj.get(rel, []):
                if other not in objects:
                    errors.append(f"{where}: {rel} names missing object '{other}'")
                elif oid not in objects[other].get(back, []):
                    errors.append(f"{where}: '{other}' does not list it in {back}")
        for gi, g in enumerate(obj.get("geometry", [])):
            gw = f"{where}.geometry[{gi}]"
            gt = g.get("type")
            if gt not in GEOMETRY_DEPTH:
                errors.append(f"{gw}: unknown geometry type '{gt}'")
                continue
            lod = g.get("lod")
            lod = lod if isinstance(lod, str) else "?"
            if not (len(lod) in (1, 3) and lod[0] in "0123" and (len(lod) == 1 or (lod[1] == "." and lod[2] in "0123"))):
                errors.append(f"{gw}: lod '{lod}' is not 0..3[.0..3]")
            b = g.get("boundaries")
            if _depth(b) != GEOMETRY_DEPTH[gt]:
                errors.append(f"{gw}: boundaries nested {_depth(b)} deep, {gt} needs {GEOMETRY_DEPTH[gt]}")
                continue
            surfaces = _rings_of(gt, b)
            for path, rings in surfaces:
                outer_normal: Optional[np.ndarray] = None
                outer_xy: Optional[np.ndarray] = None
                for ri, ring in enumerate(rings):
                    rw = f"{gw} surface {path} ring {ri}"
                    if any((not isinstance(i, int)) or i < 0 or i >= len(V) for i in ring):
                        errors.append(f"{rw}: vertex index out of range")
                        continue
                    used[ring] = True
                    if len(set(ring)) < 3:
                        errors.append(f"{rw}: fewer than 3 distinct vertices")
                        continue
                    if ring[0] == ring[-1]:
                        errors.append(f"{rw}: ring is closed (CityJSON rings are not)")
                    if any(ring[k] == ring[k - 1] for k in range(len(ring))):
                        errors.append(f"{rw}: repeated consecutive vertex")
                    if len(set(ring)) != len(ring):
                        errors.append(f"{rw}: a vertex appears twice")
                    pts = real[ring]
                    dev, normal = _plane_check(pts)
                    if dev > PLANARITY_M:
                        errors.append(f"{rw}: not planar ({dev * 1000:.1f} mm)")
                    # Orientation and simplicity in the ring's own plane.
                    axis = int(np.argmax(np.abs(normal)))
                    xy = np.delete(pts - pts.mean(axis=0), axis, axis=1)
                    if not is_simple(xy):
                        errors.append(f"{rw}: ring crosses itself")
                    nwm = _newell(pts)
                    if ri == 0:
                        outer_normal = nwm
                        outer_xy = xy
                    elif outer_normal is not None and outer_xy is not None:
                        if float(np.dot(nwm, outer_normal)) >= 0:
                            errors.append(f"{rw}: inner ring oriented like the outer one")
                        if not point_in_polygon(xy[:, 0], xy[:, 1], [tuple(p) for p in outer_xy]).all():
                            errors.append(f"{rw}: inner ring not inside the outer ring")
            if gt == "Solid":
                for sh, shell in enumerate(b):
                    edges: dict[tuple[int, int], int] = {}
                    for rings in shell:
                        for ring in rings:
                            for idx in range(len(ring)):
                                e = (ring[idx - 1], ring[idx])
                                edges[e] = edges.get(e, 0) + 1
                    bad = [e for e, n in edges.items() if n != 1 or edges.get((e[1], e[0]), 0) != 1]
                    if bad:
                        errors.append(f"{gw} shell {sh}: not closed / not consistently oriented "
                                      f"({len(bad)} edges)")
                    else:
                        vol = _volume(real, shell)
                        if vol <= 0:
                            errors.append(f"{gw} shell {sh}: faces point inwards (volume {vol:.1f} m³)")
            sem = g.get("semantics")
            if sem is not None:
                _check_semantics(sem, gt, surfaces, obj.get("type"), gw, errors)
    if len(V) and not used.all():
        errors.append(f"{int((~used).sum())} unused vertices")
    return errors, warnings


def _newell(pts: np.ndarray) -> np.ndarray:
    n = np.zeros(3)
    for k in range(len(pts)):
        a, b = pts[k], pts[(k + 1) % len(pts)]
        n += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]),
                       (a[0] - b[0]) * (a[1] + b[1])])
    norm = np.linalg.norm(n)
    return n / norm if norm > 0 else n


def _volume(real: np.ndarray, shell) -> float:
    c = real[[i for rings in shell for ring in rings for i in ring]].mean(axis=0)
    vol = 0.0
    for rings in shell:
        ring = rings[0]
        p0 = real[ring[0]] - c
        for k in range(1, len(ring) - 1):
            vol += float(np.dot(p0, np.cross(real[ring[k]] - c, real[ring[k + 1]] - c))) / 6.0
        for hole in rings[1:]:
            q0 = real[hole[0]] - c
            for k in range(1, len(hole) - 1):
                vol += float(np.dot(q0, np.cross(real[hole[k]] - c, real[hole[k + 1]] - c))) / 6.0
    return vol


def _check_semantics(sem, gt: str, surfaces, obj_type, gw: str, errors: list[str]) -> None:
    surf = sem.get("surfaces")
    values = sem.get("values")
    if not isinstance(surf, list) or values is None:
        errors.append(f"{gw}: semantics needs surfaces and values")
        return
    for i, s in enumerate(surf):
        st = s.get("type")
        if obj_type in ("Building", "BuildingPart") and st not in BUILDING_SURFACES and not str(st).startswith("+"):
            errors.append(f"{gw}: semantic surface {i} type '{st}' is not one of a Building's")
        if "parent" in s:
            p = s["parent"]
            if not isinstance(p, int) or not 0 <= p < len(surf) or i not in surf[p].get("children", []):
                errors.append(f"{gw}: semantic surface {i} parent {p} does not list it as a child")
        for c in s.get("children", []):
            if not isinstance(c, int) or not 0 <= c < len(surf) or surf[c].get("parent") != i:
                errors.append(f"{gw}: semantic surface {i} child {c} does not name it as parent")
        if st in OPENING_SURFACES and "parent" not in s:
            errors.append(f"{gw}: {st} surface {i} has no parent wall")
    for path, _ in surfaces:
        try:
            v = _value_at(values, path)
        except IndexError:
            errors.append(f"{gw}: semantics.values has no entry for surface {path}")
            continue
        if v is not None and not (isinstance(v, int) and 0 <= v < len(surf)):
            errors.append(f"{gw}: semantics value {v} at {path} is not a surface index")
    # Values must mirror the boundaries' nesting down to the surface level.
    n_values = sum(1 for _ in _flat(values))
    if n_values != len(surfaces):
        errors.append(f"{gw}: {n_values} semantic values for {len(surfaces)} surfaces")


def _flat(node):
    if isinstance(node, list):
        for x in node:
            yield from _flat(x)
    else:
        yield node

