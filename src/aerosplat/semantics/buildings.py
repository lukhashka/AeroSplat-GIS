"""Building footprints from the nDSM (ACCESS_POINTS §5.2.2).

A building is a connected run of cells that stand high enough above the bare
earth and actually have points in them. Its outline is traced along cell edges,
so the polygon is exact for the raster, then simplified.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from aerosplat.semantics.models import Building
from aerosplat.semantics.terrain import Terrain

F64 = NDArray[np.float64]
Corner = tuple[int, int]


def trace_region_outline(mask: NDArray[np.bool_]) -> list[Corner]:
    """Outer outline of a single connected True region, as (col, row) corner indices.

    Edges are emitted with the inside on the left, so chaining them is
    unambiguous and the resulting ring is counter-clockwise.
    """
    pad = np.pad(mask, 1)
    ii, jj = np.nonzero(pad)
    succ: dict[Corner, list[Corner]] = {}

    def edge(a: Corner, b: Corner) -> None:
        succ.setdefault(a, []).append(b)

    for i, j in zip(ii.tolist(), jj.tolist()):
        if not pad[i, j + 1]:
            edge((j + 1, i), (j + 1, i + 1))
        if not pad[i + 1, j]:
            edge((j + 1, i + 1), (j, i + 1))
        if not pad[i, j - 1]:
            edge((j, i + 1), (j, i))
        if not pad[i - 1, j]:
            edge((j, i), (j + 1, i))
    if not succ:
        return []

    rings: list[list[Corner]] = []
    while succ:
        start = next(iter(succ))
        ring, cur = [start], start
        while True:
            nxt = succ[cur].pop()
            if not succ[cur]:
                del succ[cur]
            if nxt == start or nxt not in succ:
                break
            ring.append(nxt)
            cur = nxt
        if len(ring) >= 4:
            rings.append(ring)
    if not rings:
        return []
    # The outer ring encloses the most area; the rest are holes.
    outer = max(rings, key=lambda r: abs(_shoelace(np.array(r, dtype=float))))
    # Undo the pad offset.
    return [(c - 1, r - 1) for c, r in outer]


def _shoelace(p: F64) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def drop_collinear(points: F64, tolerance: float = 1e-9) -> F64:
    """Remove points that lie on the segment between their neighbours.

    An outline traced along cell edges is mostly straight runs, so this alone
    turns a rectangle into its four corners -- exactly, without a tolerance to
    tune. Douglas-Peucker then only has to deal with genuine detail.
    """
    if len(points) < 3:
        return points
    prev, nxt = np.roll(points, 1, axis=0), np.roll(points, -1, axis=0)
    corner = np.abs(np.cross(points - prev, nxt - points)) > tolerance
    return points[corner] if corner.sum() >= 3 else points


def simplify(points: F64, tolerance: float) -> F64:
    """Douglas-Peucker on a closed ring (first point is not repeated)."""
    points = drop_collinear(points)
    if len(points) <= 4 or tolerance <= 0:
        return points
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[len(points) // 2] = True          # two anchors split the ring open
    stack = [(0, len(points) // 2), (len(points) // 2, len(points))]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue
        a = points[lo]
        b = points[hi % len(points)]
        seg = b - a
        n = float(np.hypot(*seg))
        rel = points[lo + 1:hi] - a
        d = (np.abs(np.cross(np.broadcast_to(seg, rel.shape), rel)) / n if n > 1e-12
             else np.hypot(rel[:, 0], rel[:, 1]))
        k = int(np.argmax(d))
        if d[k] > tolerance:
            keep[lo + 1 + k] = True
            stack += [(lo, lo + 1 + k), (lo + 1 + k, hi)]
    return points[keep]


def building_mask(terrain: Terrain, *, min_height_m: float = 2.5,
                  min_area_m2: float = 40.0, max_roughness_m: float = 0.15,
                  component_roughness_m: float = 0.10, noise_factor: float = 3.0
                  ) -> tuple[NDArray[np.int64], int]:
    """Label map of building cells (0 = none) and the number of buildings.

    Height alone is not enough: in a suburb most things above 2.5 m are trees.
    Roofs are locally flat, canopy is not, so roughness does the separating
    (thresholds are given per metre of cell size).

    It takes two passes. First every tall blob is scored by its MEDIAN
    roughness, measured before any gating -- gate the cells first and each
    surviving blob looks flat by construction, which is the whole statistic
    thrown away. Blobs that pass are then trimmed cell by cell, which cuts a
    tree touching a wall off the building it leans on.
    """
    res = terrain.model.res_m
    solid = (terrain.count > 0)
    # A fixed threshold only suits a cloud as clean as the one it was tuned on.
    # Bare ground is flat by definition, so its roughness is this cloud's own
    # noise; anything a few times above that is structure, not measurement.
    flat_ground = solid & (terrain.ndsm < 0.5)
    floor = float(np.median(terrain.roughness[flat_ground])) if flat_ground.any() else 0.0
    cell_limit = max(max_roughness_m * res, noise_factor * floor)
    comp_limit = max(component_roughness_m * res, noise_factor * floor)
    high = ndimage.binary_closing((terrain.ndsm > min_height_m) & solid,
                                  structure=np.ones((3, 3), bool)) & solid
    blobs, n_blobs = ndimage.label(high, structure=np.ones((3, 3), dtype=int))
    if n_blobs == 0:
        return blobs.astype(np.int64), 0
    blob_rough = np.asarray(ndimage.median(terrain.roughness, labels=blobs,
                                           index=np.arange(1, n_blobs + 1)))
    flat_blob = np.concatenate([[False], blob_rough <= comp_limit])

    mask = flat_blob[blobs] & (terrain.roughness <= cell_limit) & solid
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), bool)) & solid
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return labels.astype(np.int64), 0

    sizes = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    keep = np.arange(1, n + 1)[sizes * res ** 2 >= min_area_m2]
    remap = np.zeros(n + 1, dtype=np.int64)
    remap[keep] = np.arange(1, len(keep) + 1)
    return remap[labels], len(keep)


def find_buildings(terrain: Terrain, *, min_height_m: float = 2.5, min_area_m2: float = 40.0,
                   max_roughness_m: float = 0.15, component_roughness_m: float = 0.10,
                   noise_factor: float = 3.0,
                   simplify_m: float = 0.75) -> tuple[list[Building], NDArray[np.int64]]:
    """Buildings, largest first, plus the label map their ids index into (1-based)."""
    labels, n = building_mask(terrain, min_height_m=min_height_m, min_area_m2=min_area_m2,
                              max_roughness_m=max_roughness_m, noise_factor=noise_factor,
                              component_roughness_m=component_roughness_m)
    if n == 0:
        return [], labels

    m = terrain.model
    order = np.argsort(-np.bincount(labels.ravel(), minlength=n + 1)[1:]) + 1
    out: list[Building] = []
    relabel = np.zeros(n + 1, dtype=np.int64)
    for new_id, old in enumerate(order.tolist(), start=1):
        relabel[old] = new_id
        mask = labels == old
        ring = trace_region_outline(mask)
        if len(ring) < 4:
            continue
        # Corner (col, row) -> ENU; corner c sits at the low edge of cell c.
        poly = np.array(ring, dtype=float)
        poly[:, 0] = m.x0 + poly[:, 0] * m.res_m
        poly[:, 1] = m.y0 + poly[:, 1] * m.res_m
        poly = simplify(poly, simplify_m)

        ground = float(np.median(terrain.dtm[mask]))
        roof = float(np.nanpercentile(terrain.dsm[mask], 95))
        out.append(Building(
            id=f"B{new_id:02d}", footprint=[(float(x), float(y)) for x, y in poly],
            area_m2=float(mask.sum()) * m.res_m ** 2, ground_z=ground, roof_z=roof,
            height_m=max(0.0, roof - ground)))
    return out, relabel[labels]
