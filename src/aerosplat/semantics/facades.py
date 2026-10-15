"""Facades: vertical planes fitted to the wall points of each building (ACCESS_POINTS §5.2.3-4).

A vertical plane through a point cloud is a straight line in plan view, so the
fit happens in 2-D: RANSAC a line through the wall points of one building, take
its inliers, refine by total least squares, repeat on what is left. This makes
verticality exact by construction instead of a constraint to enforce.

Only points close to the footprint edge take part — roof points would otherwise
pull lines across the middle of the building.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from aerosplat.semantics.models import (Building, Facade, SurfaceKind, azimuth_of, octant_of,
                                        u_axis_for)
from aerosplat.semantics.terrain import Terrain

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class LineFit:
    point: F64          # (2,) a point on the line
    direction: F64      # (2,) unit
    inliers: NDArray[np.int64]
    rms: float


def _fit_direction(pts: F64) -> tuple[F64, F64]:
    """Total-least-squares line through 2-D points -> (centroid, unit direction)."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, vt[0]


def _perp(d: F64) -> F64:
    return np.array([-d[1], d[0]])


def _longest_run(u: F64, max_gap: float) -> NDArray[np.bool_]:
    """Mask of the longest stretch of sorted-by-u points with no gap wider than `max_gap`."""
    order = np.argsort(u)
    breaks = np.flatnonzero(np.diff(u[order]) > max_gap) + 1
    runs = np.split(order, breaks)
    best = max(runs, key=lambda r: u[r].max() - u[r].min() if len(r) else -np.inf)
    keep = np.zeros(len(u), dtype=bool)
    keep[best] = True
    return keep


def ransac_line(pts: F64, *, inlier_m: float, min_points: int, min_length: float,
                iterations: int, rng: np.random.Generator,
                max_gap_m: float = 8.0) -> LineFit | None:
    """Strongest straight line through 2-D points, or None if nothing qualifies."""
    n = len(pts)
    if n < min_points:
        return None
    best: NDArray[np.bool_] | None = None
    best_count = 0
    # A sampled pair must straddle enough distance to define a stable direction.
    min_pair = max(min_length * 0.5, inlier_m * 4)
    for _ in range(iterations):
        a, b = rng.integers(0, n, size=2)
        seg = pts[b] - pts[a]
        norm = float(np.hypot(*seg))
        if norm < min_pair:
            continue
        hit = np.abs((pts - pts[a]) @ _perp(seg / norm)) < inlier_m
        count = int(hit.sum())
        if count > best_count:
            best, best_count = hit, count
    if best is None or best_count < min_points:
        return None

    hit = best
    c = d = None
    for _ in range(3):                       # refit on inliers, then re-select
        c, d = _fit_direction(pts[hit])
        resid = np.abs((pts - c) @ _perp(d))
        grown = resid < inlier_m
        if grown.sum() < min_points:
            break
        hit = grown
    assert c is not None and d is not None
    idx = np.flatnonzero(hit)
    # A line fitted to one wall often also passes through a wall of the next
    # building along the same street. Keep only the stretch that is actually
    # continuous; the rest stays in the pool and becomes its own facade.
    idx = idx[_longest_run((pts[idx] - c) @ d, max_gap_m)]
    u = (pts[idx] - c) @ d
    if len(idx) < min_points or float(u.max() - u.min()) < min_length:
        return None
    c = c + float(u.mean()) * d
    rms = float(np.sqrt(np.mean(((pts[idx] - c) @ _perp(d)) ** 2)))
    return LineFit(point=c, direction=d, inliers=idx, rms=rms)


def merge_coplanar(fits: list[tuple[F64, F64, F64]], points: list[F64], *,
                   max_angle_deg: float = 15.0, max_offset_m: float = 2.5,
                   min_overlap: float = 0.3) -> list[list[int]]:
    """Group line fits that describe the same wall: near-parallel, near-coincident, overlapping.

    One wall rarely comes out as one fit. The cloud's points scatter about half a
    metre either side of the true surface, so RANSAC peels the wall off in
    parallel slabs a metre apart. Grouping them back is what turns eleven
    "facades" with the same azimuth into one.

    Overlap along the wall is what keeps the grouping from running away. Without
    it, a row of buildings down one street chains slab to slab into a single
    200-metre "facade", because each is parallel to and nearly in line with its
    neighbour. Duplicate slabs of one wall cover the same stretch; neighbours
    down a street follow one another, so they stay apart.

    `fits` are (point, direction, normal) in plan view, `points` the matching
    inlier sets. The result is index groups in order of first appearance.
    """
    cos_limit = math.cos(math.radians(max_angle_deg))
    parent = list(range(len(fits)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(fits)):
        for j in range(i + 1, len(fits)):
            (p_i, d_i, n_i), (p_j, _, n_j) = fits[i], fits[j]
            if abs(float(n_i @ n_j)) < cos_limit:
                continue
            # Distance between the two planes, measured both ways (they are parallel).
            if max(abs(float((p_j - p_i) @ n_i)), abs(float((p_i - p_j) @ n_j))) > max_offset_m:
                continue
            u_i, u_j = points[i][:, :2] @ d_i, points[j][:, :2] @ d_i
            span = min(float(u_i.max() - u_i.min()), float(u_j.max() - u_j.min()))
            overlap = min(float(u_i.max()), float(u_j.max())) - max(float(u_i.min()),
                                                                    float(u_j.min()))
            if overlap < min_overlap * max(span, 1e-6):
                continue
            parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(len(fits)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _wall_points(xyz: F64, terrain: Terrain, labels: NDArray[np.int64], building_id: int, *,
                 edge_cells: int, min_height_m: float) -> F64:
    """Points of one building that sit near its footprint edge and above the ground."""
    mask = labels == building_id
    # Distance (in cells) from each building cell to the outside: walls hug the edge.
    depth = ndimage.distance_transform_edt(mask)
    i, j = terrain.model.index_of(xyz[:, 0], xyz[:, 1])
    on_building = mask[i, j]
    near_edge = depth[i, j] <= edge_cells
    above = xyz[:, 2] - terrain.dtm[i, j] > min_height_m
    return xyz[on_building & near_edge & above]


def facades_of_building(xyz: F64, terrain: Terrain, labels: NDArray[np.int64],
                        building: Building, label_id: int, *, inlier_m: float = 0.8,
                        min_points: int = 40, min_length_m: float = 5.0,
                        min_height_m: float = 3.0, wall_min_height_m: float = 1.5,
                        edge_cells: int = 2, max_facades: int = 12, max_gap_m: float = 8.0,
                        merge_angle_deg: float = 15.0, merge_offset_m: float = 2.5,
                        merge_min_overlap: float = 0.3,
                        iterations: int = 400, seed: int = 0) -> list[Facade]:
    """Every wall plane of one building, named `<building>/<octant><ordinal>`."""
    pts = _wall_points(xyz, terrain, labels, label_id, edge_cells=edge_cells,
                       min_height_m=wall_min_height_m)
    if len(pts) < min_points:
        return []

    rng = np.random.default_rng(seed)
    centre = np.asarray(building.centroid, float)
    remaining = pts
    raw: list[tuple[F64, F64, F64]] = []        # (point2, direction2, normal2)
    raw_pts: list[F64] = []
    for _ in range(max_facades * 2):
        fit = ransac_line(remaining[:, :2], inlier_m=inlier_m, min_points=min_points,
                          min_length=min_length_m, iterations=iterations, rng=rng,
                          max_gap_m=max_gap_m)
        if fit is None:
            break
        normal = _perp(fit.direction)
        if float(normal @ (fit.point - centre)) < 0:      # outward: away from the building
            normal = -normal
        raw.append((fit.point, fit.direction, normal))
        raw_pts.append(remaining[fit.inliers])
        keep = np.ones(len(remaining), dtype=bool)
        keep[fit.inliers] = False
        remaining = remaining[keep]
        if len(remaining) < min_points:
            break

    found: list[tuple[F64, F64, F64, float, int]] = []   # point2, normal2, points, rms, n
    for group in merge_coplanar(raw, raw_pts, max_angle_deg=merge_angle_deg,
                                max_offset_m=merge_offset_m, min_overlap=merge_min_overlap):
        merged = np.concatenate([raw_pts[k] for k in group])
        point2, direction = _fit_direction(merged[:, :2])
        normal2 = _perp(direction)
        if float(normal2 @ (point2 - centre)) < 0:
            normal2 = -normal2
        resid = (merged[:, :2] - point2) @ normal2
        found.append((point2, normal2, merged, float(np.sqrt(np.mean(resid ** 2))), len(merged)))
    found.sort(key=lambda f: -f[4])
    found = found[:max_facades]

    # Name by compass octant, counting left to right as seen from outside.
    staged: list[tuple[str, float, Facade]] = []
    for point2, normal2, inlier_pts, rms, n_pts in found:
        az = azimuth_of((float(normal2[0]), float(normal2[1])))
        ux, uy = u_axis_for((float(normal2[0]), float(normal2[1])))
        u_axis = np.array([ux, uy, 0.0])
        v_ground = float(np.median(terrain.height_at(inlier_pts[:, 0], inlier_pts[:, 1])))
        v_max = float(np.percentile(inlier_pts[:, 2], 95))
        if v_max - v_ground < min_height_m:
            continue
        origin = np.array([point2[0], point2[1], v_ground])
        u = (inlier_pts - origin) @ u_axis
        facade = Facade(
            id="", building_id=building.id, kind=SurfaceKind.wall,
            origin=(float(origin[0]), float(origin[1]), v_ground),
            normal=(float(normal2[0]), float(normal2[1]), 0.0),
            u_axis=(float(ux), float(uy), 0.0), azimuth_deg=az,
            u_min=float(u.min()), u_max=float(u.max()), v_ground=v_ground, v_max=v_max,
            plane_rms_m=rms, n_points=n_pts)
        staged.append((octant_of(az), float(origin[:2] @ np.array([ux, uy])), facade))

    out: list[Facade] = []
    for octant in sorted({s[0] for s in staged}):
        same = sorted((s for s in staged if s[0] == octant), key=lambda s: s[1])
        for ordinal, (_, _, facade) in enumerate(same, start=1):
            out.append(facade.model_copy(update={"id": f"{building.id}/{octant}{ordinal}"}))
    return sorted(out, key=lambda f: f.id)
