"""Facade coverage maps (ACCESS_POINTS §5.2.5).

For every cell of a facade, how many captures see it well enough to trust what
is drawn there: looking at it from within `max_angle_deg` of the normal, with a
ground sampling distance of at most `max_gsd_m`, and with the cell inside the
frame. A 3DGS orthophoto of an uncovered patch is invention, not evidence, so
anything found there has to be treated as a guess.

v1 does not ray-trace occlusion, so a wall behind a tree still counts as seen;
`Coverage.occlusion_tested` records that.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from aerosplat.geo.affine import AffineMap
from aerosplat.io.colmap_model import CameraPose
from aerosplat.semantics.models import UP, Coverage, Facade

F64 = NDArray[np.float64]


def transform_pose(pose: CameraPose, affine: AffineMap) -> CameraPose:
    """Carry a world-to-camera pose through a world transform x -> A x + b.

    x_cam = R A^-1 (x' - b) + t, so R' = R A^-1 and t' = t - R' b. R' is only
    exactly a rotation when A is a similarity; it is squared up by polar
    decomposition (for the sphere -> WGS84 map the correction is < 0.11 deg).
    """
    inv = np.linalg.inv(affine.A)
    m = pose.R @ inv
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    return CameraPose(name=pose.name, R=r, t=pose.t - r @ affine.b,
                      intrinsics=pose.intrinsics)


def facade_grid(facade: Facade, cell_m: float) -> tuple[F64, int, int]:
    """Cell-centre ENU points of a facade, as (N, 3) with N = nu * nv (row-major in v)."""
    nu = max(1, int(round(facade.length / cell_m)))
    nv = max(1, int(round(facade.height / cell_m)))
    u = facade.u_min + (np.arange(nu) + 0.5) * (facade.length / nu)
    v = facade.v_ground + (np.arange(nv) + 0.5) * (facade.height / nv)
    uu, vv = np.meshgrid(u, v)
    origin = np.asarray(facade.origin, float)
    pts = (origin + uu.reshape(-1, 1) * np.asarray(facade.u_axis, float)
           + (vv.reshape(-1, 1) - origin[2]) * UP)
    return pts, nu, nv


def facade_coverage(facade: Facade, cameras: list[CameraPose], *, cell_m: float = 0.5,
                    max_angle_deg: float = 60.0, max_gsd_m: float = 0.02
                    ) -> tuple[Coverage, NDArray[np.int32]]:
    """Coverage summary plus the per-cell view count, shaped (nv, nu) with v growing up."""
    pts, nu, nv = facade_grid(facade, cell_m)
    normal = np.asarray(facade.normal, float)
    counts = np.zeros(len(pts), dtype=np.int32)
    seen_any = np.zeros(len(pts), dtype=bool)
    best_gsd = np.full(len(pts), np.inf)
    angle_sum = np.zeros(len(pts))
    angle_n = np.zeros(len(pts), dtype=np.int32)
    cos_limit = math.cos(math.radians(max_angle_deg))

    for cam in cameras:
        w = cam.centre - pts
        dist = np.linalg.norm(w, axis=1)
        ok = dist > 1e-6
        if not ok.any():
            continue
        cos = np.zeros(len(pts))
        np.divide(w @ normal, dist, out=cos, where=ok)
        focal = 0.5 * (cam.intrinsics.fx + cam.intrinsics.fy)
        gsd = np.divide(dist, focal, out=np.full(len(pts), np.inf), where=ok)
        # Geometry first (can this frame look at the cell at all), resolution
        # second: keeping them apart is what tells "never photographed" from
        # "photographed, but too far away to read".
        visible = ok & (cos >= cos_limit)
        if not visible.any():
            continue
        visible &= cam.sees(pts)
        if not visible.any():
            continue
        seen_any |= visible
        np.minimum(best_gsd, np.where(visible, gsd, np.inf), out=best_gsd)
        angle_sum += np.where(visible, np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))), 0.0)
        angle_n += visible
        counts += visible & (gsd <= max_gsd_m)

    covered = counts > 0
    summary = Coverage(
        cell_m=cell_m, cells=len(pts), visible=int(seen_any.sum()),
        covered=int(covered.sum()),
        mean_views=float(counts[covered].mean()) if covered.any() else 0.0,
        max_gsd_cm=round(max_gsd_m * 100, 2),
        best_gsd_cm=round(float(best_gsd[seen_any].min()) * 100, 2) if seen_any.any() else None,
        median_angle_deg=(round(float(np.median(angle_sum[seen_any] / angle_n[seen_any])), 1)
                          if seen_any.any() else None),
        occlusion_tested=False)
    return summary, counts.reshape(nv, nu)


def add_coverage(facades: list[Facade], cameras: list[CameraPose], **kw) -> list[Facade]:
    """`facades` with their `coverage` filled in (the grids themselves are not kept)."""
    return [f.model_copy(update={"coverage": facade_coverage(f, cameras, **kw)[0]})
            for f in facades]
