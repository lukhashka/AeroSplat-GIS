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
from aerosplat.semantics.models import UP, Building, Coverage, Facade

F64 = NDArray[np.float64]
# A facade counts as fully covered when this share of its cells is.
FULL_COVERAGE = 0.95


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


# ── what the stage reports ───────────────────────────────────────────────
def distance_to_footprint(footprint, xy=(0.0, 0.0)) -> float:
    """Horizontal distance from `xy` to a footprint polygon; 0 inside it."""
    poly = np.asarray(footprint, float)
    p = np.asarray(xy, float)
    if len(poly) == 0:
        return math.inf
    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
    edge = float(np.min(np.linalg.norm(a + t[:, None] * ab - p, axis=1)))
    # Even-odd rule: does a ray from p towards +x cross an odd number of edges?
    crosses = ((a[:, 1] > p[1]) != (b[:, 1] > p[1])) & (
        p[0] < a[:, 0] + (p[1] - a[:, 1]) * ab[:, 0] / np.where(ab[:, 1] == 0, 1e-12, ab[:, 1]))
    return 0.0 if int(crosses.sum()) % 2 else edge


def coverage_report(buildings: list[Building], facades: list[Facade], *, max_gsd_m: float,
                    n_cameras: int, near: int = 5) -> tuple[list[str], list[str]]:
    """(log lines, warnings) on how well the capture sees the facades.

    Counts are per facade, never a site-wide median: an Earth Studio orbit sees the buildings
    it circles sharply and the rest of a square kilometre from far away, so a median over
    every facade says nothing about any of them. The buildings nearest the Site origin — the
    centre of the capture — are listed one by one, with their own best GSD.
    """
    cov = [f.coverage for f in facades if f.coverage is not None]
    if not cov:
        return [], []
    thr = 100 * max_gsd_m
    any_cov = [c for c in cov if c.covered > 0]
    full = [c for c in any_cov if c.fraction >= FULL_COVERAGE]
    coarse = [c for c in cov if c.too_coarse]
    unseen = [c for c in cov if c.visible == 0]
    n = len(cov)
    lines = [f"Покриття ({n_cameras} кадрів, поріг {thr:g} см/px): з потрібною деталізацією "
             f"{len(any_cov)} із {n} фасадів ({100 * len(any_cov) / n:.0f}%), з них повністю "
             f"(≥{100 * FULL_COVERAGE:.0f}% клітин) {len(full)}; видно лише грубіше за поріг "
             f"{len(coarse)}; не видно {len(unseen)}"]

    by_id = {f.id: f for f in facades}
    ranked = sorted(buildings, key=lambda b: (distance_to_footprint(b.footprint), b.id))
    shown = [b for b in ranked if b.facade_ids][:near]
    if shown:
        lines.append("Найближчі до початку Site будівлі (фасадів з покриттям / усього, медіана "
                     "частки покритих клітин, найкраща GSD на будівлі):")
    for b in shown:
        mine = [c for c in (by_id[i].coverage for i in b.facade_ids if i in by_id)
                if c is not None]
        if not mine:
            continue
        best = [c.best_gsd_cm for c in mine if c.best_gsd_cm is not None]
        sharp = f"найкраще {min(best):.0f} см/px" if best else "не видно"
        lines.append(f"  {b.id} ({distance_to_footprint(b.footprint):.0f} м від початку, "
                     f"h {b.height_m:.0f} м): {sum(c.covered > 0 for c in mine)}/{len(mine)} "
                     f"фасадів, медіана {100 * float(np.median([c.fraction for c in mine])):.0f}%, "
                     f"{sharp}")

    warnings: list[str] = []
    if coarse and len(coarse) > 0.5 * n:
        # The GSD quoted is that of the coarse facades themselves, not the site's best.
        g = np.array([c.best_gsd_cm for c in coarse if c.best_gsd_cm is not None])
        # Rounded up: these are coarser than the bar, and "12.0" next to a 12 bar would not say so.
        finest = math.ceil(float(g.min()) * 10) / 10
        warnings.append(
            f"{len(coarse)} із {n} фасадів видно, але жодна їхня клітина не різкіша за "
            f"{thr:g} см/px (на них найкраще {finest:.1f} см/px, медіана {np.median(g):.0f}): "
            f"отвори на цих фасадах не читаються — для них потрібен ближчий обліт; фасадів з "
            f"потрібною деталізацією {len(any_cov)}")
    return lines, warnings
