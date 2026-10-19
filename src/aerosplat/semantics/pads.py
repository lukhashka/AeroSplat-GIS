"""Pad zones on roofs and in yards (ACCESS_POINTS §5.7): where a drone could put down.

Works on the semantics stage's 2.5-D rasters (`terrain.npz`: DSM, DTM, point count,
building labels) — no new data, so every run that went through `semantics` can be asked.
Per building:

* **Roof** — the building's label cells. The roof surface is a robust local reference
  (a NaN-aware median over `ref_window_m`), so a chimney or an air-conditioner a cell or two
  wide stands out of it by its height (`obstacle_h_m`, §5.7 item 3). Cells the building
  mask left out inside the outline (the stage drops rough cells: chimneys, antennae) and
  cells without points are obstacles too: unknown space counts as occupied
  (internal reliability requirements §3.3).
* **Yard** — ground cells (`DSM − DTM` below `obstacle_h_m`, observed, not a building)
  within `yard_radius_m` of the building. A tree is canopy over unseen ground, so it is an
  obstacle like any other. Sky visibility (`sky_view`) is checked, because a courtyard
  between tall walls is reachable only straight down.

Each connected flat region is then fitted with a plane: slope (`max_slope_deg`, 5° for
`land`, §5.7 item 1) and residual (`max_rms_m`, item 2) decide whether it is a pad at all.
What survives is eroded by the clearance radius (`clearance_m` = drone radius + margin +
k·σ, item 4): the result is the zone where the drone's *centre* may be put down with that
much free space around it. Regions are reported either way: an accepted `PadZone`, or a
`PadRejection` with the reason in Ukrainian. Nothing here writes or confirms anything; the
API turns zones into `candidate` access points (ADR-0009).

Known limits, on purpose stated rather than hidden: the raster is as coarse as the stage
made it (1 m by default), so obstacles smaller than a cell can disappear into the median;
roads, water and cars on the ground are not recognised (§5.7 item 6 needs a land-cover
segmentation that does not exist yet), so a yard zone may be a road; overhead wires are
never seen (§5.8).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from aerosplat.semantics.buildings import drop_collinear, trace_region_outline
from aerosplat.semantics.models import Building, TerrainModel

VERSION = "1.0.0"
F64 = NDArray[np.float64]
PadKind = Literal["roof", "ground"]


# Plane residual allowed on data from Earth Studio only (owner delegated the choice, 2026-10-19):
# its 1 m raster of a baked mesh leaves 7.5–9.3 cm on the flat roof of the dataset_1910 tower,
# so §5.7's 5 cm rejects every roof there. Real captures keep 5 cm (PadParams.max_rms_m); a
# run whose capture source is unknown or mixed keeps it too — the stricter value is the safe one.
EARTHSTUDIO_MAX_RMS_M = 0.10


@dataclass(frozen=True)
class PadParams:
    max_slope_deg: float = 5.0          # §5.7 item 1: `land`
    max_rms_m: float = 0.05             # §5.7 item 2: residual from the plane
    obstacle_h_m: float = 0.3           # §5.7 item 3: anything this far above the surface
    # Drone radius 0.5 m + margin 1.0 m + 2σ with σ = 0.25 m (half a 1-m cell): the free
    # radius around the point the drone's centre goes down on.
    clearance_m: float = 2.0
    min_usable_area_m2: float = 4.0     # of the zone the centre may be in
    ref_window_m: float = 5.0           # the roof reference median window
    yard_radius_m: float = 30.0
    sky_radius_m: float = 40.0
    sky_dirs: int = 16
    min_sky_view: float = 0.5           # ground pads: share of open sky, 1 = open field
    min_region_cells: int = 4           # smaller flat specks are not reported at all


@dataclass
class PadZone:
    kind: PadKind
    building_id: str                    # the roof's building, or the one the yard is next to
    polygon_enu: list[tuple[float, float]]   # the zone the drone's centre may go down in
    z: float                            # ENU height of the surface at `centre`
    centre: tuple[float, float]         # the point with the most clearance
    slope_deg: float
    rms_m: float
    free_area_m2: float                 # the flat, obstacle-free region before clearance
    usable_area_m2: float               # after the clearance erosion
    clearance_max_m: float              # free radius at `centre`
    sky_view: float
    notes: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """0..1 for sorting the review queue; not a probability of being safe."""
        return float(np.clip(min(1.0, self.usable_area_m2 / 25.0) * self.sky_view, 0.05, 0.9))


@dataclass
class PadRejection:
    kind: PadKind
    building_id: str
    reason: str
    code: str                           # no_roof | no_data | slope | rough | no_room | sky
    area_m2: float = 0.0
    centre: Optional[tuple[float, float]] = None


@dataclass
class PadResult:
    model: str
    zones: list[PadZone] = field(default_factory=list)
    rejected: list[PadRejection] = field(default_factory=list)


@dataclass(frozen=True)
class Rasters:
    """The semantics stage's `terrain.npz`, with the header that places it."""

    model: TerrainModel
    dsm: F64
    dtm: F64
    count: NDArray[np.int64]
    labels: NDArray[np.int64]


# ── helpers ──────────────────────────────────────────────────────────────
def plane_fit(x: F64, y: F64, z: F64) -> tuple[float, float]:
    """Least-squares plane z = a x + b y + c: (slope in degrees, RMS residual in metres)."""
    if len(z) < 3:
        return float("nan"), float("nan")
    A = np.column_stack([x - x.mean(), y - y.mean(), np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    resid = z - A @ coef
    return (math.degrees(math.atan(math.hypot(coef[0], coef[1]))),
            float(np.sqrt(np.mean(resid ** 2))))


def steps(a: F64, h: float) -> NDArray[np.bool_]:
    """Cells with a 4-neighbour more than `h` higher or lower: a roof step, a parapet, a curb.

    They cut a surface into its levels, so two plateaus joined by a ramp-like seam of the
    median filter are not fitted as one plane. NaN neighbours are ignored.
    """
    p = np.pad(a, 1, constant_values=np.nan)
    c = p[1:-1, 1:-1]
    out = np.zeros(a.shape, bool)
    for n in (p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]):
        with np.errstate(invalid="ignore"):
            out |= np.abs(c - n) > h
    return out


def local_slope(z: F64, valid: NDArray[np.bool_], size: int, res_m: float) -> F64:
    """Slope (degrees) of the least-squares plane through the valid cells of each size×size window.

    Unlike a gradient of a filtered surface this stays exact at the edge of a roof (a
    truncated window still fits the same plane) and ignores the cells left out (chimneys,
    unobserved). NaN where the window holds fewer than 3 non-collinear valid cells.
    """
    half = size // 2
    d = np.arange(-half, half + 1, dtype=float)
    dy, dx = np.meshgrid(d, d, indexing="ij")
    w = valid.astype(float)
    zw = np.where(valid, z, 0.0)

    def corr(a: F64, k: F64) -> F64:
        return ndimage.correlate(a, k, mode="constant", cval=0.0)

    ones = np.ones_like(dx)
    S1, Sx, Sy = corr(w, ones), corr(w, dx), corr(w, dy)
    Sxx, Syy, Sxy = corr(w, dx * dx), corr(w, dy * dy), corr(w, dx * dy)
    Sz, Sxz, Syz = corr(zw, ones), corr(zw, dx), corr(zw, dy)
    M = np.stack([np.stack([Sxx, Sxy, Sx], -1), np.stack([Sxy, Syy, Sy], -1),
                  np.stack([Sx, Sy, S1], -1)], -2)
    rhs = np.stack([Sxz, Syz, Sz], -1)
    out = np.full(z.shape, np.nan)
    det = np.linalg.det(M)
    ok = valid & (S1 >= 3) & (np.abs(det) > 1e-6)
    if ok.any():
        coef = np.linalg.solve(M[ok], rhs[ok][..., None])[..., 0]
        out[ok] = np.degrees(np.arctan(np.hypot(coef[:, 0], coef[:, 1]) / res_m))
    return out


def nan_median_filter(a: F64, size: int) -> F64:
    """Median over a size×size window ignoring NaN; NaN where the whole window is NaN."""
    pad = size // 2
    p = np.pad(a, pad, constant_values=np.nan)
    win = np.lib.stride_tricks.sliding_window_view(p, (size, size))
    flat = win.reshape(*a.shape, size * size)
    out = np.full(a.shape, np.nan)
    ok = ~np.all(np.isnan(flat), axis=-1)
    out[ok] = np.nanmedian(flat[ok], axis=-1)
    return out


def sky_view_cells(surface: F64, ii: NDArray[np.int64], jj: NDArray[np.int64], z0: F64,
                   res_m: float, radius_m: float, dirs: int) -> F64:
    """Sky-view factor at cells (ii, jj) from heights z0: 1 − mean sin(horizon elevation).

    The 2.5-D version of the usual horizon-angle method: for each of `dirs` azimuths the
    highest elevation angle of the surface within `radius_m`. Off the raster, and cells
    without a height, count as open sky. Vectorised over the cells.
    """
    rows, cols = surface.shape
    steps = np.arange(1, int(radius_m / res_m) + 1)
    dist = steps * res_m
    ii = np.asarray(ii, np.int64)
    jj = np.asarray(jj, np.int64)
    z0 = np.asarray(z0, float)
    total = np.zeros(len(ii))
    for k in range(dirs):
        a = 2 * math.pi * k / dirs
        I = ii[:, None] + np.rint(np.cos(a) * steps).astype(np.int64)[None, :]
        J = jj[:, None] + np.rint(np.sin(a) * steps).astype(np.int64)[None, :]
        ok = (I >= 0) & (I < rows) & (J >= 0) & (J < cols)
        h = np.full(I.shape, -np.inf)
        h[ok] = surface[I[ok], J[ok]] - np.broadcast_to(z0[:, None], I.shape)[ok]
        h[~np.isfinite(h)] = -np.inf
        gamma = np.maximum(np.max(np.arctan2(h, dist[None, :]), axis=1), 0.0)
        total += np.sin(gamma)
    return 1.0 - total / dirs


def sky_view(surface: F64, i: int, j: int, z0: float, res_m: float, radius_m: float,
             dirs: int) -> float:
    """`sky_view_cells` for one cell."""
    return float(sky_view_cells(surface, np.array([i]), np.array([j]), np.array([z0]), res_m,
                                radius_m, dirs)[0])


def outline_enu(mask: NDArray[np.bool_], model: TerrainModel, i0: int, j0: int) -> list[tuple[float, float]]:
    """Cell-edge outline of one connected region of a crop, in ENU (counter-clockwise)."""
    ring = trace_region_outline(mask)
    if len(ring) < 3:
        return []
    poly = np.array(ring, dtype=float)
    poly[:, 0] = model.x0 + (poly[:, 0] + j0) * model.res_m
    poly[:, 1] = model.y0 + (poly[:, 1] + i0) * model.res_m
    poly = drop_collinear(poly)
    return [(float(x), float(y)) for x, y in poly]


def point_in_polygon(x: F64, y: F64, poly: Sequence[tuple[float, float]]) -> NDArray[np.bool_]:
    """Even-odd rule, vectorised over points."""
    p = np.asarray(poly, float)
    inside = np.zeros(np.shape(x), bool)
    for (x1, y1), (x2, y2) in zip(p, np.roll(p, -1, axis=0)):
        crosses = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (x < xi)
    return inside


def polygon_overlap(a: Sequence[tuple[float, float]], b: Sequence[tuple[float, float]],
                    step_m: float = 0.25) -> float:
    """Shared area over the smaller polygon's area, sampled on a grid (0 when apart)."""
    pa, pb = np.asarray(a, float), np.asarray(b, float)
    lo = np.maximum(pa.min(0), pb.min(0))
    hi = np.minimum(pa.max(0), pb.max(0))
    if np.any(hi <= lo):
        return 0.0
    smaller = min(abs(_area(pa)), abs(_area(pb)))
    if smaller <= 0:
        return 0.0
    # Sample the bounding boxes' intersection, which holds the shared area.
    xs = np.arange(lo[0] + step_m / 2, hi[0], step_m)
    ys = np.arange(lo[1] + step_m / 2, hi[1], step_m)
    if not len(xs) or not len(ys):
        return 0.0
    X, Y = np.meshgrid(xs, ys)
    shared = float((point_in_polygon(X, Y, a) & point_in_polygon(X, Y, b)).sum()) * step_m ** 2
    return min(1.0, shared / smaller)


def _area(p: F64) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ── the search ───────────────────────────────────────────────────────────
def _zones_in(flat: NDArray[np.bool_], surface: F64, sky_full: F64, *, kind: PadKind,
              building_id: str, model: TerrainModel, i0: int, j0: int, params: PadParams,
              need_sky: bool, out: PadResult) -> None:
    """Fit, erode and judge every connected flat region of a crop at (i0, j0).

    `sky_full` is the whole raster's surface: the horizon reaches beyond the crop.
    """
    res = model.res_m
    regions, n = ndimage.label(flat, structure=np.ones((3, 3), int))
    for k in range(1, n + 1):
        region = regions == k
        cells = int(region.sum())
        if cells < params.min_region_cells:
            continue
        ii, jj = np.nonzero(region)
        x = model.x0 + (jj + j0 + 0.5) * res
        y = model.y0 + (ii + i0 + 0.5) * res
        z = surface[ii, jj]
        centre_guess = (float(x.mean()), float(y.mean()))
        area = cells * res ** 2
        slope, rms = plane_fit(x, y, z)
        where = "дах" if kind == "roof" else "двір"
        if not slope <= params.max_slope_deg:
            out.rejected.append(PadRejection(kind, building_id, f"{where}: ухил площини {slope:.1f}° > "
                                             f"{params.max_slope_deg:g}°", "slope", area, centre_guess))
            continue
        if not rms <= params.max_rms_m:
            out.rejected.append(PadRejection(kind, building_id, f"{where}: нерівність {100 * rms:.1f} см > "
                                             f"{100 * params.max_rms_m:g} см", "rough", area, centre_guess))
            continue
        # Distance from a cell centre to the region's edge; outside the crop is not free.
        dist = ndimage.distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1] * res - res / 2
        usable_all = region & (dist >= params.clearance_m)
        best_seen = float(dist[region].max())
        kept = refused = 0
        svf_map = np.full(region.shape, np.nan)
        if usable_all.any():
            ui, uj = np.nonzero(usable_all)
            svf_map[ui, uj] = sky_view_cells(sky_full, ui + i0, uj + j0, surface[ui, uj] + 1.0, res,
                                             params.sky_radius_m, params.sky_dirs)
            # Every cell is judged, not one point: a zone that runs from an open yard into
            # a slot between tall walls loses the slot.
            dark = usable_all & (svf_map < params.min_sky_view) if need_sky else None
            if dark is not None and dark.any():
                di, dj = np.nonzero(dark)
                out.rejected.append(PadRejection(
                    kind, building_id, f"{where}: відкритість неба {np.nanmedian(svf_map[dark]):.2f} < "
                                       f"{params.min_sky_view:g} на {dark.sum() * res ** 2:.0f} м² — туди "
                                       f"лише прямовисно між стінами", "sky", float(dark.sum()) * res ** 2,
                    (float(model.x0 + (dj.mean() + j0 + 0.5) * res),
                     float(model.y0 + (di.mean() + i0 + 0.5) * res))))
                refused += 1
                usable_all &= ~dark
        parts, m = ndimage.label(usable_all, structure=np.ones((3, 3), int))
        for p in range(1, m + 1):
            part = parts == p
            usable_area = float(part.sum()) * res ** 2
            if usable_area < params.min_usable_area_m2:
                continue
            flat_idx = int(np.argmax(np.where(part, dist, -np.inf)))
            bi, bj = np.unravel_index(flat_idx, part.shape)
            cx = model.x0 + (bj + j0 + 0.5) * res
            cy = model.y0 + (bi + i0 + 0.5) * res
            cz = float(np.median(surface[part]))
            svf = float(svf_map[bi, bj])
            # The outline is the outer ring only, so a zone with a hole (around a chimney)
            # would claim the hole as well: such a zone gets the square below instead.
            holed = bool((ndimage.binary_fill_holes(part) & ~part).any())
            polygon = [] if holed else outline_enu(part, model, i0, j0)
            ok_centroid = False
            if len(polygon) >= 3:
                px, py = _centroid(polygon)
                ci = int(math.floor((py - model.y0) / res)) - i0
                cj = int(math.floor((px - model.x0) / res)) - j0
                ok_centroid = 0 <= ci < part.shape[0] and 0 <= cj < part.shape[1] and bool(part[ci, cj])
            if not ok_centroid:
                # A bent zone's area centroid can fall outside it; then the pad is the square
                # around the best point that keeps the full clearance everywhere in it.
                h = max(res / 2, (float(dist[bi, bj]) - params.clearance_m) / math.sqrt(2))
                polygon = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
            zone = PadZone(kind=kind, building_id=building_id, polygon_enu=polygon, z=cz,
                           centre=(float(cx), float(cy)), slope_deg=slope, rms_m=rms,
                           free_area_m2=area, usable_area_m2=usable_area,
                           clearance_max_m=float(dist[bi, bj]), sky_view=svf)
            if kind == "ground":
                zone.notes.append("дороги, воду й авто не розпізнано (§5.7 п. 6): перевірте, що це не проїзд")
            if not ok_centroid:
                zone.notes.append("зона вигнута або з перешкодою всередині: майданчик — квадрат "
                                  "навколо точки з найбільшим запасом")
            out.zones.append(zone)
            kept += 1
        if kept == 0 and refused == 0:
            out.rejected.append(PadRejection(
                kind, building_id, f"{where}: рівно {area:.0f} м², але немає ділянки ≥ "
                                   f"{params.min_usable_area_m2:g} м², де центр має запас "
                                   f"{params.clearance_m:g} м від перешкод і краю (найбільший запас "
                                   f"{max(0.0, best_seen):.1f} м)", "no_room", area, centre_guess))


def _centroid(poly: Sequence[tuple[float, float]]) -> tuple[float, float]:
    p = np.asarray(poly, float)
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a2 = cross.sum()
    if abs(a2) < 1e-12:
        return float(x.mean()), float(y.mean())
    return float(((x + x1) * cross).sum() / (3 * a2)), float(((y + y1) * cross).sum() / (3 * a2))


def find_pads(r: Rasters, buildings: Sequence[Building], *, building_ids: Optional[Sequence[str]] = None,
              roofs: bool = True, ground: bool = True,
              params: PadParams = PadParams()) -> PadResult:
    """Pad zones on the roofs and in the yards of the given buildings (default: all)."""
    m = r.model
    res = m.res_m
    out = PadResult(model=f"padzones@{VERSION}")
    wanted = set(building_ids) if building_ids is not None else None
    observed = (r.count > 0) & np.isfinite(r.dsm)
    any_building = r.labels > 0
    # The surface the sky view looks along: DSM, with holes filled by the ground.
    sky_all = np.where(np.isfinite(r.dsm), r.dsm, r.dtm)
    reach = int(math.ceil((params.yard_radius_m + params.clearance_m) / res)) + 2
    for b in buildings:
        if wanted is not None and b.id not in wanted:
            continue
        try:
            label = int(b.id.lstrip("B"))
        except ValueError:
            continue
        mask = r.labels == label
        if not mask.any():
            out.rejected.append(PadRejection("roof", b.id, "будівлі немає в растрі міток", "no_roof"))
            continue
        ii, jj = np.nonzero(mask)
        i0, i1 = max(0, ii.min() - reach), min(m.rows, ii.max() + reach + 1)
        j0, j1 = max(0, jj.min() - reach), min(m.cols, jj.max() + reach + 1)
        crop = (slice(i0, i1), slice(j0, j1))
        dsm, dtm = r.dsm[crop], r.dtm[crop]
        roof, obs, built = mask[crop], observed[crop], any_building[crop]

        if roofs:
            outline = ndimage.binary_fill_holes(roof)
            win = max(3, int(round(params.ref_window_m / res)) | 1)
            surface = np.where(roof & obs, dsm, np.nan)
            ref = nan_median_filter(surface, win)
            slope = local_slope(dsm, roof & obs, win, res)
            obstacle = outline & (~roof | ~obs | (dsm - ref > params.obstacle_h_m))
            # Steps only between roof cells: the drop at the roof's own edge is not one.
            flat = (roof & obs & ~obstacle & (slope <= params.max_slope_deg)
                    & ~steps(surface, params.obstacle_h_m))
            if not flat.any():
                med = float(np.nanmedian(slope[roof])) if np.isfinite(slope[roof]).any() else float("nan")
                if not (roof & obs).any():
                    out.rejected.append(PadRejection("roof", b.id, "дах без спостережень", "no_data",
                                                     float(roof.sum()) * res ** 2))
                else:
                    out.rejected.append(PadRejection(
                        "roof", b.id, f"дах схилий: медіанний ухил {med:.0f}° > {params.max_slope_deg:g}°",
                        "slope", float(roof.sum()) * res ** 2))
            else:
                _zones_in(flat, np.where(np.isfinite(dsm), dsm, np.nan), sky_all, kind="roof",
                          building_id=b.id, model=m, i0=i0, j0=j0, params=params,
                          need_sky=False, out=out)

        if ground:
            near = ndimage.distance_transform_edt(~roof) * res <= params.yard_radius_m
            gi, gj = np.gradient(dtm, res)
            slope_g = np.degrees(np.arctan(np.hypot(gi, gj)))
            bare = obs & ~built & (dsm - dtm < params.obstacle_h_m)
            flat_g = (near & bare & (slope_g <= params.max_slope_deg)
                      & ~steps(np.where(bare, dsm, np.nan), params.obstacle_h_m))
            if flat_g.any():
                _zones_in(flat_g, np.where(np.isfinite(dsm), dsm, np.nan), sky_all, kind="ground",
                          building_id=b.id, model=m, i0=i0, j0=j0, params=params,
                          need_sky=True, out=out)
            else:
                out.rejected.append(PadRejection("ground", b.id, f"у межах {params.yard_radius_m:g} м "
                                                 f"немає рівної відкритої землі", "no_data"))
    return out
