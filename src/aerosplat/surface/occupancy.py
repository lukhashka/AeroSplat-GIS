"""Occupancy grid and distance field of a Site (SIMULATION_AND_DELIVERY §6, step 8.1a).

The grid answers one question for planning and simulation: how far is this point, path or
ray from anything we must not fly into? "Anything" is OCCUPIED (matter was reconstructed
there) or UNKNOWN (nothing says the space is empty), and both block every query the same
way (internal reliability requirements §3 p. 3). FREE needs a reason.

v1 is 2.5-D, like the DSM it starts from: in each voxel column, space ABOVE the column's
surface is free and everything below is blocked. That cannot represent the air under a
bridge, a balcony or a canopy — it is blocked, which is safe. Where the surface height
comes from is recorded per column (`ColumnSource`), because only some of it is measured:

* POINTS / CELL — measured: the highest cloud point in the voxel column, or, when the column
  is empty, in its terrain cell (the DSM; up to one terrain cell of conservative spread).
* FILLED — a gap in the cloud no wider than `fill_radius_m`: assumed no higher than the
  highest surface around it plus `assumed_margin_m`.
* SKYLINE / SKY — no data within `fill_radius_m`: unknown up to the site's skyline plus the
  margin (free above it, an assumption), or with unknown="sky" unknown all the way up.

Camera rays then carve that model (carving.py, when the dense workspace's visibility is
given): unknown voxels that segments from cameras to the points they saw cross become free —
under overhangs too — and a FILLED / SKYLINE / SKY column whose top blocked voxel is carved
becomes CARVED, its free air observed. `free_from` keeps the 2.5-D boundary per column, so a
free voxel's provenance is always known: below it, carved (observed); above it, the column's.

Queries say when a path relies on assumed free space, so a planner can treat that as a
risk rather than as evidence.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from aerosplat.semantics.models import TerrainModel
from aerosplat.surface.carving import CameraRays, carve, frame_offset_m
from aerosplat.surface.esdf import DEFAULT_TILE_VOXELS, distance_field_cm
from aerosplat.surface.models import (ASSUMED_FREE, FORMAT_VERSION, FREE, OCCUPANCY_JSON,
                                      OCCUPANCY_NPZ, OCCUPIED, OUTSIDE, UNKNOWN, CarvingStats,
                                      ColumnSource, GridSpec, OccupancyReport, OccupancyStats,
                                      state_name)

F64 = NDArray[np.float64]
SQRT3 = math.sqrt(3.0)
UNKNOWN_MODES = ("skyline", "sky")


class GridExtentError(ValueError):
    """The chosen extent cannot be built: over `max_voxels`, off the raster, or empty."""


def as_points(xyz, what: str = "points") -> F64:
    """(N, 3) finite float64 points; a single (3,) point becomes (1, 3). Fails loudly otherwise."""
    p = np.asarray(xyz, dtype=np.float64)
    if p.ndim == 1 and p.shape[0] == 3:
        p = p[None, :]
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"{what}: expected (N, 3), got shape {p.shape}")
    if len(p) == 0:
        raise ValueError(f"{what}: empty")
    finite = np.isfinite(p).all(axis=1)
    if not finite.all():
        raise ValueError(f"{what}: {int((~finite).sum())} of {len(p)} rows hold NaN/inf")
    return p


# ── the grid ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RayHits:
    """First blocked voxel along each ray."""

    distance: F64                       # metres to the face of the first blocked voxel; inf: none
    state: NDArray[np.uint8]            # what was hit (OCCUPIED / UNKNOWN / OUTSIDE); FREE: none

    @property
    def hit(self) -> NDArray[np.bool_]:
        return np.isfinite(self.distance)


@dataclass(frozen=True)
class PathClearance:
    """How close a polyline comes to blocked space."""

    length_m: float
    n_samples: int
    step_m: float                       # largest spacing between consecutive samples
    # A lower bound over the WHOLE path, between samples too (sampled minimum - step/2).
    min_clearance_m: float
    min_at: tuple[float, float, float]  # the sample with the smallest clearance
    min_at_s: float                     # ... and its arc length along the path
    # Exact arc length where the path first enters a blocked voxel, and what it is.
    first_blocked_s: Optional[float]
    first_blocked_state: Optional[str]  # "occupied" / "unknown" / "outside"
    # First sample in free space that is assumed rather than measured (ColumnSource).
    assumed_free_s: Optional[float]

    @property
    def blocked(self) -> bool:
        return self.first_blocked_s is not None


@dataclass(frozen=True, eq=False)
class OccupancyGrid:
    spec: GridSpec
    state: NDArray[np.uint8]            # (rows, cols, layers): FREE / OCCUPIED / UNKNOWN
    esdf_cm: NDArray[np.integer]        # same shape: see EsdfInfo (uint16, or int16 if signed)
    column_top: NDArray[np.float32]     # (rows, cols): surface height, NaN where unknown
    column_source: NDArray[np.uint8]    # (rows, cols): ColumnSource
    # (rows, cols): first free layer of the 2.5-D model; free voxels below it were carved.
    free_from: NDArray[np.int16]
    max_distance_m: float

    def __post_init__(self) -> None:
        shape = self.spec.shape
        if self.state.shape != shape or self.state.dtype != np.uint8:
            raise ValueError(f"state must be uint8 {shape}, got {self.state.dtype} {self.state.shape}")
        if self.esdf_cm.shape != shape or self.esdf_cm.dtype not in (np.uint16, np.int16):
            raise ValueError(f"esdf_cm must be (u)int16 {shape}, got "
                             f"{self.esdf_cm.dtype} {self.esdf_cm.shape}")
        if (self.column_top.shape != shape[:2] or self.column_source.shape != shape[:2]
                or self.free_from.shape != shape[:2]):
            raise ValueError(f"column arrays must be {shape[:2]}")
        if self.free_from.dtype != np.int16:
            raise ValueError(f"free_from must be int16, got {self.free_from.dtype}")

    @property
    def res_m(self) -> float:
        return self.spec.res_m

    @property
    def signed(self) -> bool:
        return self.esdf_cm.dtype == np.int16

    # -- lookups with the boundary rules ---------------------------------
    def _codes(self, i, j, k) -> NDArray[np.uint8]:
        """State of voxels by index, applying the rules outside the grid (models.py)."""
        s = self.spec
        out = np.full(np.shape(i), OUTSIDE, np.uint8)
        lateral = (i >= 0) & (i < s.rows) & (j >= 0) & (j < s.cols)
        out[lateral & (k < 0)] = OCCUPIED                       # underground
        m = lateral & (k >= 0)
        out[m] = self.state[i[m], j[m], np.minimum(k[m], s.layers - 1)]    # above: top layer
        return out

    def _esdf_m(self, i, j, k) -> F64:
        """Distance field at in-grid columns; k is clamped into the grid."""
        return self.esdf_cm[i, j, np.clip(k, 0, self.spec.layers - 1)].astype(np.float64) * 0.01

    def state_at(self, xyz) -> NDArray[np.uint8]:
        """FREE / OCCUPIED / UNKNOWN / OUTSIDE at ENU points."""
        return self._codes(*self.spec.index_of(as_points(xyz, "query points")))

    def source_at(self, xyz) -> NDArray[np.uint8]:
        """ColumnSource of the column under each point; 0 laterally outside the grid."""
        s = self.spec
        i, j, _ = s.index_of(as_points(xyz, "query points"))
        out = np.zeros(len(i), np.uint8)
        m = (i >= 0) & (i < s.rows) & (j >= 0) & (j < s.cols)
        out[m] = self.column_source[i[m], j[m]]
        return out

    def assumed_at(self, xyz) -> NDArray[np.bool_]:
        """Whether each point is in free space that is assumed rather than observed: above the
        2.5-D surface of a FILLED / SKYLINE column (carved voxels below it are observed).
        Above the grid the top layer's answer holds, as for the state."""
        p = as_points(xyz, "query points")
        s = self.spec
        i, j, k = s.index_of(p)
        out = np.zeros(len(p), np.bool_)
        m = self._codes(i, j, k) == FREE
        im, jm, km = i[m], j[m], np.minimum(k[m], s.layers - 1)
        out[m] = (np.isin(self.column_source[im, jm], ASSUMED_FREE)
                  & (km >= self.free_from[im, jm]))
        return out

    # -- clearance -------------------------------------------------------
    def clearance(self, xyz, *, exact_within_m: Optional[float] = None) -> F64:
        """Lower bound on the distance from each point to blocked space, in metres.

        Blocked space is every occupied, unknown or out-of-grid voxel taken as a solid cube,
        so the value never exceeds the true distance to anything the grid holds. It is 0 in
        blocked space. Near obstacles — where the voxel's distance field is at most
        `exact_within_m`, default 8 voxels — it is the exact distance to the nearest blocked
        cube; farther away it is a cheaper bound, low by up to about two voxels. It saturates
        below the field's truncation (`max_distance_m`).
        """
        p = as_points(xyz, "query points")
        s = self.spec
        codes = self._codes(*s.index_of(p))
        out = np.zeros(len(p))
        free = codes == FREE
        if free.any():
            q = p[free].copy()
            # Above the grid the columns only continue upward, so nothing gets closer than it
            # is at the top layer: the distance there is a valid lower bound.
            q[:, 2] = np.minimum(q[:, 2], s.z0 + (s.layers - 0.5) * s.res_m)
            out[free] = self._clearance_free(q, exact_within_m)
        return out

    def _clearance_free(self, q: F64, exact_within_m: Optional[float]) -> F64:
        s = self.spec
        r = s.res_m
        # For any blocked cube B and any voxel centre c:
        #   dist(q, B) >= |q_B - c| - |q - c| - (√3/2) r >= esdf(c) - |q - c| - (√3/2) r,
        # so the best of the 8 centres around q is a valid bound.
        f = np.stack([(q[:, 1] - s.y0) / r, (q[:, 0] - s.x0) / r, (q[:, 2] - s.z0) / r], 1) - 0.5
        base = np.floor(f).astype(np.int64)
        lim = np.array([s.rows - 1, s.cols - 1, s.layers - 1])
        best = np.full(len(q), -np.inf)
        for corner in np.ndindex(2, 2, 2):
            idx = np.clip(base + np.array(corner), 0, lim)
            c = s.centre_of(idx[:, 0], idx[:, 1], idx[:, 2])
            e = self._esdf_m(idx[:, 0], idx[:, 1], idx[:, 2])
            best = np.maximum(best, e - np.linalg.norm(q - c, axis=1))
        out = np.maximum(0.0, best - 0.5 * SQRT3 * r)

        limit = 8.0 * r if exact_within_m is None else float(exact_within_m)
        if limit > 0:
            i, j, k = s.index_of(q)
            near = self._esdf_m(i, j, k) <= limit
            if near.any():
                exact = self._exact_clearance(q[near], limit)
                out[near] = np.where(np.isfinite(exact), exact, out[near])
        return out

    def _exact_clearance(self, q: F64, limit_m: float) -> F64:
        """Distance from each point to the nearest blocked cube, by brute force over a window.

        Valid for points whose own voxel's field is <= limit_m. The nearest cube is then
        closer than U = |q - c| + esdf(c) - r/2 (the cube around the field's nearest centre
        holds the r/2 ball), and a cube n voxels away along any axis is at least (n - 1) r
        away, so a window of ceil(U_max / r) + 1 voxels holds it.
        """
        s = self.spec
        r = s.res_m
        w = int(math.ceil(limit_m / r + 0.5 * SQRT3 - 0.5 + 0.01 / r)) + 1
        span = np.arange(-w, w + 1)
        off = np.stack(np.meshgrid(span, span, span, indexing="ij"), -1).reshape(-1, 3)
        own = np.stack(s.index_of(q), 1)
        out = np.empty(len(q))
        chunk = max(1, 200_000 // len(off))
        for a in range(0, len(q), chunk):
            idx = own[a:a + chunk, None, :] + off[None]
            codes = self._codes(idx[..., 0], idx[..., 1], idx[..., 2])
            gap = np.abs(q[a:a + chunk, None, :] - s.centre_of(idx[..., 0], idx[..., 1], idx[..., 2]))
            d = np.sqrt((np.maximum(gap - 0.5 * r, 0.0) ** 2).sum(-1))
            d[codes == FREE] = np.inf
            out[a:a + chunk] = d.min(axis=1)
        return out

    # -- paths -----------------------------------------------------------
    def path_clearance(self, points, *, step_m: Optional[float] = None,
                       exact_within_m: Optional[float] = None) -> PathClearance:
        """Minimum clearance along a polyline, sampled every `step_m` (at most half a voxel).

        Clearance is 1-Lipschitz, so between samples it cannot drop by more than half the
        spacing: subtracting that makes the minimum a bound for the whole path. Where the
        path enters blocked space is found exactly, by casting each segment as a ray.
        """
        pts = as_points(points, "path")
        if len(pts) < 2:
            raise ValueError("a path needs at least two points")
        r = self.res_m
        step = 0.5 * r if step_m is None else float(step_m)
        if not (math.isfinite(step) and 0 < step <= 0.5 * r * (1 + 1e-9)):
            raise ValueError(f"step_m must be in (0, {0.5 * r}] (half a voxel), got {step_m}")

        seg = np.diff(pts, axis=0)
        lens = np.linalg.norm(seg, axis=1)
        n = np.maximum(1, np.ceil(lens / step - 1e-9).astype(np.int64))
        seg_id = np.repeat(np.arange(len(seg)), n)
        t = np.concatenate([np.arange(m) / m for m in n])
        s_start = np.concatenate([[0.0], np.cumsum(lens)])
        samples = np.vstack([pts[seg_id] + t[:, None] * seg[seg_id], pts[-1:]])
        s = np.concatenate([s_start[seg_id] + t * lens[seg_id], s_start[-1:]])
        spacing = float((lens / n).max())

        c = self.clearance(samples, exact_within_m=exact_within_m)
        m = int(np.argmin(c))
        lower = max(0.0, float(c[m]) - 0.5 * spacing)

        codes = self.state_at(samples)
        first_s: Optional[float] = None
        first_code = FREE
        if (codes != FREE).any():                   # covers degenerate, zero-length paths
            b = int(np.argmax(codes != FREE))
            first_s, first_code = float(s[b]), int(codes[b])
        moving = np.flatnonzero(lens > 0)
        if len(moving):
            hits = self.raycast(pts[moving], seg[moving], lens[moving])
            h = np.flatnonzero(hits.hit)
            if len(h):
                s_hit = float(s_start[moving[h[0]]] + hits.distance[h[0]])
                if first_s is None or s_hit <= first_s:
                    first_s, first_code = s_hit, int(hits.state[h[0]])
        if first_s is not None:
            lower = 0.0

        assumed = self.assumed_at(samples)
        return PathClearance(
            length_m=float(s_start[-1]), n_samples=len(samples), step_m=spacing,
            min_clearance_m=lower, min_at=(float(samples[m, 0]), float(samples[m, 1]),
                                           float(samples[m, 2])),
            min_at_s=float(s[m]), first_blocked_s=first_s,
            first_blocked_state=None if first_s is None else state_name(first_code),
            assumed_free_s=float(s[int(np.argmax(assumed))]) if assumed.any() else None)

    # -- rays ------------------------------------------------------------
    def raycast(self, origins, directions, max_range_m) -> RayHits:
        """Distance along each ray to the first blocked voxel (exact, voxel traversal).

        For simulated rangefinders and line-of-sight checks. A ray starting in blocked space
        hits at 0; one leaving the grid sideways hits OUTSIDE at the grid's edge; one going
        up keeps the column rules above the grid. Open space is crossed in jumps the distance
        field proves empty, the last voxels one face at a time (Amanatides & Woo).
        """
        o = as_points(origins, "ray origins")
        d = as_points(directions, "ray directions")
        if len(o) != len(d):
            if len(d) == 1:
                d = np.repeat(d, len(o), axis=0)
            elif len(o) == 1:
                o = np.repeat(o, len(d), axis=0)
            else:
                raise ValueError(f"{len(o)} origins but {len(d)} directions")
        norm = np.linalg.norm(d, axis=1)
        if (norm <= 1e-12).any():
            raise ValueError("ray direction of zero length")
        d = d / norm[:, None]
        rng = np.broadcast_to(np.asarray(max_range_m, np.float64), (len(o),)).copy()
        if not (np.isfinite(rng).all() and (rng > 0).all()):
            raise ValueError("max_range_m must be finite and positive")

        s = self.spec
        r = s.res_m
        n = len(o)
        dist = np.full(n, np.inf)
        hit_state = np.full(n, FREE, np.uint8)
        v = np.stack(s.index_of(o), 1)                          # (i, j, k) per ray
        codes = self._codes(v[:, 0], v[:, 1], v[:, 2])
        start_blocked = codes != FREE
        dist[start_blocked] = 0.0
        hit_state[start_blocked] = codes[start_blocked]
        active = ~start_blocked
        t = np.zeros(n)

        # Per axis in (i, j, k) order, i.e. (north, east, up).
        dirs, orig = d[:, [1, 0, 2]], o[:, [1, 0, 2]]
        lo = np.array([s.y0, s.x0, s.z0])
        step = np.sign(dirs).astype(np.int64)
        inv = np.where(dirs != 0, 1.0 / np.where(dirs != 0, dirs, 1.0), np.inf)

        # Every iteration jumps >= r or enters a new voxel, and a ray of length L crosses at
        # most 3 (L / r + 1) voxels: this bound is never reached unless something is wrong.
        for _ in range(int(math.ceil(6.0 * float(rng.max()) / r)) + 100):
            a = np.flatnonzero(active)
            if len(a) == 0:
                break
            va = v[a]
            # D(p) >= esdf(c) - |p - c| - (√3/2) r >= esdf(c) - √3 r for p in the voxel of c.
            jump = self._esdf_m(va[:, 0], va[:, 1], va[:, 2]) - SQRT3 * r - 1e-6
            jumping = jump > r
            if jumping.any():
                aj = a[jumping]
                t[aj] += jump[jumping]
                beyond = t[aj] >= rng[aj]
                active[aj[beyond]] = False
                keep = aj[~beyond]
                v[keep] = np.stack(s.index_of(o[keep] + t[keep, None] * d[keep]), 1)
            b = a[~jumping]
            if len(b):
                vb = v[b]
                face = lo + (vb + (step[b] > 0)) * r            # the face each axis exits through
                with np.errstate(invalid="ignore"):
                    tb = np.where(step[b] != 0, (face - orig[b]) * inv[b], np.inf)
                ax = np.argmin(tb, axis=1)
                tn = np.maximum(tb[np.arange(len(b)), ax], t[b])
                beyond = tn > rng[b]
                active[b[beyond]] = False
                go = ~beyond
                bg, axg = b[go], ax[go]
                v[bg, axg] += step[bg, axg]
                t[bg] = tn[go]
                cg = self._codes(v[bg, 0], v[bg, 1], v[bg, 2])
                hit = cg != FREE
                dist[bg[hit]] = t[bg[hit]]
                hit_state[bg[hit]] = cg[hit]
                active[bg[hit]] = False
        if active.any():
            raise RuntimeError("raycast did not terminate (bug)")
        return RayHits(distance=dist, state=hit_state)


# ── building ─────────────────────────────────────────────────────────────
def _disk(radius_cells: float) -> NDArray[np.bool_]:
    n = int(math.floor(radius_cells + 1e-9))
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    return (xx ** 2 + yy ** 2) <= radius_cells ** 2 + 1e-9


def build_occupancy(xyz, terrain: TerrainModel, dsm, dtm, *, res_m: float = 0.5,
                    headroom_m: float = 10.0, max_distance_m: float = 20.0,
                    fill_radius_m: float = 5.0, close_radius_m: float = 1.0,
                    assumed_margin_m: float = 1.0, unknown: str = "skyline",
                    skyline_quantile: float = 0.9999,
                    bbox: Optional[tuple[float, float, float, float]] = None,
                    max_voxels: int = 400_000_000, signed: bool = False,
                    tile_voxels: int = DEFAULT_TILE_VOXELS,
                    rays: Union[CameraRays, Callable[[], Optional[CameraRays]], None] = None,
                    carve_min_views: int = 2,
                    carve_margin_m: float = 1.0, carve_under_surface: bool = True,
                    frame_tolerance_m: float = 0.01
                    ) -> tuple[OccupancyGrid, OccupancyStats, list[str]]:
    """Voxelise a dense cloud over the semantics terrain raster (see the module docstring).

    `terrain`, `dsm`, `dtm` are what the `semantics` stage wrote (semantics.json, terrain.npz);
    the voxel size must divide the terrain cell, so the two lattices line up exactly. The
    grid spans the raster (or `bbox`, in Site ENU metres) from one voxel under the lowest
    ground to `headroom_m` above the skyline (the `skyline_quantile` of measured column tops).

    `close_radius_m` closes the surface (grey closing: dilate, then erode): pits narrower
    than about twice the radius are raised to their rim, walls and step edges stay where
    they are. In a sparse cloud a lone noise point under a roof otherwise defines its column
    and opens a free shaft into the building; the price is that slots narrower than ~2 m
    between buildings are closed too, which no corridor needs.

    With `rays` (carving.load_colmap_rays), camera rays then carve the 2.5-D model; the cloud
    must be the fused points the rays end at (or a filtered subset), in the same frame to
    `frame_tolerance_m`, or the build fails. `rays` may be a function that loads them (or
    returns None): it is called only once the grid is known to fit `max_voxels`.
    """
    p = as_points(xyz, "cloud")
    dsm = np.asarray(dsm, np.float64)
    dtm = np.asarray(dtm, np.float64)
    shape_t = (terrain.rows, terrain.cols)
    if dsm.shape != shape_t or dtm.shape != shape_t:
        raise ValueError(f"terrain arrays {dsm.shape}/{dtm.shape} do not match the raster {shape_t}")
    if not np.isfinite(dtm).all():
        raise ValueError("DTM holds NaN/inf; the semantics stage never writes that — corrupt terrain.npz?")
    if np.isinf(dsm).any():
        raise ValueError("DSM holds inf (empty cells must be NaN)")
    for name, value in (("res_m", res_m), ("headroom_m", headroom_m),
                        ("max_distance_m", max_distance_m)):
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{name} must be positive, got {value}")
    for name, value in (("fill_radius_m", fill_radius_m), ("close_radius_m", close_radius_m),
                        ("assumed_margin_m", assumed_margin_m)):
        if not (math.isfinite(value) and value >= 0):
            raise ValueError(f"{name} must be >= 0, got {value}")
    if unknown not in UNKNOWN_MODES:
        raise ValueError(f"unknown must be one of {UNKNOWN_MODES}, got {unknown!r}")
    if not (0.5 < skyline_quantile <= 1.0):
        raise ValueError(f"skyline_quantile must be in (0.5, 1], got {skyline_quantile}")
    ratio = terrain.res_m / res_m
    k_per_cell = int(round(ratio))
    if k_per_cell < 1 or abs(ratio - k_per_cell) > 1e-6:
        raise ValueError(f"voxel size {res_m} m must divide the terrain cell {terrain.res_m} m")
    if headroom_m < assumed_margin_m + 2 * res_m:
        raise ValueError(f"headroom_m={headroom_m} must exceed assumed_margin_m + 2 voxels "
                         f"({assumed_margin_m + 2 * res_m}), or the skyline rule has no room")
    if rays is not None:
        if int(carve_min_views) != carve_min_views or not 1 <= carve_min_views <= 255:
            raise ValueError(f"carve_min_views must be an integer in 1..255, got {carve_min_views}")
        if not (math.isfinite(carve_margin_m) and carve_margin_m >= 0):
            raise ValueError(f"carve_margin_m must be >= 0, got {carve_margin_m}")
        if not (math.isfinite(frame_tolerance_m) and frame_tolerance_m > 0):
            raise ValueError(f"frame_tolerance_m must be positive, got {frame_tolerance_m}")

    # Lateral lattice: the terrain raster split k_per_cell times, optionally cropped.
    i_lo, i_hi = 0, terrain.rows * k_per_cell
    j_lo, j_hi = 0, terrain.cols * k_per_cell
    if bbox is not None:
        xmin, ymin, xmax, ymax = (float(v) for v in bbox)
        if not (all(math.isfinite(v) for v in (xmin, ymin, xmax, ymax))
                and xmin < xmax and ymin < ymax):
            raise ValueError(f"bad bbox {bbox}")
        j_lo, j_hi = (max(j_lo, int(math.floor((xmin - terrain.x0) / res_m))),
                      min(j_hi, int(math.ceil((xmax - terrain.x0) / res_m))))
        i_lo, i_hi = (max(i_lo, int(math.floor((ymin - terrain.y0) / res_m))),
                      min(i_hi, int(math.ceil((ymax - terrain.y0) / res_m))))
        if i_lo >= i_hi or j_lo >= j_hi:
            raise GridExtentError(f"bbox {bbox} does not overlap the terrain raster")
    rows, cols = i_hi - i_lo, j_hi - j_lo
    x0, y0 = terrain.x0 + j_lo * res_m, terrain.y0 + i_lo * res_m

    # Highest point per voxel column.
    pi = np.floor((p[:, 1] - y0) / res_m).astype(np.int64)
    pj = np.floor((p[:, 0] - x0) / res_m).astype(np.int64)
    inside = (pi >= 0) & (pi < rows) & (pj >= 0) & (pj < cols)
    pi, pj, pz = pi[inside], pj[inside], p[inside, 2]
    top_pts = np.full(rows * cols, -np.inf)
    np.maximum.at(top_pts, pi * cols + pj, pz)
    top_pts = top_pts.reshape(rows, cols)
    low_pts = np.full(rows * cols, np.inf)                  # ... and the lowest
    np.minimum.at(low_pts, pi * cols + pj, pz)
    low_pts = low_pts.reshape(rows, cols)

    ti = (i_lo + np.arange(rows)) // k_per_cell
    tj = (j_lo + np.arange(cols)) // k_per_cell
    dsm_c = dsm[np.ix_(ti, tj)]
    dtm_c = dtm[np.ix_(ti, tj)]
    # Lowest point per terrain cell, for the columns that take their surface from the cell.
    ci = np.floor((p[:, 1] - terrain.y0) / terrain.res_m).astype(np.int64)
    cj = np.floor((p[:, 0] - terrain.x0) / terrain.res_m).astype(np.int64)
    on = (ci >= 0) & (ci < terrain.rows) & (cj >= 0) & (cj < terrain.cols)
    low_cell = np.full(terrain.rows * terrain.cols, np.inf)
    np.minimum.at(low_cell, ci[on] * terrain.cols + cj[on], p[on, 2])
    low_cell_c = low_cell.reshape(shape_t)[np.ix_(ti, tj)]
    # Gap filling runs on the whole raster, so a crop still sees the data beyond its edge.
    radius_cells = fill_radius_m / terrain.res_m
    if radius_cells >= 1:
        near = ndimage.maximum_filter(np.where(np.isfinite(dsm), dsm, -np.inf),
                                      footprint=_disk(radius_cells), mode="constant", cval=-np.inf)
    else:
        near = np.full(shape_t, -np.inf)
    near_c = near[np.ix_(ti, tj)]

    source = np.zeros((rows, cols), np.uint8)
    top = np.full((rows, cols), np.nan)
    has_pts = np.isfinite(top_pts)
    top[has_pts] = top_pts[has_pts]
    source[has_pts] = ColumnSource.POINTS
    cell = ~has_pts & np.isfinite(dsm_c)
    top[cell] = dsm_c[cell]
    source[cell] = ColumnSource.CELL
    observed = has_pts | cell
    gap = ~observed & np.isfinite(near_c)
    top[gap] = near_c[gap] + assumed_margin_m
    source[gap] = ColumnSource.FILLED
    empty = source == 0
    if not observed.any():
        raise GridExtentError("no reconstructed surface inside the grid: nothing to build "
                              "occupancy from")

    skyline_z = float(np.quantile(top[observed], skyline_quantile, method="higher"))
    z_low = float(min(dtm_c.min(), pz.min() if len(pz) else np.inf))
    z0 = math.floor(z_low / res_m) * res_m - res_m          # a whole voxel under all ground
    layers = int(math.ceil((skyline_z + headroom_m - z0) / res_m))
    spec = GridSpec(res_m=res_m, x0=x0, y0=y0, z0=z0, rows=rows, cols=cols, layers=layers)
    if layers >= np.iinfo(np.int16).max:
        raise ValueError(f"{layers} layers: more than free_from (int16) can index")
    if spec.voxels > max_voxels:
        raise GridExtentError(f"grid {rows}×{cols}×{layers} = {spec.voxels:,} voxels exceeds "
                              f"max_voxels={max_voxels:,}: give a bbox, a coarser res_m or a "
                              f"larger max_voxels")
    if callable(rays):
        # Loaded only now: reading the visibility of a large site takes most of a minute,
        # which a grid over budget should not have to wait for.
        rays = rays()
    unknown_ceiling = skyline_z + assumed_margin_m if unknown == "skyline" else None

    surf = ~empty
    closed_columns = 0
    if close_radius_m / res_m >= 1:
        # Unknown columns take part at the height they are blocked to, so a narrow free slot
        # beside unknown space closes as well.
        eff = np.where(surf, top, unknown_ceiling if unknown_ceiling is not None
                       else spec.z1 + res_m)
        closed = ndimage.grey_closing(eff, footprint=_disk(close_radius_m / res_m), mode="nearest")
        raised = surf & (closed > top + 1e-9)
        closed_columns = int(raised.sum())
        top[raised] = closed[raised]

    def k_of(z) -> NDArray[np.int64]:
        return np.floor((np.asarray(z, np.float64) - z0) / res_m).astype(np.int64)

    # Highest blocked layer per column; everything above it is free.
    k_ground = k_of(dtm_c)
    k_solid = np.full((rows, cols), layers - 1, np.int64)
    k_solid[surf] = np.maximum(k_of(top[surf]), k_ground[surf])
    if unknown_ceiling is not None:
        k_solid[empty] = np.maximum(k_of(unknown_ceiling), k_ground[empty])
        source[empty] = ColumnSource.SKYLINE
    else:
        source[empty] = ColumnSource.SKY
    pillar = (k_solid >= layers - 1) & (source != ColumnSource.SKY)
    source[pillar] = ColumnSource.PILLAR
    k_solid = np.minimum(k_solid, layers - 1)
    # Ground under a measured column is matter — below the DTM and below the lowest point
    # seen in the column (its terrain cell, for a CELL column). A DTM that runs over a canopy
    # or a bridge deck would otherwise make the air under it matter, which no camera ray
    # could ever open. Between the two the column is unknown: blocked all the same.
    lowest = np.where(has_pts, low_pts, low_cell_c)
    seen_low = np.isfinite(lowest)
    k_fill = np.where(seen_low, np.minimum(k_ground, k_of(np.where(seen_low, lowest, 0.0))),
                      k_ground)

    state = np.empty(spec.shape, np.uint8)
    kk = np.arange(layers, dtype=np.int64)[None, None, :]
    chunk = max(1, 8_000_000 // (cols * layers))
    for a in range(0, rows, chunk):
        b = min(rows, a + chunk)
        blk = np.where(kk > k_solid[a:b, :, None], FREE, UNKNOWN).astype(np.uint8)
        # Under a guessed column the ground stays unknown.
        blk[(kk <= k_fill[a:b, :, None]) & observed[a:b, :, None]] = OCCUPIED
        state[a:b] = blk
    pk = k_of(pz)
    ok = pk < layers                     # points above the grid stand in pillar columns
    state[pi[ok], pj[ok], pk[ok]] = OCCUPIED
    free_from = (k_solid + 1).astype(np.int16)

    carving: Optional[CarvingStats] = None
    if rays is not None:
        offset = frame_offset_m(p, rays.points)
        if offset > frame_tolerance_m:
            raise ValueError(f"the cloud is not the fused point set the rays end at: median "
                             f"distance to the nearest fused point {offset:.3f} m "
                             f"(> {frame_tolerance_m} m) — another run, or a wrong registration?")
        point_voxels = np.unique((pi[ok] * cols + pj[ok]) * layers + pk[ok])
        found = carve(state, spec, free_from, source, rays, point_voxels,
                      min_views=int(carve_min_views), margin_m=carve_margin_m,
                      under_surface=carve_under_surface)
        carving = CarvingStats(min_views=int(carve_min_views), margin_m=carve_margin_m,
                               under_surface=carve_under_surface, frame_offset_m=offset,
                               visibility=rays.info, **found)

    counts = np.zeros(3, np.int64)
    for a in range(0, rows, chunk):
        blk = state[a:a + chunk]
        counts += [np.count_nonzero(blk == FREE), np.count_nonzero(blk == OCCUPIED),
                   np.count_nonzero(blk == UNKNOWN)]
    esdf = distance_field_cm(state, res_m, max_distance_m, signed=signed,     # FREE == 0
                             tile_voxels=tile_voxels)
    grid = OccupancyGrid(spec=spec, state=state, esdf_cm=esdf,
                         column_top=top.astype(np.float32), column_source=source,
                         free_from=free_from, max_distance_m=max_distance_m)

    n_cols = rows * cols
    by_source = {src.name.lower(): int(np.count_nonzero(source == src)) for src in ColumnSource}
    assumed = by_source["filled"] + by_source["skyline"]
    stats = OccupancyStats(
        voxels=spec.voxels, free=int(counts[0]), occupied=int(counts[1]), unknown=int(counts[2]),
        columns=by_source, observed_fraction=float(observed.mean()),
        assumed_fraction=assumed / n_cols, closed_columns=closed_columns, skyline_z=skyline_z,
        unknown_ceiling_z=unknown_ceiling, ceiling_z=spec.z1, n_points=int(len(pz)),
        carving=carving)

    warnings: list[str] = []
    pct = {k: 100.0 * v / n_cols for k, v in by_source.items()}
    if by_source["filled"]:
        warnings.append(f"{pct['filled']:.1f}% колонок — дірки в хмарі, заповнені найвищою "
                        f"поверхнею в радіусі {fill_radius_m:g} м + {assumed_margin_m:g} м: "
                        f"вільний простір над ними — припущення")
    if by_source["skyline"]:
        warnings.append(f"{pct['skyline']:.1f}% колонок без даних у радіусі {fill_radius_m:g} м: "
                        f"невідомі до лінії дахів {unknown_ceiling:.1f} м, вище — ПРИПУЩЕННЯ "
                        f"вільного простору (unknown='skyline')")
    if by_source["sky"]:
        warnings.append(f"{pct['sky']:.1f}% колонок без даних у радіусі {fill_radius_m:g} м: "
                        f"заблоковані на всю висоту (unknown='sky')")
    if by_source["pillar"]:
        warnings.append(f"{by_source['pillar']} колонок сягають стелі сітки {spec.z1:.1f} м "
                        f"(вище за квантиль {skyline_quantile} висот + {headroom_m:g} м): "
                        f"заблоковані до неба — шум у хмарі або справді висока споруда")
    if observed.mean() < 0.5:
        warnings.append(f"лише {100 * observed.mean():.0f}% колонок мають виміряну поверхню: "
                        f"карта здебільшого складається з припущень і невідомого")
    if carving is not None:
        ended = carving.rays_total - carving.rays_unsupported
        through = carving.rays_through_occupied / ended if ended else 0.0
        if through > 0.25:
            warnings.append(f"{100 * through:.0f}% променів перетинають зайняті вокселі перед "
                            f"своєю точкою: видимість або пози камер суперечать хмарі")
    return grid, stats, warnings


# ── files ────────────────────────────────────────────────────────────────
def write_occupancy(folder: Path, grid: OccupancyGrid, report: OccupancyReport) -> None:
    """occupancy.npz, then occupancy.json — the json last, as the marker of a complete write."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / OCCUPANCY_JSON
    if marker.exists():
        marker.unlink()
    np.savez_compressed(folder / OCCUPANCY_NPZ, state=grid.state, esdf_cm=grid.esdf_cm,
                        column_top=grid.column_top, column_source=grid.column_source,
                        free_from=grid.free_from)
    marker.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def read_occupancy(folder: Path) -> tuple[OccupancyGrid, OccupancyReport]:
    folder = Path(folder)
    report = OccupancyReport(**json.loads((folder / OCCUPANCY_JSON).read_text(encoding="utf-8")))
    if report.format != FORMAT_VERSION:
        raise ValueError(f"{folder}: format {report.format!r}, this code reads {FORMAT_VERSION!r}")
    with np.load(folder / OCCUPANCY_NPZ) as z:
        grid = OccupancyGrid(spec=report.grid, state=z["state"], esdf_cm=z["esdf_cm"],
                             column_top=z["column_top"], column_source=z["column_source"],
                             free_from=z["free_from"], max_distance_m=report.esdf.max_distance_m)
    return grid, report
