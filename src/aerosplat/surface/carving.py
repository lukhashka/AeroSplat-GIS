"""Space carving: free space that camera rays prove (follow-up to the 2.5-D grid of step 8.1a).

A fused MVS point was seen by the cameras whose depth maps made it (COLMAP `fused.ply.vis`),
so the segment from each of those cameras to the point crossed nothing opaque. A voxel such a
segment crosses is evidence of free space, and the only evidence that can open space the 2.5-D
model blocks: the air under a canopy, a bridge or a balcony, and the columns where the cloud
has no surface and v1 can only assume where the air begins.

Evidence is accepted conservatively; every rule has a test that fails without it:

* visibility comes from the fusion, never from "the point is in the camera's frame" — a camera
  that frames the point may be looking at the wall in front of it;
* a segment ends at a voxel holding a point of the cloud the grid was built from (points the
  parse stage dropped as outliers do not carve), and that cloud must be a subset of the fused
  points in the grid's frame — a wrong registration fails loudly instead of carving askew;
* the last `margin_m` of each segment is not carved (depth noise along the ray);
* a segment that crosses an occupied voxel is discarded whole — its point or the occupied
  voxel is wrong, and a depth that may be wrong proves nothing — except within its point's
  own surface layer, which a grazing ray to a ground point runs through for metres;
* a voxel needs segments from `min_views` distinct cameras;
* a crossed voxel next to space that stays blocked is given back (one-voxel erosion): a segment
  proves only its own line free, and matter beside it may reach into the voxel it crosses.
  Thin traces — a noise point under a roof, one grazing ray — vanish with it.

Only UNKNOWN voxels are ever carved. OCCUPIED ones hold points or measured ground, and PILLAR
columns stay blocked: they hold a point above the grid that free space must not reach.

A voxel counts as crossed when a sample of the segment falls in it; samples are r/2 apart,
so a voxel whose corner a segment only clips may be left out — never the other way round.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.spatial import cKDTree

from aerosplat.geo.affine import AffineMap
from aerosplat.io.cloud_io import read_ply, read_ply_header
from aerosplat.io.colmap_model import CameraPose, read_model_ids
from aerosplat.surface.models import FREE, OCCUPIED, UNKNOWN, ColumnSource, GridSpec

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]

SQRT3 = math.sqrt(3.0)
# Longest stretch near a segment's end exempt from the occupied check (rays >= ~10 deg to
# the surface they end on); see _segments.
GRAZE_VOXELS = 10.0

# Columns whose free air v1 only assumes; the ones carving can turn into evidence.
ASSUMED_COLUMNS = (int(ColumnSource.FILLED), int(ColumnSource.SKYLINE), int(ColumnSource.SKY))
MEASURED_COLUMNS = (int(ColumnSource.POINTS), int(ColumnSource.CELL))


class VisibilityError(ValueError):
    """Visibility that cannot be trusted to carve with (wrong model, order or frame)."""


@dataclass(frozen=True)
class CameraRays:
    """Which camera saw which point: one segment per (point, camera) pair, in Site ENU."""

    points: F64                         # (N, 3)
    centres: F64                        # (C, 3) camera centres
    point: NDArray[np.integer]          # (M,) index into points
    camera: NDArray[np.integer]         # (M,) index into centres
    # (N, 3) unit surface normals at the points; zero rows (or None): unknown.
    normals: Optional[F64] = None
    info: dict[str, Any] = field(default_factory=dict)      # what the loader found

    def __post_init__(self) -> None:
        for name in ("points", "centres"):
            a = getattr(self, name)
            if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
                raise VisibilityError(f"{name}: expected finite (n, 3), got {a.shape}")
        if self.normals is not None and (self.normals.shape != self.points.shape
                                         or not np.isfinite(self.normals).all()):
            raise VisibilityError("normals must be finite and shaped like points")
        if self.point.shape != self.camera.shape or self.point.ndim != 1:
            raise VisibilityError("point and camera indices must be equal-length 1-D arrays")
        if len(self.point) and (self.point.min() < 0 or self.point.max() >= len(self.points)
                                or self.camera.min() < 0
                                or self.camera.max() >= len(self.centres)):
            raise VisibilityError("ray indices out of range")


# ── COLMAP visibility ────────────────────────────────────────────────────
def read_fused_visibility(path: Path, n_points: int) -> tuple[I64, I64]:
    """(point index, image index) per pair from COLMAP's fused.ply.vis.

    Layout: uint64 point count, then per point a uint32 count and that many uint32 image
    indices — positions of the images in the dense workspace's model, not image ids.
    """
    raw = np.fromfile(path, dtype="<u4")
    if len(raw) < 2:
        raise VisibilityError(f"{path}: too short for a visibility file")
    n = int(raw[0]) | (int(raw[1]) << 32)
    if n != n_points:
        raise VisibilityError(f"{path}: {n:,} entries for {n_points:,} fused points")
    body = raw[2:]
    starts = np.empty(n, np.int64)
    pos, size = 0, len(body)
    for i in range(n):                  # each entry's length is in its head: a sequential walk
        if pos >= size:
            raise VisibilityError(f"{path}: truncated at entry {i:,}")
        starts[i] = pos
        pos += 1 + int(body[pos])
    if pos != size:
        raise VisibilityError(f"{path}: {'truncated' if pos > size else 'trailing data'} "
                              f"after {n:,} entries")
    counts = body[starts].astype(np.int64)
    point = np.repeat(np.arange(n, dtype=np.int64), counts)
    first = np.repeat(starts + 1 - np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
    image = body[first + np.arange(len(point))].astype(np.int64)
    return point, image


def read_fused(path: Path) -> tuple[F64, Optional[F64]]:
    """(xyz, unit normals or None) of a COLMAP fused.ply; zero rows where a normal is unusable."""
    hdr = read_ply_header(path)
    names = [n for n, _ in hdr.props]
    if hdr.fmt == "ascii":
        xyz, _, _ = read_ply(path)
        return xyz, None
    dt = np.dtype([(n, ("<" if hdr.fmt == "binary_little_endian" else ">") + t)
                   for n, t in hdr.props])
    rec = np.fromfile(path, dtype=dt, count=hdr.count, offset=hdr.data_start)
    if len(rec) != hdr.count:
        raise VisibilityError(f"{path}: {len(rec):,} of {hdr.count:,} vertices present")
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float64)
    if not {"nx", "ny", "nz"} <= set(names):
        return xyz, None
    n = np.stack([rec["nx"], rec["ny"], rec["nz"]], axis=1).astype(np.float64)
    return xyz, _unit(n)


def _unit(n: F64) -> F64:
    """Rows scaled to unit length; rows that are not finite or nearly zero become zero."""
    norm = np.linalg.norm(n, axis=1)
    ok = np.isfinite(norm) & (norm > 1e-6)
    out = np.zeros_like(n)
    out[ok] = n[ok] / norm[ok, None]
    return out


def _in_frame(poses: list[CameraPose], xyz: F64, point: I64, image: I64,
              tol_px: float) -> NDArray[np.bool_]:
    """Whether each (point, image) pair projects in front of the camera and into its frame."""
    ok = np.zeros(len(point), np.bool_)
    order = np.argsort(image, kind="stable")
    bounds = np.searchsorted(image[order], np.arange(len(poses) + 1))
    for c, pose in enumerate(poses):
        sel = order[bounds[c]:bounds[c + 1]]
        if len(sel) == 0:
            continue
        uv, z = pose.project(xyz[point[sel]])
        k = pose.intrinsics
        with np.errstate(invalid="ignore"):
            ok[sel] = ((z > 0) & (uv[:, 0] >= -tol_px) & (uv[:, 0] <= k.width + tol_px)
                       & (uv[:, 1] >= -tol_px) & (uv[:, 1] <= k.height + tol_px))
    return ok


def load_colmap_rays(fused_ply: Path, sparse_dir: Path, affine: Optional[AffineMap] = None, *,
                     max_outside: float = 0.005, tol_px: float = 2.0,
                     sample: int = 200_000, seed: int = 0) -> CameraRays:
    """Segments camera -> fused point from a COLMAP dense workspace, mapped by `affine`.

    `sparse_dir` is the workspace's own model (dense/sparse): its images are what the indices
    in fused.ply.vis count. COLMAP numbers them in the order of the model's registered images
    — by image id in current versions, in file order in older ones. Both are tried; the order
    must put at least 1 - `max_outside` of the pairs inside their image (to `tol_px`), and
    must be the only one that does, or there is no trusted visibility at all (checked against
    the depth maps of dataset_v6: the right order frames 99.99% of the pairs, name order —
    wrong there — still 96%, so the test must be strict). Pairs outside their image are dropped.
    """
    fused_ply, sparse_dir = Path(fused_ply), Path(sparse_dir)
    vis = fused_ply.with_name(fused_ply.name + ".vis")
    if not vis.is_file():
        raise VisibilityError(f"{vis}: no visibility file next to the fused cloud")
    xyz, normals = read_fused(fused_ply)
    if not np.isfinite(xyz).all():
        raise VisibilityError(f"{fused_ply}: NaN/inf coordinates")
    point, image = read_fused_visibility(vis, len(xyz))
    entries = read_model_ids(sparse_dir)
    if len(image) and image.max() >= len(entries):
        raise VisibilityError(f"{vis}: image index {int(image.max())} but the model in "
                              f"{sparse_dir} has {len(entries)} images — not this workspace's model")

    ids = np.array([e[0] for e in entries])
    if len(np.unique(ids)) != len(ids):
        raise VisibilityError(f"{sparse_dir}: duplicate image ids")
    candidates: dict[str, I64] = {"image_id": np.argsort(ids, kind="stable")}
    if not np.array_equal(candidates["image_id"], np.arange(len(ids))):
        candidates["file"] = np.arange(len(ids))
    rng = np.random.default_rng(seed)
    probe = rng.choice(len(point), min(sample, len(point)), replace=False) if len(point) else \
        np.zeros(0, np.int64)
    rates: dict[str, float] = {}
    for name, order in candidates.items():
        poses = [entries[i][1] for i in order]
        rates[name] = float(_in_frame(poses, xyz, point[probe], image[probe], tol_px).mean()) \
            if len(probe) else 1.0
    passing = [n for n, v in rates.items() if v >= 1.0 - max_outside]
    shown = ", ".join(f"{n}: {100 * v:.2f}%" for n, v in rates.items())
    if not passing:
        raise VisibilityError(f"{vis}: no image order puts the points inside the images that "
                              f"saw them ({shown} in frame) — wrong sparse model?")
    if len(passing) > 1:
        raise VisibilityError(f"{vis}: image order is ambiguous ({shown} in frame); "
                              f"cannot tell which camera saw which point")
    chosen = passing[0]
    poses = [entries[i][1] for i in candidates[chosen]]
    inside = _in_frame(poses, xyz, point, image, tol_px)
    outside = int((~inside).sum())
    if outside > max_outside * max(1, len(point)):
        raise VisibilityError(f"{vis}: {outside:,} of {len(point):,} pairs fall outside their "
                              f"image — wrong sparse model?")

    centres = np.array([p.centre for p in poses]).reshape(-1, 3)
    a = affine if affine is not None else AffineMap.identity()
    info = {"fused": str(fused_ply), "sparse": str(sparse_dir), "images": len(poses),
            "fused_points": int(len(xyz)), "pairs": int(len(point)),
            "pairs_outside_image": outside, "image_order": chosen,
            "in_frame": {n: round(v, 5) for n, v in rates.items()},
            "normals": normals is not None}
    # Normals are covectors: they map by the inverse transpose.
    mapped = None if normals is None else _unit(normals @ np.linalg.inv(a.A))
    # int32 pairs: they stay in memory through the distance field, the build's peak.
    return CameraRays(points=a.apply(xyz), centres=a.apply(centres),
                      point=point[inside].astype(np.int32), camera=image[inside].astype(np.int32),
                      normals=mapped, info=info)


def registration_affine(path: Optional[Path]) -> Optional[AffineMap]:
    """The reconstruction -> Site ENU map of a registration report (as semantics uses it)."""
    if path is None:
        return None
    block = json.loads(Path(path).read_text(encoding="utf-8")).get("sphere_enu_to_site_enu")
    return AffineMap.from_dict(block) if block else None


def frame_offset_m(cloud_xyz: F64, fused_xyz: F64, sample: int = 20_000, seed: int = 0) -> float:
    """Median distance from cloud points to the nearest fused point: ~0 when the cloud is the
    fused set (or a filtered subset of it) in the same frame."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(cloud_xyz), min(sample, len(cloud_xyz)), replace=False)
    d, _ = cKDTree(fused_xyz).query(cloud_xyz[pick])
    return float(np.median(d))


# ── carving ──────────────────────────────────────────────────────────────
def _clip_to_box(o: F64, d: F64, length: F64, lo: F64, hi: F64) -> tuple[F64, F64]:
    """Parameter range [ta, tb] of each segment o + t d, 0 <= t <= length, inside the box."""
    ta = np.zeros(len(d))
    tb = length.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        for a in range(3):
            da = d[:, a]
            t1 = (lo[a] - o[a]) / da
            t2 = (hi[a] - o[a]) / da
            par = da == 0
            inside = (o[a] >= lo[a]) & (o[a] <= hi[a])
            t1 = np.where(par, -np.inf if inside else np.inf, t1)
            t2 = np.where(par, np.inf if inside else -np.inf, t2)
            ta = np.maximum(ta, np.minimum(t1, t2))
            tb = np.minimum(tb, np.maximum(t1, t2))
    return ta, tb


@dataclass
class _Levels:
    """Coarse-to-fine skipping of free air, which is most of every segment. A segment's
    samples are grouped into intervals of `steps[0]`; an interval survives only if its lowest
    sample lies below the highest `free_from` within its horizontal reach (`reach`, in layers,
    flattened per column: a maximum over a window that covers the interval); survivors are
    split into intervals of `steps[1]` samples and tested again, and so on. Only the last
    survivors are sampled. The skip is exact: a sample below its column's `free_from` is
    never skipped (a test compares this with sampling every segment in full)."""

    steps: list[int]
    reach: list[NDArray[np.float32]]


def _levels(free_from: NDArray[np.integer], lengths_m: tuple[float, ...], res_m: float,
            layers: int) -> _Levels:
    h = 0.5 * res_m
    # A sample on the grid's top face counts in the top layer, so a column blocked to the
    # top (free_from == layers) is reachable at any height: never skip next to one.
    blocked = np.where(free_from >= layers, np.inf, free_from).astype(np.float32)
    steps: list[int] = []
    unit = 1
    for length in sorted(lengths_m):
        n = unit * max(1, int(round(length / (unit * h))))
        if n > unit:
            steps.append(n)
            unit = n
    steps = sorted(steps, reverse=True) or [1]
    reach = []
    for n in steps:
        # Samples of an interval lie within (n - 1) h of its first one; floor() can add one.
        w = int(math.floor((n - 1) * h / res_m)) + 1
        reach.append(ndimage.maximum_filter(blocked, size=2 * w + 1, mode="nearest").ravel())
    return _Levels(steps=steps, reach=reach)


@dataclass
class _Segments:
    """One camera's segments, sampled at b + m s (m = 0..last), in voxel units of the grid
    (float32): east (j), north (i), up (k). last < 0: the segment misses the grid. Occupied
    voxels count against a segment only up to sample `last_occ` (see _segments)."""

    b: tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]
    s: tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]
    last: NDArray[np.float32]
    last_occ: NDArray[np.float32]


def _segments(o: F64, ends: F64, normals: Optional[F64], spec: GridSpec,
              margin_m: float) -> _Segments:
    """Segments from camera centre `o` to `ends`, minus the last `margin_m`.

    Near its end a segment runs alongside the surface it ends on: a grazing ray to a ground
    point spends metres inside the ground's voxel layer. Occupied voxels there are that same
    surface, not an occluder, so they do not discard the segment: with the end's normal n,
    the stretch within a voxel diagonal of the tangent plane (sqrt(3) r / |d.n|, capped at
    GRAZE_VOXELS voxels) is exempt from the occupied check. It is not carved anyway unless
    it lies outside the margin, and it carves only unknown voxels.
    """
    r, h = spec.res_m, 0.5 * spec.res_m
    lo = np.array([spec.x0, spec.y0, spec.z0])
    hi = np.array([spec.x1, spec.y1, spec.z1])
    vec = ends - o
    length = np.linalg.norm(vec, axis=1)
    d = vec / np.maximum(length, 1e-12)[:, None]
    ta, tb = _clip_to_box(o, d, length - margin_m, lo, hi)
    last = np.where(tb >= ta, np.floor((tb - ta) / h + 1e-9), -1.0)
    graze = np.full(len(d), margin_m)
    if normals is not None:
        cos = np.abs((d * normals).sum(axis=1))
        known = np.linalg.norm(normals, axis=1) > 0.5
        stretch = np.minimum(SQRT3 * r / np.maximum(cos, 1e-12), GRAZE_VOXELS * r)
        graze = np.where(known, np.maximum(margin_m, stretch), margin_m)
    t_occ = np.minimum(tb, length - graze)
    last_occ = np.where(t_occ >= ta, np.floor((t_occ - ta) / h + 1e-9), -1.0)
    first = ((o + ta[:, None] * d - lo) / r).astype(np.float32)
    step = (d * (h / r)).astype(np.float32)
    return _Segments(b=(first[:, 0], first[:, 1], first[:, 2]),
                     s=(step[:, 0], step[:, 1], step[:, 2]),
                     last=last.astype(np.float32), last_occ=last_occ.astype(np.float32))


def carve(state: NDArray[np.uint8], spec: GridSpec, free_from: NDArray[np.integer],
          column_source: NDArray[np.uint8], rays: CameraRays, point_voxels: I64, *,
          min_views: int = 2, margin_m: float = 1.0, under_surface: bool = True,
          coarse_m: tuple[float, ...] = (8.0, 1.0), chunk: int = 4096) -> dict[str, Any]:
    """Open UNKNOWN voxels of `state` that the rays prove free, and mark FILLED / SKYLINE /
    SKY columns whose top blocked voxel opened as CARVED in `column_source` — both in place.
    Returns statistics.

    `free_from` is the first free layer of each column in the 2.5-D model (everything at or
    above it is FREE before carving, everything below blocked); `point_voxels` the sorted
    linear indices of the voxels that hold cloud points.
    """
    if state.shape != spec.shape or state.dtype != np.uint8 or not state.flags.c_contiguous:
        raise ValueError("state must be a C-contiguous uint8 array of the grid's shape")
    if not 1 <= min_views <= 255:
        raise ValueError(f"min_views must be in 1..255 (uint8 counts), got {min_views}")
    if spec.voxels >= 2 ** 31:
        raise ValueError(f"{spec.voxels:,} voxels: carving indexes them with int32")
    if not (math.isfinite(margin_m) and margin_m >= 0):
        raise ValueError(f"margin_m must be >= 0, got {margin_m}")
    t_start = time.perf_counter()
    rows, cols, layers = spec.shape
    r = spec.res_m
    flat = state.reshape(-1)
    src = column_source
    carvable = np.isin(src, ASSUMED_COLUMNS + (MEASURED_COLUMNS if under_surface else ())).ravel()
    lv = _levels(free_from, coarse_m, r, layers)

    # Endpoints must lie in a voxel that holds a cloud point.
    pi, pj, pk = spec.index_of(rays.points)
    inside = (pi >= 0) & (pi < rows) & (pj >= 0) & (pj < cols) & (pk >= 0) & (pk < layers)
    supported = np.zeros(len(rays.points), np.bool_)
    if len(point_voxels):
        lin = np.where(inside, (pi * cols + pj) * layers + pk, -1)
        pos = np.minimum(np.searchsorted(point_voxels, lin), len(point_voxels) - 1)
        supported = inside & (point_voxels[pos] == lin)
    use = supported[rays.point]

    views = np.zeros(spec.voxels, np.uint8)
    order = np.flatnonzero(use)
    order = order[np.argsort(rays.camera[order], kind="stable")]
    bounds = np.searchsorted(rays.camera[order], np.arange(len(rays.centres) + 1))
    n_bad = n_samples = 0
    for c in range(len(rays.centres)):
        ray_ids = order[bounds[c]:bounds[c + 1]]
        found: list[NDArray[np.integer]] = []
        for a in range(0, len(ray_ids), chunk):
            ids = rays.point[ray_ids[a:a + chunk]]
            seg = _segments(rays.centres[c], rays.points[ids],
                            None if rays.normals is None else rays.normals[ids], spec, margin_m)
            got, bad, ns = _march(seg, lv, spec, flat, carvable, 1 << 18)
            n_samples += ns
            n_bad += int(bad.sum())
            found.append(got)
        if found:
            idx = np.concatenate(found)
            idx = idx[views[idx] < min_views]
            views[idx] += 1               # repeated indices count once: one camera, one view
    t_rays = time.perf_counter() - t_start

    eroded_stats = _apply(state, spec, views, min_views, src)
    del views
    ceiling = np.clip(free_from.astype(np.int64) - 1, 0, layers - 1)
    ii, jj = np.nonzero(np.isin(src, ASSUMED_COLUMNS))
    opened = state[ii, jj, ceiling[ii, jj]] == FREE
    upgraded = {ColumnSource(int(v)).name.lower(): int(np.count_nonzero(src[ii, jj][opened] == v))
                for v in ASSUMED_COLUMNS}
    column_source[ii[opened], jj[opened]] = ColumnSource.CARVED
    return {"rays_total": int(len(rays.point)), "rays_unsupported": int((~use).sum()),
            "rays_through_occupied": n_bad, "rays_used": int(use.sum()) - n_bad,
            "samples": n_samples, **eroded_stats, "columns_upgraded": upgraded,
            "seconds_rays": round(t_rays, 2),
            "seconds_total": round(time.perf_counter() - t_start, 2)}


def _march(seg: _Segments, lv: _Levels, spec: GridSpec, flat: NDArray[np.uint8],
           carvable: NDArray[np.bool_], batch: int) -> tuple[NDArray[np.integer],
                                                            NDArray[np.bool_], int]:
    """Carving candidates of one camera's segments: (linear indices of the unknown voxels
    crossed by segments that cross no occupied voxel, those bad segments, samples inspected).
    `carvable` is per column, flattened.

    Arrays are (segment or interval, sample) blocks, so per-segment values are gathered once
    per row, and positions are float32 / indices int32 (grids under 2^31 voxels): the work
    is memory-bound. Sample m of a segment is computed by the same expression at every level,
    so a coarse test and the samples it stands for agree to the last bit.
    """
    rows, cols, layers = spec.shape
    nq = len(seg.last)
    bx, by, bz = seg.b
    sx, sy, sz = seg.s
    last = seg.last
    bad = np.zeros(nq, np.bool_)
    if nq == 0 or last.max() < 0:
        return np.zeros(0, np.int32), bad, 0

    def column(m: NDArray[np.float32], rows_of: Any) -> NDArray[np.int32]:
        # Out-of-range values occur only past a segment's end, where results are masked.
        i = np.clip((by[rows_of] + sy[rows_of] * m).astype(np.int32), 0, rows - 1)
        j = np.clip((bx[rows_of] + sx[rows_of] * m).astype(np.int32), 0, cols - 1)
        return i * cols + j

    def survivors(m: NDArray[np.float32], n: int, rows_of: Any, reach: NDArray[np.float32]
                  ) -> NDArray[np.bool_]:
        lp = last[rows_of]
        m_end = np.minimum(m + np.float32(n - 1), lp)
        z_low = bz[rows_of] + np.minimum(sz[rows_of] * m, sz[rows_of] * m_end)
        return (m <= lp) & (z_low < reach.take(column(m, rows_of)))

    # First level: (segment, interval).
    n0 = lv.steps[0]
    m0 = (np.arange(int(last.max()) // n0 + 1) * n0).astype(np.float32)[None, :]
    q, kk = np.nonzero(survivors(m0, n0, np.s_[:, None], lv.reach[0]))
    m = (kk * n0).astype(np.float32)
    for level in range(1, len(lv.steps)):           # (parent, child): split and test again
        n = lv.steps[level]
        mm = m[:, None] + (np.arange(lv.steps[level - 1] // n) * n).astype(np.float32)[None, :]
        pi, ci = np.nonzero(survivors(mm, n, (q[:, None],), lv.reach[level]))
        q, m = q[pi], mm[pi, ci]

    # Every sample of the surviving intervals: (interval, sample). A partial last interval
    # repeats the segment's last sample, which changes nothing.
    nf = lv.steps[-1]
    off = np.arange(nf, dtype=np.float32)[None, :]
    lins: list[NDArray[np.integer]] = []
    segs: list[NDArray[np.int64]] = []
    total = 0
    per = max(1, batch // nf)
    for a in range(0, len(q), per):
        qb = q[a:a + per]
        lb = last[qb][:, None]
        mm = np.minimum(m[a:a + per, None] + off, lb)
        total += int((np.minimum(nf - 1, lb - m[a:a + per, None]) + 1).sum())
        # Inside the grid's box up to rounding at its faces: values >= -0 truncate to 0,
        # the upper faces are clamped.
        i = np.minimum((by[qb][:, None] + sy[qb][:, None] * mm).astype(np.int32), rows - 1)
        j = np.minimum((bx[qb][:, None] + sx[qb][:, None] * mm).astype(np.int32), cols - 1)
        k = np.minimum((bz[qb][:, None] + sz[qb][:, None] * mm).astype(np.int32), layers - 1)
        col = i * cols + j
        lin = col * layers + k
        st = flat.take(lin)
        occ = st == OCCUPIED
        if occ.any():
            occ &= mm <= seg.last_occ[qb][:, None]
            bad[qb[occ.any(axis=1)]] = True
        cand = (st == UNKNOWN) & carvable.take(col)
        lins.append(lin[cand])
        segs.append(np.repeat(qb, cand.sum(axis=1)))
    if not lins:
        return np.zeros(0, np.int32), bad, total
    lin_all, seg_all = np.concatenate(lins), np.concatenate(segs)
    return lin_all[~bad[seg_all]], bad, total


def _apply(state: NDArray[np.uint8], spec: GridSpec, views: NDArray[np.uint8], min_views: int,
           src: NDArray[np.uint8]) -> dict[str, Any]:
    """Open crossed voxels whose 26 neighbours are free or crossed too (one-voxel erosion).

    Blocked neighbours are taken from the grid's boundary rules: beside it unknown, below it
    underground, above it the top layer continued.
    """
    rows, cols, layers = spec.shape
    vol = views.reshape(spec.shape)
    crossed = carved = under = 0
    measured = np.isin(src, MEASURED_COLUMNS)
    chunk = max(1, 8_000_000 // max(1, cols * layers))
    for a in range(0, rows, chunk):
        b = min(rows, a + chunk)
        if vol[a:b].max() < min_views:
            continue
        raw = vol[a:b] >= min_views
        if raw.any():
            # Residual blocked space around rows [a, b), padded by one voxel on every side.
            pad = np.ones((b - a + 2, cols + 2, layers + 2), np.bool_)
            ra, rb = max(0, a - 1), min(rows, b + 1)
            blk = (state[ra:rb] != FREE) & (vol[ra:rb] < min_views)
            pad[ra - a + 1:rb - a + 1, 1:-1, 1:-1] = blk
            pad[ra - a + 1:rb - a + 1, 1:-1, -1] = blk[:, :, -1]     # above: the top layer
            near = ndimage.maximum_filter(pad.view(np.uint8), size=3,
                                          mode="nearest")[1:-1, 1:-1, 1:-1].astype(np.bool_)
            opened = raw & ~near
            crossed += int(raw.sum())
            carved += int(opened.sum())
            under += int((opened & measured[a:b, :, None]).sum())
            state[a:b][opened] = FREE
    return {"voxels_crossed": crossed, "voxels_carved": carved, "voxels_eroded": crossed - carved,
            "carved_under_surface": under, "carved_assumed": carved - under}
