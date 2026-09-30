#!/usr/bin/env python3
"""
cloud_io.py — AeroSplat-GIS

Shared point-cloud reading/filtering/export used by colmap_parser.py and
ply_parser.py. Both previously carried their own near-duplicate Python
per-point loop; this module replaces them with vectorised NumPy readers.

Output contract (unchanged, backward compatible with colmap_loader.py and
gs_trainer_v3.py): a dict saved via torch.save with

    'xyz' : float32 (N, 3)
    'rgb' : float32 (N, 3) in [0, 1]

plus a new 'meta' dict that downstream code is free to ignore.

Place at: src/reconstruction/cloud_io.py
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# COLMAP points3D.bin
# ---------------------------------------------------------------------------
#
# Record layout (little-endian, no padding):
#   point3D_id      uint64      8
#   x, y, z         float64    24
#   r, g, b         uint8       3
#   error           float64     8
#   track_length    uint64      8   -> 51 bytes fixed
#   track           track_length * (uint32 image_id + uint32 point2D_idx)
#
# Records are variable length, so a single np.frombuffer is impossible.
# Instead: one cheap pass to collect record offsets, then a vectorised
# byte-gather to extract every field at once.

_P3D_FIXED = 51


@dataclass
class Points3D:
    xyz: np.ndarray      # (N, 3) float64
    rgb: np.ndarray      # (N, 3) uint8
    error: np.ndarray    # (N,)   float64 — mean reprojection error, px
    track_len: np.ndarray  # (N,) int64  — number of images that saw the point


def read_points3d_bin(path: Path) -> Points3D:
    buf = path.read_bytes()
    if len(buf) < 8:
        raise ValueError(f"{path} is too short to be a points3D.bin")

    n = struct.unpack_from("<Q", buf, 0)[0]
    if n == 0:
        raise ValueError(f"{path} declares 0 points — SfM likely failed")

    # Pass 1: walk the variable-length records, recording where each starts.
    offsets: list[int] = []
    o = 8
    total = len(buf)
    for _ in range(n):
        if o + _P3D_FIXED > total:
            raise ValueError(
                f"{path} is truncated: header claims {n:,} points but the "
                f"file ends inside record #{len(offsets):,}")
        offsets.append(o)
        track_len = int.from_bytes(buf[o + 43:o + 51], "little")
        o += _P3D_FIXED + track_len * 8
    if o > total:
        raise ValueError(
            f"{path} is truncated: the last record's track list runs "
            f"{o - total} bytes past the end of the file")
    if o != total:
        # Not fatal, but it means the layout assumption drifted somewhere.
        print(f"  warning: {total - o:+d} trailing bytes after the last record")

    off = np.asarray(offsets, dtype=np.int64)
    raw = np.frombuffer(buf, dtype=np.uint8)

    def gather(start: int, stop: int) -> np.ndarray:
        """Extract bytes [start, stop) of every record as a (N, stop-start) array."""
        return raw[off[:, None] + np.arange(start, stop)]

    xyz = gather(8, 32).copy().view(np.float64).reshape(n, 3)
    rgb = gather(32, 35).copy()
    error = gather(35, 43).copy().view(np.float64).reshape(n)
    track = gather(43, 51).copy().view(np.int64).reshape(n)

    return Points3D(xyz=xyz, rgb=rgb, error=error, track_len=track)


# ---------------------------------------------------------------------------
# PLY
# ---------------------------------------------------------------------------

_PLY_TYPES = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}

_RED = ("red", "r", "diffuse_red")
_GREEN = ("green", "g", "diffuse_green")
_BLUE = ("blue", "b", "diffuse_blue")


@dataclass
class PlyHeader:
    fmt: str                      # ascii | binary_little_endian | binary_big_endian
    count: int                    # vertex count
    props: list[tuple[str, str]]  # [(name, numpy type char), ...] for the vertex element
    data_start: int               # byte offset just past end_header


def read_ply_header(path: Path) -> PlyHeader:
    """Parse a PLY header properly instead of guessing the record size.

    The original code assumed 15 or 27 bytes per vertex based on whether the
    substring 'nx' appeared. That happens to be right for COLMAP's two default
    outputs and silently produces garbage for anything else.
    """
    fmt = None
    count = None
    props: list[tuple[str, str]] = []
    in_vertex = False

    with path.open("rb") as f:
        first = f.readline()
        if first.strip() != b"ply":
            raise ValueError(f"{path} does not start with a 'ply' magic line")
        while True:
            line = f.readline()
            if not line:
                # The original loop spun forever here on a malformed file.
                raise ValueError(f"{path}: reached EOF without 'end_header'")
            text = line.decode("ascii", errors="replace").strip()
            if text == "end_header":
                data_start = f.tell()
                break
            parts = text.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise ValueError(
                        f"{path}: vertex element contains a list property "
                        f"('{parts[-1]}') — variable-length vertices are not "
                        f"supported")
                if parts[1] not in _PLY_TYPES:
                    raise ValueError(f"{path}: unknown property type '{parts[1]}'")
                props.append((parts[2], _PLY_TYPES[parts[1]]))

    if fmt is None:
        raise ValueError(f"{path}: no 'format' line in header")
    if count is None:
        raise ValueError(f"{path}: no 'element vertex' in header")
    if not props:
        raise ValueError(f"{path}: vertex element declares no properties")
    return PlyHeader(fmt=fmt, count=count, props=props, data_start=data_start)


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray | None, PlyHeader]:
    """Read a PLY into (xyz float64 (N,3), rgb uint8 (N,3) or None, header)."""
    hdr = read_ply_header(path)
    names = [p[0] for p in hdr.props]

    for axis in ("x", "y", "z"):
        if axis not in names:
            raise ValueError(f"{path}: vertex element has no '{axis}' property")

    if hdr.fmt == "ascii":
        # Slow path, but at least it is correct rather than reading text as
        # binary floats. COLMAP writes binary, so this is a safety net.
        table = np.loadtxt(path, skiprows=_count_header_lines(path),
                           max_rows=hdr.count, dtype=np.float64)
        table = np.atleast_2d(table)
        col = {n: table[:, i] for i, n in enumerate(names)}
    else:
        byteorder = "<" if hdr.fmt == "binary_little_endian" else ">"
        dt = np.dtype([(n, byteorder + t) for n, t in hdr.props])
        buf = np.memmap(path, dtype=np.uint8, mode="r")
        need = hdr.count * dt.itemsize
        avail = buf.size - hdr.data_start
        if avail < need:
            raise ValueError(
                f"{path} is truncated: header claims {hdr.count:,} vertices "
                f"({need:,} bytes) but only {avail:,} bytes follow the header")
        if avail > need:
            print(f"  note: {avail - need:,} trailing bytes after the vertex "
                  f"block (probably a face element — ignored)")
        rec = np.frombuffer(buf, dtype=dt, count=hdr.count,
                            offset=hdr.data_start)
        col = {n: rec[n] for n in names}

    xyz = np.stack([col["x"], col["y"], col["z"]], axis=1).astype(np.float64)

    rgb = None
    rn = _first_present(names, _RED)
    gn = _first_present(names, _GREEN)
    bn = _first_present(names, _BLUE)
    if rn and gn and bn:
        stacked = np.stack([col[rn], col[gn], col[bn]], axis=1)
        if np.issubdtype(stacked.dtype, np.floating):
            # Float colours may be 0..1 or 0..255 depending on the writer.
            if float(np.nanmax(stacked)) <= 1.5:
                stacked = stacked * 255.0
        rgb = np.clip(stacked, 0, 255).astype(np.uint8)

    return xyz, rgb, hdr


def _first_present(names: list[str], candidates) -> str | None:
    for c in candidates:
        if c in names:
            return c
    return None


def _count_header_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        for line in f:
            n += 1
            if line.strip() == b"end_header":
                return n
    raise ValueError(f"{path}: no end_header")


# ---------------------------------------------------------------------------
# Filtering and export
# ---------------------------------------------------------------------------

def filter_cloud(xyz: np.ndarray, rgb: np.ndarray,
                 error: np.ndarray | None = None,
                 track_len: np.ndarray | None = None,
                 max_error: float | None = None,
                 min_track: int | None = None,
                 mad_k: float = 0.0,
                 max_points: int | None = None,
                 seed: int = 0,
                 verbose: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply the standard cleanup passes. Returns (xyz, rgb, stats)."""
    n0 = len(xyz)
    stats: dict[str, int] = {"input": n0}
    keep = np.isfinite(xyz).all(axis=1)
    stats["dropped_nonfinite"] = int((~keep).sum())

    if max_error is not None and error is not None:
        m = error <= max_error
        stats["dropped_high_error"] = int((keep & ~m).sum())
        keep &= m

    if min_track is not None and track_len is not None:
        m = track_len >= min_track
        stats["dropped_short_track"] = int((keep & ~m).sum())
        keep &= m

    if mad_k > 0:
        # Robust per-axis outlier rejection. COLMAP sparse clouds reliably
        # contain a handful of points at effectively infinite distance; they
        # wreck the scene bounds that gs_trainer_v3 initialises scales from.
        sub = xyz[keep]
        if len(sub):
            med = np.median(sub, axis=0)
            mad = np.median(np.abs(sub - med), axis=0)
            mad = np.where(mad < 1e-9, 1e-9, mad)
            m = (np.abs(xyz - med) <= mad_k * mad).all(axis=1)
            stats["dropped_outlier"] = int((keep & ~m).sum())
            keep &= m

    xyz, rgb = xyz[keep], rgb[keep]

    if max_points is not None and len(xyz) > max_points:
        # Random, not strided — COLMAP orders points by track construction, so
        # a fixed stride is spatially biased. Same reasoning as gs_trainer_v3's
        # init_headroom subsampling.
        idx = np.random.default_rng(seed).choice(len(xyz), max_points,
                                                 replace=False)
        idx.sort()
        stats["dropped_subsample"] = len(xyz) - max_points
        xyz, rgb = xyz[idx], rgb[idx]

    stats["output"] = len(xyz)
    if verbose:
        for k, v in stats.items():
            if k not in ("input", "output") and v:
                print(f"  filtered: {k.replace('dropped_', ''):<16} {v:,}")
        print(f"  kept {stats['output']:,} / {n0:,} points "
              f"({100.0 * stats['output'] / max(n0, 1):.1f}%)")
    if stats["output"] == 0:
        raise ValueError("all points were filtered out — loosen the thresholds")
    return xyz, rgb, stats


def save_cloud(out_path: Path, xyz: np.ndarray, rgb: np.ndarray,
               source: Path, stats: dict, recenter: bool = False,
               extra_meta: dict | None = None) -> dict:
    """Write the .pt in the format the rest of the pipeline expects."""
    centroid = xyz.mean(axis=0)
    if recenter:
        xyz = xyz - centroid

    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    extent = float(np.linalg.norm(hi - lo))
    # Median nearest-neighbour spacing is the sane initial Gaussian scale;
    # estimated from a sample so it stays cheap at a million points.
    spacing = _estimate_spacing(xyz)

    meta = {
        "source": str(source),
        "num_points": int(len(xyz)),
        "centroid": centroid.tolist(),
        "recentered": bool(recenter),
        "bbox_min": lo.tolist(),
        "bbox_max": hi.tolist(),
        "extent": extent,
        "median_point_spacing": spacing,
        "filter_stats": stats,
    }
    if extra_meta:
        meta.update(extra_meta)

    data = {
        "xyz": torch.from_numpy(np.ascontiguousarray(xyz, dtype=np.float32)),
        "rgb": torch.from_numpy(np.ascontiguousarray(rgb, dtype=np.float32) / 255.0),
        "meta": meta,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(data, tmp)
    tmp.replace(out_path)   # atomic — never leave a half-written .pt behind

    print(f"  saved {len(xyz):,} points -> {out_path}")
    print(f"  bbox  {lo.round(2).tolist()} .. {hi.round(2).tolist()}  "
          f"(extent {extent:.2f})")
    print(f"  median point spacing ~ {spacing:.4f} "
          f"(a reasonable initial Gaussian scale)")
    return meta


def _estimate_spacing(xyz: np.ndarray, sample: int = 4000,
                      seed: int = 0) -> float:
    n = len(xyz)
    if n < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, min(sample, n), replace=False)
    pts = xyz[idx]
    # Brute force over a small random sample only, chunked to bound memory.
    best = np.full(len(pts), np.inf)
    chunk = 256
    for i in range(0, len(pts), chunk):
        block = pts[i:i + chunk]
        d = np.linalg.norm(block[:, None, :] - pts[None, :, :], axis=2)
        np.fill_diagonal(d[:, i:i + len(block)], np.inf)
        best[i:i + len(block)] = d.min(axis=1)
    finite = best[np.isfinite(best)]
    return float(np.median(finite)) if len(finite) else 0.0