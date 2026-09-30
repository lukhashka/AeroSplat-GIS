#!/usr/bin/env python3
"""
colmap_parser.py — AeroSplat-GIS

COLMAP sparse `points3D.bin` -> `.pt` point cloud for the GS trainer / server.

Usage:
    python src/reconstruction/colmap_parser.py --dataset zaporizhzhia
    python src/reconstruction/colmap_parser.py \
        --input data/colmap_output/sparse/0/points3D.bin \
        --output data/zaporizhzhia_cloud.pt \
        --max-error 2.0 --min-track 3 --mad-k 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cloud_io import filter_cloud, read_points3d_bin, save_cloud

PROJECT_DIR = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=None,
                   help="Dataset name; fills in the conventional input/output "
                        "paths. Keep this consistent with AEROSPLAT_DATASET "
                        "in src/server/main.py.")
    p.add_argument("--input", type=Path, default=None,
                   help="Path to points3D.bin")
    p.add_argument("--output", type=Path, default=None,
                   help="Path to the .pt to write")

    p.add_argument("--max-error", type=float, default=None,
                   help="Drop points whose mean reprojection error exceeds "
                        "this (px). ~2.0 is a sane starting point.")
    p.add_argument("--min-track", type=int, default=None,
                   help="Drop points seen by fewer than N images. 3 removes "
                        "most two-view triangulation noise.")
    p.add_argument("--mad-k", type=float, default=0.0,
                   help="Robust outlier rejection: drop points more than K "
                        "median-absolute-deviations from the median on any "
                        "axis. Try 20. 0 disables.")
    p.add_argument("--max-points", type=int, default=None,
                   help="Randomly subsample to at most N points.")
    p.add_argument("--recenter", action="store_true",
                   help="Translate the cloud to its centroid. The original "
                        "centroid is stored in meta so this stays reversible "
                        "for the GIS side.")
    p.add_argument("--seed", type=int, default=0)
    return p


def resolve_paths(args) -> tuple[Path, Path]:
    if args.input and args.output:
        return args.input, args.output
    if not args.dataset:
        raise SystemExit("Provide either --dataset or both --input and --output")
    ds = args.dataset
    inp = args.input or (PROJECT_DIR / "data" / "colmap_output" / "sparse" / "0"
                         / "points3D.bin")
    out = args.output or (PROJECT_DIR / "data" / f"{ds}_cloud.pt")
    return inp, out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inp, out = resolve_paths(args)

    if not inp.is_file():
        print(f"ERROR: {inp} not found — run sfm_runner.py first",
              file=sys.stderr)
        return 1

    print(f"Reading COLMAP sparse cloud: {inp}")
    pts = read_points3d_bin(inp)
    print(f"  {len(pts.xyz):,} points in file")
    print(f"  reprojection error: median {float(pts.error.mean()):.3f} px, "
          f"max {float(pts.error.max()):.3f} px")
    print(f"  track length: min {int(pts.track_len.min())}, "
          f"median {int(pts.track_len.mean())}, max {int(pts.track_len.max())}")

    xyz, rgb, stats = filter_cloud(
        pts.xyz, pts.rgb, error=pts.error, track_len=pts.track_len,
        max_error=args.max_error, min_track=args.min_track,
        mad_k=args.mad_k, max_points=args.max_points, seed=args.seed)

    save_cloud(out, xyz, rgb, source=inp, stats=stats, recenter=args.recenter,
               extra_meta={"kind": "colmap_sparse"})
    print("next: python src/reconstruction/dense_runner.py "
          "(or gs_trainer_v3.py to train on the sparse cloud)")
    return 0


from aerosplat.logging_utils import fix_console_encoding as _fix_console_encoding


if __name__ == "__main__":
    _fix_console_encoding()
    raise SystemExit(main())