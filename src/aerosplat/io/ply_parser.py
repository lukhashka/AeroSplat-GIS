#!/usr/bin/env python3
"""
ply_parser.py — AeroSplat-GIS

COLMAP dense `.ply` -> `.pt` point cloud for the GS trainer / server.

The dense MVS result normally lands at:
    data/colmap_output/dense/0/fused.ply
Note that `sparse/0/points3D.ply` is the SPARSE cloud in PLY clothing — if
that is what you are pointing at, use colmap_parser.py on points3D.bin
instead, which also gives you reprojection error and track length to filter on.

Usage:
    python -m aerosplat.io.ply_parser --dataset zaporizhzhia
    python -m aerosplat.io.ply_parser \
        --input data/colmap_output/dense/0/fused.ply \
        --output data/zaporizhzhia_dense.pt --mad-k 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from aerosplat.io.cloud_io import filter_cloud, read_ply, save_cloud

from aerosplat.paths import PROJECT_DIR


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=None,
                   help="Dataset name; fills in the conventional paths.")
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--mad-k", type=float, default=0.0,
                   help="Robust per-axis outlier rejection, in MADs. Try 20. "
                        "0 disables.")
    p.add_argument("--max-points", type=int, default=None,
                   help="Randomly subsample to at most N points. Useful to "
                        "match gs_trainer_v3's --max-gaussians budget.")
    p.add_argument("--recenter", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p


def resolve_paths(args) -> tuple[Path, Path]:
    if args.input and args.output:
        return args.input, args.output
    if not args.dataset:
        raise SystemExit("Provide either --dataset or both --input and --output")
    ds = args.dataset
    inp = args.input or (PROJECT_DIR / "data" / "colmap_output" / "dense" / "0"
                         / "fused.ply")
    out = args.output or (PROJECT_DIR / "data" / f"{ds}_dense.pt")
    return inp, out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inp, out = resolve_paths(args)

    if not inp.is_file():
        print(f"ERROR: {inp} not found — run dense_runner.py first",
              file=sys.stderr)
        return 1

    print(f"Reading dense PLY: {inp}")
    xyz, rgb, hdr = read_ply(inp)
    print(f"  format: {hdr.fmt}")
    print(f"  vertex properties: "
          f"{', '.join(n for n, _ in hdr.props)}")
    print(f"  {len(xyz):,} vertices")

    if rgb is None:
        print("  warning: no colour properties found — filling mid grey. "
              "The GS trainer will have to learn all colour from scratch.")
        rgb = np.full((len(xyz), 3), 128, dtype=np.uint8)

    xyz, rgb, stats = filter_cloud(xyz, rgb, mad_k=args.mad_k,
                                   max_points=args.max_points, seed=args.seed)

    save_cloud(out, xyz, rgb, source=inp, stats=stats, recenter=args.recenter,
               extra_meta={"kind": "colmap_dense",
                           "ply_properties": [n for n, _ in hdr.props]})
    print("next: python -m aerosplat.gs.trainer")
    return 0


from aerosplat.logging_utils import fix_console_encoding as _fix_console_encoding


if __name__ == "__main__":
    _fix_console_encoding()
    raise SystemExit(main())