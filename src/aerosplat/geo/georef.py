"""Pipeline stage `georef`: map a run's products from sphere ENU to WGS84 Site ENU.

Earth Studio runs reconstruct in sphere ENU (aerosplat.geo.sphere_enu, ADR-0006
amended 2026-10-15); the geo stage records the affine map sphere ENU -> WGS84
Site ENU in the registration report. This stage applies it to the cloud .pt
and the splat .pt and writes the results next to them (`*_wgs84.pt`). Those
are what the catalog registers as the run's `cloud` / `splat` and what the
server serves; the inputs stay in the reconstruction frame, which is the one
consistent with the COLMAP cameras (eval, resume).

Points map as x -> A x + b; Gaussians exactly as mu -> A mu + b,
Sigma -> A Sigma A^T (aerosplat.geo.affine). Opacity and `rgb` are unchanged.
View-dependent colour (`sh`, gsplat trainer) is a function of the viewing
direction, which the map turns too: the SH coefficients are rotated by the
rotation part of A (polar decomposition). A's remaining stretch (sphere ->
WGS84 on dataset_v6: axes 1.0026 / 0.9987 / 1) bends directions by <= 0.11°
and is ignored.

    python -m aerosplat.geo.georef --registration reports/registration.json \\
        --cloud gs/dense.pt gs/dense_wgs84.pt --splat gs/splat.pt gs/splat_wgs84.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aerosplat.geo.affine import AffineMap
from aerosplat.geo.sphere_enu import EARTH_MODEL
from aerosplat.gs.sh import rotate_sh

# Key a mapped .pt carries: which frame it is in and how it got there.
FRAME_KEY = "frame"
SITE_FRAME = "WGS84 Site ENU"


def load_affine(report_path: Path) -> tuple[AffineMap, dict[str, Any]]:
    """The sphere ENU -> WGS84 Site ENU map recorded by the geo stage."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    em = report.get("earth_model")
    model = em.get("id") if isinstance(em, dict) else None
    block = report.get("sphere_enu_to_site_enu")
    if model != EARTH_MODEL or not block:
        raise SystemExit(f"❌ {report_path}: no sphere ENU -> WGS84 Site ENU map "
                         f"(earth_model={model!r}); this run's known poses predate it -- "
                         f"re-run from the 'geo' stage")
    return AffineMap.from_dict(block), block


def frame_info(block: dict[str, Any], report_path: Path, source: Path) -> dict[str, Any]:
    return {"crs": SITE_FRAME, "earth_model": EARTH_MODEL, "mapped_from": "sphere ENU",
            "A": block["A"], "b": block["b"], "residual_max_m": block["residual_max_m"],
            "registration": str(report_path), "source": str(source)}


def _load(src: Path) -> dict[str, Any]:
    data = torch.load(src, map_location="cpu", weights_only=False)
    if FRAME_KEY in data:
        raise SystemExit(f"❌ {src} is already in {data[FRAME_KEY].get('crs')} -- "
                         f"pass the reconstruction-frame file")
    return data


def _save(data: dict[str, Any], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    torch.save(data, tmp)
    tmp.replace(dst)   # atomic, like cloud_io.save_cloud


def _tensor(a: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).to(like.dtype)


def transform_cloud(src: Path, dst: Path, affine: AffineMap, info: dict[str, Any]) -> int:
    data = _load(src)
    xyz = data["xyz"].double().numpy()
    meta = dict(data.get("meta") or {})
    if meta.get("recentered"):
        # Stored relative to the centroid: offsets map linearly, the centroid as a point.
        new = affine.apply_linear(xyz)
    else:
        new = affine.apply(xyz)
    if "centroid" in meta:
        meta["centroid"] = affine.apply(meta["centroid"]).tolist()
    if len(new):
        lo, hi = np.nanmin(new, axis=0), np.nanmax(new, axis=0)
        meta.update(bbox_min=lo.tolist(), bbox_max=hi.tolist(),
                    extent=float(np.linalg.norm(hi - lo)))
    data["xyz"] = _tensor(new, data["xyz"])
    data["meta"] = meta
    data[FRAME_KEY] = info
    _save(data, dst)
    return int(len(new))


def transform_splat(src: Path, dst: Path, affine: AffineMap, info: dict[str, Any]) -> int:
    data = _load(src)
    contract = data.get("activations") or {}
    if "scale_is_log" not in contract:
        raise SystemExit(f"❌ {src}: no activation contract (ADR-0002) -- cannot tell "
                         f"whether its scales are log-space")
    xyz = data["xyz"].double().numpy()
    scale = data["scale"].double().numpy()
    rot = data["rotation"].double().numpy()
    if contract["scale_is_log"]:
        scale = np.exp(scale)
    # Diverged Gaussians (NaN/inf, zero quaternion) keep their values; the
    # server drops non-finite rows anyway, and SVD would fail on them.
    ok = (np.isfinite(xyz).all(1) & np.isfinite(scale).all(1) & np.isfinite(rot).all(1)
          & (np.linalg.norm(rot, axis=1) > 0))
    new_xyz, new_scale, new_rot = affine.apply(xyz), scale.copy(), rot.copy()
    _, new_scale[ok], new_rot[ok] = affine.apply_gaussians(xyz[ok], scale[ok], rot[ok])
    if contract["scale_is_log"]:
        new_scale = np.log(new_scale)
    data.update(xyz=_tensor(new_xyz, data["xyz"]), scale=_tensor(new_scale, data["scale"]),
                rotation=_tensor(new_rot, data["rotation"]))
    sh = data.get("sh")
    if sh is not None and sh.shape[1] > 1:      # DC alone is direction-independent
        data["sh"] = _tensor(rotate_sh(sh.double().numpy(), affine.rotation()), sh)
    data[FRAME_KEY] = info
    _save(data, dst)
    return int(len(new_xyz))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registration", type=Path, required=True,
                    help="registration report of the run's geo stage")
    ap.add_argument("--cloud", type=Path, nargs=2, metavar=("IN", "OUT"))
    ap.add_argument("--splat", type=Path, nargs=2, metavar=("IN", "OUT"))
    args = ap.parse_args(argv)

    affine, block = load_affine(args.registration)
    diag = ", ".join(f"{affine.A[i, i]:.6f}" for i in range(3))
    print(f"sphere ENU -> {SITE_FRAME}: diag(A)={diag}, b={np.round(affine.b, 6).tolist()}, "
          f"rotation {affine.rotation_deg():.2e}°, fit residual {block['residual_max_m'] * 1000:.2f} mm")
    done = 0
    for kind, pair, fn in (("cloud", args.cloud, transform_cloud),
                           ("splat", args.splat, transform_splat)):
        if not pair:
            continue
        src, dst = pair
        if not src.is_file():
            print(f"  {kind}: {src} not found -- skipped")
            continue
        n = fn(src, dst, affine, frame_info(block, args.registration, src))
        print(f"  {kind}: {n:,} -> {dst}")
        done += 1
    if not done:
        raise SystemExit("❌ nothing to map: neither the cloud nor the splat .pt exists")
    return 0


if __name__ == "__main__":
    from aerosplat.logging_utils import fix_console_encoding

    fix_console_encoding()
    raise SystemExit(main())
