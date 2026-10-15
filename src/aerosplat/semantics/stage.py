"""Pipeline stage `semantics`: dense cloud -> terrain, buildings, facades.

    python -m aerosplat.semantics.stage --cloud gs/dense_wgs84.pt \\
        --sparse sfm/sparse/0 --registration reports/registration.json \\
        --out-dir semantics

Runs after `georef`, because everything it writes is in the Site ENU frame that
authored features are stored in (ADR-0006). The COLMAP cameras live in the
reconstruction frame, so they are carried over by the same affine map.

Outputs in `--out-dir`:
    semantics.json   terrain raster header, buildings, facades, parameters
    terrain.npz      dsm, dtm, point count, building label map
    coverage.npz     per-facade view-count grid, keyed by facade id
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from aerosplat.geo.affine import AffineMap
from aerosplat.io.colmap_model import CameraPose, read_model
from aerosplat.logging_utils import fix_console_encoding, get_logger
from aerosplat.semantics.buildings import find_buildings
from aerosplat.semantics.coverage import facade_coverage, transform_pose
from aerosplat.semantics.facades import facades_of_building
from aerosplat.semantics.models import (COVERAGE_NPZ, SEMANTICS_JSON, TERRAIN_NPZ, Facade,
                                        SemanticsResult)
from aerosplat.semantics.terrain import build_terrain

log = get_logger()


def load_cloud_xyz(path: Path) -> np.ndarray:
    """(N, 3) ENU points from a cloud .pt. torch is imported here, not at module level."""
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    if "xyz" not in data:
        raise SystemExit(f"❌ {path}: no 'xyz' — not a cloud .pt")
    return data["xyz"].numpy().astype(np.float64)


def load_cameras(sparse_dir: Optional[Path], registration: Optional[Path]) -> list[CameraPose]:
    """COLMAP poses mapped into the Site ENU frame; empty when there is no model."""
    if sparse_dir is None or not Path(sparse_dir).is_dir():
        return []
    cams = read_model(sparse_dir)
    affine = AffineMap.identity()
    if registration is not None and Path(registration).is_file():
        block = json.loads(Path(registration).read_text(encoding="utf-8")) \
            .get("sphere_enu_to_site_enu")
        if block:
            affine = AffineMap.from_dict(block)
    if np.allclose(affine.A, np.eye(3)) and np.allclose(affine.b, 0):
        return cams
    return [transform_pose(c, affine) for c in cams]


def run_semantics(cloud: Path, out_dir: Path, *, sparse: Optional[Path] = None,
                  registration: Optional[Path] = None, res_m: float = 1.0,
                  min_building_height_m: float = 2.5, min_building_area_m2: float = 40.0,
                  facade_inlier_m: float = 0.8, facade_min_length_m: float = 5.0,
                  facade_min_height_m: float = 3.0, facade_min_points: int = 40,
                  max_facades_per_building: int = 12, coverage_cell_m: float = 0.5,
                  coverage_max_angle_deg: float = 60.0, coverage_max_gsd_m: float = 0.05,
                  max_buildings: int = 400, seed: int = 0) -> SemanticsResult:
    xyz = load_cloud_xyz(cloud)
    log.info(f"  Хмара: {len(xyz):,} точок")

    terrain = build_terrain(xyz, res_m=res_m)
    m = terrain.model
    log.info(f"  Рельєф: {m.rows}×{m.cols} клітин по {res_m} м, "
             f"{m.filled_cells:,} із точками ({100 * m.filled_cells / (m.rows * m.cols):.0f}%)")

    buildings, labels = find_buildings(terrain, min_height_m=min_building_height_m,
                                       min_area_m2=min_building_area_m2)
    warnings: list[str] = []
    if len(buildings) > max_buildings:
        warnings.append(f"знайдено {len(buildings)} будівель, залишено {max_buildings} найбільших")
        buildings = buildings[:max_buildings]
    log.info(f"  Будівель: {len(buildings)}")

    facades: list[Facade] = []
    for n, b in enumerate(buildings, start=1):
        found = facades_of_building(
            xyz, terrain, labels, b, n, inlier_m=facade_inlier_m,
            min_points=facade_min_points, min_length_m=facade_min_length_m,
            min_height_m=facade_min_height_m, max_facades=max_facades_per_building,
            seed=seed + n)
        b.facade_ids = [f.id for f in found]
        facades += found
    log.info(f"  Фасадів: {len(facades)}")

    cameras = load_cameras(sparse, registration)
    grids: dict[str, np.ndarray] = {}
    if cameras:
        for i, f in enumerate(facades):
            summary, grid = facade_coverage(f, cameras, cell_m=coverage_cell_m,
                                            max_angle_deg=coverage_max_angle_deg,
                                            max_gsd_m=coverage_max_gsd_m)
            facades[i] = f.model_copy(update={"coverage": summary})
            grids[f.id] = grid
        cov = [f.coverage for f in facades if f.coverage]
        if cov:
            best = [c.best_gsd_cm for c in cov if c.best_gsd_cm is not None]
            log.info(f"  Покриття ({len(cameras)} кадрів): видно медіана "
                     f"{100 * float(np.median([c.visible_fraction for c in cov])):.0f}%, "
                     f"з потрібною деталізацією "
                     f"{100 * float(np.median([c.fraction for c in cov])):.0f}% "
                     f"(поріг {100 * coverage_max_gsd_m:.0f} см/px, "
                     f"найкраще {min(best):.0f} см/px)" if best else
                     f"  Покриття ({len(cameras)} кадрів): жоден фасад не видно")
            coarse = [c for c in cov if c.too_coarse]
            if best and len(coarse) > 0.5 * len(cov):
                warnings.append(
                    f"{len(coarse)} із {len(cov)} фасадів видно, але не різкіше ніж "
                    f"{min(best):.0f} см/px (потрібно {100 * coverage_max_gsd_m:.0f}): "
                    f"зйомка задалеко, щоб читати отвори — потрібен ближчий обліт")
                log.warning(f"  ⚠️  {warnings[-1]}")
    else:
        warnings.append("камер немає (sfm/sparse не знайдено) — карту покриття не побудовано")
        log.warning("  ⚠️  Камер не знайдено — без карти покриття")

    result = SemanticsResult(
        terrain=m, buildings=buildings, facades=facades, warnings=warnings,
        params={"res_m": res_m, "min_building_height_m": min_building_height_m,
                "min_building_area_m2": min_building_area_m2,
                "facade_inlier_m": facade_inlier_m,
                "facade_min_length_m": facade_min_length_m,
                "facade_min_height_m": facade_min_height_m,
                "coverage_cell_m": coverage_cell_m,
                "coverage_max_angle_deg": coverage_max_angle_deg,
                "coverage_max_gsd_m": coverage_max_gsd_m,
                "n_cameras": len(cameras), "seed": seed})

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SEMANTICS_JSON).write_text(
        result.model_dump_json(indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / TERRAIN_NPZ, labels=labels.astype(np.int32),
                        **terrain.to_npz())
    if grids:
        np.savez_compressed(out_dir / COVERAGE_NPZ, **grids)
    log.info(f"  Збережено: {out_dir / SEMANTICS_JSON}")
    return result


def main(argv: Optional[list[str]] = None) -> int:
    fix_console_encoding()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cloud", type=Path, required=True, help="dense cloud .pt у Site ENU")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sparse", type=Path, default=None, help="COLMAP sparse (для покриття)")
    ap.add_argument("--registration", type=Path, default=None,
                    help="reports/registration.json — афінне для поз камер")
    ap.add_argument("--res", type=float, default=1.0, dest="res_m")
    ap.add_argument("--min-building-height", type=float, default=2.5,
                    dest="min_building_height_m")
    ap.add_argument("--min-building-area", type=float, default=40.0, dest="min_building_area_m2")
    ap.add_argument("--facade-inlier", type=float, default=0.8, dest="facade_inlier_m")
    ap.add_argument("--facade-min-length", type=float, default=5.0, dest="facade_min_length_m")
    ap.add_argument("--facade-min-height", type=float, default=3.0, dest="facade_min_height_m")
    ap.add_argument("--facade-min-points", type=int, default=40)
    ap.add_argument("--max-facades-per-building", type=int, default=12)
    ap.add_argument("--coverage-cell", type=float, default=0.5, dest="coverage_cell_m")
    ap.add_argument("--coverage-max-angle", type=float, default=60.0,
                    dest="coverage_max_angle_deg")
    ap.add_argument("--coverage-max-gsd", type=float, default=0.05, dest="coverage_max_gsd_m",
                    help="м/піксель, за якої деталі фасаду ще читаються")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    kw: dict[str, Any] = {k: v for k, v in vars(args).items() if k not in ("cloud", "out_dir")}
    run_semantics(args.cloud, args.out_dir, **kw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
