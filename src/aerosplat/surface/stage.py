"""Pipeline stage `occupancy`: dense cloud + semantics terrain -> voxel grid + distance field.

    python -m aerosplat.surface.stage --cloud gs/dense_wgs84.pt \\
        --semantics-dir semantics --out-dir occupancy \\
        --fused dense/fused.ply --sparse dense/sparse --registration reports/registration.json

Runs after `semantics`, on the same cloud: the terrain raster it wrote (semantics.json,
terrain.npz) supplies the DSM for empty voxel columns and the DTM for the ground, and its
lattice is the one the voxel columns subdivide.

With `--fused`, camera rays carve the grid (aerosplat.surface.carving): the COLMAP dense
workspace's fused.ply, its visibility (fused.ply.vis) and the model its indices count
(`--sparse`, default: sparse/ next to fused.ply), mapped to Site ENU by `--registration`
like the semantics cameras. Missing workspace files leave a 2.5-D grid and a warning;
files that contradict each other or the cloud stop the stage.

Outputs in `--out-dir`:
    occupancy.npz    state, esdf_cm (rows, cols, layers); column_top, column_source (rows, cols)
    occupancy.json   grid header, bounds, ESDF convention, statistics, parameters, warnings
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from aerosplat.logging_utils import fix_console_encoding, get_logger
from aerosplat.semantics.models import SEMANTICS_JSON, TERRAIN_NPZ, SemanticsResult
from aerosplat.surface.carving import CameraRays, load_colmap_rays, registration_affine
from aerosplat.surface.esdf import DEFAULT_TILE_VOXELS
from aerosplat.surface.models import EsdfInfo, OccupancyReport, parse_bbox
from aerosplat.surface.occupancy import UNKNOWN_MODES, build_occupancy, write_occupancy

log = get_logger()


def load_terrain(semantics_dir: Path):
    """(TerrainModel, dsm, dtm) from what the semantics stage wrote."""
    sem_json, terrain_npz = semantics_dir / SEMANTICS_JSON, semantics_dir / TERRAIN_NPZ
    if not sem_json.is_file() or not terrain_npz.is_file():
        raise SystemExit(f"❌ {semantics_dir}: немає {SEMANTICS_JSON} / {TERRAIN_NPZ} — "
                         f"спершу стадія `semantics`")
    result = SemanticsResult(**json.loads(sem_json.read_text(encoding="utf-8")))
    with np.load(terrain_npz) as z:
        return result.terrain, z["dsm"].astype(np.float64), z["dtm"].astype(np.float64)


def load_rays(fused: Optional[Path], sparse: Optional[Path],
              registration: Optional[Path]) -> tuple[Optional[CameraRays], Optional[str]]:
    """(rays, None), or (None, why not) when the workspace lacks what carving needs."""
    if fused is None:
        return None, "не задано --fused"
    fused = Path(fused)
    sparse = Path(sparse) if sparse is not None else fused.parent / "sparse"
    vis = fused.with_name(fused.name + ".vis")
    for need in (fused, vis, sparse):
        if not need.exists():
            return None, f"немає {need}"
    try:
        return load_colmap_rays(fused, sparse, registration_affine(registration)), None
    except (ValueError, FileNotFoundError, KeyError) as e:
        raise SystemExit(f"❌ occupancy: видимість COLMAP не можна використати: {e}") from None


def run_occupancy(cloud: Path, semantics_dir: Path, out_dir: Path, *, res_m: float = 0.5,
                  headroom_m: float = 10.0, max_distance_m: float = 20.0,
                  fill_radius_m: float = 5.0, close_radius_m: float = 1.0,
                  assumed_margin_m: float = 1.0, unknown: str = "skyline", skyline_quantile: float = 0.9999,
                  bbox: Optional[str] = None, max_voxels: int = 400_000_000,
                  signed: bool = False, tile_voxels: int = DEFAULT_TILE_VOXELS,
                  fused: Optional[Path] = None, sparse: Optional[Path] = None,
                  registration: Optional[Path] = None, carve: bool = True,
                  carve_min_views: int = 2, carve_margin_m: float = 1.0,
                  carve_under: bool = True) -> OccupancyReport:
    from aerosplat.semantics.stage import load_cloud_xyz      # torch, only to read the .pt

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    terrain, dsm, dtm = load_terrain(semantics_dir)
    xyz = load_cloud_xyz(cloud)
    timings["load"] = time.perf_counter() - t0
    log.info(f"  Хмара: {len(xyz):,} точок; рельєф {terrain.rows}×{terrain.cols} по {terrain.res_m} м")
    rays, no_rays = load_rays(fused, sparse, registration) if carve else (None, "--no-carve")
    timings["rays"] = time.perf_counter() - t0 - timings["load"]
    if rays is not None:
        v = rays.info
        log.info(f"  Видимість: {v['pairs']:,} пар точка–камера, {v['images']} камер, "
                 f"порядок кадрів {v['image_order']}, поза кадром {v['pairs_outside_image']:,}")

    params: dict[str, Any] = {
        "res_m": res_m, "headroom_m": headroom_m, "max_distance_m": max_distance_m,
        "fill_radius_m": fill_radius_m, "close_radius_m": close_radius_m,
        "assumed_margin_m": assumed_margin_m,
        "unknown": unknown, "skyline_quantile": skyline_quantile, "bbox": bbox,
        "max_voxels": max_voxels, "signed": signed, "carve": rays is not None,
        "carve_min_views": carve_min_views, "carve_margin_m": carve_margin_m,
        "carve_under": carve_under}
    t1 = time.perf_counter()
    try:
        grid, stats, warnings = build_occupancy(
            xyz, terrain, dsm, dtm, res_m=res_m, headroom_m=headroom_m,
            max_distance_m=max_distance_m, fill_radius_m=fill_radius_m,
            close_radius_m=close_radius_m, assumed_margin_m=assumed_margin_m, unknown=unknown,
            skyline_quantile=skyline_quantile, bbox=parse_bbox(bbox) if bbox else None,
            max_voxels=max_voxels, signed=signed, tile_voxels=tile_voxels, rays=rays,
            carve_min_views=carve_min_views, carve_margin_m=carve_margin_m,
            carve_under_surface=carve_under)
    except ValueError as e:
        raise SystemExit(f"❌ occupancy: {e}") from None
    if no_rays is not None and carve:
        warnings.append(f"простір не вирізано променями камер ({no_rays}): вільне лише над "
                        f"поверхнею (2.5D), під навісами — заблоковано")
    timings["build"] = time.perf_counter() - t1
    g = grid.spec
    log.info(f"  Сітка: {g.rows}×{g.cols}×{g.layers} = {g.voxels:,} вокселів по {g.res_m} м, "
             f"z {g.z0:.1f}…{g.z1:.1f} м")
    log.info(f"  Вільно {100 * stats.free / g.voxels:.1f}%, зайнято "
             f"{100 * stats.occupied / g.voxels:.1f}%, невідомо {100 * stats.unknown / g.voxels:.1f}%; "
             f"колонки: виміряні {100 * stats.observed_fraction:.0f}%, "
             f"з припущенням {100 * stats.assumed_fraction:.0f}%")
    if stats.carving is not None:
        c = stats.carving
        log.info(f"  Промені: використано {c.rays_used:,} з {c.rays_total:,} "
                 f"(крізь зайняте {c.rays_through_occupied:,}); вирізано {c.voxels_carved:,} "
                 f"вокселів (під поверхнею {c.carved_under_surface:,}), повернуто ерозією "
                 f"{c.voxels_eroded:,}; колонок CARVED {sum(c.columns_upgraded.values()):,} "
                 f"({c.seconds_total:.1f} s)")
    for w in warnings:
        log.warning(f"  ⚠️  {w}")

    report = OccupancyReport(
        grid=g, bounds=g.bounds(),
        esdf=EsdfInfo(dtype=str(grid.esdf_cm.dtype), signed=signed, max_distance_m=max_distance_m),
        stats=stats, params=params,
        inputs={"cloud": str(cloud), "semantics_dir": str(semantics_dir),
                "n_points": int(len(xyz)), "terrain_res_m": terrain.res_m,
                "fused": None if fused is None else str(fused),
                "sparse": None if sparse is None else str(sparse),
                "registration": None if registration is None else str(registration)},
        warnings=warnings, timings_s={k: round(v, 2) for k, v in timings.items()})
    t2 = time.perf_counter()
    write_occupancy(out_dir, grid, report)
    log.info(f"  Збережено: {out_dir} (запис {time.perf_counter() - t2:.1f} s, "
             f"усього {time.perf_counter() - t0:.1f} s)")
    return report


def main(argv: Optional[list[str]] = None) -> int:
    fix_console_encoding()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cloud", type=Path, required=True, help="dense cloud .pt у Site ENU")
    ap.add_argument("--semantics-dir", type=Path, required=True,
                    help="вихід стадії semantics (semantics.json, terrain.npz)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--res", type=float, default=0.5, dest="res_m",
                    help="розмір вокселя, м (має ділити клітину рельєфу)")
    ap.add_argument("--headroom", type=float, default=10.0, dest="headroom_m",
                    help="висота сітки над лінією дахів, м")
    ap.add_argument("--max-distance", type=float, default=20.0, dest="max_distance_m",
                    help="обрізання ESDF, м")
    ap.add_argument("--fill-radius", type=float, default=5.0, dest="fill_radius_m",
                    help="найбільша дірка в хмарі, яку заповнюємо сусідами, м")
    ap.add_argument("--close-radius", type=float, default=1.0, dest="close_radius_m",
                    help="закриття поверхні: ями вужчі за ~2 радіуси піднімаються до краю, м")
    ap.add_argument("--assumed-margin", type=float, default=1.0, dest="assumed_margin_m",
                    help="запас над будь-якою припущеною висотою поверхні, м")
    ap.add_argument("--unknown", choices=UNKNOWN_MODES, default="skyline",
                    help="колонки без даних: невідомі до лінії дахів (skyline) чи до неба (sky)")
    ap.add_argument("--skyline-quantile", type=float, default=0.9999)
    ap.add_argument("--bbox", default=None,
                    help="xmin,ymin,xmax,ymax у Site ENU, м; писати --bbox=-100,-80,120,90")
    ap.add_argument("--max-voxels", type=int, default=400_000_000)
    ap.add_argument("--signed", action="store_true",
                    help="ESDF зі знаком (глибина всередині перешкод) — удвічі довше")
    ap.add_argument("--fused", type=Path, default=None,
                    help="dense/fused.ply COLMAP (поруч fused.ply.vis): промені для вирізання")
    ap.add_argument("--sparse", type=Path, default=None,
                    help="модель робочої теки dense (за замовчуванням sparse/ поруч з fused.ply)")
    ap.add_argument("--registration", type=Path, default=None,
                    help="звіт реєстрації: афінне перетворення кадру реконструкції в Site ENU")
    ap.add_argument("--no-carve", dest="carve", action="store_false",
                    help="не вирізати простір променями: лише 2.5D")
    ap.add_argument("--carve-min-views", type=int, default=2,
                    help="скільки різних камер мають перетнути воксель")
    ap.add_argument("--carve-margin", type=float, default=1.0, dest="carve_margin_m",
                    help="останні N м кожного променя не вирізаються (шум глибини)")
    ap.add_argument("--no-carve-under", dest="carve_under", action="store_false",
                    help="не відкривати простір під виміряною поверхнею (навіси, мости)")
    args = ap.parse_args(argv)

    kw: dict[str, Any] = {k: v for k, v in vars(args).items()
                          if k not in ("cloud", "semantics_dir", "out_dir")}
    run_occupancy(args.cloud, args.semantics_dir, args.out_dir, **kw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
