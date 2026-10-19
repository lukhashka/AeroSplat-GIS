"""`aerosplat export`: GIS outputs of one catalog run, recorded as the run's products.

    aerosplat export --site default/dataset_1910                       # latest run, all kinds
    aerosplat export --site dataset_1910 --run 20261019-105253 --kinds dsm,dtm,las
    aerosplat export --site dataset_1910 --kinds ortho --gsd 0.1 --crs EPSG:32618

Outputs go to `runs/<run>/exports/` (or `--out-dir`): ortho.tif, dsm.tif, dtm.tif,
ndsm.tif (COG), cloud.laz, splat.ply and `exports.json` -- the manifest with the CRS, the
georeference error budget and per-file details. Each file is registered in the catalog
as product `export_<kind>`, the manifest as `exports` (Catalog.register_products: the
run's status and finish time are not touched, so the served run does not change).

Refusals (exit code 2, nothing written): the run is not finished successfully, has no
georeference (`scene-meta.metric` would be null), or its products are not in WGS84 Site
ENU; the CRS is unusable; the area is too large for one grid. `ortho` needs the GPU
(gsplat, on Windows via tools\\gsplat_env.bat); the other kinds run on the CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from aerosplat.export.georef import (DEFAULT_CRS, SITE_FRAME, ExportRefused, SiteProjection,
                                     georeference_budget)
from aerosplat.logging_utils import fix_console_encoding
from aerosplat.store.catalog import Catalog, CatalogError
from aerosplat.store.locking import file_lock
from aerosplat.store.models import Run, RunStatus, Site
from aerosplat.store.scenes import CLOUD, REGISTRATION, SPLAT, run_metric, scene_run

KINDS = ("ortho", "dsm", "dtm", "ndsm", "las", "ply")
DEFAULT_KINDS = KINDS
FILENAMES = {"ortho": "ortho.tif", "dsm": "dsm.tif", "dtm": "dtm.tif", "ndsm": "ndsm.tif",
             "las": "cloud.laz", "ply": "splat.ply"}
EXPORTS_DIR = "exports"
MANIFEST = "exports.json"
PRODUCT_PREFIX = "export_"
MANIFEST_PRODUCT = "exports"
LOCK_TIMEOUT_S = 10.0
SCHEMA = 1

# Called as on_event(kind, state, details): state is started | done | failed.
EventFn = Callable[[str, str, dict[str, Any]], None]


@dataclass
class ExportOptions:
    crs: str = DEFAULT_CRS
    gsd_m: float = 0.25                    # orthophoto pixel, CRS metres
    dem_px_m: Optional[float] = None       # terrain pixel; default: the semantics raster's
    tile_px: int = 2048
    cog: bool = True
    las_source: str = "cloud"              # "cloud" | "splat"
    las_ext: str = ".laz"                  # ".laz" | ".las"
    max_gap_m: float = 25.0                # DTM: nodata farther than this from a measured cell
    sh_degree: Optional[int] = None        # ortho render; None = all bands the scene has
    area_enu: Optional[tuple[float, float, float, float]] = None   # ortho: e0, n0, e1, n1
    out_dir: Optional[Path] = None
    register: bool = True                  # record products in the catalog

    def describe(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k not in ("out_dir",)}
        d["area_enu"] = list(self.area_enu) if self.area_enu else None
        return d


@dataclass
class ExportTarget:
    """A run that may be exported: checked, with its georeference resolved."""

    catalog: Catalog
    site: Site
    run: Run
    run_dir: Path
    metric: dict[str, Any]
    report: dict[str, Any]
    budget: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def resolve(cls, catalog: Catalog, site_ref: str, *, epoch_id: Optional[str] = None,
                run_id: Optional[str] = None) -> "ExportTarget":
        try:
            site = catalog.find_site(site_ref)
        except CatalogError as e:
            raise ExportRefused(str(e)) from None
        if run_id is None:
            run = scene_run(catalog, site)
            if run is None:
                raise ExportRefused(f"{site.key} has no finished run with a scene to export")
        else:
            runs = [r for r in catalog.list_runs(site, epoch_id) if r.id == run_id]
            if not runs:
                raise ExportRefused(f"{site.key} has no run '{run_id}'"
                                    + (f" in epoch '{epoch_id}'" if epoch_id else ""))
            if len(runs) > 1:
                raise ExportRefused(f"run '{run_id}' exists in several epochs of {site.key}: "
                                    f"give --epoch")
            run = runs[0]
        return cls.for_run(catalog, site, run)

    @classmethod
    def for_run(cls, catalog: Catalog, site: Site, run: Run) -> "ExportTarget":
        if run.status != RunStatus.ok:
            raise ExportRefused(f"run {run.id} is '{run.status.value}', not ok: only a "
                                f"finished run is exported")
        metric = run_metric(catalog, site, run)
        if metric is None:
            raise ExportRefused(
                f"run {run.id} of {site.key} has no georeference (metric is null: "
                f"earth_model={run.earth_model}, georeferenced={run.georeferenced}). Its "
                f"coordinates are an arbitrary reconstruction frame; a GIS file from it "
                f"would be placed nowhere in particular. Re-run with known poses "
                f"(Earth Studio JSON) or another georeference.")
        report_path = catalog.product_path(site, run, REGISTRATION)
        report: dict[str, Any] = {}
        if report_path is not None and report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except ValueError:
                raise ExportRefused(f"{report_path} is not valid JSON: the run's "
                                    f"georeference cannot be stated") from None
        t = cls(catalog=catalog, site=site, run=run,
                run_dir=catalog.run_dirs(site, run.epoch_id, run.id).root,
                metric=metric, report=report)
        t.budget = georeference_budget(earth_model=run.earth_model, metric=metric,
                                       report=report)
        return t

    def product(self, kind: str) -> Path:
        path = self.catalog.product_path(self.site, self.run, kind)
        if path is None or not path.is_file():
            raise ExportRefused(f"run {self.run.id} has no '{kind}' product on disk")
        if self.run.earth_model == "sphere-enu+affine" and not path.stem.endswith("_wgs84"):
            raise ExportRefused(f"{path.name} is not the georef stage's WGS84 Site ENU file "
                                f"(earth_model sphere-enu+affine): refusing to export a "
                                f"reconstruction-frame product")
        return path


# ── loading run products (torch only here) ───────────────────────────────
def load_pt(path: Path, earth_model: str) -> dict[str, Any]:
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    frame = (data.get("frame") or {}).get("crs")
    if earth_model == "sphere-enu+affine" and frame != SITE_FRAME:
        raise ExportRefused(f"{path} is in {frame or 'an unmarked frame'}, not {SITE_FRAME}")
    return data


def cloud_points(data: dict[str, Any]) -> tuple[np.ndarray, Optional[np.ndarray], dict[str, Any]]:
    xyz = data["xyz"].double().numpy()
    meta = dict(data.get("meta") or {})
    if meta.get("recentered"):
        xyz = xyz + np.asarray(meta["centroid"], np.float64)   # stored about the centroid
    rgb = data["rgb"].double().numpy() if "rgb" in data else None
    return xyz, rgb, meta


def splat_points(data: dict[str, Any]) -> tuple[np.ndarray, Optional[np.ndarray]]:
    from aerosplat.gs.sh import sh_dc_to_rgb

    xyz = data["xyz"].double().numpy()
    if "sh" in data:
        rgb = sh_dc_to_rgb(data["sh"][:, 0, :].double().numpy())
    elif "rgb" in data:
        rgb = data["rgb"].double().numpy()
    else:
        rgb = None
    return xyz, rgb


def robust_area(xyz: np.ndarray) -> tuple[tuple[float, float, float, float], float, tuple[float, float]]:
    """ENU rectangle of a cloud (already outlier-filtered by the parse stage), its median
    height and its height range."""
    ok = np.isfinite(xyz).all(axis=1)
    p = xyz[ok]
    if len(p) == 0:
        raise ExportRefused("the cloud has no finite points: cannot size the orthophoto")
    lo, hi = p.min(axis=0), p.max(axis=0)
    return ((float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])), float(np.median(p[:, 2])),
            (float(lo[2]), float(hi[2])))


# ── semantics (terrain) ──────────────────────────────────────────────────
def terrain_of(target: ExportTarget):
    from aerosplat.semantics.models import TERRAIN_NPZ
    from aerosplat.semantics.products import find_semantics

    found = find_semantics(target.catalog, target.site, run_id=target.run.id,
                           epoch_id=target.run.epoch_id)
    if found is None:
        raise ExportRefused(f"run {target.run.id} has no semantics: DSM/DTM/nDSM come from "
                            f"its `semantics` stage (aerosplat run --site {target.site.key} "
                            f"--run-id {target.run.id} --resume --from semantics)")
    raw = json.loads((found.dir / "semantics.json").read_text(encoding="utf-8"))
    if raw.get("frame", SITE_FRAME) != SITE_FRAME:
        raise ExportRefused(f"semantics of run {target.run.id} are in {raw['frame']}, "
                            f"not {SITE_FRAME}")
    if "frame" not in raw and target.run.earth_model == "sphere-enu+affine":
        # Written before semantics.json named its frame: trust only a stage input that
        # was the georef stage's output.
        inputs = _manifest_inputs(target.run_dir, "semantics")
        if not any(Path(p).stem.endswith("_wgs84") for p in inputs):
            raise ExportRefused(f"cannot confirm that the semantics of run {target.run.id} "
                                f"were computed in {SITE_FRAME}: re-run the semantics stage")
    npz = found.dir / TERRAIN_NPZ
    if not npz.is_file():
        raise ExportRefused(f"{npz} is missing")
    return found, npz


def _manifest_inputs(run_dir: Path, stage: str) -> list[str]:
    try:
        m = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    st = (m.get("stages") or {}).get(stage) or {}
    return [str(i.get("path", "")) for i in st.get("inputs") or [] if isinstance(i, dict)]


# ── manifest ─────────────────────────────────────────────────────────────
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(out_dir: Path) -> dict[str, Any]:
    path = out_dir / MANIFEST
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except ValueError:
        return {}


def _write_manifest(out_dir: Path, data: dict[str, Any]) -> Path:
    path = out_dir / MANIFEST
    tmp = path.with_name(f".{MANIFEST}.partial")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── the export ───────────────────────────────────────────────────────────
OrthoRendererFactory = Callable[[ExportTarget, Path], Any]


def engine_renderer(target: ExportTarget, splat: Path):
    """In-process GPU renderer pinned to exactly this run's splat (not the Site's latest)."""
    from aerosplat.render.engine import RenderEngine
    from aerosplat.store.scenes import SceneSource

    engine = RenderEngine(None, None, idle_s=1e9)
    name = f"{target.site.key}@{target.run.id}"
    engine.pin_scene(SceneSource(name=name, cloud="", splat=str(splat), metric=target.metric,
                                 kind="catalog", site=target.site.key, run=target.run.id))

    def render(req):
        out = engine.render(name, req)
        if out.meta.get("run") != target.run.id:
            raise ExportRefused(f"renderer drew run {out.meta.get('run')}, not {target.run.id}")
        return out.image, out.meta

    render.close = engine.unload_all          # type: ignore[attr-defined]
    return render


def export_run(target: ExportTarget, kinds: list[str], opts: ExportOptions, *,
               renderer_factory: Optional[OrthoRendererFactory] = None,
               on_event: Optional[EventFn] = None, log: Callable[[str], None] = print
               ) -> dict[str, Any]:
    bad = [k for k in kinds if k not in KINDS]
    if bad or not kinds:
        raise ExportRefused(f"unknown export kinds {bad}; known: {list(KINDS)}")
    if opts.las_source not in ("cloud", "splat") or opts.las_ext not in (".las", ".laz"):
        raise ExportRefused("las_source must be cloud|splat and las_ext .las|.laz")
    proj = SiteProjection(target.site.origin, opts.crs)
    # Absolute: the catalog reads a relative product path as relative to the run directory,
    # so a relative --projects-root would make registration miss every file.
    out_dir = (opts.out_dir or (target.run_dir / EXPORTS_DIR)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    emit = on_event or (lambda *a: None)
    log(f"Експорт {target.site.key} · run {target.run.id} → {out_dir}")
    log(f"  CRS: {proj.crs.name} (EPSG:{proj.epsg}); висоти — {proj.describe()['heights']}")
    fe = target.budget.get("frame_error_m")
    log(f"  Геоприв'язка: {target.run.earth_model}, похибка кадру "
        f"{'невідома' if fe is None else f'{fe:.4g} м'}")
    for w in target.budget.get("warnings", []):
        log(f"  ⚠️  {w}")

    stack = ExitStack()
    try:
        stack.enter_context(file_lock(out_dir / ".export.lock", timeout=LOCK_TIMEOUT_S))
    except TimeoutError:
        raise ExportRefused(f"another export of run {target.run.id} is writing {out_dir}") from None
    with stack:
        manifest = read_manifest(out_dir)
        manifest.update({"schema": SCHEMA, "site": target.site.key, "epoch": target.run.epoch_id,
                         "run": target.run.id, "georeference": target.budget})
        files = manifest.setdefault("files", {})
        done: dict[str, Path] = {}
        # The terrain rasters share one resampling grid, so they are one step.
        steps = [("ortho", ["ortho"])] if "ortho" in kinds else []
        dem = [k for k in ("dsm", "dtm", "ndsm") if k in kinds]
        steps += [("dem", dem)] if dem else []
        steps += [(k, [k]) for k in ("las", "ply") if k in kinds]
        for step, group in steps:
            for k in group:
                emit(k, "started", {})
            try:
                details, paths = _export_kind(step, group, target, proj, opts, out_dir,
                                              renderer_factory, log)
            except Exception as e:
                for k in group:
                    emit(k, "failed", {"error": str(e)})
                raise
            for k in group:
                path = paths[k]
                size = path.stat().st_size
                files[k] = {"path": _rel(path, target.run_dir), "file": path.name,
                            "bytes": size, "sha256": _sha256(path), "created": _now(),
                            "crs": {"name": proj.crs.name, "epsg": proj.epsg},
                            "options": opts.describe(), "details": details[k]}
                done[k] = path
                emit(k, "done", {"file": path.name, "bytes": size})
                log(f"  ✅ {k}: {path.name} ({size / 2 ** 20:.1f} MB)")
            # Written after every step: a later failure keeps what is already done.
            mpath = _write_manifest(out_dir, manifest)
            if opts.register and opts.out_dir is None:
                products: dict[str, Path] = {f"{PRODUCT_PREFIX}{k}": paths[k] for k in group}
                products[MANIFEST_PRODUCT] = mpath
                run = target.catalog.register_products(target.site, target.run.epoch_id,
                                                       target.run.id, products)
                missing = sorted(k for k in products if k not in run.products)
                if missing:          # Catalog skips paths it cannot find: never silently
                    raise ExportRefused(f"files written but not registered as products "
                                        f"{missing} of run {target.run.id}")
        return manifest


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _common_tags(target: ExportTarget, proj: SiteProjection, kind: str,
                 extra: dict[str, Any]) -> dict[str, Any]:
    from aerosplat.export.raster import TAG_EXPORT, TAG_GEOREF

    return {TAG_EXPORT: {"kind": kind, "site": target.site.key, "epoch": target.run.epoch_id,
                         "run": target.run.id, "created": _now(), **extra},
            TAG_GEOREF: {**target.budget, "crs": proj.describe()}}


def _export_kind(kind: str, group: list[str], target: ExportTarget, proj: SiteProjection,
                 opts: ExportOptions, out_dir: Path,
                 renderer_factory: Optional[OrthoRendererFactory], log: Callable[[str], None]
                 ) -> tuple[dict[str, Any], dict[str, Path]]:
    if kind == "ortho":
        from aerosplat.export.ortho import export_ortho
        from aerosplat.export.raster import NorthUpGrid

        splat = target.product(SPLAT)
        # The cloud sizes the mosaic and gives the ground height for the projection fit.
        cloud_xyz = cloud_points(load_pt(target.product(CLOUD), target.run.earth_model))[0]
        area, u_ref, u_range = robust_area(cloud_xyz)
        if opts.area_enu is not None:
            e0, n0, e1, n1 = opts.area_enu
            if not (e1 > e0 and n1 > n0):
                raise ExportRefused(f"area e0,n0,e1,n1 must be increasing, got {opts.area_enu}")
            area = (e0, n0, e1, n1)
        fit = proj.fit_plane((area[0], area[2]), (area[1], area[3]), u_range, u_ref=u_ref)
        grid = NorthUpGrid.covering(fit, (area[0], area[2]), (area[1], area[3]), opts.gsd_m)
        log(f"  ортофото: {grid.width}×{grid.height} px по {opts.gsd_m} м, "
            f"плитки {opts.tile_px} px")
        renderer = (renderer_factory or engine_renderer)(target, splat)
        path = out_dir / FILENAMES["ortho"]
        details_extra = {"source": f"3DGS orthographic render of {splat.name}",
                         "area_enu": list(area), "grid": grid.describe(),
                         "projection_fit": fit.describe()}
        try:
            info = export_ortho(path, renderer=renderer, proj=proj, fit=fit, grid=grid,
                                tile_px=opts.tile_px, cog=opts.cog, sh_degree=opts.sh_degree,
                                tags=_common_tags(target, proj, "ortho", details_extra),
                                on_tile=lambda i, n: log(f"    плитка {i}/{n}"))
        finally:
            close = getattr(renderer, "close", None)
            if close is not None:
                close()
        return {"ortho": {**details_extra, **info}}, {"ortho": path}

    if kind == "dem":
        from aerosplat.export.dem import export_dem, load_terrain, terrain_fit

        found, npz = terrain_of(target)
        model = found.result.terrain
        arrays = load_terrain(npz, model)
        fit = terrain_fit(proj, model, arrays["dsm"], arrays["dtm"])
        paths = {k: out_dir / FILENAMES[k] for k in group}
        extra = {"source": f"semantics/{npz.name} of run {found.run.id} "
                           f"({model.rows}×{model.cols} cells of {model.res_m} m)",
                 "semantics_params": found.result.params, "projection_fit": fit.describe()}
        summary = export_dem(paths, model=model, dsm=arrays["dsm"], dtm=arrays["dtm"],
                             count=arrays["count"], proj=proj, fit=fit, px=opts.dem_px_m,
                             max_gap_m=opts.max_gap_m, cog=opts.cog,
                             tags=_common_tags(target, proj, "+".join(group), extra))
        per = {k: {**extra, "grid": summary["grid"], "max_gap_m": summary["max_gap_m"],
                   "resampling": summary["resampling"], **summary[k]} for k in group}
        return per, paths

    if kind == "las":
        from aerosplat.export.pointcloud import export_las

        src_kind = CLOUD if opts.las_source == "cloud" else SPLAT
        src = target.product(src_kind)
        data = load_pt(src, target.run.earth_model)
        if opts.las_source == "cloud":
            xyz, rgb, meta = cloud_points(data)
            what = (f"run cloud product {src.name} (parse stage output: "
                    f"{meta.get('source', 'dense or sparse COLMAP cloud')})")
            stats = {k: meta[k] for k in ("num_points", "spacing", "stats") if k in meta}
        else:
            xyz, rgb = splat_points(data)
            what = f"3DGS Gaussian centres of {src.name} (DC colour; not survey points)"
            stats = {}
        path = out_dir / ("cloud" + opts.las_ext)
        metadata = {"source": what, "site": target.site.key, "run": target.run.id,
                    "georeference": target.budget, "source_stats": stats}
        info = export_las(path, enu=xyz, rgb=rgb, proj=proj, metadata=metadata)
        return {"las": {"source": what, "source_stats": stats, **info}}, {"las": path}

    if kind == "ply":
        from aerosplat.export.splat_ply import inria_arrays, write_inria_ply

        src = target.product(SPLAT)
        data = load_pt(src, target.run.earth_model)
        arrays, stats = inria_arrays(data)
        o = target.site.origin
        rmode = (data.get("render") or {}).get("rasterize_mode", "classic")
        comments = [
            "AeroSplat-GIS 3DGS export (INRIA layout)",
            f"site {target.site.key} run {target.run.id}",
            f"frame {SITE_FRAME}: x east, y north, z up, metres",
            f"origin_wgs84 lat {o.lat!r} lon {o.lon!r} h {o.h!r}",
            f"earth_model {target.run.earth_model} frame_error_m {target.budget.get('frame_error_m')}",
            f"rasterize_mode {rmode}",
        ]
        path = out_dir / FILENAMES["ply"]
        n = write_inria_ply(path, arrays, comments)
        return {"ply": {"source": src.name, "frame": SITE_FRAME, "rasterize_mode": rmode,
                        "origin": target.budget.get("origin"), **stats, "written": n}}, {"ply": path}

    raise ExportRefused(f"unknown export kind {kind}")


# ── CLI ──────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="aerosplat export", description=__doc__.split("\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--site", required=True, help="project/site або id ділянки")
    ap.add_argument("--run", dest="run_id", help="run (за замовчуванням — той, що показує сервер)")
    ap.add_argument("--epoch", help="епоха run-а (якщо id run-а неоднозначний)")
    ap.add_argument("--kinds", default=",".join(DEFAULT_KINDS),
                    help=f"через кому з: {','.join(KINDS)}")
    ap.add_argument("--crs", default=DEFAULT_CRS,
                    help="'utm' (зона origin ділянки, WGS84) або проєкційна СК, напр. EPSG:32618")
    ap.add_argument("--gsd", type=float, default=0.25, dest="gsd_m", help="піксель ортофото, м")
    ap.add_argument("--dem-px", type=float, default=None, dest="dem_px_m",
                    help="піксель DSM/DTM, м (за замовчуванням — крок растру семантики)")
    ap.add_argument("--tile", type=int, default=2048, dest="tile_px", help="плитка рендера, px")
    ap.add_argument("--no-cog", action="store_true", help="звичайний tiled GeoTIFF замість COG")
    ap.add_argument("--las-source", choices=["cloud", "splat"], default="cloud")
    ap.add_argument("--las", action="store_true", help=".las замість .laz")
    ap.add_argument("--max-gap", type=float, default=25.0, dest="max_gap_m",
                    help="DTM: nodata далі за стільки метрів від виміряної клітини")
    ap.add_argument("--sh-degree", type=int, default=None)
    ap.add_argument("--area", help="ортофото: e0,n0,e1,n1 у Site ENU, м (за замовчуванням — хмара)")
    ap.add_argument("--out-dir", type=Path, help="куди писати (тоді в каталозі не реєструється)")
    ap.add_argument("--projects-root", type=Path)
    return ap


def options_from_args(args: argparse.Namespace) -> ExportOptions:
    area = None
    if args.area:
        try:
            vals = tuple(float(v) for v in args.area.split(","))
        except ValueError:
            raise ExportRefused(f"--area expects e0,n0,e1,n1, got {args.area!r}") from None
        if len(vals) != 4:
            raise ExportRefused(f"--area expects e0,n0,e1,n1, got {args.area!r}")
        area = vals
    return ExportOptions(crs=args.crs, gsd_m=args.gsd_m, dem_px_m=args.dem_px_m,
                         tile_px=args.tile_px, cog=not args.no_cog, las_source=args.las_source,
                         las_ext=".las" if args.las else ".laz", max_gap_m=args.max_gap_m,
                         sh_degree=args.sh_degree, area_enu=area, out_dir=args.out_dir)  # type: ignore[arg-type]


def main(argv: Optional[list[str]] = None, on_event: Optional[EventFn] = None) -> int:
    fix_console_encoding()
    args = build_parser().parse_args(argv)
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    catalog = Catalog.open(args.projects_root)
    try:
        opts = options_from_args(args)
        target = ExportTarget.resolve(catalog, args.site, epoch_id=args.epoch, run_id=args.run_id)
        export_run(target, kinds, opts, on_event=on_event)
    except ExportRefused as e:
        print(f"❌ Експорт відхилено: {e}", file=sys.stderr, flush=True)
        return 2
    finally:
        catalog.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
