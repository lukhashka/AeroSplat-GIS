"""Stage registry: each stage builds argv for its module's `main(argv)` and runs in-process.

The stage scripts keep their CLIs (still runnable standalone), but the pipeline
calls their `main()` directly, so a failure surfaces as a normal Python
traceback instead of an opaque subprocess exit code.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from aerosplat.geo.sphere_enu import EARTH_MODEL
from aerosplat.logging_utils import get_logger
from aerosplat.pipeline.layout import RunLayout, abs_path

Cfg = dict[str, Any]

log = get_logger()


def _flag(name: str, value: Any, out: list[str]) -> None:
    if value is not None:
        out += [name, str(value)]


def _switch(name: str, on: bool, out: list[str]) -> None:
    if on:
        out.append(name)


def _known_poses(cfg: Cfg) -> bool:
    return cfg["camera_source"] == "earthstudio_json" and cfg["known_poses"]


# ── argv builders ────────────────────────────────────────────────────────
def argv_geo(cfg: Cfg, L: RunLayout) -> list[str]:
    a = [str(L.json_path), "--dataset", L.frame_prefix, "--out-dir", str(L.known_poses_dir),
         "--footage-dir", str(L.footage), "--images-out-dir", str(L.geotagged),
         "--report", str(L.registration_report)]
    if L.origin is not None:
        a += ["--origin", ",".join(repr(v) for v in L.origin.as_tuple())]
    return a


def argv_exif(cfg: Cfg, L: RunLayout) -> list[str]:
    a = ["--images-dir", str(L.footage), "--out-dir", str(L.geotagged)]
    if cfg["telemetry_csv"]:
        _flag("--telemetry", abs_path(cfg["telemetry_csv"], L.project_dir), a)
    else:
        for k in ("lat", "lon", "altitude", "radius", "arc", "start_bearing"):
            _flag("--" + k.replace("_", "-"), cfg[k], a)
        _switch("--counter-clockwise", cfg["counter_clockwise"], a)
    _flag("--hfov", cfg["hfov"], a)
    _flag("--camera-make", cfg["camera_make"], a)
    _flag("--camera-model", cfg["camera_model"], a)
    if cfg["colmap_geo"]:
        _flag("--colmap-geo", L.geo_txt, a)
    return a


def argv_sfm(cfg: Cfg, L: RunLayout) -> list[str]:
    a = ["--dataset", L.dataset, "--images-dir", str(L.geotagged),
         "--output-dir", str(L.colmap_out), "--colmap-bin", cfg["colmap_bin"],
         "--matcher", cfg["matcher"], "--mapper", cfg["mapper"]]
    _switch("--fresh", cfg["fresh"], a)
    _flag("--seq-overlap", cfg["seq_overlap"], a)
    _flag("--vocab-tree", cfg["vocab_tree"], a)
    _flag("--spatial-neighbors", cfg["spatial_neighbors"], a)
    _flag("--spatial-max-distance", cfg["spatial_max_distance"], a)
    _flag("--max-image-size", cfg["sfm_max_image_size"], a)
    if _known_poses(cfg):
        # Real ground-truth poses + intrinsics: point_triangulator (+ optional BA)
        # instead of incremental pose-guessing.
        _flag("--known-poses-dir", L.known_poses_dir, a)
        if not cfg["refine_known_poses"]:
            a.append("--no-refine-known-poses")
    else:
        if not cfg["known_intrinsics"]:
            a.append("--no-known-intrinsics")
        _flag("--image-width", cfg["image_width"], a)
        _flag("--image-height", cfg["image_height"], a)
        _flag("--fov-deg", cfg["fov_deg"], a)
    return a


def argv_dense(cfg: Cfg, L: RunLayout) -> list[str]:
    a = ["--dataset", L.dataset, "--colmap-bin", cfg["colmap_bin"],
         "--gpu-index", cfg["gpu_index"], "--images-dir", str(L.geotagged),
         "--sparse-dir", str(L.sparse), "--dense-dir", str(L.dense)]
    for k in ("window_step", "num_samples", "num_iterations", "cache_size",
              "min_num_pixels", "max_reproj_error"):
        _flag("--" + k.replace("_", "-"), cfg[k], a)
    _flag("--max-image-size", cfg["dense_max_image_size"], a)
    _switch("--skip-undistort", cfg["skip_undistort"], a)
    _switch("--skip-stereo", cfg["skip_stereo"], a)
    return a


def argv_parse(cfg: Cfg, L: RunLayout) -> list[str]:
    if cfg["cloud_source"] == "dense":
        a = ["--input", str(L.fused_ply), "--output", str(L.dense_pt)]
    else:
        a = ["--input", str(L.points3d), "--output", str(L.cloud_pt)]
        _flag("--max-error", cfg["max_error"], a)
        _flag("--min-track", cfg["min_track"], a)
    _flag("--mad-k", cfg["mad_k"], a)
    _flag("--max-points", cfg["max_points"], a)
    _switch("--recenter", cfg["recenter"], a)
    _flag("--seed", cfg["cloud_seed"], a)
    return a


def train_inputs(cfg: Cfg, L: RunLayout) -> tuple[Path, Path]:
    """(colmap sparse dir, images dir) that train AND eval must share (same frames/intrinsics)."""
    colmap_dir, images_dir = L.sparse, L.geotagged
    if not _known_poses(cfg):
        # SIMPLE_RADIAL has distortion the trainer ignores → train on the
        # undistorted dense/{sparse,images} pair (same coordinate frame).
        u_sparse, u_images = L.dense / "sparse", L.dense / "images"
        if u_sparse.is_dir() and u_images.is_dir():
            colmap_dir, images_dir = u_sparse, u_images
            log.info(f"  train: undistorted кадри {u_images}")
        else:
            log.warning("  ⚠️  dense/images не знайдено (стадія dense не виконувалась) — "
                        "тренування на сирих кадрах без корекції дисторсії")
    return colmap_dir, images_dir


def argv_train(cfg: Cfg, L: RunLayout) -> list[str]:
    colmap_dir, images_dir = train_inputs(cfg, L)
    a = ["--input", str(L.cloud_in), "--output", str(L.splat_pt),
         "--colmap", str(colmap_dir), "--images", str(images_dir),
         "--rasterizer", cfg["rasterizer"]]
    for k in ("iterations", "max_res", "max_gaussians", "init_headroom", "init_scale",
              "densify_from", "densify_until_frac", "densify_interval", "opacity_reset",
              "grad_threshold", "chunk", "warmup_iters", "warmup_scale", "save_every", "max_screen_radius",
              "scale_reg", "max_aniso_ratio"):
        _flag("--" + k.replace("_", "-"), cfg[k], a)
    _switch("--raster-checkpoint", cfg["raster_checkpoint"], a)
    if cfg["rasterizer"] == "gsplat":
        for k in _GSPLAT:
            if k != "absgrad":
                _flag("--" + k.replace("_", "-"), cfg[k], a)
        _switch("--absgrad", cfg["absgrad"], a)
    _flag("--seed", cfg["train_seed"], a)
    _flag("--eval-every", cfg["eval_every"], a)
    _flag("--eval-interval", cfg["eval_interval"], a)
    return a


def argv_eval(cfg: Cfg, L: RunLayout) -> list[str]:
    colmap_dir, images_dir = train_inputs(cfg, L)
    a = ["--splat", str(L.splat_pt), "--colmap", str(colmap_dir), "--images", str(images_dir),
         "--out-dir", str(L.reports_dir), "--eval-every", str(cfg["eval_every"]),
         "--max-res", str(cfg["max_res"]), "--rasterizer", cfg["rasterizer"],
         "--manifest", str(L.manifest_path)]
    _switch("--no-lpips", not cfg["eval_lpips"], a)
    if _known_poses(cfg) and cfg["refine_known_poses"]:
        a += ["--known-poses-dir", str(L.known_poses_dir)]
    return a


def argv_georef(cfg: Cfg, L: RunLayout) -> list[str]:
    return ["--registration", str(L.registration_report),
            "--cloud", str(L.cloud_in), str(L.cloud_wgs84),
            "--splat", str(L.splat_pt), str(L.splat_wgs84)]


# ── registry ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Stage:
    name: str
    label: str
    module: Callable[[Cfg], str]      # importable module exposing main(argv)
    argv: Callable[[Cfg, RunLayout], list[str]]
    inputs: Callable[[Cfg, RunLayout], list[Path]]
    outputs: Callable[[Cfg, RunLayout], list[Path]]
    keys: tuple[str, ...]             # config keys this stage depends on
    skip_reason: Callable[[Cfg], str | None] = lambda cfg: None
    # Bumped when the stage's output convention changes, so `--resume` redoes
    # it (and everything downstream) instead of reusing incompatible outputs.
    version: str = ""
    # Keys that matter only for some configs; left out otherwise, so adding
    # options does not change the fingerprints of runs that do not use them.
    extra_keys: Callable[[Cfg], tuple[str, ...]] = lambda cfg: ()

    def keys_for(self, cfg: Cfg) -> tuple[str, ...]:
        return self.keys + self.extra_keys(cfg)

    def script(self, cfg: Cfg) -> Path:
        spec = importlib.util.find_spec(self.module(cfg))
        if spec is None or spec.origin is None:
            raise ModuleNotFoundError(self.module(cfg))
        return Path(spec.origin)

    def entry(self, cfg: Cfg) -> Callable[[list[str]], Any]:
        return importlib.import_module(self.module(cfg)).main


_SYNTH = ("lat", "lon", "altitude", "radius", "arc", "start_bearing", "counter_clockwise",
          "hfov", "camera_make", "camera_model", "colmap_geo", "telemetry_csv")
_SFM = ("matcher", "mapper", "fresh", "seq_overlap", "vocab_tree", "spatial_neighbors",
        "spatial_max_distance", "sfm_max_image_size", "known_intrinsics", "image_width",
        "image_height", "fov_deg", "known_poses", "refine_known_poses", "colmap_bin")
_DENSE = ("dense_max_image_size", "window_step", "num_samples", "num_iterations",
          "cache_size", "min_num_pixels", "max_reproj_error", "skip_undistort",
          "skip_stereo", "gpu_index", "colmap_bin")
_PARSE = ("cloud_source", "mad_k", "max_points", "recenter", "cloud_seed", "max_error",
          "min_track")
_GSPLAT = ("sh_degree", "sh_degree_interval", "rasterize_mode", "densify_strategy", "absgrad",
           "absgrad_threshold", "mcmc_opacity_reg", "mcmc_scale_reg", "mcmc_noise_lr",
           "appearance")
_TRAIN = ("cloud_source", "iterations", "rasterizer", "max_res", "max_gaussians",
          "init_headroom", "init_scale", "densify_from", "densify_until_frac",
          "densify_interval", "opacity_reset", "grad_threshold", "chunk",
          "raster_checkpoint", "warmup_iters", "warmup_scale", "save_every", "train_seed",
          "max_screen_radius", "scale_reg", "max_aniso_ratio",
          "known_poses", "eval_every", "eval_interval")
_EVAL = ("eval_every", "eval_lpips", "max_res", "rasterizer", "known_poses",
         "refine_known_poses")


def _geo_skip(cfg: Cfg) -> str | None:
    if cfg["camera_source"] != "earthstudio_json":
        return "camera_source='synthetic' — нічого конвертувати"
    return None


def _gsplat_keys(cfg: Cfg) -> tuple[str, ...]:
    return _GSPLAT if cfg["rasterizer"] == "gsplat" else ()


def _georef_skip(cfg: Cfg) -> str | None:
    if not _known_poses(cfg):
        return "реконструкція без відомих поз Earth Studio — метричного кадру немає"
    return None


def _exif_skip(cfg: Cfg) -> str | None:
    if cfg["camera_source"] == "earthstudio_json":
        return "camera_source='earthstudio_json' — raw_images_<ds> вже заповнено стадією 'geo'"
    return None


STAGES: dict[str, Stage] = {s.name: s for s in [
    Stage("geo", "0/7  Earth Studio JSON → відомі пози COLMAP (сферичний ENU ділянки) + кадри для SfM",
          lambda c: "aerosplat.geo.earthstudio_to_colmap", argv_geo,
          lambda c, L: [L.json_path],
          lambda c, L: [L.known_poses_dir, L.geotagged],
          ("camera_source",), _geo_skip, version=EARTH_MODEL),
    Stage("exif", "1/7  EXIF GPS injection",
          lambda c: "aerosplat.geo.exif_injector", argv_exif,
          lambda c, L: [L.footage],
          lambda c, L: [L.geotagged],
          ("camera_source",) + _SYNTH, _exif_skip),
    Stage("sfm", "2/7  COLMAP SfM (sparse)",
          lambda c: "aerosplat.sfm.sfm_runner", argv_sfm,
          lambda c, L: [L.geotagged],
          lambda c, L: [L.sparse],
          ("camera_source",) + _SFM),
    Stage("dense", "3/7  COLMAP Dense MVS",
          lambda c: "aerosplat.sfm.dense_runner", argv_dense,
          lambda c, L: [L.sparse],
          lambda c, L: [L.fused_ply],
          _DENSE),
    Stage("parse", "4/7  point cloud → .pt",
          lambda c: ("aerosplat.io.ply_parser" if c["cloud_source"] == "dense"
                  else "aerosplat.io.colmap_parser"),
          argv_parse,
          lambda c, L: [L.fused_ply if c["cloud_source"] == "dense" else L.points3d],
          lambda c, L: [L.cloud_in],
          _PARSE),
    Stage("train", "5/7  3D Gaussian Splatting (aerosplat.gs.trainer)",
          lambda c: "aerosplat.gs.trainer", argv_train,
          lambda c, L: [L.cloud_in],
          lambda c, L: [L.splat_pt],
          _TRAIN, extra_keys=_gsplat_keys),
    Stage("eval", "6/7  Оцінка якості на test-кадрах (metrics.json + report.html)",
          lambda c: "aerosplat.gs.evaluate", argv_eval,
          lambda c, L: [L.splat_pt],
          lambda c, L: [L.metrics_json, L.report_html],
          _EVAL,
          lambda c: ("eval_every=0 — test-спліту немає" if not c["eval_every"] else None)),
    Stage("georef", "7/7  Продукти: сферичний ENU → WGS84 Site ENU (афінне з reports/registration.json)",
          lambda c: "aerosplat.geo.georef", argv_georef,
          lambda c, L: [L.registration_report, L.cloud_in, L.splat_pt],
          lambda c, L: [L.cloud_wgs84, L.splat_wgs84],
          ("camera_source", "known_poses", "cloud_source"), _georef_skip),
]}
