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
from aerosplat.surface.models import FORMAT_VERSION as OCCUPANCY_FORMAT

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


def masks_dir(cfg: Cfg, L: RunLayout) -> Path:
    """Masks of the capture's frames: `masks_dir`, else `masks/` next to the images dir."""
    if cfg["masks_dir"]:
        return abs_path(cfg["masks_dir"], L.project_dir)
    return L.footage.parent / "masks"


def tensorboard_dir(L: RunLayout) -> Path:
    """gs/splat.pt -> gs/splat_tensorboard (legacy: data/<ds>_splat_tensorboard)."""
    return L.splat_pt.with_name(L.splat_pt.stem + "_tensorboard")


def train_aux_inputs(cfg: Cfg, L: RunLayout) -> list[Path]:
    """Extra inputs of the train stage that exist only with masks / depth switched on."""
    out = []
    if cfg["masks"]:
        out.append(masks_dir(cfg, L))
    if cfg["rasterizer"] == "gsplat" and cfg["depth_reg"] == "mvs":
        out.append(L.dense / "stereo" / "depth_maps")
    return out


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
        if cfg["depth_reg"] == "mvs":
            a += ["--depth-mvs", str(L.dense), "--depth-weight", str(cfg["depth_weight"]),
                  "--depth-weight-final", str(cfg["depth_weight_final"])]
        if cfg["checkpoint_every"]:
            a += ["--checkpoint-every", str(cfg["checkpoint_every"]),
                  "--checkpoint-keep", str(cfg["checkpoint_keep"])]
        _switch("--resume", cfg["train_resume"], a)
        if cfg["tensorboard"]:
            a += ["--tensorboard", str(tensorboard_dir(L))]
    if cfg["masks"]:
        if images_dir == L.dense / "images" and not cfg["masks_dir"]:
            # Masks next to the capture are drawn on the original frames; these are
            # undistorted (and possibly cropped) copies, so pixels would not line up.
            raise ValueError(
                f"masks: тренування йде на undistorted-кадрах {images_dir}, а маски з "
                f"{masks_dir(cfg, L)} намальовано на оригіналах — пікселі не збігатимуться. "
                "Задай masks_dir з масками саме цих кадрів.")
        _flag("--masks", masks_dir(cfg, L), a)
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


def _semantic_cloud(cfg: Cfg, L: RunLayout) -> Path:
    """The Site ENU cloud that semantics and occupancy both read."""
    return L.cloud_wgs84 if _known_poses(cfg) else L.cloud_in


def argv_semantics(cfg: Cfg, L: RunLayout) -> list[str]:
    a = ["--cloud", str(_semantic_cloud(cfg, L)),
         "--out-dir", str(L.semantics_dir), "--sparse", str(L.sparse),
         "--seed", str(cfg["semantics_seed"])]
    for key, flag in (("semantics_res", "--res"), ("min_building_height", "--min-building-height"),
                      ("min_building_area", "--min-building-area"),
                      ("facade_inlier", "--facade-inlier"),
                      ("facade_min_length", "--facade-min-length"),
                      ("facade_min_height", "--facade-min-height"),
                      ("facade_min_points", "--facade-min-points"),
                      ("max_facades_per_building", "--max-facades-per-building"),
                      ("coverage_cell", "--coverage-cell"),
                      ("coverage_max_angle", "--coverage-max-angle"),
                      ("coverage_max_gsd", "--coverage-max-gsd")):
        _flag(flag, cfg[key], a)
    if _known_poses(cfg):
        # The cameras are in the reconstruction frame; this is how they reach Site ENU.
        _flag("--registration", L.registration_report, a)
    return a


def argv_occupancy(cfg: Cfg, L: RunLayout) -> list[str]:
    a = ["--cloud", str(_semantic_cloud(cfg, L)), "--semantics-dir", str(L.semantics_dir),
         "--out-dir", str(L.occupancy_dir), "--unknown", cfg["occupancy_unknown"]]
    for key, flag in (("occupancy_res", "--res"), ("occupancy_headroom", "--headroom"),
                      ("occupancy_max_distance", "--max-distance"),
                      ("occupancy_fill_radius", "--fill-radius"),
                      ("occupancy_close_radius", "--close-radius"),
                      ("occupancy_assumed_margin", "--assumed-margin"),
                      ("occupancy_skyline_quantile", "--skyline-quantile"),
                      ("occupancy_max_voxels", "--max-voxels")):
        _flag(flag, cfg[key], a)
    # One token each: argparse would read a value starting with "-" as an option.
    a += [f"--extent={cfg['occupancy_extent']}", f"--radius={cfg['occupancy_radius']!r}"]
    if cfg["occupancy_bbox"]:
        a.append(f"--bbox={cfg['occupancy_bbox']}")
    if L.site_bbox is not None and L.origin is not None:
        b = L.site_bbox
        a += [f"--site-bbox={b.west!r},{b.south!r},{b.east!r},{b.north!r}",
              "--origin=" + ",".join(repr(v) for v in L.origin.as_tuple())]
    if not cfg["occupancy_carve"]:
        return a + ["--no-carve"]
    # The dense workspace: fused points, their visibility, and the model its indices count.
    a += ["--fused", str(L.fused_ply), "--sparse", str(L.dense / "sparse")]
    if _known_poses(cfg):
        # Points and cameras are in the reconstruction frame, like semantics' cameras.
        _flag("--registration", L.registration_report, a)
    _flag("--carve-min-views", cfg["occupancy_carve_min_views"], a)
    _flag("--carve-margin", cfg["occupancy_carve_margin"], a)
    _switch("--no-carve-under", not cfg["occupancy_carve_under"], a)
    return a


def _occupancy_inputs(cfg: Cfg, L: RunLayout) -> list[Path]:
    return [_semantic_cloud(cfg, L), L.semantics_dir] + (
        [L.fused_ply] if cfg["occupancy_carve"] else [])


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
    # Inputs from the layout (not the config) that change the stage's output; empty for
    # most, so their fingerprints stay as they were.
    layout_values: Callable[[RunLayout], dict[str, Any]] = lambda L: {}

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
           "mcmc_refine_stop_frac", "appearance")
_TRAIN = ("cloud_source", "iterations", "rasterizer", "max_res", "max_gaussians",
          "init_headroom", "init_scale", "densify_from", "densify_until_frac",
          "densify_interval", "opacity_reset", "grad_threshold", "chunk",
          "raster_checkpoint", "warmup_iters", "warmup_scale", "save_every", "train_seed",
          "max_screen_radius", "scale_reg", "max_aniso_ratio",
          "known_poses", "eval_every", "eval_interval")
_EVAL = ("eval_every", "eval_lpips", "max_res", "rasterizer", "known_poses",
         "refine_known_poses")
_SEMANTICS = ("semantics_res", "min_building_height", "min_building_area", "facade_inlier",
              "facade_min_length", "facade_min_height", "facade_min_points",
              "max_facades_per_building", "coverage_cell", "coverage_max_angle",
              "coverage_max_gsd", "semantics_seed", "camera_source", "known_poses",
              "cloud_source")
_OCCUPANCY = ("occupancy_res", "occupancy_headroom", "occupancy_max_distance",
              "occupancy_fill_radius", "occupancy_close_radius", "occupancy_assumed_margin",
              "occupancy_unknown",
              "occupancy_skyline_quantile", "occupancy_bbox", "occupancy_extent",
              "occupancy_radius", "occupancy_max_voxels",
              "occupancy_carve", "occupancy_carve_min_views", "occupancy_carve_margin",
              "occupancy_carve_under",
              "semantics_res", "camera_source", "known_poses", "cloud_source")


def _geo_skip(cfg: Cfg) -> str | None:
    if cfg["camera_source"] != "earthstudio_json":
        return "camera_source='synthetic' — нічого конвертувати"
    return None


def _train_extra_keys(cfg: Cfg) -> tuple[str, ...]:
    """gsplat options, masks and depth regularisation: in the fingerprint only when used.

    checkpoint_every/keep, train_resume and tensorboard never are: they do not
    change the trained splat.
    """
    keys: tuple[str, ...] = ()
    if cfg["rasterizer"] == "gsplat":
        keys += _GSPLAT
        if cfg["depth_reg"] != "none":
            keys += ("depth_reg", "depth_weight", "depth_weight_final")
    if cfg["masks"]:
        keys += ("masks", "masks_dir")
    return keys


def _georef_skip(cfg: Cfg) -> str | None:
    if not _known_poses(cfg):
        return "реконструкція без відомих поз Earth Studio — метричного кадру немає"
    return None


def _semantics_skip(cfg: Cfg) -> str | None:
    if cfg["cloud_source"] != "dense":
        return "cloud_source='sparse' — розрідженої хмари замало на фасади"
    return None


def _occupancy_skip(cfg: Cfg) -> str | None:
    if cfg["cloud_source"] != "dense":
        return "cloud_source='sparse' — semantics пропущено, рельєфу для зайнятості немає"
    return None


def _occupancy_layout(L: RunLayout) -> dict[str, Any]:
    """The Site bbox is the default extent of the grid: a new one must rebuild it."""
    return {} if L.site_bbox is None else {"site_bbox": L.site_bbox.model_dump()}


def _exif_skip(cfg: Cfg) -> str | None:
    if cfg["camera_source"] == "earthstudio_json":
        return "camera_source='earthstudio_json' — raw_images_<ds> вже заповнено стадією 'geo'"
    return None


STAGES: dict[str, Stage] = {s.name: s for s in [
    Stage("geo", "0/9  Earth Studio JSON → відомі пози COLMAP (сферичний ENU ділянки) + кадри для SfM",
          lambda c: "aerosplat.geo.earthstudio_to_colmap", argv_geo,
          lambda c, L: [L.json_path],
          lambda c, L: [L.known_poses_dir, L.geotagged],
          ("camera_source",), _geo_skip, version=EARTH_MODEL),
    Stage("exif", "1/9  EXIF GPS injection",
          lambda c: "aerosplat.geo.exif_injector", argv_exif,
          lambda c, L: [L.footage],
          lambda c, L: [L.geotagged],
          ("camera_source",) + _SYNTH, _exif_skip),
    Stage("sfm", "2/9  COLMAP SfM (sparse)",
          lambda c: "aerosplat.sfm.sfm_runner", argv_sfm,
          lambda c, L: [L.geotagged],
          lambda c, L: [L.sparse],
          ("camera_source",) + _SFM),
    Stage("dense", "3/9  COLMAP Dense MVS",
          lambda c: "aerosplat.sfm.dense_runner", argv_dense,
          lambda c, L: [L.sparse],
          lambda c, L: [L.fused_ply],
          _DENSE),
    Stage("parse", "4/9  point cloud → .pt",
          lambda c: ("aerosplat.io.ply_parser" if c["cloud_source"] == "dense"
                  else "aerosplat.io.colmap_parser"),
          argv_parse,
          lambda c, L: [L.fused_ply if c["cloud_source"] == "dense" else L.points3d],
          lambda c, L: [L.cloud_in],
          _PARSE),
    Stage("train", "5/9  3D Gaussian Splatting (aerosplat.gs.trainer)",
          lambda c: "aerosplat.gs.trainer", argv_train,
          lambda c, L: [L.cloud_in] + train_aux_inputs(c, L),
          lambda c, L: [L.splat_pt],
          _TRAIN, extra_keys=_train_extra_keys),
    Stage("eval", "6/9  Оцінка якості на test-кадрах (metrics.json + report.html)",
          lambda c: "aerosplat.gs.evaluate", argv_eval,
          lambda c, L: [L.splat_pt],
          lambda c, L: [L.metrics_json, L.report_html],
          _EVAL,
          lambda c: ("eval_every=0 — test-спліту немає" if not c["eval_every"] else None)),
    Stage("georef", "7/9  Продукти: сферичний ENU → WGS84 Site ENU (афінне з reports/registration.json)",
          lambda c: "aerosplat.geo.georef", argv_georef,
          lambda c, L: [L.registration_report, L.cloud_in, L.splat_pt],
          lambda c, L: [L.cloud_wgs84, L.splat_wgs84],
          ("camera_source", "known_poses", "cloud_source"), _georef_skip),
    Stage("semantics", "8/9  Семантика: рельєф, будівлі, фасади, карта покриття",
          lambda c: "aerosplat.semantics.stage", argv_semantics,
          lambda c, L: [_semantic_cloud(c, L)],
          lambda c, L: [L.semantics_dir],
          _SEMANTICS, _semantics_skip),
    Stage("occupancy", "9/9  Зайнятість: воксельна сітка + ESDF (кліренси, промені)",
          lambda c: "aerosplat.surface.stage", argv_occupancy,
          _occupancy_inputs,
          lambda c, L: [L.occupancy_dir],
          _OCCUPANCY, _occupancy_skip, version=OCCUPANCY_FORMAT,
          layout_values=_occupancy_layout),
]}
