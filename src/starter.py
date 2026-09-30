#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
starter.py — єдина точка входу в пайплайн AeroSplat-GIS.

footage → EXIF GPS → COLMAP SfM → COLMAP MVS → .pt cloud → 3DGS splat

Кладеться в src/ (поруч з reconstruction/ і telemetry/).

Приклади:
    python src/starter.py                          # весь пайплайн, дефолтний датасет
    python src/starter.py --dataset dataset_v6
    python src/starter.py --from sfm               # почати з SfM (кадри вже геотеговані)
    python src/starter.py --only train
    python src/starter.py --stages exif,sfm,dense
    python src/starter.py --dry-run                # тільки показати команди
    python src/starter.py --config my_run.json     # усі параметри з файлу
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from aerosplat.config import RunConfig, build_config

# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

STAGES = ["geo", "exif", "sfm", "dense", "parse", "train"]

# --------------------------------------------------------------------------
# Параметри: src/aerosplat/config.py (RunConfig, валідація pydantic).
# Перекриття: --config run.yaml|json, прапорці CLI, --set KEY=VALUE.
# --------------------------------------------------------------------------
CONFIG: dict = RunConfig().to_dict()


# --------------------------------------------------------------------------
# Похідні шляхи
# --------------------------------------------------------------------------
class Paths:
    """Усі шляхи в одному місці — щоб не тримати їх у 5 скриптах."""

    def __init__(self, cfg: dict):
        ds = cfg["dataset"]
        self.dataset = ds
        self.footage = _abs(cfg["footage_dir"])

        # ВАЖЛИВО: sfm_runner.py і dense_runner.py жорстко очікують саме
        # цю схему іменування, тому геотеговані кадри кладемо сюди.
        self.geotagged = DATA_DIR / f"raw_images_{ds}"
        self.colmap_out = DATA_DIR / f"colmap_output_{ds}"
        self.sparse = self.colmap_out / "sparse" / "0"
        self.dense = self.colmap_out / "dense"
        self.fused_ply = self.dense / "fused.ply"
        self.points3d = self.sparse / "points3D.bin"

        self.cloud_pt = DATA_DIR / f"{ds}_cloud.pt"      # sparse → .pt
        self.dense_pt = DATA_DIR / f"{ds}_dense.pt"      # dense  → .pt
        self.splat_pt = DATA_DIR / f"{ds}_splat.pt"      # 3DGS вихід
        self.geo_txt = DATA_DIR / f"{ds}_geo.txt"

        self.cloud_in = (self.dense_pt if cfg["cloud_source"] == "dense"
                         else self.cloud_pt)

        # ── earthstudio_json camera source ──────────────────────────────
        self.json_path = (_abs(cfg["json_path"]) if cfg["json_path"]
                          else DATA_DIR / ds / f"{ds}.json")
        # NOT under self.colmap_out: sfm_runner.py's --fresh wipes that
        # whole directory before every run, and known poses are ground-
        # truth INPUT data, not regenerable COLMAP working output — they
        # must survive a --fresh wipe. Keeping them beside the source JSON
        # instead.
        self.known_poses_dir = DATA_DIR / ds / "sparse_known"
        self.geo_registration_json = DATA_DIR / ds / f"{ds}_geo_registration.json"

    # скрипти
    exif_injector = SRC_DIR / "telemetry" / "exif_injector.py"
    sfm_runner = SRC_DIR / "reconstruction" / "sfm_runner.py"
    dense_runner = SRC_DIR / "reconstruction" / "dense_runner.py"
    ply_parser = SRC_DIR / "reconstruction" / "ply_parser.py"
    colmap_parser = SRC_DIR / "reconstruction" / "colmap_parser.py"
    gs_trainer = SRC_DIR / "reconstruction" / "gs_trainer_v3.py"
    earthstudio_converter = SRC_DIR / "reconstruction" / "earthstudio_to_colmap.py"


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_DIR / p)


# --------------------------------------------------------------------------
# Хелпери побудови команд
# --------------------------------------------------------------------------
def flag(name: str, value, out: list) -> None:
    """Додає --name value, якщо value не None."""
    if value is not None:
        out += [name, str(value)]


def switch(name: str, on: bool, out: list) -> None:
    if on:
        out.append(name)


def run(cmd: list, label: str, dry: bool) -> None:
    print(f"\n{'=' * 70}\n▶  {label}\n{'=' * 70}")
    printable = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
    print(f"$ {printable}\n")
    if dry:
        return
    t0 = time.perf_counter()
    res = subprocess.run([str(c) for c in cmd], cwd=str(PROJECT_DIR))
    dt = time.perf_counter() - t0
    if res.returncode != 0:
        raise SystemExit(f"\n❌ Стадія '{label}' впала (код {res.returncode}) "
                         f"після {dt:.1f}s")
    print(f"\n✅ {label} — {dt:.1f}s")


PY = sys.executable


# --------------------------------------------------------------------------
# Стадії
# --------------------------------------------------------------------------
def stage_geo(cfg: dict, p: Paths, dry: bool) -> None:
    if cfg["camera_source"] != "earthstudio_json":
        print("⏭  [geo] camera_source='synthetic' — нічого конвертувати, пропущено\n")
        return

    if not dry and not p.json_path.is_file():
        raise SystemExit(f"❌ Не знайдено Earth Studio JSON: {p.json_path}")

    cmd = [PY, Paths.earthstudio_converter,
           p.json_path,
           "--dataset", p.dataset,
           "--out-dir", p.known_poses_dir,
           "--footage-dir", p.footage,
           "--images-out-dir", p.geotagged]

    run(cmd, "0/5  Earth Studio JSON → відомі пози COLMAP + raw_images_<ds>", dry)


def stage_exif(cfg: dict, p: Paths, dry: bool) -> None:
    if cfg["camera_source"] == "earthstudio_json":
        # geo-стадія вже наповнила p.geotagged реальними кадрами (+ GPS EXIF,
        # якщо є piexif) напряму з реальних поз — синтетичний інжектор тут
        # не потрібен і не запускається.
        print("⏭  [exif] camera_source='earthstudio_json' — raw_images_<ds> "
              "вже заповнено стадією 'geo', пропущено\n")
        return

    if not p.footage.is_dir():
        raise SystemExit(f"❌ Немає теки з кадрами: {p.footage}")

    cmd = [PY, Paths.exif_injector,
           "--images-dir", p.footage,
           "--out-dir", p.geotagged]

    if cfg["telemetry_csv"]:
        flag("--telemetry", _abs(cfg["telemetry_csv"]), cmd)
    else:
        flag("--lat", cfg["lat"], cmd)
        flag("--lon", cfg["lon"], cmd)
        flag("--altitude", cfg["altitude"], cmd)
        flag("--radius", cfg["radius"], cmd)
        flag("--arc", cfg["arc"], cmd)
        flag("--start-bearing", cfg["start_bearing"], cmd)
        switch("--counter-clockwise", cfg["counter_clockwise"], cmd)

    flag("--hfov", cfg["hfov"], cmd)
    flag("--camera-make", cfg["camera_make"], cmd)
    flag("--camera-model", cfg["camera_model"], cmd)
    if cfg["colmap_geo"]:
        flag("--colmap-geo", p.geo_txt, cmd)

    run(cmd, "1/5  EXIF GPS injection", dry)


def stage_sfm(cfg: dict, p: Paths, dry: bool) -> None:
    cmd = [PY, Paths.sfm_runner,
           "--dataset", p.dataset,
           "--images-dir", p.geotagged,
           "--output-dir", p.colmap_out,
           "--colmap-bin", cfg["colmap_bin"],
           "--matcher", cfg["matcher"],
           "--mapper", cfg["mapper"]]
    switch("--fresh", cfg["fresh"], cmd)
    flag("--seq-overlap", cfg["seq_overlap"], cmd)
    flag("--vocab-tree", cfg["vocab_tree"], cmd)
    flag("--spatial-neighbors", cfg["spatial_neighbors"], cmd)
    flag("--spatial-max-distance", cfg["spatial_max_distance"], cmd)
    flag("--max-image-size", cfg["sfm_max_image_size"], cmd)

    if cfg["camera_source"] == "earthstudio_json" and cfg["known_poses"]:
        # Real ground-truth poses + intrinsics from the geo stage — replaces
        # incremental pose-guessing with point_triangulator (+ optional BA).
        # See sfm_runner.py's --known-poses-dir docstring / this session's
        # notes for why this is more accurate when real poses exist.
        flag("--known-poses-dir", p.known_poses_dir, cmd)
        if not cfg["refine_known_poses"]:
            cmd.append("--no-refine-known-poses")
    else:
        if not cfg["known_intrinsics"]:
            cmd.append("--no-known-intrinsics")
        flag("--image-width", cfg["image_width"], cmd)
        flag("--image-height", cfg["image_height"], cmd)
        flag("--fov-deg", cfg["fov_deg"], cmd)

    run(cmd, "2/5  COLMAP SfM (sparse)", dry)


def stage_dense(cfg: dict, p: Paths, dry: bool) -> None:
    # dense_runner.py не має --images-dir/--output-dir: він сам будує
    # data/raw_images_<ds> і data/colmap_output_<ds>. Тому лише --dataset.
    cmd = [PY, Paths.dense_runner,
           "--dataset", p.dataset,
           "--colmap-bin", cfg["colmap_bin"],
           "--gpu-index", cfg["gpu_index"]]
    flag("--max-image-size", cfg["dense_max_image_size"], cmd)
    flag("--window-step", cfg["window_step"], cmd)
    flag("--num-samples", cfg["num_samples"], cmd)
    flag("--num-iterations", cfg["num_iterations"], cmd)
    flag("--cache-size", cfg["cache_size"], cmd)
    flag("--min-num-pixels", cfg["min_num_pixels"], cmd)
    flag("--max-reproj-error", cfg["max_reproj_error"], cmd)
    switch("--skip-undistort", cfg["skip_undistort"], cmd)
    switch("--skip-stereo", cfg["skip_stereo"], cmd)

    run(cmd, "3/5  COLMAP Dense MVS", dry)


def stage_parse(cfg: dict, p: Paths, dry: bool) -> None:
    if cfg["cloud_source"] == "dense":
        cmd = [PY, Paths.ply_parser,
               "--input", p.fused_ply,
               "--output", p.dense_pt]
        label = "4/5  fused.ply → .pt (dense)"
    else:
        cmd = [PY, Paths.colmap_parser,
               "--input", p.points3d,
               "--output", p.cloud_pt]
        flag("--max-error", cfg["max_error"], cmd)
        flag("--min-track", cfg["min_track"], cmd)
        label = "4/5  points3D.bin → .pt (sparse)"

    flag("--mad-k", cfg["mad_k"], cmd)
    flag("--max-points", cfg["max_points"], cmd)
    switch("--recenter", cfg["recenter"], cmd)
    flag("--seed", cfg["cloud_seed"], cmd)

    run(cmd, label, dry)


def stage_train(cfg: dict, p: Paths, dry: bool) -> None:
    colmap_dir, images_dir = p.sparse, p.geotagged
    known = cfg["camera_source"] == "earthstudio_json" and cfg["known_poses"]
    if not known:
        # Модель SIMPLE_RADIAL має дисторсію, яку тренер ігнорує → тренуємось на
        # undistorted-парі dense/{sparse,images} (той самий кадр координат).
        u_sparse, u_images = p.dense / "sparse", p.dense / "images"
        if u_sparse.is_dir() and u_images.is_dir():
            colmap_dir, images_dir = u_sparse, u_images
            print(f"  train: undistorted кадри {u_images}")
        else:
            print("  ⚠️  dense/images не знайдено (стадія dense не виконувалась) — "
                  "тренування на сирих кадрах без корекції дисторсії")
    cmd = [PY, Paths.gs_trainer,
           "--input", p.cloud_in,
           "--output", p.splat_pt,
           "--colmap", colmap_dir,
           "--images", images_dir,
           "--rasterizer", cfg["rasterizer"]]
    flag("--iterations", cfg["iterations"], cmd)
    flag("--max-res", cfg["max_res"], cmd)
    flag("--max-gaussians", cfg["max_gaussians"], cmd)
    flag("--init-headroom", cfg["init_headroom"], cmd)
    flag("--init-scale", cfg["init_scale"], cmd)
    flag("--densify-from", cfg["densify_from"], cmd)
    flag("--densify-until-frac", cfg["densify_until_frac"], cmd)
    flag("--densify-interval", cfg["densify_interval"], cmd)
    flag("--opacity-reset", cfg["opacity_reset"], cmd)
    flag("--grad-threshold", cfg["grad_threshold"], cmd)
    flag("--chunk", cfg["chunk"], cmd)
    switch("--raster-checkpoint", cfg["raster_checkpoint"], cmd)
    flag("--warmup-iters", cfg["warmup_iters"], cmd)
    flag("--warmup-scale", cfg["warmup_scale"], cmd)
    flag("--save-every", cfg["save_every"], cmd)
    flag("--seed", cfg["train_seed"], cmd)

    run(cmd, "5/5  3D Gaussian Splatting (gs_trainer_v3)", dry)


RUNNERS = {"geo": stage_geo, "exif": stage_exif, "sfm": stage_sfm, "dense": stage_dense,
           "parse": stage_parse, "train": stage_train}


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------
def preflight(cfg: dict, p: Paths, stages: list) -> None:
    print("── Preflight ─────────────────────────────────────────────────")
    print(f"  проєкт:      {PROJECT_DIR}")
    print(f"  датасет:     {p.dataset}")
    print(f"  джерело камер: {cfg['camera_source']}")
    if cfg["camera_source"] == "earthstudio_json":
        print(f"  JSON:        {p.json_path}")
        print(f"  known poses: {p.known_poses_dir}  (refine={cfg['refine_known_poses']})")
    print(f"  кадри:       {p.footage}")
    print(f"  геотеговані: {p.geotagged}")
    print(f"  COLMAP out:  {p.colmap_out}")
    print(f"  хмара .pt:   {p.cloud_in}  ({cfg['cloud_source']})")
    print(f"  splat .pt:   {p.splat_pt}")
    print(f"  стадії:      {' → '.join(stages)}")

    for name in stages:
        script = {"geo": Paths.earthstudio_converter, "exif": Paths.exif_injector,
                  "sfm": Paths.sfm_runner, "dense": Paths.dense_runner,
                  "train": Paths.gs_trainer,
                  "parse": (Paths.ply_parser if cfg["cloud_source"] == "dense"
                            else Paths.colmap_parser)}[name]
        if name == "geo" and cfg["camera_source"] != "earthstudio_json":
            continue  # stage no-ops in this mode, script need not exist
        if not script.is_file():
            raise SystemExit(f"❌ Не знайдено скрипт: {script}")

    if {"sfm", "dense"} & set(stages) and shutil.which(cfg["colmap_bin"]) is None:
        print(f"  ⚠️  '{cfg['colmap_bin']}' не знайдено в PATH — "
              f"задай --colmap-bin або COLMAP_BIN")

    # вхідні дані для першої стадії
    first = stages[0]
    if first == "geo" and cfg["camera_source"] != "earthstudio_json":
        need = None  # no-op stage, nothing to check
    else:
        need = {"geo": p.json_path, "exif": p.footage, "sfm": p.geotagged,
                "dense": p.sparse,
                "parse": (p.fused_ply if cfg["cloud_source"] == "dense"
                          else p.points3d),
                "train": p.cloud_in}[first]
    if need is not None and not need.exists():
        print(f"  ⚠️  вхід для стадії '{first}' не існує: {need}")
    print()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="AeroSplat-GIS — єдина точка входу в пайплайн",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", type=Path,
                    help="JSON з будь-якими ключами CONFIG (перекриває дефолти)")
    ap.add_argument("--dataset")
    ap.add_argument("--camera-source", choices=["auto", "synthetic", "earthstudio_json"])
    ap.add_argument("--footage-dir")
    ap.add_argument("--telemetry-csv")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--altitude", type=float)
    ap.add_argument("--radius", type=float)
    ap.add_argument("--cloud-source", choices=["dense", "sparse"])
    ap.add_argument("--iterations", type=int)
    ap.add_argument("--max-gaussians", type=int)
    ap.add_argument("--rasterizer", choices=["auto", "cuda", "torch"])
    ap.add_argument("--colmap-bin")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="перекрити будь-який ключ CONFIG, напр. --set chunk=32")

    ap.add_argument("--stages", help=f"через кому, з: {','.join(STAGES)}")
    ap.add_argument("--from", dest="from_stage", choices=STAGES,
                    help="почати з цієї стадії і далі до кінця")
    ap.add_argument("--to", dest="to_stage", choices=STAGES)
    ap.add_argument("--only", choices=STAGES, help="лише одна стадія")
    ap.add_argument("--skip", help="пропустити стадії, через кому")
    ap.add_argument("--dry-run", action="store_true",
                    help="показати команди, нічого не запускати")
    return ap


def _coerce(text: str):
    low = text.lower()
    if low in ("none", "null"):
        return None
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def resolve_config(args) -> dict:
    overrides: dict = {}
    for key in ("dataset", "camera_source", "footage_dir", "telemetry_csv", "lat", "lon",
                "altitude", "radius", "cloud_source", "iterations",
                "max_gaussians", "rasterizer", "colmap_bin"):
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set очікує KEY=VALUE, отримано: {item}")
        k, v = item.split("=", 1)
        if k not in RunConfig.model_fields:
            raise SystemExit(f"Невідомий ключ конфігу: {k}")
        overrides[k] = _coerce(v)
    # _coerce("none") -> None is a meaningful override (Optional fields), so apply
    # --set values after the None-filtering done in build_config.
    none_keys = [k for k, v in overrides.items() if v is None]
    try:
        cfg = build_config(args.config, {k: v for k, v in overrides.items() if v is not None})
        if none_keys:
            data = cfg.to_dict()
            data.update({k: None for k in none_keys})
            cfg = RunConfig(**data)
    except Exception as e:  # pydantic.ValidationError / bad file
        raise SystemExit(f"❌ Невалідний конфіг: {e}")
    return cfg.to_dict()


def resolve_stages(args) -> list:
    if args.only:
        return [args.only]
    if args.stages:
        chosen = [s.strip() for s in args.stages.split(",") if s.strip()]
        bad = [s for s in chosen if s not in STAGES]
        if bad:
            raise SystemExit(f"Невідомі стадії: {bad}")
        return chosen
    start = STAGES.index(args.from_stage) if args.from_stage else 0
    end = STAGES.index(args.to_stage) + 1 if args.to_stage else len(STAGES)
    chosen = STAGES[start:end]
    if args.skip:
        skip = {s.strip() for s in args.skip.split(",")}
        chosen = [s for s in chosen if s not in skip]
    if not chosen:
        raise SystemExit("Порожній список стадій")
    return chosen


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = resolve_config(args)
    stages = resolve_stages(args)
    if cfg["camera_source"] == "auto":
        js = (_abs(cfg["json_path"]) if cfg["json_path"]
              else DATA_DIR / cfg["dataset"] / f"{cfg['dataset']}.json")
        cfg["camera_source"] = "earthstudio_json" if js.is_file() else "synthetic"
        print(f"camera_source=auto → {cfg['camera_source']}"
              f" ({'знайдено' if js.is_file() else 'немає'} {js})")
    if cfg["recenter"] and "train" in stages:
        raise SystemExit("❌ recenter=true разом зі стадією train заборонено: "
                         "хмара зсувається відносно поз COLMAP і тренування тихо ламається. "
                         "Запускай train без recenter (центрування робить сервер).")
    p = Paths(cfg)

    preflight(cfg, p, stages)

    t0 = time.perf_counter()
    for name in stages:
        RUNNERS[name](cfg, p, args.dry_run)

    if args.dry_run:
        print("\n(dry-run — нічого не виконано)")
    else:
        print(f"\n🎉 Готово за {time.perf_counter() - t0:.1f}s")
        print(f"   хмара: {p.cloud_in}")
        print(f"   splat: {p.splat_pt}")
        print(f"   далі:  AEROSPLAT_DATASET={p.dataset} "
              f"python src/server/main.py")
    return 0


def cli() -> int:
    """Console-script entry point: `aerosplat [run|doctor|serve] ...`."""
    _fix_console_encoding()
    argv = sys.argv[1:]
    cmd = argv[0] if argv and not argv[0].startswith("-") else "run"
    rest = argv[1:] if argv and not argv[0].startswith("-") else argv
    if cmd == "run":
        return main(rest)
    if cmd == "doctor":
        import doctor
        return doctor.main(rest) or 0
    if cmd == "serve":
        import runpy
        sys.argv = [str(SRC_DIR / "server" / "main.py"), *rest]
        runpy.run_path(str(SRC_DIR / "server" / "main.py"), run_name="__main__")
        return 0
    raise SystemExit(f"Невідома команда '{cmd}'. Доступні: run, doctor, serve")


from aerosplat.logging_utils import fix_console_encoding as _fix_console_encoding


if __name__ == "__main__":
    _fix_console_encoding()
    raise SystemExit(main())