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
import sys
import time
from pathlib import Path

from aerosplat.config import RunConfig, build_config
from aerosplat.logging_utils import fix_console_encoding as _fix_console_encoding
from aerosplat.logging_utils import get_logger
from aerosplat.pipeline.layout import DATA_DIR, RunLayout, abs_path
from aerosplat.pipeline.runner import StageError, preflight, run_stages

SRC_DIR = Path(__file__).resolve().parent
STAGES = ["geo", "exif", "sfm", "dense", "parse", "train"]

# Параметри: src/aerosplat/config.py (RunConfig, валідація pydantic).
# Перекриття: --config run.yaml|json, прапорці CLI, --set KEY=VALUE.
# Стадії: src/aerosplat/pipeline/stages.py (виконуються в процесі, не через subprocess).
CONFIG: dict = RunConfig().to_dict()


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
    ap.add_argument("--resume", action="store_true",
                    help="пропустити стадії, що вже успішно виконані з тим самим конфігом "
                         "(за manifest.json) і чиї виходи існують")
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
    log = get_logger()
    cfg = resolve_config(args)
    stages = resolve_stages(args)
    if cfg["camera_source"] == "auto":
        js = (abs_path(cfg["json_path"]) if cfg["json_path"]
              else DATA_DIR / cfg["dataset"] / f"{cfg['dataset']}.json")
        cfg["camera_source"] = "earthstudio_json" if js.is_file() else "synthetic"
        log.info(f"camera_source=auto → {cfg['camera_source']}"
                 f" ({'знайдено' if js.is_file() else 'немає'} {js})")
    if cfg["recenter"] and "train" in stages:
        raise SystemExit("❌ recenter=true разом зі стадією train заборонено: "
                         "хмара зсувається відносно поз COLMAP і тренування тихо ламається. "
                         "Запускай train без recenter (центрування робить сервер).")
    layout = RunLayout.from_config(cfg)

    preflight(cfg, layout, stages)

    t0 = time.perf_counter()
    try:
        run_stages(cfg, layout, stages, dry_run=args.dry_run, resume=args.resume)
    except StageError as e:
        raise SystemExit(f"\n❌ {e}") from None

    if args.dry_run:
        log.info("\n(dry-run — нічого не виконано)")
    else:
        log.info(f"\n🎉 Готово за {time.perf_counter() - t0:.1f}s")
        log.info(f"   хмара: {layout.cloud_in}")
        log.info(f"   splat: {layout.splat_pt}")
        log.info(f"   журнал: {layout.manifest_path}")
        log.info(f"   далі:  AEROSPLAT_DATASET={layout.dataset} python src/server/main.py")
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



if __name__ == "__main__":
    _fix_console_encoding()
    raise SystemExit(main())