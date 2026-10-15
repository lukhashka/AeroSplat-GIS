#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aerosplat.starter — єдина точка входу в пайплайн AeroSplat-GIS.

footage → EXIF GPS → COLMAP SfM → COLMAP MVS → .pt cloud → 3DGS splat → eval
        → georef (Earth Studio: продукти зі сферичного ENU у WGS84 Site ENU)

Приклади:
    aerosplat run --site default/dataset_v6   # ділянка з каталогу (ADR-0006): новий run
    aerosplat run --site dataset_v6 --resume  # продовжити останній run ділянки
    aerosplat run                          # legacy: data/<ds>, дефолтний датасет
    aerosplat run --dataset dataset_v6     # legacy: data/<ds> (доки міграція не завершена)
    aerosplat run --from sfm               # почати з SfM (кадри вже геотеговані)
    aerosplat run --only train
    aerosplat run --stages exif,sfm,dense
    aerosplat run --dry-run                # тільки показати команди
    aerosplat run --config my_run.json     # усі параметри з файлу
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from aerosplat.config import RunConfig, build_config
from aerosplat.logging_utils import fix_console_encoding as _fix_console_encoding
from aerosplat.logging_utils import get_logger
from aerosplat.pipeline.layout import DATA_DIR, RunLayout, abs_path
from aerosplat.pipeline.runner import StageCallback, StageError, preflight, run_stages

STAGES = ["geo", "exif", "sfm", "dense", "parse", "train", "eval", "georef"]

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
    ap.add_argument("--dataset", help="legacy layout data/<dataset> (без Site)")
    ap.add_argument("--site", help="ділянка з каталогу: project/site або id, якщо він унікальний")
    ap.add_argument("--epoch", default="baseline", help="епоха ділянки (з --site)")
    ap.add_argument("--capture", help="захоплення ділянки (з --site; потрібне, якщо їх кілька)")
    ap.add_argument("--run-id", help="id run-а (з --site); за замовчуванням новий, з часу")
    ap.add_argument("--projects-root", type=Path,
                    help="корінь дерева проєктів (за замовчуванням $AEROSPLAT_PROJECTS або projects/)")
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
    ap.add_argument("--rasterizer", choices=["auto", "cuda", "torch", "gsplat"],
                    help="gsplat: SH, antialiased, MCMC (--set sh_degree=..., densify_strategy=...); "
                         "на Windows запускати через tools\\gsplat_env.bat")
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


def main(argv=None, on_event: Optional[StageCallback] = None,
         on_run: Optional[Callable[[str, str], None]] = None) -> int:
    """`aerosplat run`. Job processes (aerosplat.jobs.child) pass `on_event` for stage
    progress and `on_run(site_key, run_id)` to learn the run before its stages start."""
    args = build_parser().parse_args(argv)
    log = get_logger()
    cfg = resolve_config(args)
    stages = resolve_stages(args)
    if args.site and args.dataset:
        raise SystemExit("❌ --site і --dataset взаємовиключні: --dataset — це legacy layout data/<ds>")
    if cfg["recenter"] and "train" in stages:
        raise SystemExit("❌ recenter=true разом зі стадією train заборонено: "
                         "хмара зсувається відносно поз COLMAP і тренування тихо ламається. "
                         "Запускай train без recenter (центрування робить сервер).")

    site_run = None
    if args.site:
        from aerosplat.pipeline.site_run import SiteRun
        from aerosplat.store.catalog import CatalogError

        try:
            site_run = SiteRun.prepare(cfg, site_ref=args.site, epoch_id=args.epoch,
                                       capture_id=args.capture, run_id=args.run_id,
                                       resume=args.resume, projects_root=args.projects_root)
        except CatalogError as e:
            raise SystemExit(f"❌ {e}") from None
        layout = site_run.layout
        log.info(f"camera_source → {cfg['camera_source']} (захоплення {site_run.capture.id})")
    else:
        if cfg["camera_source"] == "auto":
            js = (abs_path(cfg["json_path"]) if cfg["json_path"]
                  else DATA_DIR / cfg["dataset"] / f"{cfg['dataset']}.json")
            cfg["camera_source"] = "earthstudio_json" if js.is_file() else "synthetic"
            log.info(f"camera_source=auto → {cfg['camera_source']}"
                     f" ({'знайдено' if js.is_file() else 'немає'} {js})")
        layout = RunLayout.from_config(cfg)

    preflight(cfg, layout, stages)

    t0 = time.perf_counter()
    if site_run is not None and not args.dry_run:
        site_run.begin(cfg)
        if on_run is not None:
            on_run(site_run.site.key, site_run.run_id)
    try:
        run_stages(cfg, layout, stages, dry_run=args.dry_run, resume=args.resume,
                   on_event=on_event)
    except StageError as e:
        if site_run is not None and not args.dry_run:
            site_run.end(cfg, ok=False)
        raise SystemExit(f"\n❌ {e}") from None
    except BaseException:
        if site_run is not None and not args.dry_run:
            site_run.end(cfg, ok=False)
        raise
    else:
        if site_run is not None and not args.dry_run:
            site_run.end(cfg, ok=True)
    finally:
        if site_run is not None:
            site_run.close()

    if args.dry_run:
        log.info("\n(dry-run — нічого не виконано)")
    else:
        log.info(f"\n🎉 Готово за {time.perf_counter() - t0:.1f}s")
        log.info(f"   хмара: {layout.cloud_in}")
        log.info(f"   splat: {layout.splat_pt}")
        if layout.splat_wgs84.is_file() or layout.cloud_wgs84.is_file():
            log.info(f"   WGS84 Site ENU: {layout.cloud_wgs84.name}, {layout.splat_wgs84.name} "
                     f"(їх реєструє каталог і показує сервер)")
        log.info(f"   журнал: {layout.manifest_path}")
        log.info(f"   далі:  AEROSPLAT_DATASET={layout.dataset} aerosplat serve")
    return 0


def cli() -> int:
    """Console-script entry point: `aerosplat [run|doctor|serve|catalog|worker|up] ...`."""
    _fix_console_encoding()
    argv = sys.argv[1:]
    cmd = argv[0] if argv and not argv[0].startswith("-") else "run"
    rest = argv[1:] if argv and not argv[0].startswith("-") else argv
    if cmd == "run":
        return main(rest)
    if cmd == "doctor":
        from aerosplat import doctor
        return doctor.main(rest) or 0
    if cmd == "catalog":
        from aerosplat.store import cli as catalog_cli
        return catalog_cli.main(rest)
    if cmd == "serve":
        import runpy
        sys.argv = ["aerosplat serve", *rest]
        runpy.run_module("aerosplat.server.main", run_name="__main__")
        return 0
    if cmd == "worker":
        from aerosplat.jobs import worker
        return worker.main(rest)
    if cmd == "up":
        from aerosplat.server import up
        return up.main(rest)
    raise SystemExit(f"Невідома команда '{cmd}'. Доступні: run, doctor, serve, catalog, worker, up")



if __name__ == "__main__":
    _fix_console_encoding()
    raise SystemExit(main())