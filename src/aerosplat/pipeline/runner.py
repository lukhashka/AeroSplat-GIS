"""Execute stages in-process, recording each one in the run manifest."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from aerosplat.logging_utils import get_logger
from aerosplat.pipeline.layout import RunLayout
from aerosplat.pipeline.manifest import Manifest, fingerprint
from aerosplat.pipeline.stages import STAGES, Stage


class StageError(RuntimeError):
    """A stage reported failure without raising (non-zero return / SystemExit)."""


def _fingerprint(stage: Stage, cfg: dict[str, Any], L: RunLayout) -> str:
    values = {k: cfg[k] for k in stage.keys}
    values.update(dataset=L.dataset, footage=str(L.footage), json=str(L.json_path))
    if L.origin is not None:   # absent for legacy runs, keeping their old fingerprints
        values["origin"] = L.origin.as_tuple()
    if stage.version:          # likewise absent for stages that never changed convention
        values["version"] = stage.version
    return fingerprint(values)


def _invoke(stage: Stage, argv: list[str], cfg: dict[str, Any]) -> None:
    try:
        rc = stage.entry(cfg)(argv)
    except SystemExit as e:  # scripts use SystemExit("message") / sys.exit(code)
        if e.code in (None, 0):
            return
        raise StageError(f"stage '{stage.name}' exited: {e.code}") from None
    if rc not in (None, 0):
        raise StageError(f"stage '{stage.name}' returned {rc}")


def run_stages(cfg: dict[str, Any], L: RunLayout, names: list[str], *,
               dry_run: bool = False, resume: bool = False) -> None:
    log = get_logger(log_file=None if dry_run else L.log_path)
    manifest = Manifest.load(L.manifest_path)
    if not dry_run:
        site_info = None
        if L.is_site_run and L.origin is not None:
            site_info = {"site": L.site_key, "epoch": L.epoch_id, "run_id": L.run_id,
                         "capture": L.capture_id, "origin": L.origin.model_dump()}
        manifest.start_run(L.dataset, cfg, L.project_dir, extra=site_info)
    upstream_ran = False

    for name in names:
        stage = STAGES[name]
        log.info(f"\n{'=' * 70}\n▶  {stage.label}\n{'=' * 70}")

        reason = stage.skip_reason(cfg)
        if reason:
            log.info(f"⏭  [{name}] {reason}, пропущено\n")
            continue

        argv = stage.argv(cfg, L)
        inputs, outputs = stage.inputs(cfg, L), stage.outputs(cfg, L)
        fp = _fingerprint(stage, cfg, L)

        if dry_run:
            printable = " ".join(f'"{a}"' if " " in a else a for a in argv)
            log.info(f"$ python -m {stage.module(cfg)} {printable}\n")
            continue

        if resume and not upstream_ran and manifest.is_fresh(name, fp, outputs):
            log.info(f"⏭  [{name}] вже виконано з тим самим конфігом (--resume), пропущено\n")
            continue

        upstream_ran = True
        t0 = time.perf_counter()
        try:
            _invoke(stage, argv, cfg)
        except BaseException as e:
            manifest.record(name, status="failed", fingerprint=fp,
                            seconds=time.perf_counter() - t0, inputs=inputs,
                            outputs=outputs, error=f"{type(e).__name__}: {e}")
            log.error(f"\n❌ Стадія '{name}' впала після {time.perf_counter() - t0:.1f}s")
            raise
        dt = time.perf_counter() - t0
        manifest.record(name, status="ok", fingerprint=fp, seconds=dt,
                        inputs=inputs, outputs=outputs)
        log.info(f"\n✅ {stage.label} — {dt:.1f}s")


def preflight(cfg: dict[str, Any], L: RunLayout, names: list[str]) -> None:
    log = get_logger()
    log.info("── Preflight ─────────────────────────────────────────────────")
    log.info(f"  проєкт:      {L.project_dir}")
    if L.is_site_run:
        assert L.origin is not None
        log.info(f"  ділянка:     {L.site_key}  епоха={L.epoch_id}  run={L.run_id}  "
                 f"захоплення={L.capture_id}")
        log.info(f"  Site ENU:    lat={L.origin.lat:.8f} lon={L.origin.lon:.8f} h={L.origin.h:.3f}")
        log.info(f"  run:         {L.run_dir}")
    else:
        log.info(f"  датасет:     {L.dataset}  (legacy data/<ds>, без Site)")
    log.info(f"  джерело камер: {cfg['camera_source']}")
    if cfg["camera_source"] == "earthstudio_json":
        log.info(f"  JSON:        {L.json_path}")
        log.info(f"  known poses: {L.known_poses_dir}  (refine={cfg['refine_known_poses']})")
    log.info(f"  кадри:       {L.footage}")
    log.info(f"  геотеговані: {L.geotagged}")
    log.info(f"  COLMAP out:  {L.colmap_out}")
    log.info(f"  хмара .pt:   {L.cloud_in}  ({cfg['cloud_source']})")
    log.info(f"  splat .pt:   {L.splat_pt}")
    log.info(f"  manifest:    {L.manifest_path}")
    log.info(f"  стадії:      {' → '.join(names)}")

    for name in names:
        stage = STAGES[name]
        if stage.skip_reason(cfg):
            continue
        script: Path = stage.script(cfg)
        if not script.is_file():
            raise SystemExit(f"❌ Не знайдено скрипт: {script}")

    if {"sfm", "dense"} & set(names) and shutil.which(cfg["colmap_bin"]) is None:
        log.warning(f"  ⚠️  '{cfg['colmap_bin']}' не знайдено в PATH — "
                    f"задай --colmap-bin або COLMAP_BIN")

    first = next((STAGES[n] for n in names if not STAGES[n].skip_reason(cfg)), None)
    if first:
        for need in first.inputs(cfg, L):
            if not need.exists():
                log.warning(f"  ⚠️  вхід для стадії '{first.name}' не існує: {need}")
    log.info("")
