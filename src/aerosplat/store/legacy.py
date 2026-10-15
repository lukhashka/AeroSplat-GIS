"""Migrate a legacy `data/<dataset>` layout into the project tree as a Site
(TARGET_ARCHITECTURE §7.4, ADR-0006). CLI: tools/migrate_dataset.py.

    data/<ds>/footage, <ds>.json, .esp, ImagerySources.txt -> captures/<id>/{images,source}
    data/<ds>/sparse_known                                   -> runs/<id>/known_poses
    data/raw_images_<ds>                                     -> runs/<id>/images
    data/colmap_output_<ds>/{database.db,sparse,...}         -> runs/<id>/sfm/
    data/colmap_output_<ds>/dense                            -> runs/<id>/mvs
    data/<ds>_{dense,cloud,splat}.pt                         -> runs/<id>/gs/{dense,cloud,splat}.pt
    data/<ds>/run/{manifest.json,run.log,metrics.json,...}   -> runs/<id>/ (+ reports/)

The Site origin defaults to the legacy run's ENU origin, so the migrated
run's coordinates are Site ENU up to the sphere -> WGS84 transition (measured
and stored in reports/registration.json; the run is tagged 'sphere-legacy').
Copies by default; `move=True` empties the legacy layout. Anything not listed
above (e.g. personal spreadsheets in data/<ds>/) is left where it is.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from aerosplat.geo.crs import GeoPoint
from aerosplat.geo.earthstudio_to_colmap import legacy_mean_origin, load_frames, report_for_json
from aerosplat.ingest.earthstudio import frame_prefix, import_earthstudio
from aerosplat.ingest.folder import import_folder
from aerosplat.ingest.images import transfer
from aerosplat.store.catalog import DEFAULT_EPOCH, Catalog, CatalogError, NotFoundError, new_run_id
from aerosplat.store.models import CaptureSource, RunStatus, Site
from aerosplat.store.paths import RunDirs
from aerosplat.store.scenes import CLOUD, REGISTRATION, SPLAT, is_georeferenced

LEGACY_EARTH_MODEL = "sphere-legacy"


@dataclass(frozen=True)
class LegacyDataset:
    data_dir: Path
    name: str

    @property
    def root(self) -> Path:
        return self.data_dir / self.name

    @property
    def footage(self) -> Path:
        return self.root / "footage"

    @property
    def json_path(self) -> Path:
        return self.root / f"{self.name}.json"

    @property
    def source_extras(self) -> list[Path]:
        return [self.root / f"{self.name}.esp", self.root / "ImagerySources.txt"]

    @property
    def known_poses(self) -> Path:
        return self.root / "sparse_known"

    @property
    def run_dir(self) -> Path:
        return self.root / "run"

    @property
    def colmap_out(self) -> Path:
        return self.data_dir / f"colmap_output_{self.name}"

    def run_transfers(self, run: RunDirs) -> list[tuple[Path, Path]]:
        """(legacy source, run destination) for every derived artefact that exists."""
        pairs: list[tuple[Path, Path]] = [
            (self.data_dir / f"raw_images_{self.name}", run.images),
            (self.known_poses, run.known_poses),
            (self.data_dir / f"{self.name}_dense.pt", run.gs / "dense.pt"),
            (self.data_dir / f"{self.name}_cloud.pt", run.gs / "cloud.pt"),
            (self.data_dir / f"{self.name}_splat.pt", run.gs / "splat.pt"),
            (self.data_dir / f"{self.name}_geo.txt", run.root / "images_geo.txt"),
        ]
        if self.colmap_out.is_dir():
            for entry in sorted(self.colmap_out.iterdir()):
                pairs.append((entry, run.mvs if entry.name == "dense" else run.sfm / entry.name))
        if self.run_dir.is_dir():
            for entry in sorted(self.run_dir.iterdir()):
                top = entry.name in ("manifest.json", "run.log")
                pairs.append((entry, (run.root if top else run.reports) / entry.name))
        return [(s, d) for s, d in pairs if s.exists()]

    def manifest(self) -> dict[str, Any]:
        path = self.run_dir / "manifest.json"
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    def legacy_enu_origin(self) -> Optional[GeoPoint]:
        """The ENU origin the legacy run was actually built in, if recorded."""
        path = self.known_poses / "geo_registration.json"
        if not path.is_file():
            return None
        o = json.loads(path.read_text(encoding="utf-8")).get("enu_origin") or {}
        return GeoPoint(lat=o["latitude"], lon=o["longitude"], h=o["altitude"])


@dataclass
class MigrationResult:
    site: Site
    capture_id: Optional[str] = None
    run_id: Optional[str] = None
    actions: list[str] = field(default_factory=list)
    left_behind: list[str] = field(default_factory=list)


def _run_id_from_manifest(manifest: dict[str, Any]) -> str:
    stamp = manifest.get("updated")
    try:
        when = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S%z") if stamp else None
    except ValueError:
        when = None
    return f"{new_run_id(when.astimezone(timezone.utc)) if when else 'unknown'}-legacy"


def choose_origin(ds: LegacyDataset, origin: Optional[GeoPoint]) -> tuple[GeoPoint, str]:
    if origin is not None:
        return origin, "given"
    legacy = ds.legacy_enu_origin()
    if legacy is not None:
        return legacy, "legacy run ENU origin (sparse_known/geo_registration.json)"
    if ds.json_path.is_file():
        _, frames = load_frames(ds.json_path)
        return legacy_mean_origin(frames), "mean of Earth Studio frames"
    raise CatalogError(f"no geo reference for '{ds.name}' (no geo_registration.json, no "
                       f"Earth Studio JSON): pass an explicit origin (--origin LAT,LON,H)")


def migrate_dataset(catalog: Catalog, data_dir: Path, dataset: str, *,
                    project_id: str = "default", site_id: Optional[str] = None,
                    origin: Optional[GeoPoint] = None, epoch_id: str = DEFAULT_EPOCH,
                    source: Optional[CaptureSource] = None, move: bool = False,
                    dry_run: bool = False,
                    log: Callable[[str], None] = print) -> MigrationResult:
    """Idempotent: existing project/site/epoch/capture/run entries are reused, not redone."""
    ds =LegacyDataset(Path(data_dir), dataset)
    if not ds.root.is_dir():
        raise NotFoundError(f"legacy dataset not found: {ds.root}")
    site_id = site_id or dataset.lower()
    site_origin, origin_source = choose_origin(ds, origin)
    is_es = ds.json_path.is_file()
    capture_id = (f"earthstudio-{frame_prefix(ds.json_path)}".lower() if is_es
                  else f"{(source or CaptureSource.handheld).value}-{dataset}".lower())
    manifest = ds.manifest()
    has_outputs = bool(manifest) or any(
        (ds.data_dir / f"{dataset}_{k}.pt").is_file() for k in ("dense", "cloud", "splat"))
    run_id = _run_id_from_manifest(manifest) if has_outputs else None

    # ── site ─────────────────────────────────────────────────────────────
    try:
        site = catalog.get_site(project_id, site_id)
        if origin is not None and site.origin != origin:
            raise CatalogError(f"site {site.key} exists with origin {site.origin.as_tuple()}; "
                               f"its origin cannot change to {origin.as_tuple()}")
        res = MigrationResult(site=site)
        res.actions.append(f"site {site.key} exists (origin kept)")
    except NotFoundError:
        site = Site(id=site_id, project_id=project_id, name=dataset, origin=site_origin)
        res = MigrationResult(site=site)
        res.actions.append(f"create site {site.key}, origin {site_origin.as_tuple()} "
                           f"({origin_source})")
        if not dry_run:
            catalog.ensure_project(project_id)
            catalog.create_site(site)
    if not dry_run:
        catalog.ensure_epoch(site, epoch_id)
    res.actions.append(f"epoch {epoch_id}")

    # ── capture ──────────────────────────────────────────────────────────
    in_catalog = _site_exists(catalog, site)   # False only in a dry run of a new site
    existing_captures = {c.id for c in catalog.list_captures(site)} if in_catalog else set()
    if capture_id in existing_captures:
        res.actions.append(f"capture {capture_id} exists, skipped")
    else:
        verb = "move" if move else "copy"
        res.actions.append(f"{verb} {ds.footage} -> capture {capture_id}"
                           + (f" (+ {ds.json_path.name}, Earth Studio)" if is_es else ""))
        if not dry_run:
            if is_es:
                import_earthstudio(catalog, site, ds.json_path, ds.footage,
                                   capture_id=capture_id, extra_files=ds.source_extras, move=move)
            else:
                import_folder(catalog, site, ds.footage, capture_id=capture_id,
                              source=source or CaptureSource.handheld, move=move,
                              notes=f"migrated from data/{dataset}/footage")
    res.capture_id = capture_id

    # ── legacy run ───────────────────────────────────────────────────────
    if run_id is not None:
        res.run_id = run_id
        existing_runs = {r.id for r in catalog.list_runs(site)} if in_catalog else set()
        if run_id in existing_runs:
            res.actions.append(f"run {run_id} exists, skipped")
        else:
            _migrate_run(catalog, ds, site, epoch_id, run_id, capture_id, manifest,
                         move=move, dry_run=dry_run, res=res)

    known = {ds.footage, ds.json_path, ds.known_poses, ds.run_dir, *ds.source_extras}
    res.left_behind = sorted(p.name for p in ds.root.iterdir() if p not in known)
    for line in res.actions:
        log(("[dry-run] " if dry_run else "") + line)
    if res.left_behind:
        log(f"left in {ds.root} (not migrated): {', '.join(res.left_behind)}")
    return res


def main(argv: Optional[list[str]] = None) -> int:
    """CLI: tools/migrate_dataset.py, `aerosplat catalog migrate-dataset`."""
    import argparse

    from aerosplat.paths import DATA_DIR

    ap = argparse.ArgumentParser(
        prog="migrate_dataset",
        description="Перенести data/<dataset> (legacy layout) у дерево проєктів як Site.")
    ap.add_argument("dataset", help="ім'я датасету в data/, напр. dataset_v6")
    ap.add_argument("--project", default="default")
    ap.add_argument("--site", default=None, help="id ділянки (за замовчуванням = dataset)")
    ap.add_argument("--origin", type=GeoPoint.parse, default=None, metavar="LAT,LON,H",
                    help="ENU-початок Site (WGS84); за замовчуванням — ENU-початок legacy run")
    ap.add_argument("--epoch", default=DEFAULT_EPOCH)
    ap.add_argument("--source", choices=[s.value for s in CaptureSource], default=None,
                    help="джерело кадрів, якщо це не експорт Earth Studio")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--projects-root", type=Path, default=None,
                    help="корінь дерева проєктів (за замовчуванням $AEROSPLAT_PROJECTS або projects/)")
    ap.add_argument("--move", action="store_true",
                    help="перемістити замість копіювання (legacy layout спорожніє)")
    ap.add_argument("--dry-run", action="store_true", help="лише показати план")
    args = ap.parse_args(argv)

    catalog = Catalog.open(args.projects_root)
    try:
        res = migrate_dataset(catalog, args.data_dir, args.dataset, project_id=args.project,
                              site_id=args.site, origin=args.origin, epoch_id=args.epoch,
                              source=CaptureSource(args.source) if args.source else None,
                              move=args.move, dry_run=args.dry_run)
    except (CatalogError, FileExistsError) as e:
        print(f"❌ {e}")
        return 1
    finally:
        catalog.close()
    if not args.dry_run:
        print(f"✅ {res.site.key}: capture={res.capture_id} run={res.run_id}")
        print(f"   далі: AEROSPLAT_DATASET={res.site.key} aerosplat serve")
    return 0


def _site_exists(catalog: Catalog, site: Site) -> bool:
    try:
        catalog.get_site(site.project_id, site.id)
        return True
    except NotFoundError:
        return False


def _migrate_run(catalog: Catalog, ds: LegacyDataset, site: Site, epoch_id: str, run_id: str,
                 capture_id: str, manifest: dict[str, Any], *, move: bool, dry_run: bool,
                 res: MigrationResult) -> None:
    run = catalog.run_dirs(site, epoch_id, run_id)
    transfers = ds.run_transfers(run)
    for src, dst in transfers:
        res.actions.append(f"{'move' if move else 'copy'} {src} -> {dst}")
    cfg = dict(manifest.get("config") or {})
    legacy_origin = ds.legacy_enu_origin()
    same_frame = legacy_origin is not None and legacy_origin == site.origin
    georeferenced = is_georeferenced(cfg) and same_frame
    if is_georeferenced(cfg) and not same_frame:
        res.actions.append("NOTE: legacy run ENU origin differs from the Site origin -- "
                           "registered as NOT georeferenced (re-run in the Site to fix)")
    res.actions.append(f"register run {run_id} ({LEGACY_EARTH_MODEL}, "
                       f"georeferenced={georeferenced})")
    if dry_run:
        return

    catalog.create_run(site, epoch_id, run_id, capture_ids=[capture_id], config=cfg,
                       earth_model=LEGACY_EARTH_MODEL, georeferenced=georeferenced,
                       git_commit=manifest.get("git_commit"),
                       notes=f"migrated from legacy data/{ds.name} layout")
    for src, dst in transfers:
        transfer(src, dst, move=move)
    if move:   # containers whose entries moved one by one; only ever removed when empty
        for d in (ds.colmap_out, ds.run_dir):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()

    # The Earth Studio JSON now lives in the capture (it was moved there if move=True).
    json_src = next((p for p in (catalog.site_dirs(site).capture(capture_id).source
                                 / ds.json_path.name, ds.json_path) if p.is_file()), None)
    if json_src is not None:
        report = report_for_json(json_src, site.origin, "site")
        report["run_frame"] = {
            "earth_model": LEGACY_EARTH_MODEL,
            "legacy_enu_origin": legacy_origin.model_dump() if legacy_origin else None,
            "note": "built before ADR-0006 on Earth Studio's sphere from `position`; "
                    "legacy_shift is how far its coordinates are from WGS84 Site ENU",
        }
        run.reports.mkdir(parents=True, exist_ok=True)
        run.registration_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if run.manifest.is_file():
        data = json.loads(run.manifest.read_text(encoding="utf-8"))
        data["migrated_from"] = {
            "layout": f"data/{ds.name}",
            "migrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": "move" if move else "copy",
            "paths": {str(d.relative_to(run.root)): str(s) for s, d in transfers},
            "earth_model": LEGACY_EARTH_MODEL,
        }
        data.update(site=site.key, epoch=epoch_id, run_id=run_id)
        run.manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    stages = (manifest.get("stages") or {}).values()
    cloud = run.gs / ("cloud.pt" if cfg.get("cloud_source") == "sparse" else "dense.pt")
    if not cloud.is_file():
        cloud = run.gs / ("dense.pt" if cloud.name == "cloud.pt" else "cloud.pt")
    products = {SPLAT: run.gs / "splat.pt", CLOUD: cloud, "sparse": run.sfm / "sparse" / "0",
                "fused_ply": run.mvs / "fused.ply", "known_poses": run.known_poses,
                "metrics": run.reports / "metrics.json", "report": run.reports / "report.html",
                REGISTRATION: run.registration_report}
    ok = (any(p.is_file() for p in (products[SPLAT], products[CLOUD]))
          and not any(st.get("status") == "failed" for st in stages))
    catalog.finish_run(site, epoch_id, run_id, status=RunStatus.ok if ok else RunStatus.failed,
                       products=products)
