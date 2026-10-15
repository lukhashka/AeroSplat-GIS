"""`aerosplat run --site`: resolve a Site/Epoch/Capture into a RunLayout and keep
the catalog's run record in step with the pipeline (ADR-0006)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from aerosplat.paths import PROJECT_DIR
from aerosplat.pipeline.layout import RunLayout
from aerosplat.pipeline.manifest import git_commit
from aerosplat.store.catalog import Catalog, new_run_id
from aerosplat.store.models import Capture, CaptureSource, RunStatus, Site
from aerosplat.store.scenes import (CLOUD, CLOUD_RECON, REGISTRATION, SPLAT, SPLAT_RECON,
                                    is_georeferenced, run_earth_model)


@dataclass
class SiteRun:
    catalog: Catalog
    site: Site
    epoch_id: str
    run_id: str
    capture: Capture
    layout: RunLayout
    exists: bool

    @classmethod
    def prepare(cls, cfg: dict[str, Any], *, site_ref: str, epoch_id: str,
                capture_id: Optional[str], run_id: Optional[str], resume: bool,
                projects_root: Optional[Path]) -> "SiteRun":
        """Resolve everything without writing to the catalog (safe for --dry-run).

        Mutates cfg["camera_source"] when it is "auto".
        """
        catalog = Catalog.open(projects_root)
        site = catalog.find_site(site_ref)
        if capture_id:
            capture = catalog.get_capture(site, capture_id)
        else:
            captures = catalog.list_captures(site)
            if not captures:
                raise SystemExit(f"❌ У ділянці {site.key} немає захоплень. Спершу: "
                                 f"aerosplat catalog import-earthstudio --site {site.key} ...")
            if len(captures) > 1:
                raise SystemExit(f"❌ У {site.key} кілька захоплень — обери --capture: "
                                 f"{', '.join(c.id for c in captures)}")
            capture = captures[0]

        has_epoch = epoch_id in {e.id for e in catalog.list_epochs(site)}
        runs = catalog.list_runs(site, epoch_id) if has_epoch else []
        if run_id is None and resume and runs:
            run_id = runs[-1].id          # `--resume` alone continues the latest run
        run_id = run_id or new_run_id()
        exists = run_id in {r.id for r in runs}

        dirs = catalog.site_dirs(site)
        if cfg["camera_source"] == "auto":
            json_ok = bool(capture.metadata_file) and (
                dirs.capture(capture.id).root / str(capture.metadata_file)).is_file()
            cfg["camera_source"] = ("earthstudio_json"
                                    if capture.source == CaptureSource.earthstudio and json_ok
                                    else "synthetic")
        layout = RunLayout.for_site_run(cfg, site=site, site_dirs=dirs, capture=capture,
                                        epoch_id=epoch_id, run_id=run_id)
        return cls(catalog, site, epoch_id, run_id, capture, layout, exists)

    def begin(self, cfg: dict[str, Any]) -> None:
        earth_model = run_earth_model(cfg)
        if self.exists:
            # A run's earth model is fixed at creation; continuing one built
            # under another convention would mix frames inside it.
            old = self.catalog.get_run(self.site, self.epoch_id, self.run_id).earth_model
            if old != earth_model:
                raise SystemExit(f"❌ run {self.run_id} побудовано з earth_model={old}, поточний "
                                 f"конвеєр дає {earth_model} — запусти новий run "
                                 f"(без --resume/--run-id)")
        self.catalog.ensure_epoch(self.site, self.epoch_id)
        if not self.exists:
            self.catalog.create_run(self.site, self.epoch_id, self.run_id,
                                    capture_ids=[self.capture.id], config=cfg,
                                    earth_model=earth_model, georeferenced=is_georeferenced(cfg),
                                    git_commit=git_commit(PROJECT_DIR))

    def end(self, cfg: dict[str, Any], ok: bool) -> None:
        L = self.layout
        products = {"sparse": L.sparse, "fused_ply": L.fused_ply,
                    "known_poses": L.known_poses_dir, "metrics": L.metrics_json,
                    "report": L.report_html, REGISTRATION: L.registration_report}
        if is_georeferenced(cfg):
            # Served products must be in WGS84 Site ENU: only the georef stage's
            # outputs qualify; the sphere-ENU originals are kept for reference.
            products.update({SPLAT: L.splat_wgs84, CLOUD: L.cloud_wgs84,
                             SPLAT_RECON: L.splat_pt, CLOUD_RECON: L.cloud_in})
        else:
            products.update({SPLAT: L.splat_pt, CLOUD: L.cloud_in})
        self.catalog.finish_run(self.site, self.epoch_id, self.run_id,
                                status=RunStatus.ok if ok else RunStatus.failed,
                                products=products, georeferenced=is_georeferenced(cfg))

    def close(self) -> None:
        self.catalog.close()
