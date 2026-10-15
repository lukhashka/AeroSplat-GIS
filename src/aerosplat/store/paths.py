"""Physical layout of the project tree (TARGET_ARCHITECTURE §7.1).

    <root>/                              AEROSPLAT_PROJECTS, default <repo>/projects
    ├── catalog.db                       SQLite index (aerosplat.store.db)
    └── <project>/
        ├── project.yaml
        └── sites/<site>/
            ├── site.yaml                origin (fixed), bbox, kind, taxonomy packs
            ├── captures/<capture>/
            │   ├── capture.yaml
            │   ├── images/              originals (read-only)
            │   ├── images_private/      after the privacy filter
            │   ├── images.jsonl         sha256, size, pose prior per image
            │   └── source/              adapter input (Earth Studio JSON, .esp, attribution)
            ├── epochs/<epoch>/
            │   ├── epoch.yaml
            │   └── runs/<run_id>/       DERIVED: may be deleted and recomputed
            │       ├── manifest.json, run.log
            │       ├── images/          frames prepared for SfM (copies with GPS EXIF)
            │       ├── known_poses/     COLMAP seed (outside sfm/: `sfm --fresh` wipes sfm/)
            │       ├── sfm/ mvs/ gs/
            │       └── reports/         registration.json, metrics.json, report.html
            ├── authored/                AUTHORED: never written by the pipeline
            │   ├── access_points.json   drawn/confirmed features (ACCESS_POINTS §3.1)
            │   └── audit.jsonl          append-only: who changed what, and what it was
            └── missions/
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from aerosplat.paths import PROJECT_DIR

CATALOG_FILE = "catalog.db"
ACCESS_POINTS_JSON = "access_points.json"
AUDIT_JSONL = "audit.jsonl"


def default_projects_root() -> Path:
    env = os.environ.get("AEROSPLAT_PROJECTS")
    return Path(env) if env else PROJECT_DIR / "projects"


@dataclass(frozen=True)
class RunDirs:
    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def log(self) -> Path:
        return self.root / "run.log"

    @property
    def images(self) -> Path:
        return self.root / "images"

    @property
    def known_poses(self) -> Path:
        return self.root / "known_poses"

    @property
    def sfm(self) -> Path:
        return self.root / "sfm"

    @property
    def mvs(self) -> Path:
        return self.root / "mvs"

    @property
    def gs(self) -> Path:
        return self.root / "gs"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def registration_report(self) -> Path:
        return self.reports / "registration.json"


@dataclass(frozen=True)
class CaptureDirs:
    root: Path

    @property
    def capture_yaml(self) -> Path:
        return self.root / "capture.yaml"

    @property
    def images(self) -> Path:
        return self.root / "images"

    @property
    def images_private(self) -> Path:
        return self.root / "images_private"

    @property
    def images_jsonl(self) -> Path:
        return self.root / "images.jsonl"

    @property
    def source(self) -> Path:
        return self.root / "source"


@dataclass(frozen=True)
class SiteDirs:
    root: Path

    @property
    def site_yaml(self) -> Path:
        return self.root / "site.yaml"

    @property
    def captures(self) -> Path:
        return self.root / "captures"

    @property
    def epochs(self) -> Path:
        return self.root / "epochs"

    @property
    def authored(self) -> Path:
        return self.root / "authored"

    @property
    def access_points(self) -> Path:
        """Authored access points: current state, rewritten atomically (ACCESS_POINTS §3.5)."""
        return self.authored / ACCESS_POINTS_JSON

    @property
    def audit_log(self) -> Path:
        """Append-only record of who changed what in `authored/`."""
        return self.authored / AUDIT_JSONL

    @property
    def missions(self) -> Path:
        return self.root / "missions"

    def capture(self, capture_id: str) -> CaptureDirs:
        return CaptureDirs(self.captures / capture_id)

    def epoch(self, epoch_id: str) -> Path:
        return self.epochs / epoch_id

    def epoch_yaml(self, epoch_id: str) -> Path:
        return self.epoch(epoch_id) / "epoch.yaml"

    def run(self, epoch_id: str, run_id: str) -> RunDirs:
        return RunDirs(self.epoch(epoch_id) / "runs" / run_id)


@dataclass(frozen=True)
class StorePaths:
    root: Path

    @property
    def catalog_db(self) -> Path:
        return self.root / CATALOG_FILE

    def project(self, project_id: str) -> Path:
        return self.root / project_id

    def project_yaml(self, project_id: str) -> Path:
        return self.project(project_id) / "project.yaml"

    def site(self, project_id: str, site_id: str) -> SiteDirs:
        return SiteDirs(self.project(project_id) / "sites" / site_id)
