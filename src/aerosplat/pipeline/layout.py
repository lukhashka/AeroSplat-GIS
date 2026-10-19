"""All filesystem paths of one run, derived in one place.

`RunLayout` is a *view*: stages only ever see its fields, never where they
come from. Two sources exist while the migration to the project tree is in
progress (ADR-0006):

* `for_site_run` -- the project tree: inputs from a Site's capture, outputs
  under `sites/<site>/epochs/<epoch>/runs/<run_id>/` (aerosplat.store.paths);
* `from_config` -- the legacy `data/<dataset>` naming (`aerosplat run --dataset`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from aerosplat.geo.crs import GeoPoint
from aerosplat.paths import DATA_DIR, PROJECT_DIR

if TYPE_CHECKING:
    from aerosplat.store.models import BBox, Capture, Site
    from aerosplat.store.paths import SiteDirs


def abs_path(p: str | Path, root: Path = PROJECT_DIR) -> Path:
    p = Path(p)
    return p if p.is_absolute() else root / p


def _wgs84(pt: Path) -> Path:
    """gs/dense.pt -> gs/dense_wgs84.pt (data/<ds>_splat.pt -> data/<ds>_splat_wgs84.pt)."""
    return pt.with_name(f"{pt.stem}_wgs84{pt.suffix}")


@dataclass(frozen=True)
class RunLayout:
    dataset: str
    data_dir: Path
    project_dir: Path
    footage: Path
    geotagged: Path
    colmap_out: Path
    sparse: Path
    dense: Path
    fused_ply: Path
    points3d: Path
    cloud_pt: Path
    dense_pt: Path
    splat_pt: Path
    geo_txt: Path
    cloud_in: Path
    json_path: Path
    known_poses_dir: Path
    geo_registration_json: Path
    run_dir: Path
    manifest_path: Path
    reports_dir: Path
    metrics_json: Path
    report_html: Path
    registration_report: Path
    log_path: Path
    # Products mapped to WGS84 Site ENU by the georef stage (Earth Studio runs
    # reconstruct in sphere ENU, ADR-0006): what the catalog registers and serves.
    cloud_wgs84: Path
    splat_wgs84: Path
    # Semantic layer (terrain, buildings, facades) derived from the mapped cloud.
    semantics_dir: Path
    # Occupancy grid + distance field, built from the same cloud and the semantics terrain.
    occupancy_dir: Path
    # Earth Studio frames are matched as '<frame_prefix>_<n>.<ext>'.
    frame_prefix: str
    # Site ENU origin; None only for legacy runs (origin = mean of the frames).
    origin: Optional[GeoPoint] = None
    # The Site's WGS84 bbox, if it has one: the default extent of the occupancy grid.
    site_bbox: Optional["BBox"] = None
    site_key: Optional[str] = None
    epoch_id: Optional[str] = None
    run_id: Optional[str] = None
    capture_id: Optional[str] = None

    @property
    def is_site_run(self) -> bool:
        return self.site_key is not None

    @classmethod
    def from_config(cls, cfg: dict[str, Any], data_dir: Path = DATA_DIR,
                    project_dir: Path = PROJECT_DIR) -> "RunLayout":
        """Legacy layout: data/raw_images_<ds>, data/colmap_output_<ds>, data/<ds>_*.pt."""
        ds = cfg["dataset"]
        colmap_out = data_dir / f"colmap_output_{ds}"
        sparse = colmap_out / "sparse" / "0"
        dense = colmap_out / "dense"
        dense_pt = data_dir / f"{ds}_dense.pt"
        cloud_pt = data_dir / f"{ds}_cloud.pt"
        run_dir = data_dir / ds / "run"
        # NOT under colmap_out: sfm --fresh wipes that directory, and known
        # poses are ground-truth INPUT, not regenerable COLMAP output.
        known_poses = data_dir / ds / "sparse_known"
        cloud_in = dense_pt if cfg["cloud_source"] == "dense" else cloud_pt
        return cls(
            dataset=ds,
            data_dir=data_dir,
            project_dir=project_dir,
            footage=abs_path(cfg["footage_dir"], project_dir),
            geotagged=data_dir / f"raw_images_{ds}",
            colmap_out=colmap_out,
            sparse=sparse,
            dense=dense,
            fused_ply=dense / "fused.ply",
            points3d=sparse / "points3D.bin",
            cloud_pt=cloud_pt,
            dense_pt=dense_pt,
            splat_pt=data_dir / f"{ds}_splat.pt",
            geo_txt=data_dir / f"{ds}_geo.txt",
            cloud_in=cloud_in,
            json_path=(abs_path(cfg["json_path"], project_dir) if cfg["json_path"]
                       else data_dir / ds / f"{ds}.json"),
            known_poses_dir=known_poses,
            geo_registration_json=known_poses / "geo_registration.json",
            run_dir=run_dir,
            manifest_path=run_dir / "manifest.json",
            reports_dir=run_dir,
            metrics_json=run_dir / "metrics.json",
            report_html=run_dir / "report.html",
            registration_report=known_poses / "registration_report.json",
            log_path=run_dir / "run.log",
            cloud_wgs84=_wgs84(cloud_in),
            splat_wgs84=_wgs84(data_dir / f"{ds}_splat.pt"),
            semantics_dir=run_dir / "semantics",
            occupancy_dir=run_dir / "occupancy",
            frame_prefix=ds,
        )

    @classmethod
    def for_site_run(cls, cfg: dict[str, Any], *, site: "Site", site_dirs: "SiteDirs",
                     capture: "Capture", epoch_id: str, run_id: str,
                     project_dir: Path = PROJECT_DIR) -> "RunLayout":
        """Project-tree layout: one capture of `site` in, one run directory out."""
        cdirs = site_dirs.capture(capture.id)
        run = site_dirs.run(epoch_id, run_id)
        colmap_out = run.sfm
        sparse = colmap_out / "sparse" / "0"
        dense_pt = run.gs / "dense.pt"
        cloud_pt = run.gs / "cloud.pt"
        json_path = (cdirs.root / capture.metadata_file if capture.metadata_file
                     else cdirs.source / "earthstudio.json")
        cloud_in = dense_pt if cfg["cloud_source"] == "dense" else cloud_pt
        return cls(
            dataset=site.key,
            data_dir=site_dirs.root,
            project_dir=project_dir,
            footage=cdirs.images,
            geotagged=run.images,
            colmap_out=colmap_out,
            sparse=sparse,
            dense=run.mvs,
            fused_ply=run.mvs / "fused.ply",
            points3d=sparse / "points3D.bin",
            cloud_pt=cloud_pt,
            dense_pt=dense_pt,
            splat_pt=run.gs / "splat.pt",
            geo_txt=run.root / "images_geo.txt",
            cloud_in=cloud_in,
            json_path=json_path,
            known_poses_dir=run.known_poses,
            geo_registration_json=run.known_poses / "geo_registration.json",
            run_dir=run.root,
            manifest_path=run.manifest,
            reports_dir=run.reports,
            metrics_json=run.reports / "metrics.json",
            report_html=run.reports / "report.html",
            registration_report=run.registration_report,
            log_path=run.log,
            cloud_wgs84=_wgs84(cloud_in),
            splat_wgs84=_wgs84(run.gs / "splat.pt"),
            semantics_dir=run.root / "semantics",
            occupancy_dir=run.root / "occupancy",
            frame_prefix=capture.frame_prefix or site.id,
            origin=site.origin,
            site_bbox=site.bbox,
            site_key=site.key,
            epoch_id=epoch_id,
            run_id=run_id,
            capture_id=capture.id,
        )
