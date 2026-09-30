"""All filesystem paths of one run, derived from the config in one place."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_DIR / "data"


def abs_path(p: str | Path, root: Path = PROJECT_DIR) -> Path:
    p = Path(p)
    return p if p.is_absolute() else root / p


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
    log_path: Path

    @classmethod
    def from_config(cls, cfg: dict[str, Any], data_dir: Path = DATA_DIR,
                    project_dir: Path = PROJECT_DIR) -> "RunLayout":
        ds = cfg["dataset"]
        colmap_out = data_dir / f"colmap_output_{ds}"
        sparse = colmap_out / "sparse" / "0"
        dense = colmap_out / "dense"
        dense_pt = data_dir / f"{ds}_dense.pt"
        cloud_pt = data_dir / f"{ds}_cloud.pt"
        run_dir = data_dir / ds / "run"
        return cls(
            dataset=ds,
            data_dir=data_dir,
            project_dir=project_dir,
            footage=abs_path(cfg["footage_dir"], project_dir),
            # sfm_runner / dense_runner hard-code this naming scheme.
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
            cloud_in=dense_pt if cfg["cloud_source"] == "dense" else cloud_pt,
            json_path=(abs_path(cfg["json_path"], project_dir) if cfg["json_path"]
                       else data_dir / ds / f"{ds}.json"),
            # NOT under colmap_out: sfm --fresh wipes that directory, and known
            # poses are ground-truth INPUT, not regenerable COLMAP output.
            known_poses_dir=data_dir / ds / "sparse_known",
            geo_registration_json=data_dir / ds / f"{ds}_geo_registration.json",
            run_dir=run_dir,
            manifest_path=run_dir / "manifest.json",
            log_path=run_dir / "run.log",
        )
