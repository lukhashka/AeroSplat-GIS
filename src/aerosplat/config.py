"""Validated run configuration (single source of truth for pipeline parameters).

Replaces the free-form CONFIG dict in starter.py. A run is described by a
`RunConfig`; YAML/JSON files and CLI `--set KEY=VALUE` overrides are merged
into it and validated once, so typos and bad values fail before any stage runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    dataset: str = "dataset_v6"

    # -- input source ------------------------------------------------------
    footage_dir: str = "data/dataset_v6/footage"
    # "auto": Earth Studio JSON if present, otherwise synthetic.
    camera_source: Literal["auto", "synthetic", "earthstudio_json"] = "auto"
    json_path: Optional[str] = None
    known_poses: bool = True
    refine_known_poses: bool = True

    # -- EXIF / telemetry (synthetic only) --------------------------------
    telemetry_csv: Optional[str] = None
    lat: float = Field(47.0052, ge=-90, le=90)
    lon: float = Field(34.9164, ge=-180, le=180)
    altitude: float = 441.0
    radius: float = Field(250.0, gt=0)
    arc: float = Field(360.0, gt=0, le=360)
    start_bearing: float = 0.0
    counter_clockwise: bool = False
    hfov: Optional[float] = Field(None, gt=0, lt=180)
    camera_make: str = "GoogleEarthStudio"
    camera_model: str = "OrbitVirtualCam"
    colmap_geo: bool = True

    # -- SfM ---------------------------------------------------------------
    fresh: bool = True
    matcher: Literal["exhaustive", "sequential", "spatial"] = "sequential"
    seq_overlap: int = Field(10, ge=1)
    vocab_tree: Optional[str] = None
    spatial_neighbors: int = Field(30, ge=1)
    spatial_max_distance: float = Field(200.0, gt=0)
    mapper: Literal["colmap", "glomap"] = "colmap"
    sfm_max_image_size: int = Field(1600, ge=64)
    # Freeze focal length in BA only when it is truly known.
    known_intrinsics: bool = False
    image_width: int = Field(1920, ge=1)
    image_height: int = Field(1080, ge=1)
    fov_deg: float = Field(60.0, gt=0, lt=180)

    # -- dense MVS ---------------------------------------------------------
    gpu_index: str = "0"
    dense_max_image_size: int = Field(2000, ge=64)
    window_step: int = Field(2, ge=1)
    num_samples: int = Field(10, ge=1)
    num_iterations: int = Field(3, ge=1)
    cache_size: int = Field(6, ge=1)
    min_num_pixels: int = Field(8, ge=1)
    max_reproj_error: float = Field(1.5, gt=0)
    skip_undistort: bool = False
    skip_stereo: bool = False

    # -- cloud -> .pt ------------------------------------------------------
    cloud_source: Literal["dense", "sparse"] = "dense"
    mad_k: float = Field(0.0, ge=0)
    max_points: Optional[int] = Field(None, ge=1)
    recenter: bool = False
    cloud_seed: int = 0
    max_error: Optional[float] = Field(None, gt=0)
    min_track: Optional[int] = Field(None, ge=1)

    # -- 3DGS --------------------------------------------------------------
    iterations: int = Field(5500, ge=1)
    rasterizer: Literal["auto", "cuda", "torch"] = "auto"
    max_res: int = Field(1024, ge=64)
    max_gaussians: int = Field(2_000_000, ge=1)
    init_headroom: float = Field(0.7, gt=0, le=1)
    init_scale: Optional[float] = Field(None, gt=0)
    densify_from: int = Field(500, ge=0)
    densify_until_frac: float = Field(0.5, gt=0, le=1)
    densify_interval: int = Field(100, ge=1)
    opacity_reset: int = Field(3000, ge=0)
    grad_threshold: float = Field(0.0002, gt=0)
    chunk: int = Field(64, ge=1)
    raster_checkpoint: bool = False
    warmup_iters: int = Field(0, ge=0)
    warmup_scale: float = Field(0.25, gt=0, le=1)
    max_screen_radius: float = Field(20.0, ge=0)
    scale_reg: float = Field(0.01, ge=0)
    max_aniso_ratio: float = Field(10.0, gt=1)
    save_every: int = Field(0, ge=0)
    train_seed: int = 0

    # -- evaluation (Фаза 2.1) ---------------------------------------------
    # Кожен N-й кадр — test (не тренується). 0 = без спліту і без стадії eval.
    eval_every: int = Field(8, ge=0)
    eval_interval: int = Field(1000, ge=1)
    eval_lpips: bool = True

    colmap_bin: str = "colmap"

    @model_validator(mode="after")
    def _check_consistency(self) -> "RunConfig":
        if self.warmup_iters >= self.iterations and self.warmup_iters:
            raise ValueError("warmup_iters must be smaller than iterations")
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


def load_config_file(path: Path) -> dict[str, Any]:
    """Read a .yaml/.yml/.json file into a plain dict (unvalidated)."""
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def build_config(file: Path | None = None, overrides: dict[str, Any] | None = None
                 ) -> RunConfig:
    """Defaults <- config file <- overrides, validated as a whole."""
    merged: dict[str, Any] = {}
    if file:
        merged.update(load_config_file(file))
    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})
    return RunConfig(**merged)
