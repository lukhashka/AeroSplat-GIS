"""Validated run configuration (single source of truth for pipeline parameters).

Replaces the free-form CONFIG dict in starter.py. A run is described by a
`RunConfig`; YAML/JSON files and CLI `--set KEY=VALUE` overrides are merged
into it and validated once, so typos and bad values fail before any stage runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    # "auto": Inria CUDA if built, else the PyTorch tile rasterizer. "gsplat" only
    # when asked for: it JIT-builds on first use (Windows: tools/gsplat_env.bat).
    rasterizer: Literal["auto", "cuda", "torch", "gsplat"] = "auto"
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

    # -- 3DGS, rasterizer="gsplat" only (Фаза 2.2, aerosplat.gs.gsplat_backend) --
    sh_degree: int = Field(3, ge=0, le=3)
    sh_degree_interval: int = Field(1000, ge=0)        # +1 SH band every N iterations
    rasterize_mode: Literal["classic", "antialiased"] = "antialiased"
    # "mcmc": 3DGS-MCMC, fixed budget (+0.56 dB on dataset_v6, docs/BASELINE.md);
    # "default": Kerbl et al. capped at max_gaussians.
    densify_strategy: Literal["default", "mcmc"] = "mcmc"
    absgrad: bool = False                              # AbsGS gradients ("default" only)
    absgrad_threshold: float = Field(0.0008, gt=0)
    mcmc_opacity_reg: float = Field(0.01, ge=0)
    mcmc_scale_reg: float = Field(0.01, ge=0)
    mcmc_noise_lr: float = Field(5e5, ge=0)
    mcmc_refine_stop_frac: float = Field(0.85, gt=0, le=1)  # MCMC relocation until this fraction
    # "exposure": per-train-image colour affine, training only (mixed sources).
    appearance: Literal["none", "exposure"] = "none"

    # -- evaluation (Фаза 2.1) ---------------------------------------------
    # Кожен N-й кадр — test (не тренується). 0 = без спліту і без стадії eval.
    eval_every: int = Field(8, ge=0)
    eval_interval: int = Field(1000, ge=1)
    eval_lpips: bool = True

    # -- semantics: terrain, buildings, facades (Фаза 7.1, aerosplat.semantics) --
    semantics_res: float = Field(1.0, gt=0, le=10)             # розмір клітини растру, м
    min_building_height: float = Field(2.5, gt=0)              # над рельєфом, м
    min_building_area: float = Field(40.0, gt=0)               # м²
    facade_inlier: float = Field(1.2, gt=0)                    # напівтовщина шару стіни, м
    facade_min_length: float = Field(5.0, gt=0)
    facade_min_height: float = Field(3.0, gt=0)
    facade_min_points: int = Field(40, ge=3)
    max_facades_per_building: int = Field(12, ge=1)
    coverage_cell: float = Field(0.5, gt=0)                    # клітина карти покриття, м
    coverage_max_angle: float = Field(60.0, gt=0, lt=90)       # до нормалі фасаду
    # Роздільність, за якої деталі фасаду ще читаються. Орбіта Earth Studio з
    # 200 м дає ~0.10-0.25 м/px, тобто гірше за цей поріг: стадія це повідомляє.
    coverage_max_gsd: float = Field(0.05, gt=0)
    semantics_seed: int = Field(0, ge=0)

    # -- occupancy: voxel grid + ESDF (Фаза 8.1a, aerosplat.surface) --------
    # Невідоме = зайняте (internal reliability requirements §3 п. 3); кожне припущення про вільний
    # простір записується по колонках (ColumnSource) і в occupancy.json.
    occupancy_res: float = Field(0.5, gt=0, le=2)              # воксель, м; ділить semantics_res
    occupancy_headroom: float = Field(10.0, gt=0)              # сітка над лінією дахів, м
    occupancy_max_distance: float = Field(20.0, gt=0, le=600)  # обрізання ESDF, м
    occupancy_fill_radius: float = Field(5.0, ge=0)            # дірки в хмарі до цього радіуса, м
    occupancy_close_radius: float = Field(1.0, ge=0)           # ями вужчі за ~2 радіуси, м
    occupancy_assumed_margin: float = Field(1.0, ge=0)         # запас над припущеною висотою, м
    # Колонки без даних: невідомі до лінії дахів ("skyline") або до неба ("sky").
    occupancy_unknown: Literal["skyline", "sky"] = "skyline"
    occupancy_skyline_quantile: float = Field(0.9999, gt=0.5, le=1)
    # Межі сітки (aerosplat.surface.extent). Поза сіткою все заблоковано (OUTSIDE), тож місія
    # мусить уміщатися в ній; сітку ніколи не обрізають мовчки під occupancy_max_voxels.
    #   "auto"   — occupancy_bbox, якщо задано; інакше bbox ділянки (site.yaml); інакше квадрат
    #              ±occupancy_radius навколо початку Site (центру зйомки);
    #   "radius" — квадрат навколо початку Site, навіть коли в ділянки є bbox;
    #   "raster" — увесь растр рельєфу: усе, що покриває хмара (Earth Studio: ~1.6 × 1.8 км).
    occupancy_extent: Literal["auto", "radius", "raster"] = "auto"
    occupancy_radius: float = Field(150.0, gt=0, le=5000)      # півсторона квадрата, м
    occupancy_bbox: Optional[str] = None                       # "xmin,ymin,xmax,ymax", Site ENU
    occupancy_max_voxels: int = Field(400_000_000, ge=1)
    # Space carving (aerosplat.surface.carving): промені камер до точок, які вони бачили
    # (dense/fused.ply.vis), відкривають невідомий простір. Без видимості — лише 2.5D.
    occupancy_carve: bool = True
    occupancy_carve_min_views: int = Field(2, ge=1)            # різних камер на воксель
    occupancy_carve_margin: float = Field(1.0, ge=0)           # не вирізати останні N м променя
    occupancy_carve_under: bool = True                         # і під виміряною поверхнею (навіси)

    colmap_bin: str = "colmap"

    @field_validator("occupancy_bbox")
    @classmethod
    def _check_bbox(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            from aerosplat.surface.models import parse_bbox

            parse_bbox(v)
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> "RunConfig":
        if self.warmup_iters >= self.iterations and self.warmup_iters:
            raise ValueError("warmup_iters must be smaller than iterations")
        ratio = self.semantics_res / self.occupancy_res
        if round(ratio) < 1 or abs(ratio - round(ratio)) > 1e-6:
            raise ValueError(f"occupancy_res={self.occupancy_res} must divide "
                             f"semantics_res={self.semantics_res}")
        if self.occupancy_headroom < self.occupancy_assumed_margin + 2 * self.occupancy_res:
            raise ValueError("occupancy_headroom must exceed occupancy_assumed_margin + 2 voxels")
        if self.occupancy_bbox is not None and self.occupancy_extent != "auto":
            raise ValueError(f"occupancy_bbox and occupancy_extent={self.occupancy_extent!r} "
                             f"contradict each other: an explicit bbox needs extent 'auto'")
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
