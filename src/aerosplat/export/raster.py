"""North-up rasters in a projected CRS and their GeoTIFF / COG files (rasterio).

A GeoTIFF *can* carry a rotated geotransform, but COG viewers, tilers and plenty of GIS
code do not honour it. So every exported raster is north-up in the target CRS: the
orthophoto is rendered straight into that grid (aerosplat.export.ortho) and the terrain
rasters are resampled into it (aerosplat.export.dem). Grid edges are snapped to whole
multiples of the pixel size, so exports of the same Site line up pixel for pixel.

Files are written to a temporary name next to the target and moved into place only when
complete: a crash never leaves a half-written GeoTIFF under the product's name.
"""

from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from aerosplat.export.georef import ExportRefused, PlaneFit, SiteProjection

GIS_EXTRA = "pip install -e .[gis]   (rasterio, laspy, lazrs)"
MAX_SIDE_PX = 40_000            # one side of an exported raster
MAX_PIXELS = 600_000_000        # whole raster (RGBA uint8: 2.4 GB uncompressed)
TAG_EXPORT = "AEROSPLAT_EXPORT"
TAG_GEOREF = "AEROSPLAT_GEOREFERENCE"
TAG_HEIGHTS = "AEROSPLAT_HEIGHTS"


def require_rasterio():
    try:
        import rasterio
        import rasterio.shutil  # noqa: F401
    except ImportError:
        raise ExportRefused(f"GeoTIFF export needs rasterio: {GIS_EXTRA}") from None
    return rasterio


@dataclass(frozen=True)
class NorthUpGrid:
    """Pixel (row, col) covers easting [x0 + col*px, +px) and northing (y1 - (row+1)*px, y1 - row*px]."""

    x0: float
    y1: float
    px: float
    width: int
    height: int

    @classmethod
    def covering(cls, fit: PlaneFit, e_range: tuple[float, float],
                 n_range: tuple[float, float], px: float, *,
                 max_side: int = MAX_SIDE_PX, max_pixels: int = MAX_PIXELS) -> "NorthUpGrid":
        """The smallest snapped grid containing an ENU rectangle (at the fit's u_ref)."""
        if not px > 0:
            raise ExportRefused(f"pixel size must be positive, got {px}")
        ee = np.array([e_range[0], e_range[1], e_range[0], e_range[1]])
        nn = np.array([n_range[0], n_range[0], n_range[1], n_range[1]])
        x, y = fit.to_crs(ee, nn)
        x0 = math.floor(float(x.min()) / px) * px
        y1 = math.ceil(float(y.max()) / px) * px
        width = max(1, math.ceil((float(x.max()) - x0) / px - 1e-9))
        height = max(1, math.ceil((y1 - float(y.min())) / px - 1e-9))
        if width > max_side or height > max_side or width * height > max_pixels:
            raise ExportRefused(
                f"{width} x {height} px at {px} m/px is too large (max {max_side} px a side, "
                f"{max_pixels / 1e6:.0f} Mpx): use a coarser pixel or a smaller area")
        return cls(x0=float(x0), y1=float(y1), px=float(px), width=int(width), height=int(height))

    def centre_x(self, col) -> np.ndarray:
        return self.x0 + (np.asarray(col, float) + 0.5) * self.px

    def centre_y(self, row) -> np.ndarray:
        return self.y1 - (np.asarray(row, float) + 0.5) * self.px

    def edge_xy(self, col: float, row: float) -> tuple[float, float]:
        """CRS position of a pixel-edge coordinate (col, row) = (0, 0) top-left corner."""
        return self.x0 + col * self.px, self.y1 - row * self.px

    def gdal_transform(self):
        from rasterio.transform import Affine
        return Affine(self.px, 0.0, self.x0, 0.0, -self.px, self.y1)

    def describe(self) -> dict[str, Any]:
        return {"x0": self.x0, "y1": self.y1, "pixel_m": self.px, "width": self.width,
                "height": self.height,
                "bounds": [self.x0, self.y1 - self.height * self.px,
                           self.x0 + self.width * self.px, self.y1]}


def _rio_crs(proj: SiteProjection):
    from rasterio.crs import CRS as RioCRS
    epsg = proj.epsg
    return RioCRS.from_epsg(epsg) if epsg else RioCRS.from_wkt(proj.crs.to_wkt())


def tmp_path_for(path: Path) -> Path:
    return path.with_name(f".{path.stem}.partial{path.suffix}")


@contextmanager
def open_raster(path: Path, grid: NorthUpGrid, proj: SiteProjection, *, count: int,
                dtype: str, nodata: Optional[float] = None, rgba: bool = False,
                tags: Optional[dict[str, Any]] = None, units: Optional[str] = None,
                descriptions: Optional[list[str]] = None) -> Iterator[Any]:
    """A tiled GeoTIFF open for (windowed) writing at a temporary name; see `finish_raster`."""
    rasterio = require_rasterio()
    from rasterio.enums import ColorInterp

    tmp = tmp_path_for(path)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    profile: dict[str, Any] = dict(
        driver="GTiff", width=grid.width, height=grid.height, count=count, dtype=dtype,
        crs=_rio_crs(proj), transform=grid.gdal_transform(), tiled=True, blockxsize=512,
        blockysize=512, compress="deflate", BIGTIFF="IF_SAFER")
    if nodata is not None:
        profile["nodata"] = nodata
    if rgba:
        profile["photometric"] = "RGB"
    try:
        with rasterio.open(tmp, "w", **profile) as dst:
            if rgba:
                dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue,
                                   ColorInterp.alpha]
            if tags:
                dst.update_tags(**{k: v if isinstance(v, str) else json.dumps(v)
                                   for k, v in tags.items()})
            if units:
                dst.units = tuple(units for _ in range(count))
            if descriptions:
                dst.descriptions = tuple(descriptions)
            yield dst
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def finish_raster(path: Path, *, cog: bool = True, resampling: str = "average") -> Path:
    """Move the temporary GeoTIFF into place, as a Cloud Optimized GeoTIFF if `cog`."""
    rasterio = require_rasterio()
    tmp = tmp_path_for(path)
    staged = path.with_name(f".{path.stem}.cog{path.suffix}")
    try:
        if cog:
            rasterio.shutil.copy(tmp, staged, driver="COG", compress="DEFLATE",
                                 overview_resampling=resampling, BIGTIFF="IF_SAFER")
            os.replace(staged, path)
        else:
            os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
        staged.unlink(missing_ok=True)
    return path


def read_tags(path: Path) -> dict[str, Any]:
    """AeroSplat tags of an exported raster, JSON-decoded (for tests and the API)."""
    rasterio = require_rasterio()
    with rasterio.open(path) as src:
        tags = src.tags()
    out: dict[str, Any] = {}
    for k, v in tags.items():
        if k.startswith("AEROSPLAT_"):
            try:
                out[k] = json.loads(v)
            except ValueError:
                out[k] = v
    return out
