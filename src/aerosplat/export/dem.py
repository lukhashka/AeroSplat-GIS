"""DSM / DTM / nDSM GeoTIFF from the semantics stage's terrain raster (terrain.npz).

The exported rasters are the very ones buildings and facades were derived from
(aerosplat.semantics.terrain), not a recomputation, so a GIS user and the editor see the
same surface. They are resampled from the Site ENU grid into a north-up grid of the
target CRS (aerosplat.export.raster) and their heights converted from ENU u to WGS84
ellipsoidal h point by point.

Resampling never bridges a hole: bilinear where all four source cells have data, else
the nearest source cell's own value, else nodata. A DSM is a measurement (highest point
per cell); interpolating across a hole would invent one, while dropping every cell that
merely borders a hole would throw away ~40 % of a sparse cloud's cells (dataset_1910:
13 % -> 8 % filled). The nearest-cell fallback moves a value by at most half a cell
diagonal and never creates one. Likewise:

* **DTM** is a bare-earth *estimate*. Under buildings it is interpolated by design, but in
  holes far from any measured point it is pure extrapolation (nearest-cell fill), so cells
  farther than `max_gap_m` from a measured cell are written as nodata;
* **nDSM** (DSM - DTM, height above ground, no datum) is nodata wherever the DSM is,
  instead of the 0 that `Terrain.ndsm` uses internally -- 0 would claim "ground here".

nodata is -9999 (float32); heights are metres.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from aerosplat.export.georef import HEIGHT_DATUM, ExportRefused, PlaneFit, SiteProjection
from aerosplat.export.raster import TAG_HEIGHTS, NorthUpGrid, finish_raster, open_raster
from aerosplat.semantics.models import TerrainModel

F64 = NDArray[np.float64]
NODATA = -9999.0
DEFAULT_MAX_GAP_M = 25.0
DEM_KINDS = ("dsm", "dtm", "ndsm")


def sample_no_bridging(raster: F64, model: TerrainModel, e: F64, n: F64) -> F64:
    """Sample a cell-centred ENU raster at (e, n): bilinear where the four surrounding
    cells all have data, otherwise the containing cell's own value (NaN if it has none);
    NaN outside the raster.

    Within the outer half cell (outside the hull of cell centres but inside the raster)
    the edge value is extended, so the resampled raster covers the full extent.
    """
    fj = (np.asarray(e, float) - model.x0) / model.res_m - 0.5
    fi = (np.asarray(n, float) - model.y0) / model.res_m - 0.5
    outside = (fj < -0.5) | (fj > model.cols - 0.5) | (fi < -0.5) | (fi > model.rows - 0.5)
    fj = np.clip(fj, 0.0, model.cols - 1.0)
    fi = np.clip(fi, 0.0, model.rows - 1.0)
    j0 = np.minimum(np.floor(fj).astype(np.int64), max(model.cols - 2, 0))
    i0 = np.minimum(np.floor(fi).astype(np.int64), max(model.rows - 2, 0))
    j1 = np.minimum(j0 + 1, model.cols - 1)
    i1 = np.minimum(i0 + 1, model.rows - 1)
    tj, ti = fj - j0, fi - i0
    # NaN * 0 is NaN: a missing neighbour poisons the blend even at zero weight.
    v = ((1 - ti) * ((1 - tj) * raster[i0, j0] + tj * raster[i0, j1])
         + ti * ((1 - tj) * raster[i1, j0] + tj * raster[i1, j1]))
    # The cell (e, n) falls in -- the nearest centre -- is a measurement of its own.
    jn = np.clip(np.floor((np.asarray(e, float) - model.x0) / model.res_m).astype(np.int64),
                 0, model.cols - 1)
    i_n = np.clip(np.floor((np.asarray(n, float) - model.y0) / model.res_m).astype(np.int64),
                  0, model.rows - 1)
    v = np.where(np.isnan(v), raster[i_n, jn], v)
    return np.where(outside, np.nan, v)


def mask_far_from_data(dtm: F64, count: NDArray[np.integer], res_m: float,
                       max_gap_m: float) -> F64:
    """DTM with cells farther than max_gap_m from any measured cell set to NaN."""
    empty = np.asarray(count) == 0
    if not empty.any():
        return dtm.copy()
    if empty.all():
        return np.full_like(dtm, np.nan)
    dist = ndimage.distance_transform_edt(empty) * res_m
    return np.where(dist > max_gap_m, np.nan, dtm)


def load_terrain(npz: Path, model: TerrainModel) -> dict[str, NDArray[Any]]:
    # Not dict[str, F64]: `count` is the integer point count per cell, `dsm`/`dtm` are float64.
    with np.load(npz) as z:
        missing = {"dsm", "dtm", "count"} - set(z.files)
        if missing:
            raise ExportRefused(f"{npz}: no {sorted(missing)} -- not a terrain raster")
        out = {k: z[k] for k in ("dsm", "dtm", "count")}
    if out["dsm"].shape != (model.rows, model.cols) or out["dtm"].shape != out["dsm"].shape:
        raise ExportRefused(f"{npz}: rasters {out['dsm'].shape} do not match the header "
                            f"{(model.rows, model.cols)} in semantics.json")
    return out


def terrain_fit(proj: SiteProjection, model: TerrainModel, dsm: F64, dtm: F64) -> PlaneFit:
    e_range = (model.x0, model.x0 + model.cols * model.res_m)
    n_range = (model.y0, model.y0 + model.rows * model.res_m)
    lo = float(np.nanmin(dtm))
    hi = float(np.nanmax(dsm)) if np.isfinite(dsm).any() else lo
    return proj.fit_plane(e_range, n_range, (lo, hi), u_ref=float(np.nanmedian(dtm)))


def export_dem(paths: dict[str, Path], *, model: TerrainModel, dsm: F64, dtm: F64,
               count: NDArray[np.integer], proj: SiteProjection, fit: Optional[PlaneFit] = None,
               px: Optional[float] = None, max_gap_m: float = DEFAULT_MAX_GAP_M,
               cog: bool = True, tags: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Write the requested subset of {dsm, dtm, ndsm}. Returns per-kind statistics."""
    bad = set(paths) - set(DEM_KINDS)
    if bad:
        raise ValueError(f"unknown terrain products {sorted(bad)}")
    fit = fit or terrain_fit(proj, model, dsm, dtm)
    e_range = (model.x0, model.x0 + model.cols * model.res_m)
    n_range = (model.y0, model.y0 + model.rows * model.res_m)
    grid = NorthUpGrid.covering(fit, e_range, n_range, px or model.res_m)

    X, Y = np.meshgrid(grid.centre_x(np.arange(grid.width)), grid.centre_y(np.arange(grid.height)))
    E, N = fit.to_enu(X, Y)

    dtm_kept = mask_far_from_data(dtm, count, model.res_m, max_gap_m)
    sources = {"dsm": dsm, "dtm": dtm_kept,
               "ndsm": np.where(np.isfinite(dsm), dsm - dtm, np.nan)}
    summary: dict[str, Any] = {"grid": grid.describe(), "max_gap_m": max_gap_m,
                               "resampling": "bilinear; next to a hole the cell's own value; "
                                             "never across a hole"}
    for kind, path in paths.items():
        v = sample_no_bridging(sources[kind], model, E, N)
        ok = np.isfinite(v)
        if kind != "ndsm" and ok.any():
            # u -> ellipsoidal h at the cell's own position (exact, per cell).
            enu = np.column_stack([E[ok], N[ok], v[ok]])
            v[ok] = proj.enu_to_crs(enu)[:, 2]
        out = np.where(ok, v, NODATA).astype(np.float32)
        meaning = ("height above the DTM, metres (no datum)" if kind == "ndsm"
                   else HEIGHT_DATUM)
        with open_raster(path, grid, proj, count=1, dtype="float32", nodata=NODATA,
                         units="metre", descriptions=[kind.upper()],
                         tags={**(tags or {}), TAG_HEIGHTS: meaning}) as dst:
            dst.write(out, 1)
        finish_raster(path, cog=cog, resampling="average")
        summary[kind] = {"valid_fraction": float(ok.mean()),
                         "min": float(v[ok].min()) if ok.any() else None,
                         "max": float(v[ok].max()) if ok.any() else None,
                         "heights": meaning}
    return summary
