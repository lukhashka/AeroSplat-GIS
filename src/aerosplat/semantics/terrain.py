"""DSM, DTM and nDSM from a dense point cloud (ACCESS_POINTS §5.2.1).

The DTM comes from a progressive morphological filter (Zhang et al. 2003): the
surface is repeatedly opened with a growing window, and a cell is pulled down to
the opened surface only when the drop exceeds what the terrain's own slope could
explain. Buildings vanish once the window is wider than they are; hills survive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from aerosplat.semantics.models import TerrainModel

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class Terrain:
    """A TerrainModel together with its rasters. NaN marks cells with no points."""

    model: TerrainModel
    dsm: F64            # highest point per cell (NaN where empty)
    dtm: F64            # bare-earth surface (never NaN)
    count: NDArray[np.int64]
    roughness: F64      # |DSM - 3x3 median|, metres: how bumpy the surface is here

    @property
    def ndsm(self) -> F64:
        return np.where(np.isnan(self.dsm), 0.0, self.dsm - self.dtm)

    def height_at(self, east, north) -> F64:
        """Ground height (DTM) at ENU positions."""
        i, j = self.model.index_of(east, north)
        return self.dtm[i, j]

    def to_npz(self) -> dict[str, np.ndarray]:
        return {"dsm": self.dsm, "dtm": self.dtm, "count": self.count,
                "roughness": self.roughness}


def _fill_nan(a: F64) -> F64:
    """Replace NaN with the value of the nearest finite cell."""
    bad = np.isnan(a)
    if not bad.any():
        return a.copy()
    if bad.all():
        return np.zeros_like(a)
    _, (ii, jj) = ndimage.distance_transform_edt(bad, return_indices=True)
    return a[ii, jj]


def progressive_morphological_dtm(dsm: F64, res_m: float, *, max_window_m: float = 48.0,
                                  slope: float = 0.3, dh_0: float = 0.4,
                                  dh_max: float = 6.0) -> F64:
    """Bare-earth surface from a (NaN-free) DSM.

    `slope` is the steepest terrain gradient trusted as real (m/m); anything
    dropping faster than that over the current window is treated as an object.
    """
    surface = dsm.copy()
    window = 1
    while True:
        size = 2 * window + 1
        opened = ndimage.grey_opening(surface, size=(size, size), mode="nearest")
        dh = min(dh_max, slope * (size * res_m) + dh_0) if window > 1 else dh_0
        surface = np.where(surface - opened > dh, opened, surface)
        if size * res_m >= max_window_m:
            return surface
        window *= 2


def build_terrain(xyz, *, res_m: float = 1.0, max_window_m: float = 48.0,
                  slope: float = 0.3, smooth_m: float = 2.0) -> Terrain:
    """Rasterise a cloud and derive DSM / DTM.

    `smooth_m` low-passes the DTM afterwards: morphological opening leaves
    terraces at the window scale, which would otherwise show up as metre-level
    steps in `v_ground` under a facade.
    """
    p = np.asarray(xyz, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"expected (N, 3) points, got {p.shape}")
    if len(p) == 0:
        raise ValueError("empty cloud")

    x0, y0 = float(np.floor(p[:, 0].min())), float(np.floor(p[:, 1].min()))
    cols = max(1, int(np.ceil((p[:, 0].max() - x0) / res_m)) + 1)
    rows = max(1, int(np.ceil((p[:, 1].max() - y0) / res_m)) + 1)
    j = np.clip(((p[:, 0] - x0) / res_m).astype(np.int64), 0, cols - 1)
    i = np.clip(((p[:, 1] - y0) / res_m).astype(np.int64), 0, rows - 1)
    flat = i * cols + j

    count = np.bincount(flat, minlength=rows * cols).reshape(rows, cols)
    dsm = np.full(rows * cols, -np.inf)
    np.maximum.at(dsm, flat, p[:, 2])
    dsm = np.where(np.isinf(dsm), np.nan, dsm).reshape(rows, cols)

    dtm = progressive_morphological_dtm(_fill_nan(dsm), res_m, max_window_m=max_window_m,
                                        slope=slope)
    if smooth_m > 0:
        sigma = smooth_m / res_m / 2.0
        dtm = ndimage.gaussian_filter(dtm, sigma=sigma, mode="nearest")
    # Smoothing may lift the surface above a measured low point; the ground is
    # never higher than what was actually seen there.
    dtm = np.minimum(dtm, _fill_nan(dsm))

    # Roughness separates roofs from canopy: a roof is locally flat, a tree is
    # not. On dataset_v6 the per-building median is 0.04 m against 0.29 m for
    # vegetation. Colour (excess green) is a second signal, but it fails on
    # autumn foliage, which is exactly what these Earth Studio frames show.
    filled = _fill_nan(dsm)
    roughness = np.abs(filled - ndimage.median_filter(filled, size=3, mode="nearest"))

    model = TerrainModel(res_m=res_m, x0=x0, y0=y0, rows=rows, cols=cols,
                         n_points=len(p), filled_cells=int((count > 0).sum()))
    return Terrain(model=model, dsm=dsm, dtm=dtm, count=count, roughness=roughness)
