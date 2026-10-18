"""Truncated Euclidean distance field of a voxel grid, computed in tiles (numpy/scipy, CPU).

scipy's exact EDT keeps ~50 bytes of scratch per voxel (an int32 feature transform plus
float64 distances): a Site at 0.5 m would need tens of gigabytes in one piece. Truncating the
field at `max_distance_m` makes tiling exact rather than approximate: a voxel whose nearest
blocked voxel lies within the truncation distance finds it inside a halo of that width, and
every other voxel only needs to learn that it is farther than the truncation.

Boundary conditions are the occupancy rules (aerosplat.surface.models): beside the grid is
unknown and below it is underground, so both are padded as blocked. Above the grid each
column keeps the state of its top layer, which can never bring a blocked voxel closer than
the top layer itself already is, so the top needs no padding.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

DEFAULT_TILE_VOXELS = 12_000_000


def distance_field_cm(blocked: NDArray[np.bool_], res_m: float, max_distance_m: float, *,
                      signed: bool = False,
                      tile_voxels: int = DEFAULT_TILE_VOXELS) -> NDArray[np.integer]:
    """Distance from each voxel centre to the nearest blocked voxel centre, in centimetres.

    Values are rounded DOWN to whole centimetres, so a stored distance never exceeds the true
    one, and truncated at `max_distance_m`. Blocked voxels hold 0 — or, with `signed`, minus
    the distance to the nearest free voxel centre (rounded down as well, i.e. deeper).

    Returns uint16, or int16 when signed; `max_distance_m` must fit that range.
    """
    b = np.asarray(blocked)
    if b.ndim != 3 or b.dtype != np.bool_:
        raise ValueError(f"expected a 3-D boolean volume, got {b.dtype} {b.shape}")
    if not (math.isfinite(res_m) and res_m > 0):
        raise ValueError(f"res_m must be positive, got {res_m}")
    if not (math.isfinite(max_distance_m) and max_distance_m > 0):
        raise ValueError(f"max_distance_m must be positive, got {max_distance_m}")
    dtype = np.int16 if signed else np.uint16
    limit_cm = int(math.floor(max_distance_m * 100.0))
    if limit_cm > np.iinfo(dtype).max:
        raise ValueError(f"max_distance_m={max_distance_m} does not fit {np.dtype(dtype).name} "
                         f"centimetres (max {np.iinfo(dtype).max / 100:.2f} m)")

    rows, cols, layers = b.shape
    # One extra voxel beyond the truncation, so a neighbour exactly at the limit is seen.
    halo = int(math.ceil(max_distance_m / res_m)) + 1
    side = int(math.sqrt(max(1, tile_voxels) / (layers + 1)))
    # A core narrower than the halo would spend most of the work on halos: exceed the budget
    # instead (at most 4x the core's own volume is halo then).
    core = max(side - 2 * halo, 2 * halo, 16)
    out = np.empty(b.shape, dtype)

    for i0 in range(0, rows, core):
        i1 = min(rows, i0 + core)
        for j0 in range(0, cols, core):
            j1 = min(cols, j0 + core)
            a0, a1, c0, c1 = i0 - halo, i1 + halo, j0 - halo, j1 + halo
            # Everything not copied from the grid stays blocked: the lateral margin beyond the
            # grid's edge and the extra bottom layer (index 0) under it.
            win = np.ones((a1 - a0, c1 - c0, layers + 1), np.bool_)
            ia, ib = max(0, a0), min(rows, a1)
            ja, jb = max(0, c0), min(cols, c1)
            win[ia - a0:ib - a0, ja - c0:jb - c0, 1:] = b[ia:ib, ja:jb, :]
            core_sl = (slice(i0 - a0, i1 - a0), slice(j0 - c0, j1 - c0), slice(1, None))

            free = ~win
            # The bottom pad guarantees a blocked voxel, so this EDT always has a target.
            d = ndimage.distance_transform_edt(free, sampling=res_m)[core_sl]
            if signed:
                if free.any():
                    d_in = ndimage.distance_transform_edt(win, sampling=res_m)[core_sl]
                else:
                    d_in = np.full(d.shape, np.inf)
                d = np.where(free[core_sl], d, -d_in)
            cm = np.floor(np.clip(d, -max_distance_m, max_distance_m) * 100.0)
            out[i0:i1, j0:j1, :] = np.clip(cm, -limit_cm if signed else 0, limit_cm).astype(dtype)
    return out
