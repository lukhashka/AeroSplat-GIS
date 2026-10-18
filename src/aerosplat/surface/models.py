"""What the `occupancy` stage produces: free / occupied / unknown voxels and a distance field.

Geometry is Site ENU metres (ADR-0006). Voxel (i, j, k) covers

    east  [x0 + j*res, x0 + (j+1)*res)
    north [y0 + i*res, y0 + (i+1)*res)
    up    [z0 + k*res, z0 + (k+1)*res)

so (i, j) follow the row/column convention of the semantics terrain raster and arrays are
shaped (rows, cols, layers). The grid is aligned with that raster: every terrain cell is an
exact block of voxel columns.

Outside the grid the rules are fixed, and every query applies them the same way:
laterally outside = unknown (reported as OUTSIDE), below = underground (occupied), above =
the column continues with the state of its top layer.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]

# Voxel states. The grid stores the first three; OUTSIDE is only ever returned by queries.
FREE, OCCUPIED, UNKNOWN, OUTSIDE = 0, 1, 2, 3
STATE_NAMES = ("free", "occupied", "unknown", "outside")

# What the stage writes into its output directory (here so readers need not import the stage).
OCCUPANCY_JSON = "occupancy.json"
OCCUPANCY_NPZ = "occupancy.npz"
# Bumped whenever the meaning of the files changes; also the stage version for `--resume`.
FORMAT_VERSION = "occupancy-v1"


class ColumnSource(IntEnum):
    """Where the surface height of a voxel column came from.

    Free space is only ever free ABOVE a column's surface, so this is also the provenance of
    every free voxel: observed for POINTS and CELL, assumed for FILLED and SKYLINE.
    """

    POINTS = 1      # points inside this voxel column: its highest point
    CELL = 2        # no points in the column, but in its terrain cell: the cell's DSM
    FILLED = 3      # a gap in the cloud: highest DSM within fill_radius_m + assumed margin
    SKYLINE = 4     # nothing within fill_radius_m: unknown up to the site skyline + margin
    SKY = 5         # nothing within fill_radius_m, unknown="sky": unknown all the way up
    PILLAR = 6      # surface above the grid's ceiling: blocked all the way up


ASSUMED_FREE = (int(ColumnSource.FILLED), int(ColumnSource.SKYLINE))


def state_name(code: int) -> str:
    return STATE_NAMES[int(code)]


def parse_bbox(text: str) -> tuple[float, float, float, float]:
    """"xmin,ymin,xmax,ymax" (Site ENU metres) -> a validated tuple."""
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 4:
        raise ValueError(f"bbox must be 'xmin,ymin,xmax,ymax', got {text!r}")
    try:
        xmin, ymin, xmax, ymax = (float(p) for p in parts)
    except ValueError:
        raise ValueError(f"bbox must hold four numbers, got {text!r}") from None
    if not all(math.isfinite(v) for v in (xmin, ymin, xmax, ymax)):
        raise ValueError(f"bbox must be finite, got {text!r}")
    if not (xmin < xmax and ymin < ymax):
        raise ValueError(f"bbox must have xmin < xmax and ymin < ymax, got {text!r}")
    return xmin, ymin, xmax, ymax


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GridSpec(_Model):
    """Where the voxel lattice sits in Site ENU. Arrays are stored separately (occupancy.npz)."""

    res_m: float = Field(gt=0)
    x0: float
    y0: float
    z0: float
    rows: int = Field(gt=0)         # along north (y)
    cols: int = Field(gt=0)         # along east (x)
    layers: int = Field(gt=0)       # along up (z)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.rows, self.cols, self.layers

    @property
    def voxels(self) -> int:
        return self.rows * self.cols * self.layers

    @property
    def x1(self) -> float:
        return self.x0 + self.cols * self.res_m

    @property
    def y1(self) -> float:
        return self.y0 + self.rows * self.res_m

    @property
    def z1(self) -> float:
        return self.z0 + self.layers * self.res_m

    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return (self.x0, self.y0, self.z0), (self.x1, self.y1, self.z1)

    def index_of(self, xyz) -> tuple[I64, I64, I64]:
        """ENU points (N, 3) -> voxel (i, j, k), NOT clipped: out-of-grid indices are kept."""
        p = np.asarray(xyz, np.float64).reshape(-1, 3)
        r = self.res_m
        return (np.floor((p[:, 1] - self.y0) / r).astype(np.int64),
                np.floor((p[:, 0] - self.x0) / r).astype(np.int64),
                np.floor((p[:, 2] - self.z0) / r).astype(np.int64))

    def centre_of(self, i, j, k) -> F64:
        """Voxel indices -> (N, 3) ENU centres (any integers, including out-of-grid ones)."""
        r = self.res_m
        return np.stack([self.x0 + (np.asarray(j, np.float64) + 0.5) * r,
                         self.y0 + (np.asarray(i, np.float64) + 0.5) * r,
                         self.z0 + (np.asarray(k, np.float64) + 0.5) * r], axis=-1)


class EsdfInfo(_Model):
    """How `esdf_cm` in occupancy.npz is to be read."""

    units: str = "cm"
    dtype: str = "uint16"           # "int16" when signed
    signed: bool = False
    max_distance_m: float = Field(gt=0)
    convention: str = ("distance from the voxel centre to the centre of the nearest occupied or "
                       "unknown voxel, rounded down to whole cm, truncated at max_distance_m; "
                       "0 inside blocked voxels (signed: minus the distance to the nearest free "
                       "voxel centre)")


class OccupancyStats(_Model):
    voxels: int = Field(ge=0)
    free: int = Field(ge=0)
    occupied: int = Field(ge=0)
    unknown: int = Field(ge=0)
    columns: dict[str, int]                 # ColumnSource name -> number of voxel columns
    observed_fraction: float = Field(ge=0, le=1)    # columns with a measured surface
    assumed_fraction: float = Field(ge=0, le=1)     # columns whose free air is assumed
    closed_columns: int = Field(ge=0)       # surfaces raised by closing (pits, noise under roofs)
    skyline_z: float                        # quantile of the observed column tops
    unknown_ceiling_z: Optional[float]      # how high unknown columns reach (None: to the sky)
    ceiling_z: float                        # top of the grid
    n_points: int = Field(ge=0)             # cloud points inside the grid


class OccupancyReport(_Model):
    """`occupancy/occupancy.json`: grid header, parameters and findings, minus the arrays."""

    format: str = FORMAT_VERSION
    frame: str = "WGS84 Site ENU"
    grid: GridSpec
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    esdf: EsdfInfo
    stats: OccupancyStats
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    timings_s: dict[str, float] = Field(default_factory=dict)
