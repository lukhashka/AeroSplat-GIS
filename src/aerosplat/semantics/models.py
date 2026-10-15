"""What the semantics stage produces: terrain raster, buildings, facades.

Geometry is Site ENU metres (ADR-0006). A facade carries its own 2-D frame:
`origin` is the point (u, v) = (0, 0), `u_axis` runs horizontally along the wall
and `v` is simply the ENU z, so a point on the wall is
`origin + u * u_axis + (v - origin.z) * up`. `u` grows to the RIGHT of an
observer standing outside and facing the wall, which is the same direction the
addressing scheme counts columns in (ACCESS_POINTS §4).
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

F64 = NDArray[np.float64]

UP = np.array([0.0, 0.0, 1.0])
OCTANTS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def octant_of(azimuth_deg: float) -> str:
    """Compass octant of an azimuth (0 = north, 90 = east)."""
    return OCTANTS[int(round((azimuth_deg % 360.0) / 45.0)) % 8]


def azimuth_of(normal_xy: tuple[float, float]) -> float:
    """Compass azimuth of a horizontal direction (0 = north, 90 = east)."""
    return math.degrees(math.atan2(normal_xy[0], normal_xy[1])) % 360.0


def u_axis_for(normal_xy: tuple[float, float]) -> tuple[float, float]:
    """Horizontal axis pointing RIGHT for an observer outside, facing the wall.

    The observer looks along -n, so right = (-n) x up = (-ny, nx).
    """
    return (-normal_xy[1], normal_xy[0])


class SurfaceKind(str, Enum):
    wall = "wall"
    roof_flat = "roof_flat"
    roof_pitched = "roof_pitched"
    ground = "ground"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Coverage(_Model):
    """How well the captures see a facade (ACCESS_POINTS §5.2.5).

    Detections on a 3DGS orthophoto are only trustworthy where this is good:
    elsewhere the splat invents facade texture out of too few views.
    """

    cell_m: float = Field(gt=0)
    cells: int = Field(ge=0)
    # Cells some frame looks at from a usable angle, with the cell inside the
    # image -- whatever the resolution. This is the geometry of the capture.
    visible: int = Field(ge=0)
    # Of those, the cells a frame also resolves finely enough (<= max GSD).
    covered: int = Field(ge=0)
    mean_views: float = Field(ge=0)         # over covered cells
    max_gsd_cm: float = Field(gt=0)         # the bar that `covered` had to clear
    best_gsd_cm: Optional[float] = None     # finest GSD among the visible cells
    median_angle_deg: Optional[float] = None
    # True once occlusion is tested as well; v1 only checks angle, GSD and framing.
    occlusion_tested: bool = False

    @property
    def fraction(self) -> float:
        return self.covered / self.cells if self.cells else 0.0

    @property
    def visible_fraction(self) -> float:
        return self.visible / self.cells if self.cells else 0.0

    @property
    def too_coarse(self) -> bool:
        """Seen from the right angles, but never sharply enough to read detail off."""
        return self.visible > 0 and self.covered == 0


class Facade(_Model):
    """One planar wall of a building, with the 2-D frame used to place openings."""

    id: str                                 # "<building>/<octant><ordinal>", e.g. "B01/N1"
    building_id: str
    kind: SurfaceKind = SurfaceKind.wall
    origin: tuple[float, float, float]      # ENU point at (u, v) = (0, ground)
    normal: tuple[float, float, float]      # unit, outward (away from the building)
    u_axis: tuple[float, float, float]      # unit, horizontal, right when facing the wall
    azimuth_deg: float
    u_min: float
    u_max: float
    v_ground: float                         # ENU z of the ground at the wall foot
    v_max: float                            # ENU z of the top
    plane_rms_m: float = Field(ge=0)        # scatter of the points about the plane
    n_points: int = Field(ge=0)
    coverage: Optional[Coverage] = None

    @property
    def length(self) -> float:
        return self.u_max - self.u_min

    @property
    def height(self) -> float:
        return self.v_max - self.v_ground

    @property
    def name(self) -> str:
        return self.id.rsplit("/", 1)[-1]

    def point(self, u: float, v: float) -> F64:
        """(u, v) on the facade -> ENU point. `v` is the ENU height, not a local offset."""
        o = np.asarray(self.origin, float)
        return o + u * np.asarray(self.u_axis, float) + (v - o[2]) * UP

    def uv_of(self, xyz) -> tuple[F64, F64]:
        """ENU points (..., 3) -> (u, v). Points off the plane keep their in-plane u."""
        d = np.asarray(xyz, float) - np.asarray(self.origin, float)
        return d @ np.asarray(self.u_axis, float), np.asarray(xyz, float)[..., 2]

    def distance_to_plane(self, xyz) -> F64:
        """Signed distance along the outward normal (positive = in front of the wall)."""
        d = np.asarray(xyz, float) - np.asarray(self.origin, float)
        return d @ np.asarray(self.normal, float)


class Building(_Model):
    id: str                                 # "B01", stable within a run
    footprint: list[tuple[float, float]]    # ENU polygon, counter-clockwise, metres
    area_m2: float = Field(ge=0)
    ground_z: float                         # DTM at the footprint
    roof_z: float                           # 95th percentile of the roof height
    height_m: float = Field(ge=0)
    facade_ids: list[str] = Field(default_factory=list)

    @property
    def centroid(self) -> tuple[float, float]:
        p = np.asarray(self.footprint, float)
        return float(p[:, 0].mean()), float(p[:, 1].mean())


class TerrainModel(_Model):
    """DSM / DTM / nDSM on a regular ENU raster.

    Cell (i, j) covers east [x0 + j*res, x0 + (j+1)*res) and north
    [y0 + i*res, y0 + (i+1)*res); its centre is half a cell further on.
    Arrays are stored separately (`terrain.npz`), not in this record.
    """

    res_m: float = Field(gt=0)
    x0: float
    y0: float
    rows: int = Field(gt=0)
    cols: int = Field(gt=0)
    n_points: int = Field(ge=0)
    filled_cells: int = Field(ge=0)

    def cell_centres(self) -> tuple[F64, F64]:
        """(east, north) of every cell centre, each (rows, cols)."""
        x = self.x0 + (np.arange(self.cols) + 0.5) * self.res_m
        y = self.y0 + (np.arange(self.rows) + 0.5) * self.res_m
        xx, yy = np.meshgrid(x, y)
        return xx, yy

    def index_of(self, east, north) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """(east, north) -> (row, col), clipped to the raster."""
        j = np.clip(((np.asarray(east, float) - self.x0) / self.res_m).astype(np.int64),
                    0, self.cols - 1)
        i = np.clip(((np.asarray(north, float) - self.y0) / self.res_m).astype(np.int64),
                    0, self.rows - 1)
        return i, j


class SemanticsResult(_Model):
    """`semantics/semantics.json`: everything the stage found, minus the rasters."""

    frame: str = "WGS84 Site ENU"
    terrain: TerrainModel
    buildings: list[Building]
    facades: list[Facade]
    params: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    def facade(self, facade_id: str) -> Facade:
        for f in self.facades:
            if f.id == facade_id:
                return f
        raise KeyError(facade_id)
