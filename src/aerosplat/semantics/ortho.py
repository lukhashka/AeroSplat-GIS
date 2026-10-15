"""Facade orthophotos: a head-on, metric view of one wall (ACCESS_POINTS §5.3).

Detection and hand-drawing both want a picture where the facade is frontal and
metric: openings are rectangular, aligned, and all at one scale, so a pixel maps to
(u, v) by a constant. That is exactly an orthographic render along the facade's
−normal, framed to its (u, v) extent.

The render service stays generic: it knows nothing about facades. This module builds
the pose and reports the pixel grid, the same way `camera.ortho_ground_grid` reports
the georeference of a nadir ortho. Both are numpy-free and torch-free, so the API can
do this without torch or a GPU.

Why the pose comes out so clean: with `yaw = azimuth + 180` and `pitch = 0` the
camera's right axis is exactly the facade's u-axis and its down axis is world −z
(see `test_semantics_ortho`). So image columns run along u and image rows run down
the wall, with no rotation left over.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from aerosplat.render.models import MAX_SIDE, Position, RenderRequest
from aerosplat.semantics.models import Facade

# Margin around the facade, so an opening flush with the fitted edge is still fully
# visible and the operator can see the wall end.
DEFAULT_MARGIN_M = 1.0
# How far in front of the wall the camera sits. An orthographic camera's distance does
# not change the image, but everything behind it is clipped, so it has to be outside.
# Whatever stands between it and the wall occludes -- which is the truth, and worth seeing.
DEFAULT_STANDOFF_M = 25.0
# ACCESS_POINTS §5.3 asks for 1-2 cm/px to read openings; 5 cm/px is the coarsest
# that is still worth drawing on.
DEFAULT_GSD_CM = 5.0
MIN_SIDE_PX = 16


class FacadeOrthoError(ValueError):
    """A facade view that cannot be rendered as asked (too many pixels for one image)."""


class FacadeOrthoSpec(BaseModel):
    """What to render: a facade, how finely, and how much wall around it."""

    model_config = ConfigDict(extra="forbid")

    gsd_cm: float = Field(DEFAULT_GSD_CM, gt=0.1, le=100.0)     # centimetres per pixel
    margin_m: float = Field(DEFAULT_MARGIN_M, ge=0.0, le=50.0)
    standoff_m: float = Field(DEFAULT_STANDOFF_M, gt=0.0, le=5000.0)
    output: Literal["rgb", "depth", "alpha"] = "rgb"
    sh_degree: Optional[int] = Field(None, ge=0, le=3)
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)


class FacadeGrid(BaseModel):
    """Pixel ↔ (u, v) for a facade orthophoto: the facade's answer to `ground_grid`.

    Pixel (i, j) means column i, row j, and the mapping is for its *centre*:

        u = pixel00_uv[0] + i·du_uv[0] + j·dv_uv[0]
        v = pixel00_uv[1] + i·du_uv[1] + j·dv_uv[1]

    which is the same affine form as `ground_grid`, so a caller can reuse the code.
    In practice `du_uv` is (+m_per_px, 0) and `dv_uv` is (0, −m_per_px): columns run
    along u, rows run *down* the wall. The ENU point of a pixel is then
    `origin_enu + u·u_axis + (v − origin_enu.z)·ẑ`.
    """

    model_config = ConfigDict(extra="forbid")

    facade_id: str
    m_per_px: float
    pixel00_uv: tuple[float, float]
    du_uv: tuple[float, float]
    dv_uv: tuple[float, float]
    u_min: float                            # framed extent, metres in the facade frame
    u_max: float
    v_min: float
    v_max: float
    origin_enu: tuple[float, float, float]
    u_axis: tuple[float, float, float]
    normal: tuple[float, float, float]
    v_ground: float
    azimuth_deg: float

    def uv_of_pixel(self, i: float, j: float) -> tuple[float, float]:
        return (self.pixel00_uv[0] + i * self.du_uv[0] + j * self.dv_uv[0],
                self.pixel00_uv[1] + i * self.du_uv[1] + j * self.dv_uv[1])

    def pixel_of_uv(self, u: float, v: float) -> tuple[float, float]:
        """Inverse of `uv_of_pixel`. Valid because the two axes are orthogonal here."""
        du, dv = self.du_uv, self.dv_uv
        det = du[0] * dv[1] - du[1] * dv[0]
        su, sv = u - self.pixel00_uv[0], v - self.pixel00_uv[1]
        return ((su * dv[1] - sv * dv[0]) / det, (sv * du[0] - su * du[1]) / det)


def facade_view_yaw(facade: Facade) -> float:
    """Compass heading of a camera looking at the wall head-on: along −normal."""
    return (facade.azimuth_deg + 180.0) % 360.0


def facade_ortho(facade: Facade, spec: Optional[FacadeOrthoSpec] = None
                 ) -> tuple[RenderRequest, FacadeGrid]:
    """The render request for a head-on view of `facade`, and the grid it will come back on.

    The pixel size is exact: the image width is chosen first and `extent_m` is derived
    from it, so metres per pixel is `gsd_cm / 100` and not a rounding of it.
    """
    spec = spec or FacadeOrthoSpec()
    gsd = spec.gsd_cm / 100.0

    u_span = facade.length + 2 * spec.margin_m
    v_span = facade.height + 2 * spec.margin_m
    # round() before ceil(): 20 m at 5 cm/px must be 400 px, not 401 from a float wobble.
    width = max(MIN_SIDE_PX, math.ceil(round(u_span / gsd, 6)))
    height = max(MIN_SIDE_PX, math.ceil(round(v_span / gsd, 6)))
    if width > MAX_SIDE or height > MAX_SIDE:
        coarsest = max(u_span, v_span) / MAX_SIDE * 100
        raise FacadeOrthoError(
            f"facade {facade.id} is {facade.length:.1f} × {facade.height:.1f} m: at "
            f"{spec.gsd_cm:g} cm/px that is {width} × {height} px, over the {MAX_SIDE} px "
            f"limit — use {coarsest:.1f} cm/px or coarser")

    u_centre = 0.5 * (facade.u_min + facade.u_max)
    v_centre = 0.5 * (facade.v_ground + facade.v_max)
    eye = facade.point(u_centre, v_centre) + spec.standoff_m * np.asarray(facade.normal, float)

    request = RenderRequest(
        mode="ortho", position=Position(e=float(eye[0]), n=float(eye[1]), u=float(eye[2])),
        yaw_deg=facade_view_yaw(facade), pitch_deg=0.0, roll_deg=0.0,
        width=int(width), height=int(height), extent_m=width * gsd,
        output=spec.output, background=spec.background, sh_degree=spec.sh_degree)

    # Pixel (i, j) covers camera x = (i + 0.5 − cx)/s, y = (j + 0.5 − cy)/s with
    # cx = width/2, cy = height/2; x runs along u and y runs down, i.e. along −v.
    u0 = u_centre + (0.5 - width / 2) * gsd
    v0 = v_centre - (0.5 - height / 2) * gsd
    grid = FacadeGrid(
        facade_id=facade.id, m_per_px=gsd, pixel00_uv=(u0, v0), du_uv=(gsd, 0.0),
        dv_uv=(0.0, -gsd),
        u_min=u_centre - width * gsd / 2, u_max=u_centre + width * gsd / 2,
        v_min=v_centre - height * gsd / 2, v_max=v_centre + height * gsd / 2,
        origin_enu=facade.origin, u_axis=facade.u_axis, normal=facade.normal,
        v_ground=facade.v_ground, azimuth_deg=facade.azimuth_deg)
    return request, grid
