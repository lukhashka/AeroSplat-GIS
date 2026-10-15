"""Render request/response schemas. Torch-free: the API validates a request before proxying it."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_SIDE = 4096


class Position(BaseModel):
    """A point in Site ENU metres (`e`, `n`, `u`) or WGS84 (`lat`, `lon`, `h`, ellipsoidal).

    ENU is relative to the Site origin. `u` / `h` may be omitted for an orthographic
    camera: it is then put just above the scene.
    """
    model_config = ConfigDict(extra="forbid")

    e: Optional[float] = None
    n: Optional[float] = None
    u: Optional[float] = None
    lat: Optional[float] = Field(None, ge=-90, le=90)
    lon: Optional[float] = Field(None, ge=-180, le=180)
    h: Optional[float] = None

    @model_validator(mode="after")
    def _one_frame(self) -> "Position":
        enu = (self.e is not None, self.n is not None)
        geo = (self.lat is not None, self.lon is not None)
        if any(enu) and any(geo):
            raise ValueError("give either e/n/u or lat/lon/h, not both")
        if not (all(enu) or all(geo)):
            raise ValueError("position needs e and n, or lat and lon")
        if all(enu) and self.h is not None or all(geo) and self.u is not None:
            raise ValueError("u goes with e/n, h goes with lat/lon")
        return self

    @property
    def is_geodetic(self) -> bool:
        return self.lat is not None

    @property
    def vertical(self) -> Optional[float]:
        return self.h if self.is_geodetic else self.u


class RenderRequest(BaseModel):
    """What to render. Angles follow aviation convention: yaw is the compass heading of the
    optical axis (0 = north, 90 = east), pitch is above the horizon (-90 = straight down),
    roll turns the camera about its axis (positive = clockwise, right side down)."""
    model_config = ConfigDict(extra="forbid")

    mode: Literal["persp", "ortho"] = "persp"
    position: Position
    yaw_deg: float = 0.0
    pitch_deg: Optional[float] = Field(None, ge=-90, le=90)     # default: -45 persp, -90 ortho
    roll_deg: float = 0.0
    look_at: Optional[Position] = None                         # overrides yaw/pitch (persp only)
    width: int = Field(1024, ge=16, le=MAX_SIDE)
    height: int = Field(768, ge=16, le=MAX_SIDE)
    fov_deg: float = Field(60.0, gt=1.0, lt=170.0)             # vertical field of view (persp)
    extent_m: float = Field(200.0, gt=0.01, le=100_000.0)      # ortho: scene width shown, metres
    output: Literal["rgb", "depth", "alpha"] = "rgb"
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)  # 0..1, behind transparent pixels
    sh_degree: Optional[int] = Field(None, ge=0, le=3)         # None: every band the scene has

    @model_validator(mode="after")
    def _check(self) -> "RenderRequest":
        if self.mode == "ortho" and self.look_at is not None:
            raise ValueError("look_at is for perspective cameras; an orthographic one uses yaw/pitch")
        if self.mode == "persp" and self.position.vertical is None:
            raise ValueError("a perspective camera needs a height (u or h)")
        if self.look_at is not None and self.look_at.vertical is None:
            raise ValueError("look_at needs a height (u or h)")
        if not all(0.0 <= c <= 1.0 for c in self.background):
            raise ValueError("background components must be in 0..1")
        return self

    @property
    def pitch(self) -> float:
        if self.pitch_deg is not None:
            return self.pitch_deg
        return -90.0 if self.mode == "ortho" else -45.0
