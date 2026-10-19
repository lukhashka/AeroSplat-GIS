"""Camera profiles (configs/sim/cameras/<camera>.yaml): sensor, lenses, rolling shutter.

    cam = load_camera("imx290:4mm")
    cam.fx, cam.cx, cam.hfov_deg, cam.readout_ms(30), cam.gsd_m(50.0)

A profile is data shared by the simulation (render service frames, SIM-3), the localizability
map and the vehicle profile (ADR-0015 §2.6). Intrinsics follow the render service's pixel
convention (`aerosplat.render.camera`): the principal point of an ideal lens is
(width / 2, height / 2); `persp_intrinsics(width, height, vfov_deg)` gives the same focal length.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from aerosplat.paths import PROJECT_DIR

CAMERAS_DIR = PROJECT_DIR / "configs" / "sim" / "cameras"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Sensor(_Model):
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    pixel_um: float = Field(gt=0)
    effective_px: Optional[tuple[int, int]] = None
    shutter: str = Field(pattern="^(rolling|global)$")
    readout_ms: dict[int, float] = Field(default_factory=dict)   # fps -> full-frame readout
    max_fps_1080p: Optional[float] = None

    @property
    def width_mm(self) -> float:
        return self.width_px * self.pixel_um * 1e-3

    @property
    def height_mm(self) -> float:
        return self.height_px * self.pixel_um * 1e-3


class Lens(_Model):
    focal_mm: float = Field(gt=0)
    use: str = ""


class Distortion(_Model):
    model: str = "none"
    calibrated: bool = False


class CameraProfile(_Model):
    camera: str
    description: str = ""
    sensor: Sensor
    lenses: dict[str, Lens]
    distortion: Distortion = Distortion()


class Camera(_Model):
    """One camera with one lens: what a renderer, a VPS model or a planner needs."""

    name: str                       # "imx290:4mm"
    sensor: Sensor
    lens: Lens
    distortion: Distortion

    @property
    def width(self) -> int:
        return self.sensor.width_px

    @property
    def height(self) -> int:
        return self.sensor.height_px

    @property
    def fx(self) -> float:
        """Focal length in pixels (square pixels: fy = fx)."""
        return self.lens.focal_mm / (self.sensor.pixel_um * 1e-3)

    @property
    def fy(self) -> float:
        return self.fx

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def K(self) -> tuple[float, float, float, float]:
        return self.fx, self.fy, self.cx, self.cy

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.width / (2.0 * self.fx)))

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.height / (2.0 * self.fy)))

    def gsd_m(self, distance_m: float) -> float:
        """Ground sample distance at `distance_m` along the optical axis (m per pixel)."""
        return distance_m / self.fx

    def readout_ms(self, fps: float) -> float:
        """Rolling-shutter time from the first to the last row; 0 for a global shutter."""
        if self.sensor.shutter == "global":
            return 0.0
        table = self.sensor.readout_ms
        if int(fps) in table and float(int(fps)) == float(fps):
            return table[int(fps)]
        raise ValueError(f"{self.name}: no readout time for {fps:g} fps (known: "
                         f"{', '.join(str(k) for k in sorted(table))})")


def load_profile(camera: str, folder: Path = CAMERAS_DIR) -> CameraProfile:
    path = folder / f"{camera}.yaml"
    if not path.is_file():
        raise ValueError(f"no camera profile '{camera}' ({path})")
    prof = CameraProfile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if prof.camera != camera:
        raise ValueError(f"{path} describes camera '{prof.camera}', not '{camera}'")
    return prof


def load_camera(spec: str, folder: Path = CAMERAS_DIR) -> Camera:
    """'imx290:4mm' -> the IMX290 with its 4 mm lens."""
    camera, sep, lens = spec.partition(":")
    if not sep or not lens:
        raise ValueError(f"camera '{spec}' must be '<camera>:<lens>'")
    prof = load_profile(camera, folder)
    if lens not in prof.lenses:
        raise ValueError(f"camera {camera} has no lens '{lens}' ({', '.join(prof.lenses)})")
    return Camera(name=spec, sensor=prof.sensor, lens=prof.lenses[lens], distortion=prof.distortion)
