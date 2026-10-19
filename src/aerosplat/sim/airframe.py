"""What a vehicle profile says about the real airframe (configs/sim/vehicles/<name>.yaml).

The SITL part of a profile (`params`, `boot_params`) configures the simulated autopilot; the
sections here describe the hardware the profile stands for, as data for the planner (r_drone),
the VPS (camera and its mount), the companion and later the vehicle itself (ADR-0015 §6):

    frame:          wheelbase, propellers, r_drone, mass, battery, speeds; what is estimated
    cameras:        mounted cameras (role vps / vio), each a configs/sim/cameras "<camera>:<lens>"
    rangefinders:   layout (yaw / pitch relative to the nose), range
    flow:           optical-flow module
    companion:      board and MAVLink link
    nav:            how the vehicle carries GNSS (monitor-only by default for a GNSS-free fleet)
    hardware_params: autopilot parameters for the REAL vehicle only (never sent to SITL)

All sections are optional, so the SITL-only profiles (sitl-quad, sitl-quad-extnav) stay valid.
The SITL physics is always the built-in quad of configs/sitl.yaml: a profile that describes a
real frame says so in `frame.physics`, and nothing here changes how SITL flies.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aerosplat.mission.models import GnssUse

# r_drone may exceed the propeller-tip radius by this much at most (guards, tolerance); a larger
# value is a profile error rather than a margin (margins are the planner's m and k·σ).
R_DRONE_SLACK_M = 0.15


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Battery(_Model):
    chemistry: Literal["lipo", "liion"]
    cells: int = Field(ge=1, le=14)
    capacity_mah: float = Field(gt=0)


class Frame(_Model):
    """The airframe. Numbers from the vendor are named in `source`; everything else is listed in
    `estimated` (field names of this model), so nobody mistakes a guess for a measurement."""

    name: str
    wheelbase_m: float = Field(gt=0)            # motor-to-motor diagonal
    prop_diameter_m: float = Field(gt=0)
    r_drone_m: float = Field(gt=0, le=10)       # PlanningProfile.r_drone_m for this frame
    mass_takeoff_kg: Optional[float] = Field(None, gt=0)
    battery: Optional[Battery] = None
    cruise_speed_mps: Optional[float] = Field(None, gt=0)
    max_speed_mps: Optional[float] = Field(None, gt=0)
    source: str
    estimated: tuple[str, ...] = ()
    physics: str                                # what SITL actually flies

    @property
    def prop_tip_radius_m(self) -> float:
        """Radius of the circle that contains every propeller (X / + quad, centred hub)."""
        return self.wheelbase_m / 2.0 + self.prop_diameter_m / 2.0

    @model_validator(mode="after")
    def _check(self) -> "Frame":
        tip = self.prop_tip_radius_m
        if self.r_drone_m < tip:
            raise ValueError(f"r_drone_m {self.r_drone_m:g} m is inside the propeller tips "
                             f"({tip:.3f} m = wheelbase/2 + prop/2)")
        if self.r_drone_m > tip + R_DRONE_SLACK_M:
            raise ValueError(f"r_drone_m {self.r_drone_m:g} m exceeds the propeller tips ({tip:.3f} m) "
                             f"by more than {R_DRONE_SLACK_M} m: margins belong to the planner")
        if self.max_speed_mps is not None and self.cruise_speed_mps is not None \
                and self.cruise_speed_mps > self.max_speed_mps:
            raise ValueError("cruise_speed_mps exceeds max_speed_mps")
        unknown = set(self.estimated) - set(type(self).model_fields)
        if unknown:
            raise ValueError(f"estimated names unknown frame fields: {sorted(unknown)}")
        return self


class CameraMount(_Model):
    """A camera on the vehicle. Angles follow the render service (RenderRequest): pitch is above
    the horizon (-90 = straight down), yaw is relative to the nose (positive = right)."""

    name: str
    camera: str                                 # configs/sim/cameras "<camera>:<lens>"
    role: Literal["vps", "vio", "detection"]
    pitch_deg: float = Field(ge=-90, le=90)
    yaw_deg: float = Field(0.0, ge=-180, le=180)
    optional: bool = False
    note: str = ""


class Rangefinder(_Model):
    name: str
    model: str
    yaw_deg: float = Field(ge=-180, le=180)     # relative to the nose, positive = right
    pitch_deg: float = Field(ge=-90, le=90)     # -90 = down
    min_m: float = Field(ge=0)
    max_m: float = Field(gt=0)
    role: Literal["wall", "ground"]

    @model_validator(mode="after")
    def _range(self) -> "Rangefinder":
        if self.min_m >= self.max_m:
            raise ValueError(f"rangefinder {self.name}: min_m must be below max_m")
        return self


class Flow(_Model):
    model: str
    rangefinder: str                            # name of its downward rangefinder


class Companion(_Model):
    board: str
    link: str
    serial_baud: int = Field(gt=0)


class NavHardware(_Model):
    gnss: GnssUse = GnssUse.monitor             # default use of the receiver on this vehicle
    gnss_receiver: Optional[str] = None
    vps_camera: Optional[str] = None            # CameraMount.name with role vps


class Airframe(_Model):
    """The hardware sections of a vehicle profile, cross-checked against each other."""

    frame: Optional[Frame] = None
    cameras: tuple[CameraMount, ...] = ()
    rangefinders: tuple[Rangefinder, ...] = ()
    flow: Optional[Flow] = None
    companion: Optional[Companion] = None
    nav: Optional[NavHardware] = None

    def mount(self, name: str) -> CameraMount:
        for m in self.cameras:
            if m.name == name:
                return m
        raise KeyError(name)

    def vps_mount(self) -> Optional[CameraMount]:
        if self.nav is not None and self.nav.vps_camera is not None:
            return self.mount(self.nav.vps_camera)
        return next((m for m in self.cameras if m.role == "vps"), None)

    def check(self, camera: Optional[str]) -> list[str]:
        """Inconsistencies between sections (and the profile's top-level `camera`)."""
        out: list[str] = []
        names = [m.name for m in self.cameras]
        if len(set(names)) != len(names):
            out.append("camera mount names repeat")
        rf = [r.name for r in self.rangefinders]
        if len(set(rf)) != len(rf):
            out.append("rangefinder names repeat")
        if self.flow is not None and self.flow.rangefinder not in rf:
            out.append(f"flow names rangefinder '{self.flow.rangefinder}', which is not listed")
        if self.flow is not None and self.flow.rangefinder in rf:
            down = next(r for r in self.rangefinders if r.name == self.flow.rangefinder)
            if down.role != "ground" or down.pitch_deg > -80:
                out.append(f"flow rangefinder '{down.name}' does not look down")
        if self.nav is not None and self.nav.vps_camera is not None:
            if self.nav.vps_camera not in names:
                out.append(f"nav.vps_camera '{self.nav.vps_camera}' is not a mounted camera")
            elif self.mount(self.nav.vps_camera).role != "vps":
                out.append(f"nav.vps_camera '{self.nav.vps_camera}' does not have role vps")
        vps = None if out else self.vps_mount()
        if camera is not None and vps is not None and vps.camera != camera:
            out.append(f"profile camera '{camera}' is not the VPS camera '{vps.camera}'")
        return out
