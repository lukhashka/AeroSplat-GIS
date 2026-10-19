"""Vehicle profiles (configs/sim/vehicles): the SITL part and the airframe data (ADR-0015 §6).

The x500-pi5-imx290 test bed must load with the existing runner code, give the planner an r_drone
that matches its frame and propellers, resolve its VPS camera to the IMX290 4 mm, and name only
parameters of the pinned build (SITL part and the real vehicle's `hardware_params`).
"""

import math

import pytest
from pydantic import ValidationError

from aerosplat.mission.models import GnssUse, PlanningProfile
from aerosplat.sim.airframe import R_DRONE_SLACK_M
from aerosplat.sim.camera import load_camera
from aerosplat.sim.faults import RUNNER_PARAMS, injection
from aerosplat.sim.params import load_snapshot
from aerosplat.sim.runner import VEHICLES_DIR, VehicleProfile, load_vehicle
from aerosplat.sim.scenario import Fault, Wind

X500 = "x500-pi5-imx290"


@pytest.fixture(scope="module")
def fw():
    return load_snapshot(variant="extnav")


@pytest.fixture(scope="module")
def x500():
    return load_vehicle(X500)


def test_every_profile_in_the_folder_loads():
    names = sorted(p.stem for p in VEHICLES_DIR.glob("*.yaml"))
    assert {"sitl-quad", "sitl-quad-extnav", X500} <= set(names)
    for name in names:
        assert load_vehicle(name).vehicle == name


def test_old_profiles_have_no_airframe_and_keep_their_meaning():
    for name in ("sitl-quad", "sitl-quad-extnav"):
        p = load_vehicle(name)
        assert p.frame is None and p.cameras == () and p.hardware_params == {}
    assert load_vehicle("sitl-quad-extnav").camera == "imx290:4mm"


def test_r_drone_is_the_propeller_tip_radius_rounded_up(x500):
    f = x500.frame
    assert (f.wheelbase_m, f.prop_diameter_m) == (0.5, 0.254)
    tip = f.wheelbase_m / 2 + f.prop_diameter_m / 2
    assert f.prop_tip_radius_m == pytest.approx(0.377)
    assert tip <= f.r_drone_m <= tip + R_DRONE_SLACK_M
    assert f.r_drone_m == 0.40
    # The planner's own default is untouched by this profile (it is set per plan).
    assert PlanningProfile().r_drone_m == 0.5
    assert PlanningProfile(r_drone_m=f.r_drone_m).r_drone_m == 0.40


def test_estimates_are_named_and_physics_is_said_to_be_sitls(x500):
    f = x500.frame
    assert {"mass_takeoff_kg", "battery", "cruise_speed_mps", "max_speed_mps"} <= set(f.estimated)
    assert "SITL" in f.physics and "not simulated" in f.physics
    assert f.battery.cells == 4


def test_vps_camera_resolves_to_imx290_4mm(x500):
    mount = x500.airframe.vps_mount()
    assert mount is not None and mount.name == "vps_main" and mount.role == "vps"
    assert mount.camera == x500.camera == "imx290:4mm"
    assert -90 < mount.pitch_deg < 0                          # down-forward, not nadir
    cam = load_camera(mount.camera)
    assert cam.fx == pytest.approx(4.0 / 0.0029)
    assert cam.hfov_deg == pytest.approx(69.7, abs=0.1)
    # The optional VIO camera is a global shutter (no rolling-shutter readout).
    vio = x500.airframe.mount("vio")
    assert vio.optional and load_camera(vio.camera).readout_ms(60) == 0.0


def test_sensor_layout(x500):
    rf = {r.name: r for r in x500.rangefinders}
    assert rf["down"].pitch_deg == -90 and rf["down"].role == "ground"
    walls = sorted(r.yaw_deg for r in x500.rangefinders if r.role == "wall")
    assert walls == [-30, 0, 30]
    assert x500.flow is not None and x500.flow.rangefinder == "down"
    assert x500.nav.gnss is GnssUse.monitor and x500.nav.vps_camera == "vps_main"
    assert x500.companion.serial_baud == 921600


@pytest.mark.hazard("H-003")
def test_every_parameter_is_the_pinned_builds(fw, x500):
    fw.require(x500.params, X500)
    fw.require(x500.boot_params, f"{X500} (boot)")
    assert fw.unknown(x500.hardware_params) == []
    assert fw.problems(x500.hardware_params) == []
    # The companion link the hardware section declares is the one the parameters set.
    assert x500.hardware_params["SERIAL2_BAUD"] * 1000 == pytest.approx(x500.companion.serial_baud,
                                                                        rel=0.001)
    # SITL keeps its own drivers; hardware drivers never reach SITL.
    assert x500.boot_params["FLOW_TYPE"] == 10 and x500.hardware_params["FLOW_TYPE"] == 6


@pytest.mark.hazard("H-012")
def test_the_4s_pack_does_not_trip_its_own_thresholds_and_a_sag_restores_to_4s(fw, x500):
    p = x500.params
    full = x500.boot_params["SIM_BATT_VOLTAGE"]
    assert full == pytest.approx(4.2 * 4)
    assert p["BATT_CRT_VOLT"] < p["BATT_LOW_VOLT"] < full
    assert math.isclose(p["MOT_BAT_VOLT_MAX"], full)
    values = {n: i.default for n, i in fw.params.items()}
    values.update(x500.boot_params)
    values.update(x500.params)
    values.update(RUNNER_PARAMS)
    inj = injection(Fault(fault="battery_sag", at_s=1, level="low"), fw, values, Wind())
    assert p["BATT_CRT_VOLT"] < inj.inject[0].value < p["BATT_LOW_VOLT"]
    assert inj.clear[0].value == pytest.approx(full)        # not the 3S 12.6 V


def _base(**over):
    raw = load_vehicle(X500).model_dump()
    raw.update(over)
    return raw


def test_r_drone_inside_the_propellers_is_refused():
    raw = _base()
    raw["frame"]["r_drone_m"] = 0.30
    with pytest.raises(ValidationError, match="inside the propeller tips"):
        VehicleProfile.model_validate(raw)
    raw["frame"]["r_drone_m"] = 0.70
    with pytest.raises(ValidationError, match="margins belong to the planner"):
        VehicleProfile.model_validate(raw)


def test_estimated_must_name_real_fields():
    raw = _base()
    raw["frame"]["estimated"] = ["mass_kg"]
    with pytest.raises(ValidationError, match="unknown frame fields"):
        VehicleProfile.model_validate(raw)


def test_profile_camera_must_be_the_vps_camera():
    with pytest.raises(ValidationError, match="is not the VPS camera"):
        VehicleProfile.model_validate(_base(camera="imx290:6mm"))


def test_flow_needs_a_downward_rangefinder():
    raw = _base()
    raw["flow"]["rangefinder"] = "front"
    with pytest.raises(ValidationError, match="does not look down"):
        VehicleProfile.model_validate(raw)
    raw["flow"]["rangefinder"] = "nope"
    with pytest.raises(ValidationError, match="not listed"):
        VehicleProfile.model_validate(raw)


def test_nav_vps_camera_must_be_a_vps_mount():
    raw = _base()
    raw["nav"]["vps_camera"] = "vio"
    with pytest.raises(ValidationError, match="does not have role vps"):
        VehicleProfile.model_validate(raw)


def test_test_bed_vps_runs_akaze_at_960x540_on_the_45_degree_camera():
    """The operating point chosen 2026-10-19 (ADR-0015 §6–7) is what the profile carries."""
    from aerosplat import vps
    from aerosplat.sim.runner import load_vehicle

    v = load_vehicle("x500-pi5-imx290")
    (cam,) = [c for c in v.cameras if c.role == "vps"]
    assert cam.pitch_deg == -45 and cam.camera == "imx290:4mm"
    assert (cam.vps.extractor, tuple(cam.vps.resolution)) == (vps.DEFAULT_EXTRACTOR, vps.DEFAULT_RESOLUTION)


def test_vps_settings_only_on_the_vps_camera():
    import pytest
    from pydantic import ValidationError

    from aerosplat.sim.airframe import CameraMount

    with pytest.raises(ValidationError, match="vps settings on a vio camera"):
        CameraMount(name="v", camera="imx296:6mm", role="vio", pitch_deg=-30, vps={})
