"""SIM-0 smoke (L2): heartbeat -> GUIDED -> arm -> take off to 10 m -> land -> disarm.

Judged on the simulator's TRUE state (SIM_STATE); the EKF estimate is only compared against it.
Runs with `pytest -m sitl`; the fixtures in tests/conftest.py find or start SITL.
"""

import pytest

from aerosplat.sim.vehicle import TruthMonitor, horizontal_distance_m

pytestmark = pytest.mark.sitl

TAKEOFF_M = 10.0
HOLD_TOL_M = 1.0          # true height while hovering: TAKEOFF_M +- this
OVERSHOOT_M = 1.5         # true height never above TAKEOFF_M + this
GROUND_TOL_M = 0.3        # true height above the start point after landing
DRIFT_M = 2.0             # true horizontal distance from the start point, whole flight
EST_VERT_TOL_M = 1.0      # |EKF height above home - true height above ground|
EST_HORZ_TOL_M = 2.0      # |EKF position - true position|


@pytest.mark.hazard("H-003")
def test_pinned_firmware(vehicle, sitl_cfg):
    """The vehicle answering is the pinned build, not whatever SITL happened to be running."""
    fw = vehicle.firmware()
    assert fw.git_hash == sitl_cfg.ardupilot.commit[:8]
    assert fw.version == sitl_cfg.ardupilot.version
    assert fw.release_type == 255, "expected an official release build (FIRMWARE_VERSION_TYPE_OFFICIAL)"


@pytest.mark.hazard("H-004", "H-010")
def test_takeoff_land_smoke(vehicle, sitl_run):
    v = vehicle
    v.stream_defaults()
    start = v.wait_truth()
    assert not v.armed, "vehicle is armed before the test: leftover state, restart SITL"
    assert start.speed < 0.1, f"vehicle is moving before the test: {start}"
    if sitl_run.home is not None:  # SITL home = the site origin it was launched with
        off = horizontal_distance_m(start.lat, start.lon, sitl_run.home.lat, sitl_run.home.lon)
        assert off < 1.0, f"true start {off:.2f} m from the launch home {sitl_run.home.arg()}"
        assert abs(start.alt_amsl - sitl_run.home.alt) < 1.0

    ground = start.alt_amsl
    # The estimate is compared from the moment the autopilot itself calls it good (pre-arm
    # passed: EKF has an origin); before that GLOBAL_POSITION_INT carries lat = lon = 0.
    v.wait_prearm_ok(timeout=120)
    monitor = TruthMonitor(ground_alt=ground, start_lat=start.lat, start_lon=start.lon)
    v.listeners.append(monitor)
    v.set_mode("GUIDED")
    v.arm(timeout=30)
    v.takeoff(TAKEOFF_M)
    v.wait("true height near the take-off altitude",
           lambda: v.truth().alt_amsl - ground >= TAKEOFF_M - HOLD_TOL_M / 2, timeout=60)

    # Hover: the true height stays inside the band for 3 s.
    hover = v.sample(lambda: v.truth().alt_amsl - ground, seconds=3.0)
    assert min(hover) > TAKEOFF_M - HOLD_TOL_M and max(hover) < TAKEOFF_M + HOLD_TOL_M, hover

    v.set_mode("LAND")
    v.wait("landed (truth) and disarmed",
           lambda: not v.armed and v.truth().alt_amsl - ground < GROUND_TOL_M, timeout=90)
    v.listeners.remove(monitor)

    end = v.wait_truth()
    assert abs(end.alt_amsl - ground) < GROUND_TOL_M
    assert end.speed < 0.2
    assert monitor.max_truth_agl < TAKEOFF_M + OVERSHOOT_M, monitor
    assert monitor.max_truth_drift < DRIFT_M, monitor
    # The estimate agreed with the truth for the whole flight.
    assert monitor.samples > 100, monitor
    assert monitor.max_vertical_error < EST_VERT_TOL_M, monitor
    assert monitor.max_horizontal_error < EST_HORZ_TOL_M, monitor
