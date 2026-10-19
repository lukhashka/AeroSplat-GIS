"""Camera profiles (configs/sim/cameras): the IMX290 of ADR-0015 §2.6."""

import math

import pytest

from aerosplat.render.camera import persp_intrinsics
from aerosplat.sim.camera import load_camera, load_profile


@pytest.mark.parametrize("lens, focal_mm, hfov, gsd50_cm", [("2.8mm", 2.8, 89.7, 5.2),
                                                             ("4mm", 4.0, 69.7, 3.6),
                                                             ("6mm", 6.0, 49.8, 2.4)])
def test_imx290_intrinsics_per_lens(lens, focal_mm, hfov, gsd50_cm):
    cam = load_camera(f"imx290:{lens}")
    assert (cam.width, cam.height) == (1920, 1080)
    assert cam.fx == pytest.approx(focal_mm / 0.0029) and cam.fy == cam.fx
    assert (cam.cx, cam.cy) == (960.0, 540.0)
    assert cam.hfov_deg == pytest.approx(hfov, abs=0.1)              # the ADR's ≈90 / 70 / 50°
    assert cam.gsd_m(50.0) * 100 == pytest.approx(gsd50_cm, abs=0.05)  # ... and its cm/px at 50 m
    # The render service's pinhole with this vertical FOV is the same camera.
    assert persp_intrinsics(1920, 1080, cam.vfov_deg) == pytest.approx(cam.K)


def test_imx290_sensor_and_rolling_shutter():
    prof = load_profile("imx290")
    assert prof.sensor.width_mm == pytest.approx(5.568) and prof.sensor.shutter == "rolling"
    cam = load_camera("imx290:4mm")
    # 1080 rows of a 1125-line frame: 16.0 ms at 60 fps, 32.0 ms at 30 fps.
    assert cam.readout_ms(60) == pytest.approx(1080 / 1125 / 60 * 1000)
    assert cam.readout_ms(30) == pytest.approx(32.0)
    assert not cam.distortion.calibrated
    with pytest.raises(ValueError, match="no readout time"):
        cam.readout_ms(50)


def test_unknown_cameras_and_lenses_are_refused():
    with pytest.raises(ValueError, match="no lens"):
        load_camera("imx290:8mm")
    with pytest.raises(ValueError, match="no camera profile"):
        load_camera("ov9281:3mm")
    with pytest.raises(ValueError, match="<camera>:<lens>"):
        load_camera("imx290")
    assert math.isclose(load_camera("imx290:6mm").vfov_deg, 2 * math.degrees(math.atan(540 / (6 / 0.0029))))
