"""L0 tests of the fault mapping and of the pinned firmware's parameter table (no SITL).

The parameter snapshot (configs/ardupilot/<tag>.params.json) comes from the pinned build itself:
names and defaults from the running SITL, units and enumerations from its own param_parse.py.
Every name this repository writes to a vehicle is checked against it here.
"""

import re

import pytest

from aerosplat.mission.export import read_parm
from aerosplat.sim.faults import (ACTIONS, BATTERY_MARGIN_V, GPS_LOSS_DETECT_S, RUNNER_PARAMS,
                                  RUNNER_SYSID, FaultUnavailable, failsafe_trigger_problems,
                                  injection, response, wind_params)
from aerosplat.sim.params import (FirmwareParams, ParameterError, ParamInfo, load_snapshot,
                                  suggestions)
from aerosplat.sim.runner import load_vehicle
from aerosplat.sim.scenario import Fault, Wind
from aerosplat.sim.sitl import load_config


@pytest.fixture(scope="module")
def fw():
    return load_snapshot()


@pytest.fixture(scope="module")
def vehicle(fw):
    """The vehicle as a scenario sees it: firmware defaults + sitl-quad profile + runner role."""
    values = {n: i.default for n, i in fw.params.items()}
    values.update(load_vehicle("sitl-quad").params)
    values.update(RUNNER_PARAMS)
    return values


FAULTS = {
    "gps_loss": Fault(fault="gps_loss", at_s=1),
    "rc_loss": Fault(fault="rc_loss", at_s=1),
    "gcs_loss": Fault(fault="gcs_loss", at_s=1),
    "battery_low": Fault(fault="battery_sag", at_s=1, level="low"),
    "battery_critical": Fault(fault="battery_sag", at_s=1, level="critical"),
    "wind_gust": Fault(fault="wind_gust", at_s=1, speed_mps=8, dir_deg=90),
    "motor_degraded": Fault(fault="motor_degraded", at_s=1, motor=3, thrust=0.7),
}


# ── the snapshot is the pinned build's ───────────────────────────────────
@pytest.mark.hazard("H-003")
def test_snapshot_is_of_the_pinned_build(fw):
    pin = load_config().ardupilot
    assert (fw.tag, fw.commit) == (pin.tag, pin.commit)
    assert len(fw.params) > 1000
    # Copter 4.7 names and units, and none of the 4.5/4.6 ones they replaced.
    assert fw["RTL_ALT_M"].units == "m" and fw["WP_SPD"].units == "m/s"
    assert "RTL_ALT" not in fw and "WPNAV_SPEED" not in fw
    with pytest.raises(ParameterError, match="RTL_ALT_M"):
        fw["RTL_ALT"]


def test_a_snapshot_of_another_commit_is_refused(tmp_path, fw):
    import json

    doc = {"tag": fw.tag, "commit": "0" * 40, "params": {"X": {"default": 1}}}
    (tmp_path / f"{fw.tag}.params.json").write_text(json.dumps(doc))
    with pytest.raises(ParameterError, match="regenerate"):
        load_snapshot(fw.tag, fw.commit, folder=tmp_path)
    with pytest.raises(ParameterError, match="no parameter snapshot"):
        load_snapshot("Copter-9.9.9", fw.commit, folder=tmp_path)


@pytest.mark.hazard("H-003")
def test_the_8_1b_package_names_are_caught_with_their_new_names(fw):
    """The .parm that 8.1b exported (Copter 4.5/4.6 names, cm units) against Copter-4.7.2."""
    old = read_parm("RTL_ALT 1490\nRTL_ALT_FINAL 0\nRTL_CLIMB_MIN 0\nRTL_SPEED 0\n"
                    "WPNAV_SPEED 800\nWPNAV_SPEED_UP 250\nWPNAV_SPEED_DN 150\nFENCE_ENABLE 1\n")
    assert fw.unknown(old) == ["RTL_ALT", "RTL_ALT_FINAL", "RTL_CLIMB_MIN", "RTL_SPEED",
                               "WPNAV_SPEED", "WPNAV_SPEED_UP", "WPNAV_SPEED_DN"]
    msg = fw.problems(old)[0]
    for new in ("RTL_ALT_M", "RTL_ALT_FINAL_M", "RTL_CLIMB_MIN_M", "RTL_SPEED_MS"):
        assert new in msg
    assert "WP_SPD" in " ".join(suggestions("WPNAV_SPEED", fw.params)) or "WP_SPD" in msg


@pytest.mark.parametrize("values, match", [
    ({"FENCE_ACTION": 9}, "not one of"),
    ({"FENCE_TYPE": 64}, "undocumented bits"),
    ({"FENCE_TYPE": 1.5}, "not a bitmask"),
    ({"RTL_ALT_M": 5000}, "outside the documented range"),
    ({"FENCE_ALT_MAX": float("nan")}, "not a number"),
])
def test_values_the_firmware_would_not_take_are_caught(fw, values, match):
    problems = fw.problems(values)
    assert len(problems) == 1 and re.search(match, problems[0])


def test_values_are_described_in_the_firmwares_words(fw):
    assert fw["FENCE_ACTION"].describe(3) == "3 (SmartRTL or RTL or Land)"
    assert fw["FENCE_TYPE"].describe(7).startswith("7 (Max altitude, Circle")
    assert fw["RTL_ALT_M"].describe(15) == "15 m"


# ── fault mapping ────────────────────────────────────────────────────────
@pytest.mark.hazard("H-003")
@pytest.mark.parametrize("name", sorted(FAULTS))
def test_every_fault_maps_to_parameters_of_the_pinned_build(fw, vehicle, name):
    inj = injection(FAULTS[name], fw, vehicle)
    for change in inj.inject + inj.clear:
        assert change.name in fw, change.name
        assert fw[change.name].problem(change.value) is None, (change, fw[change.name])
        assert change.name.startswith("SIM_")       # faults are the simulator's, not the vehicle's
    assert [c.name for c in inj.inject] == [c.name for c in inj.clear]


def test_concrete_values_of_each_fault(fw, vehicle):
    gps = injection(FAULTS["gps_loss"], fw, vehicle)
    assert [(c.name, c.value) for c in gps.inject] == [("SIM_GPS1_ENABLE", 0.0)]
    assert [(c.name, c.value) for c in gps.clear] == [("SIM_GPS1_ENABLE", 1.0)]
    rc = injection(FAULTS["rc_loss"], fw, vehicle)
    assert [(c.name, c.value) for c in rc.inject] == [("SIM_RC_FAIL", 1.0)]
    gcs = injection(FAULTS["gcs_loss"], fw, vehicle)
    assert gcs.inject == () and gcs.heartbeats is False
    low = injection(FAULTS["battery_low"], fw, vehicle).inject[0].value
    crit = injection(FAULTS["battery_critical"], fw, vehicle).inject[0].value
    assert vehicle["BATT_CRT_VOLT"] < low < vehicle["BATT_LOW_VOLT"]
    assert crit == pytest.approx(vehicle["BATT_CRT_VOLT"] - BATTERY_MARGIN_V)
    motor = injection(FAULTS["motor_degraded"], fw, vehicle)
    assert [(c.name, c.value) for c in motor.inject] == [("SIM_ENGINE_FAIL", 4.0), ("SIM_ENGINE_MUL", 0.7)]
    gust = injection(FAULTS["wind_gust"], fw, vehicle, Wind(speed_mps=2, dir_deg=180))
    assert [(c.name, c.value) for c in gust.clear] == [("SIM_WIND_SPD", 2.0), ("SIM_WIND_DIR", 180.0),
                                                      ("SIM_WIND_TURB", 0.0)]
    assert wind_params(Wind(speed_mps=3, turbulence="moderate"), fw)["SIM_WIND_TURB"] == 2.0


@pytest.mark.hazard("H-008", "H-012")
def test_faults_that_cannot_fire_are_refused(fw, vehicle):
    """A fault whose failsafe trigger is disabled would 'pass' without testing anything."""
    no_crit = {**vehicle, "BATT_CRT_VOLT": 0.0}
    with pytest.raises(FaultUnavailable, match="BATT_CRT_VOLT is 0"):
        injection(FAULTS["battery_critical"], fw, no_crit)
    with pytest.raises(FaultUnavailable, match="BATT_LOW_VOLT is 0"):
        injection(FAULTS["battery_low"], fw, {**vehicle, "BATT_LOW_VOLT": 0.0})
    with pytest.raises(FaultUnavailable, match=f"MAV_GCS_SYSID = {RUNNER_SYSID}"):
        injection(FAULTS["gcs_loss"], fw, {**vehicle, "MAV_GCS_SYSID": 255.0})
    lacking = FirmwareParams(fw.tag, fw.commit, {k: v for k, v in fw.params.items()
                                                 if k != "SIM_GPS1_ENABLE"})
    with pytest.raises(FaultUnavailable, match="has no SIM_GPS1_ENABLE"):
        injection(FAULTS["gps_loss"], lacking, vehicle)


def test_the_vehicle_profile_is_valid_and_arms_the_critical_battery_failsafe(fw):
    prof = load_vehicle("sitl-quad")
    assert fw.problems(prof.params) == []
    assert prof.params["BATT_CRT_VOLT"] > 0 and fw.default("BATT_CRT_VOLT") == 0
    assert fw.problems(RUNNER_PARAMS) == []


# ── what the .parm prescribes ────────────────────────────────────────────
TOKENS = {"SMART_RTL": "SmartRTL", "RTL": "RTL", "LAND": "Land", "BRAKE": "Brake",
          "ALT_HOLD": "AltHold", "AUTO_RTL": "DO_LAND_START"}


@pytest.mark.parametrize("param", sorted(ACTIONS))
def test_action_table_says_what_the_firmware_says(fw, param):
    """ACTIONS is code; the firmware's own description of every value is data. They must agree:
    each mode's word appears in the description, in the same order; no mode = no action. The
    one thing the descriptions leave out is the final LAND: every RTL / SmartRTL / Brake action
    lands when it cannot enter its mode (set_mode_*_or_land_with_pause, ArduCopter/events.cpp)."""
    info = fw[param]
    assert info.values is not None
    for value, modes in ACTIONS[param].items():
        text = info.values[value]
        if not modes:
            assert re.search(r"Disabled|Report only|Warn only|NoAction", text, re.I), (param, value, text)
            continue
        if modes[-1] == "LAND" and len(modes) > 1 and "Land" not in text:
            modes = modes[:-1]
        pos = 0
        for mode in modes:
            k = text.find(TOKENS[mode], pos)
            assert k >= 0, (param, value, text, mode)
            pos = k + len(TOKENS[mode])


def test_responses_of_the_exported_package(vehicle):
    from aerosplat.mission.export import FS_OPTIONS_CONTINUE_IF_LANDING

    parm = {"FENCE_ENABLE": 1, "FENCE_ACTION": 3, "FS_THR_ENABLE": 4, "FS_GCS_ENABLE": 3,
            "FS_OPTIONS": FS_OPTIONS_CONTINUE_IF_LANDING, "BATT_FS_LOW_ACT": 3,
            "BATT_FS_CRT_ACT": 1, "FS_EKF_ACTION": 1, "FS_DR_ENABLE": 1}
    assert response("fence_breach", parm, vehicle).modes == ("SMART_RTL", "RTL", "LAND")
    rc = response("rc_loss", parm, vehicle)
    assert rc.modes[0] == "SMART_RTL" and rc.delay_s == vehicle["RC_FS_TIMEOUT"]
    gcs = response("gcs_loss", parm, vehicle)
    assert gcs.modes[0] == "SMART_RTL" and gcs.delay_s == vehicle["FS_GCS_TIMEOUT"]
    assert response("battery_low", parm, vehicle).delay_s == vehicle["BATT_LOW_TIMER"]
    assert response("battery_critical", parm, vehicle).modes == ("LAND",)
    gps = response("gps_loss", parm, vehicle)
    assert gps.modes == ("LAND",) and gps.delay_s == GPS_LOSS_DETECT_S


@pytest.mark.hazard("H-008")
def test_continue_options_and_disabled_triggers_change_the_answer(vehicle):
    parm = {"FS_THR_ENABLE": 4, "FS_GCS_ENABLE": 3, "FS_OPTIONS": 3, "BATT_FS_CRT_ACT": 1,
            "FENCE_ENABLE": 0, "FENCE_ACTION": 3}
    assert response("rc_loss", parm, vehicle).continues
    assert response("gcs_loss", parm, vehicle).continues
    assert response("fence_breach", parm, vehicle).continues
    with pytest.raises(FaultUnavailable, match="can never fire"):
        response("battery_critical", parm, {**vehicle, "BATT_CRT_VOLT": 0.0})


@pytest.mark.hazard("H-012")
def test_a_package_promising_a_failsafe_the_vehicle_cannot_trigger_is_reported(fw):
    """copter.parm leaves BATT_CRT_VOLT at 0: the package's BATT_FS_CRT_ACT would never fire."""
    defaults = {n: i.default for n, i in fw.params.items()}
    problems = failsafe_trigger_problems({"BATT_FS_CRT_ACT": 1, "BATT_FS_LOW_ACT": 3}, defaults)
    assert len(problems) == 1 and "critical battery failsafe cannot fire" in problems[0]
    with_profile = {**defaults, **load_vehicle("sitl-quad").params}
    assert failsafe_trigger_problems({"BATT_FS_CRT_ACT": 1}, with_profile) == []


def test_param_info_problem_rules():
    p = ParamInfo("X_ACT", 0, values={0: "a", 1: "b"})
    assert p.problem(1) is None and "not one of" in p.problem(2)
    assert ParamInfo("X_M", 0, units="m", low=0, high=10).problem(11) is not None
