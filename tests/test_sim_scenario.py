"""L0 tests of the scenario format (SIMULATION_AND_DELIVERY §4.5): what a file may say, and the
things it may not — anything L2 cannot honour is refused when read, never silently ignored."""

import pytest
import yaml
from pydantic import ValidationError

from aerosplat.sim.scenario import (DEFAULT_INVARIANTS, Fault, FaultKind, Invariant, Outcome,
                                    Scenario, load_scenario, resolve)

EXAMPLE = """
scenario: street-window-gps-loss
description: GPS lost while holding in front of the window
site: demo/street
epoch: baseline
scene: street
target: window
handover: hover_handoff
mission: missions/msn_01M58GPHJNMFFXSTTN08BKBS58
vehicle: sitl-quad
ardupilot: {tag: Copter-4.7.2}
environment:
  wind: {speed_mps: 3, dir_deg: 240, turbulence: light}
faults:
  - {fault: gps_loss, after: hover, at_s: 12}
expect:
  outcome: abort
  reason: gps_loss
  modes: [LAND]
  tolerances: {react_s: 6}
seeds: 3
"""


def scenario(**over):
    data = yaml.safe_load(EXAMPLE)
    for k, v in over.items():
        if v is None:
            data.pop(k, None)
        else:
            data[k] = v
    return Scenario.model_validate(data)


def expect(**kw):
    return {"outcome": "complete", "reason": "delivered", **kw}


def test_the_documented_shape_loads(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(EXAMPLE, encoding="utf-8")
    sc = load_scenario(path)
    assert sc.scenario == "street-window-gps-loss" and sc.seeds == 3
    assert sc.faults == [Fault(fault=FaultKind.gps_loss, after="hover", at_s=12)]
    assert sc.faults[0].reason == "gps_loss"
    assert sc.environment.wind.turbulence_mps == 1.0          # "light"
    assert sc.expect.checked == DEFAULT_INVARIANTS[Outcome.abort]
    assert sc.expect.tolerances.react_s == 6 and sc.expect.tolerances.hover_m == 1.0
    assert resolve(tmp_path, sc.mission) == tmp_path / "missions" / "msn_01M58GPHJNMFFXSTTN08BKBS58"


def test_every_fault_kind_with_its_own_fields():
    faults = [{"fault": "gps_loss", "at_s": 1}, {"fault": "rc_loss", "at_s": 1, "duration_s": 5},
              {"fault": "gcs_loss", "at_s": 1, "after": "over_entry:reached"},
              {"fault": "battery_sag", "at_s": 1, "level": "critical", "duration_s": 20},
              {"fault": "battery_sag", "at_s": 30, "voltage_v": 10.2},
              {"fault": "wind_gust", "at_s": 1, "speed_mps": 8, "dir_deg": 90, "duration_s": 3},
              {"fault": "motor_degraded", "at_s": 1, "motor": 2, "thrust": 0.8}]
    sc = scenario(faults=faults, expect=expect())
    assert [f.fault.value for f in sc.faults] == ["gps_loss", "rc_loss", "gcs_loss", "battery_sag",
                                                  "battery_sag", "wind_gust", "motor_degraded"]
    assert [f.reason for f in sc.faults][3:5] == ["battery_critical", "battery_low"]


def test_violation_and_refusal_expectations():
    v = scenario(faults=[], expect={"outcome": "violation", "reason": "clearance"})
    assert Invariant.clearance in v.expect.checked
    r = scenario(faults=[], expect={"outcome": "refused", "reason": "unknown_parameter"})
    assert r.expect.checked == ()
    f = scenario(faults=[], expect={"outcome": "abort", "reason": "fence_breach", "modes": ["SMART_RTL"]})
    assert f.expect.modes == ["SMART_RTL"]


# ── negative: malformed or unsupported scenarios are refused when read ───
@pytest.mark.parametrize("over, match", [
    ({"faults": [{"fault": "gps_jam", "at_s": 1}]}, "fault"),
    ({"faults": [{"fault": "gps_loss", "at_s": -1}]}, "at_s"),
    ({"faults": [{"fault": "gps_loss", "at_s": 1, "level": "low"}]}, "does not take level"),
    ({"faults": [{"fault": "battery_sag", "at_s": 1}]}, "exactly one of"),
    ({"faults": [{"fault": "battery_sag", "at_s": 1, "level": "low", "voltage_v": 10}]}, "exactly one of"),
    ({"faults": [{"fault": "battery_sag", "at_s": 1, "level": "empty"}]}, "low / critical"),
    ({"faults": [{"fault": "wind_gust", "at_s": 1}]}, "speed_mps"),
    ({"faults": [{"fault": "motor_degraded", "at_s": 1, "motor": 1}]}, "motor and thrust"),
    ({"faults": [{"fault": "gps_loss", "at_s": 1, "after": "hover:left"}]}, "reached"),
    ({"faults": [{"fault": "gps_loss", "at_s": 1, "after": "start:reached"}]}, "reached"),
    ({"faults": [{"fault": "gps_loss", "at_s": 1}, {"fault": "gps_loss", "at_s": 5}]}, "overlap"),
    ({"faults": [{"fault": "gps_loss", "at_s": 1, "duration_s": 5},
                 {"fault": "gps_loss", "at_s": 3}]}, "overlap"),
    ({"faults": [{"fault": "rc_loss", "at_s": 1}]}, "no fault causes it"),
    ({"expect": {"outcome": "complete", "reason": "landed"}}, "delivered"),
    ({"expect": {"outcome": "abort", "reason": "tired"}}, "abort reason"),
    ({"expect": {"outcome": "refused", "reason": "dunno"}}, "refusal reason"),
    ({"expect": {"outcome": "violation", "reason": "fence", "invariants": ["clearance"]}}, "not checked"),
    ({"expect": {"outcome": "violation", "reason": "speed"}}, "must name an invariant"),
    ({"expect": {"outcome": "abort", "reason": "gps_loss", "modes": ["HOVER"]}}, "unknown Copter modes"),
    ({"expect": {"outcome": "complete", "reason": "delivered", "modes": ["LAND"]}}, "failsafe modes"),
    ({"actors": [{"kind": "person", "path": [[0, 0], [1, 1]]}]}, "L3"),
    ({"environment": {"gnss": {"model": "sky_view"}}}, "SIM-2"),
    ({"environment": {"wind": {"speed_mps": 3, "turbulence": "gusty"}}}, "turbulence"),
    ({"seeds": 0}, "seeds"),
    ({"scenario": "Has Spaces"}, "scenario"),
    ({"origin": {"lat": 1, "lon": 2}}, "lat, lon, h"),
    ({"wind": 3}, "wind"),                                     # unknown top-level key
])
def test_invalid_scenarios_are_refused(over, match):
    with pytest.raises(ValidationError, match=match):
        scenario(**over)


def test_unknown_keys_inside_sections_are_refused():
    with pytest.raises(ValidationError):
        scenario(expect={"outcome": "abort", "reason": "gps_loss", "modes": ["LAND"], "speed": 1})
    with pytest.raises(ValidationError):
        scenario(environment={"wind": {"speed_mps": 3, "gust": 9}})


def test_a_yaml_that_is_not_a_mapping_is_refused(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a scenario mapping"):
        load_scenario(path)
