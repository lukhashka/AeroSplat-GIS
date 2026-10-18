"""Abstract faults -> concrete SIM_* parameters of the pinned build, and what the autopilot
should do about each one according to the mission's `.parm` (SIMULATION §4.5, §4.6).

Names are not written from memory: every parameter used here is checked against the pinned
build's own table (`aerosplat.sim.params`, configs/ardupilot/<tag>.params.json) by the tests,
and against the live vehicle by the runner before anything is injected.

    gps_loss        SIM_GPS1_ENABLE 0                     -> dead-reckoning / EKF failsafe
    rc_loss         SIM_RC_FAIL 1 (no RC pulses)          -> radio failsafe (FS_THR_ENABLE)
    gcs_loss        runner stops its GCS heartbeats       -> GCS failsafe (FS_GCS_ENABLE)
    battery_sag     SIM_BATT_VOLTAGE below BATT_LOW_VOLT / BATT_CRT_VOLT -> battery failsafe
    wind_gust       SIM_WIND_SPD / _DIR / _TURB           -> none expected
    motor_degraded  SIM_ENGINE_FAIL (mask) + SIM_ENGINE_MUL (thrust) -> none expected

`gcs_loss` has no SIM_* parameter: ArduPilot counts heartbeats from MAV_GCS_SYSID only, so the
runner makes itself the GCS (RUNNER_PARAMS) and the fault is the runner going silent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

from aerosplat.sim.params import FirmwareParams, ParameterError
from aerosplat.sim.scenario import Environment, Fault, FaultKind, Wind

RUNNER_SYSID = 245
# The runner is the vehicle's GCS: GCS loss is the runner's silence, not MAVProxy's.
RUNNER_PARAMS = {"MAV_GCS_SYSID": float(RUNNER_SYSID)}
BATTERY_MARGIN_V = 0.3          # how far below a threshold a sag goes
# GPS loss is not acted on at once: EKF3 keeps a position on the IMU alone until its variance
# passes FS_EKF_THRESH. SITL (Copter-4.7.2, SIM-1, 2026-10-19): 7.9 s, the mission continuing
# in AUTO meanwhile. Not something the package controls; a hazard of its own (H-011).
GPS_LOSS_DETECT_S = 10.0

# What each failsafe action value makes the vehicle do, as Copter modes in the order the
# firmware tries them ("SmartRTL or RTL or Land": SMART_RTL, else RTL, else LAND). The firmware's
# own words for every value are in the parameter snapshot; tests check that this table says the
# same thing. An empty tuple: no mode change (report only, or continue the mission).
ACTIONS: dict[str, dict[int, tuple[str, ...]]] = {
    "FENCE_ACTION": {0: (), 1: ("RTL", "LAND"), 2: ("LAND",), 3: ("SMART_RTL", "RTL", "LAND"),
                     4: ("BRAKE", "LAND"), 5: ("SMART_RTL", "LAND")},
    "FS_THR_ENABLE": {0: (), 1: ("RTL", "LAND"), 3: ("LAND",), 4: ("SMART_RTL", "RTL", "LAND"),
                      5: ("SMART_RTL", "LAND"), 6: ("AUTO_RTL", "RTL", "LAND"), 7: ("BRAKE", "LAND")},
    "FS_GCS_ENABLE": {0: (), 1: ("RTL", "LAND"), 3: ("SMART_RTL", "RTL", "LAND"),
                      4: ("SMART_RTL", "LAND"), 5: ("LAND",), 6: ("AUTO_RTL", "RTL", "LAND"),
                      7: ("BRAKE", "LAND")},
    "BATT_FS_LOW_ACT": {0: (), 1: ("LAND",), 2: ("RTL", "LAND"), 3: ("SMART_RTL", "RTL", "LAND"),
                        4: ("SMART_RTL", "LAND"), 6: ("AUTO_RTL", "RTL", "LAND"),
                        7: ("BRAKE", "LAND")},
    "BATT_FS_CRT_ACT": {0: (), 1: ("LAND",), 2: ("RTL", "LAND"), 3: ("SMART_RTL", "RTL", "LAND"),
                        4: ("SMART_RTL", "LAND"), 6: ("AUTO_RTL", "RTL", "LAND"),
                        7: ("BRAKE", "LAND")},
    "FS_DR_ENABLE": {0: (), 1: ("LAND",), 2: ("RTL", "LAND"), 3: ("SMART_RTL", "RTL", "LAND"),
                     4: ("SMART_RTL", "LAND"), 6: ("AUTO_RTL", "RTL", "LAND")},
    "FS_EKF_ACTION": {0: (), 1: ("LAND",), 2: ("ALT_HOLD",), 3: ("LAND",)},
}
# FS_OPTIONS bits that make a failsafe carry on in AUTO.
FS_OPT_CONTINUE_AUTO_RC = 1 << 0
FS_OPT_CONTINUE_AUTO_GCS = 1 << 1


class FaultUnavailable(ParameterError):
    """The fault cannot be injected on this vehicle as configured (no such parameter, or the
    failsafe it should provoke has its trigger disabled)."""


@dataclass(frozen=True)
class ParamChange:
    name: str
    value: float


@dataclass(frozen=True)
class Injection:
    """Concrete steps for one fault: set on inject, put back on clear."""

    fault: Fault
    inject: tuple[ParamChange, ...]
    clear: tuple[ParamChange, ...]
    heartbeats: Optional[bool] = None      # gcs_loss: runner heartbeats off (inject) / on (clear)
    note: str = ""

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.inject]


@dataclass(frozen=True)
class Response:
    """What the .parm prescribes for one trigger: the acceptable modes (preferred first), the
    parameter that says so, and how long the firmware waits before acting."""

    trigger: str                          # fence_breach, gps_loss, ...
    modes: tuple[str, ...]
    by: tuple[str, ...]                   # the parameters (and values) that decide it
    delay_s: float = 0.0
    note: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def continues(self) -> bool:
        return not self.modes


def _value(name: str, parm: Mapping[str, float], vehicle: Mapping[str, float]) -> float:
    """The value the vehicle will run with: the package's if it sets it, else the vehicle's."""
    if name in parm:
        return float(parm[name])
    if name in vehicle:
        return float(vehicle[name])
    raise FaultUnavailable(f"{name} is neither in the mission package nor on the vehicle")


def _action(name: str, parm, vehicle) -> tuple[str, ...]:
    v = int(_value(name, parm, vehicle))
    table = ACTIONS[name]
    if v not in table:
        raise FaultUnavailable(f"{name} = {v}: no mode mapping for this value (Terminate or removed)")
    return table[v]


def injection(fault: Fault, fw: FirmwareParams, vehicle: Mapping[str, float],
              wind: Wind = Wind()) -> Injection:
    """The SIM_* changes for `fault` on a vehicle whose parameters are `vehicle` (what it holds
    before the fault: thresholds come from there, and clearing puts these values back)."""

    def have(*names: str) -> None:
        missing = [n for n in names if n not in fw]
        if missing:
            raise FaultUnavailable(f"{fault.fault.value}: {fw.tag} has no {', '.join(missing)}")

    def back(name: str, default: float) -> ParamChange:
        return ParamChange(name, float(vehicle.get(name, default)))

    k = fault.fault
    if k is FaultKind.gps_loss:
        have("SIM_GPS1_ENABLE")
        return Injection(fault, (ParamChange("SIM_GPS1_ENABLE", 0.0),),
                         (back("SIM_GPS1_ENABLE", 1.0),), note="GPS 1 stops sending")
    if k is FaultKind.rc_loss:
        have("SIM_RC_FAIL")
        return Injection(fault, (ParamChange("SIM_RC_FAIL", 1.0),), (back("SIM_RC_FAIL", 0.0),),
                         note="no RC pulses (SIM_RC_FAIL 1)")
    if k is FaultKind.gcs_loss:
        have("MAV_GCS_SYSID")
        if int(vehicle.get("MAV_GCS_SYSID", -1)) != RUNNER_SYSID:
            raise FaultUnavailable(f"gcs_loss needs MAV_GCS_SYSID = {RUNNER_SYSID} (the runner as "
                                   f"GCS); the vehicle has {vehicle.get('MAV_GCS_SYSID')}")
        return Injection(fault, (), (), heartbeats=False,
                         note=f"runner (sysid {RUNNER_SYSID}) stops its GCS heartbeats")
    if k is FaultKind.battery_sag:
        have("SIM_BATT_VOLTAGE", "BATT_LOW_VOLT", "BATT_CRT_VOLT")
        low, crit = float(vehicle.get("BATT_LOW_VOLT", 0)), float(vehicle.get("BATT_CRT_VOLT", 0))
        if fault.voltage_v is not None:
            volts = float(fault.voltage_v)
        elif fault.level == "critical":
            if crit <= 0:
                raise FaultUnavailable("battery_sag critical: BATT_CRT_VOLT is 0, the critical "
                                       "battery failsafe is disabled on this vehicle")
            volts = crit - BATTERY_MARGIN_V
        else:
            if low <= 0:
                raise FaultUnavailable("battery_sag low: BATT_LOW_VOLT is 0, the low battery "
                                       "failsafe is disabled on this vehicle")
            volts = max(crit + BATTERY_MARGIN_V, low - BATTERY_MARGIN_V) if crit > 0 \
                else low - BATTERY_MARGIN_V
            if crit > 0 and not crit < volts < low:
                raise FaultUnavailable(f"battery_sag low: no voltage between BATT_CRT_VOLT {crit:g} "
                                       f"and BATT_LOW_VOLT {low:g} with {BATTERY_MARGIN_V} V margin")
        return Injection(fault, (ParamChange("SIM_BATT_VOLTAGE", round(volts, 3)),),
                         (back("SIM_BATT_VOLTAGE", 12.6),),
                         note=f"battery at {volts:.2f} V (low {low:g} V, critical {crit:g} V)")
    if k is FaultKind.wind_gust:
        have("SIM_WIND_SPD", "SIM_WIND_DIR", "SIM_WIND_TURB")
        assert fault.speed_mps is not None
        inj = (ParamChange("SIM_WIND_SPD", float(fault.speed_mps)),
               ParamChange("SIM_WIND_DIR", float(fault.dir_deg if fault.dir_deg is not None
                                                 else wind.dir_deg)),
               ParamChange("SIM_WIND_TURB", float(fault.turbulence_mps if fault.turbulence_mps
                                                  is not None else wind.turbulence_mps)))
        clr = (ParamChange("SIM_WIND_SPD", wind.speed_mps), ParamChange("SIM_WIND_DIR", wind.dir_deg),
               ParamChange("SIM_WIND_TURB", wind.turbulence_mps))
        return Injection(fault, inj, clr, note=f"wind {fault.speed_mps:g} m/s")
    if k is FaultKind.motor_degraded:
        have("SIM_ENGINE_FAIL", "SIM_ENGINE_MUL")
        assert fault.motor is not None and fault.thrust is not None
        return Injection(fault, (ParamChange("SIM_ENGINE_FAIL", float(1 << (fault.motor - 1))),
                                 ParamChange("SIM_ENGINE_MUL", float(fault.thrust))),
                         (back("SIM_ENGINE_FAIL", 0.0), back("SIM_ENGINE_MUL", 1.0)),
                         note=f"motor {fault.motor} at {fault.thrust:.0%} thrust")
    raise FaultUnavailable(f"no mapping for fault {k}")


def wind_params(wind: Wind, fw: FirmwareParams) -> dict[str, float]:
    """The steady environment (scenario `environment.wind`) as SIM_WIND_* values."""
    out = {"SIM_WIND_SPD": wind.speed_mps, "SIM_WIND_DIR": wind.dir_deg,
           "SIM_WIND_TURB": wind.turbulence_mps}
    missing = fw.unknown(out)
    if missing:
        raise FaultUnavailable(f"wind: {fw.tag} has no {', '.join(missing)}")
    return out


# SITL's own ground follows SRTM terrain at the site's coordinates while SIM_TERRAIN is 1 (the
# default; MAVProxy fetches the tiles): on the synthetic street it put the ground 2.7 m above
# home 17 m away. The world the oracles judge is the twin, so the simulator's ground is made
# flat at home's altitude, where the twin has it; L2 does not model the ground beyond that.
FLAT_GROUND = {"SIM_TERRAIN": 0.0}


def environment_params(env: Environment, fw: FirmwareParams) -> dict[str, float]:
    """Everything the runner sets on the simulator for a scenario's environment."""
    missing = fw.unknown(FLAT_GROUND)
    if missing:
        raise FaultUnavailable(f"environment: {fw.tag} has no {', '.join(missing)}")
    return {**FLAT_GROUND, **wind_params(env.wind, fw)}


def response(trigger: str, parm: Mapping[str, float], vehicle: Mapping[str, float]) -> Response:
    """What the package + vehicle make of a trigger (an abort reason, SIMULATION §4.6)."""
    opts = int(_value("FS_OPTIONS", parm, vehicle)) if ("FS_OPTIONS" in parm or "FS_OPTIONS" in vehicle) else 0
    if trigger == "fence_breach":
        if int(_value("FENCE_ENABLE", parm, vehicle)) != 1:
            return Response(trigger, (), ("FENCE_ENABLE",), note="the fence is not enabled")
        return Response(trigger, _action("FENCE_ACTION", parm, vehicle), ("FENCE_ACTION",))
    if trigger == "rc_loss":
        if opts & FS_OPT_CONTINUE_AUTO_RC:
            return Response(trigger, (), ("FS_OPTIONS",), note="FS_OPTIONS: continue in AUTO")
        delay = float(vehicle.get("RC_FS_TIMEOUT", 1.0))
        return Response(trigger, _action("FS_THR_ENABLE", parm, vehicle), ("FS_THR_ENABLE",),
                        delay_s=delay)
    if trigger == "gcs_loss":
        if opts & FS_OPT_CONTINUE_AUTO_GCS:
            return Response(trigger, (), ("FS_OPTIONS",), note="FS_OPTIONS: continue in AUTO")
        delay = _value("FS_GCS_TIMEOUT", parm, vehicle)
        return Response(trigger, _action("FS_GCS_ENABLE", parm, vehicle), ("FS_GCS_ENABLE",),
                        delay_s=delay)
    if trigger in ("battery_low", "battery_critical"):
        name = "BATT_FS_LOW_ACT" if trigger == "battery_low" else "BATT_FS_CRT_ACT"
        threshold = "BATT_LOW_VOLT" if trigger == "battery_low" else "BATT_CRT_VOLT"
        if _value(threshold, parm, vehicle) <= 0:
            raise FaultUnavailable(f"{trigger}: {threshold} = 0, the failsafe can never fire")
        return Response(trigger, _action(name, parm, vehicle), (name, threshold),
                        delay_s=_value("BATT_LOW_TIMER", parm, vehicle))
    if trigger == "gps_loss":
        # Two failsafes can answer. FS_DR_ENABLE acts when EKF3 reports dead reckoning, which it
        # does only with wind-relative navigation (drag / airspeed fusion); a copter without it
        # never reports it, and the EKF failsafe (FS_EKF_ACTION) acts once the position variance
        # passes FS_EKF_THRESH. Both are accepted; detection takes up to GPS_LOSS_DETECT_S.
        dr = _action("FS_DR_ENABLE", parm, vehicle) if "FS_DR_ENABLE" in parm or \
            "FS_DR_ENABLE" in vehicle else ()
        ekf = _action("FS_EKF_ACTION", parm, vehicle)
        modes = tuple(dict.fromkeys(dr + ekf))
        return Response(trigger, modes, ("FS_DR_ENABLE", "FS_EKF_ACTION"), delay_s=GPS_LOSS_DETECT_S,
                        note="the vehicle keeps flying on IMU alone until the EKF gives up",
                        extra={"dead_reckoning": dr, "ekf": ekf})
    raise ValueError(f"unknown trigger '{trigger}'")


def failsafe_trigger_problems(parm: Mapping[str, float], vehicle: Mapping[str, float]) -> list[str]:
    """Actions the package sets whose trigger is disabled on the vehicle: the package promises
    a reaction that can never happen. Reported, so that nobody believes in it."""
    out = []
    if parm.get("BATT_FS_CRT_ACT", 0) and not vehicle.get("BATT_CRT_VOLT", 0) \
            and not vehicle.get("BATT_CRT_MAH", 0):
        out.append("BATT_FS_CRT_ACT is set but BATT_CRT_VOLT and BATT_CRT_MAH are 0: the "
                   "critical battery failsafe cannot fire on this vehicle")
    if parm.get("BATT_FS_LOW_ACT", 0) and not vehicle.get("BATT_LOW_VOLT", 0) \
            and not vehicle.get("BATT_LOW_MAH", 0):
        out.append("BATT_FS_LOW_ACT is set but BATT_LOW_VOLT and BATT_LOW_MAH are 0")
    if parm.get("FS_GCS_ENABLE", 0) and "MAV_GCS_SYSID" in vehicle and \
            int(vehicle["MAV_GCS_SYSID"]) not in (RUNNER_SYSID, 255):
        out.append(f"FS_GCS_ENABLE is set; the GCS the vehicle listens to is sysid "
                   f"{int(vehicle['MAV_GCS_SYSID'])}")
    return out
