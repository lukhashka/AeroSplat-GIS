"""pymavlink client for SITL tests: GCS heartbeats, a message cache, commands and timed waits.

Truth and estimate are kept apart (§4.1). `truth()` is the simulator's own state (SIM_STATE);
`estimate()` is what the autopilot believes (GLOBAL_POSITION_INT, i.e. EKF3). Oracles judge
`truth()`; comparing the two is how a confidently wrong EKF becomes visible.

    with Vehicle.connect(Sitl(load_config()).endpoint()) as v:
        v.stream_defaults()
        v.wait_prearm_ok()
        v.set_mode("GUIDED"); v.arm(); v.takeoff(10)

Every wait has a wall-clock timeout and fails with the vehicle's last STATUSTEXT lines.
pymavlink is imported lazily (optional extra `aerosplat[sim]`).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")

# MAVLink numbers used below (common.xml / ardupilotmega.xml), numeric so that importing this
# module does not need pymavlink.
MSG_SYS_STATUS = 1
MSG_GLOBAL_POSITION_INT = 33
MSG_SIM_STATE = 108
MSG_AUTOPILOT_VERSION = 148
MSG_EXTENDED_SYS_STATE = 245

CMD_NAV_TAKEOFF = 22
CMD_DO_SET_MODE = 176
CMD_COMPONENT_ARM_DISARM = 400
CMD_SET_MESSAGE_INTERVAL = 511
CMD_REQUEST_MESSAGE = 512

MAV_RESULT_ACCEPTED = 0
MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
MAV_MODE_FLAG_SAFETY_ARMED = 128
MAV_SYS_STATUS_PREARM_CHECK = 0x10000000
MAV_TYPE_GCS = 6
MAV_AUTOPILOT_INVALID = 8
MAV_STATE_ACTIVE = 4
FORCE_DISARM_MAGIC = 21196

COPTER_MODES = {"STABILIZE": 0, "ALT_HOLD": 2, "AUTO": 3, "GUIDED": 4, "LOITER": 5, "RTL": 6,
                "LAND": 9, "BRAKE": 17}

EARTH_RADIUS_M = 6371008.8  # IUGG mean radius; plenty for metre-scale distances


class VehicleError(RuntimeError):
    pass


class VehicleTimeout(VehicleError):
    pass


@dataclass(frozen=True)
class GeoState:
    """Position (WGS84, metres AMSL) and NED velocity (m/s); `t` = host monotonic receive time."""
    lat: float
    lon: float
    alt_amsl: float
    vn: float
    ve: float
    vd: float
    t: float
    rel_alt: Optional[float] = None  # estimate only: height above home

    @property
    def speed(self) -> float:
        return math.sqrt(self.vn ** 2 + self.ve ** 2 + self.vd ** 2)


@dataclass(frozen=True)
class Firmware:
    version: tuple[int, int, int]
    release_type: int     # FIRMWARE_VERSION_TYPE: 255 = official
    git_hash: str         # first 8 hex digits of the commit the binary was built from


def horizontal_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Equirectangular distance; error < 1 mm at the scale of a site."""
    phi = math.radians((lat1 + lat2) / 2)
    dx = math.radians(lon2 - lon1) * math.cos(phi)
    dy = math.radians(lat2 - lat1)
    return EARTH_RADIUS_M * math.hypot(dx, dy)


def truth_from_sim_state(msg: Any, t: float) -> GeoState:
    """SIM_STATE -> truth. Uses lat_int/lon_int: ArduPilot puts degE7 into the float lat/lon."""
    lat_int = getattr(msg, "lat_int", 0)
    lon_int = getattr(msg, "lon_int", 0)
    if lat_int == 0 and lon_int == 0:
        raise VehicleError("SIM_STATE without lat_int/lon_int (firmware or pymavlink too old)")
    return GeoState(lat=lat_int * 1e-7, lon=lon_int * 1e-7, alt_amsl=float(msg.alt),
                    vn=float(msg.vn), ve=float(msg.ve), vd=float(msg.vd), t=t)


def estimate_from_global_position(msg: Any, t: float) -> GeoState:
    """GLOBAL_POSITION_INT -> EKF estimate."""
    return GeoState(lat=msg.lat * 1e-7, lon=msg.lon * 1e-7, alt_amsl=msg.alt / 1000.0,
                    vn=msg.vx / 100.0, ve=msg.vy / 100.0, vd=msg.vz / 100.0, t=t,
                    rel_alt=msg.relative_alt / 1000.0)


@dataclass
class TruthMonitor:
    """Listener that pairs each estimate with the latest truth no older than `max_skew` seconds.

    Tracks the truth height envelope above `ground_alt` (true AMSL of the vehicle at rest before
    take-off) and the worst estimate-vs-truth disagreement. The vertical comparison uses the
    estimate's height above home against the true height above ground.
    """
    ground_alt: float
    start_lat: float
    start_lon: float
    max_skew: float = 0.15
    samples: int = 0
    max_vertical_error: float = 0.0
    max_horizontal_error: float = 0.0
    max_truth_agl: float = -math.inf
    max_truth_drift: float = 0.0   # horizontal distance of truth from the start point
    _truth: Optional[GeoState] = field(default=None, repr=False)

    def __call__(self, msg: Any) -> None:
        now = time.monotonic()
        kind = msg.get_type()
        if kind == "SIM_STATE":
            tr = truth_from_sim_state(msg, now)
            self._truth = tr
            self.max_truth_agl = max(self.max_truth_agl, tr.alt_amsl - self.ground_alt)
            self.max_truth_drift = max(self.max_truth_drift, horizontal_distance_m(
                self.start_lat, self.start_lon, tr.lat, tr.lon))
        elif kind == "GLOBAL_POSITION_INT" and self._truth is not None \
                and now - self._truth.t <= self.max_skew:
            est = estimate_from_global_position(msg, now)
            tr = self._truth
            assert est.rel_alt is not None
            self.max_vertical_error = max(self.max_vertical_error,
                                          abs(est.rel_alt - (tr.alt_amsl - self.ground_alt)))
            self.max_horizontal_error = max(self.max_horizontal_error,
                                            horizontal_distance_m(est.lat, est.lon, tr.lat, tr.lon))
            self.samples += 1


class Vehicle:
    """One ArduCopter over MAVLink. Not thread-safe: every wait pumps the link itself."""

    def __init__(self, conn: Any):
        self.conn = conn
        self.target_system = 0
        self.target_component = 0
        self.latest: dict[str, Any] = {}
        self.received_at: dict[str, float] = {}
        self.statustext: list[str] = []
        self.listeners: list[Callable[[Any], None]] = []
        self._acks: dict[int, tuple[float, Any]] = {}
        self._last_heartbeat_sent = -math.inf

    @classmethod
    def connect(cls, endpoint: str, timeout: float = 30.0, source_system: int = 245,
                source_component: int = 190) -> "Vehicle":
        """Open `endpoint` (pymavlink syntax) and wait for the autopilot's heartbeat."""
        try:
            from pymavlink import mavutil
        except ImportError as e:
            raise ImportError("pymavlink is not installed: pip install -e .[sim]") from e
        conn = mavutil.mavlink_connection(endpoint, source_system=source_system,
                                          source_component=source_component, dialect="ardupilotmega")
        v = cls(conn)
        try:
            v.wait_heartbeat(timeout)
        except BaseException:
            v.close()
            raise
        return v

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Vehicle":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── link ─────────────────────────────────────────────────────────────
    def pump(self, timeout: float = 0.1) -> int:
        """Send a GCS heartbeat when due (1 Hz), then read everything pending, waiting up to
        `timeout` for the first message. Returns the number of messages handled."""
        now = time.monotonic()
        if now - self._last_heartbeat_sent >= 1.0:
            self.conn.mav.heartbeat_send(MAV_TYPE_GCS, MAV_AUTOPILOT_INVALID, 0, 0, MAV_STATE_ACTIVE)
            self._last_heartbeat_sent = now
        n = 0
        msg = self.conn.recv_match(blocking=True, timeout=timeout)
        while msg is not None:
            self._dispatch(msg)
            n += 1
            msg = self.conn.recv_match(blocking=False)
        return n

    def _dispatch(self, msg: Any) -> None:
        kind = msg.get_type()
        if kind == "BAD_DATA":
            return
        if self.target_system == 0:
            if kind != "HEARTBEAT" or msg.type == MAV_TYPE_GCS:
                return
            self.target_system, self.target_component = msg.get_srcSystem(), msg.get_srcComponent()
        if msg.get_srcSystem() != self.target_system:
            return
        if kind == "HEARTBEAT" and msg.get_srcComponent() != self.target_component:
            return  # a camera or gimbal heartbeat would otherwise overwrite the autopilot's
        now = time.monotonic()
        self.latest[kind] = msg
        self.received_at[kind] = now
        if kind == "STATUSTEXT":
            self.statustext.append(msg.text)
        elif kind == "COMMAND_ACK":
            self._acks[msg.command] = (now, msg)
        for fn in self.listeners:
            fn(msg)

    def wait(self, what: str, predicate: Callable[[], bool], timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                raise VehicleTimeout(f"timed out after {timeout:.0f} s waiting for {what}; "
                                     f"last STATUSTEXT: {self.statustext[-6:]}")
            self.pump(0.1)

    def sample(self, fn: Callable[[], T], seconds: float) -> list[T]:
        """fn() after every pump for `seconds` of wall-clock time."""
        out: list[T] = []
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.pump(0.1)
            out.append(fn())
        return out

    def wait_heartbeat(self, timeout: float = 30.0) -> Any:
        self.wait("autopilot HEARTBEAT", lambda: "HEARTBEAT" in self.latest, timeout)
        return self.latest["HEARTBEAT"]

    def fresh(self, kind: str, max_age: float = 1.0) -> Optional[Any]:
        t = self.received_at.get(kind)
        return self.latest[kind] if t is not None and time.monotonic() - t <= max_age else None

    def wait_message(self, kind: str, timeout: float = 10.0, since: Optional[float] = None) -> Any:
        """The latest `kind` received after `since` (default: now)."""
        start = time.monotonic() if since is None else since
        self.wait(kind, lambda: self.received_at.get(kind, -math.inf) >= start, timeout)
        return self.latest[kind]

    # ── commands ─────────────────────────────────────────────────────────
    def command(self, cmd: int, *params: float, timeout: float = 5.0) -> int:
        """COMMAND_LONG, resent every second until acknowledged; returns the MAV_RESULT."""
        if len(params) > 7:
            raise ValueError("COMMAND_LONG takes at most 7 parameters")
        p = [float(x) for x in params] + [0.0] * (7 - len(params))
        sent = time.monotonic()
        deadline = sent + timeout
        confirmation = 0
        while time.monotonic() < deadline:
            self.conn.mav.command_long_send(self.target_system, self.target_component, cmd,
                                            confirmation, *p)
            resend_at = min(time.monotonic() + 1.0, deadline)
            while time.monotonic() < resend_at:
                ack = self._acks.get(cmd)
                if ack is not None and ack[0] >= sent:
                    return int(ack[1].result)
                self.pump(0.1)
            confirmation = min(confirmation + 1, 255)
        raise VehicleTimeout(f"no COMMAND_ACK for command {cmd} in {timeout:.0f} s")

    def command_ok(self, cmd: int, *params: float, timeout: float = 5.0) -> None:
        result = self.command(cmd, *params, timeout=timeout)
        if result != MAV_RESULT_ACCEPTED:
            raise VehicleError(f"command {cmd}{tuple(params)} rejected: MAV_RESULT {result}; "
                               f"last STATUSTEXT: {self.statustext[-6:]}")

    def set_message_interval(self, msg_id: int, hz: float) -> None:
        self.command_ok(CMD_SET_MESSAGE_INTERVAL, msg_id, 1e6 / hz)

    def request_message(self, msg_id: int, kind: str, timeout: float = 5.0) -> Any:
        start = time.monotonic()
        self.command_ok(CMD_REQUEST_MESSAGE, msg_id)
        return self.wait_message(kind, timeout, since=start)

    def stream_defaults(self, hz: float = 10.0) -> None:
        """Truth and estimate at `hz`, health and landed state at 2 Hz (this link only)."""
        self.set_message_interval(MSG_SIM_STATE, hz)
        self.set_message_interval(MSG_GLOBAL_POSITION_INT, hz)
        self.set_message_interval(MSG_SYS_STATUS, 2.0)
        self.set_message_interval(MSG_EXTENDED_SYS_STATE, 2.0)

    # ── state ────────────────────────────────────────────────────────────
    @property
    def armed(self) -> bool:
        hb = self.latest.get("HEARTBEAT")
        return bool(hb is not None and hb.base_mode & MAV_MODE_FLAG_SAFETY_ARMED)

    @property
    def mode(self) -> Optional[str]:
        hb = self.latest.get("HEARTBEAT")
        if hb is None:
            return None
        return next((k for k, v in COPTER_MODES.items() if v == hb.custom_mode), str(hb.custom_mode))

    def truth(self, max_age: float = 1.0) -> GeoState:
        msg = self.fresh("SIM_STATE", max_age)
        if msg is None:
            raise VehicleError(f"no SIM_STATE (truth) in the last {max_age} s: not a SITL vehicle, "
                               f"or the stream was not requested (stream_defaults)")
        return truth_from_sim_state(msg, self.received_at["SIM_STATE"])

    def estimate(self, max_age: float = 1.0) -> GeoState:
        msg = self.fresh("GLOBAL_POSITION_INT", max_age)
        if msg is None:
            raise VehicleError(f"no GLOBAL_POSITION_INT in the last {max_age} s")
        return estimate_from_global_position(msg, self.received_at["GLOBAL_POSITION_INT"])

    def wait_truth(self, timeout: float = 10.0) -> GeoState:
        self.wait("SIM_STATE (truth)", lambda: self.fresh("SIM_STATE") is not None, timeout)
        return self.truth()

    def firmware(self) -> Firmware:
        av = self.request_message(MSG_AUTOPILOT_VERSION, "AUTOPILOT_VERSION")
        v = int(av.flight_sw_version)
        git_hash = bytes(av.flight_custom_version).rstrip(b"\x00").decode("ascii", "replace")
        return Firmware(((v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF), v & 0xFF, git_hash)

    # ── actions ──────────────────────────────────────────────────────────
    def wait_prearm_ok(self, timeout: float = 120.0) -> None:
        """Until SYS_STATUS reports the pre-arm checks healthy (EKF, GPS, ... settled)."""
        def ok() -> bool:
            s = self.fresh("SYS_STATUS", 3.0)
            return s is not None and bool(s.onboard_control_sensors_health & MAV_SYS_STATUS_PREARM_CHECK)
        self.wait("pre-arm checks to pass", ok, timeout)

    def set_mode(self, name: str, timeout: float = 10.0) -> None:
        self.command_ok(CMD_DO_SET_MODE, MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, COPTER_MODES[name])
        self.wait(f"mode {name}", lambda: self.mode == name, timeout)

    def arm(self, timeout: float = 30.0) -> None:
        """Arm, retrying while pre-arm checks flap (EKF still settling in SITL)."""
        deadline = time.monotonic() + timeout
        while not self.armed:
            if time.monotonic() > deadline:
                raise VehicleTimeout(f"not armed after {timeout:.0f} s; "
                                     f"last STATUSTEXT: {self.statustext[-6:]}")
            self.command(CMD_COMPONENT_ARM_DISARM, 1)
            try:
                self.wait("armed", lambda: self.armed, 2.0)
            except VehicleTimeout:
                pass

    def disarm(self, force: bool = False, timeout: float = 10.0) -> None:
        self.command_ok(CMD_COMPONENT_ARM_DISARM, 0, FORCE_DISARM_MAGIC if force else 0)
        self.wait("disarmed", lambda: not self.armed, timeout)

    def takeoff(self, alt_m: float) -> None:
        """GUIDED take-off to `alt_m` above home."""
        self.command_ok(CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, alt_m)
