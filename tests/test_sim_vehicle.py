"""L0 tests of the pymavlink client: truth/estimate decoding, the truth monitor, and the command
and wait logic against a scripted fake link (no pymavlink, no SITL)."""

import math
from collections import deque
from types import SimpleNamespace

import pytest

import aerosplat.sim.vehicle as vehicle_mod
from aerosplat.sim.vehicle import (COPTER_MODES, MAV_MODE_FLAG_SAFETY_ARMED, MAV_TYPE_GCS, TruthMonitor,
                                   Vehicle, VehicleError, VehicleTimeout, estimate_from_global_position,
                                   horizontal_distance_m, truth_from_sim_state)


class Msg:
    def __init__(self, kind, src=(1, 1), **fields):
        self._kind, self._src = kind, src
        self.__dict__.update(fields)

    def get_type(self):
        return self._kind

    def get_srcSystem(self):
        return self._src[0]

    def get_srcComponent(self):
        return self._src[1]


def sim_state(lat_int=408910458, lon_int=-742163420, alt=233.1, vn=0.0, ve=0.0, vd=0.0, **kw):
    # ArduPilot fills the float lat/lon with degE7 values; the decoder must not use them.
    return Msg("SIM_STATE", lat=float(lat_int), lon=float(lon_int), lat_int=lat_int, lon_int=lon_int,
               alt=alt, vn=vn, ve=ve, vd=vd, **kw)


def gpi(lat=408910458, lon=-742163420, alt=233100, relative_alt=0, vx=0, vy=0, vz=0):
    return Msg("GLOBAL_POSITION_INT", lat=lat, lon=lon, alt=alt, relative_alt=relative_alt,
               vx=vx, vy=vy, vz=vz)


def heartbeat(armed=False, mode="STABILIZE", src=(1, 1), mav_type=2):
    return Msg("HEARTBEAT", src=src, type=mav_type, base_mode=MAV_MODE_FLAG_SAFETY_ARMED if armed else 0,
               custom_mode=COPTER_MODES[mode])


class FakeLink:
    """Stands in for a pymavlink connection; `on_command(cmd, params)` returns replies to queue."""

    def __init__(self, on_command=None):
        self.inbox = deque()
        self.sent_commands = []
        self.heartbeats_sent = 0
        self.on_command = on_command or (lambda cmd, params: [Msg("COMMAND_ACK", command=cmd, result=0)])
        self.mav = self

    def heartbeat_send(self, *args):
        self.heartbeats_sent += 1

    def command_long_send(self, target_system, target_component, cmd, confirmation, *params):
        self.sent_commands.append((cmd, confirmation, params))
        self.inbox.extend(self.on_command(cmd, params))

    def recv_match(self, blocking=False, timeout=None):
        return self.inbox.popleft() if self.inbox else None

    def close(self):
        pass


def fake_clock(monkeypatch, step=0.0, start=0.0):
    """Replace the module's clock: each reading advances it by `step` seconds."""
    now = [start]

    def monotonic():
        now[0] += step
        return now[0]

    monkeypatch.setattr(vehicle_mod, "time", SimpleNamespace(monotonic=monotonic))
    return now


def connected(link):
    link.inbox.append(heartbeat())
    v = Vehicle(link)
    v.wait_heartbeat(timeout=1)
    return v


@pytest.mark.hazard("H-004")
def test_truth_uses_integer_coordinates():
    t = truth_from_sim_state(sim_state(alt=243.5, vn=1.0, ve=-2.0, vd=0.5), t=7.0)
    assert (t.lat, t.lon) == pytest.approx((40.8910458, -74.216342), abs=1e-9)
    assert (t.alt_amsl, t.vn, t.ve, t.vd, t.t) == (243.5, 1.0, -2.0, 0.5, 7.0)
    assert t.speed == pytest.approx(math.sqrt(5.25))
    with pytest.raises(VehicleError):  # an old SIM_STATE without lat_int is not silently misread
        truth_from_sim_state(sim_state(lat_int=0, lon_int=0), t=0.0)


def test_estimate_units():
    e = estimate_from_global_position(gpi(alt=243100, relative_alt=10000, vx=150, vy=-50, vz=-25), t=1.0)
    assert (e.alt_amsl, e.rel_alt, e.vn, e.ve, e.vd) == (243.1, 10.0, 1.5, -0.5, -0.25)


def test_horizontal_distance():
    assert horizontal_distance_m(40.0, -74.0, 40.0, -74.0) == 0.0
    assert horizontal_distance_m(40.0, -74.0, 40.00001, -74.0) == pytest.approx(1.112, abs=0.002)
    assert horizontal_distance_m(40.0, -74.0, 40.0, -73.99999) == pytest.approx(0.852, abs=0.002)


def test_truth_monitor_pairs_estimate_with_fresh_truth(monkeypatch):
    now = fake_clock(monkeypatch, start=100.0)
    m = TruthMonitor(ground_alt=233.1, start_lat=40.8910458, start_lon=-74.216342)
    m(sim_state(alt=243.1))                                   # truth: 10 m above ground
    m(gpi(relative_alt=10400, lat=408910458 + 90))            # estimate: 0.4 m high, ~1 m north
    assert m.samples == 1
    assert m.max_vertical_error == pytest.approx(0.4, abs=1e-6)
    assert m.max_horizontal_error == pytest.approx(1.0, abs=0.01)
    assert m.max_truth_agl == pytest.approx(10.0, abs=1e-4)
    now[0] += 0.5                                             # truth too old: no pairing
    m(gpi(relative_alt=0))
    assert m.samples == 1


def test_command_ack_and_mode():
    link = FakeLink()
    v = connected(link)
    assert (v.target_system, v.target_component) == (1, 1)
    assert link.heartbeats_sent == 1                          # the GCS heartbeat goes out first

    def on_command(cmd, params):
        mode = next(k for k, val in COPTER_MODES.items() if val == params[1])
        return [Msg("COMMAND_ACK", command=cmd, result=0), heartbeat(mode=mode)]

    link.on_command = on_command
    v.set_mode("GUIDED", timeout=1)
    assert v.mode == "GUIDED" and not v.armed
    assert link.sent_commands[-1][0] == 176

    link.on_command = lambda cmd, params: [Msg("COMMAND_ACK", command=cmd, result=4)]   # FAILED
    with pytest.raises(VehicleError):
        v.command_ok(400, 1, timeout=1)


def test_command_without_ack_times_out_after_resends(monkeypatch):
    fake_clock(monkeypatch, step=0.05)
    link = FakeLink(on_command=lambda cmd, params: [])
    v = connected(link)
    with pytest.raises(VehicleTimeout):
        v.command(511, 108, 1e5, timeout=3)
    confirmations = [c for cmd, c, _ in link.sent_commands if cmd == 511]
    assert len(confirmations) >= 2 and confirmations[:2] == [0, 1]   # resent with confirmation+1


def test_other_systems_and_components_are_ignored():
    link = FakeLink()
    v = connected(link)
    link.inbox.extend([
        heartbeat(armed=True, src=(2, 1)),                 # another vehicle
        heartbeat(armed=True, src=(1, 154)),               # a gimbal on ours
        heartbeat(src=(255, 190), mav_type=MAV_TYPE_GCS),  # a GCS
        sim_state(src=(2, 1), alt=999.0),
    ])
    v.pump(0)
    assert not v.armed
    with pytest.raises(VehicleError):                      # no truth from our vehicle yet
        v.truth()
    link.inbox.append(sim_state())
    v.pump(0)
    assert v.truth().alt_amsl == pytest.approx(233.1)


def test_wait_reports_statustext_on_timeout(monkeypatch):
    fake_clock(monkeypatch, step=0.1)
    link = FakeLink()
    v = connected(link)
    link.inbox.append(Msg("STATUSTEXT", text="PreArm: Need Position Estimate"))
    with pytest.raises(VehicleTimeout, match="Need Position Estimate"):
        v.wait_prearm_ok(timeout=2)
