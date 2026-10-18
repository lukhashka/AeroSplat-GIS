"""L0 tests of the mission protocol and the parameter protocol against a scripted autopilot
(no pymavlink, no SITL). The fake stores items the way ArduPilot does (Copter-4.7.2, seen in
SITL): DO / CONDITION items come back in frame 0 without a position, NAV_LAND param4 by sign."""

from collections import deque
from types import SimpleNamespace

import pytest

import mission_scenes as ms
import sim_scenarios as ss
from aerosplat.mission.export import to_wire
from aerosplat.sim import mavmission
from aerosplat.sim.mavmission import (MISSION_TYPE_FENCE, MISSION_TYPE_MISSION, MISSION_TYPE_RALLY,
                                      Item, MissionProtocolError, compare, fence_items, rally_items,
                                      upload_verified)
from aerosplat.sim.vehicle import Vehicle, VehicleError
import aerosplat.sim.vehicle as vehicle_mod


class Msg(SimpleNamespace):
    def __init__(self, kind, **fields):
        super().__init__(**fields)
        self._kind = kind

    def get_type(self):
        return self._kind

    def get_srcSystem(self):
        return 1

    def get_srcComponent(self):
        return 1


class FakeAutopilot:
    """A link whose other end answers like ArduPilot; `corrupt(item) -> item` alters storage."""

    def __init__(self, params=None, corrupt=None, clamp=None):
        self.inbox = deque([Msg("HEARTBEAT", type=2, base_mode=0, custom_mode=0)])
        self.mav = self
        self.store: dict[int, list[Item]] = {}
        self.upload: dict = {}
        self.params = dict(params or {})
        self.corrupt = corrupt or (lambda it: it)
        self.clamp = clamp or {}
        self.heartbeats = 0

    # link
    def recv_match(self, blocking=False, timeout=None):
        return self.inbox.popleft() if self.inbox else None

    def close(self):
        pass

    def heartbeat_send(self, *a):
        self.heartbeats += 1

    # mission protocol
    def mission_count_send(self, ts, tc, count, mtype):
        self.upload = {"type": mtype, "count": count, "items": []}
        if count == 0:
            self.store[mtype] = []
            self.inbox.append(Msg("MISSION_ACK", type=0, mission_type=mtype))
        else:
            self.inbox.append(Msg("MISSION_REQUEST_INT", seq=0, mission_type=mtype))

    def mission_item_int_send(self, ts, tc, seq, frame, cmd, cur, auto, p1, p2, p3, p4, x, y, z, mtype):
        it = Item(seq=seq, frame=frame, command=cmd, params=(p1, p2, p3, p4), x=x, y=y, z=z,
                  current=cur, autocontinue=auto)
        if cmd not in (16, 21, 22, 5001, 5100):         # ArduPilot keeps no frame for these
            it = Item(seq=seq, frame=0, command=cmd, params=it.params, x=0, y=0, z=0.0)
        if cmd == 21:
            it = Item(seq=seq, frame=frame, command=cmd, params=(p1, p2, p3, -1.0 if p4 < 0 else 1.0),
                      x=x, y=y, z=z)
        self.upload["items"].append(self.corrupt(it))
        if seq + 1 < self.upload["count"]:
            self.inbox.append(Msg("MISSION_REQUEST_INT", seq=seq + 1, mission_type=mtype))
        else:
            self.store[mtype] = self.upload["items"]
            self.inbox.append(Msg("MISSION_ACK", type=0, mission_type=mtype))

    def mission_request_list_send(self, ts, tc, mtype):
        self.inbox.append(Msg("MISSION_COUNT", count=len(self.store.get(mtype, [])), mission_type=mtype))

    def mission_request_int_send(self, ts, tc, seq, mtype):
        it = self.store[mtype][seq]
        self.inbox.append(Msg("MISSION_ITEM_INT", seq=seq, frame=it.frame, command=it.command,
                              param1=it.params[0], param2=it.params[1], param3=it.params[2],
                              param4=it.params[3], x=it.x, y=it.y, z=it.z, current=0,
                              autocontinue=1, mission_type=mtype))

    def mission_ack_send(self, *a):
        pass

    # parameter protocol
    def _value(self, name, index):
        return Msg("PARAM_VALUE", param_id=name, param_value=self.params[name],
                   param_type=9, param_count=len(self.params), param_index=index)

    def param_set_send(self, ts, tc, name, value, ptype):
        name = name.decode()
        if name in self.params:                         # unknown names: silence, like ArduPilot
            self.params[name] = self.clamp.get(name, value)
            self.inbox.append(self._value(name, list(self.params).index(name)))

    def param_request_read_send(self, ts, tc, name, index):
        name = name.decode() if name else list(self.params)[index]
        if name in self.params:
            self.inbox.append(self._value(name, list(self.params).index(name)))

    def param_request_list_send(self, ts, tc):
        for k, name in enumerate(self.params):
            self.inbox.append(self._value(name, k))


def connected(link) -> Vehicle:
    v = Vehicle(link)
    v.wait_heartbeat(timeout=2)
    return v


@pytest.fixture(scope="module")
def items(tmp_path_factory):
    plan = ss.window_plan(tmp_path_factory.mktemp("p"))
    return [Item.from_wire(i) for i in to_wire(plan, ms.ORIGIN)]


# ── mission ──────────────────────────────────────────────────────────────
@pytest.mark.hazard("H-002")
def test_mission_round_trip_up_to_the_firmwares_storage_rules(items):
    link = FakeAutopilot()
    v = connected(link)
    held = upload_verified(v, items, MISSION_TYPE_MISSION)
    assert len(held) == len(items)
    assert {it.frame for it in held if it.command in (178, 115)} == {0}   # normalised, accepted
    assert held[0].frame == 0                                  # home: absolute
    assert [it.frame for it in held[1:] if it.command in (16, 21, 22)] == [3] * 6


@pytest.mark.hazard("H-002")
@pytest.mark.parametrize("corrupt, match", [
    (lambda it: Item(**{**it.__dict__, "z": it.z + 0.5}) if it.seq == 6 else it, "altitude"),
    (lambda it: Item(**{**it.__dict__, "frame": 0}) if it.seq == 3 else it, "frame/command"),
    (lambda it: Item(**{**it.__dict__, "x": it.x + 50}) if it.seq == 3 else it, "position"),
    (lambda it: Item(**{**it.__dict__, "params": (3.0, 0, 0, 0)}) if it.seq == 6 else it, "param1"),
])
def test_what_the_vehicle_holds_must_be_what_was_sent(items, corrupt, match):
    v = connected(FakeAutopilot(corrupt=corrupt))
    with pytest.raises(MissionProtocolError, match=match):
        upload_verified(v, items, MISSION_TYPE_MISSION)


def test_fence_and_rally_round_trip():
    v = connected(FakeAutopilot())
    fence = fence_items([(50.45, 30.52), (50.451, 30.52), (50.451, 30.521)])
    assert all(it.params[0] == 3 for it in fence)
    assert len(upload_verified(v, fence, MISSION_TYPE_FENCE)) == 3
    assert len(upload_verified(v, rally_items([(50.45, 30.52, 9.0)]), MISSION_TYPE_RALLY)) == 1
    with pytest.raises(MissionProtocolError, match="3 or more"):
        fence_items([(50.45, 30.52), (50.451, 30.52)])


def test_compare_counts_and_home():
    a = [Item(seq=0, frame=0, command=16, params=(0, 0, 0, 0), x=1, y=1, z=100.0)]
    b = [Item(seq=0, frame=0, command=16, params=(0, 0, 0, 0), x=9, y=9, z=150.0)]
    assert compare(a, b, "m", skip_home=True) == []          # ArduPilot keeps its own home
    assert compare(a, b + b, "m") == ["m: 1 items sent, the vehicle holds 2"]


# ── parameters ───────────────────────────────────────────────────────────
@pytest.mark.hazard("H-003")
def test_parameter_set_is_confirmed_and_unknown_names_fail_loudly(monkeypatch):
    v = connected(FakeAutopilot({"RTL_ALT_M": 15.0, "FENCE_ACTION": 1.0, "WP_SPD": 10.0},
                                clamp={"WP_SPD": 20.0}))
    assert v.set_param("RTL_ALT_M", 9.0) == 9.0
    assert v.get_param("RTL_ALT_M") == 9.0
    clock = [0.0]
    monkeypatch.setattr(vehicle_mod, "time", SimpleNamespace(monotonic=lambda: clock.__setitem__(
        0, clock[0] + 0.05) or clock[0]))
    with pytest.raises(VehicleError, match="unknown to this firmware"):
        v.set_param("RTL_ALT", 900.0, timeout=0.6)              # the 4.5/4.6 name: silence
    with pytest.raises(VehicleError, match="holds 20"):
        v.set_param("WP_SPD", 30.0)
    with pytest.raises(VehicleError, match="longer than 16"):
        v.set_param("A_VERY_LONG_PARAMETER", 1.0)


def test_fetch_params_reads_the_whole_table():
    link = FakeAutopilot({f"P{k}": float(k) for k in range(30)})
    v = connected(link)
    table = v.fetch_params(timeout=5)
    assert len(table) == 30 and table["P7"] == 7.0 and v.param_count == 30


def test_gcs_heartbeats_can_be_switched_off(monkeypatch):
    link = FakeAutopilot()
    v = connected(link)
    sent = link.heartbeats
    v.send_heartbeats = False
    v._last_heartbeat_sent = -1e9
    v.pump(0)
    assert link.heartbeats == sent                              # the gcs_loss fault
    v.send_heartbeats = True
    v.pump(0)
    assert link.heartbeats == sent + 1


def test_simulation_clock_follows_the_autopilot(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(vehicle_mod, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    link = FakeAutopilot()
    v = connected(link)
    with pytest.raises(VehicleError, match="simulation clock"):
        v.sim_now()
    link.inbox.append(Msg("GLOBAL_POSITION_INT", time_boot_ms=40_000, lat=0, lon=0, alt=0,
                          relative_alt=0, vx=0, vy=0, vz=0))
    v.pump(0)
    clock[0] += 0.5
    assert v.sim_now(speedup=3) == pytest.approx(41.5)
    link.inbox.append(Msg("GLOBAL_POSITION_INT", time_boot_ms=39_900, lat=0, lon=0, alt=0,
                          relative_alt=0, vx=0, vy=0, vz=0))       # reordered: ignored
    v.pump(0)
    assert v.boot_ms == 40_000
    link.inbox.append(Msg("SYSTEM_TIME", time_boot_ms=1_000))  # reboot: the clock starts over
    v.pump(0)
    assert v.boot_ms == 1_000 and mavmission.MISSION_TYPES[0] == "mission"
