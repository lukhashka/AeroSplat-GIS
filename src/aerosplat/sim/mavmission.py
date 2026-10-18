"""MAVLink mission protocol (upload and download) for the mission, the geofence and rally points.

    MISSION_COUNT -> MISSION_REQUEST_INT(seq) ... -> MISSION_ITEM_INT(seq) ... -> MISSION_ACK

Items travel as MISSION_ITEM_INT (lat/lon in 1e-7 degrees, the same resolution the WPL files
carry). Everything that is uploaded is downloaded again and compared item by item, frame and
command included: what the autopilot will fly is what was checked (SIMULATION §4.6, altitude
frames). The home row (seq 0) is the exception: ArduPilot keeps its own home there.

pymavlink is not imported here; `Vehicle` owns the link.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Optional

from aerosplat.mission.export import (MAV_FRAME_GLOBAL, MAV_FRAME_GLOBAL_RELATIVE_ALT,
                                      POSITIONAL as POSITIONAL_COMMANDS, WireItem)
from aerosplat.sim.vehicle import Vehicle, VehicleError, VehicleTimeout

MISSION_TYPE_MISSION = 0
MISSION_TYPE_FENCE = 1
MISSION_TYPE_RALLY = 2
MISSION_TYPES = {MISSION_TYPE_MISSION: "mission", MISSION_TYPE_FENCE: "fence",
                 MISSION_TYPE_RALLY: "rally"}
MAV_MISSION_ACCEPTED = 0
CMD_FENCE_POLYGON_VERTEX_INCLUSION = 5001
CMD_RALLY_POINT = 5100
CMD_NAV_LAND = 21
# MISSION_ITEM_INT keeps 1e-7 degrees (~1.1 cm); altitudes are float32.
LATLON_TOL_E7 = 1
ALT_TOL_M = 0.01


class MissionProtocolError(VehicleError):
    pass


@dataclass(frozen=True)
class Item:
    """One item as the protocol carries it: integer lat/lon (degE7), float altitude."""

    seq: int
    frame: int
    command: int
    params: tuple[float, float, float, float]
    x: int
    y: int
    z: float
    current: int = 0
    autocontinue: int = 1

    @classmethod
    def from_wire(cls, it: WireItem) -> "Item":
        return cls(seq=it.seq, frame=it.frame, command=it.command, params=it.params,
                   x=int(round(it.lat * 1e7)), y=int(round(it.lon * 1e7)), z=float(it.alt),
                   current=it.current, autocontinue=it.autocontinue)

    @property
    def lat(self) -> float:
        return self.x * 1e-7

    @property
    def lon(self) -> float:
        return self.y * 1e-7


def fence_items(polygon_latlon: list[tuple[float, float]]) -> list[Item]:
    """One inclusion polygon as FENCE_POLYGON_VERTEX_INCLUSION items (param1 = vertex count)."""
    n = len(polygon_latlon)
    if n < 3:
        raise MissionProtocolError(f"an inclusion polygon needs 3 or more vertices, got {n}")
    return [Item(seq=k, frame=MAV_FRAME_GLOBAL, command=CMD_FENCE_POLYGON_VERTEX_INCLUSION,
                 params=(float(n), 0.0, 0.0, 0.0), x=int(round(lat * 1e7)),
                 y=int(round(lon * 1e7)), z=0.0) for k, (lat, lon) in enumerate(polygon_latlon)]


def rally_items(points: list[tuple[float, float, float]]) -> list[Item]:
    """Rally points (lat, lon, altitude above home)."""
    return [Item(seq=k, frame=MAV_FRAME_GLOBAL_RELATIVE_ALT, command=CMD_RALLY_POINT,
                 params=(0.0, 0.0, 0.0, 0.0), x=int(round(lat * 1e7)), y=int(round(lon * 1e7)),
                 z=float(alt)) for k, (lat, lon, alt) in enumerate(points)]


def _send_item(v: Vehicle, it: Item, mission_type: int) -> None:
    p1, p2, p3, p4 = (float(p) for p in it.params)
    v.conn.mav.mission_item_int_send(v.target_system, v.target_component, it.seq, it.frame,
                                     it.command, it.current, it.autocontinue, p1, p2, p3, p4,
                                     it.x, it.y, float(it.z), mission_type)


def _latest_after(v: Vehicle, kind: str, since: float) -> Optional[Any]:
    return v.latest[kind] if v.received_at.get(kind, -math.inf) >= since else None


def upload(v: Vehicle, items: list[Item], mission_type: int, timeout: float = 30.0) -> None:
    """Send `items` (seq 0..n-1) and wait for MISSION_ACK accepted. An empty list clears."""
    name = MISSION_TYPES[mission_type]
    for k, it in enumerate(items):
        if it.seq != k:
            raise MissionProtocolError(f"{name} item {k} has seq {it.seq}")
    deadline = time.monotonic() + timeout
    sent_count = time.monotonic()
    v.conn.mav.mission_count_send(v.target_system, v.target_component, len(items), mission_type)
    last_req = time.monotonic()
    while True:
        if time.monotonic() > deadline:
            raise VehicleTimeout(f"{name} upload: no MISSION_ACK in {timeout:.0f} s; "
                                 f"last STATUSTEXT: {v.statustext[-6:]}")
        v.pump(0.05)
        ack = _latest_after(v, "MISSION_ACK", sent_count)
        if ack is not None and int(getattr(ack, "mission_type", 0)) == mission_type:
            if int(ack.type) != MAV_MISSION_ACCEPTED:
                raise MissionProtocolError(f"{name} upload rejected: MAV_MISSION_RESULT {ack.type}; "
                                           f"last STATUSTEXT: {v.statustext[-6:]}")
            return
        for kind in ("MISSION_REQUEST_INT", "MISSION_REQUEST"):
            req = _latest_after(v, kind, last_req)
            if req is not None and int(getattr(req, "mission_type", 0)) == mission_type:
                if not 0 <= int(req.seq) < len(items):
                    raise MissionProtocolError(f"{name} upload: vehicle asked for seq {req.seq} "
                                               f"of {len(items)}")
                _send_item(v, items[int(req.seq)], mission_type)
                last_req = time.monotonic()
                v.received_at[kind] = -math.inf          # consume it
        if time.monotonic() - last_req > 2.0:             # lost MISSION_COUNT: start again
            v.conn.mav.mission_count_send(v.target_system, v.target_component, len(items),
                                          mission_type)
            last_req = time.monotonic()


def download(v: Vehicle, mission_type: int, timeout: float = 30.0) -> list[Item]:
    name = MISSION_TYPES[mission_type]
    deadline = time.monotonic() + timeout
    start = time.monotonic()
    v.conn.mav.mission_request_list_send(v.target_system, v.target_component, mission_type)
    count = None
    while count is None:
        if time.monotonic() > deadline:
            raise VehicleTimeout(f"{name} download: no MISSION_COUNT in {timeout:.0f} s")
        v.pump(0.05)
        msg = _latest_after(v, "MISSION_COUNT", start)
        if msg is not None and int(getattr(msg, "mission_type", 0)) == mission_type:
            count = int(msg.count)
        elif time.monotonic() - start > 2.0:
            start = time.monotonic()
            v.conn.mav.mission_request_list_send(v.target_system, v.target_component, mission_type)
    out: list[Item] = []
    while len(out) < count:
        seq = len(out)
        asked = time.monotonic()
        v.conn.mav.mission_request_int_send(v.target_system, v.target_component, seq, mission_type)
        got = None
        while got is None:
            if time.monotonic() > deadline:
                raise VehicleTimeout(f"{name} download: item {seq} of {count} missing")
            v.pump(0.05)
            msg = _latest_after(v, "MISSION_ITEM_INT", asked)
            if msg is not None and int(msg.seq) == seq \
                    and int(getattr(msg, "mission_type", 0)) == mission_type:
                got = msg
            elif time.monotonic() - asked > 1.0:
                asked = time.monotonic()
                v.conn.mav.mission_request_int_send(v.target_system, v.target_component, seq,
                                                    mission_type)
        out.append(Item(seq=seq, frame=int(got.frame), command=int(got.command),
                        params=(float(got.param1), float(got.param2), float(got.param3),
                                float(got.param4)),
                        x=int(got.x), y=int(got.y), z=float(got.z), current=int(got.current),
                        autocontinue=int(got.autocontinue)))
    v.conn.mav.mission_ack_send(v.target_system, v.target_component, MAV_MISSION_ACCEPTED,
                                mission_type)
    return out


def _normalised(it: Item) -> Item:
    """What ArduPilot (Copter-4.7.2, AP_Mission) gives back for an item, where it does not keep
    the item as sent. Seen in SITL; every rule is one storage detail of the firmware:

    * a command without a position (DO_*, CONDITION_*) has no frame in storage: it comes back
      as MAV_FRAME_GLOBAL with zero lat / lon / alt;
    * NAV_LAND param4 is stored as a loiter direction (Plane deep-stall) and comes back as
      -1 / 1 by its sign.
    Anything else that differs is a real difference."""
    if it.command not in POSITIONAL_COMMANDS and it.command not in (CMD_FENCE_POLYGON_VERTEX_INCLUSION,
                                                                     CMD_RALLY_POINT):
        it = Item(seq=it.seq, frame=MAV_FRAME_GLOBAL, command=it.command, params=it.params,
                  x=0, y=0, z=0.0, current=it.current, autocontinue=it.autocontinue)
    if it.command == CMD_NAV_LAND:
        p = it.params
        it = Item(seq=it.seq, frame=it.frame, command=it.command,
                  params=(p[0], p[1], p[2], -1.0 if p[3] < 0 else 1.0), x=it.x, y=it.y, z=it.z,
                  current=it.current, autocontinue=it.autocontinue)
    return it


def compare(sent: list[Item], back: list[Item], what: str, skip_home: bool = False) -> list[str]:
    """Differences between what was uploaded and what the vehicle holds ([] = the same, up to
    the firmware's documented storage normalisation)."""
    diffs: list[str] = []
    if len(sent) != len(back):
        return [f"{what}: {len(sent)} items sent, the vehicle holds {len(back)}"]
    for a, b in zip(sent, back):
        if skip_home and a.seq == 0:
            continue
        a, b = _normalised(a), _normalised(b)
        if (a.frame, a.command) != (b.frame, b.command):
            diffs.append(f"{what} item {a.seq}: frame/command {a.frame}/{a.command} sent, "
                         f"{b.frame}/{b.command} held")
            continue
        if abs(a.x - b.x) > LATLON_TOL_E7 or abs(a.y - b.y) > LATLON_TOL_E7:
            diffs.append(f"{what} item {a.seq}: position {a.x},{a.y} sent, {b.x},{b.y} held")
        if abs(a.z - b.z) > ALT_TOL_M:
            diffs.append(f"{what} item {a.seq}: altitude {a.z:.3f} sent, {b.z:.3f} held")
        for k, (pa, pb) in enumerate(zip(a.params, b.params), start=1):
            if not math.isclose(pa, pb, rel_tol=1e-5, abs_tol=1e-4):
                diffs.append(f"{what} item {a.seq}: param{k} {pa:g} sent, {pb:g} held")
    return diffs


def upload_verified(v: Vehicle, items: list[Item], mission_type: int,
                    timeout: float = 30.0) -> list[Item]:
    """Upload, download again, refuse any difference. Returns what the vehicle holds."""
    upload(v, items, mission_type, timeout)
    back = download(v, mission_type, timeout)
    diffs = compare(items, back, MISSION_TYPES[mission_type],
                    skip_home=mission_type == MISSION_TYPE_MISSION)
    if diffs:
        raise MissionProtocolError("the vehicle does not hold what was uploaded: " + "; ".join(diffs))
    return back
