"""What a run leaves behind: the true trajectory, the estimate next to it, and events, all on
the simulation clock (the autopilot's time since boot, SIMULATION §4.1: never wall-clock).

Truth is SIM_STATE converted to Site ENU (the same frame the plan and the occupancy grid are
in). The estimate (GLOBAL_POSITION_INT, EKF3) is recorded only to be compared, never judged.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class Event:
    t: float                    # simulation seconds since boot
    kind: str                   # mode, armed, disarmed, mission_start, current, reached, statustext,
                                # fault_on, fault_off, fence, home, landed, end, note
    detail: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class Recording:
    """Columns of truth samples (one row per SIM_STATE) and a list of events."""

    t: list[float] = field(default_factory=list)
    enu: list[tuple[float, float, float]] = field(default_factory=list)       # truth, Site ENU
    vel_ned: list[tuple[float, float, float]] = field(default_factory=list)   # truth
    yaw_deg: list[float] = field(default_factory=list)                        # truth, compass
    mode: list[str] = field(default_factory=list)
    armed: list[bool] = field(default_factory=list)
    seq: list[int] = field(default_factory=list)                 # mission item being flown to
    est_enu: list[Optional[tuple[float, float, float]]] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def add(self, *, t: float, enu, vel_ned, yaw_deg: float, mode: str, armed: bool, seq: int,
            est_enu=None) -> None:
        self.t.append(float(t))
        self.enu.append((float(enu[0]), float(enu[1]), float(enu[2])))
        self.vel_ned.append((float(vel_ned[0]), float(vel_ned[1]), float(vel_ned[2])))
        self.yaw_deg.append(float(yaw_deg))
        self.mode.append(mode)
        self.armed.append(bool(armed))
        self.seq.append(int(seq))
        self.est_enu.append(None if est_enu is None else
                            (float(est_enu[0]), float(est_enu[1]), float(est_enu[2])))

    def event(self, t: float, kind: str, detail: str = "", **data: Any) -> Event:
        e = Event(float(t), kind, detail, data)
        self.events.append(e)
        return e

    # ── views ────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.t)

    @property
    def times(self) -> F64:
        return np.asarray(self.t, float)

    @property
    def positions(self) -> F64:
        return np.asarray(self.enu, float).reshape(-1, 3)

    @property
    def estimates(self) -> F64:
        """(N, 3) with NaN rows where no estimate was paired."""
        out = np.full((len(self.t), 3), np.nan)
        for k, e in enumerate(self.est_enu):
            if e is not None:
                out[k] = e
        return out

    def events_of(self, *kinds: str) -> list[Event]:
        return [e for e in self.events if e.kind in kinds]

    def first(self, kind: str, after: float = -math.inf) -> Optional[Event]:
        return next((e for e in self.events if e.kind == kind and e.t >= after), None)

    def mode_changes(self) -> list[tuple[float, str]]:
        return [(e.t, e.detail) for e in self.events if e.kind == "mode"]

    # ── files ────────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {"t": self.t, "enu": self.enu, "vel_ned": self.vel_ned, "yaw_deg": self.yaw_deg,
                "mode": self.mode, "armed": self.armed, "seq": self.seq, "est_enu": self.est_enu,
                "events": [asdict(e) for e in self.events], "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> "Recording":
        r = cls(t=list(d["t"]), enu=[tuple(p) for p in d["enu"]],
                vel_ned=[tuple(v) for v in d["vel_ned"]], yaw_deg=list(d["yaw_deg"]),
                mode=list(d["mode"]), armed=list(d["armed"]), seq=list(d["seq"]),
                est_enu=[None if e is None else tuple(e) for e in d["est_enu"]],
                meta=dict(d.get("meta", {})))
        r.events = [Event(e["t"], e["kind"], e.get("detail", ""), e.get("data", {}))
                    for e in d["events"]]
        return r

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Recording":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
