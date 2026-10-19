"""A recorded flight as the web UI replays it: what the vehicle was doing at each sample.

The phase of a sample is the role, in the plan, of the mission item being flown to (`seq`, from
MISSION_CURRENT); the hold at H is told from the approach to it by the distance to H, with the
hover oracle's tolerances. Anything flown armed outside AUTO after the mission started is a
failsafe (or the operator) and is marked as such — the report's top view makes the same cut.

Only the recording and the plan are needed, so a real flight log converted to a `Recording`
(SIMULATION §2 idea 13) replays the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from aerosplat.mission.models import MissionPlan
from aerosplat.sim.record import Recording

# Role of the item flown to -> phase. The leg right after H is the exit whatever its item is
# called (E itself, or "over E" when E is at transit height); DO/CONDITION items act at the
# start of the next leg, so they take its phase (`phase_by_seq`).
PHASE_OF_ROLE = {
    "takeoff": "takeoff",
    "over_entry": "transit",
    "entry": "descent",
    "hover": "approach",
    "exit": "exit",
    "over_exit": "climb",
    "over_home": "return",
    "land": "land",
}
GROUND = "ground"           # disarmed
HOLD = "hold"               # at H, within the hover tolerances
PHASES = (GROUND, "takeoff", "transit", "descent", "approach", HOLD, "exit", "climb", "return",
          "land", "other")


@dataclass(frozen=True)
class Segment:
    """A run of samples [i0, i1] with one phase and failsafe state; i1 is inclusive, and the
    next segment starts at i1 so that a polyline of segments has no gaps."""

    phase: str
    failsafe: bool
    mode: str
    i0: int
    i1: int


def mission_start(rec: Recording) -> Optional[float]:
    e = rec.first("mission_start")
    return None if e is None else e.t


def phase_by_seq(plan: MissionPlan) -> dict[int, str]:
    """seq (1-based, as uploaded: seq 0 is home) -> the phase of the leg flown to it."""
    nav = [(seq, it.role) for seq, it in enumerate(plan.items, start=1) if it.enu is not None]
    at_h = next((k for k, (_, role) in enumerate(nav) if role == "hover"), None)
    out: dict[int, str] = {}
    for k, (seq, role) in enumerate(nav):
        phase = PHASE_OF_ROLE.get(role, "other")
        if at_h is not None and k == at_h + 1 and role != "land":
            phase = "exit"
        out[seq] = phase
    following = None
    for seq in range(len(plan.items), 0, -1):          # DO items: the phase of the next leg
        if seq in out:
            following = out[seq]
        else:
            out[seq] = following or (out[max(out)] if out else "other")
    return out


def failsafe_mask(rec: Recording, start_t: Optional[float] = None) -> NDArray[np.bool_]:
    """Armed, after the mission started, not in AUTO."""
    if start_t is None:
        start_t = mission_start(rec)
    t = rec.times
    armed = np.asarray(rec.armed, bool)
    auto = np.asarray(rec.mode, dtype=object) == "AUTO"
    after = t >= start_t if start_t is not None else np.zeros(len(t), bool)
    return armed & after & ~auto


def phases(rec: Recording, plan: MissionPlan, *, hover_m: float = 1.0, hover_alt_m: float = 0.5,
           failsafe: Optional[NDArray[np.bool_]] = None) -> list[str]:
    """Phase of every sample. With `failsafe` (failsafe_mask), a sample outside AUTO keeps the
    phase of the moment the vehicle left it: the mission item it was flying to no longer says
    what it is doing."""
    by_seq = phase_by_seq(plan)
    p = rec.positions
    h = np.asarray(plan.approach.hover, float)
    near_h = np.zeros(len(p), bool)
    if len(p):
        near_h = ((np.linalg.norm(p[:, :2] - h[None, :2], axis=1) <= hover_m)
                  & (np.abs(p[:, 2] - h[2]) <= hover_alt_m))
    out = []
    landing = False
    for k, (armed, seq) in enumerate(zip(rec.armed, rec.seq)):
        if not armed:
            out.append(GROUND)
            landing = False
            continue
        if landing:
            # Copter rewinds the mission to item 1 on touchdown, before it disarms.
            out.append("land")
            continue
        # Armed at seq 0 (home): about to take off.
        ph = "takeoff" if int(seq) == 0 else by_seq.get(int(seq), "other")
        if ph == "approach" and near_h[k]:
            ph = HOLD
        if failsafe is not None and failsafe[k] and k > 0 and failsafe[k - 1]:
            ph = out[-1]
        landing = ph == "land"
        out.append(ph)
    return out


def segments(phase: list[str], failsafe: NDArray[np.bool_], mode: list[str]) -> list[Segment]:
    """Runs of one phase in AUTO; outside it, runs of one mode (RTL, then LAND), labelled with
    the phase the vehicle was in when it left AUTO — the mission item no longer says anything."""
    out: list[Segment] = []
    if not phase:
        return out
    start = 0
    for k in range(1, len(phase) + 1):
        if k == len(phase) or bool(failsafe[k]) != bool(failsafe[start]):
            cut = True
        elif failsafe[start]:
            cut = mode[k] != mode[start]
        else:
            cut = phase[k] != phase[start]
        if cut:
            out.append(Segment(phase=phase[start], failsafe=bool(failsafe[start]), mode=mode[start],
                               i0=start, i1=min(k, len(phase) - 1)))
            start = k
    return out
