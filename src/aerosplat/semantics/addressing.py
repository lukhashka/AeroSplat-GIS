"""Floor rows, columns and canonical addresses (ACCESS_POINTS §4, §5.6).

Facades are almost always regular, so the rows and columns of a facade can be
recovered by clustering the openings' v- and u-centres. That gives a person a way to
say *which* window: "fourth floor, third from the left", and the system a way to
resolve it back to one opening.

A wrong floor or column is the most dangerous error in the whole system — the drone
flies to someone else's window — so this module is deliberately shy. It reports
`floor_ambiguous` whenever it is working from thin evidence, and §6.3 then blocks
confirming such a target without an explicit human check. It never invents an opening
and never moves one: the operator's drawing stays where they put it.

Two numberings per row, both kept (§4):
    `column`             the column of the facade's regular grid
    `ordinal_on_floor`   the n-th opening that actually *exists* on that row
A person saying "third window" means the second one. When they disagree, the resolver
has to offer both, which is only possible if both are stored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

# A row whose bottom edge is no higher than this above the ground is the ground floor.
GROUND_ROW_MAX_M = 1.5
# Storey height prior (§5.6). The ground floor may be taller -- shops -- so a single
# tall first interval is not by itself a reason to doubt the numbering.
FLOOR_H_MIN_M = 2.4
FLOOR_H_MAX_M = 4.2
# Openings within this of each other in v (or u) belong to the same row (or column).
ROW_TOL_M = 0.6
COLUMN_TOL_M = 0.6


@dataclass(frozen=True)
class Opening:
    """The little an address needs to know about an opening."""

    id: str
    u_center: float
    v_center: float
    height: float = 0.0        # along v; used to find the row's bottom edge
    width: float = 0.0


@dataclass
class Row:
    index: int                 # 0 = the row nearest the ground
    v_center: float
    v_bottom: float
    members: list[str] = field(default_factory=list)


@dataclass
class FacadeAddressing:
    rows: list[Row]
    column_centres: list[float]
    floor_height_m: Optional[float]           # median interval between row centres
    ambiguous: bool
    reasons: list[str] = field(default_factory=list)
    # opening id -> (floor_index, column, ordinal_on_floor)
    of: dict[str, tuple[int, int, int]] = field(default_factory=dict)


def cluster_1d(values: Sequence[float], tol: float) -> list[list[int]]:
    """Group indices whose values form a chain with no gap wider than `tol`.

    Single-linkage on a sorted line: the 1-D case of mean-shift with a flat kernel, and
    the behaviour §5.6 wants — a row is a run of openings at nearly the same height,
    however many there are, and a gap of more than `tol` starts a new one.
    """
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda i: values[i])
    groups: list[list[int]] = [[order[0]]]
    for prev, cur in zip(order, order[1:]):
        if values[cur] - values[prev] > tol:
            groups.append([cur])
        else:
            groups[-1].append(cur)
    return groups


def _centre(values: Sequence[float], group: Sequence[int]) -> float:
    return sum(values[i] for i in group) / len(group)


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def address_facade(openings: Sequence[Opening], *, v_ground: float,
                   row_tol_m: float = ROW_TOL_M, column_tol_m: float = COLUMN_TOL_M,
                   floor_offset: int = 0) -> FacadeAddressing:
    """Rows, columns and per-opening addresses for the openings of one facade.

    `floor_offset` shifts the human-facing label (a basement or a mezzanine the
    operator told us about); it never shifts `floor_index`, which always counts from
    the row at ground level.
    """
    if not openings:
        return FacadeAddressing(rows=[], column_centres=[], floor_height_m=None, ambiguous=False)

    vs = [o.v_center for o in openings]
    us = [o.u_center for o in openings]
    v_groups = cluster_1d(vs, row_tol_m)
    v_groups.sort(key=lambda g: _centre(vs, g))
    u_groups = cluster_1d(us, column_tol_m)
    u_groups.sort(key=lambda g: _centre(us, g))

    column_of: dict[str, int] = {}
    for col, group in enumerate(u_groups, start=1):
        for i in group:
            column_of[openings[i].id] = col

    rows: list[Row] = []
    for index, group in enumerate(v_groups):
        v_centre = _centre(vs, group)
        bottom = min(openings[i].v_center - openings[i].height / 2 for i in group)
        rows.append(Row(index=index, v_center=v_centre, v_bottom=bottom,
                        members=[openings[i].id for i in sorted(group, key=lambda i: us[i])]))

    # Floors count from the row at ground level. If the lowest row is well above the
    # ground, the ground row was not drawn -- trees, a fence, no coverage -- and the
    # numbering is a guess from there up.
    reasons: list[str] = []
    lowest = rows[0]
    if lowest.v_bottom - v_ground > GROUND_ROW_MAX_M:
        reasons.append(f"найнижчий ряд на {lowest.v_bottom - v_ground:.1f} м над землею "
                       f"(> {GROUND_ROW_MAX_M} м): ряд біля землі не позначено")

    intervals = [b.v_center - a.v_center for a, b in zip(rows, rows[1:])]
    floor_h = _median(intervals) if intervals else None
    # The ground floor is often taller, so the first interval gets a pass.
    checked = intervals[1:] if len(intervals) > 1 else intervals
    odd = [d for d in checked if not FLOOR_H_MIN_M <= d <= FLOOR_H_MAX_M]
    if odd:
        reasons.append(f"{len(odd)} з {len(checked)} інтервалів між рядами поза "
                       f"{FLOOR_H_MIN_M}–{FLOOR_H_MAX_M} м "
                       f"({', '.join(f'{d:.1f}' for d in odd[:4])} м)")
    if floor_offset and reasons:
        reasons.append(f"задано floor_offset={floor_offset}, а нумерація вже під питанням")

    of: dict[str, tuple[int, int, int]] = {}
    for row in rows:
        for ordinal, member in enumerate(row.members, start=1):
            of[member] = (row.index, column_of[member], ordinal)
    return FacadeAddressing(rows=rows, column_centres=[_centre(us, g) for g in u_groups],
                            floor_height_m=floor_h, ambiguous=bool(reasons), reasons=reasons, of=of)


def floor_label(floor_index: int, floor_offset: int = 0) -> str:
    """Human-facing storey number. In Ukraine the ground floor is the 1st (§4)."""
    return str(floor_index + 1 + floor_offset)


def canonical_address(site_id: str, building_id: str, facade_name: str, floor_index: int,
                      column: int) -> str:
    """`<site>/<building>/<facade>/<floor>/<column>`, e.g. `podil/B012/N1/04/03` (§4)."""
    return f"{site_id}/{building_id}/{facade_name}/{floor_index:02d}/{column:02d}"
