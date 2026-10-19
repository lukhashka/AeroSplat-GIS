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


# ── snapping to the facade's frame (§5.6 item 3) ─────────────────────────
# An opening's edge moves to its row's or column's line only when it is already this close.
SNAP_TOL_M = 0.15
# Edges closer than this to the line are left alone: not worth an audit entry.
SNAP_NOOP_M = 0.002
# After snapping, both sides must still be at least this long (the editor's own minimum).
SNAP_MIN_SIDE_M = 0.05

Edge = str          # "u0" | "u1" | "v0" | "v1": left, right, bottom, top in facade (u, v)


@dataclass(frozen=True)
class SnapBox:
    """An opening's axis-aligned box in facade (u, v), and the group it snaps within."""

    id: str
    u0: float
    u1: float
    v0: float
    v1: float
    group: str = ""            # e.g. the class: a door does not set a window's lines


@dataclass(frozen=True)
class EdgeMove:
    id: str
    edge: Edge
    old: float
    new: float
    line: str                  # "row 3", "column 2" (0-based within the group)

    @property
    def delta(self) -> float:
        return self.new - self.old


@dataclass(frozen=True)
class EdgeKept:
    """An edge further from its line than the tolerance: left where the operator put it."""

    id: str
    edge: Edge
    off_m: float
    line: str
    reason: str


@dataclass
class SnapResult:
    tol_m: float
    moves: list[EdgeMove] = field(default_factory=list)
    kept: list[EdgeKept] = field(default_factory=list)
    # id -> the new box (u0, u1, v0, v1), only for boxes that changed
    boxes: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)

    def moves_of(self, box_id: str) -> list[EdgeMove]:
        return [m for m in self.moves if m.id == box_id]


def _median_f(xs: Sequence[float]) -> float:
    return float(_median(list(xs)))


def snap_to_grid(boxes: Sequence[SnapBox], *, tol_m: float = SNAP_TOL_M,
                 row_tol_m: float = ROW_TOL_M, column_tol_m: float = COLUMN_TOL_M) -> SnapResult:
    """Snap opening edges to the facade's row and column lines (§5.6 item 3).

    Rows and columns are the same clusters the addressing uses (v- and u-centres, within a
    group). A row's lines are the medians of its members' bottom and top edges, a column's
    the medians of their left and right edges; a line needs at least two members. An edge
    moves onto its line only when it is within `tol_m` of it, so **no edge ever moves by
    more than `tol_m`**; an edge further away is reported in `kept` and left alone (that is
    a different opening, or a drawing to look at, not noise). A box that would come out
    thinner than `SNAP_MIN_SIDE_M` is not touched at all.
    """
    if tol_m <= 0:
        raise ValueError("tol_m must be positive")
    proposed: dict[str, dict[Edge, tuple[float, str]]] = {}
    out = SnapResult(tol_m=tol_m)
    groups: dict[str, list[SnapBox]] = {}
    for b in boxes:
        groups.setdefault(b.group, []).append(b)

    def consider(members: list[SnapBox], lo: Edge, hi: Edge, label: str) -> None:
        for edge in (lo, hi):
            line = _median_f([getattr(b, edge) for b in members])
            for b in members:
                off = line - getattr(b, edge)
                if abs(off) <= SNAP_NOOP_M:
                    continue
                if abs(off) > tol_m:
                    out.kept.append(EdgeKept(id=b.id, edge=edge, off_m=off, line=label,
                                             reason=f"{abs(off):.2f} м від лінії > допуску {tol_m:g} м"))
                    continue
                proposed.setdefault(b.id, {})[edge] = (line, label)

    for name, members in sorted(groups.items()):
        tag = f"{name} " if name else ""
        vc = [0.5 * (b.v0 + b.v1) for b in members]
        uc = [0.5 * (b.u0 + b.u1) for b in members]
        rows = sorted(cluster_1d(vc, row_tol_m), key=lambda g: _centre(vc, g))
        cols = sorted(cluster_1d(uc, column_tol_m), key=lambda g: _centre(uc, g))
        for k, g in enumerate(rows):
            if len(g) >= 2:
                consider([members[i] for i in g], "v0", "v1", f"{tag}ряд {k}")
        for k, g in enumerate(cols):
            if len(g) >= 2:
                consider([members[i] for i in g], "u0", "u1", f"{tag}колонка {k}")

    by_id = {b.id: b for b in boxes}
    for box_id, edges in proposed.items():
        b = by_id[box_id]
        new = {e: edges[e][0] if e in edges else getattr(b, e) for e in ("u0", "u1", "v0", "v1")}
        if new["u1"] - new["u0"] < SNAP_MIN_SIDE_M or new["v1"] - new["v0"] < SNAP_MIN_SIDE_M:
            out.kept.extend(EdgeKept(id=box_id, edge=e, off_m=v - getattr(b, e), line=label,
                                     reason="рамка вийшла б надто тонкою")
                            for e, (v, label) in edges.items())
            continue
        for e, (v, label) in sorted(edges.items()):
            out.moves.append(EdgeMove(id=box_id, edge=e, old=getattr(b, e), new=v, line=label))
        out.boxes[box_id] = (new["u0"], new["u1"], new["v0"], new["v1"])
    return out


def floor_label(floor_index: int, floor_offset: int = 0) -> str:
    """Human-facing storey number. In Ukraine the ground floor is the 1st (§4)."""
    return str(floor_index + 1 + floor_offset)


def canonical_address(site_id: str, building_id: str, facade_name: str, floor_index: int,
                      column: int) -> str:
    """`<site>/<building>/<facade>/<floor>/<column>`, e.g. `podil/B012/N1/04/03` (§4)."""
    return f"{site_id}/{building_id}/{facade_name}/{floor_index:02d}/{column:02d}"
