"""Snapping openings to the facade's rows and columns (ACCESS_POINTS §5.6 item 3).

The one rule that matters: no edge ever moves by more than the tolerance. An edge further
off is a different opening or a drawing to look at, and is left where the operator put it.
"""

import numpy as np
import pytest

from aerosplat.semantics.addressing import SNAP_TOL_M, SnapBox, snap_to_grid


def row(v0s, v1s, *, u_step=3.0, width=1.4, group="opening.window", prefix="w"):
    return [SnapBox(id=f"{prefix}{i}", u0=i * u_step, u1=i * u_step + width, v0=a, v1=b, group=group)
            for i, (a, b) in enumerate(zip(v0s, v1s))]


def test_a_ragged_row_lands_on_its_median_lines():
    boxes = row([3.00, 3.05, 2.95, 3.10], [4.50, 4.48, 4.55, 4.50])
    r = snap_to_grid(boxes)
    assert {m.edge for m in r.moves} == {"v0", "v1"}
    lines = {(m.edge, round(m.new, 6)) for m in r.moves}
    assert lines == {("v0", 3.025), ("v1", 4.5)}
    assert all(abs(m.delta) <= SNAP_TOL_M for m in r.moves)
    for box_id, (_, _, v0, v1) in r.boxes.items():
        assert (v0, v1) == pytest.approx((3.025, 4.5))
    assert r.kept == []
    # u is untouched: each column has a single opening, so it has no line to snap to.
    assert all(m.edge in ("v0", "v1") for m in r.moves)


def test_an_edge_beyond_the_tolerance_is_kept_and_reported():
    boxes = row([3.00, 3.02, 2.98, 3.40], [4.50, 4.50, 4.50, 4.52])
    r = snap_to_grid(boxes, tol_m=0.15)
    assert not any(m.id == "w3" and m.edge == "v0" for m in r.moves)
    kept = [k for k in r.kept if k.id == "w3"]
    assert [k.edge for k in kept] == ["v0"] and "допуску" in kept[0].reason
    assert kept[0].off_m == pytest.approx(3.01 - 3.40)
    # w3's top is within tolerance and still snaps.
    assert any(m.id == "w3" and m.edge == "v1" for m in r.moves)


def test_columns_snap_left_and_right_edges():
    boxes = [SnapBox(id=f"f{k}", u0=5.0 + d0, u1=6.4 + d1, v0=3.0 * k, v1=3.0 * k + 1.5)
             for k, (d0, d1) in enumerate([(0.0, 0.0), (0.06, -0.04), (-0.05, 0.03)])]
    r = snap_to_grid(boxes)
    assert {(m.id, m.edge) for m in r.moves} == {("f1", "u0"), ("f1", "u1"), ("f2", "u0"), ("f2", "u1")}
    assert all(b[0] == pytest.approx(5.0) and b[1] == pytest.approx(6.4) for b in r.boxes.values())


def test_a_door_does_not_set_the_lines_of_windows():
    windows = row([3.0, 3.1], [4.5, 4.5])
    door = [SnapBox(id="d", u0=0.0, u1=1.2, v0=3.05, v1=4.45, group="opening.door")]
    r = snap_to_grid(windows + door)
    assert not any(m.id == "d" for m in r.moves)


def test_lonely_openings_and_a_regular_facade_do_not_move():
    assert snap_to_grid(row([3.0], [4.5])).moves == []
    regular = row([3.0, 3.0, 3.0], [4.5, 4.5, 4.5])
    assert snap_to_grid(regular).moves == [] and snap_to_grid(regular).kept == []


def test_snapping_is_idempotent():
    boxes = row([3.00, 3.05, 2.95, 3.10], [4.50, 4.48, 4.55, 4.50])
    r = snap_to_grid(boxes)
    again = [SnapBox(id=b.id, u0=r.boxes.get(b.id, (b.u0, b.u1, b.v0, b.v1))[0],
                     u1=r.boxes.get(b.id, (b.u0, b.u1, b.v0, b.v1))[1],
                     v0=r.boxes.get(b.id, (b.u0, b.u1, b.v0, b.v1))[2],
                     v1=r.boxes.get(b.id, (b.u0, b.u1, b.v0, b.v1))[3], group=b.group) for b in boxes]
    assert snap_to_grid(again).moves == []


@pytest.mark.parametrize("seed", range(5))
def test_no_edge_ever_moves_further_than_the_tolerance(seed):
    rng = np.random.default_rng(seed)
    boxes = []
    for f in range(6):
        for c in range(5):
            u0 = c * 3.0 + rng.normal(0, 0.12)
            v0 = 1.0 + f * 3.0 + rng.normal(0, 0.12)
            boxes.append(SnapBox(id=f"b{f}{c}", u0=u0, u1=u0 + 1.4 + rng.normal(0, 0.1),
                                 v0=v0, v1=v0 + 1.5 + rng.normal(0, 0.1)))
    tol = float(rng.uniform(0.05, 0.2))
    r = snap_to_grid(boxes, tol_m=tol)
    assert r.moves, "a noisy grid has something to snap"
    assert max(abs(m.delta) for m in r.moves) <= tol + 1e-12
    by_id = {b.id: b for b in boxes}
    for box_id, (u0, u1, v0, v1) in r.boxes.items():
        b = by_id[box_id]
        assert max(abs(u0 - b.u0), abs(u1 - b.u1), abs(v0 - b.v0), abs(v1 - b.v1)) <= tol + 1e-12
        assert u1 > u0 and v1 > v0


def test_a_box_that_would_collapse_is_not_touched():
    boxes = [SnapBox(id="a", u0=0.0, u1=0.06, v0=0.0, v1=1.0),
             SnapBox(id="b", u0=0.0, u1=1.0, v0=3.0, v1=4.0),
             SnapBox(id="c", u0=0.12, u1=0.9, v0=6.0, v1=7.0)]
    r = snap_to_grid(boxes, tol_m=0.15)
    assert "a" not in r.boxes or r.boxes["a"][1] - r.boxes["a"][0] >= 0.05
    with pytest.raises(ValueError):
        snap_to_grid(boxes, tol_m=0.0)
