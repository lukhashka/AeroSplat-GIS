"""Facades across runs (ACCESS_POINTS §5.9): matching walls, placing and carrying features.

Synthetic facades with known answers. Most tests are about what must NOT match, or must
come back ambiguous: a wrong match carries a window onto someone else's wall.
"""

import math

import numpy as np
import pytest

from aerosplat.authored.carry import (corners_enu, foreign_on, is_native, native_features,
                                      place_feature, reanchor)
from aerosplat.authored.geometry import make_access_point
from aerosplat.authored.models import Address, Lifecycle
from aerosplat.semantics.matching import MatchParams, match_facade, match_facades, pair
from aerosplat.semantics.models import Facade, SurfaceKind, azimuth_of, u_axis_for


def wall(fid, azimuth_deg, centre_xy, length, *, z0=0.0, height=30.0, offset=0.0, shift=0.0,
         rms=0.3):
    """A vertical facade whose middle sits at `centre_xy` + `offset` along the normal and
    `shift` along the wall, `length` long, facing `azimuth_deg`."""
    a = math.radians(azimuth_deg)
    n = np.array([math.sin(a), math.cos(a)])
    u = np.array(u_axis_for((float(n[0]), float(n[1]))))
    mid = np.asarray(centre_xy, float) + offset * n + shift * u
    origin = (float(mid[0]), float(mid[1]), z0)
    return Facade(id=fid, building_id=fid.split("/")[0], kind=SurfaceKind.wall, origin=origin,
                  normal=(float(n[0]), float(n[1]), 0.0), u_axis=(float(u[0]), float(u[1]), 0.0),
                  azimuth_deg=azimuth_of((float(n[0]), float(n[1]))), u_min=-length / 2,
                  u_max=length / 2, v_ground=z0, v_max=z0 + height, plane_rms_m=rms,
                  n_points=1000)


def window(facade, u0, v0, u1, v1, **kw):
    return make_access_point(site_key="demo/pier", facade=facade, class_id="opening.window",
                             taxonomy="core@1.1", polygon_uv=[(u0, v0), (u1, v0), (u1, v1), (u0, v1)],
                             **kw)


# ── facade ↔ facade ──────────────────────────────────────────────────────
def test_a_refitted_wall_under_a_new_name_is_the_same_wall():
    a = wall("B08/SW3", 214.0, (10, -20), 55.0)
    b = wall("B04/SW1", 215.5, (10, -20), 54.0, offset=0.4, shift=0.6)
    m = match_facade(a, [b, wall("B04/NE1", 34.0, (10, 20), 95.0)])
    assert m.status == "match" and m.b_ids == ["B04/SW1"] and m.relation == "same"
    c = m.candidates[0]
    assert c.angle_deg == pytest.approx(1.5, abs=1e-6)
    assert c.offset_m < 0.45 and c.overlap_a > 0.95 and m.confidence > 0.6


def test_two_walls_merged_by_the_other_run_are_both_inside_it():
    """dataset_1910: run 1 has SW2 and SW3 where run 2 has one SW1."""
    left = wall("B08/SW2", 215.0, (-30, 0), 40.0, shift=-27.5)
    right = wall("B08/SW3", 215.0, (-30, 0), 55.0, shift=20.0)
    whole = wall("B04/SW1", 215.0, (-30, 0), 95.0, offset=0.3)
    ms = {m.a_id: m for m in match_facades([left, right], [whole])}
    for a_id, other in (("B08/SW2", "B08/SW3"), ("B08/SW3", "B08/SW2")):
        assert ms[a_id].status == "match" and ms[a_id].b_ids == ["B04/SW1"]
        assert ms[a_id].relation == "inside" and ms[a_id].shared_with == [other]
    # ... and the other way round, the long wall is split, left to right along it.
    back = match_facade(whole, [right, left])
    assert back.status == "split" and back.b_ids == ["B08/SW2", "B08/SW3"]
    assert back.confidence > 0.8


def test_two_fits_of_the_same_stretch_are_ambiguous_not_the_closer_one():
    a = wall("A/N1", 0.0, (0, 0), 30.0)
    near = wall("B/N1", 0.0, (0, 0), 30.0, offset=0.2)
    far = wall("B/N2", 0.0, (0, 0), 20.0, offset=0.9)       # within the 1 m tolerance
    m = match_facade(a, [near, far])
    assert m.status == "ambiguous" and m.b_ids == ["B/N1", "B/N2"] and m.confidence == 0.0
    assert "не вибираємо" in m.reason


@pytest.mark.parametrize("other, why", [
    (wall("B/S1", 180.0, (0, 0), 30.0), "faces the other way (the back of a thin wall)"),
    (wall("B/E1", 90.0, (0, 0), 30.0), "perpendicular"),
    (wall("B/N1", 12.0, (0, 0), 30.0), "12° off: beyond the 10° gate"),
    (wall("B/N1", 0.0, (0, 0), 30.0, offset=4.0), "a parallel wall 4 m away"),
    (wall("B/N1", 0.0, (0, 0), 20.0, shift=40.0), "the same line, further along the street"),
    (wall("B/N1", 0.0, (0, 0), 30.0, z0=40.0, height=10.0), "a different height band"),
    (wall("B/N1", 0.0, (0, 0), 30.0, shift=29.0), "touching for 1 m only"),
])
def test_what_is_not_the_same_wall(other, why):
    a = wall("A/N1", 0.0, (0, 0), 30.0)
    assert pair(a, other) is None, why
    assert match_facade(a, [other]).status == "none", why


def test_the_offset_tolerance_grows_with_the_planes_scatter():
    a = wall("A/N1", 0.0, (0, 0), 30.0, rms=0.9)
    b = wall("B/N1", 0.0, (0, 0), 30.0, offset=2.2, rms=0.85)
    assert pair(a, b) is not None                       # 2 × hypot(0.9, 0.85) = 2.48 m
    tight = wall("B/N1", 0.0, (0, 0), 30.0, offset=2.2, rms=0.1)
    assert pair(wall("A/N1", 0.0, (0, 0), 30.0, rms=0.1), tight) is None   # floor 1.0 m
    assert MatchParams().offset_tol(5.0, 5.0) == 3.0                      # and a cap


def test_building_ids_do_not_matter_and_the_prefilter_drops_nothing():
    rng = np.random.default_rng(7)
    A = [wall(f"B{i:02d}/N{i}", float(rng.uniform(0, 360)), rng.uniform(-80, 80, 2),
              float(rng.uniform(6, 60)), rms=float(rng.uniform(0.1, 0.9))) for i in range(60)]
    B = [wall(f"X{i:02d}/N1", f.azimuth_deg + float(rng.normal(0, 3)),
              np.asarray(f.origin[:2]), f.length * float(rng.uniform(0.6, 1.4)),
              offset=float(rng.normal(0, 0.8)), shift=float(rng.normal(0, 5)),
              rms=float(rng.uniform(0.1, 0.9)))
         for i, f in enumerate(A)]
    fast = match_facades(A, B)
    slow = [match_facade(a, B) for a in A]
    assert [(m.status, m.b_ids) for m in fast] == [(m.status, m.b_ids) for m in slow]
    assert sum(m.status == "match" for m in fast) > 30


# ── features across runs ─────────────────────────────────────────────────
def test_a_feature_is_native_only_to_the_exact_frame_it_was_drawn_in():
    a = wall("B04/SW1", 215.0, (0, 0), 40.0)
    other_building = wall("B04/SW1", 30.0, (200, 50), 20.0)        # same id, next run
    ap = window(a, 1.0, 3.0, 2.2, 4.5)
    assert is_native(ap, a) and not is_native(ap, other_building)
    assert native_features([ap], other_building) == []


def test_a_feature_lands_on_the_matching_wall_of_the_other_run():
    a = wall("B08/SW3", 215.0, (0, 0), 55.0)
    b = wall("B04/SW1", 216.0, (0, 0), 95.0, offset=0.3, shift=-20.0)
    ap = window(a, 10.0, 12.0, 11.4, 13.6)
    p = place_feature(ap, [b, wall("B04/NE1", 35.0, (0, 30), 95.0)])
    assert p.status == "placed" and p.facade_id == "B04/SW1"
    assert p.fit is not None and p.fit.offset_m == pytest.approx(0.3, abs=0.25)
    us = [q[0] for q in p.fit.polygon_uv]
    vs = [q[1] for q in p.fit.polygon_uv]
    # Heights are kept exactly; along the wall it moves by b's shift of the middle.
    assert sorted(set(vs)) == pytest.approx([12.0, 13.6])
    assert min(us) == pytest.approx(30.0, abs=0.4) and max(us) - min(us) == pytest.approx(1.4, abs=0.01)


def test_a_feature_where_two_fits_overlap_is_ambiguous_and_cannot_be_carried():
    a = wall("A/N1", 0.0, (0, 0), 30.0)
    b1 = wall("B/N1", 0.0, (0, 0), 30.0, offset=0.2)
    b2 = wall("B/N2", 0.0, (0, 0), 10.0, offset=0.8, shift=8.0)
    inside_both = window(a, 7.0, 3.0, 8.0, 4.0)
    only_b1 = window(a, -10.0, 3.0, -9.0, 4.0)
    assert place_feature(inside_both, [b1, b2]).status == "ambiguous"
    assert place_feature(only_b1, [b1, b2]).status == "placed"
    with pytest.raises(ValueError, match="не можна перенести"):
        reanchor(inside_both, b1, [b1, b2])


def test_a_feature_beyond_the_other_runs_facade_lands_nowhere():
    a = wall("A/N1", 0.0, (0, 0), 40.0)
    short = wall("B/N1", 0.0, (0, 0), 20.0, shift=-10.0)          # covers u -20..0 of A only
    assert place_feature(window(a, 12.0, 3.0, 13.0, 4.0), [short]).status == "none"
    assert place_feature(window(a, -12.0, 3.0, -11.0, 4.0), [short]).status == "placed"


def test_reanchor_reexpresses_the_shape_and_drops_the_address():
    a = wall("B08/SW3", 215.0, (0, 0), 55.0)
    b = wall("B04/SW1", 215.0, (0, 0), 95.0, offset=0.5, shift=-20.0)
    ap = window(a, 10.0, 12.0, 11.4, 13.6, lifecycle=Lifecycle.verified)
    ap = ap.model_copy(update={"address": Address(floor_index=4, column=3,
                                                  canonical="demo/B08/SW3/04/03")})
    moved, placement = reanchor(ap, b, [b])
    assert moved.id == ap.id and moved.lifecycle is Lifecycle.verified
    assert moved.host is not None and moved.host.facade_id == "B04/SW1" and is_native(moved, b)
    assert moved.address is None
    # The world position changes only by the plane offset, straight along the normal.
    d = np.asarray(moved.geometry.center) - np.asarray(ap.geometry.center)
    assert np.linalg.norm(d) == pytest.approx(0.5, abs=1e-6)
    assert abs(float(np.dot(d, b.normal))) == pytest.approx(0.5, abs=1e-6)
    assert np.allclose(corners_enu(moved)[:, 2], corners_enu(ap)[:, 2])
    assert ap.host.facade_id == "B08/SW3"                                # the input is untouched
    assert placement.status == "placed"


def test_foreign_on_lists_only_other_runs_features_on_this_wall():
    a = wall("B08/SW3", 215.0, (0, 0), 55.0)
    b = wall("B04/SW1", 215.0, (0, 0), 95.0, offset=0.3, shift=-20.0)
    elsewhere = wall("B04/NE1", 35.0, (0, 40), 95.0)
    mine = window(b, 0.0, 3.0, 1.0, 4.0)
    theirs = window(a, 10.0, 12.0, 11.4, 13.6)
    far = window(wall("B09/N1", 0.0, (300, 300), 20.0), 0.0, 3.0, 1.0, 4.0)
    got = foreign_on(b, [b, elsewhere], [mine, theirs, far])
    assert [p.feature_id for p in got] == [theirs.id] and got[0].status == "placed"
