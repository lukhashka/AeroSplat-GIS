"""Facade orthophotos and addressing: the pose, the pixel grid, floors and columns.

The pose and the grid are pure arithmetic on a stored facade, so they are checked here
against the render engine's own projection rather than against a picture. ACCESS_POINTS
§4 (addressing), §5.3 (rectified views), §5.6 (regularisation).
"""

import numpy as np
import pytest

from aerosplat.render import camera as cam
from aerosplat.semantics.addressing import (Opening, address_facade, canonical_address,
                                            cluster_1d, floor_label)
from aerosplat.semantics.models import Facade, SurfaceKind, azimuth_of, u_axis_for
from aerosplat.semantics.ortho import (FacadeOrthoError, FacadeOrthoSpec, facade_ortho,
                                       facade_view_yaw)


def a_wall(normal=(0.0, 1.0, 0.0), origin=(0.0, 0.0, 0.0), u_min=-10.0, u_max=10.0,
           height=12.0) -> Facade:
    n = np.asarray(normal, float)[:2]
    n = n / np.linalg.norm(n)
    ux, uy = u_axis_for((float(n[0]), float(n[1])))
    return Facade(id="B01/N1", building_id="B01", kind=SurfaceKind.wall, origin=origin,
                  normal=(float(n[0]), float(n[1]), 0.0), u_axis=(ux, uy, 0.0),
                  azimuth_deg=azimuth_of((float(n[0]), float(n[1]))), u_min=u_min, u_max=u_max,
                  v_ground=origin[2], v_max=origin[2] + height, plane_rms_m=0.1, n_points=500)


# ── the pose ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("normal", [(0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0),
                                    (-0.6, 0.8), (0.37, -0.93)])
def test_the_ortho_camera_axes_are_the_facade_axes(normal):
    """Why the pose comes out clean: with yaw = azimuth + 180 and pitch 0 the camera's
    right axis *is* the facade u-axis and its down axis is world -z. Image columns run
    along u and image rows run down the wall, with no residual rotation to undo."""
    facade = a_wall(normal=normal)
    req, _ = facade_ortho(facade)
    assert req.yaw_deg == pytest.approx(facade_view_yaw(facade))
    R = cam.view_rotation(req.yaw_deg, req.pitch, req.roll_deg)
    assert R[0] == pytest.approx(np.asarray(facade.u_axis), abs=1e-9)      # right = u
    assert R[1] == pytest.approx([0, 0, -1], abs=1e-9)                     # down = -z
    assert R[2] == pytest.approx(-np.asarray(facade.normal), abs=1e-9)     # forward = -n


def test_the_camera_stands_in_front_of_the_wall():
    facade = a_wall(normal=(0.6, 0.8), origin=(13.0, -4.0, 2.5))
    req, _ = facade_ortho(facade, FacadeOrthoSpec(standoff_m=30.0))
    eye = np.array([req.position.e, req.position.n, req.position.u])
    # An orthographic camera's distance does not change the image, but anything behind
    # it is clipped, so it has to be outside the wall.
    assert facade.distance_to_plane(eye) == pytest.approx(30.0)
    assert req.mode == "ortho" and req.pitch == 0.0 and req.roll_deg == 0.0


# ── the pixel grid ───────────────────────────────────────────────────────
@pytest.mark.parametrize("gsd_cm", [1.0, 2.0, 3.0, 5.0, 17.0])
def test_the_pixel_size_is_exactly_what_was_asked_for(gsd_cm):
    req, grid = facade_ortho(a_wall(u_max=7.3), FacadeOrthoSpec(gsd_cm=gsd_cm))
    # `extent_m` is derived from the chosen pixel width, not the other way round, so
    # metres per pixel is the number asked for rather than a rounding of it.
    assert grid.m_per_px == pytest.approx(gsd_cm / 100)
    assert req.extent_m / req.width == pytest.approx(gsd_cm / 100)
    assert req.height * grid.m_per_px == pytest.approx(grid.v_max - grid.v_min)
    assert req.width * grid.m_per_px == pytest.approx(grid.u_max - grid.u_min)


def test_the_frame_covers_the_whole_wall_plus_the_margin():
    facade = a_wall(u_min=-6.0, u_max=14.0, origin=(0.0, 0.0, 3.0), height=9.0)
    _, grid = facade_ortho(facade, FacadeOrthoSpec(gsd_cm=5, margin_m=1.5))
    assert grid.u_min <= facade.u_min - 1.5 and grid.u_max >= facade.u_max + 1.5
    assert grid.v_min <= facade.v_ground - 1.5 and grid.v_max >= facade.v_max + 1.5
    assert 0.5 * (grid.u_min + grid.u_max) == pytest.approx(0.5 * (facade.u_min + facade.u_max))
    assert 0.5 * (grid.v_min + grid.v_max) == pytest.approx(0.5 * (facade.v_ground + facade.v_max))


def test_a_pixel_maps_back_to_the_same_point_on_the_wall():
    facade = a_wall(normal=(0.37, -0.93), origin=(5.0, 11.0, 1.5), u_min=-8.0, u_max=12.0)
    _, grid = facade_ortho(facade, FacadeOrthoSpec(gsd_cm=2))
    for u, v in [(-8.0, 1.5), (0.0, 7.0), (12.0, 13.5), (3.3, 4.4)]:
        i, j = grid.pixel_of_uv(u, v)
        assert grid.uv_of_pixel(i, j) == pytest.approx((u, v), abs=1e-9)
    # u grows with the column and v falls down the rows, so an opening reads where it looks.
    assert grid.du_uv == pytest.approx((grid.m_per_px, 0.0))
    assert grid.dv_uv == pytest.approx((0.0, -grid.m_per_px))
    assert grid.pixel_of_uv(0.0, 5.0)[0] < grid.pixel_of_uv(1.0, 5.0)[0]
    assert grid.pixel_of_uv(0.0, 5.0)[1] > grid.pixel_of_uv(0.0, 6.0)[1]


def test_the_grid_agrees_with_the_render_engines_own_projection():
    """The grid is derived from the request; here it is checked against the projection
    the rasteriser will actually use, so the two cannot drift apart unnoticed."""
    facade = a_wall(normal=(-0.8, 0.6), origin=(-3.0, 7.0, 0.5), u_min=-9.0, u_max=6.0)
    req, grid = facade_ortho(facade, FacadeOrthoSpec(gsd_cm=4))
    eye = np.array([req.position.e, req.position.n, req.position.u])
    V = cam.view_matrix(eye, req.yaw_deg, req.pitch, req.roll_deg)
    K = cam.ortho_intrinsics(req.width, req.height, req.extent_m)
    uv = [(-9.0, 0.5), (0.0, 6.0), (6.0, 12.5), (2.2, 3.3)]
    world = np.array([facade.point(u, v) for u, v in uv])
    pixels, depth = cam.project(V, K, world, ortho=True)
    for (u, v), (px, py) in zip(uv, pixels):
        # project() returns the pixel edge convention; the grid works in pixel centres.
        assert grid.pixel_of_uv(u, v) == pytest.approx((px - 0.5, py - 0.5), abs=1e-6)
    assert (depth > 0).all()                       # the wall is in front of the camera


def test_the_grid_carries_the_frame_needed_to_reach_enu():
    facade = a_wall(normal=(0.6, 0.8), origin=(2.0, 3.0, 4.0))
    _, grid = facade_ortho(facade)
    assert grid.facade_id == facade.id
    assert grid.origin_enu == facade.origin and grid.u_axis == facade.u_axis
    assert grid.normal == facade.normal and grid.v_ground == facade.v_ground
    u, v = grid.uv_of_pixel(100, 60)
    assert facade.point(u, v) == pytest.approx(
        np.asarray(grid.origin_enu) + u * np.asarray(grid.u_axis)
        + (v - grid.origin_enu[2]) * np.array([0, 0, 1.0]))


def test_a_facade_too_big_for_one_image_says_what_would_fit():
    facade = a_wall(u_min=-150.0, u_max=150.0)
    with pytest.raises(FacadeOrthoError, match="cm/px or coarser"):
        facade_ortho(facade, FacadeOrthoSpec(gsd_cm=2))
    req, _ = facade_ortho(facade, FacadeOrthoSpec(gsd_cm=10))      # the hint has to work
    assert req.width <= 4096 and req.height <= 4096


def test_a_tiny_facade_still_gets_a_usable_image():
    req, _ = facade_ortho(a_wall(u_min=0.0, u_max=0.2, height=0.2),
                          FacadeOrthoSpec(gsd_cm=20, margin_m=0.0))
    assert req.width >= 16 and req.height >= 16


# ── addressing ───────────────────────────────────────────────────────────
def openings(spec, *, w=1.3, h=1.5):
    return [Opening(id=f"ap_{n:02d}", u_center=u, v_center=v, width=w, height=h)
            for n, (u, v) in enumerate(spec)]


def a_grid(cols=(-7.0, -2.4, 2.2, 6.8), rows=(1.2, 4.3, 7.4)):
    return [(u, v) for v in rows for u in cols]


def test_a_regular_facade_resolves_to_rows_and_columns():
    got = address_facade(openings(a_grid()), v_ground=0.0)
    assert [r.index for r in got.rows] == [0, 1, 2]
    assert len(got.column_centres) == 4
    assert got.floor_height_m == pytest.approx(3.1, abs=0.05)
    assert not got.ambiguous and got.reasons == []
    # Columns count left to right for an observer outside facing the wall (§4), which
    # is the direction u grows in.
    assert got.column_centres == sorted(got.column_centres)
    assert got.of["ap_00"] == (0, 1, 1)
    assert got.of["ap_11"] == (2, 4, 4)


def test_column_and_ordinal_differ_when_a_window_is_missing():
    # Nothing in column 3 on the upper floor: "the third window" is then column 4. Both
    # numberings are kept so a resolver can offer each reading instead of picking one.
    spec = a_grid(rows=(1.2,)) + [(-7.0, 4.3), (2.2, 4.3), (6.8, 4.3)]
    got = address_facade(openings(spec), v_ground=0.0)
    upper = sorted(v for v in got.of.values() if v[0] == 1)
    assert upper == [(1, 1, 1), (1, 3, 2), (1, 4, 3)]


def test_the_ground_row_decides_floor_zero():
    got = address_facade(openings(a_grid()), v_ground=0.0)
    assert got.rows[0].v_bottom == pytest.approx(0.45)
    assert [r.index for r in got.rows] == [0, 1, 2]
    # In Ukraine the ground floor is the 1st, and `floor_offset` only moves the label.
    assert [floor_label(r.index) for r in got.rows] == ["1", "2", "3"]
    assert [floor_label(r.index, floor_offset=1) for r in got.rows] == ["2", "3", "4"]


def test_a_missing_ground_row_is_flagged_rather_than_guessed():
    # Trees or a fence hide the bottom row: the numbering is then a guess from the first
    # visible row up, and §6.3 refuses to confirm such a target without a human check.
    got = address_facade(openings(a_grid(rows=(4.3, 7.4))), v_ground=0.0)
    assert got.ambiguous and "ряд біля землі" in " ".join(got.reasons)
    assert got.rows[0].index == 0               # still numbered, but not to be trusted


def test_irregular_storey_intervals_are_flagged():
    got = address_facade(openings(a_grid(rows=(1.2, 4.3, 12.0))), v_ground=0.0)
    assert got.ambiguous and "інтервал" in " ".join(got.reasons)


def test_a_tall_ground_floor_is_not_by_itself_suspicious():
    # Shops make the first storey taller; that alone must not block every target above.
    got = address_facade(openings(a_grid(rows=(1.5, 6.0, 9.0, 12.0))), v_ground=0.0)
    assert not got.ambiguous, got.reasons


def test_openings_at_slightly_different_heights_still_share_a_row():
    spec = [(-7.0, 4.25), (-2.4, 4.3), (2.2, 4.38), (6.8, 4.22)]
    got = address_facade(openings(spec), v_ground=4.0)
    assert len(got.rows) == 1 and len(got.rows[0].members) == 4


def test_clustering_breaks_only_on_a_real_gap():
    assert cluster_1d([0.0, 0.1, 5.0, 5.2, 5.3], tol=0.5) == [[0, 1], [2, 3, 4]]
    assert cluster_1d([], tol=0.5) == []
    assert cluster_1d([3.0], tol=0.5) == [[0]]
    # Order of the input must not change the grouping.
    assert cluster_1d([5.3, 0.0, 5.0, 0.1, 5.2], tol=0.5) == [[1, 3], [2, 4, 0]]


def test_members_of_a_row_are_listed_left_to_right():
    got = address_facade(openings([(6.8, 4.3), (-7.0, 4.3), (2.2, 4.3)]), v_ground=4.0)
    assert [got.of[m][2] for m in got.rows[0].members] == [1, 2, 3]
    assert got.of["ap_01"][2] == 1               # u = -7.0 is the leftmost


def test_an_empty_facade_addresses_to_nothing():
    got = address_facade([], v_ground=0.0)
    assert got.rows == [] and got.of == {} and not got.ambiguous


def test_the_canonical_address_reads_as_the_doc_writes_it():
    assert canonical_address("podil", "B012", "N1", 3, 3) == "podil/B012/N1/03/03"
