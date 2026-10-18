"""Occupancy grid + ESDF (aerosplat.surface) against synthetic scenes with known answers.

As in test_semantics, dataset_v6 cannot settle whether the geometry is right, so every claim
is checked on scenes built here or against a brute-force reference. This module decides what
a planner may fly through, so the negative cases (internal reliability requirements §5) — unknown space,
NaN, empty and malformed inputs — get at least as much attention as the positive ones.
"""

import json
import math

import numpy as np
import pytest
from scipy import ndimage

from aerosplat.semantics.terrain import build_terrain
from aerosplat.surface.esdf import distance_field_cm
from aerosplat.surface.models import (FORMAT_VERSION, FREE, OCCUPANCY_JSON, OCCUPIED, OUTSIDE,
                                      UNKNOWN, ColumnSource, GridSpec, parse_bbox)
from aerosplat.surface.occupancy import (OccupancyGrid, build_occupancy, read_occupancy,
                                         write_occupancy)

RNG = np.random.default_rng(11)

# A box building whose walls deliberately miss the voxel lattice (x0 of the grid is -30).
X0, X1, Y0, Y1, H = -9.8, 10.35, -6.2, 5.9, 10.0


# ── scene builders ───────────────────────────────────────────────────────
def ground(step: float, half: float = 30.0, z: float = 0.0, holes=()) -> np.ndarray:
    """A flat grid of points, cell-centred so no point sits on a voxel face; `holes` are
    (xmin, ymin, xmax, ymax) rectangles left without any point."""
    g = np.arange(-half + step / 2, half, step)
    gx, gy = np.meshgrid(g, g)
    p = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=1)
    for xmin, ymin, xmax, ymax in holes:
        p = p[~((p[:, 0] > xmin) & (p[:, 0] < xmax) & (p[:, 1] > ymin) & (p[:, 1] < ymax))]
    return p


def box(step: float, x0=X0, x1=X1, y0=Y0, y1=Y1, h=H) -> np.ndarray:
    """Points on the four walls (up to the roof) and the flat roof of an axis-aligned building."""
    zs = np.append(np.arange(0.0, h - 1e-9, step), h)
    pts = []
    for x in np.append(np.arange(x0, x1, step), x1):
        pts += [(x, y0, z) for z in zs] + [(x, y1, z) for z in zs]
    for y in np.append(np.arange(y0, y1, step), y1):
        pts += [(x0, y, z) for z in zs] + [(x1, y, z) for z in zs]
    rx, ry = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    roof = np.stack([rx.ravel(), ry.ravel(), np.full(rx.size, h)], axis=1)
    return np.concatenate([np.array(pts, float), roof])


def box_scene(res: float, holes=(), extra=None) -> np.ndarray:
    """Ground + building, sampled at half a voxel (well sampled: every column has points)."""
    step = res / 2
    g = ground(step, holes=holes)
    g = g[~((g[:, 0] > X0) & (g[:, 0] < X1) & (g[:, 1] > Y0) & (g[:, 1] < Y1))]
    parts = [g, box(step)] + ([np.asarray(extra, float)] if extra is not None else [])
    return np.concatenate(parts)


def build(xyz, res=0.5, **kw):
    t = build_terrain(xyz, res_m=1.0)
    return build_occupancy(xyz, t.model, t.dsm, t.dtm, res_m=res, **kw)


@pytest.fixture(scope="module", params=[0.25, 0.5], ids=["r25", "r50"])
def box_grid(request):
    res = request.param
    grid, stats, warnings = build(box_scene(res), res=res)
    return grid, res


def box_distance(p) -> np.ndarray:
    """Exact distance from points outside the building to the building or the ground (z=0)."""
    p = np.atleast_2d(p)
    dx = np.maximum(np.maximum(X0 - p[:, 0], p[:, 0] - X1), 0.0)
    dy = np.maximum(np.maximum(Y0 - p[:, 1], p[:, 1] - Y1), 0.0)
    dz = np.maximum(p[:, 2] - H, 0.0)
    return np.minimum(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2), p[:, 2])


def face_east(grid) -> float:
    """East face of the voxel that holds the east wall: what a conservative map can promise."""
    r = grid.res_m
    return grid.spec.x0 + (math.floor((X1 - grid.spec.x0) / r) + 1) * r


# ── the box: states and distances ────────────────────────────────────────
def test_box_states(box_grid):
    grid, r = box_grid
    s = grid.state_at([[0.0, 0.0, 5.0],            # inside the building: never seen
                       [X1, 0.0, 5.0],             # on the wall
                       [X1 + 3.0, 0.0, 5.0],       # in front of it
                       [0.0, 0.0, H + 2.0],        # above the roof
                       [20.0, 20.0, -0.6],         # underground, below the grid
                       [20.0, 20.0, 0.01]])        # in the ground voxel
    assert list(s) == [UNKNOWN, OCCUPIED, FREE, FREE, OCCUPIED, OCCUPIED]
    assert grid.source_at([[0.0, 0.0, 0.0]])[0] == ColumnSource.POINTS


def test_box_distance_field_within_half_a_voxel(box_grid):
    """At voxel centres, the field is the distance to the wall (or roof) within r/2."""
    grid, r = box_grid
    s = grid.spec
    xc = s.x0 + (np.arange(s.cols) + 0.5) * r
    yc = s.y0 + (np.arange(s.rows) + 0.5) * r
    zc = s.z0 + (np.arange(s.layers) + 0.5) * r
    checked = 0
    # In front of each wall, away from its edges, closer to the wall than to the ground.
    walls = [("E", xc[(xc > X1 + r) & (xc < X1 + 4.0)], lambda x: x - X1, "x"),
             ("W", xc[(xc < X0 - r) & (xc > X0 - 4.0)], lambda x: X0 - x, "x"),
             ("N", yc[(yc > Y1 + r) & (yc < Y1 + 4.0)], lambda y: y - Y1, "y"),
             ("S", yc[(yc < Y0 - r) & (yc > Y0 - 4.0)], lambda y: Y0 - y, "y")]
    zs = zc[(zc > 4.5) & (zc < H - 4.5)]
    for _, along, truth, axis in walls:
        if axis == "x":
            others = yc[(yc > Y0 + 4.5) & (yc < Y1 - 4.5)]
            pts = np.array([(a, b, z) for a in along for b in others for z in zs])
            t = truth(pts[:, 0])
        else:
            others = xc[(xc > X0 + 4.5) & (xc < X1 - 4.5)]
            pts = np.array([(b, a, z) for a in along for b in others for z in zs])
            t = truth(pts[:, 1])
        i, j, k = s.index_of(pts)
        e = grid.esdf_cm[i, j, k] * 0.01
        assert np.abs(e - t).max() <= r / 2 + 0.01
        checked += len(pts)
    # Above the roof, away from its edges.
    pts = np.array([(x, y, z) for x in xc[(xc > X0 + 5) & (xc < X1 - 5)]
                    for y in yc[(yc > Y0 + 5) & (yc < Y1 - 5)] for z in zc[(zc > H + r) & (zc < H + 4)]])
    i, j, k = s.index_of(pts)
    assert np.abs(grid.esdf_cm[i, j, k] * 0.01 - (pts[:, 2] - H)).max() <= r / 2 + 0.01
    assert checked + len(pts) > 100


def test_box_clearance_never_exceeds_the_truth(box_grid):
    grid, r = box_grid
    p = np.column_stack([RNG.uniform(-25, 25, 4000), RNG.uniform(-25, 25, 4000),
                         RNG.uniform(0.3, H + 6, 4000)])
    outside = ~((p[:, 0] >= X0) & (p[:, 0] <= X1) & (p[:, 1] >= Y0) & (p[:, 1] <= Y1)
                & (p[:, 2] <= H))
    p = p[outside]
    truth = box_distance(p)
    c = grid.clearance(p)
    assert (c <= truth + 1e-9).all()
    # Within a voxel of the truth near obstacles (exact cube distance: a voxel can stick out
    # of the true surface by up to r along each axis, so √3 r at a corner) ...
    near = truth < 5 * r
    assert (c[near] >= truth[near] - math.sqrt(3) * r - 1e-6).all()
    # ... and never more than the documented ~2 voxels farther away. (Only where the scene's
    # own obstacles are nearer than the edge of the grid, which `truth` does not know about.)
    mid = truth < 4.0
    assert (c[mid] >= truth[mid] - 2.6 * r).all()


def test_wall_clearance_is_the_distance_to_the_wall_voxel(box_grid):
    grid, r = box_grid
    face = face_east(grid)
    for d in (0.3, 1.0, 2.5, 4.0):
        c = grid.clearance([X1 + d, 0.0, 6.0], exact_within_m=5.0)[0]
        assert c == pytest.approx(X1 + d - face, abs=1e-9)
        assert 0 <= d - c < r


def test_clearance_is_zero_in_blocked_and_unknown_space(box_grid):
    grid, _ = box_grid
    c = grid.clearance([[0.0, 0.0, 5.0], [X1, 0.0, 3.0], [20.0, 20.0, -2.0],
                        [500.0, 0.0, 5.0], [0.0, -500.0, 50.0]])
    assert (c == 0).all()
    assert list(grid.state_at([[500.0, 0.0, 5.0]])) == [OUTSIDE]


# ── paths ────────────────────────────────────────────────────────────────
def test_corridor_through_a_wall_has_zero_clearance(box_grid):
    grid, r = box_grid
    start = X1 + 5.0
    pc = grid.path_clearance([[start, 0.0, 5.0], [X1 - 3.0, 0.0, 5.0]])
    assert pc.blocked and pc.min_clearance_m == 0.0
    assert pc.first_blocked_state == "occupied"
    assert pc.first_blocked_s == pytest.approx(start - face_east(grid), abs=1e-9)


def test_corridor_along_a_wall_keeps_its_clearance(box_grid):
    grid, r = box_grid
    pc = grid.path_clearance([[X1 + 3.0, -4.0, 6.0], [X1 + 3.0, 4.0, 6.0]], exact_within_m=4.0)
    assert not pc.blocked and pc.first_blocked_state is None
    assert 3.0 - r - pc.step_m / 2 - 1e-9 <= pc.min_clearance_m <= 3.0
    assert pc.step_m <= r / 2 + 1e-12
    assert pc.assumed_free_s is None                    # every column here was measured


def test_descent_onto_the_roof_is_blocked_at_the_roof(box_grid):
    grid, r = box_grid
    pc = grid.path_clearance([[0.0, 0.0, H + 6.0], [0.0, 0.0, H + 2.0], [0.0, 0.0, H - 1.0]])
    assert pc.blocked and pc.first_blocked_state == "occupied"
    roof_top = grid.spec.z0 + (math.floor((H - grid.spec.z0) / r) + 1) * r
    assert pc.first_blocked_s == pytest.approx(H + 6.0 - roof_top, abs=1e-9)


def test_path_cannot_slip_past_a_thin_pole():
    """A one-voxel pole that the straight segment only clips at a corner still blocks it."""
    res = 0.5
    pole = np.array([(5.3, 5.3, z) for z in np.arange(0.0, 8.0, 0.1)])
    grid, _, _ = build(np.concatenate([ground(res / 2), pole]), res=res)
    i, j, _ = grid.spec.index_of([[5.3, 5.3, 1.0]])
    cx, cy = grid.spec.x0 + (j[0] + 1) * res, grid.spec.y0 + (i[0] + 1) * res   # NE corner
    # A diagonal that passes 1 cm inside that corner.
    a = np.array([cx - 5.0, cy + 5.0 - 0.02, 3.0])
    b = np.array([cx + 5.0, cy - 5.0 - 0.02, 3.0])
    pc = grid.path_clearance([a, b])
    assert pc.blocked and pc.min_clearance_m == 0.0
    # Moved 30 cm off the corner, it passes, with about that much clearance.
    off = np.array([0.15, 0.15, 0.0])
    pc = grid.path_clearance([a + off, b + off])
    assert not pc.blocked
    assert 0.0 < pc.min_clearance_m <= 0.3 / math.sqrt(2) * 2


@pytest.mark.parametrize("bad, match", [
    ([[0.0, 0.0, 5.0]], "two points"),
    ([[0.0, 0.0, 5.0], [np.nan, 0.0, 5.0]], "NaN"),
    ([[0.0, 0.0, 5.0], [1.0, 0.0, np.inf]], "NaN"),
    (np.zeros((0, 3)), "empty"),
    ([[0.0, 0.0], [1.0, 1.0]], "expected"),
])
def test_path_rejects_bad_input(box_grid, bad, match):
    grid, _ = box_grid
    with pytest.raises(ValueError, match=match):
        grid.path_clearance(bad)


@pytest.mark.parametrize("step", [0.0, -0.1, 0.3, float("nan")])
def test_path_sampling_is_never_coarser_than_half_a_voxel(step):
    grid, _, _ = build(box_scene(0.5), res=0.5)
    with pytest.raises(ValueError, match="half a voxel"):
        grid.path_clearance([[15.0, 0.0, 5.0], [15.0, 5.0, 5.0]], step_m=step)


def test_degenerate_path_inside_a_wall_is_blocked(box_grid):
    grid, _ = box_grid
    pc = grid.path_clearance([[X1, 0.0, 5.0], [X1, 0.0, 5.0]])
    assert pc.blocked and pc.first_blocked_s == 0.0 and pc.length_m == 0.0


# ── rays ─────────────────────────────────────────────────────────────────
def test_ray_towards_a_wall(box_grid):
    grid, r = box_grid
    start = X1 + 6.0
    hits = grid.raycast([[start, 1.0, 4.0]], [[-1.0, 0.0, 0.0]], 50.0)
    assert hits.hit[0] and hits.state[0] == OCCUPIED
    assert hits.distance[0] == pytest.approx(start - face_east(grid), abs=1e-9)
    assert 6.0 - r < hits.distance[0] <= 6.0


def test_ray_directions_need_not_be_unit(box_grid):
    grid, _ = box_grid
    a = grid.raycast([[X1 + 6.0, 1.0, 4.0]], [[-1.0, 0.0, 0.0]], 50.0).distance
    b = grid.raycast([[X1 + 6.0, 1.0, 4.0]], [[-7.5, 0.0, 0.0]], 50.0).distance
    assert a == pytest.approx(b, abs=1e-12)


def test_ray_cases(box_grid):
    grid, r = box_grid
    hits = grid.raycast([[0.0, 0.0, 5.0],            # starts inside the building
                         [0.0, 0.0, H + 2.0],        # straight up, nothing there
                         [25.0, 25.0, 3.0],          # leaves the grid sideways
                         [20.0, 20.0, 5.0],          # straight down onto the ground
                         [X1 + 15.0, 1.0, 4.0]],     # towards the wall, out of range
                        [[1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, -1], [-1, 0, 0]],
                        [50.0, 100.0, 100.0, 100.0, 10.0])
    assert hits.distance[0] == 0.0 and hits.state[0] == UNKNOWN
    assert math.isinf(hits.distance[1]) and hits.state[1] == FREE
    assert hits.state[2] == OUTSIDE
    assert hits.distance[2] == pytest.approx(grid.spec.x1 - 25.0, abs=1e-9)
    ground_top = grid.spec.z0 + (math.floor((0.0 - grid.spec.z0) / r) + 1) * r
    assert hits.state[3] == OCCUPIED and hits.distance[3] == pytest.approx(5.0 - ground_top)
    assert not hits.hit[4]


def _march(grid, o, d, max_range, step=1e-3):
    """Reference: walk the ray in tiny steps and report the first blocked sample."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    t = np.arange(0.0, max_range, step)
    s = grid.state_at(np.asarray(o, float) + t[:, None] * d)
    k = np.flatnonzero(s != FREE)
    return (math.inf, FREE) if len(k) == 0 else (float(t[k[0]]), int(s[k[0]]))


def test_ray_matches_a_brute_force_march(box_grid):
    grid, _ = box_grid
    rng = np.random.default_rng(3)
    o = np.column_stack([rng.uniform(-28, 28, 40), rng.uniform(-28, 28, 40), rng.uniform(0.6, 18, 40)])
    o = o[grid.state_at(o) == FREE]
    d = rng.normal(size=(len(o), 3))
    hits = grid.raycast(o, d, 40.0)
    for n in range(len(o)):
        t_ref, s_ref = _march(grid, o[n], d[n], 40.0)
        if math.isinf(t_ref):
            assert not hits.hit[n]
        else:
            assert hits.distance[n] == pytest.approx(t_ref, abs=2e-3)
            assert hits.state[n] == s_ref


@pytest.mark.parametrize("o, d, rng, match", [
    ([[0.0, 0.0, 12.0]], [[0.0, 0.0, 0.0]], 10.0, "zero length"),
    ([[np.nan, 0.0, 12.0]], [[0.0, 0.0, 1.0]], 10.0, "NaN"),
    ([[0.0, 0.0, 12.0]], [[0.0, 0.0, 1.0]], 0.0, "max_range"),
    ([[0.0, 0.0, 12.0]], [[0.0, 0.0, 1.0]], float("inf"), "max_range"),
    ([[0.0, 0.0, 12.0], [1, 1, 12]], [[0.0, 0.0, 1.0], [0, 1, 0], [1, 0, 0]], 10.0, "origins"),
])
def test_ray_rejects_bad_input(box_grid, o, d, rng, match):
    grid, _ = box_grid
    with pytest.raises(ValueError, match=match):
        grid.raycast(o, d, rng)


@pytest.mark.parametrize("bad", [[np.nan, 0, 0], [[0, 0, 1], [0, np.inf, 1]], [[1, 2]], []])
def test_point_queries_reject_bad_input(box_grid, bad):
    grid, _ = box_grid
    with pytest.raises(ValueError):
        grid.clearance(bad)
    with pytest.raises(ValueError):
        grid.state_at(bad)


# ── unknown space ────────────────────────────────────────────────────────
HOLE = (12.0, -24.0, 26.0, -10.0)            # 14 x 14 m without a single point
HOLE_C = (19.0, -17.0)


def test_unknown_space_blocks_a_low_corridor():
    grid, stats, warnings = build(box_scene(0.5, holes=[HOLE]), res=0.5)
    assert grid.source_at([[*HOLE_C, 0.0]])[0] == ColumnSource.SKYLINE
    assert grid.state_at([[*HOLE_C, 3.0]])[0] == UNKNOWN
    assert grid.clearance([[*HOLE_C, 3.0]])[0] == 0.0
    pc = grid.path_clearance([[HOLE_C[0], 0.0, 3.0], [HOLE_C[0], -28.0, 3.0]])
    assert pc.blocked and pc.first_blocked_state == "unknown" and pc.min_clearance_m == 0.0
    # The hole is unknown, never free, right down to the ground.
    i, j, _ = grid.spec.index_of([[*HOLE_C, 0.0]])
    assert not (grid.state[i[0], j[0]] == FREE)[: int((H + 1.0 - grid.spec.z0) / 0.5)].any()
    assert any("лінії дахів" in w for w in warnings)
    assert stats.columns["skyline"] > 0 and stats.unknown_ceiling_z == pytest.approx(H + 1.0)


def test_skyline_mode_only_assumes_free_space_above_the_skyline():
    grid, _, _ = build(box_scene(0.5, holes=[HOLE]), res=0.5, unknown="skyline")
    high = H + 1.0 + 1.0                        # above the skyline + margin
    pc = grid.path_clearance([[HOLE_C[0], 0.0, high], [HOLE_C[0], -28.0, high]])
    assert not pc.blocked
    assert pc.assumed_free_s is not None        # ... and says the free space is assumed
    assert grid.state_at([[*HOLE_C, H + 0.9]])[0] == UNKNOWN
    assert grid.state_at([[*HOLE_C, 1000.0]])[0] == FREE      # above the grid: top layer


def test_sky_mode_blocks_unknown_columns_at_every_altitude():
    grid, stats, warnings = build(box_scene(0.5, holes=[HOLE]), res=0.5, unknown="sky")
    assert grid.source_at([[*HOLE_C, 0.0]])[0] == ColumnSource.SKY
    for z in (3.0, H + 2.0, 1000.0):
        assert grid.state_at([[*HOLE_C, z]])[0] == UNKNOWN
    pc = grid.path_clearance([[HOLE_C[0], 0.0, H + 2.0], [HOLE_C[0], -28.0, H + 2.0]])
    assert pc.blocked and pc.first_blocked_state == "unknown"
    assert stats.unknown_ceiling_z is None
    assert any("на всю висоту" in w for w in warnings)


def test_small_hole_is_filled_conservatively_and_flagged():
    hole = (14.0, 14.0, 18.0, 18.0)             # 4 x 4 m: within fill_radius_m = 5
    centre = [16.0, 16.0]
    grid, stats, _ = build(box_scene(0.5, holes=[hole]), res=0.5)
    assert grid.source_at([[*centre, 0.0]])[0] == ColumnSource.FILLED
    # Filled to the highest neighbour (ground, 0) + 1 m margin, and unknown below that.
    assert float(grid.column_top[grid.spec.index_of([[*centre, 0]])[0][0],
                                 grid.spec.index_of([[*centre, 0]])[1][0]]) == pytest.approx(1.0)
    assert grid.state_at([[*centre, 0.6]])[0] == UNKNOWN
    pc = grid.path_clearance([[16.0, 10.0, 3.0], [16.0, 22.0, 3.0]])
    assert not pc.blocked and pc.assumed_free_s is not None
    assert stats.columns["filled"] > 0

    # Without gap filling the same hole is unknown up to the skyline.
    grid0, _, _ = build(box_scene(0.5, holes=[hole]), res=0.5, fill_radius_m=0.0)
    assert grid0.source_at([[*centre, 0.0]])[0] == ColumnSource.SKYLINE
    assert grid0.path_clearance([[16.0, 10.0, 3.0], [16.0, 22.0, 3.0]]).blocked


def sparse_box(extra=None) -> np.ndarray:
    """Ground and building sampled every 1.5 m: most 0.5 m voxel columns hold no point."""
    g = ground(1.5)
    g = g[~((g[:, 0] > X0) & (g[:, 0] < X1) & (g[:, 1] > Y0) & (g[:, 1] < Y1))]
    return np.concatenate([g, box(1.5)] + ([np.asarray(extra, float)] if extra is not None else []))


def test_sparse_cloud_never_frees_the_inside_of_a_building():
    """With most voxel columns empty, the DSM, the gap fill and the closing must still keep
    every voxel inside the true building out of free space."""
    res = 0.5
    grid, _, _ = build(sparse_box(), res=res)
    s = grid.spec
    xc = s.x0 + (np.arange(s.cols) + 0.5) * res
    yc = s.y0 + (np.arange(s.rows) + 0.5) * res
    zc = s.z0 + (np.arange(s.layers) + 0.5) * res
    jj = np.flatnonzero((xc > X0) & (xc < X1))
    ii = np.flatnonzero((yc > Y0) & (yc < Y1))
    kk = np.flatnonzero((zc > 0) & (zc < H))
    assert not (grid.state[np.ix_(ii, jj, kk)] == FREE).any()


def test_closing_fills_a_noise_pit_under_a_roof():
    """A lone MVS point below a sparsely sampled roof must not open a free shaft into it."""
    noise = [[0.3, 0.3, 4.0]]
    grid, stats, _ = build(sparse_box(extra=noise), res=0.5)
    assert grid.state_at([[0.3, 0.3, 6.0]])[0] != FREE
    assert stats.closed_columns > 0
    # The failure the closing exists for: without it the shaft is free air inside the roof.
    open_, stats0, _ = build(sparse_box(extra=noise), res=0.5, close_radius_m=0.0)
    assert open_.state_at([[0.3, 0.3, 6.0]])[0] == FREE
    assert stats0.closed_columns == 0


def test_outlier_point_makes_a_pillar_not_a_taller_grid():
    spike = [[22.25, 22.25, 80.0]]
    grid, stats, warnings = build(box_scene(0.5, extra=spike), res=0.5, skyline_quantile=0.999)
    assert grid.spec.z1 < 30.0                          # the grid follows the roofs ...
    assert stats.columns["pillar"] == 1
    for z in (5.0, 29.0, 500.0):                        # ... and the spike blocks to the sky
        assert grid.state_at([[22.25, 22.25, z]])[0] == UNKNOWN
    assert grid.clearance([[23.25, 22.25, 200.0]])[0] < 1.0
    assert any("стелі" in w for w in warnings)


def test_above_the_grid_columns_continue_upward(box_grid):
    grid, _ = box_grid
    top = grid.spec.z0 + (grid.spec.layers - 0.5) * grid.res_m
    p_top, p_high = [15.0, 0.0, top], [15.0, 0.0, top + 300.0]
    assert grid.state_at([p_high])[0] == FREE
    # The bound above the grid is the one at the top layer: valid, if pessimistic.
    assert grid.clearance(p_high)[0] == pytest.approx(grid.clearance(p_top)[0])
    assert grid.clearance(p_high)[0] <= box_distance(np.array([p_high]))[0]


# ── brute-force references on random grids ───────────────────────────────
def random_grid(seed: int, rows=13, cols=11, layers=9, res=0.5) -> OccupancyGrid:
    """A random 2.5-D grid (heightfield of blocked voxels, a few UNKNOWN/OCCUPIED mixed)."""
    rng = np.random.default_rng(seed)
    heights = rng.integers(0, layers, size=(rows, cols))
    heights[rng.random((rows, cols)) < 0.08] = layers - 1        # a few pillars
    k = np.arange(layers)[None, None, :]
    state = np.where(k <= heights[:, :, None], UNKNOWN, FREE).astype(np.uint8)
    state[(state == UNKNOWN) & (rng.random(state.shape) < 0.5)] = OCCUPIED
    spec = GridSpec(res_m=res, x0=-3.0, y0=2.0, z0=-1.0, rows=rows, cols=cols, layers=layers)
    esdf = distance_field_cm(state != FREE, res, 50.0)
    return OccupancyGrid(spec=spec, state=state, esdf_cm=esdf,
                         column_top=np.full((rows, cols), np.nan, np.float32),
                         column_source=np.full((rows, cols), ColumnSource.POINTS, np.uint8),
                         free_from=(heights + 1).astype(np.int16), max_distance_m=50.0)


def brute_clearance(grid: OccupancyGrid, p: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest blocked cube, over a padded lattice that
    holds the outside-the-grid rules (unknown beside, underground below, columns above)."""
    s = grid.spec
    pad = max(s.rows, s.cols, s.layers) + 2
    ii, jj, kk = np.meshgrid(np.arange(-pad, s.rows + pad), np.arange(-pad, s.cols + pad),
                             np.arange(-pad, s.layers + 3 * pad), indexing="ij")
    ii, jj, kk = ii.ravel(), jj.ravel(), kk.ravel()
    blocked = grid._codes(ii, jj, kk) != FREE
    c = s.centre_of(ii[blocked], jj[blocked], kk[blocked])
    out = np.empty(len(p))
    for n, q in enumerate(p):
        gap = np.maximum(np.abs(c - q) - s.res_m / 2, 0.0)
        out[n] = np.sqrt((gap ** 2).sum(1)).min()
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_clearance_against_brute_force(seed):
    grid = random_grid(seed)
    s = grid.spec
    rng = np.random.default_rng(100 + seed)
    p = np.column_stack([rng.uniform(s.x0, s.x1, 300), rng.uniform(s.y0, s.y1, 300),
                         rng.uniform(s.z0, s.z1 + 3.0, 300)])
    truth = brute_clearance(grid, p)
    exact = grid.clearance(p, exact_within_m=6.0)
    coarse = grid.clearance(p, exact_within_m=0.0)
    assert (exact <= truth + 1e-9).all() and (coarse <= truth + 1e-9).all()
    assert (coarse <= exact + 1e-9).all()
    # Above the top layer centre both are bounds taken lower down (columns only continue
    # upward), so they are compared with the truth inside the grid only.
    inside = p[:, 2] <= s.z0 + (s.layers - 0.5) * s.res_m
    assert (coarse[inside] >= truth[inside] - 2.1 * s.res_m - 0.01).all()
    # Nothing in this small grid is more than 6 m from a blocked voxel: all exact.
    assert np.allclose(exact[inside], truth[inside], atol=1e-9)


@pytest.mark.parametrize("seed", [0, 1])
def test_raycast_against_brute_force_on_random_grids(seed):
    grid = random_grid(seed)
    s = grid.spec
    rng = np.random.default_rng(200 + seed)
    o = np.column_stack([rng.uniform(s.x0, s.x1, 80), rng.uniform(s.y0, s.y1, 80),
                         rng.uniform(s.z0, s.z1 + 2.0, 80)])
    d = rng.normal(size=(80, 3))
    hits = grid.raycast(o, d, 15.0)
    for n in range(len(o)):
        t_ref, s_ref = _march(grid, o[n], d[n], 15.0)
        if math.isinf(t_ref):
            assert not hits.hit[n]
        else:
            assert hits.distance[n] == pytest.approx(t_ref, abs=2e-3)
            assert hits.state[n] == s_ref


# ── the distance field itself ────────────────────────────────────────────
def reference_field(blocked: np.ndarray, res: float, limit: float, signed=False) -> np.ndarray:
    """One-piece scipy EDT with the boundary rules as explicit padding (no tiling)."""
    pad = int(math.ceil(limit / res)) + 2
    big = np.pad(blocked, ((pad, pad), (pad, pad), (1, 0)), constant_values=True)
    d = ndimage.distance_transform_edt(~big, sampling=res)
    if signed:
        d = np.where(big, -ndimage.distance_transform_edt(big, sampling=res), d)
    d = d[pad:-pad, pad:-pad, 1:]
    return np.floor(np.clip(d, -limit, limit) * 100.0)


@pytest.mark.parametrize("limit", [100.0, 6.0, 1.3])
@pytest.mark.parametrize("signed", [False, True])
def test_tiled_field_equals_one_piece(limit, signed):
    rng = np.random.default_rng(5)
    heights = rng.integers(0, 15, size=(47, 53))
    blocked = np.arange(18)[None, None, :] <= heights[:, :, None]
    blocked[rng.random(blocked.shape) < 0.002] = True              # floating specks too
    tiled = distance_field_cm(blocked, 0.5, limit, signed=signed, tile_voxels=3000)
    assert tiled.dtype == (np.int16 if signed else np.uint16)
    assert np.array_equal(tiled.astype(np.int64), reference_field(blocked, 0.5, limit, signed))


def test_field_reads_a_state_array_like_its_mask():
    """The build passes the state array itself (FREE == 0) to save a full-size copy."""
    rng = np.random.default_rng(5)
    state = rng.choice(np.array([FREE, OCCUPIED, UNKNOWN], np.uint8), size=(9, 8, 7),
                       p=[0.8, 0.1, 0.1])
    for signed in (False, True):
        assert np.array_equal(distance_field_cm(state, 0.5, 3.0, signed=signed, tile_voxels=200),
                              distance_field_cm(state != FREE, 0.5, 3.0, signed=signed))


@pytest.mark.parametrize("kw, match", [
    (dict(blocked=np.zeros((3, 3, 3), np.float32)), "boolean or uint8"),
    (dict(blocked=np.zeros((3, 3), bool)), "3-D"),
    (dict(max_distance_m=700.0), "fit"),
    (dict(max_distance_m=400.0, signed=True), "fit"),
    (dict(max_distance_m=0.0), "positive"),
    (dict(res_m=float("nan")), "positive"),
])
def test_field_rejects_bad_input(kw, match):
    args = dict(blocked=np.zeros((3, 3, 3), bool), res_m=0.5, max_distance_m=10.0)
    args.update(kw)
    with pytest.raises(ValueError, match=match):
        distance_field_cm(args.pop("blocked"), args.pop("res_m"), args.pop("max_distance_m"),
                          **args)


# ── building: validation ─────────────────────────────────────────────────
@pytest.fixture(scope="module")
def scene():
    xyz = box_scene(0.5)
    return xyz, build_terrain(xyz, res_m=1.0)


@pytest.mark.parametrize("kw, match", [
    (dict(res_m=0.3), "divide"),
    (dict(res_m=2.0), "divide"),
    (dict(res_m=0.0), "positive"),
    (dict(res_m=-0.5), "positive"),
    (dict(headroom_m=1.5), "headroom"),
    (dict(max_distance_m=float("inf")), "positive"),
    (dict(fill_radius_m=-1.0), ">= 0"),
    (dict(close_radius_m=-0.5), ">= 0"),
    (dict(assumed_margin_m=float("nan")), ">= 0"),
    (dict(unknown="free"), "unknown"),
    (dict(skyline_quantile=0.2), "skyline_quantile"),
    (dict(max_voxels=1000), "max_voxels"),
    (dict(bbox=(100.0, 100.0, 200.0, 200.0)), "overlap"),
    (dict(bbox=(5.0, 5.0, 1.0, 9.0)), "bbox"),
])
def test_build_rejects_bad_parameters(scene, kw, match):
    xyz, t = scene
    with pytest.raises(ValueError, match=match):
        build_occupancy(xyz, t.model, t.dsm, t.dtm, **kw)


def test_build_rejects_bad_clouds_and_terrain(scene):
    xyz, t = scene
    bad = xyz.copy()
    bad[17, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        build_occupancy(bad, t.model, t.dsm, t.dtm)
    with pytest.raises(ValueError, match="empty"):
        build_occupancy(np.zeros((0, 3)), t.model, t.dsm, t.dtm)
    with pytest.raises(ValueError, match="expected"):
        build_occupancy(xyz[:, :2], t.model, t.dsm, t.dtm)
    dtm = t.dtm.copy()
    dtm[3, 4] = np.nan
    with pytest.raises(ValueError, match="DTM"):
        build_occupancy(xyz, t.model, t.dsm, dtm)
    dsm = t.dsm.copy()
    dsm[3, 4] = np.inf
    with pytest.raises(ValueError, match="DSM"):
        build_occupancy(xyz, t.model, dsm, t.dtm)
    with pytest.raises(ValueError, match="do not match"):
        build_occupancy(xyz, t.model, t.dsm[:-1], t.dtm[:-1])
    # A crop that holds no reconstructed surface at all has nothing to build on.
    wide = np.concatenate([xyz, [[200.0, 200.0, 0.0]]])
    t_wide = build_terrain(wide, res_m=1.0)
    with pytest.raises(ValueError, match="no reconstructed surface"):
        build_occupancy(wide, t_wide.model, t_wide.dsm, t_wide.dtm,
                        bbox=(100.0, 100.0, 150.0, 150.0))


def test_bbox_crops_on_the_same_lattice(scene):
    xyz, t = scene
    full, _, _ = build_occupancy(xyz, t.model, t.dsm, t.dtm)
    crop, _, _ = build_occupancy(xyz, t.model, t.dsm, t.dtm, bbox=(-15.2, -10.1, 16.3, 12.0))
    r = crop.res_m
    assert crop.spec.cols < full.spec.cols and crop.spec.rows < full.spec.rows
    for off in ((crop.spec.x0 - full.spec.x0) / r, (crop.spec.y0 - full.spec.y0) / r):
        assert off == pytest.approx(round(off), abs=1e-9)
    p = np.column_stack([RNG.uniform(-15, 16, 500), RNG.uniform(-10, 12, 500),
                         RNG.uniform(0.2, 15, 500)])
    assert np.array_equal(crop.state_at(p), full.state_at(p))
    # The crop's edge is unknown, so its distances can only be smaller. (Compared in one
    # mode: near its edge the crop switches to the exact distance where the full grid still
    # uses the looser far-field bound, and both are valid bounds.)
    assert (crop.clearance(p, exact_within_m=0.0) <= full.clearance(p, exact_within_m=0.0)
            + 1e-9).all()
    di = round((crop.spec.y0 - full.spec.y0) / r)
    dj = round((crop.spec.x0 - full.spec.x0) / r)
    assert crop.spec.z0 == full.spec.z0 and crop.spec.layers == full.spec.layers
    assert (crop.esdf_cm <= full.esdf_cm[di:di + crop.spec.rows, dj:dj + crop.spec.cols]).all()
    assert crop.state_at([[-20.0, 0.0, 5.0]])[0] == OUTSIDE


def test_build_is_deterministic(scene):
    xyz, t = scene
    a, sa, wa = build_occupancy(xyz, t.model, t.dsm, t.dtm)
    b, sb, wb = build_occupancy(xyz[::-1].copy(), t.model, t.dsm, t.dtm)    # order-free too
    assert np.array_equal(a.state, b.state) and np.array_equal(a.esdf_cm, b.esdf_cm)
    assert np.array_equal(a.column_source, b.column_source)
    assert sa == sb and wa == wb


def test_stats_add_up(box_grid):
    grid, _ = box_grid
    s = grid.spec
    assert int((grid.state == FREE).sum()) + int((grid.state == OCCUPIED).sum()) \
        + int((grid.state == UNKNOWN).sum()) == s.voxels


# ── files ────────────────────────────────────────────────────────────────
def _report(grid, stats, warnings):
    from aerosplat.surface.models import EsdfInfo, OccupancyReport

    return OccupancyReport(grid=grid.spec, bounds=grid.spec.bounds(),
                           esdf=EsdfInfo(max_distance_m=grid.max_distance_m), stats=stats,
                           warnings=warnings)


def test_files_round_trip(tmp_path):
    grid, stats, warnings = build(box_scene(0.5, holes=[HOLE]), res=0.5)
    write_occupancy(tmp_path / "occ", grid, _report(grid, stats, warnings))
    back, report = read_occupancy(tmp_path / "occ")
    assert report.grid == grid.spec and report.warnings == warnings
    for name in ("state", "esdf_cm", "column_source"):
        assert np.array_equal(getattr(back, name), getattr(grid, name))
    assert np.array_equal(back.column_top, grid.column_top, equal_nan=True)
    p = np.column_stack([RNG.uniform(-29, 29, 200), RNG.uniform(-29, 29, 200),
                         RNG.uniform(0, 20, 200)])
    assert np.array_equal(back.clearance(p), grid.clearance(p))


def test_reading_refuses_other_formats_and_mismatched_arrays(tmp_path):
    grid, stats, warnings = build(box_scene(0.5), res=0.5)
    folder = tmp_path / "occ"
    write_occupancy(folder, grid, _report(grid, stats, warnings))
    data = json.loads((folder / OCCUPANCY_JSON).read_text(encoding="utf-8"))
    data["format"] = "occupancy-v0"
    (folder / OCCUPANCY_JSON).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="format"):
        read_occupancy(folder)
    data["format"] = FORMAT_VERSION
    data["grid"]["layers"] += 1
    (folder / OCCUPANCY_JSON).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="state must be"):
        read_occupancy(folder)


def test_parse_bbox():
    assert parse_bbox(" -10, -5.5,20,7 ") == (-10.0, -5.5, 20.0, 7.0)
    for bad in ("1,2,3", "a,b,c,d", "0,0,0,5", "5,0,1,5", "0,0,nan,5", "0,0,inf,5"):
        with pytest.raises(ValueError):
            parse_bbox(bad)


# ── the pipeline stage ───────────────────────────────────────────────────
def _write_cloud(path, xyz):
    torch = pytest.importorskip("torch")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"xyz": torch.tensor(xyz, dtype=torch.float32)}, path)


def test_stage_end_to_end(tmp_path):
    from aerosplat.semantics.stage import run_semantics
    from aerosplat.surface import stage

    cloud = tmp_path / "gs" / "dense_wgs84.pt"
    _write_cloud(cloud, box_scene(0.5, holes=[HOLE]))
    run_semantics(cloud, tmp_path / "semantics")
    assert stage.main(["--cloud", str(cloud), "--semantics-dir", str(tmp_path / "semantics"),
                       "--out-dir", str(tmp_path / "occupancy"), "--res", "0.5",
                       "--bbox=-28,-28,28,28"]) == 0
    grid, report = read_occupancy(tmp_path / "occupancy")
    assert report.frame == "WGS84 Site ENU" and report.format == FORMAT_VERSION
    assert report.bounds == grid.spec.bounds()
    assert report.params["res_m"] == 0.5 and report.params["unknown"] == "skyline"
    assert report.stats.voxels == grid.spec.voxels and report.stats.columns["skyline"] > 0
    assert report.warnings and {"load", "build"} <= set(report.timings_s)
    assert report.esdf.dtype == "uint16" and not report.esdf.signed
    assert grid.path_clearance([[X1 + 5.0, 0.0, 5.0], [X1 - 3.0, 0.0, 5.0]]).blocked


def test_stage_fails_loudly(tmp_path):
    from aerosplat.semantics.stage import run_semantics
    from aerosplat.surface import stage

    cloud = tmp_path / "dense.pt"
    _write_cloud(cloud, box_scene(0.5))
    common = ["--cloud", str(cloud), "--out-dir", str(tmp_path / "occupancy")]
    with pytest.raises(SystemExit, match="semantics"):
        stage.main(common + ["--semantics-dir", str(tmp_path / "nothing")])
    run_semantics(cloud, tmp_path / "semantics")
    with pytest.raises(SystemExit, match="divide"):
        stage.main(common + ["--semantics-dir", str(tmp_path / "semantics"), "--res", "0.3"])
    with pytest.raises((SystemExit, ValueError), match="bbox"):
        stage.main(common + ["--semantics-dir", str(tmp_path / "semantics"), "--bbox=1,2,3"])
    assert not (tmp_path / "occupancy" / OCCUPANCY_JSON).exists()


def test_occupancy_stage_is_registered_after_semantics():
    from aerosplat.jobs.models import PIPELINE_STAGES
    from aerosplat.pipeline.stages import STAGES

    assert PIPELINE_STAGES[-2:] == ("semantics", "occupancy")
    assert list(STAGES) == list(PIPELINE_STAGES)
    assert STAGES["occupancy"].entry({}) is not None
    assert STAGES["occupancy"].version == FORMAT_VERSION


def test_occupancy_argv_reads_what_semantics_read(es_site):
    from aerosplat.config import RunConfig
    from aerosplat.pipeline.layout import RunLayout
    from aerosplat.pipeline.stages import STAGES, argv_occupancy, argv_semantics

    cat, site, cap, _ = es_site
    cfg = RunConfig(camera_source="earthstudio_json", cloud_source="dense",
                    occupancy_bbox="-50,-50,50,50").to_dict()
    L = RunLayout.for_site_run(cfg, site=site, site_dirs=cat.site_dirs(site), capture=cap,
                               epoch_id="baseline", run_id="r1")
    a, sem = argv_occupancy(cfg, L), argv_semantics(cfg, L)
    assert a[a.index("--cloud") + 1] == sem[sem.index("--cloud") + 1] == str(L.cloud_wgs84)
    assert a[a.index("--semantics-dir") + 1] == sem[sem.index("--out-dir") + 1]
    root = cat.run_dirs(site, "baseline", "r1").root
    assert a[a.index("--out-dir") + 1] == str(root / "occupancy")
    assert "--bbox=-50,-50,50,50" in a          # one token: argparse reads "-50,..." as a flag
    assert a[a.index("--unknown") + 1] == "skyline"
    assert STAGES["occupancy"].skip_reason(cfg) is None
    sparse = RunConfig(camera_source="earthstudio_json", cloud_source="sparse").to_dict()
    assert STAGES["occupancy"].skip_reason(sparse)


def test_optional_stages_run_only_when_named():
    from aerosplat import starter

    def stages(*argv):
        return starter.resolve_stages(starter.build_parser().parse_args(list(argv)))

    assert stages() == starter.STAGES                       # a plain run is unchanged
    assert "occupancy" not in stages("--from", "sfm")
    assert stages("--from", "semantics") == ["semantics", "occupancy"]
    assert stages("--only", "occupancy") == ["occupancy"]
    assert stages("--stages", "semantics,occupancy") == ["semantics", "occupancy"]
    assert stages("--to", "occupancy")[-2:] == ["semantics", "occupancy"]
    with pytest.raises(SystemExit):
        stages("--stages", "occupancy,bogus")


def test_config_rejects_inconsistent_occupancy_settings():
    from pydantic import ValidationError

    from aerosplat.config import RunConfig

    RunConfig(occupancy_res=0.25)
    for kw in (dict(occupancy_res=0.3), dict(occupancy_res=0.5, semantics_res=0.75),
               dict(occupancy_bbox="1,2,3"), dict(occupancy_bbox="5,0,1,5"),
               dict(occupancy_headroom=1.0), dict(occupancy_unknown="free")):
        with pytest.raises(ValidationError):
            RunConfig(**kw)


def test_products_find_the_newest_occupancy(tmp_path):
    from aerosplat.store import Catalog, RunStatus, Site
    from aerosplat.surface.products import (OccupancyMissing, find_occupancy,
                                            require_occupancy)
    from conftest import ES_ORIGIN

    cat = Catalog.open(tmp_path / "projects")
    cat.ensure_project("demo")
    site = cat.create_site(Site(id="yard", project_id="demo", origin=ES_ORIGIN))
    cat.ensure_epoch(site)
    cat.create_run(site, "baseline", "r1", capture_ids=[], config={}, earth_model="wgs84",
                   georeferenced=True)
    assert find_occupancy(cat, site) is None
    with pytest.raises(OccupancyMissing):
        require_occupancy(cat, site)

    grid, stats, warnings = build(box_scene(0.5), res=0.5)
    occ = cat.run_dirs(site, "baseline", "r1").root / "occupancy"
    write_occupancy(occ, grid, _report(grid, stats, warnings))
    cat.finish_run(site, "baseline", "r1", status=RunStatus.ok, products={"occupancy": occ})
    found = require_occupancy(cat, site)
    assert found.run.id == "r1" and found.report.grid == grid.spec
    assert np.array_equal(found.load_grid().state, grid.state)

    # A file in an older format is not served as if it were current.
    data = json.loads((occ / OCCUPANCY_JSON).read_text(encoding="utf-8"))
    data["format"] = "occupancy-v0"
    (occ / OCCUPANCY_JSON).write_text(json.dumps(data), encoding="utf-8")
    assert find_occupancy(cat, site) is None
    cat.close()
