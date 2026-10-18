"""Space carving (aerosplat.surface.carving) on synthetic scenes with known answers.

Visibility is computed here against each scene's true, opaque geometry, as a perfect MVS
fusion would record it; the tests that need wrong visibility build it on purpose. Carving is
what lets free space appear where the 2.5-D model blocks it, so most tests are negative ones:
space that rays did not see — or could not have seen — must stay blocked.
"""

import struct

import numpy as np
import pytest

from aerosplat.semantics.terrain import build_terrain
from aerosplat.surface.carving import (CameraRays, VisibilityError, _apply, _levels, _march,
                                       _segments, carve, load_colmap_rays, read_fused_visibility)
from aerosplat.surface.models import FREE, OCCUPIED, UNKNOWN, ColumnSource, GridSpec
from aerosplat.surface.occupancy import build_occupancy, read_occupancy, write_occupancy

UP = np.array([0.0, 0.0, 1.0])


# ── scenes ───────────────────────────────────────────────────────────────
def sheet(x0, x1, y0, y1, z, step, normal=UP):
    """Cell-centred points of a horizontal rectangle (none on a voxel face)."""
    xs, ys = np.arange(x0 + step / 2, x1, step), np.arange(y0 + step / 2, y1, step)
    gx, gy = np.meshgrid(xs, ys)
    p = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, float(z))], axis=1)
    return p, np.tile(np.asarray(normal, float), (len(p), 1))


def walls(lo, hi, step):
    """Points on the four vertical faces of a box, with outward normals."""
    pts, nrm = [], []
    zs = np.arange(lo[2] + step / 2, hi[2], step)
    for axis, side, value in ((0, -1, lo[0]), (0, 1, hi[0]), (1, -1, lo[1]), (1, 1, hi[1])):
        other = 1 - axis
        us = np.arange(lo[other] + step / 2, hi[other], step)
        gu, gz = np.meshgrid(us, zs)
        p = np.zeros((gu.size, 3))
        p[:, axis], p[:, other], p[:, 2] = value, gu.ravel(), gz.ravel()
        n = np.zeros(3)
        n[axis] = side
        pts.append(p)
        nrm.append(np.tile(n, (len(p), 1)))
    return np.concatenate(pts), np.concatenate(nrm)


def inside(p, lo, hi):
    return np.all((p > np.asarray(lo)) & (p < np.asarray(hi)), axis=-1)


def ring(n, radius, height, phase=0.0):
    a = phase + 2 * np.pi * np.arange(n) / n
    return np.stack([radius * np.cos(a), radius * np.sin(a), np.full(n, float(height))], axis=1)


def blocked_by(cams, pts, solids, eps=0.05):
    """(C, N): the segment camera -> point (stopped eps short) passes through a solid box."""
    o = cams[:, None, :]
    d = pts[None, :, :] - o
    d -= eps * d / np.linalg.norm(d, axis=2, keepdims=True)
    hit = np.zeros(d.shape[:2], bool)
    for lo, hi in solids:
        t0, t1 = np.zeros(d.shape[:2]), np.ones(d.shape[:2])
        with np.errstate(divide="ignore", invalid="ignore"):
            for a in range(3):
                ta = (lo[a] - o[..., a]) / d[..., a]
                tb = (hi[a] - o[..., a]) / d[..., a]
                par = d[..., a] == 0
                out = par & ((o[..., a] <= lo[a]) | (o[..., a] >= hi[a]))
                ta = np.where(par, np.where(out, np.inf, -np.inf), ta)
                tb = np.where(par, np.where(out, -np.inf, np.inf), tb)
                t0 = np.maximum(t0, np.minimum(ta, tb))
                t1 = np.minimum(t1, np.maximum(ta, tb))
        hit |= t0 < t1
    return hit


def observe(pts, nrm, cams, solids, min_views=2):
    """(cloud, rays) as an MVS fusion would give them: a point is reconstructed when at least
    `min_views` cameras see it unoccluded, and those cameras are its visibility. Points behind
    a camera's back still count as seen: these cameras look everywhere."""
    seen = ~blocked_by(cams, pts, solids)
    keep = seen.sum(axis=0) >= min_views
    cam, pt = np.nonzero(seen[:, keep])
    cloud = pts[keep]
    return cloud, CameraRays(points=cloud, centres=cams, point=pt, camera=cam, normals=nrm[keep])


def build(cloud, rays=None, res=0.5, **kw):
    t = build_terrain(cloud, res_m=1.0)
    return build_occupancy(cloud, t.model, t.dsm, t.dtm, res_m=res, rays=rays, **kw)


def centres(grid, lo, hi):
    """States of the voxels whose centres lie inside the box (lo, hi)."""
    s = grid.spec
    r = s.res_m
    x = s.x0 + (np.arange(s.cols) + 0.5) * r
    y = s.y0 + (np.arange(s.rows) + 0.5) * r
    z = s.z0 + (np.arange(s.layers) + 0.5) * r
    ii = np.flatnonzero((y > lo[1]) & (y < hi[1]))
    jj = np.flatnonzero((x > lo[0]) & (x < hi[0]))
    kk = np.flatnonzero((z > lo[2]) & (z < hi[2]))
    return grid.state[np.ix_(ii, jj, kk)]


# ── a canopy: free space under an overhang only where rays reach ─────────
CANOPY = (np.array([-5.0, -5.0, 4.0]), np.array([5.0, 5.0, 4.5]))


def canopy_scene():
    lo, hi = CANOPY
    g, gn = sheet(-20, 20, -20, 20, 0.0, 0.5)
    top, tn = sheet(lo[0], hi[0], lo[1], hi[1], hi[2], 0.5)
    side, sn = walls(lo, hi, 0.25)
    return np.concatenate([g, top, side]), np.concatenate([gn, tn, sn])


LOW = np.concatenate([ring(24, 25.0, 6.0), ring(24, 18.0, 10.0, phase=0.13)])   # 13-29 deg
HIGH = np.concatenate([ring(12, 4.0, 60.0), [[0.0, 0.0, 70.0]]])                 # >= 86 deg


@pytest.fixture(scope="module")
def canopy_low():
    pts, nrm = canopy_scene()
    cloud, rays = observe(pts, nrm, np.concatenate([LOW, HIGH]), [CANOPY])
    return cloud, rays, build(cloud, rays)


def test_rays_open_space_under_a_canopy(canopy_low):
    cloud, rays, (grid, stats, _) = canopy_low
    lo, hi = CANOPY
    # Ground under the canopy was seen from the side, so it is in the cloud ...
    assert inside(cloud, [-4, -4, -0.1], [4, 4, 0.1]).sum() > 200
    # ... and the air between it and the canopy is carved where the rays went.
    band = centres(grid, [-4, -4, 1.0], [4, 4, 2.0])
    assert (band == FREE).mean() > 0.9
    assert stats.carving is not None and stats.carving.carved_under_surface > 0
    # The canopy and the voxel layer right under it stay blocked (erosion, then the slab).
    assert not (centres(grid, [lo[0] + 0.5, lo[1] + 0.5, 3.5], [hi[0] - 0.5, hi[1] - 0.5, 4.5])
                == FREE).any()
    # A flight under the canopy is clear, and the free space it uses is evidence.
    pc = grid.path_clearance([[-12.0, 0.3, 1.6], [0.0, 0.3, 1.6]])
    assert not pc.blocked and pc.assumed_free_s is None
    assert not grid.assumed_at([[0.0, 0.3, 1.6]])[0]
    # Through the canopy is still blocked.
    assert grid.path_clearance([[0.3, 0.3, 8.0], [0.3, 0.3, 2.0]]).blocked


def test_ground_under_a_canopy_is_unknown_not_matter(canopy_low):
    """The DTM runs over the canopy (a 10 m slab reads as terrain), but points were seen on
    the ground beneath it: matter is asserted only below them. Above, the 2.5-D model keeps
    the space blocked — as unknown, which rays may open, not as occupied, which they may not."""
    cloud, _, _ = canopy_low
    t = build_terrain(cloud, res_m=1.0)
    m = t.model
    assert t.dtm[int(-m.y0), int(-m.x0)] > 4.0           # the DTM is wrong here, on purpose
    plain, _, _ = build(cloud)
    assert (centres(plain, [-4, -4, 0.5], [4, 4, 4.0]) == UNKNOWN).all()
    assert (centres(plain, [-4, -4, -0.5], [4, 4, 0.0]) == OCCUPIED).all()
    assert plain.clearance([[0.3, 0.3, 2.0]])[0] == 0.0


def test_no_rays_under_the_canopy_no_free_space():
    """Near-vertical cameras cannot see under a 4 m canopy: nothing there is carved."""
    pts, nrm = canopy_scene()
    cloud, rays = observe(pts, nrm, HIGH, [CANOPY])
    grid, stats, _ = build(cloud, rays)
    lo, hi = CANOPY
    assert stats.carving is not None and stats.carving.rays_used > 0
    assert not (centres(grid, [lo[0] + 1, lo[1] + 1, 0.0], [hi[0] - 1, hi[1] - 1, lo[2]])
                == FREE).any()


def test_under_surface_carving_can_be_switched_off(canopy_low):
    cloud, rays, (grid, _, _) = canopy_low
    off, stats, _ = build(cloud, rays, carve_under_surface=False)
    assert stats.carving is not None and stats.carving.carved_under_surface == 0
    lo, hi = CANOPY
    assert not (centres(off, [lo[0] + 0.5, lo[1] + 0.5, 0.0], [hi[0] - 0.5, hi[1] - 0.5, lo[2]])
                == FREE).any()
    # The 2.5-D grid is what is left: identical to building without rays at all.
    plain, _, _ = build(cloud)
    assert np.array_equal(off.state, plain.state)


# ── occlusion: only the cameras that saw a point carve towards it ─────────
BUILDING = (np.array([-4.0, -4.0, 0.0]), np.array([4.0, 4.0, 8.0]))
CAMS = np.concatenate([ring(24, 30.0, 12.0), ring(8, 6.0, 50.0)])


def building_scene(step):
    lo, hi = BUILDING
    g, gn = sheet(-20, 20, -20, 20, 0.0, 0.5)
    keep = ~inside(g, [lo[0], lo[1], -1], [hi[0], hi[1], 1])
    roof, rn = sheet(lo[0], hi[0], lo[1], hi[1], hi[2], step)
    parts = [(g[keep], gn[keep]), (roof, rn)]
    if step <= 0.5:
        parts.append(walls(lo, hi, step))
    return np.concatenate([p for p, _ in parts]), np.concatenate([n for _, n in parts])


def frustum_visibility(cloud, rays):
    """What carving must NOT do: every camera 'sees' every point it can frame."""
    cam, pt = np.meshgrid(np.arange(len(rays.centres)), np.arange(len(cloud)), indexing="ij")
    return CameraRays(points=cloud, centres=rays.centres, point=pt.ravel(), camera=cam.ravel(),
                      normals=rays.normals)


@pytest.mark.parametrize("step", [0.5, 1.5], ids=["walls", "sparse-roof"])
def test_occluded_points_never_carve_through_a_building(step):
    pts, nrm = building_scene(step)
    cloud, rays = observe(pts, nrm, CAMS, [BUILDING])
    grid, stats, _ = build(cloud, rays)
    lo, hi = BUILDING
    assert stats.carving is not None and stats.carving.rays_used > 0
    assert not (centres(grid, lo, hi) == FREE).any()


def test_wrong_visibility_through_reconstructed_walls_is_discarded():
    pts, nrm = building_scene(0.5)
    cloud, rays = observe(pts, nrm, CAMS, [BUILDING])
    grid, stats, _ = build(cloud, frustum_visibility(cloud, rays))
    assert stats.carving is not None and stats.carving.rays_through_occupied > 0
    assert not (centres(grid, *BUILDING) == FREE).any()


def test_frustum_visibility_would_carve_an_unreconstructed_building():
    """Why visibility must come from the fusion: with walls that left no points, rays to
    cameras that merely frame a point cut straight through the building."""
    pts, nrm = building_scene(1.5)
    cloud, rays = observe(pts, nrm, CAMS, [BUILDING])
    wrong, _, _ = build(cloud, frustum_visibility(cloud, rays))
    assert (centres(wrong, *BUILDING) == FREE).any()


def test_a_noise_point_under_a_sparse_roof_does_not_open_a_shaft(monkeypatch):
    pts, nrm = building_scene(1.5)
    noise = np.array([[0.3, 0.3, 4.0]])
    pts, nrm = np.concatenate([pts, noise]), np.concatenate([nrm, [UP]])
    cloud, rays = observe(pts, nrm, CAMS, [BUILDING])
    # The fusion "saw" the noise point from every camera above the roof.
    n = len(cloud)
    above = np.flatnonzero(rays.centres[:, 2] > 20)
    rays = CameraRays(points=np.concatenate([cloud, noise]), centres=rays.centres,
                      point=np.append(rays.point, np.full(len(above), n)),
                      camera=np.append(rays.camera, above),
                      normals=np.concatenate([rays.normals, [UP]]))
    cloud = rays.points
    grid, _, _ = build(cloud, rays)
    assert not (centres(grid, *BUILDING) == FREE).any()

    # The failure the erosion exists for: opening every voxel the rays crossed lets the
    # traces through the roof into the building.
    import aerosplat.surface.carving as carving

    def open_all(state, spec, views, min_views, src):
        raw = views.reshape(spec.shape) >= min_views
        state[raw] = FREE
        n = int(raw.sum())
        return {"voxels_crossed": n, "voxels_carved": n, "voxels_eroded": 0,
                "carved_under_surface": 0, "carved_assumed": 0}

    monkeypatch.setattr(carving, "_apply", open_all)
    leaky, _, _ = build(cloud, rays)
    assert (centres(leaky, *BUILDING) == FREE).any()


# ── a hole in the cloud: skyline columns become evidence ─────────────────
HOLE_LO, HOLE_HI = np.array([-7.0, -7.0]), np.array([7.0, 7.0])
GRAZING = np.concatenate([ring(32, 35.0, 5.0), ring(32, 35.0, 3.0, phase=0.05)])   # 5-8 deg


def hole_scene():
    g, gn = sheet(-25, 25, -25, 25, 0.0, 0.5)
    keep = ~inside(g, [*HOLE_LO, -1], [*HOLE_HI, 1])
    return g[keep], gn[keep]


@pytest.fixture(scope="module")
def hole_grids():
    pts, nrm = hole_scene()
    cloud, rays = observe(pts, nrm, GRAZING, [])
    return build(cloud), build(cloud, rays)


def test_rays_turn_a_skyline_column_into_evidence(hole_grids):
    (plain, _, _), (carved, stats, _) = hole_grids
    c = [[0.25, 0.25, 3.0]]
    assert plain.source_at(c)[0] == ColumnSource.SKYLINE and plain.assumed_at(c)[0]
    assert carved.source_at(c)[0] == ColumnSource.CARVED and not carved.assumed_at(c)[0]
    assert carved.state_at([[0.25, 0.25, 1.25]])[0] == FREE         # was unknown
    assert plain.state_at([[0.25, 0.25, 1.25]])[0] == UNKNOWN
    assert stats.carving is not None and stats.carving.columns_upgraded["skyline"] > 0
    assert stats.columns["carved"] == sum(stats.carving.columns_upgraded.values())
    assert stats.assumed_fraction < hole_grids[0][1].assumed_fraction


def voxel_centres(grid, lo, hi):
    """(N, 3) centres of the voxels that touch the box (lo, hi)."""
    s = grid.spec
    r = s.res_m
    axes = [o + (np.arange(n) + 0.5) * r for o, n in ((s.x0, s.cols), (s.y0, s.rows),
                                                       (s.z0, s.layers))]
    x, y, z = (a[(a + r / 2 > lo[d]) & (a - r / 2 < hi[d])] for d, a in enumerate(axes))
    return np.stack(np.meshgrid(x, y, z, indexing="ij"), -1).reshape(-1, 3)


@pytest.mark.parametrize("height", [3.0, 0.8], ids=["above-ceiling", "below-ceiling"])
def test_rays_never_vouch_for_space_inside_an_unreconstructed_building(height):
    """A building that left no points (black glass) occludes the rays that would cross it:
    nothing inside it is carved, and its column is not turned into evidence. Taller than the
    skyline ceiling, its top is in the free air the 2.5-D model already assumes — and that
    stays flagged as an assumption. Lower, the air above it is carved as it should be."""
    hidden = (np.array([-1.5, -1.5, 0.0]), np.array([1.5, 1.5, height]))
    pts, nrm = hole_scene()
    cloud, rays = observe(pts, nrm, GRAZING, [hidden])
    grid, _, _ = build(cloud, rays)
    c = voxel_centres(grid, *hidden)
    free = grid.state_at(c) == FREE
    assert not (free & ~grid.assumed_at(c)).any()
    tall = height > 1.5                                  # the skyline ceiling is at 1.0-1.5 m
    assert free.any() == tall                           # v1's assumption, not carving
    centre = [[0.25, 0.25, 0.0]]
    assert grid.source_at(centre)[0] == (ColumnSource.SKYLINE if tall else ColumnSource.CARVED)


STEEP = ring(32, 35.0, 30.0)                            # ~40 deg: rays cross the grid's top


def test_sky_mode_opens_a_column_to_the_sky_only_where_rays_crossed_its_top():
    """With unknown="sky" a column without data is blocked to the sky. Rays across its top
    layer carve it, and only then does the column continue free above the grid."""
    pts, nrm = hole_scene()
    c = [[0.25, 0.25, 1000.0]]
    cloud, rays = observe(pts, nrm, STEEP, [])
    grid, stats, _ = build(cloud, rays, unknown="sky")
    assert grid.source_at(c)[0] == ColumnSource.CARVED
    assert grid.state_at(c)[0] == FREE and not grid.assumed_at(c)[0]
    assert stats.carving is not None and stats.carving.columns_upgraded["sky"] > 0
    cloud, rays = observe(pts, nrm, GRAZING, [])         # these never reach the top layer
    grid, _, _ = build(cloud, rays, unknown="sky")
    assert grid.source_at(c)[0] == ColumnSource.SKY and grid.state_at(c)[0] == UNKNOWN


def test_carving_is_deterministic(hole_grids):
    pts, nrm = hole_scene()
    cloud, rays = observe(pts, nrm, GRAZING, [])
    again, stats, _ = build(cloud, rays)
    grid, stats0, _ = hole_grids[1]
    assert np.array_equal(again.state, grid.state)
    assert np.array_equal(again.column_source, grid.column_source)
    timing = {"seconds_rays", "seconds_total"}
    assert (stats.carving.model_dump(exclude=timing) == stats0.carving.model_dump(exclude=timing))


def test_wrong_frame_fails_loudly():
    pts, nrm = hole_scene()
    cloud, rays = observe(pts, nrm, GRAZING, [])
    shifted = CameraRays(points=rays.points + [0.3, 0.0, 0.0], centres=rays.centres,
                         point=rays.point, camera=rays.camera)
    with pytest.raises(ValueError, match="fused point set"):
        build(cloud, shifted)


@pytest.mark.parametrize("kw, match", [(dict(carve_min_views=0), "min_views"),
                                       (dict(carve_min_views=256), "min_views"),
                                       (dict(carve_min_views=1.5), "min_views"),
                                       (dict(carve_margin_m=-1.0), "margin"),
                                       (dict(carve_margin_m=float("nan")), "margin")])
def test_build_rejects_bad_carving_parameters(hole_grids, kw, match):
    pts, nrm = hole_scene()
    cloud, rays = observe(pts, nrm, GRAZING[:4], [])
    with pytest.raises(ValueError, match=match):
        build(cloud, rays, **kw)


# ── the rules, one at a time, on hand-made grids ──────────────────────────
def flat_grid(rows=40, cols=40, layers=16, res=0.5, source=ColumnSource.SKY):
    """Ground voxels (k = 0) occupied, everything above unknown: no free air at all."""
    spec = GridSpec(res_m=res, x0=0.0, y0=0.0, z0=0.0, rows=rows, cols=cols, layers=layers)
    state = np.full(spec.shape, UNKNOWN, np.uint8)
    state[:, :, 0] = OCCUPIED
    free_from = np.full((rows, cols), layers, np.int16)
    return spec, state, free_from, np.full((rows, cols), source, np.uint8)


def ground_rays(spec, cams, step=0.5):
    """Rays from `cams` to points in the ground voxels (cell centres, z = 0.25)."""
    p, n = sheet(1, spec.x1 - 1, 1, spec.y1 - 1, 0.25, step)
    cam, pt = np.meshgrid(np.arange(len(cams)), np.arange(len(p)), indexing="ij")
    rays = CameraRays(points=p, centres=np.asarray(cams, float), point=pt.ravel(),
                      camera=cam.ravel(), normals=n)
    i, j, k = spec.index_of(p)
    return rays, np.unique((i * spec.cols + j) * spec.layers + k)


def test_one_camera_is_not_enough():
    spec, state, free_from, src = flat_grid()
    one = [[10.0, -40.0, 30.0]]
    rays, pv = ground_rays(spec, one)
    out = carve(state, spec, free_from, src, rays, pv, min_views=2)
    assert out["voxels_carved"] == 0 and not (state == FREE).any()
    spec, state, free_from, src = flat_grid()
    carve(state, spec, free_from, src, rays, pv, min_views=1)
    assert (state == FREE).any()


def test_two_cameras_carve_where_they_overlap():
    spec, state, free_from, src = flat_grid()
    rays, pv = ground_rays(spec, [[10.0, -40.0, 30.0], [-40.0, 10.0, 30.0]])
    out = carve(state, spec, free_from, src, rays, pv, min_views=2)
    assert out["voxels_carved"] > 0 and out["voxels_eroded"] > 0
    # Never the occupied ground, never anything above the rays' reach.
    assert (state[:, :, 0] == OCCUPIED).all()
    assert out["rays_through_occupied"] == 0


def test_pillar_columns_are_never_carved():
    spec, state, free_from, src = flat_grid(source=ColumnSource.PILLAR)
    rays, pv = ground_rays(spec, [[10.0, -40.0, 30.0], [-40.0, 10.0, 30.0]])
    out = carve(state, spec, free_from, src, rays, pv, min_views=1)
    assert out["voxels_carved"] == 0 and not (state == FREE).any()


def test_unsupported_endpoints_do_not_carve():
    """A fused point outside every voxel that holds a cloud point (an outlier the parse
    stage dropped, or a point outside a bbox crop) carves nothing."""
    spec, state, free_from, src = flat_grid()
    rays, _ = ground_rays(spec, [[10.0, -40.0, 30.0], [-40.0, 10.0, 30.0]])
    out = carve(state, spec, free_from, src, rays, np.zeros(0, np.int64), min_views=1)
    assert out["rays_unsupported"] == out["rays_total"] and not (state == FREE).any()


def test_erosion_gives_back_the_shell_next_to_blocked_space():
    spec = GridSpec(res_m=0.5, x0=0.0, y0=0.0, z0=0.0, rows=10, cols=10, layers=6)
    state = np.full(spec.shape, UNKNOWN, np.uint8)
    views = np.zeros(spec.voxels, np.uint8).reshape(spec.shape)
    views[2:8, 2:8, 1:5] = 2
    src = np.full((10, 10), ColumnSource.SKY, np.uint8)
    out = _apply(state, spec, views.reshape(-1), 2, src)
    expect = np.zeros(spec.shape, bool)
    expect[3:7, 3:7, 2:4] = True
    assert np.array_equal(state == FREE, expect)
    assert out["voxels_crossed"] == 6 * 6 * 4 and out["voxels_carved"] == 4 * 4 * 2

    # Beside the grid is unknown; above it, the top layer continues; free voxels are free.
    state = np.full(spec.shape, UNKNOWN, np.uint8)
    state[:, :, 5] = FREE
    views[:] = 0
    views[0:6, 2:8, 3:5] = 2
    _apply(state, spec, views.reshape(-1), 2, src)
    opened = (state == FREE)[:, :, :5]
    assert not opened[0].any()                          # row 0 touches the grid's edge
    assert opened[1:5, 3:7, 4].all() and not opened[:, :, 3].any()


def random_scene(seed, rows=24, cols=20, layers=14, res=0.5):
    rng = np.random.default_rng(seed)
    spec = GridSpec(res_m=res, x0=-3.0, y0=2.0, z0=-1.0, rows=rows, cols=cols, layers=layers)
    heights = rng.integers(0, layers - 3, size=(rows, cols))
    heights[rng.random((rows, cols)) < 0.05] = layers - 1
    k = np.arange(layers)[None, None, :]
    state = np.where(k <= heights[:, :, None], UNKNOWN, FREE).astype(np.uint8)
    state[(state == UNKNOWN) & (rng.random(state.shape) < 0.3)] = OCCUPIED
    free_from = (heights + 1).astype(np.int16)
    src = rng.choice(np.array([1, 2, 3, 4, 6], np.uint8), size=(rows, cols))
    carvable = np.isin(src, (1, 2, 3, 4)).reshape(-1)
    o = np.array([spec.x0 + rng.uniform(-30, 40), spec.y0 + rng.uniform(-30, 40), 25.0])
    ends = np.column_stack([rng.uniform(spec.x0, spec.x1, 300), rng.uniform(spec.y0, spec.y1, 300),
                            rng.uniform(spec.z0, spec.z1, 300)])
    normals = rng.normal(size=(300, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return spec, state, free_from, carvable, o, ends, normals


def brute_march(seg, spec, flat, carvable):
    """Every sample of every segment, by the same float32 expression: the reference."""
    rows, cols, layers = spec.shape
    bad = np.zeros(len(seg.last), bool)
    out = []
    for q in range(len(seg.last)):
        if seg.last[q] < 0:
            continue
        m = np.arange(int(seg.last[q]) + 1, dtype=np.float32)
        i = np.minimum((seg.b[1][q] + seg.s[1][q] * m).astype(np.int32), rows - 1)
        j = np.minimum((seg.b[0][q] + seg.s[0][q] * m).astype(np.int32), cols - 1)
        k = np.minimum((seg.b[2][q] + seg.s[2][q] * m).astype(np.int32), layers - 1)
        col = i * cols + j
        lin = col * layers + k
        st = flat[lin]
        bad[q] = ((st == OCCUPIED) & (m <= seg.last_occ[q])).any()
        if not bad[q]:
            out.append(lin[(st == UNKNOWN) & carvable[col]])
    return (np.unique(np.concatenate(out)) if out else np.zeros(0, int)), bad


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("coarse", [(8.0, 1.0), (16.0, 4.0, 1.0), (2.0,), ()])
def test_skipping_free_air_is_exact(seed, coarse):
    spec, state, free_from, carvable, o, ends, normals = random_scene(seed)
    seg = _segments(o, ends, normals, spec, margin_m=1.0)
    lv = _levels(free_from, coarse, spec.res_m, spec.layers)
    got, bad, _ = _march(seg, lv, spec, state.reshape(-1), carvable, 1 << 10)
    ref, ref_bad = brute_march(seg, spec, state.reshape(-1), carvable)
    assert np.array_equal(np.unique(got), ref)
    assert np.array_equal(bad, ref_bad)
    assert len(ref) > 0 and ref_bad.any() and not ref_bad.all()     # the case is not trivial


def test_segments_stop_short_of_their_end():
    spec, state, free_from, carvable, o, ends, normals = random_scene(0)
    h = spec.res_m / 2
    for margin in (0.0, 1.0, 3.0):
        seg = _segments(o, ends, normals, spec, margin_m=margin)
        start = np.stack(seg.b, 1) * spec.res_m + [spec.x0, spec.y0, spec.z0]
        step = np.stack(seg.s, 1) * spec.res_m
        ok = seg.last >= 0
        last = start + seg.last[:, None] * step
        dist = np.linalg.norm(ends - last, axis=1)
        assert (dist[ok] >= margin - 1e-3).all()
        assert (np.linalg.norm(step, axis=1) == pytest.approx(h, rel=1e-5))
        assert (seg.last_occ <= seg.last).all()


def test_grazing_rays_are_exempt_from_their_own_surface_only():
    """A ray along the ground runs through the ground's voxel layer for metres before its
    end point: those occupied voxels are the surface it ends on, not an occluder."""
    spec = GridSpec(res_m=0.5, x0=0.0, y0=0.0, z0=0.0, rows=4, cols=80, layers=10)
    o = np.array([0.0, 1.0, 4.0])
    end = np.array([[39.0, 1.0, 0.3]])                  # ~5.7 deg above the ground
    flat_n = np.array([[0.0, 0.0, 1.0]])
    seg = _segments(o, end, flat_n, spec, margin_m=1.0)
    seg0 = _segments(o, end, None, spec, margin_m=1.0)
    h = spec.res_m / 2
    exempt = (seg.last[0] - seg.last_occ[0]) * h
    assert exempt == pytest.approx(10 * spec.res_m - 1.0, abs=h)     # capped at GRAZE_VOXELS
    assert seg0.last_occ[0] == seg0.last[0]             # no normal: only the margin
    steep = _segments(np.array([39.0, 1.0, 30.0]), end, flat_n, spec, margin_m=1.0)
    assert steep.last_occ[0] == steep.last[0]           # sqrt(3) r / 1 < margin


# ── COLMAP visibility files ───────────────────────────────────────────────
def look_at(c, target):
    """World-to-camera rotation (x right, y down, z forward) of a camera at c."""
    f = np.asarray(target, float) - c
    f /= np.linalg.norm(f)
    right = np.cross(f, UP if abs(f[2]) < 0.99 else [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(f, right), f])


def quat(R):
    """(w, x, y, z), as COLMAP writes it."""
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return [w, x, y, z]


def framed(c, target, pts, width, focal):
    p = (pts - c) @ look_at(c, target).T
    uv = focal * p[:, :2] / p[:, 2:] + width / 2
    return (p[:, 2] > 0) & (uv >= 0).all(axis=1) & (uv <= width).all(axis=1)


def write_workspace(folder, pts, nrm, cams, targets, ids, seen_by, width=400, focal=400.0):
    """A COLMAP dense workspace: fused.ply, fused.ply.vis and sparse/{cameras,images}.txt.

    `cams` / `targets` / `ids` are in the images file's order; `seen_by[p]` lists, by that
    file position, the cameras that saw point p. The visibility file numbers images the way
    COLMAP does: by position in the model, i.e. sorted by image id."""
    folder.mkdir(parents=True, exist_ok=True)
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
            + "".join(f"property float {n}\n" for n in ("x", "y", "z", "nx", "ny", "nz"))
            + "".join(f"property uchar {c}\n" for c in ("red", "green", "blue"))
            + "end_header\n")
    rec = np.zeros(len(pts), dtype=[(n, "<f4") for n in ("x", "y", "z", "nx", "ny", "nz")]
                   + [(c, "u1") for c in ("red", "green", "blue")])
    for a, n in enumerate("xyz"):
        rec[n], rec["n" + n] = pts[:, a], nrm[:, a]
    (folder / "fused.ply").write_bytes(head.encode() + rec.tobytes())
    position = {c: n for n, c in enumerate(sorted(range(len(cams)), key=lambda c: ids[c]))}
    blob = struct.pack("<Q", len(pts)) + b"".join(
        struct.pack(f"<I{len(v)}I", len(v), *(position[c] for c in v)) for v in seen_by)
    (folder / "fused.ply.vis").write_bytes(blob)
    sp = folder / "sparse"
    sp.mkdir(exist_ok=True)
    (sp / "cameras.txt").write_text(f"1 PINHOLE {width} {width} {focal} {focal} "
                                    f"{width / 2} {width / 2}\n")
    lines = []
    for c in range(len(cams)):
        R = look_at(cams[c], targets[c])
        t = -R @ cams[c]
        lines += [f"{ids[c]} {' '.join(map(str, quat(R)))} {t[0]} {t[1]} {t[2]} 1 img{c:02d}.jpg",
                  ""]
    (sp / "images.txt").write_text("\n".join(lines) + "\n")


IDS = [7, 3, 12, 5, 9, 1, 4]                            # neither file nor name order


def strip_workspace(folder, wide=False, extra=()):
    """Seven cameras over a strip of ground, each framing its own 12 m patch (or, `wide`,
    everything). Visibility is what each camera frames, plus `extra` (point, camera) pairs."""
    pts, nrm = sheet(-16, 16, -4, 4, 0.0, 1.0)
    xs = np.linspace(-12, 12, 7)
    cams = np.stack([xs, np.zeros(7), np.full(7, 30.0)], axis=1)
    targets = cams * [1, 1, 0]
    focal = 60.0 if wide else 1200.0
    seen = np.stack([framed(cams[c], targets[c], pts, 400, focal) for c in range(7)])
    seen_by = [list(np.flatnonzero(seen[:, p])) for p in range(len(pts))]
    for p, c in extra:
        seen_by[p].append(c)
    write_workspace(folder, pts, nrm, cams, targets, IDS, seen_by, focal=focal)
    return pts, cams, seen


def test_visibility_maps_indices_to_the_right_cameras(tmp_path):
    pts, cams, seen = strip_workspace(tmp_path)
    assert 0 < seen.mean() < 0.5                          # each camera frames only a part
    rays = load_colmap_rays(tmp_path / "fused.ply", tmp_path / "sparse")
    assert rays.info["image_order"] == "image_id" and rays.info["pairs_outside_image"] == 0
    got = {(tuple(np.round(rays.points[p], 3)), tuple(np.round(rays.centres[c], 3)))
           for p, c in zip(rays.point, rays.camera)}
    cam, pt = np.nonzero(seen)
    assert got == {(tuple(np.round(pts[p], 3)), tuple(np.round(cams[c], 3)))
                   for c, p in zip(cam, pt)}
    assert np.allclose(rays.normals, [0, 0, 1])


def test_visibility_order_must_be_unambiguous(tmp_path):
    """With a wide lens every camera frames every point: the in-frame test cannot tell the
    image orders apart, so no visibility is trusted."""
    strip_workspace(tmp_path, wide=True)
    with pytest.raises(VisibilityError, match="ambiguous"):
        load_colmap_rays(tmp_path / "fused.ply", tmp_path / "sparse")


def test_pairs_outside_their_image_are_dropped_or_refused(tmp_path):
    # Point 0 sits under camera 0 and outside camera 6's frame.
    pts, cams, seen = strip_workspace(tmp_path, extra=[(0, 6)])
    assert not seen[6, 0]
    rays = load_colmap_rays(tmp_path / "fused.ply", tmp_path / "sparse")
    assert rays.info["pairs_outside_image"] == 1 and len(rays.point) == seen.sum()
    many = [(p, 6) for p in range(len(pts)) if not seen[6, p]]
    strip_workspace(tmp_path, extra=many)
    with pytest.raises(VisibilityError, match="no image order|outside"):
        load_colmap_rays(tmp_path / "fused.ply", tmp_path / "sparse")


def test_visibility_file_must_match_the_cloud(tmp_path):
    strip_workspace(tmp_path)
    vis = tmp_path / "fused.ply.vis"
    blob = vis.read_bytes()
    vis.write_bytes(blob[:-4])
    with pytest.raises(VisibilityError, match="truncated"):
        load_colmap_rays(tmp_path / "fused.ply", tmp_path / "sparse")
    vis.write_bytes(blob + b"\0\0\0\0")
    with pytest.raises(VisibilityError, match="trailing"):
        load_colmap_rays(tmp_path / "fused.ply", tmp_path / "sparse")
    vis.write_bytes(struct.pack("<Q", 3) + blob[8:])
    with pytest.raises(VisibilityError, match="entries"):
        read_fused_visibility(vis, 256)
    vis.unlink()
    with pytest.raises(VisibilityError, match="no visibility"):
        load_colmap_rays(tmp_path / "fused.ply", tmp_path / "sparse")


def test_visibility_is_mapped_into_the_site_frame(tmp_path):
    from aerosplat.geo.affine import AffineMap

    strip_workspace(tmp_path)
    a = AffineMap(np.diag([1.0025, 0.9987, 1.0]), np.array([0.4, -0.2, 0.1]))
    rays = load_colmap_rays(tmp_path / "fused.ply", tmp_path / "sparse", a)
    plain = load_colmap_rays(tmp_path / "fused.ply", tmp_path / "sparse")
    assert np.allclose(rays.points, a.apply(plain.points))
    assert np.allclose(rays.centres, a.apply(plain.centres))
    assert np.allclose(np.linalg.norm(rays.normals, axis=1), 1.0)


# ── files ────────────────────────────────────────────────────────────────
def test_carved_grid_round_trips(tmp_path, hole_grids):
    from aerosplat.surface.models import EsdfInfo, OccupancyReport

    grid, stats, warnings = hole_grids[1]
    report = OccupancyReport(grid=grid.spec, bounds=grid.spec.bounds(),
                             esdf=EsdfInfo(max_distance_m=grid.max_distance_m), stats=stats,
                             warnings=warnings)
    write_occupancy(tmp_path / "occ", grid, report)
    back, rep = read_occupancy(tmp_path / "occ")
    assert np.array_equal(back.free_from, grid.free_from)
    assert np.array_equal(back.state, grid.state)
    assert rep.stats.carving == stats.carving
    p = [[0.25, 0.25, 3.0], [0.25, 0.25, 1.25], [20.0, 20.0, 3.0]]
    assert np.array_equal(back.assumed_at(p), grid.assumed_at(p))


# ── the pipeline stage ───────────────────────────────────────────────────
def canopy_workspace(folder):
    """The canopy scene as a COLMAP dense workspace plus the cloud .pt the stage reads.
    A camera saw a point when nothing occludes it and its (wide) image frames it."""
    torch = pytest.importorskip("torch")
    pts, nrm = canopy_scene()
    cams = np.concatenate([LOW, HIGH])
    targets = np.zeros_like(cams)
    seen = ~blocked_by(cams, pts, [CANOPY])
    seen &= np.stack([framed(cams[c], targets[c], pts, 400, 90.0) for c in range(len(cams))])
    keep = seen.sum(axis=0) >= 2
    pts, nrm, seen = pts[keep], nrm[keep], seen[:, keep]
    write_workspace(folder / "dense", pts, nrm, cams, targets, list(range(1, len(cams) + 1)),
                    [list(np.flatnonzero(seen[:, p])) for p in range(len(pts))], focal=90.0)
    cloud = folder / "gs" / "dense.pt"
    cloud.parent.mkdir(parents=True)
    torch.save({"xyz": torch.tensor(pts, dtype=torch.float32)}, cloud)
    return cloud


def test_stage_carves_with_the_dense_workspace(tmp_path):
    from aerosplat.semantics.stage import run_semantics
    from aerosplat.surface import stage

    cloud = canopy_workspace(tmp_path)
    run_semantics(cloud, tmp_path / "semantics")
    common = ["--cloud", str(cloud), "--semantics-dir", str(tmp_path / "semantics"), "--res", "0.5"]
    assert stage.main(common + ["--out-dir", str(tmp_path / "occ"),
                                "--fused", str(tmp_path / "dense" / "fused.ply")]) == 0
    grid, report = read_occupancy(tmp_path / "occ")
    c = report.stats.carving
    assert c is not None and c.carved_under_surface > 0 and report.params["carve"]
    assert c.visibility["image_order"] == "image_id"
    assert report.inputs["fused"].endswith("fused.ply") and "rays" in report.timings_s
    assert (centres(grid, [-4, -4, 1.0], [4, 4, 2.0]) == FREE).mean() > 0.9

    # Without the workspace the grid is 2.5-D, and the report says so.
    assert stage.main(common + ["--out-dir", str(tmp_path / "plain"),
                                "--fused", str(tmp_path / "nowhere" / "fused.ply")]) == 0
    plain, report = read_occupancy(tmp_path / "plain")
    assert report.stats.carving is None and not report.params["carve"]
    assert any("не вирізано" in w for w in report.warnings)
    assert not (centres(plain, [-4, -4, 1.0], [4, 4, 2.0]) == FREE).any()
    assert stage.main(common + ["--out-dir", str(tmp_path / "off"), "--no-carve",
                                "--fused", str(tmp_path / "dense" / "fused.ply")]) == 0
    off, report = read_occupancy(tmp_path / "off")
    assert report.stats.carving is None and np.array_equal(off.state, plain.state)


def test_stage_refuses_a_workspace_that_contradicts_itself(tmp_path):
    from aerosplat.semantics.stage import run_semantics
    from aerosplat.surface import stage

    cloud = canopy_workspace(tmp_path)
    run_semantics(cloud, tmp_path / "semantics")
    vis = tmp_path / "dense" / "fused.ply.vis"
    vis.write_bytes(vis.read_bytes()[:-4])
    with pytest.raises(SystemExit, match="видимість"):
        stage.main(["--cloud", str(cloud), "--semantics-dir", str(tmp_path / "semantics"),
                    "--out-dir", str(tmp_path / "occ"), "--res", "0.5",
                    "--fused", str(tmp_path / "dense" / "fused.ply")])
    assert not (tmp_path / "occ").exists()


def test_occupancy_argv_passes_the_dense_workspace(es_site):
    from aerosplat.config import RunConfig
    from aerosplat.pipeline.layout import RunLayout
    from aerosplat.pipeline.stages import STAGES, argv_occupancy, argv_semantics

    cat, site, cap, _ = es_site
    cfg = RunConfig(camera_source="earthstudio_json", cloud_source="dense").to_dict()
    L = RunLayout.for_site_run(cfg, site=site, site_dirs=cat.site_dirs(site), capture=cap,
                               epoch_id="baseline", run_id="r1")
    a, sem = argv_occupancy(cfg, L), argv_semantics(cfg, L)
    assert a[a.index("--fused") + 1] == str(L.fused_ply)
    assert a[a.index("--sparse") + 1] == str(L.dense / "sparse")
    # Points and cameras reach Site ENU by the map semantics uses for its cameras.
    assert a[a.index("--registration") + 1] == sem[sem.index("--registration") + 1]
    assert a[a.index("--carve-min-views") + 1] == "2" and "--no-carve-under" not in a
    assert L.fused_ply in STAGES["occupancy"].inputs(cfg, L)
    off = RunConfig(camera_source="earthstudio_json", cloud_source="dense",
                    occupancy_carve=False, occupancy_carve_under=False).to_dict()
    b = argv_occupancy(off, L)
    assert "--no-carve" in b and "--fused" not in b
    assert L.fused_ply not in STAGES["occupancy"].inputs(off, L)
    under = RunConfig(camera_source="earthstudio_json", cloud_source="dense",
                      occupancy_carve_under=False).to_dict()
    assert "--no-carve-under" in argv_occupancy(under, L)


def test_config_rejects_bad_carving_settings():
    from pydantic import ValidationError

    from aerosplat.config import RunConfig

    RunConfig(occupancy_carve_min_views=3, occupancy_carve_margin=0.0)
    for kw in (dict(occupancy_carve_min_views=0), dict(occupancy_carve_margin=-0.5)):
        with pytest.raises(ValidationError):
            RunConfig(**kw)


def test_surface_stays_free_of_torch():
    """aerosplat.surface is numpy/scipy only: importing it (carving included) loads no torch."""
    import subprocess
    import sys

    code = ("import sys, aerosplat.surface.stage, aerosplat.surface.carving; "
            "sys.exit('torch' in sys.modules)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
