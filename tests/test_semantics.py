"""Semantics: terrain, footprints, facade fitting, coverage — against synthetic ground truth.

dataset_v6 cannot settle whether this code is right: its cloud has 3 m point
spacing and Google's rounded mesh has no crisp walls. So the geometry is checked
on scenes built here, where the answer is known exactly.
"""

import math

import numpy as np
import pytest

from aerosplat.io.colmap_model import CameraPose, Intrinsics, read_model
from aerosplat.semantics.buildings import find_buildings, simplify, trace_region_outline
from aerosplat.semantics.coverage import (coverage_report, distance_to_footprint,
                                          facade_coverage, facade_grid, transform_pose)
from aerosplat.semantics.facades import facades_of_building, merge_coplanar, ransac_line
from aerosplat.semantics.models import (Building, Coverage, Facade, azimuth_of, octant_of,
                                        u_axis_for)
from aerosplat.semantics.terrain import build_terrain

RNG = np.random.default_rng(7)


# ── scene builders ───────────────────────────────────────────────────────
def ground(size: float = 140.0, step: float = 0.7, z: float = 0.0) -> np.ndarray:
    n = int(size / step)
    x, y = np.meshgrid(np.linspace(-size / 2, size / 2, n), np.linspace(-size / 2, size / 2, n))
    return np.stack([x.ravel(), y.ravel(), np.full(x.size, z)], axis=1)


def box(centre=(0.0, 0.0), width=30.0, depth=20.0, height=12.0, yaw_deg=0.0,
        step=0.5, noise=0.0) -> np.ndarray:
    """Points on the four walls and the flat roof of a rectangular building.

    `yaw_deg` turns the box about its centre; width runs along the box's local x.
    """
    hw, hd = width / 2, depth / 2
    pts = []
    for u in np.arange(-hw, hw + step, step):                   # walls along local x
        for z in np.arange(0.0, height + step, step):
            pts += [(u, -hd, z), (u, hd, z)]
    for v in np.arange(-hd, hd + step, step):                   # walls along local y
        for z in np.arange(0.0, height + step, step):
            pts += [(-hw, v, z), (hw, v, z)]
    gx, gy = np.meshgrid(np.arange(-hw, hw, step), np.arange(-hd, hd, step))
    roof = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, height)], axis=1)
    p = np.concatenate([np.array(pts, float), roof])
    a = math.radians(yaw_deg)
    rot = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    p = p @ rot.T + np.array([centre[0], centre[1], 0.0])
    if noise:
        p = p + RNG.normal(0.0, noise, p.shape)
    return p


def tree(centre=(0.0, 0.0), radius=6.0, height=9.0, n=4000, bumps=9) -> np.ndarray:
    """A rough canopy blob: tall, but with none of a roof's local flatness.

    Built from overlapping bumps of differing height, so the highest point in
    one cell differs from its neighbour by metres -- which is what makes canopy
    rough and a roof smooth.
    """
    tops = np.stack([RNG.normal(centre[0], radius * 0.5, bumps),
                     RNG.normal(centre[1], radius * 0.5, bumps),
                     height * (0.5 + 0.5 * RNG.random(bumps))], axis=1)
    out = []
    for cx, cy, h in tops:
        r = radius * 0.45 * np.sqrt(RNG.random(n // bumps))
        th = RNG.random(n // bumps) * 2 * math.pi
        out.append(np.stack([cx + r * np.cos(th), cy + r * np.sin(th),
                             h * (1 - 0.7 * (r / (radius * 0.45)) ** 2)], axis=1))
    return np.concatenate(out)


# ── conventions ──────────────────────────────────────────────────────────
def test_azimuth_octant_and_u_axis_conventions():
    assert azimuth_of((0, 1)) == pytest.approx(0)          # normal north
    assert azimuth_of((1, 0)) == pytest.approx(90)         # normal east
    assert octant_of(0) == "N" and octant_of(91) == "E" and octant_of(359) == "N"
    assert octant_of(45) == "NE" and octant_of(225) == "SW"
    # Facing a north wall from outside you stand north looking south, so right is west.
    assert u_axis_for((0, 1)) == pytest.approx((-1, 0))
    assert u_axis_for((1, 0)) == pytest.approx((0, 1))     # east wall: right is north


# ── terrain ──────────────────────────────────────────────────────────────
def test_terrain_separates_roof_from_bare_earth():
    cloud = np.concatenate([ground(), box(height=12.0)])
    t = build_terrain(cloud, res_m=1.0)
    i, j = t.model.index_of([0.0], [0.0])
    assert t.dsm[i, j][0] == pytest.approx(12.0, abs=0.3)
    assert t.dtm[i, j][0] == pytest.approx(0.0, abs=0.5)      # building removed from the DTM
    assert t.ndsm[i, j][0] == pytest.approx(12.0, abs=0.6)
    far_i, far_j = t.model.index_of([60.0], [60.0])
    assert t.ndsm[far_i, far_j][0] == pytest.approx(0.0, abs=0.5)


def test_terrain_dtm_follows_a_slope():
    g = ground(size=140, step=0.7)
    g[:, 2] = 0.05 * g[:, 0]                                  # 5% ramp towards the east
    t = build_terrain(np.concatenate([g, box(centre=(30, 0), height=12)]), res_m=1.0)
    for e in (-50.0, 0.0, 50.0):
        i, j = t.model.index_of([e], [-50.0])
        assert t.dtm[i, j][0] == pytest.approx(0.05 * e, abs=1.0)


def test_roof_is_flat_and_canopy_is_not():
    t = build_terrain(np.concatenate([ground(), box(height=12), tree(centre=(45, 45))]),
                      res_m=1.0)
    roof_i, roof_j = t.model.index_of([0.0], [0.0])
    tree_i, tree_j = t.model.index_of([45.0], [45.0])
    assert t.roughness[roof_i, roof_j][0] < 0.1
    assert t.roughness[tree_i, tree_j][0] > 0.3


# ── footprints ───────────────────────────────────────────────────────────
def test_outline_of_a_rectangle_is_its_four_corners():
    mask = np.zeros((20, 30), bool)
    mask[5:15, 8:25] = True
    ring = np.array(trace_region_outline(mask), float)
    assert len(ring) >= 4
    corners = simplify(ring, 0.5)
    assert len(corners) == 4
    assert {(int(x), int(y)) for x, y in corners} == {(8, 5), (25, 5), (25, 15), (8, 15)}


def test_outline_is_counter_clockwise():
    mask = np.zeros((12, 12), bool)
    mask[3:9, 3:9] = True
    p = np.array(trace_region_outline(mask), float)
    area = 0.5 * float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))
    assert area > 0


def test_finds_the_building_and_rejects_the_tree():
    cloud = np.concatenate([ground(), box(centre=(-20, -10), width=30, depth=20, height=12),
                            tree(centre=(40, 35)), tree(centre=(30, -40))])
    t = build_terrain(cloud, res_m=1.0)
    buildings, labels = find_buildings(t)
    assert len(buildings) == 1
    b = buildings[0]
    assert b.area_m2 == pytest.approx(600, rel=0.2)
    assert b.height_m == pytest.approx(12, abs=1.0)
    assert b.centroid[0] == pytest.approx(-20, abs=2.0)
    assert b.centroid[1] == pytest.approx(-10, abs=2.0)
    assert labels.max() == 1


def test_two_buildings_stay_apart():
    cloud = np.concatenate([ground(), box(centre=(-30, 0), height=10),
                            box(centre=(30, 0), height=14)])
    buildings, _ = find_buildings(build_terrain(cloud, res_m=1.0))
    assert len(buildings) == 2
    assert sorted(round(b.height_m) for b in buildings) == [10, 14]


# ── facade fitting ───────────────────────────────────────────────────────
def test_longest_contiguous_run_wins():
    """A line through two walls down one street keeps only the stretch that is continuous."""
    short = np.stack([np.linspace(0, 6, 40), np.zeros(40)], axis=1)
    long = np.stack([np.linspace(30, 60, 160), np.zeros(160)], axis=1)
    fit = ransac_line(np.concatenate([short, long]), inlier_m=0.5, min_points=20,
                      min_length=5.0, iterations=300, rng=np.random.default_rng(1),
                      max_gap_m=8.0)
    assert fit is not None
    u = (np.concatenate([short, long])[fit.inliers] - fit.point) @ fit.direction
    assert u.max() - u.min() == pytest.approx(30, abs=2)
    assert len(fit.inliers) == 160


def test_merge_joins_duplicate_slabs_but_not_neighbours_down_a_street():
    d, n = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    same_wall = np.stack([np.linspace(0, 40, 50), np.zeros(50)], axis=1)
    slab = same_wall + np.array([0.0, 1.2])          # same wall, one metre out
    next_door = np.stack([np.linspace(60, 100, 50), np.zeros(50)], axis=1)
    fits = [(np.array([20.0, 0.0]), d, n), (np.array([20.0, 1.2]), d, n),
            (np.array([80.0, 0.0]), d, n)]
    pts = [np.c_[p, np.zeros(len(p))] for p in (same_wall, slab, next_door)]
    groups = merge_coplanar(fits, pts, max_angle_deg=15, max_offset_m=2.5, min_overlap=0.3)
    assert sorted(sorted(g) for g in groups) == [[0, 1], [2]]


def _facades_of(cloud, **kw):
    t = build_terrain(cloud, res_m=1.0)
    buildings, labels = find_buildings(t)
    assert len(buildings) == 1
    return facades_of_building(cloud, t, labels, buildings[0], 1, **kw)


def test_a_box_has_four_walls_with_the_right_sizes():
    facades = _facades_of(np.concatenate([ground(), box(width=30, depth=20, height=12)]))
    assert len(facades) == 4
    assert sorted(round(f.length / 5) * 5 for f in facades) == [20, 20, 30, 30]
    for f in facades:
        assert f.height == pytest.approx(12, abs=1.5)
        assert f.normal[2] == 0.0                                   # walls are vertical
        assert np.linalg.norm(f.normal) == pytest.approx(1.0)
        assert abs(np.dot(f.normal, f.u_axis)) < 1e-9               # u runs along the wall
    assert sorted(octant_of(f.azimuth_deg) for f in facades) == ["E", "N", "S", "W"]


def test_facade_normals_point_away_from_the_building():
    cloud = np.concatenate([ground(), box(centre=(10, -5), width=30, depth=20, height=12)])
    facades = _facades_of(cloud)
    centre = np.array([10.0, -5.0, 6.0])
    for f in facades:
        # Stepping out along the normal must take you further from the centre.
        here = np.asarray(f.origin) + np.array([0, 0, 6.0])
        assert (np.linalg.norm(here + np.asarray(f.normal) - centre)
                > np.linalg.norm(here - centre))
        assert f.distance_to_plane(centre) < 0                      # centre is behind the wall


def test_rotated_box_keeps_its_right_angles():
    facades = _facades_of(np.concatenate([ground(), box(width=34, depth=22, height=12,
                                                        yaw_deg=25)]))
    assert len(facades) == 4
    az = sorted(f.azimuth_deg for f in facades)
    for a, b in zip(az, az[1:]):
        assert b - a == pytest.approx(90, abs=6)
    # A turned box rasterises to a staircase, so each wall loses about a cell at
    # either end. Facade extents are good to roughly the grid resolution, no better.
    assert sorted(f.length for f in facades) == pytest.approx([22, 22, 34, 34], abs=2.5)


def test_facade_ids_number_octants_left_to_right():
    facades = _facades_of(np.concatenate([ground(), box(width=30, depth=20, height=12)]))
    for f in facades:
        assert f.id.startswith("B01/")
        assert f.name == octant_of(f.azimuth_deg) + "1"             # one wall per octant here
    assert len({f.id for f in facades}) == 4


def test_uv_round_trip_on_the_facade_plane():
    f = _facades_of(np.concatenate([ground(), box(width=30, depth=20, height=12)]))[0]
    for u, v in ((0.0, f.v_ground + 3), (f.u_max - 1, f.v_ground + 8)):
        p = f.point(u, v)
        got_u, got_v = f.uv_of(p)
        assert float(got_u) == pytest.approx(u, abs=1e-9)
        assert float(got_v) == pytest.approx(v, abs=1e-9)
        assert abs(float(f.distance_to_plane(p))) < 1e-9


def test_noisy_walls_do_not_split_into_parallel_slabs():
    """Scatter about the true surface is what makes one wall look like several."""
    g = ground()
    g[:, 2] += RNG.normal(0.0, 0.25, len(g))        # the whole cloud is noisy, as a real one is
    cloud = np.concatenate([g, box(width=30, depth=20, height=12, noise=0.25)])
    facades = _facades_of(cloud, inlier_m=1.2)
    assert len(facades) == 4
    assert len({octant_of(f.azimuth_deg) for f in facades}) == 4


# ── coverage ─────────────────────────────────────────────────────────────
def _camera_at(position, target, width=1600, height=1200, focal=1500.0) -> CameraPose:
    f = np.asarray(target, float) - np.asarray(position, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    r = np.stack([right, np.cross(f, right), f])
    return CameraPose(name="c", R=r, t=-r @ np.asarray(position, float),
                      intrinsics=Intrinsics("PINHOLE", width, height, focal, focal,
                                            width / 2, height / 2))


@pytest.fixture
def wall():
    return _facades_of(np.concatenate([ground(), box(width=30, depth=20, height=12)]))


def test_coverage_counts_a_camera_looking_straight_at_the_wall(wall):
    f = next(x for x in wall if octant_of(x.azimuth_deg) == "N")
    centre = f.point(0.5 * (f.u_min + f.u_max), f.v_ground + 6)
    cam = _camera_at(centre + np.asarray(f.normal) * 30.0, centre)
    cov, grid = facade_coverage(f, [cam], cell_m=0.5, max_gsd_m=0.05)
    assert cov.visible > 0.5 * cov.cells
    assert cov.covered == cov.visible                     # 30 m / 1500 px = 2 cm per pixel
    assert cov.best_gsd_cm == pytest.approx(2.0, abs=0.3)
    assert grid.max() == 1 and not cov.too_coarse


def test_coverage_reports_a_capture_that_is_visible_but_too_coarse(wall):
    f = next(x for x in wall if octant_of(x.azimuth_deg) == "N")
    centre = f.point(0.5 * (f.u_min + f.u_max), f.v_ground + 6)
    cam = _camera_at(centre + np.asarray(f.normal) * 200.0, centre, focal=800.0)
    cov, _ = facade_coverage(f, [cam], cell_m=0.5, max_gsd_m=0.05)
    assert cov.visible > 0 and cov.covered == 0           # 200 m / 800 px = 25 cm per pixel
    assert cov.too_coarse and cov.best_gsd_cm == pytest.approx(25, abs=2)


def test_a_camera_behind_the_wall_sees_nothing(wall):
    f = next(x for x in wall if octant_of(x.azimuth_deg) == "N")
    centre = f.point(0.5 * (f.u_min + f.u_max), f.v_ground + 6)
    cam = _camera_at(centre - np.asarray(f.normal) * 40.0, centre)
    cov, _ = facade_coverage(f, [cam], cell_m=0.5)
    assert cov.visible == 0 and cov.covered == 0 and cov.best_gsd_cm is None


def test_a_camera_too_far_to_the_side_fails_the_angle_test(wall):
    f = next(x for x in wall if octant_of(x.azimuth_deg) == "N")
    centre = f.point(0.5 * (f.u_min + f.u_max), f.v_ground + 6)
    off = np.asarray(f.normal) * 10.0 + np.asarray(f.u_axis) * 60.0      # ~80 deg off normal
    cov, _ = facade_coverage(f, [_camera_at(centre + off, centre)], cell_m=0.5,
                             max_angle_deg=60.0)
    assert cov.visible == 0


def test_facade_grid_covers_the_wall(wall):
    f = wall[0]
    pts, nu, nv = facade_grid(f, 0.5)
    assert len(pts) == nu * nv
    assert np.abs(f.distance_to_plane(pts)).max() < 1e-9
    u, v = f.uv_of(pts)
    assert u.min() > f.u_min and u.max() < f.u_max
    assert v.min() > f.v_ground and v.max() < f.v_max



# ── the coverage report of the stage ─────────────────────────────────────
def _square(cx: float, cy: float, half: float) -> list[tuple[float, float]]:
    return [(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half),
            (cx - half, cy + half)]


def _covered_facade(fid: str, *, cells=100, visible=100, covered=100, best=None) -> Facade:
    cov = Coverage(cell_m=0.5, cells=cells, visible=visible, covered=covered,
                   mean_views=1.0 if covered else 0.0, max_gsd_cm=16.0, best_gsd_cm=best)
    return Facade(id=fid, building_id=fid.split("/")[0], origin=(0, 0, 0), normal=(0, -1, 0),
                  u_axis=(1, 0, 0), azimuth_deg=180.0, u_min=0.0, u_max=10.0, v_ground=0.0,
                  v_max=10.0, plane_rms_m=0.1, n_points=100, coverage=cov)


def _site_like_dataset_1910():
    """A tower at the origin seen sharply on all 12 facades; 30 far buildings seen only from
    far away (at 9 cm/px on the tower vs 20-40 cm/px elsewhere, a 16 cm/px bar)."""
    buildings, facades = [], []
    tower = [_covered_facade(f"B01/F{k}", best=13.0) for k in range(12)]
    buildings.append(Building(id="B01", footprint=_square(5, 5, 20), area_m2=1600, ground_z=0,
                              roof_z=44, height_m=44, facade_ids=[f.id for f in tower]))
    facades += tower
    for n in range(2, 32):
        far = [_covered_facade(f"B{n:02d}/F{k}", covered=0, best=20.0 + n) for k in range(4)]
        buildings.append(Building(id=f"B{n:02d}", footprint=_square(200 + 30 * n, 0, 8),
                                  area_m2=256, ground_z=0, roof_z=15, height_m=15,
                                  facade_ids=[f.id for f in far]))
        facades += far
    return buildings, facades


def test_coverage_report_counts_facades_instead_of_a_site_wide_median():
    buildings, facades = _site_like_dataset_1910()
    # The old message: the median over every facade, which is 0 % here.
    assert float(np.median([f.coverage.fraction for f in facades])) == 0.0
    lines, warnings = coverage_report(buildings, facades, max_gsd_m=0.16, n_cameras=151)
    assert "12 із 132 фасадів" in lines[0] and "повністю (≥95% клітин) 12" in lines[0]
    assert "лише грубіше за поріг 120" in lines[0]
    tower = next(line for line in lines if line.strip().startswith("B01"))
    assert "12/12 фасадів" in tower and "медіана 100%" in tower and "13 см/px" in tower
    assert "0 м від початку" in tower                       # the origin is inside its footprint


def test_coverage_warning_quotes_the_gsd_of_the_coarse_facades_not_the_sites_best():
    buildings, facades = _site_like_dataset_1910()
    _, warnings = coverage_report(buildings, facades, max_gsd_m=0.16, n_cameras=151)
    assert len(warnings) == 1
    w = warnings[0]
    assert w.startswith("120 із 132 фасадів")
    assert "найкраще 22.0 см/px" in w                      # the coarse ones, not 13 (the tower)
    assert "13" not in w.split("(")[1].split(")")[0]
    assert "з потрібною деталізацією 12" in w


def test_coverage_report_lists_the_buildings_nearest_the_origin_first():
    buildings, facades = _site_like_dataset_1910()
    lines, _ = coverage_report(list(reversed(buildings)), facades, max_gsd_m=0.16, n_cameras=1,
                               near=3)
    listed = [line.strip().split(" ")[0] for line in lines if line.startswith("  ")]
    assert listed == ["B01", "B02", "B03"]


def test_coverage_report_without_any_coverage_or_with_nothing_seen():
    buildings, facades = _site_like_dataset_1910()
    bare = [f.model_copy(update={"coverage": None}) for f in facades]
    assert coverage_report(buildings, bare, max_gsd_m=0.16, n_cameras=0) == ([], [])
    unseen = [_covered_facade("B01/F0", visible=0, covered=0, best=None)]
    lines, warnings = coverage_report(buildings[:1], unseen, max_gsd_m=0.16, n_cameras=3)
    assert "0 із 1 фасадів" in lines[0] and "не видно 1" in lines[0]
    assert lines[-1].strip().endswith("не видно") and not warnings


def test_coverage_report_does_not_warn_when_most_facades_are_sharp():
    buildings, facades = _site_like_dataset_1910()
    sharp = [f.model_copy(update={"coverage": f.coverage.model_copy(update={"covered": 50})})
             for f in facades]
    _, warnings = coverage_report(buildings, sharp, max_gsd_m=0.16, n_cameras=151)
    assert warnings == []


def test_distance_to_footprint_is_zero_inside_and_euclidean_outside():
    sq = _square(0, 0, 10)
    assert distance_to_footprint(sq, (0, 0)) == 0.0
    assert distance_to_footprint(sq, (13, 0)) == pytest.approx(3.0)
    assert distance_to_footprint(sq, (13, 14)) == pytest.approx(5.0)

# ── COLMAP model + frame transfer ────────────────────────────────────────
def test_read_a_text_colmap_model(tmp_path):
    (tmp_path / "cameras.txt").write_text("# c\n1 SIMPLE_PINHOLE 640 480 500 320 240\n")
    (tmp_path / "images.txt").write_text(
        "# i\n1 1 0 0 0 1 2 3 1 a.jpg\n\n2 1 0 0 0 5 5 5 1 b.jpg\n\n", encoding="utf-8")
    cams = read_model(tmp_path)
    assert [c.name for c in cams] == ["a.jpg", "b.jpg"]
    k = cams[0].intrinsics
    assert (k.fx, k.fy, k.cx, k.cy, k.width) == (500, 500, 320, 240, 640)
    assert np.allclose(cams[0].R, np.eye(3))
    assert np.allclose(cams[0].centre, [-1, -2, -3])


def test_projection_and_framing():
    cam = _camera_at([0, -20, 0], [0, 0, 0], width=640, height=480, focal=500)
    uv, z = cam.project(np.array([[0.0, 0.0, 0.0]]))
    assert uv[0] == pytest.approx([320, 240]) and z[0] == pytest.approx(20)
    assert cam.sees(np.array([[0.0, 0.0, 0.0]]))[0]
    assert not cam.sees(np.array([[0.0, -40.0, 0.0]]))[0]       # behind the camera
    assert not cam.sees(np.array([[50.0, 0.0, 0.0]]))[0]        # outside the frame


def test_pose_carried_through_an_affine_projects_the_same_point():
    from aerosplat.geo.affine import AffineMap

    cam = _camera_at([5, -30, 12], [0, 0, 2], width=800, height=600, focal=900)
    a = AffineMap(np.diag([1.0025, 0.9987, 1.0]), np.array([0.4, -0.2, 0.1]))
    moved = transform_pose(cam, a)
    world = np.array([[3.0, 1.0, 4.0], [-6.0, 2.5, 0.0]])
    before, _ = cam.project(world)
    after, _ = moved.project(a.apply(world))
    assert np.abs(after - before).max() < 0.5                   # sub-pixel at 800x600
    assert np.allclose(moved.R @ moved.R.T, np.eye(3), atol=1e-12)


# ── pipeline wiring ──────────────────────────────────────────────────────
def test_semantics_stage_is_registered_before_occupancy():
    from aerosplat.jobs.models import PIPELINE_STAGES
    from aerosplat.pipeline.stages import STAGES

    assert PIPELINE_STAGES[-2:] == ("semantics", "occupancy")      # occupancy reads its terrain
    assert list(STAGES) == list(PIPELINE_STAGES)
    assert STAGES["semantics"].entry({}) is not None            # module imports, main() exists


def test_semantics_argv_points_at_the_mapped_cloud_and_the_run_dir(es_site):
    from aerosplat.config import RunConfig
    from aerosplat.pipeline.layout import RunLayout
    from aerosplat.pipeline.stages import STAGES, argv_semantics

    cat, site, cap, _ = es_site
    cfg = RunConfig(camera_source="earthstudio_json", cloud_source="dense").to_dict()
    L = RunLayout.for_site_run(cfg, site=site, site_dirs=cat.site_dirs(site), capture=cap,
                               epoch_id="baseline", run_id="r1")
    a = argv_semantics(cfg, L)
    # Earth Studio reconstructs in sphere ENU, so semantics must read the mapped
    # products and carry the cameras over with the same affine.
    assert a[a.index("--cloud") + 1] == str(L.cloud_wgs84)
    assert a[a.index("--registration") + 1] == str(L.registration_report)
    assert a[a.index("--out-dir") + 1] == str(cat.run_dirs(site, "baseline", "r1").root
                                              / "semantics")
    assert a[a.index("--sparse") + 1] == str(L.sparse)
    assert STAGES["semantics"].skip_reason(cfg) is None

    sparse_cfg = RunConfig(camera_source="earthstudio_json", cloud_source="sparse").to_dict()
    assert "замало" in (STAGES["semantics"].skip_reason(sparse_cfg) or "")


def test_semantics_argv_without_known_poses_uses_the_unmapped_cloud(tmp_path):
    from aerosplat.config import RunConfig
    from aerosplat.pipeline.layout import RunLayout
    from aerosplat.pipeline.stages import argv_semantics

    cfg = RunConfig(camera_source="synthetic", cloud_source="dense").to_dict()
    L = RunLayout.from_config(cfg, data_dir=tmp_path / "data", project_dir=tmp_path)
    a = argv_semantics(cfg, L)
    assert a[a.index("--cloud") + 1] == str(L.cloud_in)
    assert "--registration" not in a            # no sphere ENU detour to undo
