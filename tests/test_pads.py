"""Pad zones on roofs and in yards (ACCESS_POINTS §5.7) on synthetic rasters with known answers.

A flat roof with a chimney, a pitched roof that must be refused, a yard with a tree, a
courtyard too deep to see the sky from, and a roof nobody saw.
"""

import math

import numpy as np
import pytest

from aerosplat.semantics.models import Building, TerrainModel
from aerosplat.semantics.pads import (PadParams, Rasters, find_pads, plane_fit, point_in_polygon,
                                      polygon_overlap, sky_view)

N = 100


def scene():
    """A 100 × 100 m ground plane at z = 0, 1-m cells, every cell observed."""
    model = TerrainModel(res_m=1.0, x0=0.0, y0=0.0, rows=N, cols=N, n_points=N * N * 10,
                         filled_cells=N * N)
    dsm = np.zeros((N, N))
    dtm = np.zeros((N, N))
    count = np.full((N, N), 10, np.int64)
    labels = np.zeros((N, N), np.int64)
    return model, dsm, dtm, count, labels


def building(bid, rows, cols, roof_z):
    i0, i1 = rows
    j0, j1 = cols
    return Building(id=bid, footprint=[(j0, i0), (j1, i0), (j1, i1), (j0, i1)],
                    area_m2=float((i1 - i0) * (j1 - j0)), ground_z=0.0, roof_z=roof_z,
                    height_m=roof_z)


def put_block(dsm, labels, rows, cols, z, label):
    dsm[rows[0]:rows[1], cols[0]:cols[1]] = z
    labels[rows[0]:rows[1], cols[0]:cols[1]] = label


def cell_xy(i, j):
    return j + 0.5, i + 0.5


def test_a_flat_roof_with_a_chimney_is_a_pad_that_keeps_clear_of_the_chimney():
    model, dsm, dtm, count, labels = scene()
    put_block(dsm, labels, (20, 40), (20, 40), 10.0, 1)
    # A 2 × 2 m chimney, 1.5 m above the roof; the stage's roughness gate drops it from
    # the building mask, which is exactly how it arrives here.
    dsm[24:26, 24:26] = 11.5
    labels[24:26, 24:26] = 0
    r = find_pads(Rasters(model, dsm, dtm, count, labels), [building("B01", (20, 40), (20, 40), 10.0)],
                  ground=False)
    roof = [z for z in r.zones if z.kind == "roof"]
    assert len(roof) == 1, r.rejected
    z = roof[0]
    assert z.building_id == "B01" and z.z == pytest.approx(10.0)
    assert z.slope_deg == pytest.approx(0.0, abs=1e-6) and z.rms_m == pytest.approx(0.0, abs=1e-9)
    # The landing point keeps the clearance from the chimney and from the roof edge.
    chimney = np.array([25.0, 25.0])                 # its centre (x = col, y = row)
    c = np.array(z.centre)
    assert np.max(np.abs(c - chimney)) - 1.0 >= PadParams().clearance_m
    assert min(c[0] - 20, 40 - c[0], c[1] - 20, 40 - c[1]) >= PadParams().clearance_m
    # The zone never contains the chimney, and every vertex of it keeps clear of it.
    poly = np.array(z.polygon_enu)
    assert not point_in_polygon(np.array([25.0]), np.array([25.0]), z.polygon_enu)[0]
    # Euclidean distance from each outline vertex to the chimney square; vertices sit on
    # cell corners, half a cell's diagonal from the centres the clearance is measured at.
    gap = np.hypot(*np.maximum(np.abs(poly - chimney) - 1.0, 0.0).T)
    assert (gap >= PadParams().clearance_m - math.sqrt(2) / 2 - 1e-9).all()
    assert 0 < z.usable_area_m2 < z.free_area_m2 < 400


def test_without_the_chimney_the_zone_is_the_roof_eroded_by_the_clearance():
    model, dsm, dtm, count, labels = scene()
    put_block(dsm, labels, (20, 40), (20, 40), 10.0, 1)
    r = find_pads(Rasters(model, dsm, dtm, count, labels), [building("B01", (20, 40), (20, 40), 10.0)],
                  ground=False)
    (z,) = r.zones
    xs = [p[0] for p in z.polygon_enu]
    ys = [p[1] for p in z.polygon_enu]
    # Cells whose centre is ≥ 2 m from the edge: centres 22.5…37.5, so cells 22…37.
    assert (min(xs), max(xs), min(ys), max(ys)) == (22.0, 38.0, 22.0, 38.0)
    assert z.usable_area_m2 == 16 * 16 and z.centre == pytest.approx((30.5, 30.5), abs=1.0)


def test_a_stepped_roof_gives_one_zone_per_level_not_one_rough_plane():
    model, dsm, dtm, count, labels = scene()
    put_block(dsm, labels, (20, 40), (20, 60), 10.0, 1)
    dsm[20:40, 40:60] = 13.0                          # a 3-m higher wing, same building
    r = find_pads(Rasters(model, dsm, dtm, count, labels), [building("B01", (20, 40), (20, 60), 13.0)],
                  ground=False)
    assert sorted(round(z.z) for z in r.zones) == [10, 13], r.rejected
    assert all(z.rms_m < 1e-9 for z in r.zones)


def test_a_parapet_keeps_the_zone_away_from_the_roof_edge():
    model, dsm, dtm, count, labels = scene()
    put_block(dsm, labels, (20, 40), (20, 40), 10.0, 1)
    dsm[20, 20:40] = dsm[39, 20:40] = 11.0            # a 1-m parapet along two sides
    r = find_pads(Rasters(model, dsm, dtm, count, labels), [building("B01", (20, 40), (20, 40), 10.0)],
                  ground=False)
    (z,) = r.zones
    ys = [p[1] for p in z.polygon_enu]
    assert min(ys) >= 21 + PadParams().clearance_m - 0.5 and max(ys) <= 39 - PadParams().clearance_m + 0.5


def test_a_pitched_roof_is_refused_with_its_slope():
    model, dsm, dtm, count, labels = scene()
    jj = np.arange(N)
    pitch = 10.0 + math.tan(math.radians(30)) * (jj - 50)
    dsm[20:40, 50:70] = pitch[50:70][None, :]
    labels[20:40, 50:70] = 2
    r = find_pads(Rasters(model, dsm, dtm, count, labels), [building("B02", (20, 40), (50, 70), 21.0)],
                  ground=False)
    assert r.zones == []
    (rej,) = r.rejected
    assert rej.code == "slope" and rej.building_id == "B02" and "30°" in rej.reason


def test_a_gently_sloped_roof_above_the_land_limit_is_refused_too():
    model, dsm, dtm, count, labels = scene()
    jj = np.arange(N)
    dsm[20:40, 50:70] = (10.0 + math.tan(math.radians(8)) * (jj - 50))[50:70][None, :]
    labels[20:40, 50:70] = 2
    r = find_pads(Rasters(model, dsm, dtm, count, labels), [building("B02", (20, 40), (50, 70), 13.0)],
                  ground=False)
    assert r.zones == [] and r.rejected[0].code == "slope"


def test_a_rough_roof_is_refused_with_its_residual():
    model, dsm, dtm, count, labels = scene()
    put_block(dsm, labels, (20, 40), (20, 40), 10.0, 1)
    rng = np.random.default_rng(3)
    dsm[20:40, 20:40] += rng.uniform(-0.12, 0.12, (20, 20))       # gravel heaps, rubble
    r = find_pads(Rasters(model, dsm, dtm, count, labels), [building("B01", (20, 40), (20, 40), 10.0)],
                  ground=False)
    assert r.zones == [] and {x.code for x in r.rejected} == {"rough"}


def test_a_roof_nobody_saw_is_not_a_pad():
    model, dsm, dtm, count, labels = scene()
    put_block(dsm, labels, (20, 40), (20, 40), 10.0, 1)
    count[20:40, 20:40] = 0
    dsm[20:40, 20:40] = np.nan
    r = find_pads(Rasters(model, dsm, dtm, count, labels), [building("B01", (20, 40), (20, 40), 10.0)],
                  ground=False)
    assert r.zones == [] and r.rejected[0].code == "no_data"


def test_a_small_roof_has_no_room_for_the_clearance():
    model, dsm, dtm, count, labels = scene()
    put_block(dsm, labels, (20, 24), (20, 30), 10.0, 1)           # 4 m deep
    r = find_pads(Rasters(model, dsm, dtm, count, labels), [building("B01", (20, 24), (20, 30), 10.0)],
                  ground=False)
    assert r.zones == [] and r.rejected[0].code == "no_room"


def test_a_yard_with_a_tree_keeps_clear_of_the_canopy():
    model, dsm, dtm, count, labels = scene()
    put_block(dsm, labels, (40, 60), (40, 60), 12.0, 1)
    ii, jj = np.mgrid[0:N, 0:N]
    tree = np.hypot(ii + 0.5 - 30.0, jj + 0.5 - 50.0) <= 4.0          # canopy south of the house
    rng = np.random.default_rng(1)
    dsm[tree] = 6.0 + rng.uniform(-0.8, 0.8, tree.sum())
    r = find_pads(Rasters(model, dsm, dtm, count, labels), [building("B01", (40, 60), (40, 60), 12.0)],
                  roofs=False)
    ground = [z for z in r.zones if z.kind == "ground"]
    assert ground, r.rejected
    for z in ground:
        assert z.building_id == "B01" and z.z == pytest.approx(0.0) and z.sky_view > 0.5
        assert "дороги" in z.notes[0]                            # what we cannot tell, said
        poly = np.array(z.polygon_enu)
        d = np.hypot(poly[:, 0] - 50.0, poly[:, 1] - 30.0)
        # Outline vertices sit on cell corners: half a cell's diagonal of slack.
        assert d.min() >= 4.0 + PadParams().clearance_m - 0.75
        cx, cy = z.centre
        assert math.hypot(cx - 50.0, cy - 30.0) >= 4.0 + PadParams().clearance_m - 0.5
        # ... and keeps clear of the house too.
        assert not (40 - 2 < cx < 60 + 2 and 40 - 2 < cy < 60 + 2)


def test_a_deep_courtyard_is_refused_for_its_sky():
    model, dsm, dtm, count, labels = scene()
    put_block(dsm, labels, (20, 80), (20, 44), 40.0, 1)
    put_block(dsm, labels, (20, 80), (52, 76), 40.0, 2)        # an 8-m slot between 40-m walls
    params = PadParams(yard_radius_m=6.0)
    r = find_pads(Rasters(model, dsm, dtm, count, labels), [building("B01", (20, 80), (20, 44), 40.0)],
                  roofs=False, params=params)
    slot = [z for z in r.zones if 44 < z.centre[0] < 52]
    assert slot == []
    assert any(x.code == "sky" and 44 < x.centre[0] < 52 for x in r.rejected)


def test_only_the_requested_buildings_are_searched():
    model, dsm, dtm, count, labels = scene()
    put_block(dsm, labels, (20, 40), (20, 40), 10.0, 1)
    put_block(dsm, labels, (60, 80), (60, 80), 10.0, 2)
    b = [building("B01", (20, 40), (20, 40), 10.0), building("B02", (60, 80), (60, 80), 10.0)]
    r = find_pads(Rasters(model, dsm, dtm, count, labels), b, building_ids=["B02"], ground=False)
    assert {z.building_id for z in r.zones} == {"B02"}


def test_helpers():
    x, y = np.meshgrid(np.arange(5.0), np.arange(5.0))
    slope, rms = plane_fit(x.ravel(), y.ravel(), (0.1 * x).ravel())
    assert slope == pytest.approx(math.degrees(math.atan(0.1))) and rms == pytest.approx(0.0, abs=1e-12)
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert polygon_overlap(sq, [(5, 5), (15, 5), (15, 15), (5, 15)]) == pytest.approx(0.25, abs=0.02)
    assert polygon_overlap(sq, [(20, 20), (30, 20), (30, 30)]) == 0.0
    open_field = np.zeros((50, 50))
    assert sky_view(open_field, 25, 25, 1.0, 1.0, 20.0, 16) == pytest.approx(1.0)
    pit = np.full((50, 50), 30.0)
    pit[20:30, 20:30] = 0.0
    assert sky_view(pit, 25, 25, 1.0, 1.0, 20.0, 16) < 0.2
