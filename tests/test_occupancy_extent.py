"""Where the occupancy grid is built (aerosplat.surface.extent) and what the stage does with it.

dataset_1910 (Earth Studio, one 15-storey tower) reconstructed 1.6 × 1.8 km; a grid over all of
it was 2.6 billion voxels and the stage failed after 53 s. The default is now a stated extent
(the Site's bbox, or a square around the Site origin), recorded in occupancy.json; the grid is
never cropped to fit, and the camera visibility is read only once the grid is known to fit.
"""

import math

import numpy as np
import pytest

from aerosplat.geo.crs import EnuFrame, GeoPoint
from aerosplat.store.models import BBox
from aerosplat.surface.extent import choose_extent, parse_site_bbox, site_bbox_enu
from aerosplat.surface.occupancy import read_occupancy

ORIGIN = GeoPoint(lat=39.76, lon=-75.574, h=-30.0)


# ── choosing ─────────────────────────────────────────────────────────────
def site_box(half_m: float, origin: GeoPoint = ORIGIN) -> BBox:
    """A WGS84 box about ±half_m around the origin."""
    dlat = half_m / 111_320.0
    dlon = half_m / (111_320.0 * math.cos(math.radians(origin.lat)))
    return BBox(west=origin.lon - dlon, south=origin.lat - dlat, east=origin.lon + dlon,
                north=origin.lat + dlat)


def test_precedence_bbox_then_site_bbox_then_radius():
    sb = site_box(80.0)
    c = choose_extent(bbox="-10,-20,30,40", radius_m=150, site_bbox=sb, origin=ORIGIN)
    assert (c.source, c.box) == ("bbox", (-10.0, -20.0, 30.0, 40.0))
    c = choose_extent(radius_m=150, site_bbox=sb, origin=ORIGIN)
    assert c.source == "site_bbox"
    assert c.box == pytest.approx((-80, -80, 80, 80), abs=0.5)
    c = choose_extent(radius_m=150)
    assert (c.source, c.box) == ("radius", (-150, -150, 150, 150))
    assert "±150" in c.why and "немає bbox" in c.why


def test_explicit_modes():
    sb = site_box(80.0)
    c = choose_extent(extent="radius", radius_m=60, site_bbox=sb, origin=ORIGIN)
    assert (c.source, c.box) == ("radius", (-60, -60, 60, 60))
    c = choose_extent(extent="raster", site_bbox=sb, origin=ORIGIN)
    assert (c.source, c.box) == ("raster", None)


@pytest.mark.parametrize("kw, match", [
    (dict(extent="raster", bbox="0,0,1,1"), "contradict"),
    (dict(extent="radius", bbox="0,0,1,1"), "contradict"),
    (dict(extent="everything"), "extent"),
    (dict(radius_m=0.0), "radius"),
    (dict(radius_m=float("nan")), "radius"),
    (dict(site_bbox=site_box(10.0)), "origin"),
    (dict(bbox="1,2,3"), "bbox"),
])
def test_contradictory_or_incomplete_choices_fail(kw, match):
    with pytest.raises(ValueError, match=match):
        choose_extent(**kw)


def test_site_bbox_maps_to_an_enu_box_that_holds_it():
    sb = BBox(west=-75.576, south=39.758, east=-75.571, north=39.763)
    box = site_bbox_enu(sb, ORIGIN)
    frame = EnuFrame(ORIGIN)
    lat = np.linspace(sb.south, sb.north, 101)
    lon = np.linspace(sb.west, sb.east, 101)
    gl, go = np.meshgrid(lat, lon)
    enu = frame.from_geodetic(gl.ravel(), go.ravel(), np.full(gl.size, ORIGIN.h))
    assert enu[:, 0].min() >= box[0] - 1e-6 and enu[:, 0].max() <= box[2] + 1e-6
    assert enu[:, 1].min() >= box[1] - 1e-6 and enu[:, 1].max() <= box[3] + 1e-6
    assert box[2] - box[0] == pytest.approx(0.005 * 111_320 * math.cos(math.radians(39.76)),
                                            rel=0.01)


def test_parse_site_bbox():
    b = parse_site_bbox("-75.576, 39.758,-75.571,39.763")
    assert (b.west, b.south, b.east, b.north) == (-75.576, 39.758, -75.571, 39.763)
    for bad in ("1,2,3", "a,b,c,d", "-75.5,39.8,-75.6,39.9", "0,50,1,40"):
        with pytest.raises(ValueError):
            parse_site_bbox(bad)


# ── the stage ────────────────────────────────────────────────────────────
HALF = 60.0                     # the "cloud" reaches ±60 m: much more than the building
BUILDING = (-8.2, 9.7, -6.1, 5.8, 12.0)     # x0, x1, y0, y1, height


def wide_scene(step: float = 0.25) -> np.ndarray:
    g = np.arange(-HALF + step / 2, HALF, step)
    gx, gy = np.meshgrid(g, g)
    ground = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    x0, x1, y0, y1, h = BUILDING
    ground = ground[~((ground[:, 0] > x0) & (ground[:, 0] < x1)
                      & (ground[:, 1] > y0) & (ground[:, 1] < y1))]
    zs = np.append(np.arange(0.0, h, step), h)
    walls = []
    for x in np.append(np.arange(x0, x1, step), x1):
        walls += [(x, y0, z) for z in zs] + [(x, y1, z) for z in zs]
    for y in np.append(np.arange(y0, y1, step), y1):
        walls += [(x0, y, z) for z in zs] + [(x1, y, z) for z in zs]
    rx, ry = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    roof = np.stack([rx.ravel(), ry.ravel(), np.full(rx.size, h)], axis=1)
    return np.concatenate([ground, np.array(walls, float), roof])


@pytest.fixture(scope="module")
def wide(tmp_path_factory):
    torch = pytest.importorskip("torch")
    from aerosplat.semantics.stage import run_semantics

    root = tmp_path_factory.mktemp("wide")
    cloud = root / "dense_wgs84.pt"
    torch.save({"xyz": torch.tensor(wide_scene(), dtype=torch.float32)}, cloud)
    run_semantics(cloud, root / "semantics")
    return cloud, root / "semantics"


def run(wide, out, **kw):
    from aerosplat.surface.stage import run_occupancy

    cloud, sem = wide
    return run_occupancy(cloud, sem, out, carve=False, **kw)


def test_default_is_a_square_around_the_origin_and_it_is_recorded(wide, tmp_path):
    report = run(wide, tmp_path / "occ", radius_m=20.0)
    g = report.grid
    assert (g.x0, g.y0) == pytest.approx((-20.0, -20.0)) and (g.x1, g.y1) == pytest.approx((20, 20))
    e = report.extent
    assert e is not None and e.source == "radius" and e.requested == (-20, -20, 20, 20)
    assert e.grid == pytest.approx((-20, -20, 20, 20)) and e.clipped == []
    assert e.raster[0] <= -HALF + 1 and e.raster[2] >= HALF - 1
    assert "±20" in e.detail and report.params["extent"] == "auto"
    _, again = read_occupancy(tmp_path / "occ")
    assert again.extent == e


def test_raster_extent_takes_everything_the_cloud_covers(wide, tmp_path):
    report = run(wide, tmp_path / "occ", extent="raster")
    assert report.extent.source == "raster" and report.extent.requested is None
    assert report.grid.x1 - report.grid.x0 >= 2 * HALF - 1


def test_site_bbox_is_the_default_when_the_site_has_one(wide, tmp_path):
    sb = site_box(25.0)
    report = run(wide, tmp_path / "occ", radius_m=10.0,
                 site_bbox=f"{sb.west!r},{sb.south!r},{sb.east!r},{sb.north!r}",
                 origin=",".join(repr(v) for v in ORIGIN.as_tuple()))
    assert report.extent.source == "site_bbox"
    assert report.grid.x0 == pytest.approx(-25.0, abs=0.6)
    assert report.grid.y1 == pytest.approx(25.0, abs=0.6)


def test_an_extent_beyond_the_cloud_is_reported_not_silently_smaller(wide, tmp_path):
    report = run(wide, tmp_path / "occ", radius_m=100.0)
    assert sorted(report.extent.clipped) == sorted(["захід", "південь", "схід", "північ"])
    assert any("закінчується раніше" in w for w in report.warnings)


def test_buildings_cut_by_the_grid_edge_are_named(wide, tmp_path):
    report = run(wide, tmp_path / "occ", radius_m=6.0)
    assert any("перетинають межу сітки" in w and "B01" in w for w in report.warnings)
    full = run(wide, tmp_path / "occ2", radius_m=20.0)
    assert not any("перетинають межу сітки" in w for w in full.warnings)


def test_over_budget_fails_with_a_hint_before_reading_the_visibility(wide, tmp_path, monkeypatch):
    from aerosplat.surface import stage

    loaded: list[int] = []
    monkeypatch.setattr(stage, "load_rays", lambda *a: loaded.append(1) or (None, "test"))
    cloud, sem = wide
    with pytest.raises(SystemExit) as e:
        stage.run_occupancy(cloud, sem, tmp_path / "occ", radius_m=50.0, max_voxels=100_000,
                            fused=tmp_path / "fused.ply")
    msg = str(e.value.code)
    assert "max_voxels" in msg and "±50" in msg and "occupancy_radius" in msg
    assert "не обрізано" in msg
    assert loaded == []                                  # failed before the slow part
    assert not (tmp_path / "occ" / "occupancy.json").exists()
    # Within budget the visibility is read, once.
    stage.run_occupancy(cloud, sem, tmp_path / "occ", radius_m=10.0, fused=tmp_path / "fused.ply")
    assert loaded == [1]


def test_only_extent_failures_carry_the_extent_hint(wide, tmp_path):
    with pytest.raises(SystemExit) as e:
        run(wide, tmp_path / "occ", res_m=0.3)                 # does not divide the 1 m cell
    assert "divide" in str(e.value.code) and "occupancy_radius" not in str(e.value.code)
    with pytest.raises(SystemExit) as e:
        run(wide, tmp_path / "occ", bbox="500,500,600,600")    # off the cloud
    assert "overlap" in str(e.value.code) and "occupancy_radius" in str(e.value.code)


def test_stage_cli_takes_the_extent_options(wide, tmp_path):
    from aerosplat.surface import stage

    cloud, sem = wide
    assert stage.main(["--cloud", str(cloud), "--semantics-dir", str(sem), "--out-dir",
                       str(tmp_path / "occ"), "--no-carve", "--extent", "radius",
                       "--radius", "15"]) == 0
    _, report = read_occupancy(tmp_path / "occ")
    assert report.extent.source == "radius" and report.grid.x0 == pytest.approx(-15.0)
    with pytest.raises(SystemExit, match="contradict"):
        stage.main(["--cloud", str(cloud), "--semantics-dir", str(sem), "--out-dir",
                    str(tmp_path / "occ"), "--extent", "raster", "--bbox=-5,-5,5,5"])
    with pytest.raises(SystemExit, match="origin"):
        stage.main(["--cloud", str(cloud), "--semantics-dir", str(sem), "--out-dir",
                    str(tmp_path / "occ"), "--site-bbox=-75.58,39.75,-75.57,39.77"])


# ── the pipeline ─────────────────────────────────────────────────────────
def test_config_extent_options():
    from pydantic import ValidationError

    from aerosplat.config import RunConfig

    c = RunConfig()
    assert (c.occupancy_extent, c.occupancy_radius, c.occupancy_bbox) == ("auto", 150.0, None)
    RunConfig(occupancy_bbox="-165,-150,135,150")                   # dataset_1910 by hand
    for bad in (dict(occupancy_extent="raster", occupancy_bbox="0,0,1,1"),
                dict(occupancy_extent="radius", occupancy_bbox="0,0,1,1"),
                dict(occupancy_radius=0), dict(occupancy_extent="all")):
        with pytest.raises(ValidationError):
            RunConfig(**bad)


def test_argv_passes_the_extent_and_the_site_bbox(es_site):
    import dataclasses

    from aerosplat.config import RunConfig
    from aerosplat.pipeline import runner
    from aerosplat.pipeline.layout import RunLayout
    from aerosplat.pipeline.stages import STAGES, argv_occupancy

    cat, site, cap, _ = es_site
    cfg = RunConfig(camera_source="earthstudio_json", cloud_source="dense").to_dict()
    L = RunLayout.for_site_run(cfg, site=site, site_dirs=cat.site_dirs(site), capture=cap,
                               epoch_id="baseline", run_id="r1")
    a = argv_occupancy(cfg, L)
    assert "--extent=auto" in a and "--radius=150.0" in a
    assert not any(x.startswith(("--site-bbox", "--bbox")) for x in a)    # the site has none

    sb = site_box(90.0, site.origin)
    Lb = dataclasses.replace(L, site_bbox=sb)
    a = argv_occupancy(cfg, Lb)
    sbox = next(x for x in a if x.startswith("--site-bbox="))
    assert parse_site_bbox(sbox.split("=", 1)[1]) == sb
    origin = next(x for x in a if x.startswith("--origin="))
    assert GeoPoint.parse(origin.split("=", 1)[1]) == site.origin
    # A Site bbox is an input of the stage: a new one must rebuild the grid on --resume,
    # while the other stages keep their fingerprints.
    occ = STAGES["occupancy"]
    assert runner._fingerprint(occ, cfg, L) != runner._fingerprint(occ, cfg, Lb)
    for name in ("sfm", "train", "semantics"):
        assert runner._fingerprint(STAGES[name], cfg, L) == runner._fingerprint(STAGES[name], cfg, Lb)
