"""GIS export (Phase 3.2): known answers on synthetic data.

The question every test asks is the GIS one: does a thing at a known Site ENU point land
at the right longitude/latitude (and ellipsoidal height) in the exported file, read back
with independent tools (rasterio, laspy, pyproj), and is an export without georeference
refused rather than written?
"""

import math

import numpy as np
import pytest
from pyproj import CRS, Proj, Transformer

from aerosplat.export.georef import (ExportRefused, SiteProjection,
                                     georeference_budget, resolve_crs, utm_crs_for)
from aerosplat.export.splat_ply import (inria_arrays, inria_to_activated, property_names,
                                        read_inria_ply, write_inria_ply)
from aerosplat.geo.crs import EnuFrame, GeoPoint
from aerosplat.render import camera as cam

ORIGIN = GeoPoint(lat=40.89104597, lon=-74.21634205, h=233.0)       # dataset_v6
TO_LONLAT = Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True)


def lonlat_of_enu(enu):
    lat, lon, h = EnuFrame(ORIGIN).to_geodetic(np.asarray(enu, float))
    return np.asarray(lon), np.asarray(lat), np.asarray(h)


def metres_between(lon1, lat1, lon2, lat2):
    """Ground distance of two nearby WGS84 positions (local flat approximation)."""
    k = 111_320.0
    return math.hypot((lon1 - lon2) * k * math.cos(math.radians(lat1)), (lat1 - lat2) * k)


# ── CRS choice and projection ────────────────────────────────────────────
def test_utm_zone_of_the_origin():
    assert utm_crs_for(ORIGIN).to_epsg() == 32618
    assert utm_crs_for(GeoPoint(lat=-33.9, lon=18.4)).to_epsg() == 32734     # Cape Town
    # Bergen: EPSG zones by longitude (31), not the MGRS Norway exception (32)
    assert utm_crs_for(GeoPoint(lat=60.4, lon=5.3)).to_epsg() == 32631


@pytest.mark.parametrize("spec, why", [
    ("EPSG:4326", "not a projected"),
    ("EPSG:2263", "not metres"),            # NY State Plane, US survey feet
    ("EPSG:32633", "not defined at the Site"),
    ("EPSG:999999", "unknown CRS"),
])
def test_unusable_crs_is_refused(spec, why):
    with pytest.raises(ExportRefused, match=why):
        resolve_crs(ORIGIN, spec)


def test_other_datum_carries_its_accuracy():
    proj = SiteProjection(ORIGIN, "EPSG:26918")       # NAD83 / UTM 18N
    assert proj.datum_accuracy_m > 0
    assert SiteProjection(ORIGIN).datum_accuracy_m == 0.0


def test_enu_to_utm_is_exact_and_round_trips():
    proj = SiteProjection(ORIGIN)
    enu = np.array([[0.0, 0.0, 0.0], [412.3, -287.9, 35.2], [-600.0, 650.0, -20.0]])
    xyh = proj.enu_to_crs(enu)
    lon, lat, h = lonlat_of_enu(enu)
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True).transform(lon, lat)
    np.testing.assert_allclose(xyh[:, 0], x, atol=1e-6)
    np.testing.assert_allclose(xyh[:, 1], y, atol=1e-6)
    np.testing.assert_allclose(xyh[:, 2], h, atol=1e-6)
    # u is not h: 600 m out the tangent plane is ~5 cm above the ellipsoid
    assert xyh[2, 2] - (ORIGIN.h - 20.0) == pytest.approx(848.5 ** 2 / (2 * 6.371e6), abs=0.01)
    np.testing.assert_allclose(proj.crs_to_enu(xyh[:, 0], xyh[:, 1], xyh[:, 2]), enu, atol=1e-6)


def test_plane_fit_is_the_utm_similarity():
    proj = SiteProjection(ORIGIN)
    fit = proj.fit_plane((-500, 500), (-500, 500), (-50, 100), u_ref=0.0)
    assert fit.residual_max_m < 1e-3
    f = Proj("EPSG:32618").get_factors(ORIGIN.lon, ORIGIN.lat)
    # scale = UTM point scale x (R / (R + h)) for the origin height; rotation = convergence
    assert fit.scale_factor == pytest.approx(f.meridional_scale * 6.371e6 / (6.371e6 + ORIGIN.h),
                                             rel=2e-6)
    assert fit.rotation_deg == pytest.approx(f.meridian_convergence, abs=1e-4)
    assert abs(fit.anisotropy) < 1e-6
    # the lean of a point 100 m above u_ref at 707 m from the origin: d / R * dh
    assert fit.lean_m == pytest.approx(707 / 6.371e6 * 100, rel=0.05)
    e, n = fit.to_enu(*fit.to_crs(123.0, -77.0))
    assert (e, n) == pytest.approx((123.0, -77.0), abs=1e-9)


def test_too_large_an_area_is_refused():
    proj = SiteProjection(ORIGIN)
    with pytest.raises(ExportRefused, match="too large"):
        proj.fit_plane((-60_000, 60_000), (-60_000, 60_000), (0, 10), u_ref=0.0)


def test_budget_states_what_is_known_and_what_is_not():
    report = {"altitude_reference": "unverified", "sphere_to_wgs84": {"frames": {"max_m": 0.96}}}
    b = georeference_budget(earth_model="sphere-enu+affine",
                            metric={"frame_error_m": 0.00022, "origin": {"lat": 1}}, report=report)
    assert b["frame_error_m"] == 0.00022 and "affine" in b["frame_error_meaning"]
    assert b["altitude_reference"] == "unverified"
    assert b["sphere_to_wgs84_corrected"]["frames"]["max_m"] == 0.96
    assert b["not_included"] and b["warnings"] == []
    legacy = georeference_budget(earth_model="sphere-legacy", metric={"frame_error_m": 1.53},
                                 report={"legacy_shift": {"max_m": 1.53}})
    assert legacy["legacy_shift"]["max_m"] == 1.53 and "1.53" in legacy["warnings"][0]
    unknown = georeference_budget(earth_model="wgs84", metric={"frame_error_m": None})
    assert unknown["warnings"] == ["frame error of this run is unknown"]


def test_site_frame_name_matches_the_georef_stage():
    pytest.importorskip("torch")
    from aerosplat.export.georef import SITE_FRAME
    from aerosplat.geo.georef import SITE_FRAME as STAGE_FRAME
    assert SITE_FRAME == STAGE_FRAME


# ── orthophoto: render into the grid, verify, mosaic ─────────────────────
MARK = np.array([37.3, -21.6, 4.0])      # a red disk on the ground, ENU
MARK_R = 1.5


def enu_colour(e, n):
    """Smooth ground colour (so seams and shifts show) plus the marker disk."""
    r = np.clip(128 + 0.9 * e, 0, 255)
    g = np.clip(128 + 0.9 * n, 0, 255)
    b = np.full_like(r, 60.0)
    disk = (e - MARK[0]) ** 2 + (n - MARK[1]) ** 2 < MARK_R ** 2
    r, g, b = np.where(disk, 255.0, r), np.where(disk, 0.0, g), np.where(disk, 0.0, b)
    return np.stack([r, g, b], axis=-1)


class FakeRenderer:
    """Draws the ENU colour field through the real ortho camera maths (no GPU).

    Opaque inside a 120 m square around the origin, transparent outside it.
    `shift_px` misreports the ground grid, like a renderer whose convention drifted.
    """

    def __init__(self, shift_px=0.0, run="r1"):
        self.calls = []
        self.shift_px = shift_px
        self.run = run

    def __call__(self, req):
        self.calls.append(req)
        V = cam.view_matrix(np.array([req.position.e, req.position.n, 50.0]), req.yaw_deg,
                            req.pitch, req.roll_deg)
        K = cam.ortho_intrinsics(req.width, req.height, req.extent_m)
        g = cam.ortho_ground_grid(V, K, req.width, req.height)
        cols, rows = np.meshgrid(np.arange(req.width), np.arange(req.height))
        en = (np.asarray(g["pixel00_enu"]) + cols[..., None] * np.asarray(g["du_enu"])
              + rows[..., None] * np.asarray(g["dv_enu"]))
        inside = (np.abs(en[..., 0]) < 60) & (np.abs(en[..., 1]) < 60)
        alpha = np.where(inside, 255, 0).astype(np.uint8)
        if req.output == "alpha":
            image = alpha
        else:
            image = np.round(enu_colour(en[..., 0], en[..., 1]) * (alpha[..., None] / 255.0)
                             ).astype(np.uint8)
        if self.shift_px:
            g = dict(g, pixel00_enu=[g["pixel00_enu"][0] + self.shift_px * g["m_per_px"],
                                     g["pixel00_enu"][1]])
        return image, {"ground_grid": g, "width": req.width, "height": req.height, "run": self.run}


def _ortho(tmp_path, renderer, tile_px=64, gsd=0.5, area=(-70, -70, 70, 70)):
    pytest.importorskip("rasterio")
    from aerosplat.export.ortho import export_ortho
    from aerosplat.export.raster import NorthUpGrid

    proj = SiteProjection(ORIGIN)
    fit = proj.fit_plane((area[0], area[2]), (area[1], area[3]), (0.0, 30.0), u_ref=4.0)
    grid = NorthUpGrid.covering(fit, (area[0], area[2]), (area[1], area[3]), gsd)
    path = tmp_path / "ortho.tif"
    info = export_ortho(path, renderer=renderer, proj=proj, fit=fit, grid=grid, tile_px=tile_px,
                        tags={"AEROSPLAT_EXPORT": {"kind": "ortho"}})
    return path, info, grid


def test_ortho_marker_lands_at_its_lonlat(tmp_path):
    import rasterio

    path, info, grid = _ortho(tmp_path, FakeRenderer())
    assert info["tiles"] > 4 and info["registration_px_max"] < 0.01
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 32618 and src.count == 4
        assert src.transform.b == 0 and src.transform.d == 0           # north-up
        assert src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
        rgba = src.read()
        T = src.transform
    red = (rgba[0] > 240) & (rgba[1] < 20) & (rgba[3] == 255)
    rows, cols = np.nonzero(red)
    x, y = T @ (cols.mean() + 0.5, rows.mean() + 0.5)
    lon, lat = TO_LONLAT.transform(x, y)
    want_lon, want_lat, _ = lonlat_of_enu(MARK)
    assert metres_between(lon, lat, float(want_lon), float(want_lat)) < 0.1 * grid.px + 0.02


def test_ortho_mosaic_has_no_seams(tmp_path):
    """Every opaque pixel holds the colour of its own ENU position: tiles line up exactly."""
    import rasterio

    path, _, grid = _ortho(tmp_path, FakeRenderer(), tile_px=48)
    proj = SiteProjection(ORIGIN)
    with rasterio.open(path) as src:
        rgba = src.read()
    rows, cols = np.nonzero(rgba[3] == 255)
    enu = proj.crs_to_enu(grid.centre_x(cols), grid.centre_y(rows), ORIGIN.h + 4.0)
    want = enu_colour(enu[:, 0], enu[:, 1])
    got = rgba[:3, rows, cols].T.astype(float)
    smooth = ~((enu[:, 0] - MARK[0]) ** 2 + (enu[:, 1] - MARK[1]) ** 2 < (MARK_R + 1) ** 2)
    assert np.abs(got[smooth] - want[smooth]).max() <= 1.0
    # transparent where nothing was reconstructed, not black ground
    assert (rgba[3] == 0).any() and rgba[3][rgba[3] < 255].max() == 0


def test_ortho_refuses_a_misregistered_renderer(tmp_path):
    with pytest.raises(ExportRefused, match="differs from the plan"):
        _ortho(tmp_path, FakeRenderer(shift_px=0.5))
    assert not list(tmp_path.glob("*.tif")), "nothing written, not even a partial file"


def test_ortho_tiles_are_rendered_in_the_utm_grid():
    """The camera is yawed by the grid rotation and scaled by the UTM scale factor."""
    from aerosplat.export.ortho import grid_yaw_and_scale

    fit = SiteProjection(ORIGIN).fit_plane((-100, 100), (-100, 100), (0, 10), u_ref=0.0)
    yaw, enu_per_m = grid_yaw_and_scale(fit)
    assert yaw == pytest.approx(fit.rotation_deg, abs=1e-6)
    assert enu_per_m == pytest.approx(1.0 / fit.scale_factor, rel=1e-6)


# ── DSM / DTM / nDSM ─────────────────────────────────────────────────────
def _terrain(res=1.0):
    from aerosplat.semantics.models import TerrainModel

    model = TerrainModel(res_m=res, x0=-40.0, y0=-30.0, rows=60, cols=80, n_points=1,
                         filled_cells=1)
    e, n = model.cell_centres()
    dtm = 10.0 + 0.05 * e - 0.02 * n                  # gently sloping ground
    dsm = dtm.copy()
    building = (np.abs(e - 5) < 8) & (np.abs(n - 4) < 6)
    dsm[building] += 12.0
    count = np.ones(dsm.shape, np.int64)
    dsm[40:44, 60:66] = np.nan                        # a hole in the data
    count[40:44, 60:66] = 0
    count[:, :3] = 0                                  # a strip never seen
    dsm[:, :3] = np.nan
    return model, dsm, dtm, count


def test_dem_heights_land_where_they_belong(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from aerosplat.export.dem import NODATA, export_dem, terrain_fit
    from aerosplat.export.raster import read_tags

    model, dsm, dtm, count = _terrain()
    proj = SiteProjection(ORIGIN)
    paths = {k: tmp_path / f"{k}.tif" for k in ("dsm", "dtm", "ndsm")}
    summary = export_dem(paths, model=model, dsm=dsm, dtm=dtm, count=count, proj=proj,
                         fit=terrain_fit(proj, model, dsm, dtm), max_gap_m=2.0,
                         tags={"AEROSPLAT_EXPORT": {"kind": "dem"}})
    assert 0.5 < summary["dsm"]["valid_fraction"] < 1.0
    with rasterio.open(paths["dsm"]) as src:
        assert src.crs.to_epsg() == 32618 and src.nodata == NODATA
        a, T = src.read(1), src.transform
    # a ground cell well away from the building and holes: its height is the exact
    # ellipsoidal h of the DTM surface at that cell's lon/lat
    row, col = 20, 10
    x, y = T @ (col + 0.5, row + 0.5)
    lon, lat = TO_LONLAT.transform(x, y)
    enu = EnuFrame(ORIGIN).from_geodetic(lat, lon, ORIGIN.h + 10.0)
    u = 10.0 + 0.05 * enu[0] - 0.02 * enu[1]
    _, _, h = lonlat_of_enu([enu[0], enu[1], u])
    assert a[row, col] == pytest.approx(float(h), abs=0.005)
    # on the roof: 12 m higher
    roof_col, roof_row = ~T @ tuple(proj.enu_to_crs([5.0, 4.0, 0.0])[:2])
    roof = a[int(roof_row), int(roof_col)]
    _, _, h_roof = lonlat_of_enu([5.0, 4.0, 10.0 + 0.05 * 5 - 0.02 * 4 + 12.0])
    assert roof == pytest.approx(float(h_roof), abs=0.01)
    # the hole is nodata, not interpolated
    hole_col, hole_row = ~T @ tuple(proj.enu_to_crs([model.x0 + 63.0, model.y0 + 42.0, 0.0])[:2])
    hole = int(hole_row), int(hole_col)
    assert a[hole] == NODATA
    with rasterio.open(paths["dtm"]) as src:
        d = src.read(1)
    assert d[hole] != NODATA            # 2 m from data: kept (bare-earth estimate)
    assert (d == NODATA).any()                      # the unseen strip is beyond max_gap_m
    with rasterio.open(paths["ndsm"]) as src:
        nd = src.read(1)
    assert nd[int(roof_row), int(roof_col)] == pytest.approx(12.0, abs=0.01)
    assert nd[hole] == NODATA
    assert "ellipsoidal" in read_tags(paths["dsm"])["AEROSPLAT_HEIGHTS"]


def test_resampling_never_bridges_a_hole():
    from aerosplat.export.dem import sample_no_bridging
    from aerosplat.semantics.models import TerrainModel

    model = TerrainModel(res_m=1.0, x0=0.0, y0=0.0, rows=3, cols=4, n_points=1, filled_cells=1)
    r = np.arange(12.0).reshape(3, 4)
    sample = lambda e, n: sample_no_bridging(r, model, np.array([e]), np.array([n]))[0]  # noqa: E731
    assert sample(1.5, 1.5) == 5.0                     # a cell centre
    assert sample(1.0, 1.5) == 4.5                     # halfway between two centres
    assert np.isnan(sample(4.6, 1.5))                  # outside the raster
    r[1, 2] = np.nan                                   # a hole
    assert np.isnan(sample(2.5, 1.5))                  # in the hole: nothing to report
    assert sample(1.9, 1.5) == 5.0                     # next to it: the cell's own value
    assert sample(0.5, 1.5) == 4.0                     # away from it: plain bilinear
    assert sample(1.0, 1.5) == 4.5


# ── LAS ──────────────────────────────────────────────────────────────────
def test_las_points_land_at_their_lonlat_and_height(tmp_path):
    laspy = pytest.importorskip("laspy")
    from aerosplat.export.pointcloud import export_las, read_metadata

    enu = np.array([[0.0, 0.0, 0.0], [250.0, -130.0, 12.5], [-480.0, 410.0, 61.0],
                    [np.nan, 0, 0]])
    rgb = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], float)
    path = tmp_path / "cloud.laz"
    info = export_las(path, enu=enu, rgb=rgb, proj=SiteProjection(ORIGIN),
                      metadata={"source": "test", "georeference": {"frame_error_m": 0.0002}})
    assert info["points"] == 3 and info["dropped_non_finite"] == 1
    las = laspy.read(str(path))
    assert las.header.version.minor == 4 and las.header.global_encoding.wkt
    assert CRS(las.header.parse_crs()).to_epsg() == 32618
    lon, lat = TO_LONLAT.transform(np.asarray(las.x), np.asarray(las.y))
    want_lon, want_lat, want_h = lonlat_of_enu(enu[:3])
    for i in range(3):
        assert metres_between(lon[i], lat[i], want_lon[i], want_lat[i]) < 0.002
    np.testing.assert_allclose(np.asarray(las.z), want_h, atol=0.001)
    assert int(las.red[0]) == 65535 and int(las.green[1]) == 65535
    meta = read_metadata(path)
    assert meta["georeference"]["frame_error_m"] == 0.0002 and "ellipsoidal" in meta["heights"]


# ── 3DGS .ply ────────────────────────────────────────────────────────────
def _splat(n=50, degree=2, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    return {"xyz": rng.normal(size=(n, 3)) * 100, "scale": np.exp(rng.normal(size=(n, 3)) - 2),
            "rotation": q / np.linalg.norm(q, axis=1, keepdims=True),
            "opacity": rng.uniform(0.01, 0.99, size=(n, 1)),
            "sh": rng.normal(size=(n, (degree + 1) ** 2, 3)) * 0.3,
            "activations": {"scale_is_log": False, "opacity_is_logit": False}}


def test_ply_round_trip_and_inria_layout(tmp_path):
    scene = _splat()
    arrays, stats = inria_arrays(scene)
    assert stats["sh_degree"] == 2 and stats["dropped_invalid"] == 0
    path = tmp_path / "splat.ply"
    write_inria_ply(path, arrays, ["frame WGS84 Site ENU", "origin_wgs84 lat 40.9"])
    back, comments = read_inria_ply(path)
    assert "origin_wgs84 lat 40.9" in comments
    act = inria_to_activated(back)
    np.testing.assert_allclose(act["xyz"], scene["xyz"], rtol=1e-6, atol=1e-4)
    np.testing.assert_allclose(act["scale"], scene["scale"], rtol=1e-5)
    np.testing.assert_allclose(act["opacity"], scene["opacity"][:, 0], rtol=1e-5)
    np.testing.assert_allclose(act["rotation"], scene["rotation"], atol=1e-6)
    np.testing.assert_allclose(act["sh"], scene["sh"], atol=1e-6)
    # header: the INRIA order, f_rest channel-major (index c * K + k)
    header = path.read_bytes().split(b"end_header")[0].decode()
    props = [ln.split()[-1] for ln in header.splitlines() if ln.startswith("property")]
    assert props == property_names(24)
    assert props[:9] == ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    K = 8
    assert back["f_rest"][0, 1 * K + 3] == pytest.approx(scene["sh"][0, 1 + 3, 1], abs=1e-6)
    # raw values: logit opacity and log scale, as every reader expects
    assert back["opacity"][0] == pytest.approx(math.log(scene["opacity"][0, 0]
                                                        / (1 - scene["opacity"][0, 0])), abs=1e-5)
    assert back["scale"][0, 0] == pytest.approx(math.log(scene["scale"][0, 0]), abs=1e-5)


def test_ply_rgb_scene_drops_diverged_and_refuses_without_contract():
    scene = _splat(n=5)
    del scene["sh"]
    scene["rgb"] = np.full((5, 3), 0.75)
    scene["xyz"][1] = np.nan
    scene["rotation"][2] = 0.0
    arrays, stats = inria_arrays(scene)
    assert stats["gaussians"] == 3 and stats["dropped_invalid"] == 2 and stats["sh_degree"] == 0
    assert arrays["f_rest"].shape == (3, 0)
    np.testing.assert_allclose(0.5 + 0.28209479177387814 * arrays["f_dc"], 0.75)
    del scene["activations"]
    with pytest.raises(ExportRefused, match="activation contract"):
        inria_arrays(scene)

