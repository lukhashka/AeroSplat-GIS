"""Golden/invariant tests protecting the empirically-found Earth Studio axis convention,
the sphere-ENU frame of the Site that the known poses are written in, and the affine map
sphere ENU -> WGS84 Site ENU recorded for the products (ADR-0006, amended 2026-10-15)."""
import json

import numpy as np
import pytest

from conftest import FIXTURES
from aerosplat.geo.affine import AffineMap
from aerosplat.geo.crs import EnuFrame, GeoPoint, ecef_to_enu_rotation
from aerosplat.geo.earthstudio_to_colmap import (EARTH_MODEL, EARTH_R, convert,
                                                 frame_coordinates, legacy_mean_origin,
                                                 load_frames, mat_to_quat_wxyz, rot_x, rot_y,
                                                 rot_z)
from aerosplat.geo.sphere_enu import fit_box, fit_site_affine, sphere_enu_to_site_enu

JSON = FIXTURES / "earthstudio_3frames.json"
# Site origin used for the golden file (fixed, NOT derived from the frames).
SITE_ORIGIN = GeoPoint(lat=40.8942, lon=-74.2173, h=233.0)


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _data_lines(text):
    return [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]


def _poses(images_txt):
    out = []
    for row in (ln.split() for ln in _data_lines(images_txt.read_text())):
        R = _quat_to_mat([float(v) for v in row[1:5]])
        t = np.array([float(v) for v in row[5:8]])
        out.append((R, -R.T @ t))
    return out


def test_rotations_orthonormal():
    R = rot_x(0.3) @ rot_y(-1.1) @ rot_z(2.0)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(R) - 1) < 1e-12


def test_quat_roundtrip_all_branches():
    rng = np.random.default_rng(1)
    for _ in range(200):
        a = rng.uniform(-np.pi, np.pi, 3)
        R = rot_x(a[0]) @ rot_y(a[1]) @ rot_z(a[2])
        np.testing.assert_allclose(_quat_to_mat(mat_to_quat_wxyz(R)), R, atol=1e-9)
    for R in (np.diag([1.0, -1, -1]), np.diag([-1.0, 1, -1]), np.diag([-1.0, -1, 1])):
        np.testing.assert_allclose(_quat_to_mat(mat_to_quat_wxyz(R)), R, atol=1e-9)


def _convert(tmp_path, origin=SITE_ORIGIN, json_path=JSON, name="sparse_known"):
    out = tmp_path / name
    report = convert(str(json_path), "fixture", str(out), origin=origin)
    return out, report


def test_images_txt_matches_golden(tmp_path):
    out, _ = _convert(tmp_path)
    got = _data_lines((out / "images.txt").read_text())
    gold = _data_lines((FIXTURES / "earthstudio_3frames_images.golden.txt").read_text())
    assert len(got) == len(gold) == 3
    for a, b in zip(got, gold):
        pa, pb = a.split(), b.split()
        np.testing.assert_allclose([float(v) for v in pa[1:8]],
                                   [float(v) for v in pb[1:8]], rtol=0, atol=1e-5)
        assert pa[8:] == pb[8:]


def test_camera_centres_are_sphere_enu_from_coordinate(tmp_path):
    """Centres sit on Earth Studio's sphere: back in geocentric xyz (same origin and R),
    each is at radius EARTH_R + altitude along the spherical angles of `coordinate`."""
    out, _ = _convert(tmp_path)
    _, frames = load_frames(JSON)
    lat, lon, alt = frame_coordinates(frames)
    la0, lo0 = np.radians(SITE_ORIGIN.lat), np.radians(SITE_ORIGIN.lon)
    o_ecef = (EARTH_R + SITE_ORIGIN.h) * np.array(
        [np.cos(la0) * np.cos(lo0), np.cos(la0) * np.sin(lo0), np.sin(la0)])
    R = ecef_to_enu_rotation(SITE_ORIGIN.lat, SITE_ORIGIN.lon)
    geo = json.loads((out / "geo_registration.json").read_text())
    for i, ((_, c), rec) in enumerate(zip(_poses(out / "images.txt"), geo["images"])):
        xyz = o_ecef + R.T @ c
        r = np.linalg.norm(xyz)
        assert r - EARTH_R == pytest.approx(alt[i], abs=1e-3)
        assert np.degrees(np.arcsin(xyz[2] / r)) == pytest.approx(lat[i], abs=1e-9)
        assert np.degrees(np.arctan2(xyz[1], xyz[0])) == pytest.approx(lon[i], abs=1e-9)
        np.testing.assert_allclose(rec["enu_xyz"], c, atol=1e-5)
    site = EnuFrame(SITE_ORIGIN).from_geodetic(lat, lon, alt)
    np.testing.assert_allclose([rec["site_enu_xyz"] for rec in geo["images"]], site, atol=1e-6)
    assert geo["enu_origin"] == {"latitude": 40.8942, "longitude": -74.2173, "altitude": 233.0}
    assert geo["origin_source"] == "site" and geo["earth_model"] == EARTH_MODEL


def test_sphere_enu_is_legacy_frame_without_position_offset(tmp_path):
    """Against the legacy golden (built from `position`, same origin = mean of frames):
    orientations are identical and centres differ only by `position`'s constant radial
    offset, i.e. ~1.2 m straight down near the origin."""
    _, frames = load_frames(JSON)
    out, report = _convert(tmp_path, origin=legacy_mean_origin(frames))
    offset = report["export_check"]["radial_offset_m"]["mean"]
    assert offset == pytest.approx(1.2, abs=0.01)
    new = _poses(out / "images.txt")
    old = _poses(FIXTURES / "earthstudio_3frames_images.sphere.golden.txt")
    for (Rn, cn), (Ro, co) in zip(new, old):
        np.testing.assert_allclose(Rn, Ro, atol=1e-8)
        np.testing.assert_allclose(cn - co, [0.0, 0.0, -offset], atol=1e-4)


def test_affine_maps_sphere_enu_to_wgs84_site_enu(tmp_path):
    """The recorded map takes the sphere-ENU centres to their exact WGS84 Site ENU positions
    and is sub-millimetre over the whole 1 km x 1 km x 500 m fit box."""
    out, report = _convert(tmp_path)
    geo = json.loads((out / "geo_registration.json").read_text())
    block = report["sphere_enu_to_site_enu"]
    assert geo["sphere_enu_to_site_enu"] == block
    aff = AffineMap.from_dict(block)
    # ~41 N: east stretched by the prime-vertical radius, north shrunk by the meridional one.
    np.testing.assert_allclose(np.diag(aff.A), [1.002558, 0.998712, 1.0], atol=2e-6)
    assert np.abs(aff.A - np.diag(np.diag(aff.A))).max() < 1e-9
    assert np.abs(aff.b).max() < 1e-3
    assert block["residual_max_m"] < 5e-4 and block["frames_residual_max_m"] < 5e-4
    sphere = np.array([rec["enu_xyz"] for rec in geo["images"]])
    site = np.array([rec["site_enu_xyz"] for rec in geo["images"]])
    np.testing.assert_allclose(aff.apply(sphere), site, atol=5e-4)
    pts = np.random.default_rng(0).uniform([-500, -500, -250], [500, 500, 250], (500, 3))
    exact = sphere_enu_to_site_enu(SITE_ORIGIN, pts)
    assert np.linalg.norm(aff.apply(pts) - exact, axis=1).max() < 5e-4
    # The map depends on the Site origin alone: every run of a Site shares it.
    np.testing.assert_allclose(fit_site_affine(SITE_ORIGIN)[0].A, aff.A, rtol=0, atol=1e-15)
    assert len(fit_box()) == block["fit"]["points"]


def test_transition_report(tmp_path):
    out, report = _convert(tmp_path)
    assert json.loads((out / "registration_report.json").read_text()) == report
    assert report["schema"] == 2 and report["earth_model"]["id"] == EARTH_MODEL
    check = report["export_check"]
    assert check["angular_residual_max_m"] < 1e-3          # convention holds
    assert check["radial_offset_m"]["spread"] < 1e-3          # constant offset only
    tr = report["sphere_to_wgs84"]
    # Sphere vs ellipsoid at ~41 N: ~0.1-0.3 % horizontal scale, i.e. ~1-2 m at 500 m.
    assert 0.5 < tr["site_1km2"]["max_horizontal_m"] < 3.0
    assert tr["site_1km2"]["max_vertical_m"] < 0.01
    assert tr["frames"]["n"] == 3 and tr["frames"]["max_m"] < tr["site_1km2"]["max_m"]
    # The pre-ADR-0006 frame (from `position`) also sat ~1.2 m high.
    assert report["legacy_shift"]["max_vertical_m"] == pytest.approx(1.2, abs=0.05)


def test_origin_is_fixed_not_derived_from_frames(tmp_path):
    """Two captures of the same site (overlapping frame subsets) put the shared frame at
    the same Site ENU position; the legacy mean origin does not."""
    data = json.loads(JSON.read_text())
    a, b = dict(data), dict(data)
    a["cameraFrames"], b["cameraFrames"] = data["cameraFrames"][:2], data["cameraFrames"][1:]
    ja, jb = tmp_path / "a.json", tmp_path / "b.json"
    ja.write_text(json.dumps(a))
    jb.write_text(json.dumps(b))

    def shared_centre(origin, tag):
        oa, _ = _convert(tmp_path, origin, ja, f"{tag}a")
        ob, _ = _convert(tmp_path, origin, jb, f"{tag}b")
        return _poses(oa / "images.txt")[1][1], _poses(ob / "images.txt")[0][1]

    ca, cb = shared_centre(SITE_ORIGIN, "site")
    np.testing.assert_allclose(ca, cb, atol=1e-6)
    la, lb = shared_centre(None, "legacy")
    assert np.linalg.norm(la - lb) > 10.0      # mean origin moved with the capture


def test_legacy_origin_without_site(tmp_path, capsys):
    out, report = _convert(tmp_path, origin=None)
    assert "no Site origin" in capsys.readouterr().out
    assert report["origin_source"].startswith("frames_mean")
    _, frames = load_frames(JSON)
    o = legacy_mean_origin(frames)
    geo = json.loads((out / "geo_registration.json").read_text())
    assert geo["enu_origin"]["latitude"] == pytest.approx(o.lat)


def test_intrinsics_from_fov(tmp_path):
    out, _ = _convert(tmp_path)
    p = _data_lines((out / "cameras.txt").read_text())[0].split()
    assert p[1] == "SIMPLE_PINHOLE"
    h, f = int(p[3]), float(p[4])
    fov = json.loads(JSON.read_text())["cameraFrames"][0]["fovVertical"]
    assert abs(f - (h / 2) / np.tan(np.radians(fov) / 2)) < 1e-3
