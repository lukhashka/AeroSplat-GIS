"""Golden/invariant tests protecting the empirically-found Earth Studio axis convention
and the Site-ENU (WGS84) conversion that replaced the per-capture spherical frame."""
import json

import numpy as np
import pytest

from conftest import FIXTURES
from aerosplat.geo.crs import EnuFrame, GeoPoint
from aerosplat.geo.earthstudio_to_colmap import (convert, frame_coordinates, legacy_mean_origin,
                                                 load_frames, mat_to_quat_wxyz, rot_x, rot_y,
                                                 rot_z)

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


def test_camera_centres_are_wgs84_site_enu(tmp_path):
    """Centres = pyproj ENU of the WGS84 coordinates around the Site origin."""
    out, _ = _convert(tmp_path)
    _, frames = load_frames(JSON)
    expected = EnuFrame(SITE_ORIGIN).from_geodetic(*frame_coordinates(frames))
    geo = json.loads((out / "geo_registration.json").read_text())
    for (_, c), exp, rec in zip(_poses(out / "images.txt"), expected, geo["images"]):
        np.testing.assert_allclose(c, exp, atol=1e-3)
        np.testing.assert_allclose(rec["enu_xyz"], exp, atol=1e-6)
    assert geo["enu_origin"] == {"latitude": 40.8942, "longitude": -74.2173, "altitude": 233.0}
    assert geo["origin_source"] == "site" and geo["earth_model"].startswith("WGS84")


def test_wgs84_keeps_sphere_rotations_and_moves_centres_within_report(tmp_path):
    """Against the legacy sphere golden (same origin = mean of frames): orientations are
    identical and centres move by exactly the reported legacy shift."""
    _, frames = load_frames(JSON)
    out, report = _convert(tmp_path, origin=legacy_mean_origin(frames))
    new = _poses(out / "images.txt")
    old = _poses(FIXTURES / "earthstudio_3frames_images.sphere.golden.txt")
    shifts = []
    for (Rn, cn), (Ro, co) in zip(new, old):
        np.testing.assert_allclose(Rn, Ro, atol=1e-8)
        shifts.append(np.linalg.norm(cn - co))
    assert max(shifts) == pytest.approx(report["legacy_shift"]["max_m"], abs=1e-4)
    assert 0.05 < max(shifts) < 2.0



def test_transition_report(tmp_path):
    out, report = _convert(tmp_path)
    assert json.loads((out / "registration_report.json").read_text()) == report
    check = report["export_check"]
    assert check["angular_residual_max_m"] < 1e-3          # convention holds
    assert check["radial_offset_m"]["spread"] < 1e-3          # constant offset only
    tr = report["sphere_to_wgs84"]
    # Sphere vs ellipsoid at ~41°N: ~0.1-0.3 % horizontal scale, i.e. ~1-2 m at 500 m.
    assert 0.5 < tr["site_1km2"]["max_horizontal_m"] < 3.0
    assert tr["site_1km2"]["max_vertical_m"] < 0.01
    assert tr["frames"]["n"] == 3 and tr["frames"]["max_m"] < tr["site_1km2"]["max_m"]


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
