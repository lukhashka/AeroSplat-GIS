"""Golden/invariant tests protecting the empirically-found Earth Studio axis convention."""
import json

import numpy as np

from conftest import FIXTURES
from earthstudio_to_colmap import convert, mat_to_quat_wxyz, rot_x, rot_y, rot_z


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _data_lines(text):
    return [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]


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


def _convert(tmp_path):
    out = tmp_path / "sparse_known"
    convert(str(FIXTURES / "earthstudio_3frames.json"), "fixture", str(out))
    return out


def test_images_txt_matches_golden(tmp_path):
    out = _convert(tmp_path)
    got = _data_lines((out / "images.txt").read_text())
    gold = _data_lines((FIXTURES / "earthstudio_3frames_images.golden.txt").read_text())
    assert len(got) == len(gold) == 3
    for a, b in zip(got, gold):
        pa, pb = a.split(), b.split()
        np.testing.assert_allclose([float(v) for v in pa[1:8]],
                                   [float(v) for v in pb[1:8]], rtol=0, atol=1e-5)
        assert pa[8:] == pb[8:]


def test_camera_centers_match_geo_registration(tmp_path):
    out = _convert(tmp_path)
    geo = json.loads((out / "geo_registration.json").read_text())
    rows = [ln.split() for ln in _data_lines((out / "images.txt").read_text())]
    for row, rec in zip(rows, geo["images"]):
        R = _quat_to_mat([float(v) for v in row[1:5]])
        t = np.array([float(v) for v in row[5:8]])
        np.testing.assert_allclose(-R.T @ t, rec["enu_xyz"], atol=1e-3)


def test_intrinsics_from_fov(tmp_path):
    out = _convert(tmp_path)
    p = _data_lines((out / "cameras.txt").read_text())[0].split()
    assert p[1] == "SIMPLE_PINHOLE"
    h, f = int(p[3]), float(p[4])
    fov = json.loads((FIXTURES / "earthstudio_3frames.json").read_text())["cameraFrames"][0]["fovVertical"]
    assert abs(f - (h / 2) / np.tan(np.radians(fov) / 2)) < 1e-3
