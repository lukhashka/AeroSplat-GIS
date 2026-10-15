"""Affine maps on points and 3D Gaussians, and the georef stage that carries an Earth
Studio run's products from sphere ENU to WGS84 Site ENU (ADR-0006, amended 2026-10-15)."""
import json

import numpy as np
import pytest
import torch

from conftest import ES_JSON, write_pt
from aerosplat.geo.affine import AffineMap, quat_to_rotmat, rotmat_to_quat
from aerosplat.geo.crs import GeoPoint
from aerosplat.geo.earthstudio_to_colmap import convert
from aerosplat.geo.georef import FRAME_KEY, SITE_FRAME, load_affine, main

SITE_ORIGIN = GeoPoint(lat=40.8942, lon=-74.2173, h=233.0)


def _gaussians(n, rng, aniso=False):
    xyz = rng.normal(size=(n, 3)) * 100
    if aniso:   # well-separated axis lengths, so every axis is identifiable
        scale = np.stack([np.full(n, 0.1), np.full(n, 0.3), np.full(n, 1.0)], 1)
        scale = scale * rng.uniform(0.5, 2.0, (n, 1))
    else:
        scale = np.exp(rng.normal(size=(n, 3)))
    q = rng.normal(size=(n, 4))
    return xyz, scale, q / np.linalg.norm(q, axis=1, keepdims=True)


def _cov(scale, q):
    M = quat_to_rotmat(q) * scale[:, None, :]
    return M @ np.swapaxes(M, 1, 2)


def _quat_close(a, b):
    """Max per-component deviation of unit quaternions, sign-agnostic (q ~ -q)."""
    return np.minimum(np.abs(a - b), np.abs(a + b)).max()


def test_quaternion_round_trip_all_branches():
    rng = np.random.default_rng(3)
    q = rng.normal(size=(500, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q = np.vstack([q, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]])
    back = rotmat_to_quat(quat_to_rotmat(q))
    assert (back[:, 0] >= 0).all()
    assert _quat_close(back, q) < 1e-12


@pytest.mark.parametrize("A", [
    np.diag([1.002558, 0.998712, 1.0]),
    np.array([[2.0, 0.3, 0.0], [0.0, 1.0, 0.1], [-0.2, 0.0, 0.5]]),
])
def test_gaussian_covariance_maps_exactly(A):
    rng = np.random.default_rng(0)
    xyz, scale, q = _gaussians(2000, rng)
    aff = AffineMap(A, np.array([1.0, -2.0, 3.0]))
    xyz2, scale2, q2 = aff.apply_gaussians(xyz, scale, q)
    np.testing.assert_allclose(xyz2, xyz @ A.T + aff.b, rtol=0, atol=1e-9)
    want = A @ _cov(scale, q) @ A.T
    assert np.abs(_cov(scale2, q2) - want).max() < 1e-12 * np.abs(want).max()
    np.testing.assert_allclose(np.linalg.norm(q2, axis=1), 1.0, atol=1e-12)
    assert (scale2 > 0).all()


def test_identity_leaves_gaussians_unchanged():
    xyz, scale, q = _gaussians(1000, np.random.default_rng(1))
    xyz2, scale2, q2 = AffineMap.identity().apply_gaussians(xyz, scale, q)
    np.testing.assert_allclose(xyz2, xyz, atol=0)
    np.testing.assert_allclose(scale2, scale, rtol=1e-12)
    assert _quat_close(q2, q) < 1e-9


def test_near_identity_keeps_axis_order_and_orientation():
    """The sphere->WGS84 map is ~0.26 % off identity: anisotropic Gaussians keep their axes."""
    xyz, scale, q = _gaussians(1000, np.random.default_rng(2), aniso=True)
    _, scale2, q2 = AffineMap(np.diag([1.002558, 0.998712, 1.0]), np.zeros(3)
                              ).apply_gaussians(xyz, scale, q)
    assert np.abs(scale2 / scale - 1).max() < 3e-3
    assert _quat_close(q2, q) < 1e-2


def test_affine_must_preserve_orientation():
    with pytest.raises(ValueError):
        AffineMap.from_dict({"A": np.diag([1.0, 1.0, -1.0]).tolist(), "b": [0, 0, 0]})


# ── the stage ─────────────────────────────────────────────────────────────
@pytest.fixture
def run(tmp_path):
    """A geo-stage registration report plus sphere-ENU cloud and splat products."""
    known = tmp_path / "known_poses"
    report = tmp_path / "reports" / "registration.json"
    convert(str(ES_JSON), "fixture", str(known), origin=SITE_ORIGIN, report_path=str(report))
    gs = tmp_path / "gs"
    write_pt(gs / "dense.pt", "dense", 300)
    write_pt(gs / "splat.pt", "splat", 300)
    return report, gs


def _georef(report, gs, *extra):
    return main(["--registration", str(report),
                 "--cloud", str(gs / "dense.pt"), str(gs / "dense_wgs84.pt"),
                 "--splat", str(gs / "splat.pt"), str(gs / "splat_wgs84.pt"), *extra])


def test_stage_maps_cloud_and_splat(run):
    report, gs = run
    assert _georef(report, gs) == 0
    aff, block = load_affine(report)
    src_c, out_c = torch.load(gs / "dense.pt"), torch.load(gs / "dense_wgs84.pt")
    np.testing.assert_allclose(out_c["xyz"].double().numpy(),
                               aff.apply(src_c["xyz"].double().numpy()), atol=1e-4)
    assert out_c["xyz"].dtype == src_c["xyz"].dtype
    assert torch.equal(out_c["rgb"], src_c["rgb"])
    assert out_c[FRAME_KEY]["crs"] == SITE_FRAME and out_c[FRAME_KEY]["A"] == block["A"]

    src, out = torch.load(gs / "splat.pt"), torch.load(gs / "splat_wgs84.pt")
    np.testing.assert_allclose(out["xyz"].double().numpy(),
                               aff.apply(src["xyz"].double().numpy()), atol=1e-4)
    want = aff.A @ _cov(src["scale"].double().numpy(), src["rotation"].double().numpy()) @ aff.A.T
    got = _cov(out["scale"].double().numpy(), out["rotation"].double().numpy())
    assert np.abs(got - want).max() < 1e-6 * np.abs(want).max()
    for key in ("opacity", "rgb"):
        assert torch.equal(out[key], src[key])
    assert out["activations"] == src["activations"] and FRAME_KEY not in src


def test_stage_handles_log_scales_and_diverged_gaussians(run):
    report, gs = run
    data = torch.load(gs / "splat.pt")
    scale = data["scale"].clone()
    data["scale"] = torch.log(scale)
    data["activations"] = {"scale_is_log": True, "opacity_is_logit": False}
    data["xyz"][0, 0] = float("nan")
    torch.save(data, gs / "splat.pt")
    assert _georef(report, gs) == 0
    out = torch.load(gs / "splat_wgs84.pt")
    assert torch.isnan(out["xyz"][0, 0]) and torch.equal(out["rotation"][0], data["rotation"][0])
    ratio = torch.exp(out["scale"][1:]).double() / scale[1:].double()
    assert (ratio - 1).abs().max() < 3e-3          # still log-space, ~0.26 % map


def test_stage_maps_recentered_cloud(run):
    report, gs = run
    data = torch.load(gs / "dense.pt")
    centroid = data["xyz"].double().mean(0)
    data["xyz"] = (data["xyz"].double() - centroid).float()
    data["meta"] = {"recentered": True, "centroid": centroid.tolist()}
    torch.save(data, gs / "dense.pt")
    _georef(report, gs)
    aff, _ = load_affine(report)
    out = torch.load(gs / "dense_wgs84.pt")
    absolute = out["xyz"].double().numpy() + np.array(out["meta"]["centroid"])
    want = aff.apply(data["xyz"].double().numpy() + centroid.numpy())
    np.testing.assert_allclose(absolute, want, atol=1e-4)


def test_stage_refuses_mapped_input_and_old_reports(run, tmp_path):
    report, gs = run
    _georef(report, gs)
    with pytest.raises(SystemExit, match="already in"):
        main(["--registration", str(report),
              "--cloud", str(gs / "dense_wgs84.pt"), str(tmp_path / "twice.pt")])
    old = json.loads(report.read_text())
    old.pop("sphere_enu_to_site_enu")
    old["earth_model"] = {"input": "sphere", "output": "WGS84"}
    report.write_text(json.dumps(old))
    with pytest.raises(SystemExit, match="re-run from the 'geo' stage"):
        _georef(report, gs)


def test_stage_needs_at_least_one_product(run, tmp_path):
    report, _ = run
    with pytest.raises(SystemExit, match="nothing to map"):
        main(["--registration", str(report),
              "--splat", str(tmp_path / "missing.pt"), str(tmp_path / "out.pt")])
