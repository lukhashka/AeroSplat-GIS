import struct

import numpy as np
import pytest

from aerosplat.io.cloud_io import filter_cloud, read_points3d_bin


def _write_points3d_bin(path, pts):
    """pts: list of (xyz, rgb, error, track_len). Layout per COLMAP points3D.bin."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        for i, (xyz, rgb, err, tl) in enumerate(pts):
            f.write(struct.pack("<Q3d3Bd", i + 1, *xyz, *rgb, err))
            f.write(struct.pack("<Q", tl))
            for k in range(tl):
                f.write(struct.pack("<II", k, k))


def test_points3d_roundtrip(tmp_path):
    pts = [((1.0, 2.0, 3.0), (10, 20, 30), 0.5, 2),
           ((-4.5, 0.25, 9.0), (255, 0, 1), 1.5, 0),
           ((7.0, 8.0, 9.0), (0, 0, 0), 0.1, 5)]
    p = tmp_path / "points3D.bin"
    _write_points3d_bin(p, pts)
    r = read_points3d_bin(p)
    np.testing.assert_allclose(r.xyz, [q[0] for q in pts])
    np.testing.assert_array_equal(r.rgb, [q[1] for q in pts])
    np.testing.assert_allclose(r.error, [q[2] for q in pts])
    np.testing.assert_array_equal(r.track_len, [q[3] for q in pts])


def test_points3d_truncated(tmp_path):
    p = tmp_path / "points3D.bin"
    _write_points3d_bin(p, [((0, 0, 0), (0, 0, 0), 0.0, 3)])
    p.write_bytes(p.read_bytes()[:-6])
    with pytest.raises(ValueError):
        read_points3d_bin(p)


def _cloud(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)), rng.integers(0, 255, (n, 3)).astype(np.uint8)


def test_filter_nonfinite_and_error():
    xyz, rgb = _cloud()
    xyz[0] = np.nan
    err = np.ones(len(xyz))
    err[1] = 9.0
    out, _, st = filter_cloud(xyz, rgb, error=err, max_error=2.0, verbose=False)
    assert st["dropped_nonfinite"] == 1 and st["dropped_high_error"] == 1
    assert len(out) == len(xyz) - 2


def test_filter_mad_removes_far_outlier():
    xyz, rgb = _cloud()
    xyz[5] = (1e6, 1e6, 1e6)
    out, _, st = filter_cloud(xyz, rgb, mad_k=10.0, verbose=False)
    assert st["dropped_outlier"] >= 1 and np.abs(out).max() < 100


def test_filter_subsample_deterministic():
    xyz, rgb = _cloud()
    a, _, _ = filter_cloud(xyz, rgb, max_points=50, seed=3, verbose=False)
    b, _, _ = filter_cloud(xyz, rgb, max_points=50, seed=3, verbose=False)
    assert len(a) == 50 and np.array_equal(a, b)


def test_filter_all_dropped_raises():
    xyz, rgb = _cloud()
    with pytest.raises(ValueError):
        filter_cloud(xyz, rgb, error=np.full(len(xyz), 5.0), max_error=1.0, verbose=False)


def test_save_cloud_contract(tmp_path):
    torch = pytest.importorskip("torch")
    from aerosplat.io.cloud_io import save_cloud
    xyz, rgb = _cloud(50)
    out = tmp_path / "c.pt"
    meta = save_cloud(out, xyz, rgb, source=tmp_path, stats={}, recenter=True)
    d = torch.load(out, map_location="cpu", weights_only=False)
    assert d["xyz"].dtype == torch.float32 and d["rgb"].max() <= 1.0
    assert float(d["xyz"].mean(0).abs().max()) < 1e-5  # recentered
    assert meta["recentered"] is True
