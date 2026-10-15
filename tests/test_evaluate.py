import json
import math

import numpy as np
import pytest
import torch

from aerosplat.geo.compare_poses import pose_errors
from aerosplat.gs import evaluate, report
from aerosplat.gs.trainer import (
    Camera, TileRasterizer, build_covariance_3d, find_test_twins, split_cameras,
)


def _cam(name, img=None, W=32, H=32):
    R = torch.eye(3)
    return Camera(R, torch.zeros(3), 40.0, 40.0, W / 2, H / 2, W, H, name, img)


def test_split_every_8th_deterministic():
    cams = [_cam(f"f{i:03d}.jpg") for i in range(24)][::-1]  # порядок вхідних не важливий
    train, test = split_cameras(cams, 8)
    assert [c.name for c in test] == ["f000.jpg", "f008.jpg", "f016.jpg"]
    assert len(train) == 21 and not {c.name for c in train} & {c.name for c in test}


def test_split_disabled_or_too_few():
    cams = [_cam(f"f{i}.jpg") for i in range(10)]
    assert split_cameras(cams, 0) == (cams, [])
    assert split_cameras(cams, 8) == (cams, [])


def _rz(deg):
    a = math.radians(deg)
    return torch.tensor([[math.cos(a), -math.sin(a), 0.0],
                         [math.sin(a), math.cos(a), 0.0],
                         [0.0, 0.0, 1.0]])


def _orbit_cam(name, bearing_deg, radius=100.0, center_jitter=(0.0, 0.0, 0.0)):
    """Camera on a horizontal orbit, rotated with its bearing (distinct pose per frame)."""
    R = _rz(bearing_deg)
    a = math.radians(bearing_deg)
    C = torch.tensor([radius * math.cos(a), radius * math.sin(a), 0.0]) + torch.tensor(center_jitter)
    return Camera(R, -R @ C, 40.0, 40.0, 16, 16, 32, 32, name, None)


def test_split_drops_closing_orbit_twin_of_test_frame():
    # 17 distinct frames ≈21° apart (≈37 m) + a closing frame that repeats frame 0
    # the way BA leaves it: ≈0.4 m and 0.09° off (dataset_v6: 00 ↔ 30).
    cams = [_orbit_cam(f"v_{i:02d}.jpg", i * 360.0 / 17) for i in range(17)]
    cams.append(_orbit_cam("v_17.jpg", 0.09, center_jitter=(0.2, 0.2, 0.1)))
    train, test = split_cameras(cams, 8)
    assert [c.name for c in test] == ["v_00.jpg", "v_08.jpg", "v_16.jpg"]
    assert "v_17.jpg" not in {c.name for c in train}
    assert len(train) == 14


def test_find_test_twins_needs_both_position_and_rotation():
    test = [_orbit_cam("t.jpg", 0.0)]
    ring = [_orbit_cam(f"r{i}.jpg", b) for i, b in enumerate((45.0, 90.0, 135.0, 180.0))]
    same_spot_turned = _orbit_cam("turned.jpg", 0.0)
    same_spot_turned.R = _rz(10.0)
    same_spot_turned.t = -same_spot_turned.R @ test[0].center()
    same_view_far = _orbit_cam("far.jpg", 0.0, center_jitter=(0.0, 30.0, 0.0))
    near_twin = _orbit_cam("twin.jpg", 0.05, center_jitter=(0.5, 0.0, 0.0))
    train = ring + [same_spot_turned, same_view_far, near_twin]
    assert find_test_twins(train, test) == {"twin.jpg": "t.jpg"}
    assert find_test_twins(ring, test) == {}


def _scene():
    n = 50
    g = torch.Generator().manual_seed(0)
    xyz = torch.randn(n, 3, generator=g) * 0.3
    xyz[:, 2] += 3.0
    scale = torch.full((n, 3), 0.15)
    rot = torch.zeros(n, 4); rot[:, 0] = 1
    return {"xyz": xyz, "scale": scale, "rotation": rot,
            "opacity": torch.full((n,), 0.8), "rgb": torch.rand(n, 3, generator=g),
            "cov3d": build_covariance_3d(torch.log(scale), rot)}


def test_evaluate_views_identical_image_is_perfect(tmp_path):
    dev = torch.device("cpu")
    params, rast = _scene(), TileRasterizer()
    cam = _cam("a.jpg")
    gt, _ = rast.render(params, cam, cam.view_matrix(dev), torch.zeros(3))
    cam.image = gt.detach()
    res = evaluate.evaluate_views(params, rast, [cam], dev, lpips_fn=None, keep_images=True)
    assert res["mean_psnr"] > 60 and res["mean_ssim"] > 0.999
    assert res["mean_lpips"] is None and res["n_test_views"] == 1

    report.write_report(tmp_path / "r.html", {**res, "system": {}}, res["_images"])
    assert "<img" in (tmp_path / "r.html").read_text(encoding="utf-8")
    json.dumps({k: v for k, v in res.items() if k != "_images"})


def test_evaluate_views_worse_when_images_differ():
    dev = torch.device("cpu")
    params, rast = _scene(), TileRasterizer()
    cam = _cam("a.jpg", torch.rand(32, 32, 3))
    res = evaluate.evaluate_views(params, rast, [cam], dev)
    assert res["mean_psnr"] < 20


def test_pick_views_dedupes():
    pv = [{"name": "a", "psnr": 1.0}, {"name": "b", "psnr": 2.0}]
    assert report.pick_views(pv, 5) == ["a", "b"]


def test_pose_errors_zero_and_rotation():
    q = np.array([1.0, 0, 0, 0]); t = np.array([1.0, 2, 3])
    known = {"a": {"q": q, "t": t}}
    z = pose_errors(known, known)
    assert z and z["pos_max_m"] == 0 and z["rot_max_deg"] < 1e-6
    q90 = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    r = pose_errors(known, {"a": {"q": q90, "t": t}})
    assert r and r["rot_max_deg"] == pytest.approx(90.0, abs=1e-4)
    assert pose_errors(known, {"b": {"q": q, "t": t}}) is None


def test_camera_sampler_covers_every_camera_each_epoch():
    from aerosplat.gs.trainer import CameraSampler
    s = CameraSampler(7)
    for _ in range(3):
        assert sorted(s.next() for _ in range(7)) == list(range(7))


def test_anisotropy_penalty_only_above_ratio():
    import math, torch
    from aerosplat.gs.trainer import anisotropy_penalty
    ok = torch.tensor([[0.0, math.log(5), 0.0]])
    needle = torch.tensor([[0.0, math.log(30), 0.0]])
    assert anisotropy_penalty(ok, 10.0).item() == 0.0
    assert anisotropy_penalty(needle, 10.0).item() > 0.0
