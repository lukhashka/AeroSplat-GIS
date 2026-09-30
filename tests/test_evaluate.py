import json

import numpy as np
import pytest
import torch

from aerosplat.geo.compare_poses import pose_errors
from aerosplat.gs import evaluate, report
from aerosplat.gs.trainer import Camera, TileRasterizer, build_covariance_3d, split_cameras


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
