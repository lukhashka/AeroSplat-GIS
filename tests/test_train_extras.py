"""Trainer extras of ROADMAP 2.2: masks, MVS depth regularisation, checkpoint/resume, TensorBoard.

CPU tests cover file conventions, losses and wiring; GPU tests (CUDA + a usable
gsplat build; on Windows under tools/gsplat_env.bat) train a tiny synthetic
scene: resume vs a straight run, depth pulling geometry, masks hiding a
corrupted region, and the CLI end to end on files laid out like a real run.
"""
import math
import os

import numpy as np
import pytest
import torch
from PIL import Image

from aerosplat.config import RunConfig
from aerosplat.gs import checkpoint as ckpt
from aerosplat.gs.supervision import (
    attach_masks, attach_mvs_depth, depth_loss, depth_map_path, depth_weight_at, find_mask,
    read_colmap_array, resize_depth, resize_mask, write_colmap_array,
)
from aerosplat.gs.trainer import (
    Camera, load_colmap_cameras, masked_photometric_loss, photometric_loss,
)


# ── masked photometric loss ─────────────────────────────────────────────
def _imgs(seed=0, h=24, w=32):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(3, h, w, generator=g), torch.rand(3, h, w, generator=g)


def test_masked_loss_with_full_mask_is_the_plain_loss():
    pred, gt = _imgs()
    full = torch.ones(24, 32, dtype=torch.bool)
    torch.testing.assert_close(masked_photometric_loss(pred, gt, full), photometric_loss(pred, gt))


def test_masked_pixels_give_no_gradient_and_do_not_change_the_loss():
    pred, gt = _imgs(1)
    valid = torch.ones(24, 32, dtype=torch.bool)
    valid[:, 20:] = False                                   # e.g. a moving object
    p = pred.clone().requires_grad_(True)
    loss = masked_photometric_loss(p, gt, valid)
    loss.backward()
    assert float(p.grad[:, :, 20:].abs().max()) == 0.0
    assert float(p.grad[:, :, :20].abs().sum()) > 0.0
    garbage = pred.clone()
    garbage[:, :, 20:] = float("nan")                       # whatever the render has there
    torch.testing.assert_close(masked_photometric_loss(garbage, gt, valid), loss.detach())


def test_masked_loss_is_normalised_by_valid_pixels():
    gt = torch.zeros(3, 24, 32)
    pred = torch.full((3, 24, 32), 0.5)
    valid = torch.zeros(24, 32, dtype=torch.bool)
    valid[:4, :4] = True
    l_small = masked_photometric_loss(pred, gt, valid, lambda_ssim=0.0)
    assert math.isclose(float(l_small), 0.5, rel_tol=1e-6)   # mean |err| over valid pixels
    nothing = torch.zeros(24, 32, dtype=torch.bool)
    assert float(masked_photometric_loss(pred, gt, nothing)) == 0.0   # fully masked: no NaN


# ── mask files ──────────────────────────────────────────────────────────
def _write_mask(path, arr):
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(path)


def _cam(name, W=32, H=24, f=30.0, R=None, t=None):
    return Camera(torch.eye(3) if R is None else R, torch.zeros(3) if t is None else t,
                  f, f, W / 2, H / 2, W, H, name, torch.zeros(H, W, 3))


def test_mask_naming_convention_colmap_first_then_stem(tmp_path):
    _write_mask(tmp_path / "a.jpeg.png", np.full((24, 32), 255))
    _write_mask(tmp_path / "a.png", np.zeros((24, 32)))
    _write_mask(tmp_path / "b.png", np.zeros((24, 32)))
    assert find_mask(str(tmp_path), "a.jpeg") == str(tmp_path / "a.jpeg.png")
    assert find_mask(str(tmp_path), "b.jpeg") == str(tmp_path / "b.png")
    assert find_mask(str(tmp_path), "c.jpeg") is None


def test_attach_masks_resamples_conservatively_and_checks_aspect(tmp_path):
    big = np.full((48, 64), 255, dtype=np.uint8)
    big[:, 33] = 0                                          # a 1-px invalid column at 2x
    _write_mask(tmp_path / "a.jpeg.png", big)
    cams = [_cam("a.jpeg"), _cam("b.jpeg")]
    assert attach_masks(cams, str(tmp_path)) == 1
    m = cams[0].mask
    assert m.shape == (24, 32) and m.dtype == torch.bool
    assert not bool(m[:, 16].any())                         # its footprint touched the column
    assert bool(m[:, 15].all()) and bool(m[:, 17].all())
    assert cams[1].mask is None                             # no file: the whole frame is used

    _write_mask(tmp_path / "b.jpeg.png", np.full((32, 32), 255))   # square mask, 4:3 frame
    with pytest.raises(ValueError, match="співвідношення"):
        attach_masks([_cam("b.jpeg")], str(tmp_path))


def test_attach_masks_fails_loudly_when_nothing_matches(tmp_path):
    with pytest.raises(FileNotFoundError):
        attach_masks([_cam("a.jpeg")], str(tmp_path / "missing"))
    _write_mask(tmp_path / "other.png", np.zeros((24, 32)))
    with pytest.raises(FileNotFoundError, match="a.jpeg.png"):
        attach_masks([_cam("a.jpeg")], str(tmp_path))


def test_resize_helpers():
    m = torch.ones(4, 4, dtype=torch.bool)
    m[0, 0] = False
    assert resize_mask(m, 2, 2).tolist() == [[False, True], [True, True]]
    d = torch.tensor([[1.0, 2.0], [3.0, 0.0]])
    up = resize_depth(d, 4, 4)
    assert set(up.unique().tolist()) == {0.0, 1.0, 2.0, 3.0}       # never blended
    assert float(up[3, 3]) == 0.0 and float(up[0, 0]) == 1.0


# ── COLMAP depth maps ───────────────────────────────────────────────────
def test_colmap_array_layout_and_roundtrip(tmp_path):
    h, w = 3, 5
    # COLMAP's Mat<float>::Write: "w&h&c&" then data[(slice*h + row)*w + col]
    flat = np.array([1000 * r + c for r in range(h) for c in range(w)], dtype=np.float32)
    path = tmp_path / "d.bin"
    with open(path, "wb") as f:
        f.write(b"5&3&1&")
        flat.tofile(f)
    arr = read_colmap_array(str(path))
    assert arr.shape == (3, 5) and arr[2, 4] == 2004 and arr[1, 0] == 1000
    write_colmap_array(str(tmp_path / "e.bin"), arr)
    assert (tmp_path / "e.bin").read_bytes() == path.read_bytes()
    (tmp_path / "bad.bin").write_bytes(b"5&3&1&" + b"\0" * 8)
    with pytest.raises(ValueError):
        read_colmap_array(str(tmp_path / "bad.bin"))


def _write_colmap_txt(sparse, cams, W, H):
    """Text model: one PINHOLE camera per image, identity rotations not required."""
    os.makedirs(sparse, exist_ok=True)
    with open(os.path.join(sparse, "cameras.txt"), "w") as f:
        for i, c in enumerate(cams, 1):
            f.write(f"{i} PINHOLE {W} {H} {c['f']} {c['f']} {W / 2} {H / 2}\n")
    with open(os.path.join(sparse, "images.txt"), "w") as f:
        for i, c in enumerate(cams, 1):
            q, t = c["q"], c["t"]
            f.write(f"{i} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {i} {c['name']}\n\n")


def _dense_fixture(tmp_path, W=64, H=48, shift=0.0, f=60.0):
    dense = tmp_path / "dense"
    spec = [{"name": f"v{i}.png", "q": (1.0, 0.0, 0.0, 0.0), "t": (0.1 * i, 0.0, 2.0), "f": f}
            for i in range(3)]
    _write_colmap_txt(str(dense / "sparse"), spec, W, H)
    os.makedirs(dense / "stereo" / "depth_maps")
    for i, s in enumerate(spec):
        d = np.full((H, W), 2.0 + i, dtype=np.float32)
        d[:, :8] = 0.0                                       # MVS hole
        d[0, 8] = np.nan
        write_colmap_array(depth_map_path(str(dense), s["name"]), d)
    train_spec = [dict(s, t=(s["t"][0] + shift, 0.0, 2.0)) for s in spec]
    _write_colmap_txt(str(tmp_path / "sparse"), train_spec, W, H)
    return str(dense), load_colmap_cameras(str(tmp_path / "sparse"))


def test_attach_mvs_depth_resamples_and_keeps_holes(tmp_path):
    dense, cams = _dense_fixture(tmp_path)
    cams = [c.scaled(0.5) for c in cams]                     # training at half resolution
    assert attach_mvs_depth(cams, dense) == 3
    d = cams[1].depth
    assert d.shape == (24, 32)
    assert float(d[:, :4].abs().max()) == 0.0                # hole stays a hole
    assert float(d[5, 10]) == 3.0 and bool(torch.isfinite(d).all())


def test_attach_mvs_depth_rejects_another_reconstruction(tmp_path):
    dense, cams = _dense_fixture(tmp_path, shift=0.5)        # same names, other poses
    with pytest.raises(ValueError, match="поза"):
        attach_mvs_depth(cams, dense)


def test_attach_mvs_depth_requires_distortion_free_training_cameras(tmp_path):
    dense, cams = _dense_fixture(tmp_path)
    assert attach_mvs_depth(cams, dense, train_sparse=str(tmp_path / "sparse")) == 3
    cam_txt = tmp_path / "sparse" / "cameras.txt"
    lines = cam_txt.read_text().splitlines()
    # SIMPLE_RADIAL f cx cy k: same pinhole part, but pixels differ from the undistorted map
    cam_txt.write_text("\n".join(f"{p[0]} SIMPLE_RADIAL {p[2]} {p[3]} {p[4]} {p[6]} {p[7]} 0.05"
                                 for p in (ln.split() for ln in lines)) + "\n")
    with pytest.raises(ValueError, match="SIMPLE_RADIAL"):
        attach_mvs_depth(cams, dense, train_sparse=str(tmp_path / "sparse"))


def test_attach_mvs_depth_rejects_other_intrinsics(tmp_path):
    dense, _ = _dense_fixture(tmp_path)
    _, cams = _dense_fixture(tmp_path / "x", f=90.0)
    with pytest.raises(ValueError, match="fx"):
        attach_mvs_depth(cams, dense)


# ── depth loss and schedule ─────────────────────────────────────────────
def test_depth_loss_is_relative_masked_and_points_the_right_way():
    target = torch.tensor([[10.0, 20.0], [0.0, 40.0]])      # 0 = no target
    pred = torch.tensor([[11.0, 18.0], [99.0, 40.0]], requires_grad=True)
    loss = depth_loss(pred, target)
    assert math.isclose(float(loss), (0.1 + 0.1 + 0.0) / 3, rel_tol=1e-6)
    loss.backward()
    assert pred.grad[0, 0] > 0 and pred.grad[0, 1] < 0 and pred.grad[1, 0] == 0
    # scale-free: the same relative errors in a 10x bigger scene
    assert math.isclose(float(depth_loss(pred.detach() * 10, target * 10)), float(loss), rel_tol=1e-6)
    valid = torch.tensor([[True, False], [True, True]])
    assert math.isclose(float(depth_loss(pred.detach(), target, valid)), 0.1 / 2, rel_tol=1e-6)
    none = depth_loss(pred, torch.zeros(2, 2))
    assert float(none) == 0.0 and none.requires_grad


def test_depth_weight_schedule():
    assert depth_weight_at(5, 100, 0.0, 0.01) == 0.0
    assert math.isclose(depth_weight_at(0, 100, 1.0, 0.01), 1.0)
    assert math.isclose(depth_weight_at(50, 100, 1.0, 0.01), 0.1)
    assert math.isclose(depth_weight_at(100, 100, 1.0, 0.01), 0.01)
    assert math.isclose(depth_weight_at(500, 100, 1.0, 0.01), 0.01)
    with pytest.raises(ValueError):
        depth_weight_at(1, 100, 1.0, 0.0)


# ── checkpoint files ────────────────────────────────────────────────────
def test_checkpoints_are_atomic_and_pruned(tmp_path):
    d = ckpt.checkpoint_dir(str(tmp_path / "gs" / "splat.pt"))
    assert d == str(tmp_path / "gs" / "splat_checkpoints")
    for it in (100, 200, 300):
        ckpt.write_checkpoint(d, it, {"x": torch.full((2,), float(it))}, keep=2)
    assert [i for i, _ in ckpt.list_checkpoints(d)] == [200, 300]
    assert not [n for n in os.listdir(d) if n.endswith(".tmp")]
    # a crash mid-write leaves only a .tmp, which is never picked up
    (tmp_path / "gs" / "splat_checkpoints" / "ckpt_0000400.pt.tmp").write_bytes(b"partial")
    path, payload = ckpt.load_latest_checkpoint(d, "cpu")
    assert path.endswith("ckpt_0000300.pt") and payload["iteration"] == 300
    assert float(payload["x"][0]) == 300.0 and payload["version"] == ckpt.CHECKPOINT_VERSION
    assert ckpt.clear_checkpoints(d) == 3 and ckpt.list_checkpoints(d) == []
    assert ckpt.load_latest_checkpoint(d, "cpu") is None


def test_corrupt_newest_checkpoint_falls_back_to_older(tmp_path, capsys):
    d = str(tmp_path)
    ckpt.write_checkpoint(d, 10, {"x": torch.zeros(1)}, keep=5)
    ckpt.write_checkpoint(d, 20, {"x": torch.ones(1)}, keep=5)
    newest = ckpt.list_checkpoints(d)[-1][1]
    with open(newest, "r+b") as f:
        f.truncate(40)
    path, payload = ckpt.load_latest_checkpoint(d, "cpu")
    assert payload["iteration"] == 10
    assert "не читається" in capsys.readouterr().out


def test_signature_diff_names_what_changed():
    a = {"options": {"sh_degree": 3, "mcmc": True}, "train_images": ["a", "b"]}
    b = {"options": {"sh_degree": 0, "mcmc": True}, "train_images": ["a"]}
    diff = ckpt.signature_diff(a, b)
    assert diff == ["options.sh_degree: 3 → 0", "train_images"]
    assert ckpt.signature_diff(a, a) == []


def test_resume_neutral_options_do_not_enter_the_signature():
    from aerosplat.gs.gsplat_backend import GsplatOptions, run_signature

    cams = [_cam("a"), _cam("b")]
    xyz = torch.zeros(5, 3)
    base = run_signature(GsplatOptions(), train_cams=cams, extent=10.0, xyz0=xyz)
    same = run_signature(GsplatOptions(checkpoint_every=7, checkpoint_keep=9, resume=True,
                                       eval_interval=3, save_every=5, tensorboard_dir="x"),
                         train_cams=cams, extent=10.0, xyz0=xyz)
    assert base == same
    cams[0].mask = torch.ones(24, 32, dtype=torch.bool)
    other = run_signature(GsplatOptions(depth_weight=0.1), train_cams=cams, extent=10.0, xyz0=xyz)
    assert set(ckpt.signature_diff(base, other)) == {"options.depth_weight: 0.0 → 0.1",
                                                     "masked_images"}


# ── config, pipeline and CLI wiring ─────────────────────────────────────
def test_config_defaults_off_and_gsplat_only_options():
    cfg = RunConfig()
    assert (cfg.depth_reg, cfg.checkpoint_every, cfg.train_resume, cfg.tensorboard, cfg.masks) == \
        ("none", 0, False, False, False)
    RunConfig(masks=True)                                   # masks work with every rasterizer
    RunConfig(rasterizer="gsplat", depth_reg="mvs", checkpoint_every=500, train_resume=True,
              tensorboard=True)
    for bad in ({"depth_reg": "mvs"}, {"checkpoint_every": 100}, {"train_resume": True},
                {"tensorboard": True}):
        with pytest.raises(ValueError, match="gsplat"):
            RunConfig(**bad)
    with pytest.raises(ValueError):
        RunConfig(rasterizer="gsplat", depth_weight=0)
    with pytest.raises(ValueError):
        RunConfig(rasterizer="gsplat", checkpoint_keep=0)


def test_train_argv_inputs_and_fingerprint_for_extras(tmp_path):
    from aerosplat.pipeline.layout import RunLayout
    from aerosplat.pipeline.runner import _fingerprint
    from aerosplat.pipeline.stages import STAGES, argv_train

    base = RunConfig(camera_source="synthetic", rasterizer="gsplat").to_dict()
    L = RunLayout.from_config(base, data_dir=tmp_path / "data", project_dir=tmp_path)
    stage = STAGES["train"]
    plain = argv_train(base, L)
    for flag in ("--depth-mvs", "--checkpoint-every", "--resume", "--tensorboard", "--masks"):
        assert flag not in plain
    assert stage.inputs(base, L) == [L.cloud_in]

    ops = dict(base, checkpoint_every=250, checkpoint_keep=3, train_resume=True, tensorboard=True)
    a = argv_train(ops, L)
    assert a[a.index("--checkpoint-every") + 1] == "250" and "--resume" in a
    assert a[a.index("--tensorboard") + 1] == str(L.splat_pt.parent / f"{L.splat_pt.stem}_tensorboard")
    assert _fingerprint(stage, ops, L) == _fingerprint(stage, base, L)   # same splat

    full = dict(base, depth_reg="mvs", depth_weight=0.5, masks=True)
    a = argv_train(full, L)
    assert a[a.index("--depth-mvs") + 1] == str(L.dense)
    assert a[a.index("--depth-weight") + 1] == "0.5"
    assert a[a.index("--masks") + 1] == str(L.footage.parent / "masks")
    assert stage.inputs(full, L) == [L.cloud_in, L.footage.parent / "masks",
                                     L.dense / "stereo" / "depth_maps"]
    assert _fingerprint(stage, full, L) != _fingerprint(stage, base, L)
    assert "depth_weight" in stage.keys_for(full) and "masks_dir" in stage.keys_for(full)

    own = dict(base, masks=True, masks_dir="elsewhere/m")
    a = argv_train(own, L)
    assert a[a.index("--masks") + 1] == str(tmp_path / "elsewhere" / "m")

    # Without known poses training switches to the undistorted dense/images: masks drawn on
    # the original frames would be misaligned there, so only an explicit masks_dir is taken.
    (L.dense / "sparse").mkdir(parents=True)
    (L.dense / "images").mkdir()
    with pytest.raises(ValueError, match="undistorted"):
        argv_train(dict(base, masks=True), L)
    assert "--masks" in argv_train(own, L)


def test_trainer_cli_passes_extras(monkeypatch):
    from aerosplat.gs import trainer

    seen = {}
    monkeypatch.setattr(trainer, "train", lambda **kw: seen.update(kw))
    trainer.main(["--self-test", "--output", "x.pt", "--rasterizer", "gsplat",
                  "--depth-mvs", "d", "--depth-weight", "0.3", "--depth-weight-final", "0.03",
                  "--checkpoint-every", "100", "--checkpoint-keep", "4", "--resume",
                  "--tensorboard", "tb", "--masks", "m"])
    assert seen["depth_mvs_dir"].endswith("d") and seen["depth_weight"] == 0.3
    assert seen["depth_weight_final"] == 0.03 and seen["checkpoint_every"] == 100
    assert seen["checkpoint_keep"] == 4 and seen["resume"] is True
    assert seen["tensorboard_dir"].endswith("tb") and seen["masks_dir"].endswith("m")
    seen.clear()
    trainer.main(["--self-test", "--output", "x.pt"])
    assert seen["depth_weight"] == 0.0 and seen["checkpoint_every"] == 0
    assert seen["resume"] is False and seen["masks_dir"] is None and seen["tensorboard_dir"] is None


def test_train_refuses_gsplat_only_extras_on_other_rasterizers(tmp_path):
    from aerosplat.gs.trainer import train

    out = str(tmp_path / "x.pt")
    for kw in ({"resume": True}, {"checkpoint_every": 10}, {"tensorboard_dir": "tb"},
               {"depth_mvs_dir": "d", "depth_weight": 0.1}):
        with pytest.raises(ValueError, match="gsplat"):
            train(pt_path="", output_path=out, self_test=True, rasterizer_kind="torch", **kw)
    with pytest.raises(ValueError, match="depth"):
        train(pt_path="", output_path=out, self_test=True, rasterizer_kind="gsplat",
              depth_mvs_dir="d")
    with pytest.raises(ValueError, match="self-test"):
        train(pt_path="", output_path=out, self_test=True, rasterizer_kind="torch",
              masks_dir=str(tmp_path))


def test_tensorboard_missing_package_is_a_clear_error(monkeypatch):
    import builtins

    from aerosplat.gs import gsplat_backend

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch.utils.tensorboard":
            raise ImportError("TensorBoard logging requires TensorBoard")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="pip install tensorboard"):
        gsplat_backend._tensorboard_writer("tb")


# ═══════════════════════════════════════════════════════════════════════
#  GPU: tiny synthetic scene through gsplat
# ═══════════════════════════════════════════════════════════════════════
def _gsplat_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA недоступна")
    from aerosplat.gs.gsplat_backend import require_gsplat

    try:
        require_gsplat()
    except (ImportError, RuntimeError) as e:
        pytest.skip(str(e))


def _look_at(C, target=(0.0, 0.0, 0.0)):
    C = torch.tensor(C, dtype=torch.float32)
    f = torch.nn.functional.normalize(torch.tensor(target) - C, dim=0)
    r = torch.nn.functional.normalize(torch.linalg.cross(f, torch.tensor([0.0, 0.0, 1.0])), dim=0)
    d = torch.linalg.cross(f, r)
    R = torch.stack([r, d, f])
    return R, -R @ C


def _tiny_scene(n_views=10, W=64, H=48, seed=0):
    """Textured plane z=0 seen from above; images and true depth rendered by gsplat."""
    from aerosplat.gs.gsplat_backend import GsplatRasterizer

    g = torch.Generator().manual_seed(seed)
    n = 900
    side = int(math.sqrt(n))
    u = torch.linspace(-1.2, 1.2, side)
    xyz = torch.stack(torch.meshgrid(u, u, indexing="ij"), -1).reshape(-1, 2)
    xyz = torch.cat([xyz, torch.zeros(len(xyz), 1)], 1)
    rgb = torch.rand(len(xyz), 3, generator=g)
    gt = {"xyz": xyz, "scale": torch.full((len(xyz), 3), 0.05), "rgb": rgb,
          "rotation": torch.tensor([[1.0, 0, 0, 0]]).repeat(len(xyz), 1),
          "opacity": torch.full((len(xyz),), 0.95)}
    gt["scale"][:, 2] = 0.005
    gt = {k: v.cuda() for k, v in gt.items()}
    raster = GsplatRasterizer("classic")
    cams = []
    for i in range(n_views):
        a = 2 * math.pi * i / n_views
        R, t = _look_at((1.2 * math.cos(a), 1.2 * math.sin(a), 2.5))
        cam = Camera(R, t, 55.0, 55.0, W / 2, H / 2, W, H, f"v{i:02d}.png")
        bg = torch.zeros(3, device="cuda")
        with torch.no_grad():
            img, depth, alpha, _ = raster.rasterize_with_depth(gt, cam, cam.view_matrix("cuda"), bg)
        cam.image = img.clamp(0, 1).cpu()
        cam.depth = torch.where(alpha > 0.5, depth, torch.zeros_like(depth)).cpu()
        cams.append(cam)
    return gt, cams


def _train(opts_kw, cams, init_xyz, init_rgb, out, test_cams=(), extent=None):
    from aerosplat.gs.gsplat_backend import GsplatOptions, train_gsplat
    from aerosplat.gs.trainer import scene_extent

    opts = GsplatOptions(**dict(dict(sh_degree=0, rasterize_mode="classic", eval_interval=0,
                                     scale_reg=0.0), **opts_kw))
    torch.manual_seed(0)
    return train_gsplat(opts, xyz0=init_xyz.cuda(), rgb0=init_rgb.cuda(),
                        init_scale=torch.full((len(init_xyz),), 0.04, device="cuda"),
                        train_cams=cams, test_cams=list(test_cams),
                        extent=extent or scene_extent(cams),
                        device=torch.device("cuda"), output_path=str(out))


def _mean_psnr(model, cams):
    from aerosplat.gs.evaluate import mean_test_psnr
    from aerosplat.gs.gsplat_backend import GsplatRasterizer

    return mean_test_psnr(model.get_params(), GsplatRasterizer("classic"), cams, torch.device("cuda"))


def _noisy_init(gt, sigma=0.25, seed=1):
    g = torch.Generator().manual_seed(seed)
    xyz = gt["xyz"].cpu().clone()
    xyz[:, 2] += torch.randn(len(xyz), generator=g) * sigma
    return xyz, gt["rgb"].cpu()


def _interrupted(monkeypatch):
    """The run's end-of-training cleanup is skipped, as if it crashed right after the
    last checkpoint: the checkpoints stay for a resume."""
    from aerosplat.gs import gsplat_backend

    monkeypatch.setattr(gsplat_backend, "clear_checkpoints", lambda d: 0)
    return lambda: monkeypatch.setattr(gsplat_backend, "clear_checkpoints", ckpt.clear_checkpoints)


@pytest.mark.parametrize("strategy,appearance", [("mcmc", "none"), ("default", "exposure")])
def test_gpu_resume_matches_a_straight_run(tmp_path, capsys, monkeypatch, strategy, appearance):
    _gsplat_or_skip()
    gt, cams = _tiny_scene()
    for c in cams:
        c.depth = None
    xyz, rgb = _noisy_init(gt)
    kw = dict(num_iterations=200, densify_strategy=strategy, appearance=appearance,
              densify_from=40, densify_until=160, densify_interval=40, max_gaussians=1500,
              opacity_reset_interval=120, checkpoint_every=50, checkpoint_keep=2)
    out = tmp_path / "gs" / "splat.pt"
    d = ckpt.checkpoint_dir(str(out))
    restore = _interrupted(monkeypatch)
    straight = _train(kw, cams, xyz, rgb, out)
    log_a = capsys.readouterr().out
    n_a, psnr_a = straight.n_gaussians, _mean_psnr(straight, cams)
    # every 50, newest 2 kept, none at the final iteration (the .pt is the result)
    assert [i for i, _ in ckpt.list_checkpoints(d)] == [100, 150]

    # interrupted after 100: resume from there to 200
    os.remove(ckpt.list_checkpoints(d)[-1][1])
    os.remove(out)
    restore()
    resumed = _train(dict(kw, resume=True), cams, xyz, rgb, out)
    log_b = capsys.readouterr().out
    assert "Resume:" in log_b and "з ітерації 101" in log_b
    assert out.is_file() and ckpt.list_checkpoints(d) == [] and not os.path.exists(d)

    assert resumed.n_gaussians == n_a
    psnr_b = _mean_psnr(resumed, cams)
    same = all(torch.equal(straight.splats[k], resumed.splats[k]) for k in straight.splats)
    print(f"resume: PSNR {psnr_a:.4f} vs {psnr_b:.4f}, params bit-identical: {same}")
    # Not bit-exact: gsplat's backward accumulates with CUDA atomics, so two straight runs
    # differ by the same amount (measured max |Δparam| ~1e-3 both ways on this scene).
    assert abs(psnr_b - psnr_a) < 0.3, (psnr_a, psnr_b)
    mean_loss = [float(line.split()[-1]) for log in (log_a, log_b)
                 for line in log.splitlines() if "Середній loss" in line]
    assert abs(mean_loss[0] - mean_loss[1]) < 0.02 * mean_loss[0], mean_loss


def test_gpu_resume_refuses_a_changed_config(tmp_path, monkeypatch):
    _gsplat_or_skip()
    gt, cams = _tiny_scene(n_views=4)
    xyz, rgb = _noisy_init(gt)
    out = tmp_path / "s.pt"
    d = ckpt.checkpoint_dir(str(out))
    kw = dict(num_iterations=20, densify_from=1000, checkpoint_every=10)
    restore = _interrupted(monkeypatch)
    _train(kw, cams, xyz, rgb, out)
    restore()
    assert [i for i, _ in ckpt.list_checkpoints(d)] == [10]
    with pytest.raises(RuntimeError, match="sh_degree"):
        _train(dict(kw, resume=True, sh_degree=1), cams, xyz, rgb, out)
    assert [i for i, _ in ckpt.list_checkpoints(d)] == [10]       # refused: nothing touched
    # without resume a new run supersedes the old checkpoints and cleans up when done
    _train(kw, cams, xyz, rgb, out)
    assert ckpt.list_checkpoints(d) == []


def _depth_error(model, cams):
    from aerosplat.gs.gsplat_backend import GsplatRasterizer

    raster, bg = GsplatRasterizer("classic"), torch.zeros(3, device="cuda")
    errs = []
    with torch.no_grad():
        for cam in cams:
            _, d, a, _ = raster.rasterize_with_depth(model.get_params(), cam,
                                                    cam.view_matrix("cuda"), bg)
            t = cam.depth.cuda()
            m = (t > 0) & (a > 0.5)
            errs.append(float(((d[m] - t[m]).abs() / t[m]).mean()))
    return sum(errs) / len(errs)


def test_gpu_depth_regularisation_pulls_geometry_to_the_target(tmp_path):
    _gsplat_or_skip()
    gt, cams = _tiny_scene()
    xyz, rgb = _noisy_init(gt, sigma=0.1)
    # extent only scales the position lr here (densification is off): ~20x the
    # default, so 300 iterations can move splats by the noise amplitude at all
    kw = dict(num_iterations=300, densify_strategy="default", densify_from=10_000)
    plain = _train(kw, cams, xyz, rgb, tmp_path / "plain.pt", extent=30.0)
    with_depth = _train(dict(kw, depth_weight=1.0, depth_weight_final=0.1), cams, xyz, rgb,
                        tmp_path / "depth.pt", extent=30.0)
    e_plain, e_depth = _depth_error(plain, cams), _depth_error(with_depth, cams)
    # measured: 0.0125 -> 0.0076 (seed 0), 0.0120 -> 0.0075 (seed 1); weight 3 -> 0.0045
    assert e_depth < 0.75 * e_plain, (e_plain, e_depth)
    # the price on the training views stays small (measured −0.4…−0.6 dB)
    assert _mean_psnr(with_depth, cams) > _mean_psnr(plain, cams) - 1.5


def test_gpu_masks_hide_a_corrupted_region(tmp_path):
    _gsplat_or_skip()
    gt, cams = _tiny_scene()
    for c in cams:
        c.depth = None
    clean = [Camera(c.R, c.t, c.fx, c.fy, c.cx, c.cy, c.W, c.H, c.name, c.image.clone())
             for c in cams]
    for c in cams:                                  # a "moving object": magenta block
        c.image[10:30, 20:44] = torch.tensor([1.0, 0.0, 1.0])
    xyz, rgb = _noisy_init(gt, sigma=0.05)
    kw = dict(num_iterations=300, densify_strategy="default", densify_from=10_000)
    unmasked = _train(kw, cams, xyz, rgb, tmp_path / "u.pt")
    for c in cams:
        c.mask = torch.ones(c.H, c.W, dtype=torch.bool)
        c.mask[10:30, 20:44] = False
    masked = _train(kw, cams, xyz, rgb, tmp_path / "m.pt")
    p_unmasked, p_masked = _mean_psnr(unmasked, clean), _mean_psnr(masked, clean)
    assert p_masked > p_unmasked + 2.0, (p_unmasked, p_masked)


def test_gpu_cli_end_to_end_with_masks_depth_and_resume(tmp_path, capsys, monkeypatch):
    """Files laid out like a run: sparse model, frames, dense/ depth maps, masks, cloud."""
    _gsplat_or_skip()
    from aerosplat.gs.trainer import main

    gt, cams = _tiny_scene(n_views=6)
    images, masks, sparse = tmp_path / "images", tmp_path / "masks", tmp_path / "sparse"
    dense = tmp_path / "dense"
    for d in (images, masks, dense / "stereo" / "depth_maps"):
        d.mkdir(parents=True)
    spec = []
    for c in cams:
        Image.fromarray((c.image.numpy() * 255).round().astype(np.uint8)).save(images / c.name)
        m = np.full((c.H, c.W), 255, np.uint8)
        m[:4] = 0
        Image.fromarray(m).save(masks / (c.name + ".png"))
        write_colmap_array(depth_map_path(str(dense), c.name), c.depth.numpy())
        q = _quat(c.R)
        spec.append({"name": c.name, "q": q, "t": c.t.tolist(), "f": c.fx})
    _write_colmap_txt(str(sparse), spec, cams[0].W, cams[0].H)
    _write_colmap_txt(str(dense / "sparse"), spec, cams[0].W, cams[0].H)
    xyz, rgb = _noisy_init(gt, sigma=0.1)
    torch.save({"xyz": xyz, "rgb": rgb}, tmp_path / "cloud.pt")
    out = tmp_path / "gs" / "splat.pt"
    argv = ["--input", str(tmp_path / "cloud.pt"), "--output", str(out), "--colmap", str(sparse),
            "--images", str(images), "--rasterizer", "gsplat", "--iterations", "40",
            "--max-res", "64", "--sh-degree", "0", "--densify-from", "10", "--densify-interval",
            "10", "--max-gaussians", "1200", "--depth-mvs", str(dense), "--depth-weight", "0.5",
            "--masks", str(masks), "--checkpoint-every", "20", "--eval-every", "0"]
    restore = _interrupted(monkeypatch)
    assert main(argv) == 0
    log = capsys.readouterr().out
    assert "Маски: 6 з 6" in log and "Глибина MVS (geometric): 6 з 6" in log
    assert "depth 0.5→0.01" in log and "| depth " in log
    assert out.is_file()
    d = str(tmp_path / "gs" / "splat_checkpoints")
    assert [i for i, _ in ckpt.list_checkpoints(d)] == [20]
    restore()
    out.unlink()
    assert main(argv + ["--resume"]) == 0
    assert "з ітерації 21" in capsys.readouterr().out
    assert out.is_file() and ckpt.list_checkpoints(d) == []


def _quat(R):
    """Rotation matrix -> COLMAP qvec (w, x, y, z)."""
    m = R.double()
    w = math.sqrt(max(0.0, 1.0 + float(m[0, 0] + m[1, 1] + m[2, 2]))) / 2
    if w > 1e-6:
        return (w, float(m[2, 1] - m[1, 2]) / (4 * w), float(m[0, 2] - m[2, 0]) / (4 * w),
                float(m[1, 0] - m[0, 1]) / (4 * w))
    # 180° rotations do not occur in these fixtures
    raise ValueError("rotation too close to 180°")
