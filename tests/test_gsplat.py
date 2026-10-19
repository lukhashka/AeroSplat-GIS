"""gsplat backend of the trainer (ROADMAP 2.2): SH maths, the .pt contract with SH,
budgeted densification, config/pipeline wiring, georef of view-dependent colour.

CPU tests need neither CUDA nor gsplat (densifier tests need the gsplat package
for its pure-torch ops). GPU tests run only with CUDA and a usable gsplat build
(on Windows: under tools/gsplat_env.bat).
"""
import math

import numpy as np
import pytest
import torch

from aerosplat.config import RunConfig
from aerosplat.geo.affine import AffineMap
from aerosplat.geo.georef import transform_splat
from aerosplat.gs import evaluate
from aerosplat.gs.gsplat_backend import (
    ExposureCorrection, GsplatOptions, SplatModel, save_splat,
)
from aerosplat.gs.sh import (
    C0, eval_sh, eval_sh_basis, num_sh_bases, rgb_to_sh_dc, rotate_sh, sh_dc_to_rgb, sh_degree_of,
)
from aerosplat.gs.trainer import Camera


def _random_rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _unit(rng, n):
    d = rng.normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


# ── SH maths ─────────────────────────────────────────────────────────────
def test_sh_basis_is_orthonormal_on_the_sphere():
    # Catches a wrong constant or sign mix-up in any of the 16 basis functions.
    rng = np.random.default_rng(0)
    Y = eval_sh_basis(3, _unit(rng, 400_000))
    gram = 4 * math.pi * (Y.T @ Y) / len(Y)
    np.testing.assert_allclose(gram, np.eye(16), atol=0.02)


def test_sh_basis_matches_gsplat_reference():
    impl = pytest.importorskip("gsplat.cuda._torch_impl")
    rng = np.random.default_rng(1)
    dirs = _unit(rng, 50)
    coeffs = rng.normal(size=(50, 16, 3))
    ref = impl._spherical_harmonics(3, torch.from_numpy(dirs), torch.from_numpy(coeffs))
    np.testing.assert_allclose(eval_sh(coeffs, dirs) - 0.5, ref.numpy(), atol=1e-10)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_rotate_sh_reproduces_colour_along_rotated_directions(degree):
    rng = np.random.default_rng(degree)
    R = _random_rotation(rng)
    sh = rng.normal(size=(20, num_sh_bases(degree), 3))
    dirs = _unit(rng, 20)
    rotated = rotate_sh(sh, R)
    # a frame change d' = R d must leave the colour of every view unchanged
    np.testing.assert_allclose(eval_sh(rotated, dirs @ R.T), eval_sh(sh, dirs), atol=1e-10)
    np.testing.assert_allclose(rotated[:, 0], sh[:, 0], atol=1e-12)      # DC is invariant


def test_rotate_sh_identity_is_exact():
    sh = np.random.default_rng(5).normal(size=(7, 16, 3))
    np.testing.assert_allclose(rotate_sh(sh, np.eye(3)), sh, atol=1e-12)


def test_sh_degree_and_dc_helpers():
    assert [sh_degree_of(k) for k in (1, 4, 9, 16)] == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        sh_degree_of(5)
    with pytest.raises(ValueError):
        sh_degree_of(25)          # degree 4 is not supported
    rgb = torch.rand(10, 3)
    torch.testing.assert_close(sh_dc_to_rgb(rgb_to_sh_dc(rgb)), rgb)
    assert float(sh_dc_to_rgb(np.array(10.0))) == 1.0       # clamped, numpy works too
    assert math.isclose(float(sh_dc_to_rgb(np.array(1.0))), 0.5 + C0)


# ── model, export, activation contract ──────────────────────────────────
def _splat_model(n=32, sh_degree=3):
    g = torch.Generator().manual_seed(0)
    xyz = torch.randn(n, 3, generator=g) * 5
    rgb = torch.rand(n, 3, generator=g) * 0.9 + 0.05
    model = SplatModel(xyz, rgb, torch.full((n,), 0.2), sh_degree=sh_degree)
    return model, xyz, rgb


def test_export_keeps_activation_contract_and_dc_as_rgb(tmp_path):
    model, xyz, rgb = _splat_model()
    with torch.no_grad():
        model.splats["shN"].normal_()          # view-dependent part must not leak into rgb
    path = tmp_path / "splat.pt"
    save_splat(model, str(path), "antialiased")
    scene = torch.load(path, weights_only=False)

    assert scene["activations"] == {"scale_is_log": False, "opacity_is_logit": False}
    assert scene["render"] == {"rasterizer": "gsplat", "rasterize_mode": "antialiased"}
    assert scene["sh_degree"] == 3 and scene["sh"].shape == (32, 16, 3)
    torch.testing.assert_close(scene["xyz"], xyz)
    torch.testing.assert_close(scene["rgb"], rgb)                       # DC -> rgb exactly
    torch.testing.assert_close(scene["sh"][:, 0], rgb_to_sh_dc(rgb))
    torch.testing.assert_close(scene["scale"], torch.full((32, 3), 0.2))
    assert scene["opacity"].shape == (32,)
    torch.testing.assert_close(scene["opacity"], torch.full((32,), 0.1))
    torch.testing.assert_close(scene["rotation"][:, 0], torch.ones(32))
    assert all(v.device.type == "cpu" for v in scene.values() if isinstance(v, torch.Tensor))

    params, render = evaluate.load_scene(str(path), torch.device("cpu"))
    assert render["rasterize_mode"] == "antialiased"
    assert params["sh"].shape == (32, 16, 3) and "cov3d" in params


def test_export_rgb_is_clamped_dc_colour():
    model, _, _ = _splat_model(n=4, sh_degree=0)
    assert "shN" not in model.splats
    with torch.no_grad():
        model.splats["sh0"][0] = 5.0           # far outside [0, 1] as a colour
    out = model.export()
    assert out["sh"].shape == (4, 1, 3) and out["sh_degree"] == 0
    torch.testing.assert_close(out["rgb"][0], torch.ones(3))
    assert out["sh"][0, 0, 0] == 5.0           # the raw coefficient is kept lossless


def test_legacy_splat_loads_without_sh(tmp_path):
    from conftest import write_pt

    path = tmp_path / "old.pt"
    write_pt(path, "splat")
    params, render = evaluate.load_scene(str(path), torch.device("cpu"))
    assert "sh" not in params and render == {}


def test_options_validate_and_schedule_sh():
    opts = GsplatOptions(sh_degree=3, sh_degree_interval=1000)
    assert [opts.active_sh_degree(i) for i in (1, 999, 1000, 2500, 3000, 9000)] == [0, 0, 1, 2, 3, 3]
    assert GsplatOptions(sh_degree=2, sh_degree_interval=0).active_sh_degree(1) == 2
    for bad in ({"rasterize_mode": "mip"}, {"densify_strategy": "abs"}, {"appearance": "embed"},
                {"sh_degree": 4}):
        with pytest.raises(ValueError):
            GsplatOptions(**bad)


def test_exposure_starts_neutral_and_has_no_shared_shift():
    exp = ExposureCorrection(5)
    img = torch.rand(4, 6, 3)
    torch.testing.assert_close(exp(img, 2), img)
    with torch.no_grad():
        exp.delta.fill_(0.3)                    # the same correction for every image ...
    torch.testing.assert_close(exp(img, 1), img)  # ... is the neutral appearance
    with torch.no_grad():
        exp.delta.zero_()
        exp.delta[0, :, 3] = 0.5                # bias on image 0 only, minus the mean bias
    assert torch.allclose(exp(img, 0) - img, torch.full_like(img, 0.5 * 4 / 5))
    assert torch.allclose(exp(img, 1) - img, torch.full_like(img, -0.5 / 5))


# ── densification (gsplat's pure-torch ops) ─────────────────────────────
def _meta_for(n, grads, radii_px, W=100, H=50):
    means2d = torch.zeros(1, n, 2, requires_grad=True)
    means2d.grad = torch.as_tensor(grads, dtype=torch.float32).reshape(1, n, 2)
    radii = torch.as_tensor(radii_px, dtype=torch.int32).reshape(1, n, 1).repeat(1, 1, 2)
    return {"means2d": means2d, "radii": radii, "width": W, "height": H, "gaussian_ids": None}


def test_densifier_respects_budget_and_prefers_high_gradients():
    pytest.importorskip("gsplat.strategy.ops")
    from aerosplat.gs.gsplat_backend import GsplatDensifier, create_optimizers

    n = 10
    model, _, _ = _splat_model(n=n)          # scale 0.2 < percent_dense * extent: clones
    opts = create_optimizers(model, extent=100.0)
    dens = GsplatDensifier(model, grad_threshold=0.0002)
    # NDC gradient = px gradient * W/2: 1e-4 px * 50 = 5e-3 > threshold for the first six
    g = np.zeros((n, 2))
    g[:6, 0] = np.linspace(1e-4, 6e-4, 6)
    dens.update_stats(_meta_for(n, g, np.full(n, 3)))
    assert int(dens.state["count"].sum()) == n

    dens.densify_and_prune(opts, extent=100.0, max_gaussians=13)       # room for 3 only
    assert model.n_gaussians == 13
    # the three highest gradients (indices 3, 4, 5) were the ones cloned
    means = model.splats["means"].detach()
    for i in (3, 4, 5):
        assert (means == means[i]).all(dim=1).sum() == 2
    for name, opt in opts.items():                                      # Adam follows
        assert opt.param_groups[0]["params"][0] is model.splats[name]
    assert dens.state["grad2d"].shape == (13,) and float(dens.state["grad2d"].sum()) == 0.0


def test_densifier_prunes_screen_wide_but_not_split_children():
    pytest.importorskip("gsplat.strategy.ops")
    from aerosplat.gs.gsplat_backend import GsplatDensifier, create_optimizers

    n = 6
    model, _, _ = _splat_model(n=n)                    # all > percent_dense * extent: splits
    with torch.no_grad():
        model.splats["scales"][0] = math.log(0.5)
    opts = create_optimizers(model, extent=10.0)
    dens = GsplatDensifier(model, grad_threshold=0.0002)
    g = np.zeros((n, 2))
    g[0, 0] = 1e-3
    dens.update_stats(_meta_for(n, g, [40, 3, 3, 3, 3, 40]))   # 0 and 5 are screen-wide
    dens.densify_and_prune(opts, extent=10.0, max_gaussians=100, max_screen_radius=20.0)
    # parent 0 replaced by two children (kept), splat 5 pruned: 6 - 1 + 2 - 1
    assert model.n_gaussians == 6
    assert torch.allclose(torch.exp(model.splats["scales"][-2:]), torch.full((2, 3), 0.5 / 1.6))


# ── config and pipeline wiring ──────────────────────────────────────────
def test_config_accepts_gsplat_options():
    assert RunConfig().rasterizer == "auto"                 # gsplat stays opt-in
    cfg = RunConfig(rasterizer="gsplat", sh_degree=2, densify_strategy="default",
                    appearance="exposure")
    assert cfg.rasterize_mode == "antialiased" and cfg.absgrad is False
    assert RunConfig().densify_strategy == GsplatOptions().densify_strategy == "mcmc"
    for bad in ({"sh_degree": 4}, {"rasterize_mode": "mip"}, {"densify_strategy": "abs"},
                {"rasterizer": "inria"}, {"absgrad_threshold": 0}):
        with pytest.raises(ValueError):
            RunConfig(**bad)


def test_gsplat_preset_is_valid():
    from conftest import ROOT
    from aerosplat.config import build_config

    cfg = build_config(ROOT / "configs" / "gsplat-earthstudio.yaml")
    assert (cfg.rasterizer, cfg.densify_strategy, cfg.sh_degree) == ("gsplat", "mcmc", 0)
    # Owner's decision 2026-10-19: MVS depth on for Earth Studio (geometry -20 %, docs/BASELINE.md).
    assert (cfg.depth_reg, cfg.depth_weight, cfg.depth_weight_final) == ("mvs", 1.0, 0.01)


def test_train_argv_and_fingerprint_carry_gsplat_keys_only_for_gsplat(tmp_path):
    from aerosplat.pipeline.layout import RunLayout
    from aerosplat.pipeline.stages import STAGES, argv_train

    base = RunConfig(camera_source="synthetic").to_dict()
    L = RunLayout.from_config(base, data_dir=tmp_path / "data", project_dir=tmp_path)
    plain = argv_train(base, L)
    assert "--sh-degree" not in plain and "--absgrad" not in plain
    assert STAGES["train"].keys_for(base) == STAGES["train"].keys     # old fingerprints kept

    gs = dict(base, rasterizer="gsplat", absgrad=True, densify_strategy="mcmc")
    a = argv_train(gs, L)
    assert a[a.index("--rasterizer") + 1] == "gsplat"
    assert a[a.index("--densify-strategy") + 1] == "mcmc" and "--absgrad" in a
    assert a[a.index("--rasterize-mode") + 1] == "antialiased"
    assert "sh_degree" in STAGES["train"].keys_for(gs)


def test_trainer_cli_passes_gsplat_options(monkeypatch):
    from aerosplat.gs import trainer

    seen = {}
    monkeypatch.setattr(trainer, "train", lambda **kw: seen.update(kw))
    trainer.main(["--self-test", "--output", "x.pt", "--rasterizer", "gsplat", "--sh-degree", "2",
                  "--rasterize-mode", "classic", "--densify-strategy", "mcmc", "--absgrad",
                  "--appearance", "exposure", "--mcmc-noise-lr", "1000"])
    assert seen["rasterizer_kind"] == "gsplat" and seen["sh_degree"] == 2
    assert seen["rasterize_mode"] == "classic" and seen["densify_strategy"] == "mcmc"
    assert seen["absgrad"] is True and seen["appearance"] == "exposure"
    assert seen["mcmc_noise_lr"] == 1000 and seen["sh_degree_interval"] == 1000


# ── georef of view-dependent colour ─────────────────────────────────────
def _write_sh_splat(path, n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    sh = torch.randn(n, 16, 3, generator=g, dtype=torch.float64).float()
    rot = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=1)
    torch.save({"xyz": torch.randn(n, 3, generator=g) * 50, "scale": torch.full((n, 3), 0.3),
                "rotation": rot, "opacity": torch.full((n,), 0.7), "rgb": sh_dc_to_rgb(sh[:, 0]),
                "sh": sh, "sh_degree": 3,
                "activations": {"scale_is_log": False, "opacity_is_logit": False},
                "render": {"rasterizer": "gsplat", "rasterize_mode": "antialiased"}}, path)
    return sh


def test_georef_rotates_sh_with_the_frame(tmp_path):
    rng = np.random.default_rng(7)
    R = _random_rotation(rng)
    src, dst = tmp_path / "splat.pt", tmp_path / "splat_wgs84.pt"
    sh = _write_sh_splat(src)
    aff = AffineMap(1.5 * R, np.array([10.0, -3.0, 2.0]))     # similarity: exact for SH
    transform_splat(src, dst, aff, {"crs": "test"})
    out = torch.load(dst, weights_only=False)

    dirs = _unit(rng, 40)
    want = eval_sh(sh.double().numpy(), dirs)                   # colour seen along d before ...
    got = eval_sh(out["sh"].double().numpy(), dirs @ R.T)       # ... and along A d after
    np.testing.assert_allclose(got, want, atol=1e-5)
    torch.testing.assert_close(out["rgb"], sh_dc_to_rgb(sh[:, 0]))
    assert out["render"]["rasterize_mode"] == "antialiased" and out["sh_degree"] == 3


def test_georef_dataset_v6_map_leaves_sh_unchanged(tmp_path):
    # sphere ENU -> WGS84 Site ENU on dataset_v6: diagonal A, no rotation part
    src, dst = tmp_path / "splat.pt", tmp_path / "splat_wgs84.pt"
    sh = _write_sh_splat(src, seed=3)
    aff = AffineMap(np.diag([1.002558, 0.998712, 1.0]), np.zeros(3))
    assert aff.rotation_deg() == pytest.approx(0.0, abs=1e-9)
    transform_splat(src, dst, aff, {"crs": "test"})
    torch.testing.assert_close(torch.load(dst, weights_only=False)["sh"], sh)


# ── GPU: gsplat itself ──────────────────────────────────────────────────
def _gsplat_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA недоступна")
    from aerosplat.gs.gsplat_backend import require_gsplat

    try:
        require_gsplat()
    except (ImportError, RuntimeError) as e:
        pytest.skip(str(e))


def _gpu_scene(n=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    xyz = torch.randn(n, 3, generator=g) * 0.4
    xyz[:, 2] += 3.0
    rot = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=1)
    p = {"xyz": xyz, "scale": torch.rand(n, 3, generator=g) * 0.05 + 0.01, "rotation": rot,
         "opacity": torch.rand(n, generator=g) * 0.8 + 0.1, "rgb": torch.rand(n, 3, generator=g)}
    return {k: v.cuda() for k, v in p.items()}


def test_gpu_gsplat_classic_matches_tile_rasterizer():
    _gsplat_or_skip()
    from aerosplat.gs.gsplat_backend import GsplatRasterizer
    from aerosplat.gs.trainer import TileRasterizer, build_covariance_3d

    p = _gpu_scene()
    p["cov3d"] = build_covariance_3d(torch.log(p["scale"]), p["rotation"])
    cam = Camera(torch.eye(3), torch.zeros(3), 120.0, 120.0, 64, 48, 128, 96, "c")
    view, bg = cam.view_matrix("cuda"), torch.zeros(3, device="cuda")
    # Same camera for both: they sample pixel centres (+0.5), like Inria and COLMAP.
    ref, _ = TileRasterizer().render(p, cam, view, bg)
    out, meta = GsplatRasterizer("classic").render(p, cam, view, bg)
    assert out.shape == ref.shape == (96, 128, 3)
    diff = (out - ref).abs()
    assert float(diff.max()) < 2e-2, float(diff.max())
    assert float(diff.mean()) < 1e-3, float(diff.mean())


def test_gpu_gsplat_sh_matches_numpy_and_short_train(tmp_path):
    _gsplat_or_skip()
    from gsplat import spherical_harmonics

    from aerosplat.gs.trainer import train

    rng = np.random.default_rng(2)
    dirs, coeffs = _unit(rng, 64), rng.normal(size=(64, 16, 3))
    got = spherical_harmonics(3, torch.from_numpy(dirs).float().cuda(),
                              torch.from_numpy(coeffs).float().cuda())
    np.testing.assert_allclose(got.cpu().numpy(), eval_sh(coeffs, dirs) - 0.5, atol=1e-4)

    out = tmp_path / "st.pt"
    train(pt_path="", output_path=str(out), self_test=True, rasterizer_kind="gsplat",
          num_iterations=60, densify_from=20, densify_interval=20, sh_degree_interval=20)
    scene = torch.load(out, weights_only=False)
    assert scene["sh"].shape[1] == 16 and scene["render"]["rasterize_mode"] == "antialiased"
    assert torch.isfinite(scene["xyz"]).all()
