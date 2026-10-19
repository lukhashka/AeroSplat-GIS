"""VPS (aerosplat.vps, ADR-0015 layer A) on the CPU: localizer geometry and refusals, the atlas
file, the builder on a synthetic textured plane, perturbations, and the companion's imports.

The safety property under test: the localizer returns a pose that is right, or a refusal with
a reason — never a confidently wrong pose.
"""

import subprocess
import sys

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="aerosplat[vps] (opencv-python-headless) not installed")

from aerosplat.vps import perturb  # noqa: E402
from aerosplat.vps.atlas import Atlas, AtlasError, concat_keyframes  # noqa: E402
from aerosplat.vps.build import DepthFilter, View, build_atlas, grid_views, keypoint_depths  # noqa: E402
from aerosplat.vps.features import Features, bucket, make_extractor  # noqa: E402
from aerosplat.vps.geometry import angles, backproject, pose_error, project, rotation  # noqa: E402
from aerosplat.vps.localize import REASONS, Localizer, PosePrior  # noqa: E402
from vps_scenes import PlaneRenderer  # noqa: E402

W, H = 960, 540
F = 1379.3 * W / 1920            # IMX290 4 mm at 960 px
K = (F, F, W / 2 - 0.5, H / 2 - 0.5)


def tiny_atlas(n_kf: int = 4, spread: float = 50.0) -> Atlas:
    rng = np.random.default_rng(0)
    kfs = []
    for i in range(n_kf):
        C = np.array([rng.uniform(-spread, spread), rng.uniform(-spread, spread), 40.0])
        R = rotation(90.0 * i, -45.0)
        kfs.append({"R": R, "C": C, "look": C + 50 * R[2], "global": np.zeros(576, np.float32),
                    "xy": rng.uniform(0, 500, (5, 2)), "desc": rng.integers(0, 255, (5, 32)).astype(np.uint8),
                    "xyz_enu": rng.uniform(-50, 50, (5, 3))})
    return concat_keyframes(kfs, np.zeros(3), {
        "intrinsics": {"width": W, "height": H, "fx": F, "fy": F, "cx": K[2], "cy": K[3]},
        "extractor": {"name": "orb", "params": {"n_max": 500}}, "ground_z": 0.0})


@pytest.fixture(scope="module")
def loc():
    return Localizer(tiny_atlas())


def scene(n=300, seed=1, C=(5.0, -3.0, 40.0), yaw=30.0, pitch=-45.0, roll=2.0):
    """3D points on uneven ground seen by a camera; returns (uv, X, R, C) of the visible ones."""
    rng = np.random.default_rng(seed)
    R = rotation(yaw, pitch, roll)
    C = np.asarray(C, float)
    uv = np.stack([rng.uniform(5, W - 5, 4 * n), rng.uniform(5, H - 5, 4 * n)], axis=1)
    # intersect pixel rays with z = 0, then lift by a few metres (buildings): non-planar points
    d = np.stack([(uv[:, 0] - K[2]) / K[0], (uv[:, 1] - K[3]) / K[1], np.ones(len(uv))], 1) @ R
    t = -C[2] / d[:, 2]
    ok = (t > 0) & (t < 300)
    X = C + t[ok, None] * d[ok]
    X[:, 2] += rng.uniform(0, 15, len(X))
    uv2, z = project(R, C, K, X)
    vis = (z > 0) & (uv2[:, 0] > 0) & (uv2[:, 0] < W) & (uv2[:, 1] > 0) & (uv2[:, 1] < H)
    return uv2[vis][:n], X[vis][:n], R, C


# ── localizer geometry ──────────────────────────────────────────────────
def test_exact_correspondences_recover_the_pose(loc):
    uv, X, R, C = scene()
    fix = loc.solve(uv, X, K, (W, H))
    assert fix.ok, fix.detail
    err = pose_error(fix.R, fix.C, R, C)
    assert err["pos_m"] < 1e-3 and err["rot_deg"] < 1e-3
    assert angles(fix.R) == pytest.approx(angles(R), abs=1e-3)
    # σ: residual floor (0.7 px) + the twin's own 0.3 m; positive definite, not absurd
    assert 0.3 <= fix.sigma_h_m < 1.0 and 0 < fix.sigma_yaw_deg < 1.0
    assert np.all(np.linalg.eigvalsh(fix.cov) > 0)


def test_noise_and_outliers_still_give_the_pose(loc):
    rng = np.random.default_rng(3)
    uv, X, R, C = scene(seed=3)
    uv = uv + rng.normal(0, 0.7, uv.shape)
    bad = rng.choice(len(uv), len(uv) // 4, replace=False)
    uv[bad] = rng.uniform(0, [W, H], (len(bad), 2))
    fix = loc.solve(uv, X, K, (W, H))
    assert fix.ok, fix.detail
    err = pose_error(fix.R, fix.C, R, C)
    assert err["horiz_m"] < 0.3 and err["yaw_deg"] < 0.2
    assert err["horiz_m"] < 2 * fix.sigma_h_m


def test_too_few_matches_is_refused(loc):
    uv, X, _, _ = scene(n=20)
    fix = loc.solve(uv, X, K, (W, H))
    assert not fix.ok and fix.reason == "too_few_matches" and fix.R is None


def test_points_in_one_corner_are_poor_geometry(loc):
    uv, X, R, C = scene(n=2000, seed=4)
    corner = (uv[:, 0] < W / 4) & (uv[:, 1] < H / 4)
    fix = loc.solve(uv[corner], X[corner], K, (W, H))
    assert corner.sum() >= 40
    assert not fix.ok and fix.reason == "poor_geometry"


def test_two_poses_supported_equally_are_ambiguous(loc):
    # The same image points explained by two cameras 10 m apart (a repeated facade): half of the
    # matches belong to each. Neither may be returned.
    uv_a, X_a, _, _ = scene(n=200, seed=5, C=(0.0, 0.0, 40.0))
    uv_b, X_b, _, _ = scene(n=200, seed=6, C=(10.0, 0.0, 40.0))
    fix = loc.solve(np.vstack([uv_a, uv_b]), np.vstack([X_a, X_b]), K, (W, H))
    assert not fix.ok and fix.reason == "ambiguous", (fix.reason, fix.detail)


def test_pose_far_from_the_prior_is_refused(loc):
    uv, X, R, C = scene()
    yaw = angles(R)[0]
    far = PosePrior(C=(C[0] + 30, C[1], C[2]), yaw_deg=yaw, pitch_deg=-45, sigma_pos_m=3.0)
    fix = loc.solve(uv, X, K, (W, H), far)
    assert not fix.ok and fix.reason == "prior_mismatch"
    assert fix.candidate is not None                 # the rejected pose is kept for diagnosis
    turned = PosePrior(C=tuple(C), yaw_deg=yaw + 60, pitch_deg=-45, sigma_yaw_deg=5.0)
    assert loc.solve(uv, X, K, (W, H), turned).reason == "prior_mismatch"
    near = PosePrior(C=(C[0] + 2, C[1] - 1, C[2]), yaw_deg=yaw + 3, pitch_deg=-45)
    assert loc.solve(uv, X, K, (W, H), near).ok


def test_garbage_correspondences_are_refused_never_a_pose(loc):
    rng = np.random.default_rng(7)
    for seed in range(5):
        uv, X, _, _ = scene(seed=10 + seed)
        fix = loc.solve(rng.uniform(0, [W, H], uv.shape), X, K, (W, H))
        assert not fix.ok and fix.reason in REASONS


def test_noisy_correspondences_are_refused(loc):
    rng = np.random.default_rng(8)
    uv, X, _, _ = scene(seed=8)
    fix = loc.solve(uv + rng.normal(0, 6.0, uv.shape), X, K, (W, H))
    assert not fix.ok


def test_bad_input_is_refused(loc):
    img = np.zeros((H, W), np.float32)
    assert loc.localize(img, K).reason == "bad_input"
    assert loc.localize(np.zeros((H, W), np.uint8), (0, F, 1, 1)).reason == "bad_input"
    assert loc.localize(np.zeros((H, W), np.uint8), (F, F, np.nan, 1)).reason == "bad_input"
    # a blank frame has no features
    assert loc.localize(np.full((H, W), 128, np.uint8), K).reason == "no_features"


# ── atlas file ──────────────────────────────────────────────────────────
def test_atlas_round_trip_and_hash(tmp_path):
    a = tiny_atlas()
    p = tmp_path / "atlas.npz"
    digest = a.save(p)
    b = Atlas.load(p, expect_hash=digest)
    for name in ("kf_R", "kf_C", "xy", "desc", "xyz", "kf_offsets"):
        assert np.array_equal(getattr(a, name), getattr(b, name))
    assert b.meta["content_hash"] == digest == b.content_hash()
    with pytest.raises(AtlasError, match="not the expected"):
        Atlas.load(p, expect_hash="0" * 64)


def test_tampered_atlas_is_refused(tmp_path):
    a = tiny_atlas()
    p = tmp_path / "atlas.npz"
    a.save(p)
    with np.load(p) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["xyz"] = arrays["xyz"] + np.float32(0.5)          # every point moved by 0.5 m
    np.savez(tmp_path / "bad.npz", **arrays)
    with pytest.raises(AtlasError, match="content hash"):
        Atlas.load(tmp_path / "bad.npz")
    (tmp_path / "junk.npz").write_bytes(b"not a zip")
    with pytest.raises(AtlasError):
        Atlas.load(tmp_path / "junk.npz")


def test_inconsistent_atlas_is_refused():
    a = tiny_atlas()
    a.kf_offsets = a.kf_offsets.copy()
    a.kf_offsets[-1] += 1
    with pytest.raises(AtlasError, match="partition"):
        a.validate()
    b = tiny_atlas()
    b.meta = {**b.meta, "extractor": {}}
    with pytest.raises(AtlasError, match="extractor"):
        b.validate()


# ── builder and end to end on a synthetic plane ─────────────────────────
def test_keypoint_depth_drops_silhouettes_and_holes():
    depth = np.full((100, 100), 50.0, np.float32)
    depth[:, 60:] = 80.0                     # a roof edge
    depth[10:20, 10:20] = 0.0                # no data
    xy = np.array([[30.0, 50.0], [60.0, 50.0], [15.0, 15.0], [90.0, 50.0], [1.0, 1.0]])
    d = keypoint_depths(depth, xy, DepthFilter())
    assert d[0] == 50.0 and d[3] == 80.0
    assert np.isnan(d[1]) and np.isnan(d[2]) and np.isnan(d[4])


def test_backproject_inverts_project():
    R, C = rotation(123.0, -40.0, 3.0), np.array([1.0, 2.0, 30.0])
    X = np.array([[10.0, 40.0, 0.0], [-20.0, 25.0, 5.0]])
    uv, z = project(R, C, K, X)
    assert np.allclose(backproject(R, C, K, uv, z), X)


def test_bucketing_spreads_features():
    rng = np.random.default_rng(0)
    xy = np.vstack([rng.uniform(0, 50, (900, 2)), rng.uniform(0, 1000, (100, 2))]).astype(np.float32)
    f = Features(xy, np.zeros((1000, 32), np.uint8), np.r_[np.full(900, 10.0), np.ones(100)].astype(np.float32))
    b = bucket(f, 1000, 1000, 200, (4, 4))
    assert len(b) == 200
    assert (b.xy[:, 0] > 250).sum() + (b.xy[:, 1] > 250).sum() > 50   # not all from the hot corner


@pytest.fixture(scope="module")
def plane_atlas():
    w, h = 640, 360
    f = 1379.3 * w / 1920
    k = (f, f, w / 2 - 0.5, h / 2 - 0.5)
    r = PlaneRenderer(seed=0)
    views = grid_views((0.0, 0.0), 40.0, 20.0, [30.0, 50.0], [0.0, 90.0, 180.0, 270.0], -45.0,
                       lambda e, n: 0.0)
    atlas, st = build_atlas(r, views, w, h, k, make_extractor("orb", n_max=800), {"run": "synthetic"})
    return atlas, st, r, k, (w, h)


def test_built_atlas_points_lie_on_the_plane(plane_atlas):
    atlas, st, *_ = plane_atlas
    assert st["keyframes"] == atlas.n_keyframes == 200
    z = atlas.points_enu(slice(None))[:, 2]
    assert np.percentile(np.abs(z), 99) < 0.05
    assert atlas.meta["intrinsics"]["convention"].startswith("opencv")


def test_novel_views_localize_on_the_synthetic_plane(plane_atlas, tmp_path):
    atlas, _, r, k, (w, h) = plane_atlas
    atlas.save(tmp_path / "a.npz")
    loc = Localizer(Atlas.load(tmp_path / "a.npz"))
    rng = np.random.default_rng(1)
    for _ in range(4):
        v = View((rng.uniform(-30, 30), rng.uniform(-30, 30), rng.uniform(25, 55)),
                 rng.uniform(0, 360), -45 + rng.uniform(-5, 5), rng.uniform(-5, 5))
        img, _ = r.render(v, w, h, k)
        prior = PosePrior(C=(v.C[0] + 2, v.C[1] - 2, v.C[2] + 1), yaw_deg=v.yaw_deg + 4,
                          pitch_deg=v.pitch_deg, roll_deg=v.roll_deg, sigma_pos_m=3.0)
        fix = loc.localize(perturb.apply(img, perturb.draw("mild", rng)), k, prior)
        assert fix.ok, (fix.reason, fix.detail)
        err = pose_error(fix.R, fix.C, v.R, np.asarray(v.C))
        assert err["horiz_m"] < 0.3 and err["yaw_deg"] < 0.3
        assert set(fix.timings_ms) >= {"extract", "retrieve", "match", "pnp", "verify", "total"}


def test_a_repetitive_scene_gives_no_false_fix():
    """A texture that repeats every 12 m: every place looks like another. Refuse, never alias."""
    w, h = 640, 360
    f = 1379.3 * w / 1920
    k = (f, f, w / 2 - 0.5, h / 2 - 0.5)
    r = PlaneRenderer(seed=0, period_m=12.0)
    views = grid_views((0.0, 0.0), 40.0, 20.0, [30.0, 50.0], [0.0, 90.0, 180.0, 270.0], -45.0,
                       lambda e, n: 0.0)
    atlas, _ = build_atlas(r, views, w, h, k, make_extractor("orb", n_max=800), {"run": "periodic"})
    loc = Localizer(atlas)
    rng = np.random.default_rng(1)
    for _ in range(6):
        v = View((rng.uniform(-30, 30), rng.uniform(-30, 30), rng.uniform(25, 55)),
                 rng.uniform(0, 360), -45 + rng.uniform(-5, 5), rng.uniform(-5, 5))
        img, _ = r.render(v, w, h, k)
        fix = loc.localize(img, k, PosePrior(C=(v.C[0] + 2, v.C[1] - 2, v.C[2]), yaw_deg=v.yaw_deg + 3,
                                             pitch_deg=v.pitch_deg, roll_deg=v.roll_deg))
        if fix.ok:
            assert pose_error(fix.R, fix.C, v.R, np.asarray(v.C))["pos_m"] <= 2.0


def test_a_prior_outside_the_atlas_is_refused(plane_atlas):
    atlas, _, r, k, (w, h) = plane_atlas
    loc = Localizer(atlas)
    v = View((150.0, 150.0, 40.0), 45.0, -45.0)        # textured, 140 m from any keyframe view
    img, _ = r.render(v, w, h, k)
    fix = loc.localize(img, k, PosePrior(C=v.C, yaw_deg=45.0, pitch_deg=-45.0))
    assert not fix.ok and fix.reason == "outside_atlas"


def test_without_a_prior_the_answer_is_right_or_a_refusal(plane_atlas):
    atlas, _, r, k, (w, h) = plane_atlas
    loc = Localizer(atlas)
    rng = np.random.default_rng(2)
    for _ in range(4):
        v = View((rng.uniform(-30, 30), rng.uniform(-30, 30), rng.uniform(25, 55)),
                 rng.uniform(0, 360), -45.0)
        img, _ = r.render(v, w, h, k)
        fix = loc.localize(img, k, None)
        assert fix.stats.get("retrieval") in (None, "global")
        if fix.ok:
            assert pose_error(fix.R, fix.C, v.R, np.asarray(v.C))["pos_m"] <= 2.0


# ── perturbations and the companion's imports ───────────────────────────
def test_perturbations_are_reproducible_and_clean_is_identity():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (60, 80, 3)).astype(np.uint8)
    assert perturb.apply(img, perturb.draw("clean", rng)) is img
    p = perturb.draw("strong", rng)
    a, b = perturb.apply(img, p), perturb.apply(img, p)
    assert np.array_equal(a, b) and not np.array_equal(a, img) and a.dtype == np.uint8
    with pytest.raises(ValueError):
        perturb.draw("night", rng)


def test_unknown_extractor_is_refused():
    with pytest.raises(ValueError, match="unknown extractor"):
        make_extractor("superpoint")
    assert make_extractor("akaze", n_max=10).name == "akaze"


def test_companion_side_imports_neither_torch_nor_gsplat():
    code = ("import sys, aerosplat.vps.localize, aerosplat.vps.bench, aerosplat.vps.perturb; "
            "bad = [m for m in ('torch', 'gsplat', 'pydantic', 'aerosplat.render.engine') "
            "if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
