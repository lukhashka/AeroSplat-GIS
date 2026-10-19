"""VPS localizer (ADR-0015 layer A): camera frame + atlas -> pose in Site ENU, or a refusal.

Runs on the companion: numpy + OpenCV, CPU only, no torch / gsplat (a test imports this module
in a fresh interpreter and checks).

    loc = Localizer(Atlas.load(path, expect_hash=package_hash))
    fix = loc.localize(image, K, prior=PosePrior(C=..., yaw_deg=..., pitch_deg=..., ...))
    if fix.ok: send ODOMETRY(fix.C, fix.R, fix.cov) else: no fix (reason in fix.reason)

Steps: features of the frame -> candidate keyframes (nearest to the prior's view, or the
global descriptor without a prior) -> descriptor matches against their 3D points (ratio test
against the nearest *different* 3D point) -> PnP-RANSAC + LM refinement -> checks.

**Fail-safe, not fail-silent**: every check that fails is a
refusal with a reason, never a pose:

    no_features       too few keypoints in the frame (blank, dark, blurred)
    outside_atlas     the prior looks at a place the atlas does not cover
    too_few_matches   too few descriptor matches
    pnp_failed        RANSAC found no pose
    too_few_inliers   / low_inlier_ratio: the pose rests on too little evidence
    poor_geometry     inliers bunched in a corner of the frame, or a singular information matrix
    high_residual     inlier reprojection RMS too high for a correct pose
    implausible_pose  points behind the camera, camera under the scene, or far outside the atlas
    sigma_too_large   the estimated σ exceeds what the fix is allowed to claim
    ambiguous         a second, different pose is supported by a large share of the outliers
    prior_mismatch    the pose disagrees with the prior beyond its own σ (HOLD, ADR-0015 §2.5)

Covariance: the Gauss-Newton information of the inlier reprojections (numerical Jacobian over a
world-frame rotation and the camera centre), scaled by the residual (with a floor) and
`sigma_scale`, plus `sigma_map_m`² of the twin itself on the position. The bench reports how well
this σ predicts the actual error; on rendered data it is a statement about the method only.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

from aerosplat.vps.atlas import Atlas, AtlasError
from aerosplat.vps.features import Extractor, global_descriptor, make_extractor, require_cv2
from aerosplat.vps.geometry import (Intrinsics, K_matrix, angles, look_point, project, rotation,
                                    rotation_angle_deg, rotvec_to_R, wrap_deg)

REASONS = ("bad_input", "no_features", "outside_atlas", "too_few_matches", "pnp_failed",
           "too_few_inliers", "low_inlier_ratio", "poor_geometry", "high_residual",
           "implausible_pose", "sigma_too_large", "ambiguous", "prior_mismatch")


@dataclass(frozen=True)
class PosePrior:
    """What the vehicle believes before the fix: position (EKF), heading (EKF / compass), and the
    camera's elevation and roll (AHRS + mount, known well without GNSS)."""

    C: tuple[float, float, float]
    yaw_deg: float
    pitch_deg: float
    roll_deg: float = 0.0
    sigma_pos_m: float = 5.0
    sigma_yaw_deg: float = 10.0

    @property
    def R(self) -> np.ndarray:
        return rotation(self.yaw_deg, self.pitch_deg, self.roll_deg)


@dataclass(frozen=True)
class LocalizerConfig:
    n_candidates: int = 6
    max_look_dist_m: float = 60.0     # + 3σ of the prior: nearest keyframe view farther = outside
    ratio: float = 0.8
    distinct_m: float = 1.0
    max_hamming_frac: float = 0.30    # of the descriptor bits
    min_matches: int = 30
    ransac_px: float = 4.0            # query pixels
    ransac_iters: int = 2000
    ransac_confidence: float = 0.9999
    min_inliers: int = 40
    min_inlier_ratio: float = 0.30
    grid: tuple[int, int] = (4, 4)
    min_cells: int = 5
    max_rms_px: float = 2.5
    sigma_px_floor: float = 0.7
    sigma_scale: float = 2.0
    sigma_map_m: float = 0.3
    max_sigma_h_m: float = 1.5
    max_sigma_yaw_deg: float = 3.0
    ambiguity_ratio: float = 0.5
    ambiguity_pos_m: float = 3.0
    ambiguity_rot_deg: float = 5.0
    prior_gate_k: float = 3.0
    prior_gate_floor_m: float = 2.0
    prior_gate_floor_deg: float = 5.0
    atlas_margin_m: float = 60.0


@dataclass
class Fix:
    ok: bool
    reason: Optional[str] = None
    detail: str = ""
    R: Optional[np.ndarray] = None            # world -> camera
    C: Optional[np.ndarray] = None            # camera centre, Site ENU
    cov: Optional[np.ndarray] = None          # 6x6: centre (e, n, u) then world rotation
    sigma_h_m: Optional[float] = None         # sqrt(σe² + σn²)
    sigma_z_m: Optional[float] = None
    sigma_yaw_deg: Optional[float] = None
    stats: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    candidate: Optional[dict[str, Any]] = None   # the pose a refusal rejected (for diagnosis)

    @property
    def angles(self) -> Optional[tuple[float, float, float]]:
        return None if self.R is None else angles(self.R)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("R", "C", "cov"):
            v = getattr(self, k)
            d[k] = None if v is None else np.asarray(v).tolist()
        if self.R is not None:
            d["yaw_deg"], d["pitch_deg"], d["roll_deg"] = angles(self.R)
        return d


def _refuse(reason: str, detail: str, stats: dict[str, Any], timings: dict[str, float],
            candidate: Optional[dict[str, Any]] = None) -> Fix:
    assert reason in REASONS
    return Fix(ok=False, reason=reason, detail=detail, stats=stats, timings_ms=timings,
               candidate=candidate)


class Localizer:
    def __init__(self, atlas: Atlas, config: Optional[LocalizerConfig] = None,
                 extractor: Optional[Extractor] = None):
        self.cv2 = require_cv2()
        atlas.validate()
        self.atlas = atlas
        self.cfg = config or LocalizerConfig()
        ex = atlas.meta["extractor"]
        self.extractor = extractor or make_extractor(ex["name"], **ex["params"])
        if self.extractor.name != ex["name"]:
            raise AtlasError(f"atlas built with {ex['name']}, localizer uses {self.extractor.name}")
        if atlas.desc.shape[1] == 0:
            raise AtlasError("atlas has empty descriptors")
        self.bits = atlas.desc.shape[1] * 8
        self.ground_z = float(atlas.meta.get("ground_z", np.median(atlas.points_enu(slice(None))[:, 2])))
        self.kf_lo = atlas.kf_C.min(0)
        self.kf_hi = atlas.kf_C.max(0)
        gl = atlas.kf_global.astype(np.float32)
        self.kf_global = gl
        self.matcher = self.cv2.BFMatcher(self.cv2.NORM_HAMMING, crossCheck=False)

    # ── retrieval ────────────────────────────────────────────────────────
    def candidates(self, gray: np.ndarray, prior: Optional[PosePrior]
                   ) -> tuple[Optional[np.ndarray], str]:
        a, cfg = self.atlas, self.cfg
        if prior is not None:
            R = prior.R
            C = np.asarray(prior.C, float)
            look = look_point(R, C, self.ground_z)
            d_look = np.linalg.norm((a.kf_look - look)[:, :2], axis=1)
            d_cam = np.linalg.norm(a.kf_C - C, axis=1)
            fwd = a.kf_R[:, 2, :2]
            fq = R[2, :2]
            # heading disagreement matters as much as 20 m of look point
            cosang = (fwd @ fq) / np.maximum(np.linalg.norm(fwd, axis=1) * np.linalg.norm(fq), 1e-9)
            score = d_look + 0.5 * d_cam + 20.0 * (1.0 - cosang)
            order = np.argsort(score, kind="stable")
            limit = cfg.max_look_dist_m + 3.0 * prior.sigma_pos_m
            if d_look[order[0]] > limit:
                return None, (f"nearest keyframe view {d_look[order[0]]:.0f} m from the prior's "
                              f"(limit {limit:.0f} m)")
            return order[:cfg.n_candidates], "prior"
        g = global_descriptor(gray)
        if g.shape[0] != self.kf_global.shape[1]:
            return None, "global descriptor size differs from the atlas"
        sim = self.kf_global @ g
        return np.argsort(-sim, kind="stable")[:cfg.n_candidates], "global"

    # ── matching ─────────────────────────────────────────────────────────
    def match(self, desc: np.ndarray, kfs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Query feature index -> atlas feature index for accepted matches."""
        a, cfg = self.atlas, self.cfg
        rows = np.concatenate([np.arange(a.kf_offsets[k], a.kf_offsets[k + 1]) for k in kfs])
        if rows.size == 0 or desc.shape[0] == 0:
            return np.zeros(0, int), np.zeros(0, int)
        knn = self.matcher.knnMatch(desc, a.desc[rows], k=4)
        xyz = a.xyz[rows]
        max_d = cfg.max_hamming_frac * self.bits
        q_idx, a_idx, dist = [], [], []
        for ms in knn:
            if not ms:
                continue
            m0 = ms[0]
            if m0.distance > max_d:
                continue
            p0 = xyz[m0.trainIdx]
            second = None
            for m in ms[1:]:
                if np.linalg.norm(xyz[m.trainIdx] - p0) > cfg.distinct_m:
                    second = m
                    break
            if second is not None and m0.distance >= cfg.ratio * second.distance:
                continue
            q_idx.append(m0.queryIdx)
            a_idx.append(int(rows[m0.trainIdx]))
            dist.append(m0.distance)
        if not q_idx:
            return np.zeros(0, int), np.zeros(0, int)
        q, at, d = np.array(q_idx), np.array(a_idx), np.array(dist)
        # one query keypoint per atlas point (the closest descriptor)
        order = np.lexsort((d, at))
        keep = order[np.concatenate([[True], at[order][1:] != at[order][:-1]])]
        return q[keep], at[keep]

    # ── pose ─────────────────────────────────────────────────────────────
    def _ransac(self, obj: np.ndarray, img: np.ndarray, Km: np.ndarray, thr: float
                ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        cv2, cfg = self.cv2, self.cfg
        if len(obj) < 4:
            return None
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj, img, Km, None, iterationsCount=cfg.ransac_iters, reprojectionError=thr,
            confidence=cfg.ransac_confidence, flags=cv2.SOLVEPNP_AP3P)
        if not ok or inl is None or len(inl) < 4:
            return None
        inl = inl.ravel()
        for _ in range(2):
            rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], Km, None, rvec, tvec)
            proj, _ = cv2.projectPoints(obj, rvec, tvec, Km, None)
            res = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
            new = np.flatnonzero(res < thr)
            if len(new) < 4:
                return None
            inl = new
        return rvec, tvec, inl

    def _covariance(self, R: np.ndarray, C: np.ndarray, K: Intrinsics, X: np.ndarray,
                    uv: np.ndarray, sigma_floor_px: float) -> tuple[np.ndarray, float, float]:
        """6x6 covariance of (C, world rotation), residual σ (px), and σ of the heading (deg)."""
        def resid(p: np.ndarray) -> np.ndarray:
            Rp = R @ rotvec_to_R(p[3:])
            u, _ = project(Rp, C + p[:3], K, X)
            return (u - uv).ravel()

        def yaw_of(p: np.ndarray) -> float:
            return angles(R @ rotvec_to_R(p[3:]))[0]

        steps = np.array([1e-3, 1e-3, 1e-3, 1e-5, 1e-5, 1e-5])
        r0 = resid(np.zeros(6))
        J = np.empty((r0.size, 6))
        g = np.zeros(6)
        for i in range(6):
            dp = np.zeros(6)
            dp[i] = steps[i]
            J[:, i] = (resid(dp) - resid(-dp)) / (2 * steps[i])
            if i >= 3:
                g[i] = wrap_deg(yaw_of(dp) - yaw_of(-dp)) / (2 * steps[i])
        dof = max(r0.size - 6, 1)
        s2 = max(float(r0 @ r0) / dof, sigma_floor_px ** 2)
        info = J.T @ J
        if not np.all(np.isfinite(info)) or np.linalg.cond(info) > 1e14:
            raise np.linalg.LinAlgError("singular information matrix")
        cov = s2 * np.linalg.inv(info) * self.cfg.sigma_scale ** 2
        cov[:3, :3] += np.eye(3) * self.cfg.sigma_map_m ** 2
        depth = float(np.median(np.linalg.norm(X - C, axis=1)))
        yaw_var = float(g @ cov @ g) + math.degrees(self.cfg.sigma_map_m / max(depth, 1.0)) ** 2
        return cov, math.sqrt(s2), math.sqrt(max(yaw_var, 0.0))

    def localize(self, image: np.ndarray, K: Intrinsics, prior: Optional[PosePrior] = None) -> Fix:
        cv2, cfg, a = self.cv2, self.cfg, self.atlas
        tm: dict[str, float] = {}
        stats: dict[str, Any] = {}
        t_all = time.perf_counter()

        def lap(name: str, t0: float) -> float:
            t = time.perf_counter()
            tm[name] = round((t - t0) * 1000.0, 3)
            return t

        def done(f: Fix) -> Fix:
            tm["total"] = round((time.perf_counter() - t_all) * 1000.0, 3)
            return f

        if image is None or image.ndim not in (2, 3) or image.dtype != np.uint8 or image.size == 0:
            return done(_refuse("bad_input", "expected an 8-bit grey or BGR image", stats, tm))
        if len(K) != 4 or not all(math.isfinite(v) for v in K) or K[0] <= 0 or K[1] <= 0:
            return done(_refuse("bad_input", f"bad intrinsics {K}", stats, tm))
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        t = time.perf_counter()
        feats = self.extractor.detect(gray)
        t = lap("extract", t)
        stats["features"] = len(feats)
        if len(feats) < cfg.min_matches:
            return done(_refuse("no_features", f"{len(feats)} keypoints", stats, tm))

        kfs, how = self.candidates(gray, prior)
        t = lap("retrieve", t)
        if kfs is None:
            return done(_refuse("outside_atlas", how, stats, tm))
        stats["retrieval"] = how
        stats["keyframes"] = [int(k) for k in kfs]

        qi, ai = self.match(feats.desc, kfs)
        t = lap("match", t)
        stats["matches"] = int(len(qi))
        if len(qi) < cfg.min_matches:
            return done(_refuse("too_few_matches", f"{len(qi)} < {cfg.min_matches}", stats, tm))

        return self.solve(feats.xy[qi], a.points_enu(ai), K, (w, h), prior, stats=stats,
                          timings=tm, t_start=t_all)

    def solve(self, uv: np.ndarray, X: np.ndarray, K: Intrinsics, size: tuple[int, int],
              prior: Optional[PosePrior] = None, *, stats: Optional[dict[str, Any]] = None,
              timings: Optional[dict[str, float]] = None, t_start: Optional[float] = None) -> Fix:
        """Pose from 2D-3D matches (uv in OpenCV pixels of an image of `size` = (w, h), X in
        Site ENU) with every check of the module docstring. `localize` ends here; tests call
        it directly with exact correspondences."""
        cv2, cfg = self.cv2, self.cfg
        tm = {} if timings is None else timings
        stats = {} if stats is None else stats
        t_all = time.perf_counter() if t_start is None else t_start

        def lap(name: str, t0: float) -> float:
            t = time.perf_counter()
            tm[name] = round((t - t0) * 1000.0, 3)
            return t

        def done(f: Fix) -> Fix:
            tm["total"] = round((time.perf_counter() - t_all) * 1000.0, 3)
            return f

        w, h = size
        X = np.asarray(X, np.float64).reshape(-1, 3)
        img = np.asarray(uv, np.float64).reshape(-1, 2)
        qi = np.arange(len(X))
        stats.setdefault("matches", int(len(X)))
        if len(X) < cfg.min_matches:
            return done(_refuse("too_few_matches", f"{len(X)} < {cfg.min_matches}", stats, tm))
        t = time.perf_counter()
        centre = X.mean(0)
        obj = (X - centre).astype(np.float64)
        Km = K_matrix(K)
        thr = cfg.ransac_px
        sol = self._ransac(obj, img, Km, thr)
        t = lap("pnp", t)
        if sol is None:
            return done(_refuse("pnp_failed", "no RANSAC solution", stats, tm))
        rvec, tvec, inl = sol
        Rcv, _ = cv2.Rodrigues(rvec)
        R = Rcv
        C = -Rcv.T @ tvec.ravel() + centre
        n_in = len(inl)
        ratio = n_in / len(qi)
        stats.update(inliers=int(n_in), inlier_ratio=round(ratio, 4))
        cand = {"C": C.tolist(), "yaw_deg": angles(R)[0]}

        # -- checks ------------------------------------------------------------
        if n_in < cfg.min_inliers:
            return done(_refuse("too_few_inliers", f"{n_in} < {cfg.min_inliers}", stats, tm, cand))
        if ratio < cfg.min_inlier_ratio:
            return done(_refuse("low_inlier_ratio", f"{ratio:.2f} < {cfg.min_inlier_ratio}",
                                stats, tm, cand))
        gx, gy = cfg.grid
        cells = np.unique(np.clip((img[inl, 0] * gx / w).astype(int), 0, gx - 1)
                          + gx * np.clip((img[inl, 1] * gy / h).astype(int), 0, gy - 1))
        stats["cells"] = int(cells.size)
        if cells.size < cfg.min_cells:
            return done(_refuse("poor_geometry", f"inliers in {cells.size} of {gx * gy} cells "
                                                 f"(need {cfg.min_cells})", stats, tm, cand))
        Xin, uvin = X[inl], img[inl]
        uvp, z = project(R, C, K, Xin)
        res = np.linalg.norm(uvp - uvin, axis=1)
        rms = float(np.sqrt(np.mean(res ** 2)))
        stats["rms_px"] = round(rms, 3)
        if rms > cfg.max_rms_px:
            return done(_refuse("high_residual", f"RMS {rms:.2f} px", stats, tm, cand))
        if np.any(z <= 0):
            return done(_refuse("implausible_pose", f"{int(np.sum(z <= 0))} inliers behind the camera",
                                stats, tm, cand))
        lo, hi = self.kf_lo - cfg.atlas_margin_m, self.kf_hi + cfg.atlas_margin_m
        if np.any(C < lo) or np.any(C > hi):
            return done(_refuse("implausible_pose", "camera outside the atlas volume", stats, tm, cand))
        if C[2] <= float(Xin[:, 2].min()):
            return done(_refuse("implausible_pose", "camera below every point it sees",
                                stats, tm, cand))
        try:
            cov, sig_px, sig_yaw = self._covariance(R, C, K, Xin, uvin, cfg.sigma_px_floor)
        except np.linalg.LinAlgError as e:
            return done(_refuse("poor_geometry", str(e), stats, tm, cand))
        sig_h = math.sqrt(cov[0, 0] + cov[1, 1])
        sig_z = math.sqrt(cov[2, 2])
        stats.update(sigma_px=round(sig_px, 3))
        if sig_h > cfg.max_sigma_h_m or sig_yaw > cfg.max_sigma_yaw_deg:
            return done(_refuse("sigma_too_large", f"σ_h {sig_h:.2f} m, σ_yaw {sig_yaw:.2f}°",
                                stats, tm, cand))
        # ambiguity: does a different pose explain many of the remaining matches?
        rest = np.setdiff1d(np.arange(len(qi)), inl)
        if len(rest) >= cfg.min_inliers:
            X2 = X[rest]
            c2 = X2.mean(0)
            sol2 = self._ransac((X2 - c2), img[rest], Km, thr)
            if sol2 is not None and len(sol2[2]) >= max(cfg.min_inliers, cfg.ambiguity_ratio * n_in):
                R2, _ = cv2.Rodrigues(sol2[0])
                C2 = -R2.T @ sol2[1].ravel() + c2
                dpos = float(np.linalg.norm(C2 - C))
                drot = rotation_angle_deg(R2, R)
                stats["second_inliers"] = int(len(sol2[2]))
                if dpos > cfg.ambiguity_pos_m or drot > cfg.ambiguity_rot_deg:
                    t = lap("verify", t)
                    return done(_refuse("ambiguous", f"second pose {dpos:.1f} m / {drot:.1f}° away "
                                                     f"with {len(sol2[2])} inliers", stats, tm, cand))
        if prior is not None:
            d = float(np.hypot(*(C[:2] - np.asarray(prior.C[:2], float))))
            gate = cfg.prior_gate_k * math.hypot(prior.sigma_pos_m, sig_h) + cfg.prior_gate_floor_m
            dyaw = abs(wrap_deg(angles(R)[0] - prior.yaw_deg))
            gate_yaw = cfg.prior_gate_k * math.hypot(prior.sigma_yaw_deg, sig_yaw) + cfg.prior_gate_floor_deg
            stats.update(prior_dist_m=round(d, 2), prior_dyaw_deg=round(dyaw, 2))
            if d > gate or dyaw > gate_yaw:
                lap("verify", t)
                return done(_refuse("prior_mismatch", f"{d:.1f} m (gate {gate:.1f}) / {dyaw:.1f}° "
                                                      f"(gate {gate_yaw:.1f}) from the prior",
                                    stats, tm, cand))
        lap("verify", t)
        return done(Fix(ok=True, R=R, C=C, cov=cov, sigma_h_m=sig_h, sigma_z_m=sig_z,
                        sigma_yaw_deg=sig_yaw, stats=stats, timings_ms=tm))
