"""Appearance perturbations: make a rendered twin frame look a little more like a camera frame.

Photometric (gain, contrast, gamma), optics (Gaussian blur), sensor (Gaussian read noise) and
the codec (JPEG). A *rough* stand-in for the domain gap: it does not model season, shadows that
moved, parked cars, construction, a different time of day, lens distortion residuals or motion
blur from vibration. The bench says so wherever it prints these numbers. `rolling_shutter` adds
the skew of a turning rolling-shutter camera (small-angle, rotation only; SIM-3).

Parameters are drawn once per query (`draw`) and stored with the query set, so the companion
re-creates exactly the same frames from the clean images (`apply`). `severe` is not a camera in
daylight: dusk, defocus, heavy noise — a level meant to make VPS refuse (SIM-3 fault
`vps_degrade`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aerosplat.vps.features import require_cv2

LEVELS: dict[str, dict[str, tuple[float, float]]] = {
    # name: parameter -> (low, high), drawn uniformly
    "clean": {},
    "mild": {"gain": (0.85, 1.15), "contrast": (0.85, 1.15), "gamma": (0.9, 1.1),
             "blur_sigma": (0.0, 0.8), "noise_sigma": (1.0, 3.0), "jpeg_quality": (80, 95)},
    "strong": {"gain": (0.6, 1.4), "contrast": (0.6, 1.3), "gamma": (0.75, 1.3),
               "blur_sigma": (0.6, 1.6), "noise_sigma": (3.0, 7.0), "jpeg_quality": (50, 75)},
    "severe": {"gain": (0.15, 0.3), "contrast": (0.4, 0.6), "gamma": (1.3, 1.6),
               "blur_sigma": (3.0, 4.5), "noise_sigma": (10.0, 16.0), "jpeg_quality": (25, 40)},
}


def draw(level: str, rng: np.random.Generator) -> dict[str, float]:
    if level not in LEVELS:
        raise ValueError(f"unknown perturbation level '{level}' (known: {', '.join(LEVELS)})")
    out: dict[str, float] = {}
    for k, (lo, hi) in LEVELS[level].items():
        out[k] = float(rng.uniform(lo, hi))
    if out:
        out["seed"] = int(rng.integers(0, 2 ** 31 - 1))
    return out


def apply(img: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    """Perturbed copy of an 8-bit image (grey or BGR); `p` from `draw` (empty = unchanged)."""
    if not p:
        return img
    cv2 = require_cv2()
    x = img.astype(np.float32) / 255.0
    mean = float(x.mean())
    x = (x - mean) * p.get("contrast", 1.0) + mean
    x = np.clip(x * p.get("gain", 1.0), 0.0, 1.0) ** p.get("gamma", 1.0)
    x = x * 255.0
    s = p.get("blur_sigma", 0.0)
    if s > 0.05:
        x = cv2.GaussianBlur(x, (0, 0), s)
    n = p.get("noise_sigma", 0.0)
    if n > 0:
        rng = np.random.default_rng(int(p.get("seed", 0)))
        x = x + rng.normal(0.0, n, x.shape).astype(np.float32)
    out = np.clip(np.rint(x), 0, 255).astype(np.uint8)
    q = p.get("jpeg_quality")
    if q is not None:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(round(q))])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        out = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    return out


def rolling_shutter(img: np.ndarray, omega_cam: tuple[float, float, float], readout_s: float,
                    K: tuple[float, float, float, float]) -> np.ndarray:
    """A frame as a rolling shutter records it while the camera turns at `omega_cam` (rad/s,
    camera axes: x right, y down, z optical), from the frame of a global shutter at the middle
    row's time. Row v is read `readout_s * (v / (h - 1) - 1/2)` after the middle row; small-angle
    model (pure rotation, no translation, no motion blur): a turn δ moves an image point by
    du = fx (δz·y - δy), dv = fy (δx - δz·x) with x, y its normalised coordinates."""
    if readout_s <= 0 or not any(abs(w) > 0 for w in omega_cam):
        return img
    cv2 = require_cv2()
    h, w = img.shape[:2]
    fx, fy, cx, cy = K
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    t = (readout_s * (v / max(h - 1, 1) - 0.5)).astype(np.float32)
    dx, dy, dz = (float(o) * t for o in omega_cam)
    xn, yn = (u - cx) / fx, (v - cy) / fy
    du = fx * (dz * yn - dy)
    dv = fy * (dx - dz * xn)
    return cv2.remap(img, (u - du).astype(np.float32), (v - dv).astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)
