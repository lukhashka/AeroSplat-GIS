"""Synthetic scenes for the VPS tests: a textured ground plane z = 0 rendered by ray casting.

CPU only (numpy + OpenCV). The texture is multi-scale noise plus rectangles (corners for the
detectors); `period_m` tiles it, which makes every place look like every other (ambiguity).
"""

from __future__ import annotations

import numpy as np

from aerosplat.vps.build import View
from aerosplat.vps.geometry import Intrinsics

EXTENT_M = 400.0      # texture covers [-200, 200] m
TEX_PX = 4096


def make_texture(seed: int = 0, size: int = TEX_PX) -> np.ndarray:
    import cv2
    rng = np.random.default_rng(seed)
    tex = np.zeros((size, size), np.float32)
    for sigma, amp in ((40.0, 60.0), (12.0, 40.0), (4.0, 30.0), (1.5, 20.0)):
        n = rng.normal(0, 1, (size, size)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), sigma)
        tex += amp * n / (n.std() + 1e-6)
    for _ in range(3000):
        x, y = rng.integers(0, size - 40, 2)
        w, h = rng.integers(4, 40, 2)
        tex[y:y + h, x:x + w] += rng.uniform(-60, 60)
    tex = tex - tex.min()
    return np.clip(tex / tex.max() * 255.0, 0, 255).astype(np.uint8)


class PlaneRenderer:
    """Renderer protocol of aerosplat.vps.build over a textured plane."""

    def __init__(self, seed: int = 0, period_m: float | None = None):
        self.tex = make_texture(seed)
        self.period_m = period_m

    def render(self, view: View, width: int, height: int, K: Intrinsics):
        import cv2
        fx, fy, cx, cy = K
        R, C = view.R, np.asarray(view.C, float)
        u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
        d_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
        d_w = d_cam @ R                      # world direction for unit z-depth
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -C[2] / d_w[..., 2]          # z-depth of the hit
        hit = (t > 0) & np.isfinite(t) & (t < 1000)
        x = C[0] + t * d_w[..., 0]
        y = C[1] + t * d_w[..., 1]
        if self.period_m is not None:
            x = np.mod(x, self.period_m)
            y = np.mod(y, self.period_m)
        scale = TEX_PX / EXTENT_M
        mx = ((x + EXTENT_M / 2) * scale).astype(np.float32)
        my = ((EXTENT_M / 2 - y) * scale).astype(np.float32)
        g = cv2.remap(self.tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        g = np.where(hit, g, 0).astype(np.uint8)
        depth = np.where(hit, t, 0.0).astype(np.float32)
        return np.repeat(g[..., None], 3, axis=2), depth
