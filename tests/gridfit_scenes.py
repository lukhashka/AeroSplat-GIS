"""Synthetic facade orthophotos with a known answer, for `aerosplat.semantics.gridfit`.

A scene is painted in facade (u, v) metres and sampled on the same pixel grid the
facade orthophoto uses (`semantics.ortho`: columns along u, rows down the wall), so a
test can say where every window *is* and check the fit against it to the centimetre.

The colours are what the dataset_1910 renders look like: a beige wall, bluish glass a
little darker than the wall, dark green foliage, near-black shadows under balconies.
Every scene gets a soft blur and pixel noise, because a 3DGS ortho is never crisp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from aerosplat.semantics.gridfit import CoverageRaster, PixelGrid, box_mean

WALL = (205.0, 190.0, 165.0)
GLASS = (95.0, 120.0, 150.0)
LIGHT_GLASS = (160.0, 185.0, 210.0)      # glass reflecting the sky: bluer, barely darker
SLAB = (235.0, 228.0, 215.0)
SHADOW = (55.0, 55.0, 58.0)
FOLIAGE = (55.0, 85.0, 45.0)
NAVY = (30.0, 40.0, 75.0)                # a parked van, a dark tree in shade: not green

Rect = tuple[float, float, float, float]  # u0, u1, v0, v1


@dataclass
class Scene:
    rgb: np.ndarray                      # (H, W, 3) uint8
    alpha: np.ndarray                    # (H, W) float 0..1
    grid: PixelGrid
    bounds: tuple[float, float, float, float]   # u_min, u_max, v_min, v_max of the wall
    windows: list[Rect] = field(default_factory=list)   # where the openings really are
    coverage: Optional[CoverageRaster] = None
    front_m: Optional[np.ndarray] = None

    def kwargs(self) -> dict:
        return {"coverage": self.coverage, "front_m": self.front_m, "bounds_uv": self.bounds}


class Painter:
    """Paints rectangles and ellipses given in (u, v) metres onto the pixel grid."""

    def __init__(self, width_m: float, height_m: float, px: float, margin_m: float = 1.0):
        self.px = px
        self.width_m, self.height_m = width_m, height_m
        w = int(math.ceil(round((width_m + 2 * margin_m) / px, 6)))
        h = int(math.ceil(round((height_m + 2 * margin_m) / px, 6)))
        self.grid = PixelGrid(m_per_px=px, u00=-margin_m + px / 2, v00=height_m + margin_m - px / 2)
        self.u = self.grid.u_of(np.arange(w))[None, :]
        self.v = self.grid.v_of(np.arange(h))[:, None]
        self.rgb = np.zeros((h, w, 3))
        self.alpha = np.zeros((h, w))
        self.front = np.zeros((h, w))
        self.paint((0.0, width_m, 0.0, height_m), WALL)
        self.alpha[self.inside((0.0, width_m, 0.0, height_m))] = 1.0

    def inside(self, r: Rect) -> np.ndarray:
        u0, u1, v0, v1 = r
        return (self.u >= u0) & (self.u <= u1) & (self.v >= v0) & (self.v <= v1)

    def paint(self, r: Rect, colour, front_m: Optional[float] = None) -> None:
        m = self.inside(r)
        self.rgb[m] = colour
        if front_m is not None:
            self.front[m] = front_m

    def ellipse(self, cu: float, cv: float, ru: float, rv: float, colour,
                front_m: Optional[float] = None, speckle: Optional[np.random.Generator] = None) -> None:
        m = ((self.u - cu) / ru) ** 2 + ((self.v - cv) / rv) ** 2 <= 1.0
        self.rgb[m] = colour
        if speckle is not None:
            # Foliage has dark gaps between the leaves, which read as glass to a darkness cue.
            dark = m & (speckle.random(m.shape) < 0.25)
            self.rgb[dark] = (20.0, 30.0, 20.0)
        self.alpha[m] = 1.0
        if front_m is not None:
            self.front[m] = front_m

    def finish(self, *, noise: float = 4.0, seed: int = 0, shade: float = 0.12,
               blur_m: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
        """Soften, shade across the wall (sun), add noise; returns (rgb uint8, alpha)."""
        rgb = self.rgb.copy()
        k = max(1, int(round(blur_m / self.px)))
        if k > 1:
            rgb = np.stack([box_mean(rgb[..., c], k) for c in range(3)], axis=-1)
        rgb *= (1.0 - shade * (self.u / max(self.width_m, 1.0)))[..., None]
        rng = np.random.default_rng(seed)
        rgb += rng.normal(0.0, noise, rgb.shape)
        rgb[self.alpha < 0.5] = 0.0
        return np.clip(rgb, 0, 255).astype(np.uint8), self.alpha.copy()


def grid_windows(*, width_m: float, height_m: float, floor_m: float, bay_m: float,
                 win_w: float, win_h: float, sill_m: float = 0.9, first_u: float = 0.7,
                 pair_gap_m: Optional[float] = None) -> list[Rect]:
    """Every window of a regular facade: one per bay, or a pair when `pair_gap_m` is set."""
    out: list[Rect] = []
    k = 0
    while k * floor_m + sill_m + win_h <= height_m - 0.3:
        v0 = k * floor_m + sill_m
        m = 0
        while True:
            u0 = first_u + m * bay_m
            us = [u0] if pair_gap_m is None else [u0, u0 + win_w + pair_gap_m]
            if us[-1] + win_w > width_m - 0.2:
                break
            out.extend((a, a + win_w, v0, v0 + win_h) for a in us)
            m += 1
        k += 1
    return out


def facade(*, width_m: float = 30.0, height_m: float = 21.0, px: float = 0.08,
           floor_m: float = 3.0, bay_m: float = 2.5, win_w: float = 1.2, win_h: float = 1.5,
           sill_m: float = 0.9, first_u: float = 0.7, pair_gap_m: Optional[float] = None,
           glass=GLASS, skip: tuple[int, ...] = (), slabs: bool = True, noise: float = 4.0,
           seed: int = 0) -> tuple[Painter, list[Rect]]:
    """A regular facade, not yet finished: callers add trees, shadows and the like."""
    p = Painter(width_m, height_m, px)
    if slabs:
        k = 1
        while k * floor_m < height_m:
            p.paint((0.0, width_m, k * floor_m - 0.08, k * floor_m + 0.08), SLAB)
            k += 1
    windows = grid_windows(width_m=width_m, height_m=height_m, floor_m=floor_m, bay_m=bay_m,
                           win_w=win_w, win_h=win_h, sill_m=sill_m, first_u=first_u,
                           pair_gap_m=pair_gap_m)
    kept = [w for i, w in enumerate(windows) if i not in skip]
    for w in kept:
        p.paint(w, glass)
    return p, kept


def scene(p: Painter, windows: list[Rect], *, coverage: Optional[CoverageRaster] = None,
          depth: bool = False, noise: float = 4.0, seed: int = 0) -> Scene:
    rgb, alpha = p.finish(noise=noise, seed=seed)
    return Scene(rgb=rgb, alpha=alpha, grid=p.grid, bounds=(0.0, p.width_m, 0.0, p.height_m),
                 windows=windows, coverage=coverage,
                 front_m=p.front.copy() if depth else None)


def full_coverage(width_m: float, height_m: float, cell_m: float = 0.5,
                  holes: tuple[Rect, ...] = ()) -> CoverageRaster:
    """A coverage grid laid out the way the stage does it, with zero-view holes."""
    nu, nv = int(math.ceil(width_m / cell_m)), int(math.ceil(height_m / cell_m))
    views = np.full((nv, nu), 6, np.int32)
    for u0, u1, v0, v1 in holes:
        i0, i1 = int(math.floor(u0 / cell_m)), int(math.ceil(u1 / cell_m))
        j0, j1 = int(math.floor(v0 / cell_m)), int(math.ceil(v1 / cell_m))
        views[max(0, j0):j1, max(0, i0):i1] = 0
    return CoverageRaster(views=views, u_min=0.0, v_min=0.0, cell_m=cell_m)


def overlaps(a: Rect, b: Rect) -> bool:
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def centre(r: Rect) -> tuple[float, float]:
    return 0.5 * (r[0] + r[1]), 0.5 * (r[2] + r[3])
