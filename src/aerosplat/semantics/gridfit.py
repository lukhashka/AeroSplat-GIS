"""Window candidates from the periodic grid of a facade orthophoto (ROADMAP 7.3, first step).

Regular facades repeat: storeys at a fixed height, bays at a fixed pitch along the wall.
On the head-on, metric facade orthophoto (`semantics.ortho`) that regularity is a strong
deterministic prior, and this module turns it into *candidates* — never into verified
openings (ADR-0009). A person confirms or rejects every one in the facade editor.

The steps, each a function of its own so a later detector can replace one of them:

1. **masks** — rendered (alpha), covered (the facade coverage grid: where no capture
   resolved the wall, 3DGS invents texture, §5.3), vegetation (green) and, when a depth
   render is given, anything standing well in front of the wall plane (§5.5);
2. **cues** — per-pixel "looks like an opening" maps, robustly normalised: darkness and
   blueness (`CUES`; another cue is one more entry there);
3. **columns** — the bay pitch from the masked autocorrelation of each cue's detrended
   column profile, then the opening strips inside one folded bay (Otsu on the fold);
4. **agreement** — both cues are scored. When both see columns but different ones, the
   facade is *refused* rather than guessed: balcony shadows read dark but not blue, and a
   grid laid on them puts every box on a shadow;
5. **rows** — the storey height from the profile restricted to the opening columns,
   checked against the storey prior shared with addressing (2.4–4.2 m);
6. **cells** — every (storey, bay, strip) cell is compared with the wall reference of its
   own storey — the median cue of the non-opening pixels there, within a bay or so of the
   cell — and kept only if it stands out from it, and is rendered, covered and not behind
   foliage or an occluder. A grid that leaves most of its visible cells empty is refused.

A refusal is a result, not an error: `GridfitResult.refusals` says why, in words for the
operator. Numpy only and deterministic: the same picture always gives the same answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from aerosplat.semantics.addressing import FLOOR_H_MAX_M, FLOOR_H_MIN_M

# Bump on any change that can change the output for the same picture: `provenance.model`
# records it on every candidate, so a reviewer can tell which rules proposed what.
VERSION = "1.0.0"
MODEL = f"gridfit@{VERSION}"

F64 = NDArray[np.float64]
Mask = NDArray[np.bool_]

# Human-facing names of the cues, for refusal messages and candidate notes.
CUE_LABEL = {"dark": "темнота", "blue": "синява", "dark+blue": "темнота+синява"}
# A run narrower than this in the folded bay is a frame or a seam, not an opinion.
MIN_OPINION_M = 0.25


# ── inputs ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PixelGrid:
    """Pixel centre (i, j) ↔ (u, v): `u = u00 + i·m`, `v = v00 − j·m` (`FacadeGrid`'s layout).

    Columns run along u and rows run *down* the wall; the facade orthophoto is always
    rendered that way (`semantics.ortho`), so a rotated grid is refused outright.
    """

    m_per_px: float
    u00: float
    v00: float

    @classmethod
    def of(cls, grid) -> "PixelGrid":
        """From a `FacadeGrid` (or anything with `m_per_px`, `pixel00_uv`, `du_uv`, `dv_uv`)."""
        m = float(grid.m_per_px)
        du, dv = grid.du_uv, grid.dv_uv
        if not (abs(du[0] - m) < 1e-9 * max(1.0, m) and abs(du[1]) < 1e-12
                and abs(dv[0]) < 1e-12 and abs(dv[1] + m) < 1e-9 * max(1.0, m)):
            raise ValueError("gridfit needs an axis-aligned facade grid: du = (m, 0), dv = (0, −m)")
        return cls(m_per_px=m, u00=float(grid.pixel00_uv[0]), v00=float(grid.pixel00_uv[1]))

    def u_of(self, i) -> F64:
        return self.u00 + np.asarray(i, float) * self.m_per_px

    def v_of(self, j) -> F64:
        return self.v00 - np.asarray(j, float) * self.m_per_px


@dataclass(frozen=True)
class CoverageRaster:
    """The facade coverage grid (`views[j][i]`, row 0 at the bottom of the wall).

    Cell (i, j) covers u from `u_min + i·cell_m` and v from `v_min + j·cell_m`, exactly
    as `CoverageGrid` on the API; a zero means no capture resolved that patch.
    """

    views: NDArray[np.int32]
    u_min: float
    v_min: float
    cell_m: float

    def mask(self, grid: PixelGrid, shape: tuple[int, int]) -> Mask:
        """True where the pixel centre falls in a cell with at least one resolving view."""
        h, w = shape
        views = np.asarray(self.views)
        if views.ndim != 2 or views.size == 0 or self.cell_m <= 0:
            return np.zeros(shape, bool)
        ci = np.floor((grid.u_of(np.arange(w)) - self.u_min) / self.cell_m).astype(np.int64)
        rj = np.floor((grid.v_of(np.arange(h)) - self.v_min) / self.cell_m).astype(np.int64)
        inside = ((rj[:, None] >= 0) & (rj[:, None] < views.shape[0])
                  & (ci[None, :] >= 0) & (ci[None, :] < views.shape[1]))
        vals = views[np.clip(rj, 0, views.shape[0] - 1)][:, np.clip(ci, 0, views.shape[1] - 1)]
        return inside & (vals > 0)


@dataclass(frozen=True)
class GridfitParams:
    """Every threshold in one place. The defaults are what `VERSION` was measured with."""

    floor_prior_m: tuple[float, float] = (FLOOR_H_MIN_M, FLOOR_H_MAX_M)
    # Searched wider than the prior on purpose: a storey found at 1.3 or 5.2 m is a
    # reason to refuse, which a search confined to the prior could never report.
    floor_search_m: tuple[float, float] = (1.5, 8.5)
    bay_search_m: tuple[float, float] = (0.9, 12.0)
    # `opening.window` size prior (taxonomy); the API passes the Site's own.
    opening_w_m: tuple[float, float] = (0.5, 2.6)
    opening_h_m: tuple[float, float] = (0.5, 2.6)
    # Normalised autocorrelation a cue's profile must reach to count as periodic.
    min_periodicity: float = 0.25
    # The folded bay must separate openings from wall by this much (robust z) for a cue
    # to count as periodic at all: a mullion's ripple repeats too, faintly.
    min_contrast: float = 0.3
    # Each cue must confirm at least this share of the other's opening columns.
    min_agreement: float = 0.75
    # Of the cells nothing masks, at least this share must look like openings, or the
    # grid is not the facade's: a regular facade has most of its cells filled.
    min_fill: float = 0.5
    # A cell's median cue must exceed its storey's wall reference by this many spreads.
    min_score: float = 1.0
    # Per cell: fractions of pixels that must be rendered / covered, and that may be
    # foliage / occluded before the cell is given up -- a sliver, no more: a pane half
    # behind a crown is not evidence of anything.
    min_rendered: float = 0.9
    min_covered: float = 0.8
    max_vegetation: float = 0.02
    max_occluded: float = 0.02
    # Per facade: the share of the picture that must be usable at all.
    min_usable: float = 0.3
    # Green index (G − (R+B)/2) / (R+G+B) above which a pixel is foliage.
    vegetation_index: float = 0.06
    # A surface this far in front of the wall plane is an occluder, not the wall (§5.5).
    # Balconies stick out 1–1.5 m, so this must stay above that.
    occluder_m: float = 2.0


# ── outputs ──────────────────────────────────────────────────────────────
class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Refusal(_Model):
    code: str                       # stable, for tests and statistics
    message: str                    # for the operator, in Ukrainian like the rest of the UI


class Strip(_Model):
    """One opening column inside a bay: where it starts along u (mod the pitch) and how wide."""

    phase_m: float
    width_m: float


class CueFit(_Model):
    """What one cue says about the columns of the facade."""

    cue: str
    periodic: bool                  # repeats, and strongly enough to be openings
    bay_m: Optional[float] = None
    strength: float = 0.0           # normalised autocorrelation at the bay pitch
    contrast: float = 0.0           # opening class − wall class in the folded bay (robust z)
    widest_m: float = 0.0           # widest run the cue stands out on, window-sized or not
    strips: list[Strip] = Field(default_factory=list)   # window-sized runs only


class RowFit(_Model):
    floor_m: float
    strength: float
    band_phase_m: float             # bottom of the opening band, v mod floor_m
    band_h_m: float


class GridCell(_Model):
    """One cell of the fitted grid, kept or not, with the reason."""

    u0: float
    u1: float
    v0: float
    v1: float
    row: int                        # 0 = lowest storey band found (not a floor address)
    bay: int                        # 0 = leftmost bay found, facing the wall
    strip: int
    score: Optional[float] = None   # (cell median − storey wall reference) / spread
    rendered: float
    covered: float
    vegetation: float
    occluded: float
    accepted: bool
    reason: str = ""                # why not: not_rendered | uncovered | occluded | vegetation | wall
    confidence: float = 0.0


class GridfitResult(_Model):
    model: str = MODEL
    m_per_px: float
    usable_fraction: float
    cue: Optional[str] = None       # the cue the cells were scored on
    agreement: Optional[float] = None   # IoU of the two cues' column masks, when both are periodic
    cues: list[CueFit] = Field(default_factory=list)
    bay_m: Optional[float] = None
    strips: list[Strip] = Field(default_factory=list)
    floor: Optional[RowFit] = None
    refusals: list[Refusal] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    cells: list[GridCell] = Field(default_factory=list)

    @property
    def refused(self) -> bool:
        return bool(self.refusals)

    @property
    def accepted(self) -> list[GridCell]:
        return [c for c in self.cells if c.accepted]

    @property
    def proposals(self) -> list[GridCell]:
        """The cells worth showing a person: none at all from a refused facade."""
        return [] if self.refused else self.accepted

    def note(self, cell: GridCell) -> str:
        """What a candidate says about where it came from, for the reviewer (Ukrainian UI)."""
        parts = [self.model]
        if self.floor is not None:
            parts.append(f"поверх {self.floor.floor_m:.2f} м")
        if self.bay_m is not None:
            parts.append(f"крок колонок {self.bay_m:.2f} м")
        if self.cue:
            cue = CUE_LABEL.get(self.cue, self.cue)
            parts.append(f"ознака {cue}" + (f" (згода {self.agreement:.2f})"
                                            if self.agreement is not None else ""))
        if cell.score is not None:
            parts.append(f"z={cell.score:.1f}")
        parts.append(f"ряд {cell.row}, колонка {cell.bay}.{cell.strip}")
        return "; ".join(parts) + ". Сітка, не детектор: перевірте, що це справді отвір."


# ── cues ─────────────────────────────────────────────────────────────────
def cue_dark(rgb: F64) -> F64:
    """Glass in a daylight render reads darker than the wall around it."""
    return -(0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])


def cue_blue(rgb: F64) -> F64:
    """Glass reflects the sky; a shadow is dark but not blue."""
    return (rgb[..., 2] - rgb[..., 0]) / (rgb.sum(axis=-1) + 1e-3)


# Higher = more opening-like. Order matters only for reporting.
CUES: dict[str, Callable[[F64], F64]] = {"dark": cue_dark, "blue": cue_blue}


def green_index(rgb: F64) -> F64:
    return (rgb[..., 1] - 0.5 * (rgb[..., 0] + rgb[..., 2])) / (rgb.sum(axis=-1) + 1e-3)


def robust_z(x: F64, valid: Mask) -> Optional[F64]:
    """(x − median) / spread over the valid pixels; None when the cue is flat there."""
    v = x[valid]
    if v.size < 16:
        return None
    med = float(np.median(v))
    mad = 1.4826 * float(np.median(np.abs(v - med)))
    # A synthetic wall can be one exact colour, which collapses the MAD to zero.
    spread = max(mad, 0.25 * float(v.std()))
    if spread < 1e-6:
        return None
    return (x - med) / spread


# ── 1-D signal helpers ───────────────────────────────────────────────────
def box_mean(mask: NDArray, k: int) -> F64:
    """Mean of `mask` in a k×k window around every pixel (integral image, zero padding)."""
    k = max(1, int(k) | 1)
    r = k // 2
    a = np.pad(np.asarray(mask, float), r)
    s = np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    h, w = np.asarray(mask).shape
    tot = s[k:k + h, k:k + w] - s[:h, k:k + w] - s[k:k + h, :w] + s[:h, :w]
    return tot / (k * k)


def profile(values: F64, mask: Mask, axis: int) -> tuple[F64, F64]:
    """Mean of `values` over the masked pixels along `axis`, and how many there were."""
    w = mask.sum(axis=axis).astype(float)
    s = np.where(mask, values, 0.0).sum(axis=axis)
    return s / np.maximum(w, 1.0), w


def detrend(p: F64, m: Mask, win_px: float) -> F64:
    """Remove slow shading (sun, ambient occlusion) with a masked moving average."""
    n = len(p)
    r = max(1, int(win_px) // 2)
    cx = np.concatenate(([0.0], np.cumsum(np.where(m, p, 0.0))))
    cw = np.concatenate(([0.0], np.cumsum(m.astype(float))))
    lo = np.clip(np.arange(n) - r, 0, n)
    hi = np.clip(np.arange(n) + r + 1, 0, n)
    sx, sw = cx[hi] - cx[lo], cw[hi] - cw[lo]
    return np.where(m, p - sx / np.maximum(sw, 1e-9), 0.0)


def masked_autocorr(x: F64, m: Mask) -> tuple[F64, F64]:
    """Normalised autocorrelation over the overlap of valid samples, and the overlap size.

    `r[k] = Σ x_i x_{i+k} / sqrt(Σ x_i² · Σ x_{i+k}²)`, all sums over i where both samples
    are valid. A tree or an uncovered patch then shortens the overlap instead of
    pretending to be wall.
    """
    n = len(x)
    size = 1 << int(math.ceil(math.log2(max(2, 2 * n))))
    xm = np.where(m, x, 0.0)
    mm = m.astype(float)
    X, M, X2 = (np.fft.rfft(a, size) for a in (xm, mm, xm * xm))

    def corr(a, b) -> F64:      # Σ_i a_{i+k} b_i
        return np.fft.irfft(a * np.conj(b), size)[:n]

    num = corr(X, X)
    e_shift, e_base = corr(X2, M), corr(M, X2)
    cnt = np.rint(corr(M, M))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / np.sqrt(np.maximum(e_shift * e_base, 1e-18))
    r[cnt < 2] = 0.0
    return r, cnt


def _parabolic(r: F64, k: int) -> float:
    if 0 < k < len(r) - 1:
        a, b, c = r[k - 1], r[k], r[k + 1]
        den = a - 2 * b + c
        if den < 0:
            return k + 0.5 * (a - c) / den
    return float(k)


@dataclass(frozen=True)
class Period:
    px: float                       # period in samples (sub-sample)
    strength: float                 # normalised autocorrelation at the period


def find_period(x: F64, m: Mask, lo_px: float, hi_px: float, min_strength: float) -> Optional[Period]:
    """The fundamental period of a masked profile inside [lo, hi], or None.

    The strongest autocorrelation peak wins, unless a peak at an integer fraction of it
    is nearly as strong (then that one is the fundamental: a grid repeats at 2P too).
    The period is then refined on the peaks at its multiples, which turns a sub-pixel
    error at one lag into one divided by the number of periods on the wall.
    """
    n = len(x)
    if int(m.sum()) < 8:
        return None
    r, cnt = masked_autocorr(x, m)
    lags = np.arange(n)
    # At least one and a half periods of overlap behind every lag considered.
    usable = cnt >= 1.5 * lags
    lo, hi = max(2, int(math.ceil(lo_px))), min(int(hi_px), n - 2)
    peaks = [k for k in range(lo, hi + 1)
             if usable[k] and r[k] >= min_strength and r[k] >= r[k - 1] and r[k] > r[k + 1]]
    if not peaks:
        return None
    best = max(peaks, key=lambda k: r[k])
    for k in sorted(peaks):
        if k >= best:
            break
        ratio = best / k
        if r[k] >= 0.85 * r[best] and round(ratio) >= 2 and abs(ratio - round(ratio)) < 0.1:
            best = k
            break
    p = _parabolic(r, best)
    strength = float(r[best])
    n_valid = int(m.sum())
    pts = [(1, p)]
    mult = 2
    while True:
        c = mult * p
        k0 = int(round(c))
        if k0 >= n - 2 or cnt[k0] < max(2.0 * p, 0.25 * n_valid):
            break
        w = max(2, int(0.08 * p))
        a, b = max(1, k0 - w), min(n - 2, k0 + w)
        k = a + int(np.argmax(r[a:b + 1]))
        if r[k] >= 0.5 * strength and a < k < b:
            pts.append((mult, _parabolic(r, k)))
        mult += 1
    ms = np.array([q[0] for q in pts], float)
    ls = np.array([q[1] for q in pts], float)
    return Period(px=float((ms * ls).sum() / (ms * ms).sum()), strength=strength)


def fold(x: F64, m: Mask, coord: F64, period: float, nb: int) -> F64:
    """Average a masked profile modulo `period` of its metric coordinate, into `nb` bins."""
    idx = np.floor(np.mod(coord, period) / period * nb).astype(np.int64) % nb
    s = np.bincount(idx, weights=np.where(m, x, 0.0), minlength=nb)
    c = np.bincount(idx, weights=m.astype(float), minlength=nb)
    out = np.where(c > 0, s / np.maximum(c, 1e-9), np.nan)
    if np.isnan(out).all():
        return np.zeros(nb)
    out = np.where(np.isnan(out), np.nanmedian(out), out)
    # A light circular [1 2 1] smoothing: the fold of a few periods is still noisy.
    return 0.25 * np.roll(out, 1) + 0.5 * out + 0.25 * np.roll(out, -1)


def otsu(values: F64) -> float:
    """The threshold that best splits `values` into two classes (Otsu, 1979)."""
    v = np.sort(np.asarray(values, float))
    n = len(v)
    if n < 2 or v[0] == v[-1]:
        return float(v[0]) if n else 0.0
    c = np.cumsum(v)
    i = np.arange(1, n)
    w0 = i / n
    m0 = c[:-1] / i
    m1 = (c[-1] - c[:-1]) / (n - i)
    between = w0 * (1 - w0) * (m0 - m1) ** 2
    k = int(np.argmax(between))
    return float(0.5 * (v[k] + v[k + 1]))


def circular_runs(high: Mask) -> list[tuple[int, int]]:
    """Runs of True in a circular array: [(start, length)], longest first."""
    n = len(high)
    if high.all():
        return [(0, n)]
    if not high.any():
        return []
    s = int(np.argmin(high))            # start scanning just after a low bin
    out: list[tuple[int, int]] = []
    run, start = 0, 0
    for step in range(1, n + 1):
        i = (s + step) % n
        if high[i]:
            if run == 0:
                start = i
            run += 1
        elif run:
            out.append((start, run))
            run = 0
    if run:
        out.append((start, run))
    return sorted(out, key=lambda r: (-r[1], r[0]))


def in_strips(coord: F64, period: float, strips: list[Strip]) -> Mask:
    """Which coordinates fall inside one of the strips, repeated every `period`."""
    out = np.zeros(np.shape(coord), bool)
    for s in strips:
        out |= np.mod(np.asarray(coord) - s.phase_m, period) < s.width_m
    return out


# ── the steps ────────────────────────────────────────────────────────────
@dataclass
class Masks:
    rendered: Mask
    covered: Mask
    vegetation: Mask
    occluded: Mask

    @property
    def valid(self) -> Mask:
        return self.rendered & self.covered & ~self.vegetation & ~self.occluded


def build_masks(rgb: F64, alpha: F64, grid: PixelGrid, coverage: Optional[CoverageRaster],
                front_m: Optional[F64], params: GridfitParams) -> Masks:
    shape = (int(alpha.shape[0]), int(alpha.shape[1]))
    rendered = alpha > 0.5
    covered = coverage.mask(grid, shape) if coverage is not None else np.ones(shape, bool)
    # Foliage is speckled with dark gaps that read as glass; close them over ~0.4 m.
    green = rendered & (green_index(rgb) > params.vegetation_index)
    k = max(3, int(round(0.4 / grid.m_per_px)))
    vegetation = green | (rendered & (box_mean(green, k) > 0.3))
    if front_m is not None:
        occ = rendered & (np.nan_to_num(front_m, nan=0.0) > params.occluder_m)
        occluded = occ | (rendered & (box_mean(occ, k) > 0.3))
    else:
        occluded = np.zeros(shape, bool)
    return Masks(rendered=rendered, covered=covered, vegetation=vegetation, occluded=occluded)


@dataclass
class _Columns:
    fit: CueFit
    period_m: float
    mask: Mask                      # per image column: inside a window-sized opening strip
    # Per image column: inside any run the cue stands out on (noise slivers aside), of
    # whatever width. That is the cue's opinion about the columns, which the other cue
    # is checked against.
    opinion: Mask
    # The cue repeats a run wider than any window: a loggia, a pier, a balcony stack.
    too_wide: bool = False


def fit_columns(cue: str, z: F64, valid: Mask, grid: PixelGrid,
                params: GridfitParams) -> _Columns:
    """Bay pitch and the window-sized opening strips of one cue map.

    A cue counts as periodic only if its profile repeats (autocorrelation) *and* the
    repeat is strong: the folded bay must separate into an opening class that stands
    clearly above the wall class. A faint ripple at a mullion's pitch repeats too.
    """
    px = grid.m_per_px
    w = valid.shape[1]
    prof, weight = profile(z, valid, axis=0)
    ok = weight >= 0.3 * max(1.0, float(weight.max()))
    lo, hi = params.bay_search_m
    x = detrend(prof, ok, 1.5 * hi / px)
    per = find_period(x, ok, lo / px, hi / px, params.min_periodicity)
    if per is None:
        none = np.zeros(w, bool)
        return _Columns(CueFit(cue=cue, periodic=False), 0.0, none, none)
    period_m = per.px * px
    nb = max(16, int(round(per.px)))
    u = grid.u_of(np.arange(w))
    f = fold(x, ok, u, period_m, nb)
    high = f > otsu(f)
    contrast = float(f[high].mean() - f[~high].mean()) if high.any() and (~high).any() else 0.0
    runs = [Strip(phase_m=round(start / nb * period_m, 4), width_m=round(n / nb * period_m, 4))
            for start, n in circular_runs(high)]
    w_lo, w_hi = params.opening_w_m
    strips = sorted((r for r in runs if w_lo <= r.width_m <= w_hi), key=lambda r: r.phase_m)
    fit = CueFit(cue=cue, periodic=contrast >= params.min_contrast, bay_m=round(period_m, 4),
                 strength=round(per.strength, 4), contrast=round(contrast, 3),
                 widest_m=round(max((r.width_m for r in runs), default=0.0), 3), strips=strips)
    said = [r for r in runs if r.width_m >= MIN_OPINION_M]
    return _Columns(fit, period_m, in_strips(u, period_m, strips) & ok,
                    in_strips(u, period_m, said) & ok, any(r.width_m > w_hi for r in runs))


def _runs(mask: Mask) -> list[tuple[int, int]]:
    """[start, stop) of every run of True in a 1-D mask."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.nonzero(edges == 1)[0].tolist(), np.nonzero(edges == -1)[0].tolist()))


def confirmed_share(a: Mask, b: Mask) -> float:
    """Share of `a`'s opening columns (by width) that `b` confirms.

    A run of `a` across the facade is confirmed when `b` covers at least 30 % of it, or
    when a run of `b` lies at least half inside it. Containment either way, not IoU:
    blueness marks the whole pane while darkness marks only its shaded part, so the
    two strips differ in width but not in place -- and it is the place that matters.
    """
    b_runs = _runs(b)
    total = confirmed = 0
    for i0, i1 in _runs(a):
        n = i1 - i0
        total += n
        covered = int(b[i0:i1].sum())
        inside = any(min(i1, j1) - max(i0, j0) >= 0.5 * (j1 - j0) for j0, j1 in b_runs
                     if j1 > i0 and j0 < i1)
        if covered >= 0.3 * n or inside:
            confirmed += n
    return confirmed / total if total else 0.0


def agreement(a: _Columns, b: _Columns) -> float:
    """How far two cues agree on the opening columns: each must confirm the other.

    A cue that repeats something wider than any window does not agree with anything:
    on a facade of balcony stacks darkness marks the whole stack, and that would
    "confirm" every pane blueness found inside it.
    """
    if a.too_wide or b.too_wide:
        return 0.0
    return min(confirmed_share(a.opinion, b.opinion), confirmed_share(b.opinion, a.opinion))


@dataclass
class _Rows:
    fit: Optional[RowFit]
    refusal: Optional[Refusal]


def fit_rows(z: F64, valid: Mask, column_mask: Mask, grid: PixelGrid,
             params: GridfitParams) -> _Rows:
    """Storey height and the opening band, from the opening columns only."""
    px = grid.m_per_px
    sel = valid & column_mask[None, :]
    prof, weight = profile(z, sel, axis=1)
    ok = weight >= 0.2 * max(1.0, float(weight.max()))
    lo, hi = params.floor_search_m
    x = detrend(prof, ok, 1.5 * hi / px)
    per = find_period(x, ok, lo / px, hi / px, params.min_periodicity)
    if per is None:
        return _Rows(None, Refusal(
            code="no_floors", message="Поверхи не повторюються в колонках отворів: "
                                      "періодичності по висоті не знайдено"))
    floor_m = per.px * px
    f_lo, f_hi = params.floor_prior_m
    if not f_lo <= floor_m <= f_hi:
        return _Rows(None, Refusal(
            code="floor_out_of_prior",
            message=f"Крок поверхів {floor_m:.2f} м поза апріорними {f_lo}–{f_hi} м: "
                    f"це не сітка поверхів (або половина/подвоєння кроку)"))
    nb = max(16, int(round(per.px)))
    v = grid.v_of(np.arange(valid.shape[0]))
    f = fold(x, ok, v, floor_m, nb)
    runs = circular_runs(f > otsu(f))
    h_lo, h_hi = params.opening_h_m
    runs = [r for r in runs if h_lo <= r[1] / nb * floor_m <= h_hi]
    if not runs:
        return _Rows(None, Refusal(
            code="no_band", message=f"На поверсі {floor_m:.2f} м немає смуги висотою вікна "
                                    f"({h_lo}–{h_hi} м)"))
    start, n = runs[0]
    return _Rows(RowFit(floor_m=round(floor_m, 4), strength=round(per.strength, 4),
                        band_phase_m=round(start / nb * floor_m, 4),
                        band_h_m=round(n / nb * floor_m, 4)), None)


def _instances(phase: float, width: float, period: float, lo: float, hi: float) -> list[float]:
    """Starts of every repetition of [phase, phase + width) mod period that fits in [lo, hi]."""
    slack = 0.2 * width
    k0 = math.ceil((lo - slack - phase) / period)
    out = []
    k = k0
    while phase + k * period + width <= hi + slack:
        out.append(phase + k * period)
        k += 1
    return out


def _pixel_span(a: float, b: float, origin: float, step: float, n: int) -> tuple[int, int]:
    """Pixels whose centre lies in [a, b] along an axis `origin + i·step` (step may be < 0)."""
    i0, i1 = sorted(((a - origin) / step, (b - origin) / step))
    return max(0, int(math.ceil(i0 - 1e-9))), min(n, int(math.floor(i1 + 1e-9)) + 1)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def score_cells(z: F64, masks: Masks, grid: PixelGrid, bay_m: float, strips: list[Strip],
                rows: RowFit, bounds: tuple[float, float, float, float], period_q: float,
                agree_q: float, params: GridfitParams) -> tuple[list[GridCell], list[str]]:
    """Lay the grid over the facade and decide every cell against its storey's wall."""
    h, w = z.shape
    px = grid.m_per_px
    u_lo, u_hi, v_lo, v_hi = bounds
    valid = masks.valid
    floors = _instances(rows.band_phase_m, rows.band_h_m, rows.floor_m, v_lo, v_hi)
    boxes: list[tuple[int, int, int, float, float, float, float]] = []
    for si, s in enumerate(strips):
        starts = _instances(s.phase_m, s.width_m, bay_m, u_lo, u_hi)
        for k, v0 in enumerate(floors):
            for b in starts:
                bay = int(round((b - s.phase_m) / bay_m))
                boxes.append((k, bay, si, b, b + s.width_m, v0, v0 + rows.band_h_m))
    notes: list[str] = []
    if not boxes:
        return [], notes
    bay0 = min(b[1] for b in boxes)

    # Everything that is not an opening cell (with a little margin for the blurred
    # frame) is wall: that is the reference a cell has to stand out from.
    opening = np.zeros((h, w), bool)
    margin = 0.15
    for _, _, _, u0, u1, v0, v1 in boxes:
        i0, i1 = _pixel_span(u0 - margin, u1 + margin, grid.u00, px, w)
        j0, j1 = _pixel_span(v0 - margin, v1 + margin, grid.v00, -px, h)
        opening[j0:j1, i0:i1] = True
    ib0, ib1 = _pixel_span(u_lo, u_hi, grid.u00, px, w)
    wall = valid & ~opening
    wall[:, :ib0] = False
    wall[:, ib1:] = False
    wall_all = z[wall]
    if wall_all.size < 50:
        notes.append("замало стіни між отворами для опорного рівня")
        return [], notes
    global_ref = float(np.median(wall_all))

    # The reference is local: the wall of the cell's own storey, within a bay or so of
    # it. Sun and ambient occlusion shade a long wall unevenly, and a storey-wide median
    # would score a darker end of the wall as glass.
    pad = 0.5 * (rows.floor_m - rows.band_h_m)
    reach = max(bay_m, 2.5)
    floor_rows = [_pixel_span(v0 - pad, v0 + rows.band_h_m + pad, grid.v00, -px, h) for v0 in floors]
    floor_ref: dict[int, float] = {}
    residuals: list[F64] = []
    for k, (j0, j1) in enumerate(floor_rows):
        vals = z[j0:j1][wall[j0:j1]]
        floor_ref[k] = float(np.median(vals)) if vals.size >= 50 else global_ref
        u = u_lo
        while u < u_hi:
            a0, a1 = _pixel_span(u, u + reach, grid.u00, px, w)
            seg = z[j0:j1, a0:a1][wall[j0:j1, a0:a1]]
            if seg.size >= 30:
                residuals.append(seg - np.median(seg))
            u += reach
    res = np.concatenate(residuals) if residuals else wall_all - global_ref
    spread = max(1.4826 * float(np.median(np.abs(res))), 0.05)
    fell_back = 0

    cells: list[GridCell] = []
    for k, bay, si, u0, u1, v0, v1 in boxes:
        i0, i1 = _pixel_span(u0, u1, grid.u00, px, w)
        j0, j1 = _pixel_span(v0, v1, grid.v00, -px, h)
        n_full = max(1, int(round((u1 - u0) / px)) * int(round((v1 - v0) / px)))
        n_in = max(0, i1 - i0) * max(0, j1 - j0)

        def frac(m: Mask) -> float:
            return float(m[j0:j1, i0:i1].sum()) / n_full if n_in else 0.0

        rendered = min(1.0, frac(masks.rendered))
        covered = min(1.0, frac(masks.covered & masks.rendered))
        vegetation = min(1.0, frac(masks.vegetation))
        occluded = min(1.0, frac(masks.occluded))
        # The score reads the inner part only: the frame of a 3DGS window is a blur.
        du, dv = 0.15 * (u1 - u0), 0.15 * (v1 - v0)
        a0, a1 = _pixel_span(u0 + du, u1 - du, grid.u00, px, w)
        b0, b1 = _pixel_span(v0 + dv, v1 - dv, grid.v00, -px, h)
        inner = z[b0:b1, a0:a1][valid[b0:b1, a0:a1]]
        fj0, fj1 = floor_rows[k]
        uc = 0.5 * (u0 + u1)
        c0, c1 = _pixel_span(uc - reach, uc + reach, grid.u00, px, w)
        near = z[fj0:fj1, c0:c1][wall[fj0:fj1, c0:c1]]
        if near.size >= 50:
            ref = float(np.median(near))
        else:
            ref = floor_ref[k]
            fell_back += 1
        score = (float(np.median(inner)) - ref) / spread if inner.size >= 4 else None

        if rendered < params.min_rendered:
            reason = "not_rendered"
        elif covered < params.min_covered:
            reason = "uncovered"
        elif occluded > params.max_occluded:
            reason = "occluded"
        elif vegetation > params.max_vegetation:
            reason = "vegetation"
        elif score is None or score < params.min_score:
            reason = "wall"
        else:
            reason = ""
        conf = 0.0
        if not reason and score is not None:
            conf = period_q * agree_q * covered * _sigmoid((score - params.min_score) / 0.5)
            conf = float(min(0.95, max(0.05, conf)))
        cells.append(GridCell(
            u0=round(u0, 4), u1=round(u1, 4), v0=round(v0, 4), v1=round(v1, 4), row=k,
            bay=bay - bay0, strip=si, score=None if score is None else round(score, 3),
            rendered=round(rendered, 3), covered=round(covered, 3),
            vegetation=round(vegetation, 3), occluded=round(occluded, 3),
            accepted=not reason, reason=reason, confidence=round(conf, 3)))
    if fell_back:
        notes.append(f"{fell_back} клітинок без стіни поруч — опорний рівень усього поверху")
    return cells, notes


# ── the whole thing ──────────────────────────────────────────────────────
def _said(c: _Columns) -> str:
    """What one cue claims about the columns, for a refusal message."""
    if c.too_wide or not c.fit.strips:
        return (f"{CUE_LABEL[c.fit.cue]} — повторюється смуга {c.fit.widest_m:.1f} м "
                f"на крок {c.fit.bay_m:.2f} м, ширша за вікно")
    return f"{CUE_LABEL[c.fit.cue]} — смуги вікон з кроком {c.fit.bay_m:.2f} м"


def _as_rgb01(rgb) -> F64:
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError(f"rgb must be (H, W, 3), got {a.shape}")
    a = a[..., :3].astype(float)
    return a / 255.0 if a.max(initial=0.0) > 1.0 + 1e-9 else a


def gridfit(rgb, alpha, grid, *, coverage: Optional[CoverageRaster] = None,
            front_m: Optional[F64] = None,
            bounds_uv: Optional[tuple[float, float, float, float]] = None,
            params: Optional[GridfitParams] = None) -> GridfitResult:
    """Fit the window grid of one facade orthophoto and decide every cell.

    `rgb` (H, W, 3) in 0..255 or 0..1, `alpha` (H, W) in 0..1 (or 0..255), `grid` the
    orthophoto's `FacadeGrid` or a `PixelGrid`. `coverage` the facade coverage grid,
    `front_m` how far each pixel's surface stands in front of the wall plane (from a
    depth render), `bounds_uv` = (u_min, u_max, v_min, v_max) of the facade itself, so
    cells are not laid on the margin around it.
    """
    params = params or GridfitParams()
    pg = grid if isinstance(grid, PixelGrid) else PixelGrid.of(grid)
    rgb01 = _as_rgb01(rgb)
    al = np.asarray(alpha, float)
    if al.ndim == 3:
        al = al[..., 0]
    if al.max(initial=0.0) > 1.0 + 1e-9:
        al = al / 255.0
    if al.shape != rgb01.shape[:2]:
        raise ValueError(f"alpha {al.shape} does not match rgb {rgb01.shape[:2]}")
    if front_m is not None and np.shape(front_m) != al.shape:
        raise ValueError(f"front_m {np.shape(front_m)} does not match rgb {al.shape}")

    masks = build_masks(rgb01, al, pg, coverage, front_m, params)
    valid = masks.valid
    rendered_frac = float(masks.rendered.mean())
    usable = float(valid.sum()) / max(1, int(masks.rendered.sum()))
    out = GridfitResult(m_per_px=pg.m_per_px, usable_fraction=round(usable, 3))

    if rendered_frac < 0.2:
        out.refusals.append(Refusal(code="not_rendered", message=(
            f"На ортофото майже немає фасаду (відрендерено {100 * rendered_frac:.0f}% кадру)")))
        return out
    if usable < params.min_usable:
        out.refusals.append(Refusal(code="no_coverage", message=(
            f"Придатна лише {100 * usable:.0f}% стіни (потрібно ≥ {100 * params.min_usable:.0f}%): "
            f"решта не покрита зйомкою, за деревами чи затулена")))
        return out

    # Cues, each normalised over the usable wall so they are comparable.
    zs: dict[str, F64] = {}
    for name, fn in CUES.items():
        z = robust_z(fn(rgb01), valid)
        if z is not None:
            zs[name] = z
    cols = {name: fit_columns(name, z, valid, pg, params) for name, z in zs.items()}
    out.cues = [c.fit for c in cols.values()]
    periodic = {n: c for n, c in cols.items() if c.fit.periodic}
    if not periodic:
        told = ", ".join(f"{CUE_LABEL[c.fit.cue]}: " + (
            f"крок {c.fit.bay_m:.2f} м, контраст {c.fit.contrast:.2f}" if c.fit.bay_m
            else "без періоду") for c in cols.values())
        out.refusals.append(Refusal(code="no_columns", message=(
            f"Колонки отворів не повторюються чітко за жодною ознакою ({told or 'ознак немає'}): "
            f"фасад не регулярний, без вікон або текстура надто слабка")))
        return out

    col_ok = valid.sum(axis=0) >= 0.3 * max(1, int(valid.sum(axis=0).max()))
    if len(periodic) >= 2:
        a, b = (periodic[n] for n in list(periodic)[:2])
        agree = agreement(a, b)
        out.agreement = round(agree, 3)
        if agree < params.min_agreement:
            out.refusals.append(Refusal(code="cues_disagree", message=(
                f"Ознаки бачать різні колонки ({_said(a)}; {_said(b)}; збіг {100 * agree:.0f}% < "
                f"{100 * params.min_agreement:.0f}%): темне може бути тінями балконів, а не "
                f"склом — розмітьте вручну")))
            return out
        # Both see the same columns: score on both at once, so a cell has to be dark
        # *and* blue to stand out, and a shadow gets only half the credit.
        cue_name = "dark+blue"
        z = 0.5 * (zs[a.fit.cue] + zs[b.fit.cue])
        chosen = fit_columns(cue_name, z, valid, pg, params)
        if not (chosen.fit.periodic and chosen.fit.strips):
            chosen = max((a, b), key=lambda c: c.fit.strength)
            cue_name = chosen.fit.cue
            z = zs[cue_name]
        agree_q = 1.0
    else:
        chosen = next(iter(periodic.values()))
        cue_name = chosen.fit.cue
        z = zs[cue_name]
        others = [CUE_LABEL[n] for n in zs if n != cue_name]
        out.notes.append(f"колонки бачить лише одна ознака ({CUE_LABEL[cue_name]})"
                         + (f"; {', '.join(others)} — без чіткого періоду" if others else ""))
        agree_q = 0.8
    out.cue = cue_name
    out.bay_m = chosen.fit.bay_m
    out.strips = chosen.fit.strips
    if not chosen.fit.strips:
        w_lo, w_hi = params.opening_w_m
        out.refusals.append(Refusal(code="no_strips", message=(
            f"У кроці колонок {chosen.period_m:.2f} м немає смуг ширини вікна ({w_lo}–{w_hi} м; "
            f"найширша {chosen.fit.widest_m:.1f} м): повторюється щось інше — пілястри, "
            f"лоджії, тіні")))
        return out

    rows = fit_rows(z, valid, chosen.mask, pg, params)
    if rows.fit is None:
        assert rows.refusal is not None
        out.refusals.append(rows.refusal)
        return out
    out.floor = rows.fit

    # Lay cells only on the facade itself and on the part of the picture that is wall.
    ii = np.nonzero(col_ok)[0]
    row_ok = valid.sum(axis=1) >= 0.1 * max(1, int(valid.sum(axis=1).max()))
    jj = np.nonzero(row_ok)[0]
    u_lo, u_hi = float(pg.u_of(ii[0])), float(pg.u_of(ii[-1]))
    v_lo, v_hi = float(pg.v_of(jj[-1])), float(pg.v_of(jj[0]))
    if bounds_uv is not None:
        u_lo, u_hi = max(u_lo, bounds_uv[0]), min(u_hi, bounds_uv[1])
        v_lo, v_hi = max(v_lo, bounds_uv[2]), min(v_hi, bounds_uv[3])
    period_q = math.sqrt(max(0.0, min(1.0, chosen.fit.strength))
                         * max(0.0, min(1.0, rows.fit.strength)))
    cells, notes = score_cells(z, masks, pg, chosen.period_m, chosen.fit.strips, rows.fit,
                               (u_lo, u_hi, v_lo, v_hi), period_q, agree_q, params)
    out.notes.extend(notes)
    out.cells = cells
    if not cells:
        out.refusals.append(Refusal(code="no_cells", message=(
            "Сітка не дала жодної клітинки всередині фасаду")))
        return out
    seen = [c for c in cells if c.reason in ("", "wall")]
    filled = sum(c.accepted for c in seen)
    if len(seen) >= 8 and filled < params.min_fill * len(seen):
        out.refusals.append(Refusal(code="sparse_grid", message=(
            f"Лише {filled} з {len(seen)} видимих клітинок сітки схожі на отвори "
            f"(потрібно ≥ {100 * params.min_fill:.0f}%): сітка не описує цей фасад")))
    return out
