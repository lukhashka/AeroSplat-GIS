"""Clearance of a corridor against the occupancy grid (aerosplat.surface), with reasons.

Two questions are asked of every leg of a mission:

1. does the path itself enter blocked space (occupied, unknown, outside the grid)? —
   `OccupancyGrid.path_clearance`, exact, voxel traversal;
2. does the path keep the leg's radius `r = r_drone + m + k·σ` from everything blocked?

For a leg that ends on a surface on purpose — the approach to a window ends `standoff` in front
of its wall, the takeoff starts on the ground — (2) must not count that surface, or no window
could ever be approached: the distance to it is held by onboard guidance (rangefinders,
facade-relative localisation; ADR-0009 p. 4, SIMULATION §3.3). Those legs are checked here by
brute force over a small window of voxels in which everything at or behind the surface's
plane (within `surface_relief_m` in front of it) is left out. Everything else — a balcony
above the window, a tree, the next wing of the building, unknown space — still counts.

Unknown and outside-the-grid voxels block exactly like occupied ones (DOMAIN §3 p. 3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from aerosplat.mission.models import Code, Finding, Severity, Vec3
from aerosplat.surface.models import FREE, OCCUPIED, state_name
from aerosplat.surface.occupancy import OccupancyGrid, as_points

F64 = NDArray[np.float64]
SQRT3 = math.sqrt(3.0)
UP = np.array([0.0, 0.0, 1.0])
# Brute force is O(samples x blocked voxels in the window): keep the window bounded.
MAX_WINDOW_VOXELS = 30_000_000


@dataclass(frozen=True)
class Plane:
    """A surface a leg ends on: points with (p - point) . normal <= offset are that surface."""

    point: Vec3
    normal: Vec3
    offset_m: float
    name: str


@dataclass
class LegCheck:
    min_clearance_m: float
    blocked_state: Optional[str]            # the path itself enters this (None: it does not)
    blocked_at: Optional[Vec3]
    assumed_free_at: Optional[Vec3]
    offenders: dict[Code, Vec3] = field(default_factory=dict)   # reason -> nearest voxel
    samples: Optional[F64] = None


def sample_path(points, step_m: float) -> F64:
    """Points along a polyline, at most `step_m` apart, ends included."""
    p = as_points(points, "path")
    out = [p[:1]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / step_m - 1e-9)))
        t = (np.arange(1, n + 1) / n)[:, None]
        out.append(a + t * (b - a))
    return np.concatenate(out)


def surface_z(grid: OccupancyGrid, x: float, y: float) -> Optional[float]:
    """Height of the 2.5-D surface (ground, roof) under (x, y); None where it is unknown."""
    s = grid.spec
    i, j, _ = s.index_of([[x, y, s.z0]])
    if not (0 <= i[0] < s.rows and 0 <= j[0] < s.cols):
        return None
    v = float(grid.column_top[i[0], j[0]])
    return v if math.isfinite(v) else None


def _blocked_in_window(grid: OccupancyGrid, lo: F64, hi: F64) -> tuple[F64, NDArray[np.uint8]]:
    """Centres and states of the non-free voxels whose cubes intersect the box [lo, hi]."""
    s = grid.spec
    i0, j0, k0 = (int(v[0]) for v in s.index_of(lo[None]))
    i1, j1, k1 = (int(v[0]) for v in s.index_of(hi[None]))
    n = (i1 - i0 + 1) * (j1 - j0 + 1) * (k1 - k0 + 1)
    if n > MAX_WINDOW_VOXELS:
        raise ValueError(f"corridor window of {n} voxels is too large for an exact check")
    ii, jj, kk = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), np.arange(k0, k1 + 1),
                             indexing="ij")
    centres = s.centre_of(ii.ravel(), jj.ravel(), kk.ravel())
    # state_at applies the grid's rules beyond its bounds (outside, underground, above).
    states = grid.state_at(centres)
    keep = states != FREE
    return centres[keep], states[keep]


def masked_clearance(grid: OccupancyGrid, samples: F64, radius_m: float,
                     exclude: Optional[Plane] = None) -> tuple[F64, F64, NDArray[np.uint8], F64]:
    """Exact distance from each sample to the nearest blocked voxel cube, leaving out the
    voxels of `exclude`. Exact below `radius_m + res`; inf means at least that far.

    Returns (clearance, nearest centres, nearest states, offending pairs): the pairs are
    (sample index, voxel centre, voxel state) for every voxel closer than `radius_m`."""
    r = grid.res_m
    reach = radius_m + r
    lo = samples.min(axis=0) - reach
    hi = samples.max(axis=0) + reach
    centres, states = _blocked_in_window(grid, lo, hi)
    if exclude is not None and len(centres):
        n = np.asarray(exclude.normal, float)
        d = (centres - np.asarray(exclude.point, float)) @ n
        keep = d > exclude.offset_m
        centres, states = centres[keep], states[keep]
    out = np.full(len(samples), np.inf)
    near_c = np.full((len(samples), 3), np.nan)
    near_s = np.zeros(len(samples), np.uint8)
    pairs: list[F64] = []
    if len(centres) == 0:
        return out, near_c, near_s, np.zeros((0, 5))
    chunk = max(1, 4_000_000 // len(centres))
    for a in range(0, len(samples), chunk):
        q = samples[a:a + chunk]
        gap = np.abs(q[:, None, :] - centres[None, :, :])
        d = np.sqrt((np.maximum(gap - 0.5 * r, 0.0) ** 2).sum(-1))
        m = np.argmin(d, axis=1)
        rows = np.arange(len(q))
        out[a:a + chunk] = d[rows, m]
        near_c[a:a + chunk] = centres[m]
        near_s[a:a + chunk] = states[m]
        si, ci = np.nonzero(d < radius_m)
        if len(si):
            pairs.append(np.column_stack([si + a, centres[ci], states[ci].astype(float)]))
    return out, near_c, near_s, (np.concatenate(pairs) if pairs else np.zeros((0, 5)))


def classify(samples: F64, pairs: F64, res_m: float) -> dict[Code, Vec3]:
    """Name what is inside the corridor: unknown space, something overhead, something else.

    `pairs` are (sample index, voxel centre, voxel state) for every voxel closer than the
    radius. A voxel is overhead when its lower face is above the sample it threatens.
    """
    out: dict[Code, Vec3] = {}
    if len(pairs) == 0:
        return out
    si = pairs[:, 0].astype(int)
    c = pairs[:, 1:4]
    st = pairs[:, 4].astype(int)
    d = np.linalg.norm(c - samples[si], axis=1)
    order = np.argsort(d)
    for idx in order:
        if st[idx] != OCCUPIED:
            code = Code.unknown_space
        elif c[idx, 2] - 0.5 * res_m >= samples[si[idx], 2]:
            code = Code.overhang_above
        else:
            code = Code.obstacle_in_corridor
        if code not in out:
            out[code] = (float(c[idx, 0]), float(c[idx, 1]), float(c[idx, 2]))
    return out


def check_leg(grid: OccupancyGrid, points, radius_m: float,
              exclude: Optional[Plane] = None) -> LegCheck:
    """Both questions for one leg (see the module docstring)."""
    pts = as_points(points, "leg")
    r = grid.res_m
    step = 0.5 * r
    samples = sample_path(pts, step)
    blocked_state: Optional[str] = None
    blocked_at: Optional[Vec3] = None
    if len(pts) >= 2 and float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) > 0:
        pc = grid.path_clearance(pts, step_m=step)
        if pc.blocked:
            blocked_state = pc.first_blocked_state
            s = float(pc.first_blocked_s or 0.0)
            seg = np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(pts, axis=0), axis=1)])
            k = min(int(np.searchsorted(seg, s, side="right")) - 1, len(pts) - 2)
            t = (s - seg[k]) / max(seg[k + 1] - seg[k], 1e-12)
            at = pts[k] + t * (pts[k + 1] - pts[k])
            blocked_at = (float(at[0]), float(at[1]), float(at[2]))
    else:
        code = int(grid.state_at(pts[:1])[0])
        if code != FREE:
            blocked_state = state_name(code)
            blocked_at = (float(pts[0, 0]), float(pts[0, 1]), float(pts[0, 2]))

    # The samples are `step` apart and clearance is 1-Lipschitz, so a sampled value at least
    # radius + step/2 proves the radius for the whole leg (the convention of path_clearance).
    eff = radius_m + 0.5 * step
    reach = eff + r
    if exclude is None:
        c = grid.clearance(samples)
        exact = c.copy()
        near = c < eff
        offenders: dict[Code, Vec3] = {}
        if near.any():
            # Only where the cheap bound says "too close": measure exactly and name what is there.
            cl, _, _, pairs = masked_clearance(grid, samples[near], eff)
            exact[near] = np.minimum(cl, reach)
            offenders = classify(samples[near], pairs, r)
    else:
        cl, _, _, pairs = masked_clearance(grid, samples, eff, exclude)
        exact = np.minimum(cl, reach)
        offenders = classify(samples, pairs, r)
    min_c = 0.0 if blocked_state is not None else max(0.0, float(exact.min()) - 0.5 * step)

    assumed = grid.assumed_at(samples)
    assumed_at = None
    if assumed.any():
        a = samples[int(np.argmax(assumed))]
        assumed_at = (float(a[0]), float(a[1]), float(a[2]))
    return LegCheck(min_clearance_m=min_c, blocked_state=blocked_state, blocked_at=blocked_at,
                    assumed_free_at=assumed_at, offenders=offenders, samples=samples)


def findings_of(leg: str, check: LegCheck, radius_m: float, *, risk_only: bool = False,
                what: str = "") -> list[Finding]:
    sev = Severity.risk if risk_only else Severity.blocking
    out: list[Finding] = []
    if check.blocked_state is not None:
        code = Code.obstacle_in_corridor if check.blocked_state == "occupied" else Code.unknown_space
        out.append(Finding(code=code, severity=sev, leg=leg, at=check.blocked_at,
                           detail=f"{what}the path enters {check.blocked_state} space"))
    names = {Code.unknown_space: "unknown space (counts as occupied)",
             Code.overhang_above: "something overhead", Code.obstacle_in_corridor: "an obstacle"}
    for code, at in check.offenders.items():
        if any(f.code is code for f in out):
            continue
        out.append(Finding(code=code, severity=sev, leg=leg,
                           at=at, detail=f"{what}{names[code]} within the {radius_m:.2f} m "
                                         f"corridor radius (clearance {check.min_clearance_m:.2f} m)"))
    if check.assumed_free_at is not None and not out:
        out.append(Finding(code=Code.assumed_free_air, severity=Severity.risk, leg=leg,
                           at=check.assumed_free_at,
                           detail="the leg relies on free air that is assumed, not observed "
                                  "(occupancy skyline / gap fill)"))
    return out


def column_ceiling(grid: OccupancyGrid, xy: F64, radius_m: float) -> tuple[float, Optional[Vec3]]:
    """Top face of the highest blocked voxel in every column within `radius_m` of the given
    (N, 2) points; inf when some such column is blocked to the sky (or outside the grid)."""
    s = grid.spec
    r = s.res_m
    w = int(math.ceil(radius_m / r)) + 1
    di, dj = np.meshgrid(np.arange(-w, w + 1), np.arange(-w, w + 1), indexing="ij")
    disk = (np.maximum(np.abs(di) - 1, 0) * r) ** 2 + (np.maximum(np.abs(dj) - 1, 0) * r) ** 2 \
        <= radius_m ** 2
    di, dj = di[disk], dj[disk]
    i, j, _ = s.index_of(np.column_stack([xy, np.full(len(xy), s.z0)]))
    cols = np.unique(np.column_stack([(i[:, None] + di[None]).ravel(),
                                      (j[:, None] + dj[None]).ravel()]), axis=0)
    inside = (cols[:, 0] >= 0) & (cols[:, 0] < s.rows) & (cols[:, 1] >= 0) & (cols[:, 1] < s.cols)
    if not inside.all():
        c = cols[~inside][0]
        at = s.centre_of(c[0], c[1], 0)
        return math.inf, (float(at[0]), float(at[1]), float(s.z1))
    blocked = grid.state[cols[:, 0], cols[:, 1], :] != FREE
    if blocked[:, -1].any():
        c = cols[int(np.argmax(blocked[:, -1]))]
        at = s.centre_of(c[0], c[1], s.layers - 1)
        return math.inf, (float(at[0]), float(at[1]), float(at[2]))
    any_b = blocked.any(axis=1)
    if not any_b.any():
        return s.z0, None
    top = s.layers - 1 - np.argmax(blocked[:, ::-1], axis=1)
    top = np.where(any_b, top, -1)
    m = int(np.argmax(top))
    at = s.centre_of(cols[m, 0], cols[m, 1], top[m])
    return float(s.z0 + (top[m] + 1) * r), (float(at[0]), float(at[1]), float(at[2]))
