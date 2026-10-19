"""Which facade of one run is which facade of another (ACCESS_POINTS §5.9).

Facade ids are per run: the semantics stage numbers walls from scratch every time, so the
tower wall that run 1 called `B08/SW3` is `B04/SW1` in run 2, and `B04/SW1` of run 1 may be
a different building altogether. Authored features keep the frame they were drawn in
(`HostFacade`), and this module is what relates such a frame to the walls of another run.

Everything is geometry in Site ENU, which every run of a Site shares (ADR-0006), so no
re-anchoring transform is needed between runs; what differs is the fit. Two facades are
the same wall when:

* their outward normals agree (`max_angle_deg`),
* each plane passes through the other where they overlap (`offset_m` within a tolerance
  that grows with the planes' own scatter, `plane_rms_m`), and
* they share a stretch along the wall and a band of height.

The answer is never a silent guess (internal reliability requirements §3.1). Per facade of run A:

* `match`     — one facade of B, with the relation spelled out (`same`, A `inside` a longer
                B that merged several walls, B a `part` of A, or a `partial` overlap);
* `split`     — several facades of B, side by side, each covering its own stretch of A;
* `ambiguous` — two or more facades of B cover the *same* stretch of A (parallel planes
                within tolerance, a duplicate fit): nothing is chosen;
* `none`      — no facade of B lies on that wall.

Matching never moves or rewrites anything; carrying a feature over is a separate, explicit
step (`aerosplat.authored.carry`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np

from aerosplat.semantics.models import Facade

VERSION = "1.0.0"

MatchStatus = Literal["match", "split", "ambiguous", "none"]
Relation = Literal["same", "inside", "part", "partial"]


@dataclass(frozen=True)
class MatchParams:
    # Normals further apart than this are different walls. Fitted normals of one wall in
    # two runs of dataset_1910 differ by 0.3–2.3°.
    max_angle_deg: float = 10.0
    # Plane offset tolerance: `rms_k` × the two planes' scatter combined, within the bounds.
    # The scatter is real (0.6 m median on dataset_1910, an Earth Studio mesh), and two fits
    # of one wall a metre apart are normal; two walls closer than this are not told apart
    # and come back `ambiguous` when they overlap.
    rms_k: float = 2.0
    min_offset_tol_m: float = 1.0
    max_offset_tol_m: float = 3.0
    # Shared stretch along the wall: at least this many metres, or this share of the
    # shorter of the two facades if that is shorter still.
    min_overlap_m: float = 2.0
    min_overlap_frac: float = 0.5
    # Shared height, as a share of the lower of the two facades.
    min_v_overlap_frac: float = 0.3
    # Two candidates of B compete for A when they cover the same part of it by this
    # share of the shorter of their two stretches.
    same_stretch_frac: float = 0.3
    # Relation labels: a facade covered by at least this share counts as covered.
    covered_frac: float = 0.5

    def offset_tol(self, rms_a: float, rms_b: float) -> float:
        return float(np.clip(self.rms_k * math.hypot(rms_a, rms_b),
                             self.min_offset_tol_m, self.max_offset_tol_m))


@dataclass(frozen=True)
class PairScore:
    """How facade `b` of the other run lies against facade `a`."""

    b_id: str
    angle_deg: float
    offset_m: float               # plane-to-plane distance where they overlap
    offset_tol_m: float
    overlap_m: float              # shared stretch along the wall
    overlap_a: float              # ... as a share of a's length
    overlap_b: float              # ... as a share of b's length
    overlap_v: float              # shared height, as a share of the lower facade
    span_a: tuple[float, float]   # the shared stretch, in a's u
    score: float                  # 0..1: overlap_a, discounted by angle and offset


@dataclass
class FacadeMatch:
    a_id: str
    status: MatchStatus
    b_ids: list[str] = field(default_factory=list)       # along a's u for a split
    relation: Optional[Relation] = None                  # for a single `match`
    confidence: float = 0.0
    candidates: list[PairScore] = field(default_factory=list)
    reason: str = ""
    # Other facades of run A that matched the same facade of B: B is one wall where A had
    # several (set by `match_facades`, which sees all of A).
    shared_with: list[str] = field(default_factory=list)


def _span(a: Facade, b: Facade) -> tuple[float, float]:
    """b's extent along the wall, in a's u."""
    z = 0.5 * (b.v_ground + b.v_max)
    ends = np.array([b.point(b.u_min, z), b.point(b.u_max, z)])
    u, _ = a.uv_of(ends)
    return float(min(u)), float(max(u))


def pair(a: Facade, b: Facade, params: MatchParams = MatchParams()) -> Optional[PairScore]:
    """Score `b` against `a`, or None when they are not the same wall."""
    na, nb = np.asarray(a.normal, float), np.asarray(b.normal, float)
    angle = math.degrees(math.acos(float(np.clip(np.dot(na, nb), -1.0, 1.0))))
    if angle > params.max_angle_deg:
        return None
    lo_b, hi_b = _span(a, b)
    lo, hi = max(a.u_min, lo_b), min(a.u_max, hi_b)
    overlap = hi - lo
    need = min(params.min_overlap_m, params.min_overlap_frac * min(a.length, b.length))
    if overlap <= 0 or overlap < need:
        return None
    v_lo, v_hi = max(a.v_ground, b.v_ground), min(a.v_max, b.v_max)
    lower = max(1e-6, min(a.height, b.height))
    overlap_v = max(0.0, v_hi - v_lo) / lower
    if overlap_v < params.min_v_overlap_frac:
        return None
    # Offset where they overlap, measured both ways: from the middle of the shared stretch
    # on a to b's plane, and from the same spot on b back to a's plane.
    z_mid = 0.5 * (v_lo + v_hi)
    p_a = a.point(0.5 * (lo + hi), z_mid)
    u_b, _ = b.uv_of(p_a)
    p_b = b.point(float(u_b), z_mid)
    offset = max(abs(float(b.distance_to_plane(p_a))), abs(float(a.distance_to_plane(p_b))))
    tol = params.offset_tol(a.plane_rms_m, b.plane_rms_m)
    if offset > tol:
        return None
    overlap_a = overlap / max(a.length, 1e-6)
    overlap_b = overlap / max(b.length, 1e-6)
    score = (min(1.0, overlap_a) * min(1.0, overlap_v)
             * (1.0 - 0.5 * angle / params.max_angle_deg) * (1.0 - 0.5 * offset / tol))
    return PairScore(b_id=b.id, angle_deg=angle, offset_m=offset, offset_tol_m=tol,
                     overlap_m=overlap, overlap_a=overlap_a, overlap_b=overlap_b,
                     overlap_v=overlap_v, span_a=(lo, hi), score=float(score))


def _compete(p: PairScore, q: PairScore, frac: float) -> bool:
    shared = min(p.span_a[1], q.span_a[1]) - max(p.span_a[0], q.span_a[0])
    shorter = min(p.span_a[1] - p.span_a[0], q.span_a[1] - q.span_a[0])
    return shared > frac * shorter


def _relation(c: PairScore, covered: float) -> Relation:
    if c.overlap_a >= covered and c.overlap_b >= covered:
        return "same"
    if c.overlap_a >= covered:
        return "inside"
    if c.overlap_b >= covered:
        return "part"
    return "partial"


RELATION_UK = {
    "same": "та сама стіна",
    "inside": "фасад — частина довшого фасаду іншого run-а (там стіни злито)",
    "part": "інший run бачить лише частину цього фасаду",
    "partial": "фасади перекриваються лише частково",
}


def match_facade(a: Facade, others: Sequence[Facade],
                 params: MatchParams = MatchParams()) -> FacadeMatch:
    """Where facade `a` is among `others` (the facades of another run)."""
    cands = [s for s in (pair(a, b, params) for b in others) if s is not None]
    cands.sort(key=lambda s: (-s.score, s.b_id))
    if not cands:
        return FacadeMatch(a_id=a.id, status="none",
                           reason=f"жоден фасад іншого run-а не лежить на цій стіні "
                                  f"(кут ≤ {params.max_angle_deg:g}°, зсув площини ≤ "
                                  f"{params.offset_tol(a.plane_rms_m, a.plane_rms_m):.1f} м, "
                                  f"спільна ділянка ≥ {params.min_overlap_m:g} м)")
    clashes = [(p, q) for i, p in enumerate(cands) for q in cands[i + 1:]
               if _compete(p, q, params.same_stretch_frac)]
    if clashes:
        ids = sorted({s.b_id for pq in clashes for s in pq})
        p, q = clashes[0]
        return FacadeMatch(
            a_id=a.id, status="ambiguous", b_ids=ids, candidates=cands,
            reason=f"{', '.join(ids)} покривають ту саму ділянку стіни "
                   f"(напр. {p.b_id}: зсув {p.offset_m:.2f} м, {q.b_id}: {q.offset_m:.2f} м) — "
                   f"не вибираємо")
    if len(cands) == 1:
        c = cands[0]
        rel = _relation(c, params.covered_frac)
        return FacadeMatch(
            a_id=a.id, status="match", b_ids=[c.b_id], relation=rel, confidence=c.score,
            candidates=cands,
            reason=f"{RELATION_UK[rel]}: кут {c.angle_deg:.1f}°, зсув {c.offset_m:.2f} м, "
                   f"спільна ділянка {c.overlap_m:.1f} м ({100 * c.overlap_a:.0f}% цього фасаду)")
    along = sorted(cands, key=lambda s: s.span_a[0])
    covered = min(1.0, sum(s.overlap_a for s in along))
    geo = float(np.mean([s.score / max(min(1.0, s.overlap_a), 1e-6) for s in along]))
    return FacadeMatch(
        a_id=a.id, status="split", b_ids=[s.b_id for s in along], confidence=covered * geo,
        candidates=along,
        reason=f"стіну поділено між {len(along)} фасадами іншого run-а "
               f"({', '.join(s.b_id for s in along)}), разом {100 * covered:.0f}% її довжини; "
               f"кожну точку доступу треба переносити окремо")


def match_facades(a_facades: Sequence[Facade], b_facades: Sequence[Facade],
                  params: MatchParams = MatchParams()) -> list[FacadeMatch]:
    """`match_facade` for every facade of run A against all facades of run B.

    Building ids are deliberately ignored: they are renumbered per run, and a building of
    one run can be two of another. A vectorised prefilter (normal angle, distance between
    the walls' middles) keeps a site of ~2000 × ~1300 facades well under a second; it only
    drops pairs `pair` would reject anyway.
    """
    if not a_facades or not b_facades:
        return [match_facade(a, [], params) for a in a_facades]
    nb = np.array([b.normal for b in b_facades], float)
    mid_b = np.array([b.point(0.5 * (b.u_min + b.u_max), b.v_ground) for b in b_facades])[:, :2]
    half_b = np.array([0.5 * b.length for b in b_facades])
    cos_max = math.cos(math.radians(params.max_angle_deg))
    # Middles of two walls that share a stretch are at most the two half-lengths apart along
    # the wall, plus the offset and what the angle adds across it.
    slack = 1.0 + math.sin(math.radians(params.max_angle_deg))
    out = []
    for a in a_facades:
        mid_a = a.point(0.5 * (a.u_min + a.u_max), a.v_ground)[:2]
        near = ((nb @ np.asarray(a.normal, float) >= cos_max - 1e-9)
                & (np.hypot(*(mid_b - mid_a).T)
                   <= (0.5 * a.length + half_b) * slack + params.max_offset_tol_m + 1.0))
        out.append(match_facade(a, [b_facades[i] for i in np.flatnonzero(near)], params))

    # A facade of B that several facades of A matched one-to-one is a merge of them: each of
    # those is `inside` it, whatever the overlap shares say.
    holders: dict[str, list[str]] = {}
    for m in out:
        if m.status == "match":
            holders.setdefault(m.b_ids[0], []).append(m.a_id)
    for m in out:
        if m.status == "match" and len(holders[m.b_ids[0]]) > 1:
            m.shared_with = [x for x in holders[m.b_ids[0]] if x != m.a_id]
            if m.relation == "same":
                m.relation = "inside"
                m.reason = (f"{RELATION_UK['inside']}, разом із {', '.join(m.shared_with)}"
                            + m.reason[len(RELATION_UK['same']):])
    return out
