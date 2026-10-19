"""Takeoff spots near a target, from the occupancy grid (step 8.1b, ACCESS_POINTS §7.2).

Home defaults to the Site origin: the centre of the capture, which is usually next to — or on —
the target building (dataset_1910: beside the tower's roof). Finding a usable home by hand took
several scripts there. A spot qualifies when it is

1. **ground**: a measured column (POINTS / CELL), outside every building footprint, within
   `ground_tol_m` of the semantics DTM — not a roof, a deck or a parked truck;
2. **open within the takeoff corridor**: no blocked voxel (occupied, unknown, or outside the
   grid) higher than its surface plus the relief, closer than the corridor radius
   r = r_drone + m + k·hypot(σ_georef, σ_GNSS) — the planner's own radius and its own rule for
   the surface a leg starts on. Unknown and assumed columns block like everything else;
3. **a clear vertical column** when checked exactly the way the planner will check it: the home
   tests of `check_home`, the takeoff leg up to the transit height towards the target, and the
   transit itself, each with `check_leg` — so a proposed home never fails the plan's home or
   takeoff checks. "The target" is where the planner's transit ends (`transit_point`): for an
   opening, E of the direction the planner's own cone search takes, which depends on nothing
   home-specific and so is found once — tilted when the direct approach is blocked or comes
   too close to the grid's edge. Where no home could be planned (the cone leaves the grid),
   `transit_point` refuses as the planner would, before any column is checked.

(2) is evaluated for every column at once on the 2-D column tops; (3) only for the best
candidates, nearest the target first, at least `spacing_m` apart, so the list names distinct
places rather than adjacent voxels of one car park. With no candidate, the search says why —
how many columns failed which test — and names the most open ground it saw and the σ_GNSS
that would let it qualify. It never relaxes a criterion on its own.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from aerosplat.authored.models import AccessPoint
from aerosplat.mission.corridor import SQRT3, Plane, check_leg
from aerosplat.mission.models import (MissionTarget, PlanningProfile, TakeoffCandidate,
                                      TakeoffSearch)
from aerosplat.mission.planner import (ConeChoice, check_home, corridor_radii, opening_approach,
                                       require_inside_grid, transit_altitude)
from aerosplat.semantics.models import SEMANTICS_JSON, TERRAIN_NPZ, SemanticsResult, TerrainModel
from aerosplat.surface.models import FREE, ColumnSource
from aerosplat.surface.occupancy import OccupancyGrid

F64 = NDArray[np.float64]
MEASURED = (int(ColumnSource.POINTS), int(ColumnSource.CELL))
# How far around a spot its open ground is measured, beyond the corridor radius.
OPEN_REACH_M = 10.0
# The order in which columns are rejected; each column counts under its first failed test.
REASONS = ("near_grid_edge", "not_measured", "building", "not_ground", "obstacle_within_radius",
           "home_check", "transit_blocked", "transit_too_high", "takeoff_leg", "transit_leg",
           "not_checked")


@dataclass(frozen=True)
class Ground:
    """What says where the ground is: the semantics terrain raster of the same run."""

    terrain: TerrainModel
    dtm: F64                                # (rows, cols)
    labels: Optional[NDArray[np.integer]]   # building map (0 = none); None: not available

    def cells_of(self, x: F64, y: F64) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.bool_]]:
        t = self.terrain
        ti = np.floor((y - t.y0) / t.res_m).astype(np.int64)
        tj = np.floor((x - t.x0) / t.res_m).astype(np.int64)
        inside = (ti >= 0) & (ti < t.rows) & (tj >= 0) & (tj < t.cols)
        return np.clip(ti, 0, t.rows - 1), np.clip(tj, 0, t.cols - 1), inside


def load_ground(semantics_dir: Path) -> Ground:
    sem_json, npz = Path(semantics_dir) / SEMANTICS_JSON, Path(semantics_dir) / TERRAIN_NPZ
    if not sem_json.is_file() or not npz.is_file():
        raise FileNotFoundError(f"{semantics_dir}: no {SEMANTICS_JSON} / {TERRAIN_NPZ} — the "
                                f"takeoff search needs the semantics terrain (DTM, buildings)")
    result = SemanticsResult(**json.loads(sem_json.read_text(encoding="utf-8")))
    with np.load(npz) as z:
        labels = z["labels"] if "labels" in z.files else None
        return Ground(terrain=result.terrain, dtm=z["dtm"].astype(np.float64), labels=labels)


def corridor_radius(profile: PlanningProfile, sigma_georef_m: float) -> tuple[float, float]:
    """(r, σ_transit): the planner's radius for takeoff, transit and landing."""
    sigma = math.hypot(sigma_georef_m, profile.sigma_gnss_m)
    return profile.r_drone_m + profile.transit_margin_m + profile.k_sigma * sigma, sigma


def _verify(grid: OccupancyGrid, x: float, y: float, z: float, target: F64,
            target_z: Optional[float], radius: float, relief: float,
            profile: PlanningProfile) -> tuple[Optional[str], float, list[str]]:
    """(failed test or None, transit height, notes): the plan's own checks for this home."""
    r = grid.res_m
    _, home_f = check_home(grid, x, y, z, relief)
    if home_f:
        return "home_check", math.nan, [f.code.value for f in home_f]
    z_t, _ = transit_altitude(grid, (x, y), target, radius)
    if not math.isfinite(z_t):
        return "transit_blocked", z_t, []
    if target_z is not None:
        z_t = max(z_t, target_z)                  # as the planner: never below E
    if z_t - z > profile.max_alt_rel_m:
        return "transit_too_high", z_t, []
    plane = Plane(point=(x, y, z), normal=(0.0, 0.0, 1.0), offset_m=relief, name="home")
    takeoff = check_leg(grid, [(x, y, z + relief + r), (x, y, z_t)], radius, plane)
    if takeoff.blocked_state is not None or takeoff.offenders or takeoff.assumed_free_at:
        return "takeoff_leg", z_t, []
    notes: list[str] = []
    transit = check_leg(grid, [(x, y, z_t), (float(target[0]), float(target[1]), z_t)], radius)
    if transit.blocked_state is not None or transit.offenders:
        return "transit_leg", z_t, []
    if transit.assumed_free_at is not None:
        notes.append("the transit towards the target relies on assumed free air (a risk the "
                     "plan will name)")
    return None, z_t, notes


@dataclass(frozen=True)
class TransitEnd:
    """Where the transit to a target ends, as the planner flies it."""

    point: F64
    note: str = ""          # how the planner's approach put it there ("": the direct approach)


def transit_point(target: MissionTarget, ap: AccessPoint, grid: OccupancyGrid,
                  profile: Optional[PlanningProfile] = None) -> TransitEnd:
    """E of the approach the planner takes to an opening, or the pad itself.

    For an opening that is the planner's own cone search (`planner.opening_approach`): it
    depends on nothing home-specific, so it runs once here. E is `entry_distance` in front of
    the opening when the direct approach is clear and fits the grid, else on the direction
    the planner tilts to. Where the planner would refuse every home (the target, or E and H of
    every direction, too close to the grid's edge), this refuses the same way, before any home
    is checked: no home can help there, only a bigger grid.

    A pad's E is above it, at the transit height; the planner tilts it only when the direct
    approach is blocked or falls short of the grid's edge, and then by a transit height that
    depends on home. The search does not model that tilt: it checks the transit to the pad
    itself, and when the pad's direct approach falls short of the edge it refuses up front
    (outside_grid, naming why) instead of failing every home's transit check.
    """
    prof = profile or PlanningProfile()
    if ap.geometry is not None:
        ch = opening_approach(target, ap, grid, prof)
        return TransitEnd(ch.E, _approach_note(ch, target.handover.approach_cone_deg))
    if ap.pad is None:
        raise ValueError(f"access point {ap.id} has neither a geometry nor a pad")
    c = np.asarray(ap.pad.center, float)
    _, r_app, r_tr = corridor_radii(target, prof)
    require_inside_grid(grid, [("target", c, r_app)])
    require_inside_grid(grid, [("entry E and hover H above the pad", c, max(r_tr, r_app))], note=(
        "; the takeoff search checks the transit to the pad's direct approach, since a tilted "
        "one depends on each home's transit height (plan with a given home_enu to let the "
        "planner try the cone)"))
    return TransitEnd(c)


def _approach_note(ch: ConeChoice, cone_deg: float) -> str:
    cone = f"{cone_deg:g}° cone"
    if not ch.clear:
        shown = "the direct approach" if ch.tilt == 0.0 else f"the least tilted ({ch.tilt:.1f}°)"
        return (f"the approach is blocked in every direction of the {cone} that fits the grid, "
                f"so a plan to this target is infeasible from any home; the transit is checked "
                f"to E of {shown}, the one the plan shows")
    if ch.tilt == 0.0:
        return ""
    why = ("the direct approach is blocked" if ch.direct_fits else
           "the direct approach comes closer to the grid's edge than r + 2 voxels")
    return (f"the transit ends at E of the planner's approach, tilted {ch.tilt:.1f}° in the "
            f"{cone}: {why}")


def find_takeoff_spots(grid: OccupancyGrid, ground: Ground, target, *,
                       sigma_georef_m: float, profile: Optional[PlanningProfile] = None,
                       max_distance_m: float = 150.0, n: int = 5,
                       spacing_m: Optional[float] = None, ground_tol_m: float = 1.0,
                       max_checks: int = 40, approach_note: str = "") -> TakeoffSearch:
    """Ranked takeoff spots within `max_distance_m` of `target`, where the transit ends (see
    the module docstring). `approach_note`: how the planner's approach put it there
    (`transit_point`), carried into the result and its message."""
    prof = profile or PlanningProfile()
    tgt = np.asarray(target, float).reshape(-1)
    if tgt.shape not in ((2,), (3,)) or not np.isfinite(tgt).all():
        raise ValueError(f"target must be a finite (x, y[, z]) point, got {target!r}")
    for name, v in (("max_distance_m", max_distance_m), ("ground_tol_m", ground_tol_m),
                    ("sigma_georef_m", sigma_georef_m)):
        if not (math.isfinite(v) and v >= 0):
            raise ValueError(f"{name} must be >= 0, got {v}")
    if n < 1 or max_checks < 1:
        raise ValueError("n and max_checks must be >= 1")
    s = grid.spec
    r = s.res_m
    radius, sigma_t = corridor_radius(prof, sigma_georef_m)
    relief = prof.surface_relief_m + 0.5 * SQRT3 * r          # as the planner's home plane
    need = radius + 0.25 * r                                  # check_leg: radius + step/2
    spacing = max(2.0 * radius, 10.0) if spacing_m is None else float(spacing_m)
    edge = radius + 2.0 * r                                   # planner.require_inside_grid
    rejected = {k: 0 for k in REASONS}
    tx, ty = float(tgt[0]), float(tgt[1])
    target3 = np.array([tx, ty, float(tgt[2]) if len(tgt) == 3 else 0.0])

    # Columns within reach, plus a margin for the open-ground measure.
    reach = need + OPEN_REACH_M + r
    i0, i1 = (max(0, int(math.floor((ty - max_distance_m - reach - s.y0) / r))),
              min(s.rows, int(math.ceil((ty + max_distance_m + reach - s.y0) / r))))
    j0, j1 = (max(0, int(math.floor((tx - max_distance_m - reach - s.x0) / r))),
              min(s.cols, int(math.ceil((tx + max_distance_m + reach - s.x0) / r))))
    grid_limited = (tx - max_distance_m < s.x0 + edge or tx + max_distance_m > s.x1 - edge
                    or ty - max_distance_m < s.y0 + edge or ty + max_distance_m > s.y1 - edge)
    base = dict(target=(tx, ty, float(target3[2])), radius_m=radius, relief_m=relief,
                sigma_transit_m=sigma_t, max_distance_m=max_distance_m, spacing_m=spacing,
                ground_tol_m=ground_tol_m, grid_limited=grid_limited, approach_note=approach_note)
    if i0 >= i1 or j0 >= j1:
        return TakeoffSearch(**base, searched_columns=0, rejected=rejected, checked=0,
                             candidates=[], message=_with_note(
                                 f"the target ({tx:.1f}, {ty:.1f}) is not within "
                                 f"{max_distance_m:g} m of the grid", approach_note))

    ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
    xc = s.x0 + (jj + 0.5) * r
    yc = s.y0 + (ii + 0.5) * r
    dist = np.hypot(xc - tx, yc - ty)
    in_disk = dist <= max_distance_m
    searched = int(in_disk.sum())

    # Highest blocked layer per column (-1: none; layers - 1: blocked to the sky).
    blocked = grid.state[i0:i1, j0:j1, :] != FREE
    any_b = blocked.any(axis=2)
    k_top = np.where(any_b, s.layers - 1 - np.argmax(blocked[:, :, ::-1], axis=2), -1)
    del blocked

    z = grid.column_top[i0:i1, j0:j1].astype(np.float64)
    src = grid.column_source[i0:i1, j0:j1]
    margin = np.minimum.reduce([xc - s.x0, s.x1 - xc, yc - s.y0, s.y1 - yc])
    ti, tj, on_raster = ground.cells_of(xc, yc)
    dtm = np.where(on_raster, ground.dtm[ti, tj], np.nan)
    in_building = (np.zeros_like(in_disk) if ground.labels is None
                   else on_raster & (ground.labels[ti, tj] > 0))

    alive = in_disk.copy()
    for reason, bad in (("near_grid_edge", margin < edge),
                        ("not_measured", ~np.isin(src, MEASURED) | ~np.isfinite(z)),
                        ("building", in_building),
                        ("not_ground", ~(np.abs(z - dtm) <= ground_tol_m))):
        hit = alive & bad
        rejected[reason] = int(hit.sum())
        alive &= ~hit

    # Open ground around each surviving column: distance to the nearest column that holds a
    # blocked voxel above the column's surface + relief (first layer with its centre higher).
    k_need = np.floor((np.where(alive, z, 0.0) + relief - s.z0) / r - 0.5).astype(np.int64) + 1
    open_ub = np.zeros(z.shape)           # upper bound: nearest centre - r/2
    open_lb = np.zeros(z.shape)           # lower bound: nearest centre - r·√2/2
    cap = need + OPEN_REACH_M
    # Beyond the grid is OUTSIDE, which blocks: a ring of obstacles where the window meets it.
    ring = ((int(i0 == 0), int(i1 == s.rows)), (int(j0 == 0), int(j1 == s.cols)))
    for k in np.unique(k_need[alive]):
        sel = alive & (k_need == k)
        obstacle = np.pad(k_top >= k, ring, constant_values=True)
        if not obstacle.any():
            open_ub[sel] = open_lb[sel] = cap
            continue
        d = ndimage.distance_transform_edt(~obstacle)[ring[0][0]:obstacle.shape[0] - ring[0][1],
                                                       ring[1][0]:obstacle.shape[1] - ring[1][1]] * r
        open_ub[sel] = d[sel] - 0.5 * r
        open_lb[sel] = d[sel] - 0.5 * math.sqrt(2.0) * r
    open_ub = np.minimum(open_ub, cap)
    open_lb = np.minimum(open_lb, cap)

    def candidate(idx: tuple[int, int], rank: int, z_t: Optional[float],
                  notes: list[str]) -> TakeoffCandidate:
        a, b = idx
        return TakeoffCandidate(
            rank=rank, enu=(float(xc[a, b]), float(yc[a, b]), float(z[a, b])),
            distance_m=float(dist[a, b]), open_radius_m=round(float(open_ub[a, b]), 2),
            transit_z=None if z_t is None else float(z_t),
            climb_m=None if z_t is None else float(z_t - z[a, b]),
            above_dtm_m=float(z[a, b] - dtm[a, b]), notes=notes)

    hit = alive & (open_ub < need)
    rejected["obstacle_within_radius"] = int(hit.sum())
    too_tight = hit                       # ground that only lacks room: the near misses
    alive &= ~hit

    # Sure passes of the 2-D test first (lower bound >= need), then the marginal ones: both
    # nearest the target first. Exact checks decide; the bounds only order the work.
    sure = alive & (open_lb >= need)
    order = [np.argwhere(m) for m in (sure, alive & ~sure)]
    picks = [o[np.argsort(dist[o[:, 0], o[:, 1]], kind="stable")] for o in order if len(o)]
    queue = np.concatenate(picks) if picks else np.zeros((0, 2), np.int64)

    found: list[TakeoffCandidate] = []
    checked = 0
    for pos, (a, b) in enumerate(queue):
        if len(found) >= n:
            break
        p = np.array([xc[a, b], yc[a, b]])
        if any(math.hypot(p[0] - c.enu[0], p[1] - c.enu[1]) < spacing for c in found):
            continue
        if checked >= max_checks:
            rejected["not_checked"] = len(queue) - pos
            break
        checked += 1
        why, z_t, notes = _verify(grid, float(p[0]), float(p[1]), float(z[a, b]), target3,
                                  float(tgt[2]) if len(tgt) == 3 else None, radius, relief, prof)
        if why is None:
            found.append(candidate((int(a), int(b)), len(found) + 1, z_t, notes))
        else:
            rejected[why] += 1

    best_open = None
    sigma_fit = None
    if not found and too_tight.any() and not rejected["not_checked"]:
        # Only ground that failed for lack of room: a σ that buys room is then the answer.
        # (An unfinished search names no near miss: the answer may be among the unchecked.)
        a, b = (int(v) for v in np.unravel_index(np.argmax(np.where(too_tight, open_ub, -1.0)),
                                                 open_ub.shape))
        best_open = candidate((a, b), 1, None, ["does not qualify: shown for comparison"])
        # r_drone + m + k·hypot(σ_georef, σ_GNSS) + r/4 <= open radius
        room = (float(open_lb[a, b]) - 0.25 * r - prof.r_drone_m - prof.transit_margin_m)
        if prof.k_sigma > 0 and room > 0:
            s_t = room / prof.k_sigma
            if s_t > sigma_georef_m:
                sigma_fit = math.sqrt(s_t ** 2 - sigma_georef_m ** 2)
    return TakeoffSearch(**base, searched_columns=searched, rejected=rejected, checked=checked,
                         candidates=found, best_open=best_open, sigma_gnss_would_fit_m=sigma_fit,
                         message=_with_note(_message(found, base, searched, rejected, prof,
                                                     sigma_georef_m, best_open, sigma_fit),
                                            approach_note))


def _with_note(text: str, note: str) -> str:
    return f"{text}. {note[0].upper()}{note[1:]}" if note else text


def _message(found, base, searched, rejected, prof, sigma_georef_m, best_open, sigma_fit) -> str:
    where = (f"within {base['max_distance_m']:g} m of the target"
             + (" (cut by the grid's edge)" if base["grid_limited"] else ""))
    r = base["radius_m"]
    profile = (f"r = {r:.2f} m (r_drone {prof.r_drone_m:g} + m {prof.transit_margin_m:g} + "
               f"{prof.k_sigma:g}·σ {base['sigma_transit_m']:.2f}; σ_GNSS {prof.sigma_gnss_m:g}, "
               f"σ_georef {sigma_georef_m:g})")
    if found:
        c = found[0]
        return (f"{len(found)} takeoff spot(s) {where} for {profile}; nearest at "
                f"({c.enu[0]:.1f}, {c.enu[1]:.1f}), {c.distance_m:.0f} m from the target, "
                f"≈{c.open_radius_m:.1f} m of open ground")
    why = ", ".join(f"{k} {v}" for k, v in rejected.items() if v)
    if rejected["not_checked"]:
        # The budget of exact checks ran out first: that is not "there is none".
        return (f"no takeoff spot confirmed {where} for {profile}: the exact checks stopped "
                f"after {sum(rejected[k] for k in REASONS[5:-1])} failures with "
                f"{rejected['not_checked']} candidates left unchecked — raise max_checks to "
                f"settle it ({why})")
    text = (f"no takeoff spot {where} for {profile}: of {searched} columns, {why or 'none qualified'}")
    if best_open is not None:
        text += (f". The most open ground is at ({best_open.enu[0]:.1f}, {best_open.enu[1]:.1f}), "
                 f"{best_open.distance_m:.0f} m from the target, with ≈{best_open.open_radius_m:.1f} m "
                 f"clear")
        text += (f"; it would qualify with σ_GNSS ≤ {sigma_fit:.2f} m (if nothing else fails)"
                 if sigma_fit is not None else
                 "; no σ_GNSS would make it qualify with this r_drone and margin")
    return text


# ── command line ─────────────────────────────────────────────────────────
def main(argv: Optional[list[str]] = None) -> int:
    from aerosplat.logging_utils import fix_console_encoding
    from aerosplat.surface.occupancy import read_occupancy

    fix_console_encoding()
    ap = argparse.ArgumentParser(description="Takeoff spots near a target, from a run's occupancy")
    ap.add_argument("--occupancy", type=Path, required=True, help="occupancy/ of a run")
    ap.add_argument("--semantics", type=Path, default=None,
                    help="semantics/ of the same run (default: the one occupancy was built from)")
    ap.add_argument("--target", required=True, help="x,y[,z] in Site ENU, m")
    ap.add_argument("--sigma-georef", type=float, required=True, dest="sigma_georef_m")
    ap.add_argument("--sigma-gnss", type=float, default=None, help="default: the profile's")
    ap.add_argument("--max-distance", type=float, default=150.0, dest="max_distance_m")
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--json", action="store_true", help="print the whole result as JSON")
    args = ap.parse_args(argv)

    grid, report = read_occupancy(args.occupancy)
    sem = args.semantics or Path(str(report.inputs.get("semantics_dir") or args.occupancy.parent
                                     / "semantics"))
    prof = PlanningProfile() if args.sigma_gnss is None else PlanningProfile(
        sigma_gnss_m=args.sigma_gnss)
    target = [float(v) for v in args.target.split(",")]
    res = find_takeoff_spots(grid, load_ground(sem), target, sigma_georef_m=args.sigma_georef_m,
                             profile=prof, max_distance_m=args.max_distance_m, n=args.n)
    if args.json:
        print(res.model_dump_json(indent=2))
        return 0 if res.candidates else 1
    print(res.message)
    for c in res.candidates:
        print(f"  #{c.rank}: ({c.enu[0]:.1f}, {c.enu[1]:.1f}, {c.enu[2]:.1f})  "
              f"{c.distance_m:.0f} m from the target, ≈{c.open_radius_m:.1f} m open, "
              f"climb {c.climb_m:.1f} m" + ("; " + "; ".join(c.notes) if c.notes else ""))
    return 0 if res.candidates else 1


if __name__ == "__main__":
    raise SystemExit(main())
