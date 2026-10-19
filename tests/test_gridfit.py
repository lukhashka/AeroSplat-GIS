"""Window candidates from the periodic grid of a facade orthophoto (ROADMAP 7.3, step 1).

Every scene is synthetic with a known answer (`gridfit_scenes`): the windows are painted
at metric positions, so a fit can be checked to the centimetre, and every way the method
should *refuse* has a scene of its own. Negative cases outnumber the positive ones on
purpose: a wrong candidate costs a reviewer's attention at best and, missed, sends a
drone to a wall (ADR-0009).
"""

import numpy as np
import pytest

from aerosplat.authored.candidates import Proposal, make_candidates, overlap, rect_on, screen
from aerosplat.authored.geometry import make_access_point, rect_uv
from aerosplat.authored.models import Lifecycle, Origin
from aerosplat.semantics.gridfit import (MODEL, CoverageRaster, GridfitParams, PixelGrid,
                                         Refusal, agreement, find_period, gridfit, otsu)
from aerosplat.semantics.models import Facade, SurfaceKind, azimuth_of, u_axis_for
from gridfit_scenes import (FOLIAGE, GLASS, LIGHT_GLASS, NAVY, SHADOW, SLAB, Painter, centre,
                            facade, full_coverage, grid_windows, overlaps, scene)


def fit(sc, **params):
    return gridfit(sc.rgb, sc.alpha, sc.grid, params=GridfitParams(**params), **sc.kwargs())


def matched(result, windows, tol=0.1):
    """Every true window has a proposed cell whose centre is within `tol` in u and v."""
    cells = [((c.u0 + c.u1) / 2, (c.v0 + c.v1) / 2) for c in result.proposals]
    out = []
    for w in windows:
        cu, cv = centre(w)
        hit = [c for c in cells if abs(c[0] - cu) <= tol and abs(c[1] - cv) <= tol]
        out.append(bool(hit))
    return out


def cell_rect(c):
    return (c.u0, c.u1, c.v0, c.v1)


# ── positive: a regular grid comes back exactly ──────────────────────────
@pytest.mark.parametrize("px, floor_m, bay_m, win_w, win_h", [
    (0.08, 3.0, 2.5, 1.2, 1.5),        # the dataset_1910 ortho scale
    (0.05, 2.8, 3.2, 1.4, 1.6),        # finer, the editor default
    (0.10, 3.3, 1.8, 1.0, 1.7),        # coarser, narrow bays
])
def test_a_regular_grid_gives_exact_periods_and_centres(px, floor_m, bay_m, win_w, win_h):
    p, windows = facade(px=px, floor_m=floor_m, bay_m=bay_m, win_w=win_w, win_h=win_h,
                        width_m=34.0, height_m=24.0)
    r = fit(scene(p, windows))
    assert not r.refused, r.refusals
    assert r.bay_m == pytest.approx(bay_m, abs=0.02)
    assert r.floor.floor_m == pytest.approx(floor_m, abs=0.02)
    assert all(matched(r, windows, tol=0.1))
    # ... and nothing besides the windows: one proposal per window.
    assert len(r.proposals) == len(windows)
    for c in r.proposals:
        assert c.u1 - c.u0 == pytest.approx(win_w, abs=0.15)
        assert c.v1 - c.v0 == pytest.approx(win_h, abs=0.15)
        assert 0.05 <= c.confidence <= 0.95


def test_window_pairs_become_two_strips_per_bay():
    p, windows = facade(bay_m=4.0, win_w=0.9, pair_gap_m=0.5, width_m=36.0)
    r = fit(scene(p, windows, seed=3))
    assert not r.refused
    assert len(r.strips) == 2
    assert all(matched(r, windows)) and len(r.proposals) == len(windows)


def test_glass_reflecting_the_sky_is_still_found():
    # Light glass is barely darker than the wall; blueness carries it.
    p, windows = facade(glass=LIGHT_GLASS)
    r = fit(scene(p, windows, seed=4))
    assert not r.refused
    assert all(matched(r, windows))


def test_the_result_names_the_model_and_is_deterministic():
    p, windows = facade()
    sc = scene(p, windows)
    a, b = fit(sc), fit(sc)
    assert a.model == MODEL and MODEL.startswith("gridfit@")
    assert a.model_dump() == b.model_dump()


# ── negative: what must never become a candidate ─────────────────────────
def test_a_missing_window_is_not_proposed():
    p, windows = facade(skip=(7, 23, 40))
    all_windows = grid_windows(width_m=30.0, height_m=21.0, floor_m=3.0, bay_m=2.5,
                               win_w=1.2, win_h=1.5)
    missing = [w for i, w in enumerate(all_windows) if i in (7, 23, 40)]
    r = fit(scene(p, windows, seed=5))
    assert not r.refused
    assert all(matched(r, windows))
    for w in missing:
        assert not any(overlaps(cell_rect(c), w) for c in r.proposals)
        cell = next(c for c in r.cells if overlaps(cell_rect(c), w))
        assert cell.reason == "wall"


def test_trees_in_front_give_no_candidates_on_the_foliage():
    rng = np.random.default_rng(7)
    p, windows = facade()
    trees = [(7.0, 3.0, 4.5, 5.0), (22.0, 2.5, 3.5, 4.0)]
    for cu, cv, ru, rv in trees:
        p.ellipse(cu, cv, ru, rv, FOLIAGE, speckle=rng)
    r = fit(scene(p, windows, seed=8))
    assert not r.refused
    for c in r.proposals:
        uu, vv = np.meshgrid(np.linspace(c.u0, c.u1, 40), np.linspace(c.v0, c.v1, 40))
        for cu, cv, ru, rv in trees:
            # Not even a corner of a proposal is foliage: at most a 2 % sliver of it.
            inside = ((uu - cu) / ru) ** 2 + ((vv - cv) / rv) ** 2 <= 1.0
            assert inside.mean() <= 0.02, c
    assert any(c.reason == "vegetation" for c in r.cells)
    # The windows well clear of the trees are still there.
    clear = [w for w in windows if w[2] > 9.5]
    assert all(matched(r, clear))


def test_an_occluder_that_is_not_green_is_caught_by_depth():
    p, windows = facade()
    p.ellipse(15.0, 2.5, 4.0, 2.8, NAVY, front_m=6.0)
    van = (11.0, 19.0, -0.3, 5.3)
    r = fit(scene(p, windows, depth=True))
    assert not r.refused
    assert not any(overlaps(cell_rect(c), van) and abs((c.u0 + c.u1) / 2 - 15) < 3.5
                   and (c.v0 + c.v1) / 2 < 4.5 for c in r.proposals)
    assert any(c.reason == "occluded" for c in r.cells)


def test_a_non_periodic_wall_is_refused_not_boxed():
    p = Painter(30.0, 21.0, 0.08)
    r = fit(scene(p, []))
    assert r.refused and r.refusals[0].code == "no_columns"
    assert r.proposals == [] and r.cells == []
    assert "Колонки" in r.refusals[0].message


@pytest.mark.parametrize("seed", range(5))
def test_randomly_placed_openings_are_refused(seed):
    rng = np.random.default_rng(100 + seed)
    p = Painter(30.0, 21.0, 0.08)
    for _ in range(60):
        u0, v0 = rng.uniform(0, 28), rng.uniform(0, 19)
        p.paint((u0, u0 + rng.uniform(0.5, 2.0), v0, v0 + rng.uniform(0.5, 2.0)), GLASS)
    r = fit(scene(p, [], seed=seed))
    assert r.refused
    assert r.proposals == []


def test_balcony_shadows_disagreeing_with_the_glass_are_refused():
    # Each bay: a glass strip, then a stack of balconies whose shadow is near black.
    # Darkness finds the shadows, blueness the glass: two column structures.
    p = Painter(30.0, 21.0, 0.08)
    k = 0
    while k * 3.0 + 2.6 < 21.0:
        for m in range(10):
            u = m * 3.0
            p.paint((u + 0.6, u + 1.6, k * 3.0 + 0.9, k * 3.0 + 2.4), LIGHT_GLASS)
            p.paint((u + 1.9, u + 2.9, k * 3.0 + 0.2, k * 3.0 + 2.6), SHADOW)
        k += 1
    r = fit(scene(p, []))
    assert r.refused and r.refusals[0].code == "cues_disagree"
    assert r.agreement is not None and r.agreement < GridfitParams().min_agreement
    assert r.proposals == []
    assert "тінями балконів" in r.refusals[0].message


def test_a_cue_repeating_something_wider_than_a_window_disagrees():
    # A whole balcony stack reads dark (3.4 m), the glass beside it reads blue:
    # the dark stack would "contain" every pane, and must not count as agreeing.
    p = Painter(30.0, 21.0, 0.08)
    for m in range(6):
        u = m * 5.0
        p.paint((u + 0.4, u + 3.8, 0.0, 21.0), SHADOW)
        k = 0
        while k * 3.0 + 2.4 < 21.0:
            p.paint((u + 1.4, u + 2.6, k * 3.0 + 0.9, k * 3.0 + 2.4), LIGHT_GLASS)
            k += 1
    r = fit(scene(p, []))
    assert r.refused and r.refusals[0].code in ("cues_disagree", "no_strips")
    assert r.proposals == []


def test_no_candidates_in_uncovered_cells():
    hole = (15.0, 30.0, 10.5, 21.0)
    p, windows = facade()
    r = fit(scene(p, windows, coverage=full_coverage(30.0, 21.0, holes=(hole,))))
    assert not r.refused
    assert not any(overlaps(cell_rect(c), hole) for c in r.proposals)
    assert any(c.reason == "uncovered" for c in r.cells)
    covered = [w for w in windows if not overlaps(w, hole)]
    assert all(matched(r, covered))


def test_a_facade_with_no_coverage_at_all_is_refused():
    p, windows = facade()
    r = fit(scene(p, windows, coverage=full_coverage(30.0, 21.0, holes=((0, 30, 0, 21),))))
    assert r.refused and r.refusals[0].code == "no_coverage"
    assert r.proposals == []


def test_a_facade_that_was_not_rendered_is_refused():
    p, windows = facade()
    sc = scene(p, windows)
    sc.alpha[:] = 0.0
    sc.rgb[:] = 0
    r = fit(sc)
    assert r.refused and r.refusals[0].code == "not_rendered"


@pytest.mark.parametrize("floor_m, win_h, sill_m, height_m", [
    (5.4, 2.0, 0.9, 27.0),             # one storey of a warehouse, or two storeys as one
    (1.9, 0.9, 0.5, 21.0),             # tile courses, not storeys
])
def test_a_storey_height_outside_the_prior_is_refused(floor_m, win_h, sill_m, height_m):
    p, windows = facade(floor_m=floor_m, win_h=win_h, sill_m=sill_m, height_m=height_m)
    r = fit(scene(p, windows))
    assert r.refused and r.refusals[0].code == "floor_out_of_prior"
    assert "2.4–4.2" in r.refusals[0].message
    assert r.proposals == []


def test_horizontal_bands_without_columns_are_refused():
    p = Painter(30.0, 21.0, 0.08)
    for k in range(7):
        p.paint((0.0, 30.0, k * 3.0 + 0.9, k * 3.0 + 2.4), GLASS)
    r = fit(scene(p, []))
    assert r.refused and r.refusals[0].code == "no_columns"


def test_full_height_glazing_without_storeys_is_refused():
    p = Painter(30.0, 21.0, 0.08)
    for m in range(12):
        p.paint((m * 2.5 + 0.7, m * 2.5 + 1.9, 0.0, 21.0), GLASS)
    r = fit(scene(p, []))
    assert r.refused and r.refusals[0].code == "no_floors"


def test_a_facade_narrower_than_two_bays_is_refused():
    p, windows = facade(width_m=3.5)
    r = fit(scene(p, windows))
    assert r.refused and r.proposals == []


def test_a_grid_that_is_mostly_wall_is_refused():
    # A third of the cells have a window: periodic in both directions, but the grid
    # does not describe this facade, and two thirds of it would be wrong boxes.
    p = Painter(30.0, 21.0, 0.08)
    for k in range(1, 7):
        p.paint((0.0, 30.0, k * 3.0 - 0.08, k * 3.0 + 0.08), SLAB)
    every = grid_windows(width_m=30.0, height_m=21.0, floor_m=3.0, bay_m=2.5, win_w=1.2, win_h=1.5)
    per_floor = len([w for w in every if w[2] < 3.0])
    kept = [w for i, w in enumerate(every) if (i // per_floor + i % per_floor) % 3 == 0]
    for w in kept:
        p.paint(w, GLASS)
    r = fit(scene(p, kept))
    assert r.refused and r.refusals[0].code == "sparse_grid"
    assert "28 з 84" in r.refusals[0].message
    # Every accepted cell is in fact a window; the facade is refused all the same,
    # because a fit that leaves two thirds of its grid empty cannot be told apart from
    # a wrong fit -- and on dataset_1910 a right grid fills 86-91 % of its cells.
    assert len(r.accepted) == len(kept) and r.proposals == []


def test_a_faint_ripple_is_not_a_column_structure():
    # Mullion-like stripes 3 % darker than the wall: periodic, but not openings.
    p = Painter(30.0, 21.0, 0.08)
    for m in range(22):
        p.paint((m * 1.35 + 0.2, m * 1.35 + 1.05, 0.0, 21.0), (198.0, 184.0, 160.0))
    r = fit(scene(p, [], noise=6.0))
    assert r.refused and r.proposals == []


def test_a_rotated_grid_or_mismatched_layers_are_errors_not_guesses():
    class Rotated:
        m_per_px = 0.08
        pixel00_uv = (0.0, 0.0)
        du_uv = (0.0, 0.08)
        dv_uv = (-0.08, 0.0)

    with pytest.raises(ValueError, match="axis-aligned"):
        PixelGrid.of(Rotated())
    p, windows = facade()
    sc = scene(p, windows)
    with pytest.raises(ValueError, match="alpha"):
        gridfit(sc.rgb, sc.alpha[:-1], sc.grid)


def test_period_and_threshold_helpers_on_known_signals():
    n, period = 600, 37.5
    x = np.sin(2 * np.pi * np.arange(n) / period)
    m = np.ones(n, bool)
    m[200:260] = False                 # a tree: a gap, not a zero
    got = find_period(x, m, 10, 120, 0.25)
    assert got is not None and got.px == pytest.approx(period, abs=0.05)
    assert find_period(np.random.default_rng(0).normal(size=n), m, 10, 120, 0.25) is None
    assert otsu(np.r_[np.zeros(30), np.ones(10)]) == pytest.approx(0.5)


def test_agreement_is_containment_not_iou():
    from aerosplat.semantics.gridfit import _Columns, CueFit
    w = 200
    narrow, wide, elsewhere = (np.zeros(w, bool) for _ in range(3))
    for k in range(0, w, 50):
        narrow[k + 10:k + 18] = True     # darkness: the shaded half of a pane
        wide[k + 5:k + 25] = True        # blueness: the whole pane
        elsewhere[k + 30:k + 40] = True  # something else, between the panes

    def col(mask, too_wide=False):
        return _Columns(CueFit(cue="x", periodic=True), 50.0, mask, mask, too_wide)

    assert agreement(col(narrow), col(wide)) == 1.0
    assert agreement(col(narrow), col(elsewhere)) == 0.0
    assert agreement(col(narrow, too_wide=True), col(wide)) == 0.0


# ── candidates: proposals meet what people already decided ───────────────
def _wall(fid="B01/N1", normal=(0.0, 1.0), origin=(0.0, 0.0, 0.0), length=30.0, height=21.0):
    n = np.asarray(normal, float)
    ux, uy = u_axis_for((float(n[0]), float(n[1])))
    return Facade(id=fid, building_id=fid.split("/")[0], kind=SurfaceKind.wall, origin=origin,
                  normal=(float(n[0]), float(n[1]), 0.0), u_axis=(ux, uy, 0.0),
                  azimuth_deg=azimuth_of((float(n[0]), float(n[1]))), u_min=0.0, u_max=length,
                  v_ground=origin[2], v_max=origin[2] + height, plane_rms_m=0.3, n_points=1000)


def _feature(wall, rect, lifecycle, class_id="opening.window"):
    return make_access_point(site_key="demo/pier", facade=wall, class_id=class_id,
                             taxonomy="core@1.1", polygon_uv=rect_uv(rect[0], rect[2], rect[1], rect[3]),
                             lifecycle=lifecycle)


def test_proposals_become_candidates_and_never_verified():
    p, windows = facade()
    r = fit(scene(p, windows))
    wall = _wall()
    props = [Proposal(cell_rect(c), c.confidence, r.note(c)) for c in r.proposals]
    aps = make_candidates(props, site_key="demo/pier", facade=wall, class_id="opening.window",
                          taxonomy="core@1.1", model=r.model, created_by="pipeline:gridfit",
                          run_id="r1", epoch_id="baseline", evidence=["ortho:B01/N1@8cm"])
    assert len(aps) == len(windows)
    for ap in aps:
        assert ap.lifecycle is Lifecycle.candidate
        assert ap.provenance.origin is Origin.pipeline
        assert ap.provenance.model == MODEL
        assert ap.provenance.created_by == "pipeline:gridfit"
        assert 0.0 < ap.confidence < 1.0
        assert "поверх 3.0" in ap.notes and "крок колонок 2.50" in ap.notes


def test_cells_on_verified_or_rejected_features_are_skipped():
    wall = _wall()
    props = [Proposal((0.7, 1.9, 0.9, 2.4), 0.8, ""), Proposal((3.2, 4.4, 0.9, 2.4), 0.8, ""),
             Proposal((5.7, 6.9, 0.9, 2.4), 0.8, ""), Proposal((8.2, 9.4, 0.9, 2.4), 0.8, "")]
    existing = [_feature(wall, (0.6, 2.0, 0.8, 2.5), Lifecycle.verified),
                _feature(wall, (3.2, 4.4, 0.9, 2.4), Lifecycle.rejected),
                _feature(wall, (5.75, 6.95, 0.95, 2.45), Lifecycle.candidate)]
    keep, dropped = screen(props, wall, existing)
    assert [k.rect for k in keep] == [(8.2, 9.4, 0.9, 2.4)]
    assert dropped == {"verified": 1, "rejected": 1, "candidate": 1}


def test_a_hand_drawn_pair_blocks_both_panes_inside_it():
    wall = _wall()
    pair = _feature(wall, (0.5, 3.5, 0.8, 2.6), Lifecycle.verified)
    props = [Proposal((0.7, 1.9, 0.9, 2.4), 0.8, ""), Proposal((2.1, 3.3, 0.9, 2.4), 0.8, "")]
    keep, dropped = screen(props, wall, [pair])
    assert keep == [] and dropped == {"verified": 2}


def test_a_rejected_feature_from_a_renamed_facade_still_blocks():
    # The same wall, fitted again by a newer run: other name, origin 0.4 m along u and
    # 0.3 m off the old plane. The rejection must still hold (§5.9, §3.4).
    old = _wall("B08/SW3", origin=(0.0, 0.0, 0.0))
    new = _wall("B04/SW1", origin=(-0.4, 0.3, 0.0))
    rejected = _feature(old, (3.2, 4.4, 0.9, 2.4), Lifecycle.rejected)
    r = rect_on(new, rejected)
    # A north wall's u runs west (−x): the old u = 3.2 is x = −3.2, which is u = 2.8 from
    # the new origin at x = −0.4.
    assert r is not None and r[0] == pytest.approx(2.8, abs=1e-6)
    keep, dropped = screen([Proposal((2.8, 4.0, 0.9, 2.4), 0.9, "")], new, [rejected])
    assert keep == [] and dropped == {"rejected": 1}


def test_features_on_other_walls_do_not_block():
    wall = _wall()
    around_the_corner = _wall("B01/E1", normal=(1.0, 0.0), origin=(30.0, -10.0, 0.0))
    far_behind = _wall("B02/N1", origin=(0.0, -12.0, 0.0))
    others = [_feature(around_the_corner, (0.7, 1.9, 0.9, 2.4), Lifecycle.rejected),
              _feature(far_behind, (0.7, 1.9, 0.9, 2.4), Lifecycle.rejected)]
    keep, dropped = screen([Proposal((0.7, 1.9, 0.9, 2.4), 0.9, "")], wall, others)
    assert len(keep) == 1 and dropped == {}


def test_overlap_is_measured_against_the_smaller_shape():
    assert overlap((0, 1, 0, 1), (0, 1, 0, 1)) == 1.0
    assert overlap((0, 1, 0, 1), (1, 2, 0, 1)) == 0.0
    assert overlap((0, 1, 0, 1), (-5, 5, -5, 5)) == 1.0          # inside a big hand-drawn box
    assert overlap((0, 1, 0, 1), (0.9, 2, 0, 1)) == pytest.approx(0.1)


def test_a_refused_result_proposes_nothing_even_with_accepted_cells():
    p, windows = facade()
    r = fit(scene(p, windows))
    assert r.proposals
    r.refusals.append(Refusal(code="sparse_grid", message="test"))
    assert r.accepted and r.proposals == []


def test_coverage_raster_follows_the_api_layout():
    # Row 0 of the coverage grid is the bottom of the wall.
    views = np.zeros((4, 2), np.int32)
    views[0, :] = 5
    cov = CoverageRaster(views=views, u_min=0.0, v_min=0.0, cell_m=1.0)
    g = PixelGrid(m_per_px=0.5, u00=0.25, v00=3.75)        # 4 × 8 px over a 2 × 4 m wall
    m = cov.mask(g, (8, 4))
    assert m[-2:, :].all() and not m[:-2, :].any()
