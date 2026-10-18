"""Targets (aerosplat.mission.targets): only a verified, confirmed access point becomes a target.

ADR-0009 and ACCESS_POINTS §6.3 as tests. Most of them are refusals, because what matters
is what the system will NOT let an operator do: confirm a candidate, a stale or ambiguous
access point, a target whose σ is unknown or too large, or loosen the taxonomy's numbers.
"""

import json

import pytest
import yaml

import mission_scenes as ms
from aerosplat.authored.models import Address, ApFlag, ApUncertainty, Lifecycle
from aerosplat.mission.models import (TARGETS_JSON, Code, HandoverOverride, MissionRefused,
                                      TargetState)
from aerosplat.mission.targets import ap_fingerprint, planning_findings
from aerosplat.taxonomy import HandoverMode, load_taxonomy, taxonomy_root


@pytest.fixture
def tax():
    return load_taxonomy()


@pytest.fixture
def stores(tmp_path):
    return ms.stores(tmp_path)


def refusal(stores, tax, ap, **kw):
    authored, targets = stores
    authored.create(ap)
    kw = {"sigma_ap_m": 0.1, "sigma_georef_m": 0.3, "confirm": True, **kw}
    with pytest.raises(MissionRefused) as e:
        targets.create(tax, ap.id, by="tester", **kw)
    return {f.code for f in e.value.findings}


# ── positive ─────────────────────────────────────────────────────────────
def test_a_verified_window_becomes_a_confirmed_target_with_taxonomy_numbers(stores, tax):
    authored, targets = stores
    ap = authored.create(ms.window_ap())
    t = targets.create(tax, ap.id, by="lukhashka", sigma_ap_m=0.1, sigma_georef_m=0.3,
                       confirm=True)
    assert t.state is TargetState.confirmed and t.confirmed_by == "op:lukhashka"
    h = t.handover
    assert (h.mode, h.standoff_m, h.min_clearance_m, h.approach_cone_deg, h.entry_distance_m) == \
        (HandoverMode.hover_handoff, 3.0, 1.5, 30.0, 12.0)
    assert h.max_sigma_m == 1.5 and h.source == "core@1.1 opening.window"
    assert t.sigma.target_m == pytest.approx((0.1 ** 2 + 0.3 ** 2) ** 0.5)
    assert t.sigma.georef.source == "stated by op:lukhashka"
    assert t.ap_fingerprint == ap_fingerprint(ap)
    # The confirmation is in the audit log, with the operator and the target as confirmed.
    last = authored.audit()[-1]
    assert last.op == "target_confirm" and last.by == "op:lukhashka"
    assert last.feature_id == ap.id and last.after["id"] == t.id


def test_a_draft_is_confirmed_later_and_both_steps_are_audited(stores, tax):
    authored, targets = stores
    ap = authored.create(ms.window_ap())
    draft = targets.create(tax, ap.id, by="resolver", sigma_ap_m=0.1, sigma_georef_m=0.3)
    assert draft.state is TargetState.draft and draft.confirmed_by is None
    done = targets.confirm(tax, draft.id, by="lukhashka")
    assert done.state is TargetState.confirmed and done.confirmed_by == "op:lukhashka"
    ops = [e.op for e in authored.audit()]
    assert ops[-2:] == ["target_create", "target_confirm"]
    assert authored.audit()[-1].before["state"] == "draft"


def test_the_access_points_own_uncertainty_wins_over_a_stated_one(stores, tax):
    authored, targets = stores
    unc = ApUncertainty(center_cov=((0.04, 0, 0), (0, 0.09, 0), (0, 0, 0.01)))
    ap = authored.create(ms.window_ap(uncertainty=unc))
    t = targets.create(tax, ap.id, by="t", sigma_ap_m=0.01, sigma_georef_m=0.3, confirm=True)
    assert t.sigma.ap.value_m == pytest.approx(0.3)            # sqrt of the largest eigenvalue
    assert t.sigma.ap.source == f"{ap.id}.uncertainty"


def test_more_careful_overrides_are_kept(stores, tax):
    authored, targets = stores
    ap = authored.create(ms.window_ap())
    t = targets.create(tax, ap.id, by="t", sigma_ap_m=0.1, sigma_georef_m=0.3, confirm=True,
                       override=HandoverOverride(standoff_m=4.0, approach_cone_deg=20.0))
    assert (t.handover.standoff_m, t.handover.approach_cone_deg) == (4.0, 20.0)
    assert t.handover.overridden == ["approach_cone_deg", "standoff_m"]


def test_targets_are_authored_data_next_to_the_access_points(stores, tax, tmp_path):
    authored, targets = stores
    ap = authored.create(ms.roof_pad_ap())
    t = targets.create(tax, ap.id, by="t", sigma_ap_m=0.2, sigma_georef_m=0.5, confirm=True)
    on_disk = json.loads((tmp_path / "authored" / TARGETS_JSON).read_text(encoding="utf-8"))
    assert [x["id"] for x in on_disk["targets"]] == [t.id]
    assert targets.get(t.id) == t
    assert t.handover.mode is HandoverMode.land                # pad.roof_flat's default


# ── negative: the access point ───────────────────────────────────────────
@pytest.mark.parametrize("lifecycle, code", [
    (Lifecycle.candidate, Code.not_verified),
    (Lifecycle.rejected, Code.not_verified),
    (Lifecycle.stale, Code.stale),
])
def test_only_a_verified_access_point_becomes_a_target(stores, tax, lifecycle, code):
    assert refusal(stores, tax, ms.window_ap(lifecycle=lifecycle)) == {code}


def test_not_redetected_blocks(stores, tax):
    assert refusal(stores, tax, ms.window_ap(flags=[ApFlag.not_redetected])) == {Code.not_redetected}


def test_an_ambiguous_floor_blocks(stores, tax):
    ap = ms.window_ap(address=Address(floor_index=3, floor_ambiguous=True))
    assert refusal(stores, tax, ap) == {Code.floor_ambiguous}


def test_sigma_above_the_modes_threshold_blocks(stores, tax):
    # Earth Studio's absolute accuracy (~2-3 m) against hover_handoff's 1.5 m gate.
    assert refusal(stores, tax, ms.window_ap(), sigma_georef_m=2.5) == {Code.sigma_too_large}


def test_unknown_sigma_blocks_and_nothing_is_written(stores, tax, tmp_path):
    authored, _ = stores
    assert refusal(stores, tax, ms.window_ap(), sigma_ap_m=None, sigma_georef_m=None) == \
        {Code.sigma_unknown}
    assert not (tmp_path / "authored" / TARGETS_JSON).exists()
    assert [e.op for e in authored.audit()] == ["create"]      # only the access point itself


def test_a_mode_the_class_does_not_allow_blocks(stores, tax):
    assert refusal(stores, tax, ms.window_ap(), mode=HandoverMode.land) == {Code.mode_not_allowed}


@pytest.mark.parametrize("override", [
    HandoverOverride(standoff_m=2.0),            # closer to the person than the taxonomy says
    HandoverOverride(min_clearance_m=1.0),
    HandoverOverride(approach_cone_deg=60.0),
    HandoverOverride(entry_distance_m=6.0),
], ids=["standoff", "clearance", "cone", "entry"])
def test_an_override_may_not_loosen_the_taxonomy(stores, tax, override):
    assert refusal(stores, tax, ms.window_ap(), override=override) == \
        {Code.override_relaxes_taxonomy}


def test_a_constraint_is_never_a_target(stores, tax):
    got = refusal(stores, tax, ms.window_ap(class_id="constraint.obstacle"))
    assert got == {Code.unsupported_kind, Code.no_handover}


def test_a_missing_access_point_is_refused(stores, tax):
    _, targets = stores
    with pytest.raises(MissionRefused) as e:
        targets.create(tax, "ap_00000000000000000000000000", by="t", sigma_ap_m=0.1,
                       sigma_georef_m=0.1)
    assert [f.code for f in e.value.findings] == [Code.ap_missing]


def test_every_reason_is_reported_at_once(stores, tax):
    ap = ms.window_ap(lifecycle=Lifecycle.stale, address=Address(floor_ambiguous=True))
    assert refusal(stores, tax, ap, sigma_georef_m=3.0) == \
        {Code.stale, Code.floor_ambiguous, Code.sigma_too_large}


def test_a_mode_without_a_threshold_in_the_taxonomy_blocks(stores, tmp_path):
    core = yaml.safe_load((taxonomy_root() / "core.yaml").read_text(encoding="utf-8"))
    del core["handover_modes"]["hover_handoff"]
    root = tmp_path / "tax"
    root.mkdir()
    (root / "core.yaml").write_text(yaml.safe_dump(core, allow_unicode=True), encoding="utf-8")
    assert refusal(stores, load_taxonomy(root=root), ms.window_ap()) == {Code.no_threshold}


# ── negative: confirming, revoking, and what planning re-checks ──────────
def test_a_revoked_target_cannot_be_confirmed_or_planned(stores, tax):
    authored, targets = stores
    ap = authored.create(ms.window_ap())
    draft = targets.create(tax, ap.id, by="t", sigma_ap_m=0.1, sigma_georef_m=0.3)
    gone = targets.revoke(draft.id, by="t", reason="wrong building")
    with pytest.raises(MissionRefused):
        targets.confirm(tax, draft.id, by="t")
    assert [f.code for f in planning_findings(gone, ap)] == [Code.target_revoked]
    assert authored.audit()[-1].op == "target_revoke"


def test_a_draft_whose_access_point_changed_cannot_be_confirmed(stores, tax):
    authored, targets = stores
    ap = authored.create(ms.window_ap())
    draft = targets.create(tax, ap.id, by="t", sigma_ap_m=0.1, sigma_georef_m=0.3)
    moved = ap.geometry.model_copy(update={"center": (1.0, 15.0, 9.0)})
    authored.replace(ap.model_copy(update={"geometry": moved}, deep=True))
    with pytest.raises(MissionRefused) as e:
        targets.confirm(tax, draft.id, by="t")
    assert Code.target_changed in {f.code for f in e.value.findings}


def test_planning_rechecks_the_access_point_as_it_is_now(stores, tax):
    authored, targets = stores
    ap = authored.create(ms.window_ap())
    t = targets.create(tax, ap.id, by="t", sigma_ap_m=0.1, sigma_georef_m=0.3, confirm=True)
    assert planning_findings(t, ap) == []
    draft = t.model_copy(update={"state": TargetState.draft})
    assert [f.code for f in planning_findings(draft, ap)] == [Code.target_not_confirmed]
    assert [f.code for f in planning_findings(t, None)] == [Code.ap_missing]
    ambiguous = ap.model_copy(update={"address": Address(floor_ambiguous=True)})
    assert [f.code for f in planning_findings(t, ambiguous)] == [Code.floor_ambiguous]
