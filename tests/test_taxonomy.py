"""Taxonomy packs: the shipped `core`, the validator, and the cross-pack checks (ADR-0007)."""

import pytest
import yaml

from aerosplat.taxonomy import (ApKind, Taxonomy, TaxonomyError, load_pack, load_taxonomy,
                                pack_names, taxonomy_root)
from aerosplat.taxonomy.loader import _version_ok, cached_taxonomy
from aerosplat.taxonomy.models import TaxonomyPack

MINIMAL = {
    "pack": "core", "version": "1.0.0",
    "attributes": [{"id": "state", "labels": {"uk": "Стан"}, "values": ["open", "closed"]}],
    "classes": [{"id": "opening.window", "labels": {"uk": "Вікно"}, "orientation": "vertical",
                 "attributes": ["state"],
                 "handover": {"allowed": ["hover_handoff"], "default": "hover_handoff",
                              "standoff_m": 3.0, "min_clearance_m": 1.5,
                              "approach_cone_deg": 30}}],
}


def write(tmp_path, name, data):
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def pack(**over):
    data = {k: (v.copy() if isinstance(v, list) else v) for k, v in MINIMAL.items()}
    data.update(over)
    return data


# ── the pack that actually ships ─────────────────────────────────────────
def test_core_pack_has_the_classes_access_points_promises():
    tax = load_taxonomy()
    assert set(tax.classes) == {"opening.window", "opening.door", "pad.roof_flat",
                                "pad.roof_pitched", "pad.ground", "constraint.obstacle",
                                "constraint.vegetation", "constraint.no_go"}
    assert tax.version_string == "core@1.0"          # major.minor, as §2.5 writes it
    assert tax.ids_of_kind(ApKind.opening) == ["opening.door", "opening.window"]


def test_core_openings_carry_a_size_prior_and_approach_numbers():
    tax = load_taxonomy()
    for cid in tax.ids_of_kind(ApKind.opening):
        cls = tax.get(cid)
        assert cls.size_prior_m is not None, f"{cid} has no size prior to filter detections with"
        assert cls.prompts, f"{cid} has no open-vocabulary prompts"
        handover = tax.handover_of(cid)
        assert handover is not None and handover.default in handover.allowed
    window = tax.get("opening.window")
    assert window.size_prior_m is not None
    assert window.size_prior_m.admits(1.4, 1.6)
    assert not window.size_prior_m.admits(4.0, 1.6)
    assert window.citygml == "Window"


def test_a_constraint_is_not_a_target_so_it_has_no_approach_numbers():
    tax = load_taxonomy()
    for cid in tax.ids_of_kind(ApKind.constraint):
        assert tax.handover_of(cid) is None


def test_every_shipped_pack_loads():
    names = sorted(p.stem for p in taxonomy_root().glob("*.yaml"))
    assert "core" in names
    for name in names:
        load_pack(name)


# ── validation inside one pack ───────────────────────────────────────────
@pytest.mark.parametrize("bad, message", [
    ({"id": "window", "labels": {"uk": "В"}}, "String should match"),
    ({"id": "thing.window", "labels": {"uk": "В"}}, "is not a kind"),
    ({"id": "opening.a.b.c", "labels": {"uk": "В"}}, "String should match"),
    ({"id": "opening.w", "labels": {"uk": "В"}, "parent": "opening.w"}, "its own parent"),
    ({"id": "opening.w", "labels": {"uk": "В"}, "parent": "pad.roof_flat"}, "different kind"),
    ({"id": "constraint.x", "labels": {"uk": "В"},
      "handover": {"allowed": ["land"], "default": "land", "standoff_m": 1,
                   "min_clearance_m": 1, "approach_cone_deg": 10}}, "has no handover"),
    ({"id": "opening.w", "labels": {"uk": "В"}, "size_prior_m": {"width": [2, 1],
                                                                "height": [1, 2]}},
     "lo <= hi"),
    ({"id": "opening.w", "labels": {"uk": "В"},
      "handover": {"allowed": ["land"], "default": "drop", "standoff_m": 1,
                   "min_clearance_m": 1, "approach_cone_deg": 10}}, "is not in allowed"),
])
def test_a_malformed_class_is_refused(tmp_path, bad, message):
    write(tmp_path, "core", pack(classes=[bad]))
    with pytest.raises(TaxonomyError, match=message):
        load_taxonomy(root=tmp_path)


def test_an_attribute_needs_values_or_free_text(tmp_path):
    write(tmp_path, "core", pack(attributes=[{"id": "state", "labels": {"uk": "С"}}]))
    with pytest.raises(TaxonomyError, match="give values"):
        load_taxonomy(root=tmp_path)
    write(tmp_path, "core", pack(attributes=[{"id": "state", "labels": {"uk": "С"},
                                              "values": ["a"], "free_text": True}]))
    with pytest.raises(TaxonomyError, match="exclusive"):
        load_taxonomy(root=tmp_path)


def test_a_default_outside_the_value_list_is_refused(tmp_path):
    write(tmp_path, "core", pack(attributes=[{"id": "state", "labels": {"uk": "С"},
                                              "values": ["open"], "default": "shut"}]))
    with pytest.raises(TaxonomyError, match="is not a value"):
        load_taxonomy(root=tmp_path)


def test_a_file_that_disagrees_with_its_own_name_is_refused(tmp_path):
    write(tmp_path, "core", pack(pack="logistics"))
    with pytest.raises(TaxonomyError, match="declares pack"):
        load_taxonomy(root=tmp_path)


def test_duplicate_ids_inside_a_pack_are_refused(tmp_path):
    write(tmp_path, "core", pack(classes=[MINIMAL["classes"][0], MINIMAL["classes"][0]]))
    with pytest.raises(TaxonomyError, match="duplicate class ids"):
        load_taxonomy(root=tmp_path)


def test_an_unknown_field_is_refused_rather_than_ignored(tmp_path):
    # A typo in a pack must fail loudly: silently dropping `size_prior` would ship a
    # class with no size filter and nobody would notice until fusion.
    write(tmp_path, "core", pack(classes=[{**MINIMAL["classes"][0], "size_prior": {}}]))
    with pytest.raises(TaxonomyError, match="[Ee]xtra"):
        load_taxonomy(root=tmp_path)


def test_a_missing_pack_says_where_it_looked(tmp_path):
    with pytest.raises(TaxonomyError, match="not found"):
        load_taxonomy(root=tmp_path)


# ── validation across packs ──────────────────────────────────────────────
def test_a_pack_extends_core_and_inherits_its_approach_numbers(tmp_path):
    write(tmp_path, "core", pack())
    write(tmp_path, "logistics", {
        "pack": "logistics", "version": "1.1.0", "requires": {"core": ">=1.0"},
        "classes": [{"id": "opening.dock_door", "parent": "opening.window",
                     "labels": {"uk": "Ворота доку", "en": "Dock door"},
                     "attributes": []}]})
    tax = load_taxonomy(["logistics"], root=tmp_path)
    assert tax.version_string == "core@1.0+logistics@1.1"
    assert tax.ancestors("opening.dock_door") == ["opening.window"]
    # The child says nothing about attributes or approach, so it takes the parent's.
    assert [a.id for a in tax.attributes_of("opening.dock_door")] == ["state"]
    handover = tax.handover_of("opening.dock_door")
    assert handover is not None and handover.standoff_m == 3.0
    assert tax.get("opening.dock_door").pack == "logistics"


def test_core_is_always_loaded_and_listed_first():
    assert pack_names(["logistics", "core", "logistics"]) == ["core", "logistics"]
    assert pack_names([]) == ["core"]


def test_a_parent_from_a_pack_that_was_not_loaded_is_refused(tmp_path):
    write(tmp_path, "core", pack())
    write(tmp_path, "x", {"pack": "x", "version": "1.0.0",
                          "classes": [{"id": "opening.a", "parent": "opening.ghost",
                                       "labels": {"uk": "A"}}]})
    with pytest.raises(TaxonomyError, match="does not exist"):
        load_taxonomy(["x"], root=tmp_path)


def test_an_unsatisfied_requirement_is_refused(tmp_path):
    write(tmp_path, "core", pack(version="1.0.0"))
    write(tmp_path, "x", {"pack": "x", "version": "1.0.0", "requires": {"core": ">=1.4"},
                          "classes": []})
    with pytest.raises(TaxonomyError, match="requires core>=1.4"):
        load_taxonomy(["x"], root=tmp_path)
    write(tmp_path, "y", {"pack": "y", "version": "1.0.0", "requires": {"nope": ">=1.0"},
                          "classes": []})
    with pytest.raises(TaxonomyError, match="not loaded"):
        load_taxonomy(["y"], root=tmp_path)


def test_two_packs_cannot_define_the_same_id(tmp_path):
    write(tmp_path, "core", pack())
    write(tmp_path, "x", {"pack": "x", "version": "1.0.0", "classes": MINIMAL["classes"]})
    with pytest.raises(TaxonomyError, match="defined by both"):
        load_taxonomy(["x"], root=tmp_path)
    write(tmp_path, "x", {"pack": "x", "version": "1.0.0",
                          "attributes": MINIMAL["attributes"], "classes": []})
    with pytest.raises(TaxonomyError, match="defined by both"):
        load_taxonomy(["x"], root=tmp_path)


def test_an_unknown_attribute_on_a_class_is_refused(tmp_path):
    write(tmp_path, "core", pack(classes=[{"id": "opening.w", "labels": {"uk": "В"},
                                           "attributes": ["ghost"]}]))
    with pytest.raises(TaxonomyError, match="unknown attributes"):
        load_taxonomy(root=tmp_path)


def test_a_parent_loop_is_refused():
    a = {"id": "opening.a", "parent": "opening.b", "labels": {"uk": "A"}}
    b = {"id": "opening.b", "parent": "opening.a", "labels": {"uk": "B"}}
    with pytest.raises(TaxonomyError, match="loops at"):
        Taxonomy([TaxonomyPack(pack="core", version="1.0.0", classes=[a, b])])


@pytest.mark.parametrize("have, want, ok", [
    ("1.0.0", ">=1.0", True), ("1.0.0", ">=1.1", False), ("2.3.4", ">=1.9.9", True),
    ("1.2.0", ">=1.2.1", False), ("1.0.0", "*", True), ("1.0.0", "", True),
])
def test_version_requirements(have, want, ok):
    assert _version_ok(have, want) is ok


def test_an_unsupported_requirement_form_is_refused():
    with pytest.raises(TaxonomyError, match="unsupported version requirement"):
        _version_ok("1.0.0", "^1.0")


# ── what a feature is checked against ────────────────────────────────────
def test_attribute_values_are_checked_against_the_class():
    tax = load_taxonomy()
    tax.validate_attributes("opening.window", {"state": "closed", "glazing": "glass"})
    with pytest.raises(TaxonomyError, match="has no attribute"):
        tax.validate_attributes("opening.window", {"slope_deg": "3"})
    with pytest.raises(TaxonomyError, match="is not one of"):
        tax.validate_attributes("opening.window", {"state": "ajar"})
    # `label_text` is free text, so anything goes -- it is an OCR'd dock number.
    tax.validate_attributes("opening.window", {"label_text": "Д-12"})


def test_an_unknown_class_names_the_taxonomy_version():
    tax = load_taxonomy()
    with pytest.raises(TaxonomyError, match="core@1.0"):
        tax.get("opening.teleporter")


def test_the_cache_reloads_a_pack_that_changed(tmp_path):
    write(tmp_path, "core", pack())
    first = cached_taxonomy(root=tmp_path)
    assert cached_taxonomy(root=tmp_path) is first               # same files, same object
    write(tmp_path, "core", pack(version="1.3.0"))
    # Classes are data; editing a pack has to take effect without restarting the server.
    assert cached_taxonomy(root=tmp_path).version_string == "core@1.3"
