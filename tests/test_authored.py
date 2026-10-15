"""Authored access points: the record, the store and the audit log (ADR-0006, §3.1-§3.5)."""

import json

import pytest

from aerosplat.authored import (AccessPoint, AuthoredError, AuthoredStore, Lifecycle, NotFound,
                                geometry_from_uv, make_access_point, new_ap_id, rect_uv)
from aerosplat.authored.models import AP_PREFIX, actor
from aerosplat.semantics.models import Facade, SurfaceKind, azimuth_of, u_axis_for


def a_facade(normal=(0.0, 1.0, 0.0), origin=(0.0, 0.0, 0.0), **over) -> Facade:
    ux, uy = u_axis_for((normal[0], normal[1]))
    kw = dict(id="B01/N1", building_id="B01", kind=SurfaceKind.wall, origin=origin,
              normal=(normal[0], normal[1], 0.0), u_axis=(ux, uy, 0.0),
              azimuth_deg=azimuth_of((normal[0], normal[1])), u_min=-10.0, u_max=10.0,
              v_ground=origin[2], v_max=origin[2] + 12.0, plane_rms_m=0.1, n_points=500)
    kw.update(over)
    return Facade(**kw)


def a_point(facade=None, u=0.0, v=3.0, w=1.4, h=1.6, **over) -> AccessPoint:
    facade = facade or a_facade()
    kw = dict(site_key="demo/pier", facade=facade, class_id="opening.window",
              taxonomy="core@1.0",
              polygon_uv=rect_uv(u - w / 2, v - h / 2, u + w / 2, v + h / 2),
              created_by="op:test")
    kw.update(over)
    return make_access_point(**kw)


@pytest.fixture
def store(tmp_path):
    return AuthoredStore(tmp_path / "pier" / "authored", "demo/pier")


# ── identifiers ──────────────────────────────────────────────────────────
def test_ids_are_ulids_that_sort_by_creation_time():
    early, late = new_ap_id(1_700_000_000_000), new_ap_id(1_900_000_000_000)
    assert early.startswith(AP_PREFIX) and len(early) == len(AP_PREFIX) + 26
    assert early < late
    assert len({new_ap_id() for _ in range(500)}) == 500
    # Crockford base32: no I, L, O or U, so an id read aloud is unambiguous.
    assert not (set(early[len(AP_PREFIX):]) & set("ILOU"))


def test_the_record_refuses_an_id_that_is_not_one(tmp_path):
    with pytest.raises(ValueError, match="String should match"):
        a_point(feature_id="ap_hello")


# ── geometry ─────────────────────────────────────────────────────────────
def test_the_enu_block_follows_from_the_drawn_polygon():
    # Wall facing north at the origin: u then runs west, so u = -6 is 6 m east.
    facade = a_facade()
    g = geometry_from_uv(facade, rect_uv(-6.7, 2.0, -5.3, 3.6))
    assert g.width == pytest.approx(1.4) and g.height == pytest.approx(1.6)
    assert g.u_center == pytest.approx(-6.0) and g.v_center == pytest.approx(2.8)
    assert g.center == pytest.approx((6.0, 0.0, 2.8))
    assert g.normal == (0.0, 1.0, 0.0) and g.frame == "site_enu"
    assert g.u_bounds == pytest.approx((-6.7, -5.3))
    assert g.v_bounds == pytest.approx((2.0, 3.6))


def test_v_is_the_enu_height_not_an_offset():
    # A facade whose foot is 40 m up still takes v straight from the ENU z.
    facade = a_facade(origin=(0.0, 0.0, 40.0))
    g = geometry_from_uv(facade, rect_uv(0.0, 41.0, 1.0, 42.0))
    assert g.center[2] == pytest.approx(41.5)
    assert g.v_center == pytest.approx(41.5)


def test_a_sliver_is_refused_as_a_stray_click():
    with pytest.raises(ValueError, match="too thin"):
        geometry_from_uv(a_facade(), rect_uv(0.0, 0.0, 0.01, 2.0))
    with pytest.raises(ValueError, match="at least three points"):
        geometry_from_uv(a_facade(), [(0.0, 0.0), (1.0, 1.0)])


def test_the_feature_carries_the_frame_it_was_drawn_in():
    # Not just the facade id: a recomputed run may move or rename the facade, and the
    # stored plane is what a re-anchoring pass compares against (§5.9).
    facade = a_facade()
    ap = a_point(facade)
    assert ap.host.facade_id == "B01/N1" and ap.host.building_id == "B01"
    assert ap.host.origin == facade.origin and ap.host.u_axis == facade.u_axis
    assert ap.host.v_ground == facade.v_ground
    assert ap.provenance.run_id is None          # drawn by hand, not produced by a run


def test_kind_must_match_the_class():
    record = a_point().model_dump(by_alias=True)
    record["kind"] = "pad"
    with pytest.raises(ValueError, match="does not match class"):
        AccessPoint(**record)


# ── the file on disk ─────────────────────────────────────────────────────
def test_an_empty_site_reads_as_empty_without_creating_anything(store):
    assert store.load().features == []
    assert store.features() == []
    assert store.audit() == []
    assert not store.dir.exists()


def test_the_wire_shape_uses_class_as_the_doc_does(store):
    store.create(a_point())
    data = json.loads(store.state_path.read_text(encoding="utf-8"))
    assert data["schema"] == 1 and data["site_key"] == "demo/pier"
    feature = data["features"][0]
    assert feature["class"] == "opening.window" and "class_" not in feature
    assert feature["geometry"]["frame"] == "site_enu"
    assert AccessPoint(**feature).class_ == "opening.window"


def test_a_drawn_feature_is_verified_from_the_start(store):
    # §3.4: the pipeline proposes candidates; a person drawing one has already verified it.
    assert store.create(a_point()).lifecycle is Lifecycle.verified


def test_round_trip_and_filtering(store):
    w1 = store.create(a_point(u=-5.0))
    w2 = store.create(a_point(u=0.0, lifecycle=Lifecycle.rejected))
    other = a_facade(normal=(1.0, 0.0, 0.0), origin=(10.0, 0.0, 0.0), id="B01/E1")
    w3 = store.create(a_point(other, u=1.0))
    assert [f.id for f in store.features()] == sorted([w1.id, w2.id, w3.id])
    assert [f.id for f in store.features(facade_id="B01/E1")] == [w3.id]
    assert [f.id for f in store.features(lifecycle=[Lifecycle.verified])] == \
        sorted([w1.id, w3.id])
    assert store.get(w2.id).lifecycle is Lifecycle.rejected


def test_the_same_id_cannot_be_stored_twice(store):
    ap = store.create(a_point())
    with pytest.raises(AuthoredError, match="already exists"):
        store.create(ap)


def test_missing_features_raise_a_readable_error(store):
    for call in (lambda: store.get("ap_01ABCDEFGHJKMNPQRSTVWXYZ00"),
                 lambda: store.delete("ap_01ABCDEFGHJKMNPQRSTVWXYZ00"),
                 lambda: store.replace(a_point())):
        with pytest.raises(NotFound, match="no access point"):
            call()


def test_a_corrupt_state_file_says_which_file(store):
    store.dir.mkdir(parents=True)
    store.state_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(AuthoredError, match="access_points.json"):
        store.load()


def test_a_half_written_file_cannot_replace_a_good_one(store, monkeypatch):
    store.create(a_point())
    good = store.state_path.read_text(encoding="utf-8")

    def boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr("os.replace", boom)
    with pytest.raises(OSError):
        store.create(a_point(u=4.0))
    # The write went to a temporary file first, so the previous state survived intact.
    assert store.state_path.read_text(encoding="utf-8") == good


# ── the audit log ────────────────────────────────────────────────────────
def test_every_change_is_appended_with_what_it_was_before(store):
    ap = store.create(a_point(), by="lukhashka")
    store.replace(ap.model_copy(update={"lifecycle": Lifecycle.rejected}),
                  by="lukhashka", what="lifecycle")
    store.delete(ap.id, by="olena")

    log = store.audit()
    assert [e.op for e in log] == ["create", "update", "delete"]
    assert [e.by for e in log] == ["op:lukhashka", "op:lukhashka", "op:olena"]
    assert all(e.feature_id == ap.id and e.site_key == "demo/pier" for e in log)
    assert log[0].before is None and log[0].after is not None
    assert log[1].before is not None and log[1].before["lifecycle"] == "verified"
    assert log[1].after is not None and log[1].after["lifecycle"] == "rejected"
    assert log[2].after is None
    # Append-only: one line per change, and the file is never rewritten.
    assert len(store.audit_path.read_text(encoding="utf-8").strip().splitlines()) == 3


def test_the_log_keeps_the_edit_history_on_the_record(store):
    ap = store.create(a_point())
    assert ap.provenance.edits == []
    once = store.replace(ap.model_copy(update={"notes": "перевірено"}), by="a", what="notes")
    twice = store.replace(once.model_copy(update={"notes": "ще раз"}), by="b", what="notes")
    assert [e.by for e in twice.provenance.edits] == ["op:a", "op:b"]
    assert [e.what for e in twice.provenance.edits] == ["notes", "notes"]


def test_who_created_a_feature_is_not_the_callers_to_rewrite(store):
    ap = store.create(a_point(created_by="op:lukhashka"))
    forged = ap.model_copy(deep=True)
    forged.provenance.created_by = "op:someone_else"
    assert store.replace(forged, by="mallory").provenance.created_by == "op:lukhashka"


def test_replace_many_writes_the_state_once_and_one_audit_line_each(store):
    ids = [store.create(a_point(u=u)).id for u in (-4.0, 0.0, 4.0)]
    updated = [store.get(i).model_copy(update={"notes": "ok"}) for i in ids]
    store.replace_many(updated, by="lukhashka", what="addressed", op="address")
    log = [e for e in store.audit() if e.op == "address"]
    assert len(log) == 3 and {e.feature_id for e in log} == set(ids)
    assert all(f.notes == "ok" for f in store.features())


def test_a_bad_line_in_the_log_does_not_hide_the_good_ones(store):
    store.create(a_point())
    with store.audit_path.open("a", encoding="utf-8") as fh:
        fh.write("not json at all\n\n")
    store.create(a_point(u=5.0))
    assert [e.op for e in store.audit()] == ["create", "create"]


def test_actor_falls_back_to_the_os_user(monkeypatch):
    assert actor("lukhashka") == "op:lukhashka"
    assert actor("svc:worker") == "svc:worker"
    monkeypatch.setenv("AEROSPLAT_OPERATOR", "olena")
    assert actor() == "op:olena"
    monkeypatch.delenv("AEROSPLAT_OPERATOR")
    monkeypatch.delenv("USERNAME", raising=False)
    monkeypatch.delenv("USER", raising=False)
    assert actor() == "op:unknown"


def test_authored_files_sit_outside_every_run_directory(tmp_path):
    from aerosplat.store.paths import StorePaths

    site = StorePaths(tmp_path).site("demo", "pier")
    run = site.run("baseline", "r1").root
    for path in (site.access_points, site.audit_log):
        assert path.parent == site.authored
        assert run not in path.parents            # ADR-0006: a run may be deleted
    assert AuthoredStore.for_site(site, "demo/pier").state_path == site.access_points
