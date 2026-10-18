"""Export for ArduPilot (aerosplat.mission.export): formats, altitude frames, round trips, fence.

SIMULATION §4.6 asks that a deliberately wrong altitude frame be caught by validation and not
in flight, and that the geofence hold the trajectory; ADR-0009 p. 5 that nothing leaves without
a confirmed target. Those are the negative cases here. The positive ones pin the formats down:
what comes back out of a file must land within 1 cm of the planned Site ENU point.
"""

import json
import math
from dataclasses import replace

import numpy as np
import pytest

import mission_scenes as ms
from aerosplat.mission.corridor import sample_path
from aerosplat.mission.export import (MAV_FRAME_GLOBAL, MAV_FRAME_GLOBAL_RELATIVE_ALT,
                                      ExportRefused, check_wire, distance_to_boundary,
                                      enu_from_geodetic, enu_of_waypoints, format_waypoints, point_in_polygon,
                                      read_parm, read_plan, read_waypoints, render, to_wire,
                                      wire_to_enu)
from aerosplat.mission.models import AltitudeFrame, AltitudeFrameError, PlanningProfile, Verdict
from aerosplat.mission.planner import plan_mission
from aerosplat.mission.store import MissionStore

QUIET = PlanningProfile(wire_height_m=5.0)


@pytest.fixture(scope="module")
def street():
    return ms.street_grid()


def make(tmp_path, grid, ap=None, profile=QUIET, **kw):
    ap = ap or ms.window_ap()
    target, authored, targets = ms.confirmed(tmp_path, ap)
    p = plan_mission(target=target, ap=ap, grid=grid, occupancy=ms.occupancy_ref(),
                     origin=ms.ORIGIN, by="op:tester", profile=profile, **kw)
    return p, authored, targets


@pytest.fixture(params=["window", "roof_pad"])
def feasible(request, tmp_path, street):
    ap = ms.window_ap() if request.param == "window" else ms.roof_pad_ap()
    p, _, _ = make(tmp_path, street, ap)
    assert p.verdict is Verdict.feasible, p.findings
    return p


def planned_points(p):
    return {seq: it.enu for seq, it in enumerate(p.items, start=1) if it.enu is not None}


def errors(back, p):
    want = planned_points(p)
    assert set(back) == set(want)
    return [float(np.linalg.norm(np.subtract(back[s], want[s]))) for s in want]


# ── positive: formats and round trips ────────────────────────────────────
def test_waypoints_round_trip_to_site_enu_within_a_centimetre(feasible):
    name, media, text = render(feasible, ms.ORIGIN, "waypoints")
    assert name == "mission.waypoints" and text.startswith("QGC WPL 110\n")
    assert max(errors(enu_of_waypoints(text, ms.ORIGIN), feasible)) < 0.01


def test_plan_round_trips_and_carries_fence_and_rally(feasible):
    _, media, text = render(feasible, ms.ORIGIN, "plan")
    doc = json.loads(text)
    assert media == "application/json" and doc["fileType"] == "Plan"
    m = doc["mission"]
    assert m["firmwareType"] == 3 and m["globalPlanAltitudeMode"] == 1
    assert all(i["AltitudeMode"] == 1 for i in m["items"] if "AltitudeMode" in i)
    assert max(errors(wire_to_enu(read_plan(text), ms.ORIGIN), feasible)) < 0.01
    poly = doc["geoFence"]["polygons"]
    assert len(poly) == 1 and poly[0]["inclusion"] is True and len(poly[0]["polygon"]) >= 3
    assert len(doc["rallyPoints"]["points"]) == 1
    assert doc["aerosplat"]["altitude_frame"] == "relative_home"
    assert doc["aerosplat"]["home_amsl_verified"] is False


def test_mission_item_int_quantisation_keeps_a_centimetre(feasible):
    # MAVLink MISSION_ITEM_INT carries degrees x 1e7: what the autopilot actually stores.
    q = [replace(w, lat=round(w.lat * 1e7) / 1e7, lon=round(w.lon * 1e7) / 1e7)
         for w in to_wire(feasible, ms.ORIGIN)]
    assert max(errors(wire_to_enu(q, ms.ORIGIN), feasible)) < 0.01


def test_relative_altitudes_are_above_the_home_point(feasible):
    items = to_wire(feasible, ms.ORIGIN)
    hz = feasible.home.enu[2]
    for w, it in zip(items[1:], feasible.items):
        if it.enu is not None:
            assert w.frame == MAV_FRAME_GLOBAL_RELATIVE_ALT
            # Over a few tens of metres the ellipsoid and the ENU plane agree to millimetres.
            assert w.alt == pytest.approx(it.enu[2] - hz, abs=0.005)
    assert items[0].frame == MAV_FRAME_GLOBAL                 # home: absolute, by the format
    assert items[0].alt == pytest.approx(feasible.home.h_ellipsoid)
    assert items[-1].command == 21 and items[-1].alt == pytest.approx(0.0, abs=1e-6)


def test_the_geofence_contains_the_whole_corridor(feasible):
    poly = feasible.fence.polygon_enu
    for leg in feasible.legs:
        for p in sample_path(np.asarray(leg.points), 0.25):
            assert point_in_polygon(p[0], p[1], poly), (leg.name, p)
            assert distance_to_boundary(p[0], p[1], poly) >= leg.radius_m, (leg.name, p)
    top = max(it.enu[2] for it in feasible.items if it.enu is not None) - feasible.home.enu[2]
    assert feasible.fence.alt_max_rel_m >= top + 1.0
    reach = max(math.hypot(x - feasible.home.enu[0], y - feasible.home.enu[1]) for x, y in poly)
    assert feasible.fence.radius_m >= reach


def test_the_fence_polygon_in_the_plan_file_is_the_planned_one(feasible):
    from aerosplat.geo.crs import EnuFrame
    doc = json.loads(render(feasible, ms.ORIGIN, "plan")[2])
    ll = np.asarray(doc["geoFence"]["polygons"][0]["polygon"])
    back = enu_from_geodetic(EnuFrame(ms.ORIGIN), ll[:, 0], ll[:, 1],
                             np.full(len(ll), feasible.home.h_ellipsoid))
    assert np.abs(back[:, :2] - np.asarray(feasible.fence.polygon_enu)).max() < 0.01


def test_parm_package_matches_the_plan(feasible):
    params = read_parm(render(feasible, ms.ORIGIN, "parm")[2])
    z_rel = feasible.transit_z - feasible.home.enu[2]
    assert params["FENCE_ENABLE"] == 1 and params["FENCE_TYPE"] == 7
    assert params["FENCE_ALT_MAX"] == feasible.fence.alt_max_rel_m > z_rel
    assert params["RTL_ALT"] == math.ceil(z_rel * 100)                 # centimetres
    assert params["RTL_ALT"] / 100 < params["FENCE_ALT_MAX"]          # RTL stays inside the fence
    assert params["FENCE_RADIUS"] >= 30
    assert params["FS_THR_ENABLE"] == 4 and params["BATT_FS_CRT_ACT"] == 1


def test_export_writes_files_and_audits(tmp_path, street):
    p, authored, targets = make(tmp_path, street)
    store = MissionStore(tmp_path / "missions", ms.SITE)
    store.save(p)
    assert store.load(p.id) == p and store.ids() == [p.id]
    for fmt in ("waypoints", "plan", "parm"):
        path, _, _ = store.export(p, fmt, origin=ms.ORIGIN, targets=targets, by="lukhashka")
        assert path.parent == tmp_path / "missions" / p.id and path.is_file()
    exports = [e for e in authored.audit() if e.op == "mission_export"]
    assert len(exports) == 3 and exports[0].by == "op:lukhashka"
    assert exports[0].after["mission_id"] == p.id


# ── negative: what may not leave ─────────────────────────────────────────
def test_an_infeasible_mission_is_not_exported(tmp_path, street):
    p, _, _ = make(tmp_path, street, profile=PlanningProfile(wire_height_m=5.0, max_alt_rel_m=5.0))
    assert p.verdict is Verdict.infeasible
    for fmt in ("waypoints", "plan", "parm"):
        with pytest.raises(ExportRefused, match="transit_too_high"):
            render(p, ms.ORIGIN, fmt)


def test_risks_need_a_person_to_accept_them(tmp_path, street):
    p, _, _ = make(tmp_path, street, profile=PlanningProfile())          # wire_risk
    assert p.verdict is Verdict.feasible_with_risks
    with pytest.raises(ExportRefused, match="wire_risk"):
        render(p, ms.ORIGIN, "waypoints")
    assert render(p, ms.ORIGIN, "waypoints", accept_risks=True)[2].startswith("QGC WPL 110")


def test_a_target_revoked_after_planning_stops_the_export(tmp_path, street):
    p, _, targets = make(tmp_path, street)
    targets.revoke(p.target.id, by="t", reason="customer cancelled")
    with pytest.raises(ExportRefused, match="target_revoked"):
        MissionStore(tmp_path / "m", ms.SITE).export(p, "waypoints", origin=ms.ORIGIN,
                                                     targets=targets)


def test_an_access_point_edited_after_planning_stops_the_export(tmp_path, street):
    p, authored, targets = make(tmp_path, street)
    ap = authored.get(p.target.ap_id)
    moved = ap.geometry.model_copy(update={"center": (2.0, 15.0, 9.0)})
    authored.replace(ap.model_copy(update={"geometry": moved}, deep=True))
    with pytest.raises(ExportRefused, match="target_changed"):
        MissionStore(tmp_path / "m", ms.SITE).export(p, "plan", origin=ms.ORIGIN, targets=targets)


def test_an_unknown_format_is_refused(tmp_path, street):
    p, _, _ = make(tmp_path, street)
    with pytest.raises(ValueError, match="unknown format"):
        render(p, ms.ORIGIN, "kml")


# ── negative: altitude frames (SIMULATION §4.6 "Кадри висот") ────────────
def test_an_amsl_export_is_refused(tmp_path, street):
    p, _, _ = make(tmp_path, street)
    amsl = p.model_copy(update={"altitude_frame": AltitudeFrame.amsl})
    with pytest.raises(AltitudeFrameError, match="unverified"):
        render(amsl, ms.ORIGIN, "waypoints")


def test_absolute_altitudes_in_relative_items_are_caught(tmp_path, street):
    # The classic mix-up: AMSL-ish numbers (home height + z) written into relative items.
    p, _, _ = make(tmp_path, street)
    items = to_wire(p, ms.ORIGIN)
    mixed = [items[0]] + [replace(w, alt=w.alt + items[0].alt) if w.frame == 3 else w
                          for w in items[1:]]
    with pytest.raises(AltitudeFrameError, match="wrong altitude frame"):
        check_wire(p, ms.ORIGIN, mixed)


def test_a_single_item_in_another_frame_is_caught_when_reading(tmp_path, street):
    p, _, _ = make(tmp_path, street)
    items = to_wire(p, ms.ORIGIN)
    k = next(i for i, w in enumerate(items) if i and w.frame == MAV_FRAME_GLOBAL_RELATIVE_ALT)
    items[k] = replace(items[k], frame=MAV_FRAME_GLOBAL, alt=items[k].alt + items[0].alt)
    with pytest.raises(AltitudeFrameError, match=f"item {k}"):
        read_waypoints(format_waypoints(items))


def test_a_home_row_that_is_not_absolute_is_caught(tmp_path, street):
    p, _, _ = make(tmp_path, street)
    items = to_wire(p, ms.ORIGIN)
    items[0] = replace(items[0], frame=MAV_FRAME_GLOBAL_RELATIVE_ALT)
    with pytest.raises(AltitudeFrameError, match="home row"):
        read_waypoints(format_waypoints(items))


@pytest.mark.parametrize("where, value", [("global", 2), ("item", 2), ("item", 0)])
def test_a_plan_file_in_another_altitude_mode_is_caught(tmp_path, street, where, value):
    p, _, _ = make(tmp_path, street)
    doc = json.loads(render(p, ms.ORIGIN, "plan")[2])
    if where == "global":
        doc["mission"]["globalPlanAltitudeMode"] = value
    else:
        next(i for i in doc["mission"]["items"] if "AltitudeMode" in i)["AltitudeMode"] = value
    with pytest.raises(AltitudeFrameError):
        read_plan(json.dumps(doc))


@pytest.mark.parametrize("mangle, match", [
    (lambda t: t.replace("QGC WPL 110", "QGC WPL 100"), "not a"),
    (lambda t: t.replace("\t1\n", "\n", 1), "columns"),
    (lambda t: t.replace("\n1\t", "\n7\t", 1), "seq"),
    (lambda t: t.replace("\t9.000\t", "\tnan\t", 1), "NaN"),
], ids=["header", "columns", "seq", "nan"])
def test_malformed_waypoint_files_are_refused(tmp_path, street, mangle, match):
    p, _, _ = make(tmp_path, street)
    text = render(p, ms.ORIGIN, "waypoints")[2]
    with pytest.raises(ValueError, match=match):
        read_waypoints(mangle(text))
