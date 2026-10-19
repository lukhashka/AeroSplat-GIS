"""API v2 missions (aerosplat.server.api_mission): targets -> plan -> export, over HTTP.

The Site's newest run carries the synthetic street of tests/mission_scenes.py as its occupancy
product, so the endpoints are exercised against the same known answers as the planner tests.
Refusals must come back with their named reasons, and nothing infeasible, unaccepted or
unconfirmed may come out of the export endpoint.
"""

import pytest

pytest.importorskip("httpx")
from fastapi import FastAPI                                            # noqa: E402
from fastapi.testclient import TestClient                              # noqa: E402

import mission_scenes as ms                                            # noqa: E402
from aerosplat.authored.models import Lifecycle                        # noqa: E402
from aerosplat.authored.store import AuthoredStore                     # noqa: E402
from aerosplat.mission.export import enu_of_waypoints, read_parm      # noqa: E402
from aerosplat.server import api_mission, api_semantics, api_v2        # noqa: E402
from aerosplat.store import Catalog, Site                              # noqa: E402
from aerosplat.store.models import RunStatus                           # noqa: E402
from aerosplat.surface.models import EsdfInfo, OccupancyReport, OccupancyStats  # noqa: E402
from aerosplat.surface.occupancy import write_occupancy                # noqa: E402

BASE = "/api/v2/sites/demo/street"
QUIET = {"wire_height_m": 5.0}


@pytest.fixture(scope="module")
def street():
    return ms.street_grid()


@pytest.fixture
def env(tmp_path, monkeypatch, street):
    """A Site at ms.ORIGIN whose run r1 has the street as occupancy; (client, catalog root)."""
    root = tmp_path / "projects"
    monkeypatch.setenv("AEROSPLAT_PROJECTS", str(root))
    monkeypatch.setenv("AEROSPLAT_OPERATOR", "tester")
    cat = Catalog.open(root)
    cat.ensure_project("demo")
    site = cat.create_site(Site(id="street", project_id="demo", origin=ms.ORIGIN))
    cat.ensure_epoch(site)
    cat.create_run(site, "baseline", "r1", capture_ids=[], config={}, earth_model="wgs84",
                   georeferenced=True)
    occ = cat.run_dirs(site, "baseline", "r1").root / "occupancy"
    s = street.state
    stats = OccupancyStats(voxels=int(s.size), free=int((s == 0).sum()), occupied=int((s == 1).sum()),
                           unknown=int((s == 2).sum()), columns={"POINTS": s.shape[0] * s.shape[1]},
                           observed_fraction=1.0, assumed_fraction=0.0, closed_columns=0,
                           skyline_z=20.0, unknown_ceiling_z=21.0, ceiling_z=street.spec.z1,
                           n_points=0)
    write_occupancy(occ, street, OccupancyReport(
        grid=street.spec, bounds=street.spec.bounds(),
        esdf=EsdfInfo(max_distance_m=street.max_distance_m), stats=stats,
        params={"unknown": "skyline"}))
    ms.write_street_semantics(occ.parent / "semantics")         # for the takeoff search
    cat.finish_run(site, "baseline", "r1", status=RunStatus.ok, products={"occupancy": occ})
    authored = AuthoredStore.for_site(cat.site_dirs(site), site.key)
    missions = cat.site_dirs(site).missions
    cat.close()

    app = FastAPI()
    app.include_router(api_v2.router)
    app.include_router(api_semantics.router)
    app.include_router(api_mission.router)
    api_v2._CATALOGS.clear()
    with TestClient(app) as c:
        yield c, authored, missions
    for handle in api_v2._CATALOGS.values():
        handle.close()
    api_v2._CATALOGS.clear()


def target(client, ap_id, **kw):
    body = {"ap_id": ap_id, "sigma_ap_m": 0.1, "sigma_georef_m": 0.3, "confirm": True,
            "by": "lukhashka", **kw}
    return client.post(f"{BASE}/targets", json=body)


def mission(client, target_id, **kw):
    return client.post(f"{BASE}/missions", json={"target_id": target_id, "profile": QUIET, **kw})


def reasons(r):
    return {x["code"] for x in r.json()["detail"]["reasons"]}


# ── the whole way ────────────────────────────────────────────────────────
def test_click_to_waypoints(env):
    client, authored, missions = env
    ap = authored.create(ms.window_ap())
    t = target(client, ap.id)
    assert t.status_code == 201 and t.json()["state"] == "confirmed"
    m = mission(client, t.json()["id"])
    assert m.status_code == 201, m.text
    plan = m.json()
    assert plan["verdict"] == "feasible" and plan["occupancy"]["run_id"] == "r1"
    assert plan["altitude_frame"] == "relative_home"
    assert (missions / plan["id"] / "mission.json").is_file()
    assert client.get(f"{BASE}/missions/{plan['id']}").json()["id"] == plan["id"]
    assert [x["id"] for x in client.get(f"{BASE}/missions").json()] == [plan["id"]]

    wpl = client.post(f"{BASE}/missions/{plan['id']}/export/waypoints", json={"by": "lukhashka"})
    assert wpl.status_code == 200 and wpl.text.startswith("QGC WPL 110")
    assert "attachment" in wpl.headers["content-disposition"]
    back = enu_of_waypoints(wpl.text, ms.ORIGIN)
    hover = next(i + 1 for i, it in enumerate(plan["items"]) if it["role"] == "hover")
    assert back[hover] == pytest.approx(tuple(plan["approach"]["hover"]), abs=0.01)
    parm = read_parm(client.post(f"{BASE}/missions/{plan['id']}/export/parm", json={}).text)
    assert parm["FENCE_ENABLE"] == 1
    assert client.post(f"{BASE}/missions/{plan['id']}/export/plan", json={}).status_code == 200
    assert {p.name for p in (missions / plan["id"]).iterdir()} == \
        {"mission.json", "mission.waypoints", "mission.plan", "mission.parm"}
    assert [e.op for e in authored.audit()].count("mission_export") == 3


def test_a_draft_is_confirmed_through_the_api(env):
    client, authored, _ = env
    ap = authored.create(ms.roof_pad_ap())
    draft = target(client, ap.id, confirm=False).json()
    assert draft["state"] == "draft"
    assert mission(client, draft["id"]).status_code == 409
    done = client.post(f"{BASE}/targets/{draft['id']}/confirm", json={"by": "lukhashka"})
    assert done.status_code == 200 and done.json()["confirmed_by"] == "op:lukhashka"
    assert mission(client, draft["id"]).json()["verdict"] == "feasible"


# ── refusals ─────────────────────────────────────────────────────────────
def test_a_candidate_access_point_is_not_a_target(env):
    client, authored, _ = env
    ap = authored.create(ms.window_ap(lifecycle=Lifecycle.candidate))
    r = target(client, ap.id)
    assert r.status_code == 409 and reasons(r) == {"not_verified"}
    assert client.get(f"{BASE}/targets").json() == []


def test_sigma_over_the_gate_is_refused_with_its_reason(env):
    client, authored, _ = env
    ap = authored.create(ms.window_ap())
    r = target(client, ap.id, sigma_georef_m=2.5)
    assert r.status_code == 409 and reasons(r) == {"sigma_too_large"}


def test_an_unconfirmed_target_gets_no_mission(env):
    client, authored, missions = env
    ap = authored.create(ms.window_ap())
    draft = target(client, ap.id, confirm=False).json()
    r = mission(client, draft["id"])
    assert r.status_code == 409 and reasons(r) == {"target_not_confirmed"}
    assert not missions.exists() or not any(missions.iterdir())


def test_an_infeasible_mission_is_kept_but_not_exported(env):
    client, authored, _ = env
    ap = authored.create(ms.window_ap())
    t = target(client, ap.id).json()
    m = mission(client, t["id"], profile={**QUIET, "max_alt_rel_m": 5.0}).json()
    assert m["verdict"] == "infeasible"
    assert {f["code"] for f in m["findings"]} >= {"transit_too_high"}
    r = client.post(f"{BASE}/missions/{m['id']}/export/waypoints", json={})
    assert r.status_code == 409 and "transit_too_high" in r.text


def test_risks_must_be_accepted_by_name(env):
    client, authored, _ = env
    ap = authored.create(ms.window_ap())
    t = target(client, ap.id).json()
    m = mission(client, t["id"], profile={}).json()          # default: wire_risk at 9 m
    assert m["verdict"] == "feasible_with_risks"
    url = f"{BASE}/missions/{m['id']}/export/plan"
    assert client.post(url, json={}).status_code == 409
    ok = client.post(url, json={"accept_risks": True, "by": "lukhashka"})
    assert ok.status_code == 200
    last = authored.audit()[-1]
    assert last.op == "mission_export" and last.after["risks_accepted"] == ["wire_risk"]


def test_an_amsl_mission_is_refused(env):
    client, authored, _ = env
    ap = authored.create(ms.window_ap())
    t = target(client, ap.id).json()
    r = mission(client, t["id"], altitude_frame="amsl")
    assert r.status_code == 422 and "unverified" in r.text


def test_a_revoked_target_stops_the_export(env):
    client, authored, _ = env
    ap = authored.create(ms.window_ap())
    t = target(client, ap.id).json()
    m = mission(client, t["id"]).json()
    api_mission.TargetStore(authored).revoke(t["id"], by="t")
    r = client.post(f"{BASE}/missions/{m['id']}/export/waypoints", json={})
    assert r.status_code == 409 and "target_revoked" in r.text


@pytest.mark.parametrize("call", [
    lambda c: target(c, "ap_00000000000000000000000000"),
    lambda c: mission(c, "tgt_00000000000000000000000000"),
    lambda c: c.get(f"{BASE}/missions/msn_00000000000000000000000000"),
    lambda c: c.get(f"{BASE}/missions/..%2F..%2Fauthored"),
], ids=["access-point", "target", "mission", "traversal"])
def test_unknown_things_are_404(env, call):
    assert call(env[0]).status_code == 404


def test_an_unknown_export_format_is_404(env):
    client, authored, _ = env
    ap = authored.create(ms.window_ap())
    m = mission(client, target(client, ap.id).json()["id"]).json()
    assert client.post(f"{BASE}/missions/{m['id']}/export/kml", json={}).status_code == 404


def test_a_site_without_occupancy_cannot_plan(env, tmp_path):
    client, authored, _ = env
    cat = Catalog.open(tmp_path / "projects")
    other = cat.create_site(Site(id="empty", project_id="demo", origin=ms.ORIGIN))
    store = AuthoredStore.for_site(cat.site_dirs(other), other.key)
    cat.close()
    ap = store.create(ms.window_ap(site_key=other.key))
    t = client.post("/api/v2/sites/demo/empty/targets",
                    json={"ap_id": ap.id, "sigma_ap_m": 0.1, "sigma_georef_m": 0.3,
                          "confirm": True}).json()
    r = client.post("/api/v2/sites/demo/empty/missions", json={"target_id": t["id"]})
    assert r.status_code == 404 and "occupancy" in r.text


# ── takeoff spots, and the grid's edge ───────────────────────────────────
WIDE = {"wire_height_m": 5.0, "sigma_gnss_m": 15.0}         # r = 32.5 m: nothing fits the street


def test_takeoff_spots_for_a_target(env):
    client, authored, _ = env
    ap = authored.create(ms.window_ap())
    t = target(client, ap.id).json()
    r = client.post(f"{BASE}/targets/{t['id']}/takeoff", json={"profile": QUIET, "n": 3})
    assert r.status_code == 200, r.text
    res = r.json()
    assert len(res["candidates"]) == 3 and res["target"] == pytest.approx([0.0, 3.0, 9.0])
    assert res["radius_m"] == pytest.approx(5.559, abs=1e-3)
    assert res["candidates"][0]["distance_m"] < 1.0
    none = client.post(f"{BASE}/targets/{t['id']}/takeoff", json={"profile": WIDE}).json()
    assert none["candidates"] == [] and none["message"].startswith("no takeoff spot")
    assert none["best_open"]["transit_z"] is None                 # JSON-safe, never NaN
    assert client.post(f"{BASE}/targets/tgt_nope/takeoff", json={}).status_code == 404


def test_a_mission_can_take_its_home_from_the_search(env):
    client, authored, _ = env
    ap = authored.create(ms.window_ap())
    t = target(client, ap.id).json()
    best = client.post(f"{BASE}/targets/{t['id']}/takeoff",
                       json={"profile": QUIET}).json()["candidates"][0]
    m = mission(client, t["id"], home_search=True)
    assert m.status_code == 201, m.text
    plan = m.json()
    assert plan["home"]["enu"] == pytest.approx(best["enu"])
    assert plan["home"]["source"].startswith("takeoff search: nearest of 5 spots")
    assert not [f for f in plan["findings"]
                if f["leg"] == "takeoff" or f["code"].startswith("home_")]


def test_home_search_refuses_when_there_is_no_spot(env):
    client, authored, missions = env
    ap = authored.create(ms.window_ap())
    t = target(client, ap.id).json()
    r = client.post(f"{BASE}/missions", json={"target_id": t["id"], "profile": WIDE,
                                              "home_search": True})
    assert r.status_code == 409 and reasons(r) == {"no_takeoff_spot"}
    assert "no takeoff spot" in r.json()["detail"]["reasons"][0]["detail"]
    assert not missions.exists() or not any(missions.iterdir())    # nothing stored
    both = mission(client, t["id"], home_search=True, home_enu=[0.0, 0.0, 0.0])
    assert both.status_code == 422


def test_takeoff_search_needs_the_semantics_of_the_same_run(env, tmp_path):
    import shutil

    client, authored, _ = env
    ap = authored.create(ms.window_ap())
    t = target(client, ap.id).json()
    sem = tmp_path / "projects" / "demo" / "sites" / "street" / "epochs" / "baseline" / "runs" \
        / "r1" / "semantics"
    assert (sem / "semantics.json").is_file()
    shutil.rmtree(sem)
    r = client.post(f"{BASE}/targets/{t['id']}/takeoff", json={})
    assert r.status_code == 404 and "semantics" in r.json()["detail"]


def test_a_home_off_the_grid_is_refused_with_a_hint(env):
    client, authored, missions = env
    ap = authored.create(ms.window_ap())
    t = target(client, ap.id).json()
    r = mission(client, t["id"], home_enu=[100.0, 100.0, 0.0])
    assert r.status_code == 409 and reasons(r) == {"outside_grid"}
    assert "occupancy_bbox=" in r.json()["detail"]["reasons"][0]["detail"]
