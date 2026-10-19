"""API v2 flights (aerosplat.server.api_sim) and the replay labels (aerosplat.sim.replay): what
the web UI's "Місія / проліт" page draws, read from a mission folder and its SITL runs.

The mission is the synthetic street of tests/sim_scenarios.py (window on the south wall of the
building north of home, H at (0, 12, 9), E at (0, 3, 9)); its runs are written here the way
aerosplat.sim.report writes them, from flights flown exactly along the plan (ss.fly). A
waypoint served by the API must be the row of the exported mission.waypoints, and the frames
must agree: Site ENU <-> WGS84 through the same EnuFrame the export used.
"""

import json

import numpy as np
import pytest

pytest.importorskip("httpx")
from fastapi import FastAPI                                            # noqa: E402
from fastapi.testclient import TestClient                              # noqa: E402

import mission_scenes as ms                                            # noqa: E402
import sim_scenarios as ss                                             # noqa: E402
from aerosplat.geo.crs import EnuFrame                                 # noqa: E402
from aerosplat.mission.export import enu_from_geodetic, read_waypoints  # noqa: E402
from aerosplat.server import api_sim, api_v2                           # noqa: E402
from aerosplat.sim import replay                                       # noqa: E402
from aerosplat.sim.oracles import clearance_series                     # noqa: E402
from aerosplat.sim.record import Recording                             # noqa: E402
from aerosplat.sim.report import (CLEARANCE_JSON, backfill_clearance,  # noqa: E402
                                  clearance_dict, write_clearance)
from aerosplat.store import Catalog, Site                              # noqa: E402
from aerosplat.surface.models import EsdfInfo, OccupancyReport, OccupancyStats  # noqa: E402
from aerosplat.surface.occupancy import write_occupancy                # noqa: E402

BASE = "/api/v2/sites/demo/street"
DELIVERED = {"outcome": "complete", "reason": "delivered"}


@pytest.fixture(scope="module")
def grid():
    return ms.street_grid()


@pytest.fixture(scope="module")
def plan(tmp_path_factory, grid):
    return ss.window_plan(tmp_path_factory.mktemp("plan"), grid)


@pytest.fixture(scope="module")
def flight(plan):
    return ss.fly(plan)


@pytest.fixture
def env(tmp_path, monkeypatch, plan):
    """A Site at ms.ORIGIN with the window mission exported into its missions/; (client, folder)."""
    root = tmp_path / "projects"
    monkeypatch.setenv("AEROSPLAT_PROJECTS", str(root))
    cat = Catalog.open(root)
    cat.ensure_project("demo")
    site = cat.create_site(Site(id="street", project_id="demo", origin=ms.ORIGIN))
    missions = cat.site_dirs(site).missions
    cat.close()
    folder = ss.export(plan, missions / plan.id)

    app = FastAPI()
    app.include_router(api_v2.router)
    app.include_router(api_sim.router)
    api_v2._CATALOGS.clear()
    with TestClient(app) as c:
        yield c, folder
    for handle in api_v2._CATALOGS.values():
        handle.close()
    api_v2._CATALOGS.clear()


def write_run(folder, plan, rec, name="street-window-delivered", *, seed=None, grid=None,
              passed=True, judged=None):
    """A run as aerosplat.sim.report leaves it: report.json, trajectory.json, clearance.json."""
    out = folder / "sim" / name / (f"seed-{seed}" if seed is not None else "")
    out.mkdir(parents=True)
    rec.meta = {"scenario": name, "speedup": 3.0, "origin": ms.ORIGIN.model_dump(),
                "home_enu": list(plan.home.enu), "mission": plan.id}
    rec.save(out / "trajectory.json")
    metrics = {}
    if grid is not None:
        series = clearance_series(rec, ss.context(plan, DELIVERED, grid=grid))
        write_clearance(series, out)
        metrics = {"min_clearance_m": float(np.min(series.clearance)), "at_t": float(series.t[0])}
    if judged is not None:
        metrics["min_clearance_m"] = judged
    report = {"schema": 1, "scenario": ss.scenario(name, DELIVERED).model_dump(mode="json"),
              "seed": seed or 0, "passed": passed, "summary": "delivered: all invariants hold",
              "outcome": "complete", "reason": None, "detail": "",
              "expected": {"outcome": "complete", "reason": "delivered",
                           "invariants": {"clearance": "pass"}},
              "verdicts": [{"invariant": "clearance", "status": "pass", "detail": "ok",
                            "expected": "pass", "metrics": metrics}],
              "timing": {"mission_sim_s": float(rec.t[-1] - rec.t[0])},
              "firmware": {"tag": "Copter-4.7.2"}, "warnings": [], "faults": [],
              "events": [], "samples": len(rec)}
    (out / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return out


# ── geometry: frames ─────────────────────────────────────────────────────
def test_a_waypoint_is_where_the_exported_file_puts_it(env, plan):
    client, folder = env
    g = client.get(f"{BASE}/missions/{plan.id}/geometry")
    assert g.status_code == 200, g.text
    g = g.json()
    rows = {w.seq: w for w in read_waypoints((folder / "mission.waypoints").read_text(encoding="utf-8"))}
    assert [w["seq"] for w in g["waypoints"]] == [k for k, it in enumerate(plan.items, start=1)
                                                  if it.enu is not None]
    for w in g["waypoints"]:
        row = rows[w["seq"]]
        # mission.waypoints carries 9 decimals of a degree (0.1 mm) and 3 of a metre
        assert w["lat"] == pytest.approx(row.lat, abs=1e-9)
        assert w["lon"] == pytest.approx(row.lon, abs=1e-9)
        assert w["alt_rel_m"] == pytest.approx(row.alt, abs=1e-3)
    hover = next(w for w in g["waypoints"] if w["role"] == "hover")
    assert hover["enu"] == pytest.approx(list(plan.approach.hover), abs=1e-12)
    assert g["target"]["hover"]["enu"] == pytest.approx(list(plan.approach.hover), abs=1e-12)
    # ... and back: WGS84 -> Site ENU lands on H to the millimetre
    back = enu_from_geodetic(EnuFrame(ms.ORIGIN), hover["lat"], hover["lon"], hover["h_ellipsoid"])[0]
    assert back == pytest.approx(list(plan.approach.hover), abs=1e-3)
    assert g["origin"] == ms.ORIGIN.model_dump()
    assert g["frame"] == "site_enu"


def test_the_target_is_drawn_on_its_facade(env, plan):
    client, _ = env
    t = client.get(f"{BASE}/missions/{plan.id}/geometry").json()["target"]
    ap = plan.access_point
    outline = np.asarray(t["polygon_enu"], float)
    assert len(outline) == len(ap.geometry.polygon_uv) == 4
    n = np.asarray(ap.host.normal, float)
    o = np.asarray(ap.host.origin, float)
    assert (outline - o) @ n == pytest.approx(np.zeros(4), abs=1e-9)       # on the wall plane
    assert sorted(outline[:, 2]) == pytest.approx(sorted(v for _, v in ap.geometry.polygon_uv))
    centre = (outline.min(axis=0) + outline.max(axis=0)) / 2
    assert centre == pytest.approx(list(ap.geometry.center), abs=1e-9)
    hover = np.asarray(t["hover"]["enu"], float)
    assert float((hover - o) @ n) == pytest.approx(plan.target.handover.standoff_m, abs=1e-6)
    assert t["heading_deg"] == pytest.approx(plan.approach.heading_deg)
    assert t["kind"] == "opening" and t["facade_id"] == ap.host.facade_id
    assert len(t["lonlat"]) == 4


def test_fence_legs_and_frame_box(env, plan):
    client, _ = env
    g = client.get(f"{BASE}/missions/{plan.id}/geometry").json()
    hz = plan.home.enu[2]
    assert g["fence"]["floor_z"] == pytest.approx(hz)
    assert g["fence"]["ceiling_z"] == pytest.approx(hz + plan.fence.alt_max_rel_m)
    assert len(g["fence"]["lonlat"]) == len(plan.fence.polygon_enu)
    assert [leg["name"] for leg in g["legs"]] == [leg.name for leg in plan.legs]
    assert [leg["radius_m"] for leg in g["legs"]] == [leg.radius_m for leg in plan.legs]
    lo, hi = np.asarray(g["bbox_enu"][0]), np.asarray(g["bbox_enu"][1])
    for w in g["waypoints"]:
        assert np.all(lo <= np.asarray(w["enu"])) and np.all(np.asarray(w["enu"]) <= hi)
    assert g["twin"] == {"run_id": plan.occupancy.run_id, "epoch_id": plan.occupancy.epoch_id,
                         "format": plan.occupancy.format, "res_m": plan.occupancy.res_m,
                         "scene": "demo/street", "legacy": False}


def test_a_mission_planned_on_a_legacy_cloud_points_at_that_dataset(plan):
    site = Site(id="dataset_v6", project_id="default", origin=ms.ORIGIN)
    legacy = plan.model_copy(update={"occupancy": plan.occupancy.model_copy(
        update={"run_id": "legacy:dataset_v6_dense.pt"})})
    twin = api_sim.twin_scene(legacy, site)
    assert (twin.scene, twin.legacy) == ("dataset_v6", True)
    assert api_sim.twin_scene(plan, site).scene == "default/dataset_v6"


# ── runs ─────────────────────────────────────────────────────────────────
def test_runs_are_listed_with_their_seeds(env, plan, flight, grid):
    client, folder = env
    write_run(folder, plan, flight, grid=grid)
    write_run(folder, plan, ss.fly(plan), "street-window-gps-loss", seed=0, passed=False)
    write_run(folder, plan, ss.fly(plan), "street-window-gps-loss", seed=1)
    (folder / "sim" / "not a scenario").mkdir()                        # ignored, not an error
    runs = client.get(f"{BASE}/missions/{plan.id}/sims").json()
    assert [(r["scenario"], r["seed"]) for r in runs] == [
        ("street-window-delivered", None), ("street-window-gps-loss", 0),
        ("street-window-gps-loss", 1)]
    assert [r["has_clearance"] for r in runs] == [True, False, False]
    assert [r["passed"] for r in runs] == [True, False, True]
    assert runs[0]["verdicts"] == [{"invariant": "clearance", "status": "pass",
                                    "expected": "pass", "detail": "ok"}]
    assert runs[0]["firmware"] == "Copter-4.7.2" and runs[0]["samples"] == len(flight)
    assert client.get(f"{BASE}/sims").json() == runs
    report = client.get(f"{BASE}/missions/{plan.id}/sims/street-window-gps-loss/report?seed=1")
    assert report.status_code == 200 and report.json()["seed"] == 1


def test_a_mission_without_runs_has_none(env, plan):
    client, _ = env
    assert client.get(f"{BASE}/missions/{plan.id}/sims").json() == []
    assert client.get(f"{BASE}/sims").json() == []


def test_what_is_not_there_is_404(env, plan, flight):
    client, folder = env
    write_run(folder, plan, flight)
    m = f"{BASE}/missions/{plan.id}"
    assert client.get(f"{m}/sims/street-window-delivered/report").status_code == 200
    assert client.get(f"{m}/sims/nope/trajectory").status_code == 404
    assert client.get(f"{m}/sims/street-window-delivered/report?seed=3").status_code == 404
    assert client.get(f"{m}/sims/..%2F..%2Fmission.json/report").status_code == 404
    assert client.get(f"{m}/sims/Street/report").status_code == 404     # not a scenario name
    assert client.get(f"{BASE}/missions/msn_nope/geometry").status_code == 404
    assert client.get("/api/v2/sites/demo/nowhere/sims").status_code == 404


# ── the trajectory as the page replays it ────────────────────────────────
def test_phases_follow_the_plan_and_a_failsafe_is_marked(env, plan, flight):
    client, folder = env
    # GNSS lost while holding at H: the vehicle leaves AUTO and lands where it is (H-011)
    t_hold = next(e.t for e in flight.events if e.kind == "current" and e.detail == "hover")
    rec = ss.with_mode(flight, t_hold + 15.0, "LAND")
    write_run(folder, plan, rec)
    tr = client.get(f"{BASE}/missions/{plan.id}/sims/street-window-delivered/trajectory")
    assert tr.status_code == 200, tr.text
    tr = tr.json()
    n = len(rec)
    assert all(len(tr[k]) == n for k in ("t", "enu", "lonlat", "alt_rel_m", "speed_mps", "phase",
                                         "failsafe", "mode", "yaw_deg", "seq"))
    # The street's E is at transit height: transit ends at E, the approach starts there, and
    # the hold at H is told from the approach to it.
    segs = tr["segments"]
    assert [(s["phase"], s["failsafe"], s["mode"]) for s in segs] == [
        ("takeoff", False, "AUTO"), ("transit", False, "AUTO"), ("approach", False, "AUTO"),
        ("hold", False, "AUTO"), ("hold", True, "LAND"), ("ground", False, "LAND")]
    held = [k for k, (p, f) in enumerate(zip(tr["phase"], tr["failsafe"])) if p == "hold" and not f]
    h = np.asarray(plan.approach.hover)
    assert max(np.linalg.norm(np.asarray(tr["enu"][k]) - h) for k in held) <= 1.0
    # the failsafe: armed, after the mission start, not in AUTO — and it keeps its phase
    fs = np.asarray(tr["failsafe"])
    assert all(tr["mode"][k] == "LAND" and tr["phase"][k] == "hold" for k in np.flatnonzero(fs))
    assert not any(fs[k] for k, a in enumerate(tr["armed"]) if not a)
    # segments tile the recording without gaps
    assert segs[0]["i0"] == 0 and segs[-1]["i1"] == n - 1
    assert all(a["i1"] == b["i0"] for a, b in zip(segs, segs[1:]))
    assert tr["phases"] == ["ground", "takeoff", "transit", "approach", "hold"]
    # heights above home, and WGS84 that converts back onto the truth
    assert tr["alt_rel_m"] == pytest.approx([p[2] - plan.home.enu[2] for p in rec.enu])
    lon, lat = np.asarray(tr["lonlat"]).T
    back = enu_from_geodetic(EnuFrame(ms.ORIGIN), lat, lon, np.full(n, ms.ORIGIN.h))
    assert back[:, :2] == pytest.approx(np.asarray(rec.enu)[:, :2], abs=0.02)   # h unused on the map
    assert tr["origin_offset_m"] == 0.0 and tr["mission_start_t"] == 10.0


def test_a_whole_delivery_reads_in_flight_order(plan, flight):
    fs = replay.failsafe_mask(flight)
    ph = replay.phases(flight, plan, failsafe=fs)
    order = [s.phase for s in replay.segments(ph, fs, flight.mode)]
    assert order == ["takeoff", "transit", "approach", "hold", "exit", "return", "land", "ground"]
    assert not fs.any()


def test_clearance_comes_with_the_samples_it_belongs_to(env, plan, flight, grid):
    client, folder = env
    out = write_run(folder, plan, flight, grid=grid)
    stored = json.loads((out / CLEARANCE_JSON).read_text(encoding="utf-8"))
    tr = client.get(f"{BASE}/missions/{plan.id}/sims/street-window-delivered/trajectory").json()
    c = tr["clearance"]
    assert len(c["clearance_m"]) == len(tr["t"])
    by_t = dict(zip(stored["t"], stored["clearance_m"]))
    for t, v in zip(tr["t"], c["clearance_m"]):
        assert v == by_t.get(round(t, 3))                    # None outside the airborne samples
    assert c["clearance_m"][-1] is None                      # disarmed
    assert sum(v is not None for v in c["clearance_m"]) == len(stored["t"])
    assert c["required_m"] == pytest.approx(plan.profile.r_drone_m + min(
        plan.target.handover.min_clearance_m, plan.profile.transit_margin_m))
    # a run recorded without clearance.json still replays, without it
    (out / CLEARANCE_JSON).unlink()
    assert client.get(f"{BASE}/missions/{plan.id}/sims/street-window-delivered/trajectory"
                      ).json()["clearance"] is None


# ── replay labels ────────────────────────────────────────────────────────
def test_do_items_take_the_phase_of_the_leg_they_start(plan):
    by_seq = replay.phase_by_seq(plan)
    nav = [seq for seq, it in enumerate(plan.items, start=1) if it.enu is not None]
    for seq, it in enumerate(plan.items, start=1):
        if it.enu is None:
            assert by_seq[seq] == by_seq[next(s for s in nav if s > seq)]
    roles = {it.role: seq for seq, it in enumerate(plan.items, start=1)}
    assert by_seq[roles["over_exit"]] == "exit"            # the leg right after H, whatever its name
    assert by_seq[roles["over_entry"]] == "transit" and by_seq[roles["land"]] == "land"


def test_the_mission_rewinding_on_touchdown_is_still_the_landing(plan, flight):
    rec = Recording.from_dict(flight.to_dict())
    last_armed = max(k for k, a in enumerate(rec.armed) if a)
    t, p = rec.t[last_armed], rec.enu[last_armed]
    for k in range(3):        # Copter: "Mission: 1 Takeoff" after touchdown, still armed
        rec.add(t=t + 0.01 * (k + 1), enu=p, vel_ned=(0, 0, 0), yaw_deg=0.0, mode="AUTO",
                armed=True, seq=1)
    order = sorted(range(len(rec)), key=lambda k: rec.t[k])
    rec = Recording(t=[rec.t[k] for k in order], enu=[rec.enu[k] for k in order],
                    vel_ned=[rec.vel_ned[k] for k in order], yaw_deg=[rec.yaw_deg[k] for k in order],
                    mode=[rec.mode[k] for k in order], armed=[rec.armed[k] for k in order],
                    seq=[rec.seq[k] for k in order], est_enu=[None] * len(order), events=rec.events)
    ph = replay.phases(rec, plan)
    armed = [k for k, a in enumerate(rec.armed) if a]
    assert ph[armed[0]] == "takeoff"
    assert ph[armed[-1]] == "land" and ph[armed[-3]] == "land"
    assert ph[-1] == replay.GROUND


# ── clearance.json ───────────────────────────────────────────────────────
def test_clearance_json_says_null_rather_than_nan():
    from aerosplat.sim.oracles import ClearanceSeries

    s = ClearanceSeries(t=np.array([1.0, 1.1]), clearance=np.array([3.0, 4.0]),
                        target_distance=np.array([np.nan, 2.5]), required=2.0, state=["free", "free"],
                        cap=4.0)
    d = clearance_dict(s)
    assert d["target_distance_m"] == [None, 2.5] and d["cap_m"] == 4.0
    assert clearance_dict(ClearanceSeries(np.zeros(0), np.zeros(0), np.zeros(0), 2.0, []))["cap_m"] is None


def _occupancy(folder, grid):
    s = grid.state
    stats = OccupancyStats(voxels=int(s.size), free=int((s == 0).sum()), occupied=int((s == 1).sum()),
                           unknown=int((s == 2).sum()), columns={"POINTS": s.shape[0] * s.shape[1]},
                           observed_fraction=1.0, assumed_fraction=0.0, closed_columns=0,
                           skyline_z=20.0, unknown_ceiling_z=21.0, ceiling_z=grid.spec.z1, n_points=0)
    write_occupancy(folder, grid, OccupancyReport(
        grid=grid.spec, bounds=grid.spec.bounds(), esdf=EsdfInfo(max_distance_m=grid.max_distance_m),
        stats=stats, params={"unknown": "skyline"}))
    return folder


def test_backfill_writes_what_the_runner_would_have(tmp_path, plan, flight, grid):
    mdir = ss.export(plan, tmp_path / plan.id)
    expected = write_run(mdir, plan, flight, grid=grid)
    want = json.loads((expected / CLEARANCE_JSON).read_text(encoding="utf-8"))
    (expected / CLEARANCE_JSON).unlink()
    occ = _occupancy(tmp_path / "occ", grid)
    path = backfill_clearance(expected, occ)
    assert path == expected / CLEARANCE_JSON
    assert json.loads(path.read_text(encoding="utf-8")) == want


def test_backfill_refuses_a_grid_the_run_was_not_judged_on(tmp_path, plan, flight, grid):
    mdir = ss.export(plan, tmp_path / plan.id)
    run = write_run(mdir, plan, flight, judged=0.42)
    with pytest.raises(ValueError, match="not the occupancy it flew against"):
        backfill_clearance(run, _occupancy(tmp_path / "occ", grid))
    assert not (run / CLEARANCE_JSON).exists()
