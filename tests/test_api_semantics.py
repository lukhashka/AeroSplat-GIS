"""API v2 semantics: taxonomy, facades, facade orthophotos, authored access points.

The render service is stubbed: what matters here is that the API builds the right pose
from the stored facade and hands back a grid a pixel can be read through, not what the
GPU draws. ACCESS_POINTS §3-§5, ADR-0006 (authored data outlives runs).
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

pytest.importorskip("httpx")
from fastapi import FastAPI                                            # noqa: E402
from fastapi.testclient import TestClient                              # noqa: E402

from aerosplat.geo.crs import GeoPoint                                 # noqa: E402
from aerosplat.semantics.models import (COVERAGE_NPZ, SEMANTICS_JSON, Building,  # noqa: E402
                                        Coverage, Facade, SemanticsResult, SurfaceKind,
                                        TerrainModel, azimuth_of, u_axis_for)
from aerosplat.server import api_semantics, api_v2                     # noqa: E402
from aerosplat.store import Catalog, Site                              # noqa: E402
from aerosplat.store.models import RunStatus                           # noqa: E402
from aerosplat.store.scenes import SceneLookupError                    # noqa: E402

ORIGIN = GeoPoint(lat=40.8942, lon=-74.2173, h=233.0)
BASE = "/api/v2/sites/demo/pier"


@pytest.fixture
def stub_render():
    """Stands in for the render service, and records the pose it was asked for."""
    seen: dict = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            # Like the real service, the meta names the run it rendered; `wrong_run` makes
            # the stub misbehave as a service older than the API would.
            run = seen.get("wrong_run", seen["body"].get("run"))
            self.send_header("X-Render-Meta", json.dumps({"stub": True, "width": 1, "height": 1,
                                                          "run": run}))
            self.end_headers()
            self.wfile.write(b"\x89PNG-stub")

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", seen
    srv.shutdown()


def _facade(fid, normal, origin, u_min=-10.0, u_max=10.0, height=12.0, coverage=None):
    n = np.asarray(normal, float)
    ux, uy = u_axis_for((float(n[0]), float(n[1])))
    return Facade(id=fid, building_id=fid.split("/")[0], kind=SurfaceKind.wall, origin=origin,
                  normal=(float(n[0]), float(n[1]), 0.0), u_axis=(ux, uy, 0.0),
                  azimuth_deg=azimuth_of((float(n[0]), float(n[1]))), u_min=u_min, u_max=u_max,
                  v_ground=origin[2], v_max=origin[2] + height, plane_rms_m=0.09,
                  n_points=4200, coverage=coverage)


@pytest.fixture
def site(tmp_path, monkeypatch, stub_render):
    """A Site whose newest run produced semantics: one building, two facades."""
    monkeypatch.setenv("AEROSPLAT_PROJECTS", str(tmp_path / "projects"))
    monkeypatch.setenv("AEROSPLAT_RENDER_URL", stub_render[0])
    monkeypatch.setenv("AEROSPLAT_OPERATOR", "tester")

    cat = Catalog.open(tmp_path / "projects")
    cat.ensure_project("demo")
    s = cat.create_site(Site(id="pier", project_id="demo", origin=ORIGIN))
    cat.ensure_epoch(s)
    cat.create_run(s, "baseline", "r1", capture_ids=[], config={}, earth_model="wgs84",
                   georeferenced=True)
    dirs = cat.run_dirs(s, "baseline", "r1")
    sem = dirs.root / "semantics"
    sem.mkdir(parents=True)

    cov = Coverage(cell_m=0.5, cells=960, visible=900, covered=700, mean_views=8.0,
                   max_gsd_cm=5.0, best_gsd_cm=2.1, median_angle_deg=28.0)
    north = _facade("B01/N1", (0.0, 1.0), (0.0, 0.0, 0.0), coverage=cov)
    east = _facade("B01/E1", (1.0, 0.0), (10.0, 0.0, 0.0), u_min=-6.0, u_max=6.0,
                   coverage=cov.model_copy(update={"covered": 0, "best_gsd_cm": 11.0}))
    building = Building(id="B01", footprint=[(-10, -12), (10, -12), (10, 0), (-10, 0)],
                        area_m2=240.0, ground_z=0.0, roof_z=12.0, height_m=12.0,
                        facade_ids=["B01/E1", "B01/N1"])
    result = SemanticsResult(
        terrain=TerrainModel(res_m=1.0, x0=-40.0, y0=-40.0, rows=80, cols=80,
                             n_points=900_000, filled_cells=5400),
        buildings=[building], facades=[east, north], warnings=["зйомка задалеко"])
    (sem / SEMANTICS_JSON).write_text(result.model_dump_json(), encoding="utf-8")
    grid = np.full((24, 40), 7, np.int32)
    grid[16:, 24:] = 0                                   # the top-right corner is invention
    np.savez_compressed(sem / COVERAGE_NPZ, **{"B01/N1": grid})
    splat = dirs.gs
    splat.mkdir(parents=True)
    (splat / "splat_wgs84.pt").write_bytes(b"stub")
    cat.finish_run(s, "baseline", "r1", status=RunStatus.ok,
                   products={"semantics": sem, "splat": splat / "splat_wgs84.pt"})
    cat.close()

    app = FastAPI()
    app.include_router(api_v2.router)
    app.include_router(api_semantics.router)
    api_v2._CATALOGS.clear()
    with TestClient(app) as c:
        yield c, stub_render[1]
    for handle in api_v2._CATALOGS.values():
        handle.close()
    api_v2._CATALOGS.clear()


@pytest.fixture
def client(site):
    return site[0]


def draw(client, u0, v0, u1, v1, class_id="opening.window", facade="B01/N1", **extra):
    body = {"facade_id": facade, "class": class_id,
            "polygon_uv": [[u0, v0], [u1, v0], [u1, v1], [u0, v1]], **extra}
    return client.post(f"{BASE}/access-points", json=body)


# ── taxonomy ─────────────────────────────────────────────────────────────
def test_the_site_serves_the_packs_it_names(client):
    body = client.get(f"{BASE}/taxonomy").json()
    assert body["version"] == "core@1.1"
    assert body["packs"] == [{"pack": "core", "version": "1.1.0"}]
    window = next(c for c in body["classes"] if c["id"] == "opening.window")
    assert window["labels"]["uk"] == "Вікно" and window["kind"] == "opening"
    # Attributes and approach numbers arrive resolved, so a picker needs no inheritance logic.
    assert "state" in [a["id"] for a in window["attributes"]]
    assert window["handover"]["default"] == "hover_handoff"


# ── semantics and facades ────────────────────────────────────────────────
def test_semantics_comes_from_the_newest_run_that_produced_it(client):
    body = client.get(f"{BASE}/semantics").json()
    assert body["run_id"] == "r1" and body["epoch_id"] == "baseline"
    assert [f["id"] for f in body["facades"]] == ["B01/E1", "B01/N1"]
    assert body["warnings"] == ["зйомка задалеко"]
    assert body["scene"] == "demo/pier"


def test_a_site_without_semantics_says_what_to_run(client, tmp_path):
    client.post("/api/v2/projects/demo/sites",
                json={"id": "empty", "origin": ORIGIN.model_dump()})
    r = client.get("/api/v2/sites/demo/empty/semantics")
    assert r.status_code == 404 and "semantics" in r.json()["detail"]


def test_a_facade_arrives_with_its_coverage_map(client):
    body = client.get(f"{BASE}/facades/B01/N1").json()
    assert body["facade"]["id"] == "B01/N1" and body["building"]["id"] == "B01"
    grid = body["coverage_grid"]
    assert (grid["nu"], grid["nv"]) == (40, 24) and grid["cell_m"] == 0.5
    assert grid["u_min"] == -10.0 and grid["v_min"] == 0.0
    # Row 0 is the bottom of the wall, as the stage writes it: the editor shades the
    # zeros, which is where 3DGS invented the texture (§5.3).
    assert grid["views"][0][0] == 7 and grid["views"][23][39] == 0
    assert grid["max_views"] == 7


def test_a_facade_without_a_coverage_grid_still_loads(client):
    body = client.get(f"{BASE}/facades/B01/E1").json()
    assert body["coverage_grid"] is None
    assert body["facade"]["coverage"]["covered"] == 0     # seen, never sharply enough


def test_an_unknown_facade_is_a_404(client):
    r = client.get(f"{BASE}/facades/B01/S9")
    assert r.status_code == 404 and "no facade" in r.json()["detail"]


# ── the orthophoto ───────────────────────────────────────────────────────
def test_the_ortho_pose_looks_at_the_wall_head_on(site):
    client, seen = site
    r = client.post(f"{BASE}/facades/B01/N1/ortho", json={"gsd_cm": 5, "margin_m": 1.0})
    assert r.status_code == 200 and r.content == b"\x89PNG-stub"
    assert seen["path"] == "/scenes/demo/pier/render"
    sent = seen["body"]
    assert sent["mode"] == "ortho" and sent["pitch_deg"] == 0.0 and sent["roll_deg"] == 0.0
    assert sent["yaw_deg"] == 180.0                      # the wall faces north
    assert (sent["width"], sent["height"]) == (440, 280)  # (20 + 2) m and (12 + 2) m at 5 cm
    assert sent["extent_m"] == pytest.approx(22.0)
    assert sent["position"] == {"e": 0.0, "n": 25.0, "u": 6.0}


def test_the_response_carries_the_grid_a_pixel_is_read_through(site):
    client, _ = site
    r = client.post(f"{BASE}/facades/B01/N1/ortho", json={"gsd_cm": 5})
    meta = json.loads(r.headers["x-render-meta"])
    assert meta["stub"] is True                          # the service's own meta is kept
    grid = meta["facade_grid"]
    assert grid["facade_id"] == "B01/N1" and grid["m_per_px"] == pytest.approx(0.05)
    assert grid["du_uv"] == [0.05, 0.0] and grid["dv_uv"] == [0.0, -0.05]
    assert meta["facade"]["length_m"] == 20.0 and meta["gsd_cm"] == 5.0
    assert meta["run"] == "r1"
    # The centre pixel of the image is the centre of the wall.
    u = grid["pixel00_uv"][0] + 219.5 * grid["du_uv"][0]
    v = grid["pixel00_uv"][1] + 139.5 * grid["dv_uv"][1]
    assert (u, v) == pytest.approx((0.0, 6.0))


def test_a_gsd_too_fine_for_one_image_is_refused_before_the_gpu(site):
    client, seen = site
    r = client.post(f"{BASE}/facades/B01/N1/ortho", json={"gsd_cm": 0.2})
    assert r.status_code == 422 and "cm/px or coarser" in r.json()["detail"]
    assert "path" not in seen                            # nothing reached the render service


def test_the_ortho_needs_a_trained_scene(client, monkeypatch):
    def no_scene(cat, site, run):
        raise SceneLookupError(f"run {run.id} of {site.key} has no trained scene (no splat product)")

    monkeypatch.setattr(api_semantics, "run_scene_source", no_scene)
    r = client.post(f"{BASE}/facades/B01/N1/ortho", json={})
    assert r.status_code == 404 and "no trained scene" in r.json()["detail"]


def test_the_render_service_being_down_says_how_to_start_it(client, monkeypatch):
    monkeypatch.setenv("AEROSPLAT_RENDER_URL", "http://127.0.0.1:9")
    r = client.post(f"{BASE}/facades/B01/N1/ortho", json={})
    assert r.status_code == 503 and "aerosplat up" in r.json()["detail"]


def add_run(run_id, facades=None, *, splat=True):
    """Another, newer run of the same Site; same facades as r1 unless told otherwise."""
    cat = api_v2.get_catalog()
    s = cat.get_site("demo", "pier")
    cat.create_run(s, "baseline", run_id, capture_ids=[], config={}, earth_model="wgs84",
                   georeferenced=True)
    dirs = cat.run_dirs(s, "baseline", run_id)
    old = cat.run_dirs(s, "baseline", "r1").root / "semantics" / SEMANTICS_JSON
    result = SemanticsResult(**json.loads(old.read_text(encoding="utf-8")))
    if facades is not None:
        result = result.model_copy(update={"facades": facades})
    sem = dirs.root / "semantics"
    sem.mkdir(parents=True)
    (sem / SEMANTICS_JSON).write_text(result.model_dump_json(), encoding="utf-8")
    products = {"semantics": sem}
    if splat:
        dirs.gs.mkdir(parents=True)
        (dirs.gs / "splat_wgs84.pt").write_bytes(b"stub")
        products["splat"] = dirs.gs / "splat_wgs84.pt"
    cat.finish_run(s, "baseline", run_id, status=RunStatus.ok, products=products)


def test_the_ortho_of_a_pinned_run_comes_from_that_runs_scene(site):
    """Session 2026-10-18…19, error 20: `?run=` pinned the facade but not the splat."""
    client, seen = site
    add_run("r2")
    assert client.post(f"{BASE}/facades/B01/N1/ortho", json={}).status_code == 200
    assert seen["body"]["run"] == "r2"                   # default: the newest run, by name
    r = client.post(f"{BASE}/facades/B01/N1/ortho", json={"run_id": "r1"})
    assert r.status_code == 200 and seen["body"]["run"] == "r1"
    assert json.loads(r.headers["x-render-meta"])["run"] == "r1"


def test_a_run_without_its_own_splat_is_refused_rather_than_mixed(site):
    client, seen = site
    add_run("r2", splat=False)
    r = client.post(f"{BASE}/facades/B01/N1/ortho", json={})
    assert r.status_code == 404 and "r2" in r.json()["detail"]
    assert "path" not in seen                            # the newest scene was not used instead
    body = client.get(f"{BASE}/semantics").json()
    assert body["run_id"] == "r2" and body["scene"] is None
    assert [(x["id"], x["scene"]) for x in body["runs"]] == [("r2", False), ("r1", True)]
    assert client.get(f"{BASE}/semantics?run_id=r1").json()["scene"] == "demo/pier"
    assert client.post(f"{BASE}/facades/B01/N1/ortho", json={"run_id": "r1"}).status_code == 200


def test_a_render_service_that_answers_with_another_run_is_caught(site):
    client, seen = site
    seen["wrong_run"] = "r0"                             # e.g. a service older than the API
    r = client.post(f"{BASE}/facades/B01/N1/ortho", json={})
    assert r.status_code == 502 and "r0" in r.json()["detail"]


def test_the_ortho_route_wins_over_the_facade_detail_route(site):
    """Both paths start with a greedy `{facade_id:path}`; order decides, so it is pinned."""
    client, seen = site
    assert client.post(f"{BASE}/facades/B01/N1/ortho", json={}).status_code == 200
    assert seen["body"]["yaw_deg"] == 180.0
    assert client.get(f"{BASE}/facades/B01/N1").json()["facade"]["id"] == "B01/N1"


# ── authored access points ───────────────────────────────────────────────
def test_a_drawn_opening_is_stored_in_the_shape_the_doc_specifies(client):
    r = draw(client, -7.65, 3.55, -6.35, 5.05, attributes={"state": "closed"}, by="lukhashka")
    assert r.status_code == 201, r.text
    ap = r.json()
    assert ap["class"] == "opening.window" and ap["kind"] == "opening"
    assert ap["taxonomy"] == "core@1.1" and ap["site_key"] == "demo/pier"
    assert ap["lifecycle"] == "verified"                 # drawn by hand (§3.4)
    assert ap["geometry"]["width"] == pytest.approx(1.3)
    assert ap["geometry"]["height"] == pytest.approx(1.5)
    assert ap["geometry"]["center"] == pytest.approx([7.0, 0.0, 4.3])
    assert ap["host"]["facade_id"] == "B01/N1" and ap["host"]["v_ground"] == 0.0
    assert ap["provenance"]["origin"] == "manual"
    assert ap["provenance"]["created_by"] == "op:lukhashka"
    assert ap["provenance"]["evidence"] == ["ortho:B01/N1"]
    assert ap["provenance"]["run_id"] == "r1"            # which ortho it was drawn on


def test_features_live_outside_the_run_and_survive_it(client, tmp_path):
    import shutil

    ap = draw(client, 0.0, 1.0, 1.4, 2.6).json()
    site_dir = tmp_path / "projects" / "demo" / "sites" / "pier"
    assert (site_dir / "authored" / "access_points.json").is_file()
    # A run directory is derived data and may be deleted (ADR-0006); the drawing is not.
    shutil.rmtree(site_dir / "epochs" / "baseline" / "runs" / "r1" / "semantics")
    kept = client.get(f"{BASE}/access-points").json()
    assert [f["id"] for f in kept] == [ap["id"]]
    assert kept[0]["host"]["facade_id"] == "B01/N1"      # keyed to the facade, not the run


def test_an_unknown_class_or_attribute_is_refused(client):
    assert draw(client, 0, 1, 1, 2, class_id="opening.teleporter").status_code == 422
    r = draw(client, 0, 1, 1, 2, attributes={"state": "ajar"})
    assert r.status_code == 422 and "is not one of" in r.json()["detail"]
    r = draw(client, 0, 1, 1, 2, attributes={"slope_deg": "3"})
    assert r.status_code == 422 and "has no attribute" in r.json()["detail"]


def test_a_sliver_and_an_unknown_facade_are_refused(client):
    assert draw(client, 0.0, 1.0, 0.01, 2.0).status_code == 422
    assert draw(client, 0, 1, 1, 2, facade="B01/S9").status_code == 404


def test_a_change_keeps_the_history_and_is_audited(client):
    ap = draw(client, -7.65, 3.55, -6.35, 5.05, by="lukhashka").json()
    r = client.patch(f"{BASE}/access-points/{ap['id']}",
                     json={"lifecycle": "rejected", "notes": "це не вікно", "by": "olena"})
    assert r.status_code == 200
    changed = r.json()
    assert changed["lifecycle"] == "rejected" and changed["notes"] == "це не вікно"
    assert changed["provenance"]["created_by"] == "op:lukhashka"     # not rewritten
    assert [e["by"] for e in changed["provenance"]["edits"]] == ["op:olena"]

    log = client.get(f"{BASE}/audit").json()
    assert [e["op"] for e in log] == ["create", "update"]
    assert log[1]["before"]["lifecycle"] == "verified"
    assert log[1]["after"]["lifecycle"] == "rejected"
    assert log[1]["what"] == "lifecycle, notes"


def test_reshaping_uses_the_frame_the_feature_was_drawn_in(client):
    ap = draw(client, -7.65, 3.55, -6.35, 5.05).json()
    r = client.patch(f"{BASE}/access-points/{ap['id']}",
                     json={"polygon_uv": [[-8.0, 3.0], [-6.0, 3.0], [-6.0, 5.0], [-8.0, 5.0]]})
    g = r.json()["geometry"]
    assert (g["width"], g["height"]) == pytest.approx((2.0, 2.0))
    assert g["u_center"] == pytest.approx(-7.0) and g["v_center"] == pytest.approx(4.0)
    assert g["center"] == pytest.approx([7.0, 0.0, 4.0])


def test_reclassifying_moves_the_kind_with_the_class(client):
    ap = draw(client, 0.0, 0.0, 1.2, 2.2).json()
    r = client.patch(f"{BASE}/access-points/{ap['id']}", json={"class": "opening.door"})
    assert r.json()["class"] == "opening.door" and r.json()["kind"] == "opening"
    assert client.patch(f"{BASE}/access-points/{ap['id']}",
                        json={"class": "opening.nope"}).status_code == 422


def test_listing_filters_by_facade_and_lifecycle(client):
    a = draw(client, -5.0, 1.0, -3.6, 2.6).json()
    b = draw(client, 0.0, 1.0, 1.4, 2.6).json()
    client.patch(f"{BASE}/access-points/{b['id']}", json={"lifecycle": "rejected"})
    assert len(client.get(f"{BASE}/access-points").json()) == 2
    assert len(client.get(f"{BASE}/access-points?facade=B01/E1").json()) == 0
    kept = client.get(f"{BASE}/access-points?lifecycle=verified").json()
    assert [f["id"] for f in kept] == [a["id"]]
    # Rejected features stay: they are hard negatives for training (§3.4).
    assert len(client.get(f"{BASE}/access-points?lifecycle=rejected").json()) == 1


def test_deleting_is_audited_and_only_works_once(client):
    ap = draw(client, 0.0, 1.0, 1.4, 2.6).json()
    assert client.delete(f"{BASE}/access-points/{ap['id']}?by=olena").status_code == 200
    assert client.delete(f"{BASE}/access-points/{ap['id']}").status_code == 404
    assert client.patch(f"{BASE}/access-points/{ap['id']}", json={"notes": "x"}).status_code == 404
    log = client.get(f"{BASE}/audit").json()
    assert log[-1]["op"] == "delete" and log[-1]["by"] == "op:olena"
    assert log[-1]["after"] is None and log[-1]["before"]["id"] == ap["id"]


def test_the_facade_count_ignores_rejected_features(client):
    a = draw(client, -5.0, 1.0, -3.6, 2.6).json()
    draw(client, 0.0, 1.0, 1.4, 2.6)
    assert client.get(f"{BASE}/facades/B01/N1").json()["features"] == 2
    client.patch(f"{BASE}/access-points/{a['id']}", json={"lifecycle": "rejected"})
    assert client.get(f"{BASE}/facades/B01/N1").json()["features"] == 1
    assert client.get(f"{BASE}/semantics").json()["features_by_facade"] == {"B01/N1": 1}


# ── addressing over the API ──────────────────────────────────────────────
def test_addressing_writes_floors_and_columns_onto_the_features(client):
    for v in (1.2, 4.3, 7.4):
        for u in (-7.0, -2.4, 2.2, 6.8):
            assert draw(client, u - 0.65, v - 0.75, u + 0.65, v + 0.75).status_code == 201
    body = client.post(f"{BASE}/facades/B01/N1/address", json={"by": "lukhashka"}).json()
    assert [r["floor_index"] for r in body["rows"]] == [0, 1, 2]
    assert [r["floor_label"] for r in body["rows"]] == ["1", "2", "3"]
    assert len(body["column_centres"]) == 4
    assert body["floor_height_m"] == pytest.approx(3.1, abs=0.05)
    assert not body["ambiguous"] and body["reasons"] == []

    stored = client.get(f"{BASE}/access-points").json()
    assert all(f["address"]["canonical"] for f in stored)
    ground_left = min(stored, key=lambda f: (f["geometry"]["v_center"], f["geometry"]["u_center"]))
    assert ground_left["address"]["canonical"] == "pier/B01/N1/00/01"
    assert ground_left["address"]["ordinal_on_floor"] == 1
    assert client.get(f"{BASE}/audit?limit=100").json()[-1]["op"] == "address"


def test_a_facade_whose_ground_row_is_missing_comes_back_ambiguous(client):
    # The most dangerous error in the system is the wrong floor, so thin evidence is
    # reported rather than smoothed over (§5.6); §6.3 then blocks confirmation.
    for v in (4.3, 7.4):
        draw(client, -7.65, v - 0.75, -6.35, v + 0.75)
    body = client.post(f"{BASE}/facades/B01/N1/address", json={}).json()
    assert body["ambiguous"] and body["reasons"]
    assert all(f["address"]["floor_ambiguous"] for f in client.get(f"{BASE}/access-points").json())


def test_addressing_ignores_rejected_openings(client):
    a = draw(client, -7.65, 0.45, -6.35, 1.95).json()
    draw(client, -2.4, 0.45, -1.1, 1.95)
    client.patch(f"{BASE}/access-points/{a['id']}", json={"lifecycle": "rejected"})
    body = client.post(f"{BASE}/facades/B01/N1/address", json={}).json()
    assert len(body["column_centres"]) == 1
    assert next(f for f in body["features"] if f["id"] == a["id"])["address"] is None


def _newer_run_without(tmp_path, facade_id: str = "B01/N1") -> None:
    """A newer run whose semantics name the walls differently: `facade_id` is gone."""
    cat = Catalog.open(tmp_path / "projects")
    s = cat.get_site("demo", "pier")
    cat.create_run(s, "baseline", "r2", capture_ids=[], config={}, earth_model="wgs84",
                   georeferenced=True)
    sem = cat.run_dirs(s, "baseline", "r2").root / "semantics"
    sem.mkdir(parents=True)
    other = _facade("B07/S1", (0.0, -1.0), (0.0, -12.0, 0.0))
    assert other.id != facade_id
    result = SemanticsResult(
        terrain=TerrainModel(res_m=1.0, x0=-40.0, y0=-40.0, rows=80, cols=80,
                             n_points=900_000, filled_cells=5400),
        buildings=[], facades=[other], warnings=[])
    (sem / SEMANTICS_JSON).write_text(result.model_dump_json(), encoding="utf-8")
    cat.finish_run(s, "baseline", "r2", status=RunStatus.ok, products={"semantics": sem})
    cat.close()


def test_an_older_run_is_edited_and_addressed_through_its_run_id(client, tmp_path):
    # Features drawn on an older run keep that run's facade frame. Once a newer run names
    # the walls differently, the editor reaches the old facade only by asking for its run.
    draw(client, -7.65, 0.45, -6.35, 1.95)
    _newer_run_without(tmp_path)
    assert client.get(f"{BASE}/facades/B01/N1").status_code == 404
    assert client.post(f"{BASE}/facades/B01/N1/address", json={}).status_code == 404
    assert draw(client, -2.4, 0.45, -1.1, 1.95).status_code == 404

    detail = client.get(f"{BASE}/facades/B01/N1?run_id=r1")
    assert detail.status_code == 200 and detail.json()["run_id"] == "r1"
    assert draw(client, -2.4, 0.45, -1.1, 1.95, run_id="r1").status_code == 201
    body = client.post(f"{BASE}/facades/B01/N1/address", json={"run_id": "r1"}).json()
    assert len(body["column_centres"]) == 2 and [r["floor_index"] for r in body["rows"]] == [0]
    assert client.post(f"{BASE}/facades/B01/N1/address",
                       json={"run_id": "no-such-run"}).status_code == 404


def test_addressing_an_empty_facade_changes_nothing(client):
    body = client.post(f"{BASE}/facades/B01/N1/address", json={}).json()
    assert body["rows"] == [] and body["features"] == [] and not body["ambiguous"]
    assert client.get(f"{BASE}/audit").json() == []


# ── pad zones (§5.7) ─────────────────────────────────────────────────────
def _write_terrain(run_id="r1"):
    """terrain.npz for the fixture's raster (80 × 80 m from (-40, -40)): B01's 20 × 12 m flat
    roof at 12 m, the ground at 0 everywhere else."""
    cat = api_v2.get_catalog()
    s = cat.get_site("demo", "pier")
    dsm = np.zeros((80, 80))
    labels = np.zeros((80, 80), np.int32)
    dsm[28:40, 30:50] = 12.0                             # rows: north -12..0, cols: east -10..10
    labels[28:40, 30:50] = 1
    np.savez_compressed(cat.run_dirs(s, "baseline", run_id).root / "semantics" / "terrain.npz",
                        dsm=dsm, dtm=np.zeros((80, 80)), count=np.full((80, 80), 9),
                        roughness=np.zeros((80, 80)), labels=labels)


def test_pad_candidates_from_the_roof_and_the_yard_are_candidates_only(client):
    _write_terrain()
    dry = client.post(f"{BASE}/pads/candidates", json={"building_ids": ["B01"], "dry_run": True}).json()
    assert dry["model"].startswith("padzones@") and dry["dry_run"]
    roof = [z for z in dry["zones"] if z["kind"] == "roof"]
    assert len(roof) == 1 and roof[0]["z"] == pytest.approx(12.0)
    assert roof[0]["usable_area_m2"] == 16 * 8                     # 20 × 12 eroded by 2 m
    assert any(z["kind"] == "ground" for z in dry["zones"])
    assert client.get(f"{BASE}/audit").json() == []                 # nothing stored

    out = client.post(f"{BASE}/pads/candidates", json={"building_ids": ["B01"], "by": "tester"}).json()
    cands = out["candidates"]
    assert cands and {c["lifecycle"] for c in cands} == {"candidate"}
    assert {c["provenance"]["origin"] for c in cands} == {"pipeline"}
    by_class = {c["class"]: c for c in cands}
    assert by_class["pad.roof_flat"]["pad"]["building_id"] == "B01"
    assert by_class["pad.ground"]["pad"]["building_id"] is None and "дороги" in by_class["pad.ground"]["notes"]
    assert by_class["pad.roof_flat"]["attributes"] == {"surface": "unknown", "hazard": "unknown"}
    assert 0 < by_class["pad.roof_flat"]["confidence"] < 1
    assert [e["op"] for e in client.get(f"{BASE}/audit").json()] == ["create"] * len(cands)

    # A second press proposes nothing new; a rejected pad is not proposed again either.
    again = client.post(f"{BASE}/pads/candidates", json={"building_ids": ["B01"]}).json()
    assert again["candidates"] == [] and again["skipped"] == {"candidate": len(cands)}
    roof_id = by_class["pad.roof_flat"]["id"]
    client.patch(f"{BASE}/access-points/{roof_id}", json={"lifecycle": "rejected"})
    third = client.post(f"{BASE}/pads/candidates", json={"building_ids": ["B01"], "dry_run": True}).json()
    assert third["candidates"] == [] and third["skipped"].get("rejected") == 1


def test_pad_candidates_say_why_not(client):
    _write_terrain()
    r = client.post(f"{BASE}/pads/candidates",
                    json={"building_ids": ["B01"], "ground": False, "clearance_m": 7.0, "dry_run": True})
    body = r.json()
    assert body["zones"] == [] and body["rejected"][0]["code"] == "no_room"
    assert client.post(f"{BASE}/pads/candidates", json={"building_ids": ["B99"]}).status_code == 404
    add_run("r2")                                                   # newer, without terrain.npz
    r = client.post(f"{BASE}/pads/candidates", json={"building_ids": ["B01"]})
    assert r.status_code == 404 and "terrain.npz" in r.json()["detail"]


# ── snapping to the facade's frame (§5.6 item 3) ─────────────────────────
def test_regularize_snaps_within_the_tolerance_and_audits_every_move(client):
    ids = [draw(client, -9.0 + 3 * i, 3.0 + dv0, -7.6 + 3 * i, 4.5 + dv1).json()["id"]
           for i, (dv0, dv1) in enumerate([(0.0, 0.0), (0.06, -0.03), (-0.04, 0.02), (0.0, 0.0),
                                           (0.4, 0.0)])]
    before = len(client.get(f"{BASE}/audit").json())
    dry = client.post(f"{BASE}/facades/B01/N1/regularize", json={"dry_run": True}).json()
    assert dry["dry_run"] and dry["tol_m"] == 0.15
    assert {(m["id"], m["edge"]) for m in dry["moves"]} == {
        (ids[1], "v0"), (ids[2], "v0"), (ids[1], "v1"), (ids[2], "v1")}
    assert all(abs(m["delta"]) <= 0.15 for m in dry["moves"])
    assert [(k["id"], k["edge"]) for k in dry["kept"]] == [(ids[4], "v0")]   # 0.4 m off: left
    assert len(client.get(f"{BASE}/audit").json()) == before                 # nothing stored

    out = client.post(f"{BASE}/facades/B01/N1/regularize", json={"by": "tester"}).json()
    assert sorted(f["id"] for f in out["features"]) == sorted(ids[1:3])
    stored = {f["id"]: f for f in client.get(f"{BASE}/access-points?facade=B01/N1").json()}
    for i in (1, 2):
        vs = [p[1] for p in stored[ids[i]]["geometry"]["polygon_uv"]]
        assert (min(vs), max(vs)) == pytest.approx((3.0, 4.5))
        assert "прив'язка до сітки" in stored[ids[i]]["provenance"]["edits"][-1]["what"]
    assert [p[1] for p in stored[ids[4]]["geometry"]["polygon_uv"]][0] == pytest.approx(3.4)
    tail = client.get(f"{BASE}/audit").json()[before:]
    assert [e["op"] for e in tail] == ["regularize", "regularize"]
    assert all("низ" in e["what"] and "верх" in e["what"] for e in tail)
    assert all(e["before"]["geometry"] != e["after"]["geometry"] for e in tail)
    again = client.post(f"{BASE}/facades/B01/N1/regularize", json={}).json()
    assert again["moves"] == [] and again["features"] == []


def test_regularize_leaves_rejected_and_other_runs_openings_alone(client):
    a = draw(client, -9.0, 3.0, -7.6, 4.5).json()
    b = draw(client, -6.0, 3.1, -4.6, 4.5).json()
    client.patch(f"{BASE}/access-points/{b['id']}", json={"lifecycle": "rejected"})
    out = client.post(f"{BASE}/facades/B01/N1/regularize", json={}).json()
    assert out["moves"] == []                                     # a row of one is no line
    _rerun_with_renamed_walls()
    out = client.post(f"{BASE}/facades/B01/N1/regularize", json={}).json()   # r2's B01/N1
    assert out["run_id"] == "r2" and out["features"] == [] and out["skipped"] == {}
    still = client.get(f"{BASE}/access-points?facade=B01/N1&run_id=r1").json()
    assert [f["geometry"] for f in still if f["id"] == a["id"]] == [a["geometry"]]


# ── facades across runs (§5.9) ───────────────────────────────────────────
def _rerun_with_renamed_walls():
    """r2: r1's north wall refitted 0.3 m out and renamed `B07/N3`; `B01/N1` is now another
    building's wall 100 m away (ids are per run)."""
    same_wall = _facade("B07/N3", (0.0, 1.0), (0.5, 0.3, 0.0))
    stranger = _facade("B01/N1", (0.0, 1.0), (100.0, 50.0, 0.0))
    add_run("r2", facades=[same_wall, stranger])


def test_facade_matches_pair_walls_by_geometry_not_by_name(client):
    _rerun_with_renamed_walls()
    body = client.get(f"{BASE}/facade-matches?from_run=r1").json()
    assert body["from_run"] == "r1" and body["to_run"] == "r2"
    got = {m["a_id"]: m for m in body["matches"]}
    assert got["B01/N1"]["status"] == "match" and got["B01/N1"]["b_ids"] == ["B07/N3"]
    assert got["B01/N1"]["relation"] == "same" and got["B01/N1"]["confidence"] > 0.5
    assert got["B01/E1"]["status"] == "none" and got["B01/E1"]["reason"]
    assert body["counts"] == {"match": 1, "none": 1}
    only = client.get(f"{BASE}/facade-matches?from_run=r1&status=none").json()["matches"]
    assert [m["a_id"] for m in only] == ["B01/E1"]
    assert client.get(f"{BASE}/facade-matches?from_run=nope").status_code == 404


def test_a_same_named_facade_of_another_run_does_not_show_or_address_its_features(client):
    ap = draw(client, -7.65, 0.45, -6.35, 1.95).json()            # on r1's B01/N1
    _rerun_with_renamed_walls()
    listed = client.get(f"{BASE}/access-points?facade=B01/N1&run_id=r2").json()
    assert listed == []                                           # r2's B01/N1 is another wall
    mine = client.get(f"{BASE}/access-points?facade=B01/N1&run_id=r1").json()
    assert [f["id"] for f in mine] == [ap["id"]]
    assert client.get(f"{BASE}/semantics").json()["features_by_facade"] == {}
    assert client.get(f"{BASE}/semantics?run_id=r1").json()["features_by_facade"] == {"B01/N1": 1}
    assert client.get(f"{BASE}/facades/B01/N1").json()["features"] == 0
    body = client.post(f"{BASE}/facades/B01/N1/address", json={}).json()
    assert body["rows"] == [] and body["features"] == []


def test_features_of_another_run_are_shown_on_the_matching_wall_and_carried_on_request(client):
    ap = draw(client, -7.65, 0.45, -6.35, 1.95).json()
    _rerun_with_renamed_walls()
    carried = client.get(f"{BASE}/facades/B07/N3/carried").json()
    assert [c["feature"]["id"] for c in carried] == [ap["id"]]
    c = carried[0]
    assert c["placement"]["status"] == "placed" and c["placement"]["facade_id"] == "B07/N3"
    assert c["here"]["offset_m"] == pytest.approx(0.3, abs=1e-6)
    # u runs west on a north wall, so moving the origin 0.5 m east moves u by +0.5.
    us = sorted({p[0] for p in c["here"]["polygon_uv"]})
    assert us == pytest.approx([-7.15, -5.85])
    assert client.get(f"{BASE}/facades/B01/N1/carried").json() == []   # the stranger
    # Nothing moved by itself: the feature is still on r1's frame.
    still = client.get(f"{BASE}/access-points?facade=B01/N1&run_id=r1").json()
    assert [f["id"] for f in still] == [ap["id"]]

    r = client.post(f"{BASE}/access-points/{ap['id']}/reanchor",
                    json={"facade_id": "B07/N3", "by": "tester"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["from_facade"] == "B01/N1" and out["feature"]["host"]["facade_id"] == "B07/N3"
    assert out["feature"]["lifecycle"] == "verified"
    moved = client.get(f"{BASE}/access-points?facade=B07/N3&run_id=r2").json()
    assert [f["id"] for f in moved] == [ap["id"]]
    last = client.get(f"{BASE}/audit").json()[-1]
    assert last["op"] == "reanchor" and "B01/N1" in last["what"] and "0.30 м" in last["what"]
    assert last["before"]["host"]["facade_id"] == "B01/N1"
    again = client.post(f"{BASE}/access-points/{ap['id']}/reanchor", json={"facade_id": "B07/N3"})
    assert again.status_code == 409


def test_a_feature_is_not_carried_onto_a_wall_it_does_not_land_on(client):
    ap = draw(client, -7.65, 0.45, -6.35, 1.95).json()
    _rerun_with_renamed_walls()
    r = client.post(f"{BASE}/access-points/{ap['id']}/reanchor", json={"facade_id": "B01/N1"})
    assert r.status_code == 409 and "не можна перенести" in r.json()["detail"]
    assert client.get(f"{BASE}/audit").json()[-1]["op"] == "create"   # nothing written
