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
            self.send_header("X-Render-Meta", '{"stub":true,"width":1,"height":1}')
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
    assert body["version"] == "core@1.0"
    assert body["packs"] == [{"pack": "core", "version": "1.0.0"}]
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
    monkeypatch.setattr(api_v2, "_scene_name", lambda cat, s: None)
    r = client.post(f"{BASE}/facades/B01/N1/ortho", json={})
    assert r.status_code == 404 and "no trained scene" in r.json()["detail"]


def test_the_render_service_being_down_says_how_to_start_it(client, monkeypatch):
    monkeypatch.setenv("AEROSPLAT_RENDER_URL", "http://127.0.0.1:9")
    r = client.post(f"{BASE}/facades/B01/N1/ortho", json={})
    assert r.status_code == 503 and "aerosplat up" in r.json()["detail"]


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
    assert ap["taxonomy"] == "core@1.0" and ap["site_key"] == "demo/pier"
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


def test_addressing_an_empty_facade_changes_nothing(client):
    body = client.post(f"{BASE}/facades/B01/N1/address", json={}).json()
    assert body["rows"] == [] and body["features"] == [] and not body["ambiguous"]
    assert client.get(f"{BASE}/audit").json() == []
