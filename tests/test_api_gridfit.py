"""API: POST …/facades/{id}/candidates/gridfit — window candidates from the facade grid.

The render service is stubbed with one that paints a synthetic facade (`gridfit_scenes`)
into whatever it is asked for — rgb, alpha or depth — on exactly the pixel grid the API
computed, so the whole path runs: pose, three renders, fit, screening against what is
already authored, storing, audit. ADR-0009: nothing it stores is ever `verified`.
"""

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("httpx")
from fastapi import FastAPI                                            # noqa: E402
from fastapi.testclient import TestClient                              # noqa: E402

from aerosplat.geo.crs import GeoPoint                                 # noqa: E402
from aerosplat.semantics.gridfit import MODEL                          # noqa: E402
from aerosplat.semantics.models import (COVERAGE_NPZ, SEMANTICS_JSON, Building,  # noqa: E402
                                        Coverage, Facade, SemanticsResult, SurfaceKind,
                                        TerrainModel)
from aerosplat.server import api_semantics, api_v2                     # noqa: E402
from aerosplat.store import Catalog, Site                              # noqa: E402
from aerosplat.store.models import RunStatus                           # noqa: E402
from aerosplat.store.scenes import SceneLookupError                    # noqa: E402
from gridfit_scenes import NAVY, Painter, facade as paint_facade       # noqa: E402

ORIGIN = GeoPoint(lat=50.45, lon=30.52, h=180.0)
BASE = "/api/v2/sites/demo/tower"
URL = f"{BASE}/facades/B01/N1/candidates/gridfit"
# 30 × 22 m with a 1 m margin is 400 × 300 px at 8 cm: whole pixels both ways, so the
# stub's painter and the API's facade grid are the same grid.
WIDTH_M, HEIGHT_M = 30.0, 22.0
DEPTH_MAX_M = 40.0


def _painted(kind: str):
    """The facade the stub paints, by scenario name."""
    if kind == "plain":
        p = Painter(WIDTH_M, HEIGHT_M, 0.08)
        return p, []
    p, windows = paint_facade(width_m=WIDTH_M, height_m=HEIGHT_M)
    if kind == "van":
        p.ellipse(15.0, 2.5, 4.0, 2.8, NAVY, front_m=6.0)
    return p, windows


@pytest.fixture
def stub_render():
    """A render service that paints the scenario into rgb, alpha or depth as asked."""
    state = {"scenario": "grid", "outputs": [], "broken": False}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["outputs"].append(body["output"])
            state.setdefault("runs", []).append(body.get("run"))
            p, _ = _painted(state["scenario"])
            rgb, alpha = p.finish(seed=1)
            w, h = body["width"], body["height"]
            assert rgb.shape[:2] == (h, w), (rgb.shape, w, h)
            meta = {"width": w, "height": h, "run": body.get("run")}
            if body["output"] == "rgb":
                img = Image.fromarray(rgb)
            elif body["output"] == "alpha":
                img = Image.fromarray((alpha * 255).astype(np.uint8))
            else:
                # The camera plane stands 25 m in front of the wall (`standoff_m`).
                depth = np.where(alpha > 0.5, 25.0 - p.front, 0.0)
                img = Image.fromarray((depth / DEPTH_MAX_M * 65535).round().astype(np.uint16))
                meta["depth"] = {"max_m": DEPTH_MAX_M, "no_data": 0}
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            data = b"not a png" if state["broken"] else buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("X-Render-Meta", json.dumps(meta))
            self.end_headers()
            self.wfile.write(data)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", state
    srv.shutdown()


@pytest.fixture
def site(tmp_path, monkeypatch, stub_render):
    monkeypatch.setenv("AEROSPLAT_PROJECTS", str(tmp_path / "projects"))
    monkeypatch.setenv("AEROSPLAT_RENDER_URL", stub_render[0])
    monkeypatch.setenv("AEROSPLAT_OPERATOR", "tester")

    cat = Catalog.open(tmp_path / "projects")
    cat.ensure_project("demo")
    s = cat.create_site(Site(id="tower", project_id="demo", origin=ORIGIN))
    cat.ensure_epoch(s)
    cat.create_run(s, "baseline", "r1", capture_ids=[], config={}, earth_model="wgs84",
                   georeferenced=True)
    dirs = cat.run_dirs(s, "baseline", "r1")
    sem = dirs.root / "semantics"
    sem.mkdir(parents=True)
    # A north wall: u runs west, so the origin sits at its east end.
    wall = Facade(id="B01/N1", building_id="B01", kind=SurfaceKind.wall, origin=(0.0, 0.0, 0.0),
                  normal=(0.0, 1.0, 0.0), u_axis=(-1.0, 0.0, 0.0), azimuth_deg=0.0,
                  u_min=0.0, u_max=WIDTH_M, v_ground=0.0, v_max=HEIGHT_M, plane_rms_m=0.2,
                  n_points=9000,
                  coverage=Coverage(cell_m=0.5, cells=2640, visible=2640, covered=2640,
                                    mean_views=6.0, max_gsd_cm=12.0, best_gsd_cm=9.0))
    building = Building(id="B01", footprint=[(-30, -12), (0, -12), (0, 0), (-30, 0)],
                        area_m2=360.0, ground_z=0.0, roof_z=HEIGHT_M, height_m=HEIGHT_M,
                        facade_ids=["B01/N1"])
    result = SemanticsResult(
        terrain=TerrainModel(res_m=1.0, x0=-50.0, y0=-50.0, rows=100, cols=100,
                             n_points=10_000, filled_cells=9000),
        buildings=[building], facades=[wall])
    (sem / SEMANTICS_JSON).write_text(result.model_dump_json(), encoding="utf-8")
    np.savez_compressed(sem / COVERAGE_NPZ, **{"B01/N1": np.full((45, 60), 6, np.int32)})
    dirs.gs.mkdir(parents=True)
    (dirs.gs / "splat_wgs84.pt").write_bytes(b"stub")
    cat.finish_run(s, "baseline", "r1", status=RunStatus.ok,
                   products={"semantics": sem, "splat": dirs.gs / "splat_wgs84.pt"})
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


def n_windows(kind="grid"):
    return len(_painted(kind)[1])


def features(client):
    return client.get(f"{BASE}/access-points").json()


# ── what it stores ───────────────────────────────────────────────────────
def test_the_grid_becomes_candidates_never_verified(client):
    r = client.post(URL, json={})
    assert r.status_code == 200, r.text
    out = r.json()
    assert not out["refused"] and out["model"] == MODEL
    assert out["result"]["floor"]["floor_m"] == pytest.approx(3.0, abs=0.02)
    assert out["result"]["bay_m"] == pytest.approx(2.5, abs=0.02)
    assert len(out["candidates"]) == out["proposed"] == n_windows()
    stored = features(client)
    assert len(stored) == n_windows()
    for f in stored:
        assert f["lifecycle"] == "candidate"
        assert f["class"] == "opening.window"
        assert f["provenance"]["origin"] == "pipeline"
        assert f["provenance"]["model"] == MODEL
        assert f["provenance"]["created_by"] == "pipeline:gridfit"
        assert f["provenance"]["evidence"] == ["ortho:B01/N1@8cm"]
        assert 0 < f["confidence"] < 1
        assert "поверх 3.0" in f["notes"]
    # Who pressed the button is in the audit, the proposer in the provenance.
    audit = client.get(f"{BASE}/audit").json()
    assert len(audit) == n_windows()
    assert {a["by"] for a in audit} == {"op:tester"} and all("gridfit@" in a["what"] for a in audit)


def test_it_renders_rgb_alpha_and_depth(site):
    client, stub = site
    client.post(URL, json={"dry_run": True})
    assert stub["outputs"] == ["rgb", "alpha", "depth"]
    stub["outputs"].clear()
    client.post(URL, json={"dry_run": True, "depth": False})
    assert stub["outputs"] == ["rgb", "alpha"]


def test_a_dry_run_stores_nothing(client):
    out = client.post(URL, json={"dry_run": True}).json()
    assert out["dry_run"] and len(out["candidates"]) == n_windows()
    assert features(client) == []
    assert client.get(f"{BASE}/audit").json() == []


# ── what it must not store ───────────────────────────────────────────────
def test_pressing_twice_adds_nothing(client):
    client.post(URL, json={})
    again = client.post(URL, json={}).json()
    assert again["candidates"] == []
    assert again["skipped"] == {"candidate": n_windows()}
    assert len(features(client)) == n_windows()


def test_a_rejected_candidate_is_not_proposed_again(client):
    first = client.post(URL, json={}).json()["candidates"][0]
    client.delete(f"{BASE}/access-points/{first['id']}")         # a mistake: gone
    second = client.post(URL, json={}).json()["candidates"]
    assert len(second) == 1                                       # ... so it comes back
    client.patch(f"{BASE}/access-points/{second[0]['id']}", json={"lifecycle": "rejected"})
    third = client.post(URL, json={}).json()
    assert third["candidates"] == []                              # a rejection sticks
    assert third["skipped"]["rejected"] == 1


def test_a_hand_drawn_window_is_not_duplicated(client):
    body = {"facade_id": "B01/N1", "class": "opening.window",
            "polygon_uv": [[0.6, 0.8], [2.0, 0.8], [2.0, 2.5], [0.6, 2.5]]}
    assert client.post(f"{BASE}/access-points", json=body).status_code == 201
    out = client.post(URL, json={}).json()
    assert out["skipped"] == {"verified": 1}
    assert len(out["candidates"]) == n_windows() - 1


def test_a_refused_facade_stores_nothing_and_says_why(site):
    client, stub = site
    stub["scenario"] = "plain"
    out = client.post(URL, json={}).json()
    assert out["refused"] and out["candidates"] == [] and out["proposed"] == 0
    assert out["result"]["refusals"][0]["code"] == "no_columns"
    assert out["result"]["refusals"][0]["message"]
    assert features(client) == []


def test_depth_keeps_candidates_off_an_occluder(site):
    client, stub = site
    stub["scenario"] = "van"
    blind = client.post(URL, json={"dry_run": True, "depth": False}).json()
    seen = client.post(URL, json={"dry_run": True}).json()
    reasons = [c["reason"] for c in seen["result"]["cells"]]
    assert "occluded" in reasons
    assert len(seen["candidates"]) < len(blind["candidates"])


def test_unknown_facade_is_a_404(client):
    r = client.post(f"{BASE}/facades/B01/Z9/candidates/gridfit", json={})
    assert r.status_code == 404


def test_no_trained_scene_is_a_404(client, monkeypatch):
    def no_scene(cat, site, run):
        raise SceneLookupError(f"run {run.id} of {site.key} has no trained scene (no splat product)")

    monkeypatch.setattr(api_semantics, "run_scene_source", no_scene)
    r = client.post(URL, json={})
    assert r.status_code == 404 and "scene" in r.json()["detail"]


def test_gridfit_renders_the_runs_own_scene(client, stub_render):
    client.post(URL, json={"dry_run": True})
    assert stub_render[1]["runs"] and set(stub_render[1]["runs"]) == {"r1"}


def test_the_render_service_being_down_is_a_503(client, monkeypatch):
    monkeypatch.setenv("AEROSPLAT_RENDER_URL", "http://127.0.0.1:9")
    r = client.post(URL, json={})
    assert r.status_code == 503
    assert features(client) == []


def test_an_unreadable_render_is_a_502_and_stores_nothing(site):
    client, stub = site
    stub["broken"] = True
    r = client.post(URL, json={})
    assert r.status_code == 502
    assert features(client) == []


def test_unknown_request_fields_are_refused(client):
    assert client.post(URL, json={"lifecycle": "verified"}).status_code == 422
    assert client.post(URL, json={"gsd_cm": 0.5}).status_code == 422
