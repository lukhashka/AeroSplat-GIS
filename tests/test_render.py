"""Render service: camera maths, request schema, HTTP layer, API v2 proxy; GPU render vs reference."""

import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
from pydantic import ValidationError

from aerosplat.render import camera as cam
from aerosplat.render.models import Position, RenderRequest


# ── camera maths (CPU) ───────────────────────────────────────────────────
def cam_xyz(V, point):
    return V[:3, :3] @ np.asarray(point, float) + V[:3, 3]


def test_rotation_is_proper_for_any_pose():
    for yaw in (0, 37, 90, 181, -120):
        for pitch in (-90, -45, 0, 30, 90):
            for roll in (0, 25, -60):
                R = cam.view_rotation(yaw, pitch, roll)
                assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
                assert np.linalg.det(R) == pytest.approx(1.0)


def test_facing_north_east_is_right_and_up_is_up():
    V = cam.view_matrix([0, 0, 0], yaw_deg=0, pitch_deg=0)
    east = cam_xyz(V, [10, 100, 0])
    assert east[0] > 0 and east[2] > 0 and east[1] == pytest.approx(0)      # right, in front
    assert cam_xyz(V, [0, 100, 10])[1] < 0                                  # above = image y < 0
    west = cam_xyz(V, [-10, 100, 0])
    assert west[0] < 0


def test_yaw_is_compass_heading():
    V = cam.view_matrix([0, 0, 0], yaw_deg=90, pitch_deg=0)          # facing east
    assert cam_xyz(V, [100, 0, 0]) == pytest.approx([0, 0, 100])
    assert cam_xyz(V, [100, -10, 0])[0] > 0                           # south is to the right


def test_nadir_with_yaw_zero_is_north_up():
    V = cam.view_matrix([0, 0, 100], yaw_deg=0, pitch_deg=-90)
    north, east = cam_xyz(V, [0, 10, 0]), cam_xyz(V, [10, 0, 0])
    assert north[1] < 0 and north[0] == pytest.approx(0)               # north: up in the image
    assert east[0] > 0 and east[1] == pytest.approx(0)                 # east: right
    assert north[2] == pytest.approx(100)


def test_roll_positive_is_clockwise():
    V = cam.view_matrix([0, 0, 0], 0, 0, roll_deg=90)
    # Camera rolled clockwise: the world turns counter-clockwise, so what was at 3 o'clock
    # (east of the axis) moves to 12 o'clock.
    assert cam_xyz(V, [10, 100, 0])[1] < 0
    assert cam_xyz(V, [10, 100, 0])[0] == pytest.approx(0, abs=1e-9)


def test_yaw_pitch_to_round_trip():
    eye = np.array([5.0, -3.0, 20.0])
    yaw, pitch = cam.yaw_pitch_to(eye, eye + [30, 40, -50])
    V = cam.view_matrix(eye, yaw, pitch)
    p = cam_xyz(V, eye + [30, 40, -50])
    assert p[0] == pytest.approx(0, abs=1e-9) and p[1] == pytest.approx(0, abs=1e-9) and p[2] > 0
    with pytest.raises(ValueError):
        cam.yaw_pitch_to(eye, eye)


def test_persp_intrinsics_vertical_fov():
    fx, fy, cx, cy = cam.persp_intrinsics(1000, 500, 90.0)
    assert fx == fy == pytest.approx(250.0) and (cx, cy) == (500, 250)


def test_ortho_grid_only_for_nadir():
    K = cam.ortho_intrinsics(1000, 1000, 500.0)                         # 2 px per metre
    V = cam.view_matrix([100, 200, 50], 0, -90)
    g = cam.ortho_ground_grid(V, K, 1000, 1000)
    assert g["m_per_px"] == pytest.approx(0.5)
    assert g["du_enu"] == pytest.approx([0.5, 0.0]) and g["dv_enu"] == pytest.approx([0.0, -0.5])
    # the principal point (u = v = 500, i.e. the centre of "pixel" 499.5) is the camera's ground point
    c = np.array(g["pixel00_enu"]) + 499.5 * np.array(g["du_enu"]) + 499.5 * np.array(g["dv_enu"])
    assert c == pytest.approx([100, 200])
    assert cam.ortho_ground_grid(cam.view_matrix([0, 0, 50], 0, -60), K, 1000, 1000) is None
    assert cam.ortho_ground_grid(cam.view_matrix([0, 0, 50], 0, 90), K, 1000, 1000) is None


def test_ortho_grid_follows_yaw():
    K = cam.ortho_intrinsics(100, 100, 100.0)
    V = cam.view_matrix([0, 0, 10], yaw_deg=90, pitch_deg=-90)         # image-up = east
    g = cam.ortho_ground_grid(V, K, 100, 100)
    assert g["dv_enu"] == pytest.approx([-1.0, 0.0], abs=1e-9)         # +v (down) = west
    px, _ = cam.project(V, K, np.array([[40.0, 0.0, 0.0]]), ortho=True)
    assert px[0] == pytest.approx([50.0, 10.0])                        # 40 m east: up, centre column


# ── request schema ───────────────────────────────────────────────────────
def test_position_needs_one_complete_frame():
    assert Position(e=1, n=2, u=3).vertical == 3
    assert Position(lat=40, lon=-74, h=5).is_geodetic
    for bad in ({"e": 1}, {"lat": 1}, {"e": 1, "n": 2, "lat": 3, "lon": 4}, {"e": 1, "n": 2, "h": 3}):
        with pytest.raises(ValidationError):
            Position(**bad)


def test_request_defaults_and_rules():
    p = {"e": 0, "n": 0, "u": 10}
    assert RenderRequest(position=p).pitch == -45
    assert RenderRequest(mode="ortho", position={"e": 0, "n": 0}).pitch == -90
    for bad in ({"position": {"e": 0, "n": 0}},                              # persp without height
                {"mode": "ortho", "position": p, "look_at": p},
                {"position": p, "width": 100000},
                {"position": p, "fov_deg": 0.5},
                {"position": p, "background": [2, 0, 0]},
                {"position": p, "output": "normals"}):
        with pytest.raises(ValidationError):
            RenderRequest(**bad)


# ── HTTP layer with a fake engine (no GPU) ───────────────────────────────
class FakeEngine:
    def __init__(self):
        self.calls = []

    def render(self, scene, req):
        from aerosplat.render.engine import GpuOutOfMemory, Rendered, SceneNotFound
        self.calls.append((scene, req))
        if scene == "none/none":
            raise SceneNotFound("no trained scene")
        if scene == "oom/oom":
            raise GpuOutOfMemory("full")
        h, w = req.height, req.width
        image = {"rgb": np.full((h, w, 3), 200, np.uint8), "alpha": np.full((h, w), 255, np.uint8),
                 "depth": np.full((h, w), 777, np.uint16)}[req.output]
        return Rendered(image=image, meta={"scene": scene, "render_ms": 1.0})

    def status(self):
        return {"device": "fake", "vram_allocated_mb": 0, "idle_unload_s": 1, "loaded": []}

    def scenes(self, fresh=False):
        from types import SimpleNamespace
        return {"demo/pier": SimpleNamespace(splat="x"), "pier": SimpleNamespace(splat="x"),
                "empty": SimpleNamespace(splat="")}

    def unload_idle(self):
        return []

    def unload_all(self):
        pass


@pytest.fixture
def render_client():
    from fastapi.testclient import TestClient

    from aerosplat.render.app import create_app

    eng = FakeEngine()
    with TestClient(create_app(eng)) as c:
        c.engine = eng
        yield c


BODY = {"position": {"e": 0, "n": -50, "u": 30}, "yaw_deg": 10, "width": 64, "height": 32}


def test_service_returns_png_with_meta(render_client):
    from PIL import Image

    r = render_client.post("/scenes/demo/pier/render", json=BODY)
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(r.content)).size == (64, 32)
    assert json.loads(r.headers["x-render-meta"])["scene"] == "demo/pier"       # slash in the name
    assert render_client.engine.calls[0][1].yaw_deg == 10


def test_service_depth_is_16_bit_png(render_client):
    from PIL import Image

    r = render_client.post("/scenes/demo/pier/render", json={**BODY, "output": "depth"})
    arr = np.array(Image.open(io.BytesIO(r.content)))
    assert arr.dtype in (np.uint16, np.int32) and arr.max() == 777 and arr.shape == (32, 64)


def test_service_errors(render_client):
    assert render_client.post("/scenes/none/none/render", json=BODY).status_code == 404
    assert render_client.post("/scenes/oom/oom/render", json=BODY).status_code == 503
    assert render_client.post("/scenes/demo/pier/render", json={"position": {}}).status_code == 422
    assert render_client.get("/health").json()["status"] == "ok"
    assert render_client.get("/scenes").json() == ["demo/pier", "pier"]


# ── API v2 proxy ─────────────────────────────────────────────────────────
@pytest.fixture
def stub_service():
    """A tiny HTTP server standing in for the render service."""
    seen = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok","device":"stub","loaded":[]}')

        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            code = 422 if seen["body"].get("yaw_deg") == 999 else 200
            self.send_response(code)
            if code == 200:
                self.send_header("Content-Type", "image/png")
                self.send_header("X-Render-Meta", '{"stub":true}')
                self.end_headers()
                self.wfile.write(b"\x89PNG-stub")
            else:
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"detail":"stub says no"}')

    srv = HTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}", seen
    srv.shutdown()


@pytest.fixture
def v2(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from aerosplat.server import api_v2

    monkeypatch.setenv("AEROSPLAT_PROJECTS", str(tmp_path / "projects"))
    app = FastAPI()
    app.include_router(api_v2.router)
    with TestClient(app) as c:
        assert c.post("/api/v2/projects", json={"id": "demo", "name": "Demo"}).status_code == 201
        origin = {"lat": 40.89, "lon": -74.22, "h": 233.0}
        assert c.post("/api/v2/projects/demo/sites", json={"id": "pier", "name": "Pier",
                                                            "origin": origin}).status_code == 201
        yield c, api_v2
    for cat in api_v2._CATALOGS.values():
        cat.close()
    api_v2._CATALOGS.clear()


def _give_site_a_scene(v2_api, monkeypatch):
    monkeypatch.setattr(v2_api, "_scene_name", lambda cat, site: site.key)


def test_proxy_forwards_request_and_meta(v2, stub_service, monkeypatch):
    c, api = v2
    url, seen = stub_service
    monkeypatch.setenv("AEROSPLAT_RENDER_URL", url)
    _give_site_a_scene(api, monkeypatch)
    r = c.post("/api/v2/sites/demo/pier/render", json=BODY)
    assert r.status_code == 200 and r.content == b"\x89PNG-stub"
    assert r.headers["x-render-meta"] == '{"stub":true}'
    assert seen["path"] == "/scenes/demo/pier/render"
    assert seen["body"]["position"] == {"e": 0, "n": -50, "u": 30} and seen["body"]["yaw_deg"] == 10
    assert c.get("/api/v2/render/status").json()["online"] is True


def test_proxy_passes_service_errors_through(v2, stub_service, monkeypatch):
    c, api = v2
    monkeypatch.setenv("AEROSPLAT_RENDER_URL", stub_service[0])
    _give_site_a_scene(api, monkeypatch)
    r = c.post("/api/v2/sites/demo/pier/render", json={**BODY, "yaw_deg": 999})
    assert r.status_code == 422 and r.json()["detail"] == "stub says no"


def test_proxy_validates_before_forwarding(v2, stub_service, monkeypatch):
    c, api = v2
    monkeypatch.setenv("AEROSPLAT_RENDER_URL", stub_service[0])
    _give_site_a_scene(api, monkeypatch)
    assert c.post("/api/v2/sites/demo/pier/render", json={"position": {"e": 1}}).status_code == 422
    assert "path" not in stub_service[1]


def test_proxy_offline_site_without_scene_and_unknown_site(v2, monkeypatch):
    c, api = v2
    monkeypatch.setenv("AEROSPLAT_RENDER_URL", "http://127.0.0.1:9")            # nothing listens
    assert c.get("/api/v2/render/status").json() == {"online": False}
    assert c.post("/api/v2/sites/demo/pier/render", json=BODY).status_code == 404    # no scene yet
    _give_site_a_scene(api, monkeypatch)
    r = c.post("/api/v2/sites/demo/pier/render", json=BODY)
    assert r.status_code == 503 and "offline" in r.json()["detail"]
    assert c.post("/api/v2/sites/demo/nope/render", json=BODY).status_code == 404


# ── a scene pinned to one run (facade editor `?run=`, ACCESS_POINTS §5.9) ─
@pytest.fixture
def two_runs(tmp_path):
    """demo/pier with r1 (older) and r2 (newer), both with a splat; r0 has none."""
    from aerosplat.geo.crs import GeoPoint
    from aerosplat.store import Catalog, Site
    from aerosplat.store.models import RunStatus

    root = tmp_path / "projects"
    cat = Catalog.open(root)
    cat.ensure_project("demo")
    s = cat.create_site(Site(id="pier", project_id="demo",
                             origin=GeoPoint(lat=40.89, lon=-74.22, h=233.0)))
    cat.ensure_epoch(s)
    for run_id, has_splat in (("r0", False), ("r1", True), ("r2", True)):
        cat.create_run(s, "baseline", run_id, earth_model="wgs84", georeferenced=True)
        gs = cat.run_dirs(s, "baseline", run_id).gs
        gs.mkdir(parents=True)
        (gs / "cloud_wgs84.pt").write_bytes(b"cloud")
        products = {"cloud": gs / "cloud_wgs84.pt"}
        if has_splat:
            (gs / "splat_wgs84.pt").write_bytes(run_id.encode())
            products["splat"] = gs / "splat_wgs84.pt"
        cat.finish_run(s, "baseline", run_id, status=RunStatus.ok, products=products)
    cat.close()
    return root


def test_a_pinned_scene_is_that_run_or_an_error(two_runs):
    from pathlib import Path

    from aerosplat.store.scenes import SceneLookupError, discover_scenes, pinned_scene

    assert discover_scenes(two_runs, None)["demo/pier"].run == "r2"      # newest by default
    src = pinned_scene(two_runs, "demo/pier", "r1")
    assert src.run == "r1" and Path(src.splat).read_bytes() == b"r1"
    assert src.metric is not None and src.metric["run"] == "r1"
    for run, why in (("r0", "no trained scene"), ("r9", "no run 'r9'")):
        with pytest.raises(SceneLookupError, match=why):
            pinned_scene(two_runs, "demo/pier", run)
    with pytest.raises(SceneLookupError, match="<project>/<site>"):
        pinned_scene(two_runs, "pier", "r1")
    Path(src.splat).unlink()
    with pytest.raises(SceneLookupError, match="missing"):
        pinned_scene(two_runs, "demo/pier", "r1")


def test_the_engine_never_answers_a_pinned_run_with_another(two_runs):
    """Resolution only (no GPU): the engine is built without its gsplat constructor."""
    import threading as th

    from aerosplat.render.engine import RenderEngine, SceneNotFound

    eng = RenderEngine.__new__(RenderEngine)
    eng.projects_root, eng.data_dir = two_runs, None
    eng._lock, eng._scenes, eng._listing = th.RLock(), {}, (-1e9, {})
    assert eng._resolve("demo/pier").run == "r2"
    assert eng._resolve("demo/pier", "r2").run == "r2"
    assert eng._resolve("demo/pier", "r1").run == "r1"
    assert eng._resolve("pier", "r1").run == "r1"                        # the bare alias too
    for run in ("r0", "r9"):
        with pytest.raises(SceneNotFound):
            eng._resolve("demo/pier", run)
    assert RenderRequest(position={"e": 0, "n": 0, "u": 1}, run="r1").run == "r1"
    with pytest.raises(ValidationError):
        RenderRequest(position={"e": 0, "n": 0, "u": 1}, run="../r1")


# ── GPU: render a synthetic scene and compare with the reference projection ─
def _gsplat_ready():
    try:
        import torch

        from aerosplat.gs.gsplat_backend import require_gsplat
        if not torch.cuda.is_available():
            return False
        require_gsplat()
        return True
    except Exception:
        return False


needs_gsplat = pytest.mark.skipif(not _gsplat_ready(), reason="gsplat + CUDA (tools\\gsplat_env.bat)")


@pytest.fixture
def gpu_engine(tmp_path, monkeypatch):
    import torch

    from aerosplat.geo.crs import GeoPoint
    from aerosplat.render.engine import RenderEngine
    from aerosplat.store.scenes import SceneSource, metric_block

    pts = torch.tensor([[10.0, 50.0, 0.0], [-20.0, 80.0, 5.0], [0.0, 30.0, -5.0], [40.0, -10.0, 0.0]])
    n = len(pts)
    splat = {"xyz": pts, "scale": torch.full((n, 3), 1.0), "rotation": torch.tensor([[1.0, 0, 0, 0]] * n),
             "opacity": torch.full((n,), 0.99), "rgb": torch.ones(n, 3),
             "activations": {"scale_is_log": False, "opacity_is_logit": False}}
    path = tmp_path / "s_splat.pt"
    torch.save(splat, path)
    origin = GeoPoint(lat=40.89, lon=-74.22, h=233.0)
    src = SceneSource(name="s", cloud="", splat=str(path), kind="catalog",
                      metric=metric_block(origin, earth_model="wgs84", source="test"))
    monkeypatch.setattr("aerosplat.render.engine.discover_scenes", lambda root, data: {"s": src})
    eng = RenderEngine(None, None, idle_s=0.05)
    yield eng, pts.numpy()
    eng.unload_all()


def _centroid(alpha):
    ys, xs = np.nonzero(alpha > 128)
    return np.array([xs.mean() + 0.5, ys.mean() + 0.5]) if len(xs) else None


@needs_gsplat
def test_gpu_perspective_matches_reference(gpu_engine):
    eng, pts = gpu_engine
    req = RenderRequest(position={"e": 0, "n": 0, "u": 0}, yaw_deg=0, pitch_deg=0, width=640,
                        height=480, fov_deg=60, output="alpha")
    out = eng.render("s", req)
    V = np.array(out.meta["view_matrix"])
    K = tuple(out.meta["intrinsics"][k] for k in ("fx", "fy", "cx", "cy"))
    target = pts[0]                                                    # 50 m north, 10 m east
    px, depth = cam.project(V, K, target[None])
    mask = np.zeros_like(out.image)
    x0, y0 = int(px[0, 0]) - 12, int(px[0, 1]) - 12
    mask[max(y0, 0):y0 + 25, max(x0, 0):x0 + 25] = out.image[max(y0, 0):y0 + 25, max(x0, 0):x0 + 25]
    got = _centroid(mask)
    assert got is not None and np.abs(got - px[0]).max() < 1.0


@needs_gsplat
def test_gpu_depth_and_geodetic_position(gpu_engine):
    eng, pts = gpu_engine
    body = dict(yaw_deg=0, pitch_deg=0, width=320, height=240, output="depth")
    out = eng.render("s", RenderRequest(position={"e": 0, "n": 0, "u": 0}, **body))
    d = out.meta["depth"]
    metres = out.image.astype(np.float64) / 65535.0 * d["max_m"]
    lit = out.image > 0
    assert lit.any() and out.image[~lit].max(initial=0) == 0
    assert metres[lit].min() > 20 and metres[lit].max() <= 85                  # nearest 30, farthest 80
    # the same pose given as WGS84 reproduces the ENU camera
    geo = out.meta["camera_geo"]
    out2 = eng.render("s", RenderRequest(position=geo, **body))
    assert np.allclose(out2.meta["camera_enu"], out.meta["camera_enu"], atol=1e-3)


@needs_gsplat
def test_gpu_ortho_nadir_matches_ground_grid(gpu_engine):
    eng, pts = gpu_engine
    req = RenderRequest(mode="ortho", position={"e": 0, "n": 0}, extent_m=200, width=800, height=800,
                        output="alpha")
    out = eng.render("s", req)
    g = out.meta["ground_grid"]
    assert out.meta["camera_enu"][2] > pts[:, 2].max()                      # above the scene
    p00 = np.array(g["pixel00_enu"])
    du, dv = np.array(g["du_enu"]), np.array(g["dv_enu"])
    for target in pts[[0, 3]]:                                              # isolated points
        u, v = np.linalg.solve(np.stack([du, dv], axis=1), target[:2] - p00)
        win = np.zeros_like(out.image)
        x0, y0 = int(u) - 10, int(v) - 10
        win[y0:y0 + 21, x0:x0 + 21] = out.image[y0:y0 + 21, x0:x0 + 21]
        got = _centroid(win)
        assert got is not None and np.abs(got - (np.array([u, v]) + 0.5)).max() < 1.0


@needs_gsplat
def test_gpu_errors_and_idle_unload(gpu_engine):
    from aerosplat.render.engine import RenderError, SceneNotFound

    eng, _ = gpu_engine
    with pytest.raises(SceneNotFound):
        eng.render("nope", RenderRequest(position={"e": 0, "n": 0, "u": 1}))
    with pytest.raises(RenderError):                                       # tilted ortho needs a height
        eng.render("s", RenderRequest(mode="ortho", pitch_deg=-60, position={"e": 0, "n": 0}))
    eng.render("s", RenderRequest(position={"e": 0, "n": 0, "u": 1}, width=32, height=32))
    assert eng.status()["loaded"]
    time.sleep(0.1)
    assert eng.unload_idle() == ["s"] and not eng.status()["loaded"]
