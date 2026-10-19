"""API tests for aerosplat/server/main.py: scenes come from a temp catalog (CPU only)."""

import importlib
import json
import shutil
import struct
import sys

import pytest
import torch

from conftest import write_pt
from aerosplat.geo.crs import GeoPoint
from aerosplat.geo.sphere_enu import EARTH_MODEL
from aerosplat.store import Capture, CaptureSource, Catalog, RunStatus, Site
from aerosplat.store.scenes import CLOUD, REGISTRATION, SPLAT, SceneSource, legacy_scenes

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

N = 200
ORIGIN = GeoPoint(lat=40.8942, lon=-74.2173, h=233.0)


def _load_server():
    sys.modules.pop("aerosplat.server.main", None)  # fresh module state per test
    return importlib.import_module("aerosplat.server.main")


def add_site_run(root, site_id, *, splat=True, georeferenced=True, run_id="20261001-000000"):
    """A site in `root`'s catalog with one finished run holding .pt products."""
    cat = Catalog.open(root)
    cat.ensure_project("demo")
    site = cat.create_site(Site(id=site_id, project_id="demo", origin=ORIGIN))
    cat.ensure_epoch(site)
    cat.add_capture(site, Capture(id="es", source=CaptureSource.earthstudio), [])
    cfg = {"camera_source": "earthstudio_json" if georeferenced else "synthetic"}
    cat.create_run(site, "baseline", run_id, capture_ids=["es"], config=cfg,
                   georeferenced=georeferenced)
    rd = cat.run_dirs(site, "baseline", run_id)
    write_pt(rd.gs / "dense.pt", "dense", N)
    products = {CLOUD: rd.gs / "dense.pt"}
    if splat:
        write_pt(rd.gs / "splat.pt", "splat", N)
        products[SPLAT] = rd.gs / "splat.pt"
    cat.finish_run(site, "baseline", run_id, status=RunStatus.ok, products=products)
    cat.close()


@pytest.fixture
def env(tmp_path, monkeypatch):
    projects, data = tmp_path / "projects", tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("AEROSPLAT_PROJECTS", str(projects))
    monkeypatch.setenv("AEROSPLAT_DATA", str(data))
    monkeypatch.delenv("AEROSPLAT_DATASET", raising=False)
    return projects, data


@pytest.fixture
def server(env):
    projects, _ = env
    add_site_run(projects, "t")
    add_site_run(projects, "nosplat", splat=False, georeferenced=False)
    mod = _load_server()
    assert mod.DEFAULT_DATASET == "demo/nosplat"     # first catalog scene, sorted
    mod.DEFAULT_DATASET = "demo/t"
    return TestClient(mod.app), mod


def _add_scene(mod, tmp_path, name, xyz=None):
    path = tmp_path / f"{name}.pt"
    xyz = torch.randn(N, 3) if xyz is None else xyz
    torch.save({"xyz": xyz, "rgb": torch.rand(N, 3)}, path)
    mod.SCENES[name] = SceneSource(name=name, cloud=str(path), splat=str(path), metric=None,
                                   kind="legacy")
    return path


def test_dense_chunk_shape_and_headers(server):
    client, mod = server
    r = client.get("/api/v1/spatial-chunk", params={"mode": "dense", "dataset": "t"})
    assert r.status_code == 200
    assert int(r.headers["x-gaussians-count"]) == len(r.content) // (mod.DENSE_STRIDE * 4)
    assert r.headers["x-dataset"] == "demo/t"           # bare site id -> canonical key
    floats = struct.unpack(f"<{len(r.content) // 4}f", r.content)
    assert all(v == v and abs(v) < 1e6 for v in floats)  # finite, centred


def test_splat_chunk_is_padded_to_texture_rows(server):
    client, mod = server
    r = client.get("/api/v1/spatial-chunk", params={"mode": "splat"})
    assert r.status_code == 200
    padded = int(r.headers["x-padded-count"])
    assert padded % mod.SPLATS_PER_TEXTURE_ROW == 0
    assert len(r.content) == padded * mod.SPLAT_STRIDE * 4
    assert int(r.headers["x-gaussians-count"]) <= padded


def test_etag_gives_304(server):
    client, _ = server
    first = client.get("/api/v1/spatial-chunk", params={"mode": "dense"})
    again = client.get("/api/v1/spatial-chunk", params={"mode": "dense"},
                       headers={"If-None-Match": first.headers["etag"]})
    assert again.status_code == 304 and not again.content


def test_step_subsamples(server):
    client, _ = server
    full = client.get("/api/v1/spatial-chunk", params={"mode": "dense"})
    sub = client.get("/api/v1/spatial-chunk", params={"mode": "dense", "step": 4})
    assert int(sub.headers["x-gaussians-count"]) < int(full.headers["x-gaussians-count"])


@pytest.mark.parametrize("params,code", [
    ({"mode": "bogus"}, 400),
    ({"dataset": "nope"}, 404),
    ({"step": 0}, 422),
    ({"step": 65}, 422),
])
def test_chunk_errors(server, params, code):
    client, _ = server
    assert client.get("/api/v1/spatial-chunk", params=params).status_code == code


def test_missing_splat_product_is_404(server):
    client, _ = server
    r = client.get("/api/v1/spatial-chunk", params={"mode": "splat", "dataset": "demo/nosplat"})
    assert r.status_code == 404


def test_malformed_pt_is_500(server, tmp_path):
    client, mod = server
    torch.save({"xyz": torch.zeros(3, 3)}, tmp_path / "bad.pt")
    mod.SCENES["bad"] = SceneSource(name="bad", cloud=str(tmp_path / "bad.pt"),
                                    splat=str(tmp_path / "bad.pt"), metric=None, kind="legacy")
    r = client.get("/api/v1/spatial-chunk", params={"mode": "dense", "dataset": "bad"})
    assert r.status_code == 500 and "missing keys" in r.text


def test_nonfinite_points_are_dropped(server, tmp_path):
    client, mod = server
    xyz = torch.randn(N, 3)
    xyz[0, 0] = float("nan")
    _add_scene(mod, tmp_path, "nan", xyz)
    meta = client.get("/api/v1/scene-meta", params={"mode": "dense", "dataset": "nan"}).json()
    assert meta["dropped_nonfinite"] == 1 and meta["count"] == N - 1


def test_scene_meta_contract(server):
    client, mod = server
    meta = client.get("/api/v1/scene-meta", params={"mode": "splat"}).json()
    assert meta["dataset"] == "demo/t" and meta["stride_floats"] == mod.SPLAT_STRIDE
    assert meta["splat_layout"] == mod.SPLAT_LAYOUT
    assert meta["activations"] is not None
    assert meta["max_addressable_splats"] == 2 ** 22
    dense = client.get("/api/v1/scene-meta", params={"mode": "dense"}).json()
    assert dense["splat_layout"] is None and dense["center"] == meta["center"]


def test_scene_meta_metric_from_catalog(server):
    client, _ = server
    m = client.get("/api/v1/scene-meta", params={"mode": "splat", "dataset": "t"}).json()["metric"]
    assert m["units_to_metres"] == 1.0 and m["crs"] == "ENU"
    assert m["origin"] == {"lat": ORIGIN.lat, "lon": ORIGIN.lon, "h": ORIGIN.h}
    assert m["earth_model"] == "wgs84" and m["frame_error_m"] is None
    assert m["site"] == "demo/t" and m["run"] == "20261001-000000"
    # not georeferenced (synthetic GPS / free SfM): honest None, not a guess
    rel = client.get("/api/v1/scene-meta", params={"mode": "dense", "dataset": "nosplat"}).json()
    assert rel["metric"] is None


def test_alias_and_canonical_share_cache(server):
    client, mod = server
    client.get("/api/v1/spatial-chunk", params={"mode": "dense", "dataset": "t"})
    client.get("/api/v1/spatial-chunk", params={"mode": "dense", "dataset": "demo/t"})
    assert [k[0] for k in mod._PAYLOADS] == ["demo/t"]


def test_status_and_reload(server):
    client, _ = server
    client.get("/api/v1/spatial-chunk", params={"mode": "dense"})
    st = client.get("/api/v1/status").json()
    assert st["datasets"]["demo/t"] == {"splat": True, "dense": True}
    assert st["datasets"]["demo/nosplat"]["splat"] is False
    assert st["scenes"]["demo/t"] == {"kind": "catalog", "site": "demo/t",
                                      "run": "20261001-000000", "aliases": ["t"], "metric": True}
    assert len(st["cached_payloads"]) == 1
    assert client.post("/api/v1/reload", params={"dataset": "t"}).json()["dropped_payloads"] == 1
    assert client.get("/api/v1/status").json()["cached_payloads"] == []


def test_full_reload_rereads_catalog(server, env):
    client, _ = server
    add_site_run(env[0], "later")
    assert client.get("/api/v1/scene-meta", params={"dataset": "later"}).status_code == 404
    assert "demo/later" in client.post("/api/v1/reload").json()["scenes"]
    assert client.get("/api/v1/scene-meta", params={"dataset": "later"}).status_code == 200


def test_newer_run_replaces_scene(server, env):
    client, mod = server
    cat = Catalog.open(env[0])
    site = cat.get_site("demo", "t")
    cat.create_run(site, "baseline", "20261002-000000", georeferenced=True)
    rd = cat.run_dirs(site, "baseline", "20261002-000000")
    write_pt(rd.gs / "splat.pt", "splat", 10)
    cat.finish_run(site, "baseline", "20261002-000000", status=RunStatus.ok,
                   products={SPLAT: rd.gs / "splat.pt"})
    cat.close()
    client.post("/api/v1/reload")
    assert mod.SCENES["demo/t"].run == "20261002-000000"


def test_legacy_outputs_are_still_served(env):
    _, data = env
    write_pt(data / "old_dense.pt", "dense", N)
    write_pt(data / "old_splat.pt", "splat", N)
    mod = _load_server()
    client = TestClient(mod.app)
    assert mod.DEFAULT_DATASET == "old" and mod.SCENES["old"].kind == "legacy"
    meta = client.get("/api/v1/scene-meta", params={"mode": "splat"}).json()
    assert meta["dataset"] == "old" and meta["metric"] is None   # no geo registration


@pytest.mark.parametrize("deleted", [False, True])
def test_catalog_scene_shadows_legacy_only_while_its_files_exist(env, caplog, deleted):
    """A catalog site overrides the legacy scene of the same name, but not once its
    run folder is gone: then `?dataset=old` is the legacy data/old_*.pt again."""
    projects, data = env
    write_pt(data / "old_dense.pt", "dense", N)
    write_pt(data / "old_splat.pt", "splat", N)
    add_site_run(projects, "old")
    if deleted:
        cat = Catalog.open(projects)
        shutil.rmtree(cat.site_dirs(cat.get_site("demo", "old")).root)
        cat.close()
    mod = _load_server()
    client = TestClient(mod.app)
    meta = client.get("/api/v1/scene-meta", params={"dataset": "old"})
    assert meta.status_code == 200
    if deleted:
        assert "demo/old" not in mod.SCENES and mod.SCENES["old"].kind == "legacy"
        assert meta.json()["dataset"] == "old"
        assert "demo/old: run 20261001-000000" in caplog.text
        assert client.get("/api/v2/sites/demo/old").json()["scene"] is None
    else:
        assert mod.SCENES["old"] is mod.SCENES["demo/old"]
        assert mod.SCENES["old"].kind == "catalog" and meta.json()["dataset"] == "demo/old"
        assert client.get("/api/v2/sites/demo/old").json()["scene"] == "demo/old"
    for cat in mod.api_v2._CATALOGS.values():
        cat.close()
    mod.api_v2._CATALOGS.clear()


def test_run_with_missing_files_falls_back_to_older_run(server, env, caplog):
    client, mod = server
    cat = Catalog.open(env[0])
    site = cat.get_site("demo", "t")
    cat.create_run(site, "baseline", "20261002-000000", georeferenced=True)
    rd = cat.run_dirs(site, "baseline", "20261002-000000")
    write_pt(rd.gs / "splat.pt", "splat", 10)
    cat.finish_run(site, "baseline", "20261002-000000", status=RunStatus.ok,
                   products={SPLAT: rd.gs / "splat.pt"})
    cat.close()
    (rd.gs / "splat.pt").unlink()
    client.post("/api/v1/reload")
    assert mod.SCENES["demo/t"].run == "20261001-000000"
    assert "беру старіший run 20261001-000000" in caplog.text


def test_scene_meta_metric_for_sphere_enu_run(env):
    """An Earth Studio run (sphere ENU + affine): the served products are its WGS84 Site
    ENU files and frame_error_m is the affine map's residual."""
    projects, _ = env
    cat = Catalog.open(projects)
    cat.ensure_project("demo")
    site = cat.create_site(Site(id="es", project_id="demo", origin=ORIGIN))
    cat.ensure_epoch(site)
    cat.create_run(site, "baseline", "r1", earth_model=EARTH_MODEL, georeferenced=True)
    rd = cat.run_dirs(site, "baseline", "r1")
    write_pt(rd.gs / "dense_wgs84.pt", "dense", N)
    write_pt(rd.gs / "splat_wgs84.pt", "splat", N)
    rd.reports.mkdir(parents=True)
    rd.registration_report.write_text(json.dumps(
        {"sphere_enu_to_site_enu": {"residual_max_m": 0.0002}}))
    cat.finish_run(site, "baseline", "r1", status=RunStatus.ok,
                   products={SPLAT: rd.gs / "splat_wgs84.pt", CLOUD: rd.gs / "dense_wgs84.pt",
                             REGISTRATION: rd.registration_report})
    cat.close()
    client = TestClient(_load_server().app)
    m = client.get("/api/v1/scene-meta", params={"mode": "splat", "dataset": "es"}).json()["metric"]
    assert m["earth_model"] == EARTH_MODEL and m["frame_error_m"] == 0.0002
    assert m["datum"].startswith("WGS84") and m["origin"]["lat"] == ORIGIN.lat


def test_legacy_sphere_enu_dataset_is_metric_only_once_mapped(env):
    """`aerosplat run --dataset` with Earth Studio poses: data/<ds>_*_wgs84.pt (georef stage)
    are served with a metric block; the sphere-ENU originals alone are not called metric."""
    _, data = env
    write_pt(data / "es_dense.pt", "dense", N)
    write_pt(data / "es_splat.pt", "splat", N)
    (data / "es" / "sparse_known").mkdir(parents=True)
    (data / "es" / "sparse_known" / "geo_registration.json").write_text(json.dumps({
        "earth_model": EARTH_MODEL, "origin_source": "frames_mean (legacy)",
        "enu_origin": {"latitude": ORIGIN.lat, "longitude": ORIGIN.lon, "altitude": ORIGIN.h},
        "sphere_enu_to_site_enu": {"residual_max_m": 0.0002}}))
    (data / "es" / "run").mkdir()
    (data / "es" / "run" / "manifest.json").write_text(json.dumps(
        {"config": {"camera_source": "earthstudio_json", "known_poses": True,
                    "cloud_source": "dense"}}))
    scene = legacy_scenes(data)["es"]
    assert scene.metric is None and scene.cloud == str(data / "es_dense.pt")

    write_pt(data / "es_dense_wgs84.pt", "dense", N)
    write_pt(data / "es_splat_wgs84.pt", "splat", N)
    scenes = legacy_scenes(data)
    assert set(scenes) == {"es"}                       # *_wgs84.pt is not a dataset of its own
    scene = scenes["es"]
    assert scene.cloud == str(data / "es_dense_wgs84.pt")
    assert scene.splat == str(data / "es_splat_wgs84.pt")
    assert scene.metric["earth_model"] == EARTH_MODEL and scene.metric["frame_error_m"] == 0.0002


def test_no_scenes_is_404_and_creates_nothing(env):
    projects, _ = env
    mod = _load_server()
    client = TestClient(mod.app)
    assert mod.DEFAULT_DATASET is None
    r = client.get("/api/v1/scene-meta")
    assert r.status_code == 404 and "No scenes" in r.text
    assert not projects.exists()          # the server never creates a catalog


def test_v2_router_and_scene_refresh(server, env):
    client, mod = server
    projects, _ = env
    assert client.get("/api/v2/sites").status_code == 200
    first = client.post("/api/v2/scenes/refresh").json()
    assert first["changed"] == [] and "demo/t" in first["scenes"]
    # A worker finishing a run of a new site in another process is picked up
    # on refresh, without restarting the server.
    add_site_run(projects, "fresh")
    assert "demo/fresh" not in mod.SCENES
    second = client.post("/api/v2/scenes/refresh").json()
    assert second["changed"] == ["demo/fresh"] and "demo/fresh" in mod.SCENES
    for cat in mod.api_v2._CATALOGS.values():
        cat.close()
    mod.api_v2._CATALOGS.clear()


def test_root_redirect_and_ui_fallback(server, tmp_path, monkeypatch):
    client, mod = server
    monkeypatch.setattr(mod, "UI_DIST", tmp_path / "no-build")
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/web/"
    assert client.get("/app/sites/x").status_code == 404
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>ui</title>")
    (dist / "assets" / "a.js").write_text("console.log(1)")
    monkeypatch.setattr(mod, "UI_DIST", dist)
    assert client.get("/", follow_redirects=False).headers["location"] == "/app/"
    assert "<title>ui</title>" in client.get("/app/sites/demo/t").text   # SPA route
    assert client.get("/app/assets/a.js").text == "console.log(1)"
    for escape in ("/app/../../pyproject.toml", "/app/..%2F..%2F..%2Fpyproject.toml"):
        r = client.get(escape)
        assert "[project]" not in r.text            # never a file outside dist/


def test_up_axis_follows_the_frame(server):
    """ENU is Z-up → Rx(−90°) keeps up up; Rx(+90°) put the camera under the ground."""
    client, _ = server
    enu = client.get("/api/v1/scene-meta", params={"mode": "splat", "dataset": "t"}).json()
    assert enu["metric"] is not None
    assert enu["up_axis_rotation"] == {"axis": "x", "degrees": -90.0}
    free = client.get("/api/v1/scene-meta", params={"mode": "dense", "dataset": "nosplat"}).json()
    assert free["metric"] is None
    assert free["up_axis_rotation"] == {"axis": "x", "degrees": 90.0}


def test_rasterize_mode_reaches_the_viewer(server, tmp_path):
    client, mod = server
    # no `render` key: an Inria/torch file, which is classic
    meta = client.get("/api/v1/scene-meta", params={"mode": "splat"}).json()
    assert meta["rasterize_mode"] == "classic" and meta["effective_step"] == 1
    path = tmp_path / "aa.pt"
    write_pt(path, "splat", N)
    data = torch.load(path)
    data["render"] = {"rasterizer": "gsplat", "rasterize_mode": "antialiased"}
    torch.save(data, path)
    mod.SCENES["aa"] = SceneSource(name="aa", cloud=str(path), splat=str(path), metric=None,
                                   kind="legacy")
    meta = client.get("/api/v1/scene-meta", params={"mode": "splat", "dataset": "aa"}).json()
    assert meta["rasterize_mode"] == "antialiased"


def test_zero_quaternions_are_dropped(server, tmp_path):
    client, mod = server
    path = tmp_path / "zq.pt"
    write_pt(path, "splat", N)
    data = torch.load(path)
    data["rotation"][:3] = 0.0
    torch.save(data, path)
    mod.SCENES["zq"] = SceneSource(name="zq", cloud=str(path), splat=str(path), metric=None,
                                   kind="legacy")
    meta = client.get("/api/v1/scene-meta", params={"mode": "splat", "dataset": "zq"}).json()
    assert meta["dropped_nonfinite"] == 3 and meta["count"] == N - 3


def test_splat_ceiling_is_enforced(server, monkeypatch):
    """Past the fp32 texel-addressing ceiling the server thins instead of corrupting."""
    client, mod = server
    monkeypatch.setattr(mod, "MAX_ADDRESSABLE_SPLATS", 60)
    meta = client.get("/api/v1/scene-meta", params={"mode": "splat"}).json()
    assert 0 < meta["count"] <= 60
    assert meta["effective_step"] == 4          # ceil(200 / 60)


def test_quantile_handles_more_than_2_pow_24(server):
    _, mod = server
    v = torch.rand(17_000_000)
    assert 0.98 < mod._quantile(v, 0.99) <= 1.0


def test_dense_etag_follows_the_splat_frame(server, env):
    client, mod = server
    first = client.get("/api/v1/spatial-chunk", params={"mode": "dense"}).headers["etag"]
    mod._FRAMES["demo/t"] = mod.SceneFrame(center=(1.0, 2.0, 3.0), extent=10.0,
                                           scene_scale=10.0, source="splat.pt")
    mod._PAYLOADS.clear()
    again = client.get("/api/v1/spatial-chunk", params={"mode": "dense"}).headers["etag"]
    assert first != again


def test_classic_viewer_files_revalidate(server):
    client, _ = server
    r = client.get("/web/appGLSL.js")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"


def test_ui_modules_are_served_as_javascript(server, tmp_path, monkeypatch):
    """MapLibre's worker is a .mjs module; served as text/plain (the Windows registry's type)
    the browser refuses it and no GeoJSON layer of a map ever draws."""
    client, mod = server
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html>", encoding="utf-8")
    (dist / "assets" / "maplibre-gl-worker-x.mjs").write_text("export {};", encoding="utf-8")
    monkeypatch.setattr(mod, "UI_DIST", dist)
    r = client.get("/app/assets/maplibre-gl-worker-x.mjs")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/javascript")


def test_the_mission_viewer_page_is_served(server):
    client, _ = server
    r = client.get("/web/mission.html")
    assert r.status_code == 200 and "mission-overlay.js" in r.text
    assert client.get("/web/mission-overlay.js").status_code == 200
