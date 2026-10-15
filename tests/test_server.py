"""API tests for aerosplat/server/main.py: scenes come from a temp catalog (CPU only)."""

import importlib
import struct
import sys

import pytest
import torch

from conftest import write_pt
from aerosplat.geo.crs import GeoPoint
from aerosplat.store import Capture, CaptureSource, Catalog, RunStatus, Site
from aerosplat.store.scenes import CLOUD, SPLAT, SceneSource

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


def test_no_scenes_is_404_and_creates_nothing(env):
    projects, _ = env
    mod = _load_server()
    client = TestClient(mod.app)
    assert mod.DEFAULT_DATASET is None
    r = client.get("/api/v1/scene-meta")
    assert r.status_code == 404 and "No scenes" in r.text
    assert not projects.exists()          # the server never creates a catalog
