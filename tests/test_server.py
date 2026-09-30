"""API tests for aerosplat/server/main.py on tiny synthetic .pt files (CPU only)."""

import importlib
import sys
import struct

import pytest
import torch

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

N = 200


def _load_server():
    sys.modules.pop("aerosplat.server.main", None)  # fresh module state per test
    return importlib.import_module("aerosplat.server.main")


@pytest.fixture
def server(tmp_path, monkeypatch):
    mod = _load_server()
    g = torch.Generator().manual_seed(0)
    xyz = torch.randn(N, 3, generator=g) * 10 + torch.tensor([500.0, 200.0, 50.0])
    torch.save({"xyz": xyz, "rgb": torch.rand(N, 3, generator=g)}, tmp_path / "cloud.pt")
    rot = torch.nn.functional.normalize(torch.randn(N, 4, generator=g), dim=1)
    torch.save({
        "xyz": xyz, "scale": torch.full((N, 3), 0.1), "rotation": rot,
        "opacity": torch.full((N, 1), 0.8), "rgb": torch.rand(N, 3, generator=g),
        "activations": {"scale_is_log": False, "opacity_is_logit": False},
    }, tmp_path / "splat.pt")
    torch.save({"xyz": xyz, "rgb": torch.rand(N, 3)}, tmp_path / "onlycloud.pt")
    monkeypatch.setattr(mod, "DATASET_PATHS", {
        "t": {"cloud": str(tmp_path / "cloud.pt"), "splat": str(tmp_path / "splat.pt")},
        "nosplat": {"cloud": str(tmp_path / "onlycloud.pt"), "splat": str(tmp_path / "none.pt")},
    })
    monkeypatch.setattr(mod, "DEFAULT_DATASET", "t")
    mod._PAYLOADS.clear()
    mod._FRAMES.clear()
    return TestClient(mod.app), mod


def test_dense_chunk_shape_and_headers(server):
    client, mod = server
    r = client.get("/api/v1/spatial-chunk", params={"mode": "dense", "dataset": "t"})
    assert r.status_code == 200
    assert int(r.headers["x-gaussians-count"]) == len(r.content) // (mod.DENSE_STRIDE * 4)
    assert r.headers["x-dataset"] == "t"
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


def test_missing_splat_file_is_404(server):
    client, _ = server
    r = client.get("/api/v1/spatial-chunk", params={"mode": "splat", "dataset": "nosplat"})
    assert r.status_code == 404


def test_malformed_pt_is_500(server, tmp_path):
    client, mod = server
    torch.save({"xyz": torch.zeros(3, 3)}, tmp_path / "bad.pt")
    mod.DATASET_PATHS["bad"] = {"cloud": str(tmp_path / "bad.pt"), "splat": str(tmp_path / "bad.pt")}
    r = client.get("/api/v1/spatial-chunk", params={"mode": "dense", "dataset": "bad"})
    assert r.status_code == 500 and "missing keys" in r.text


def test_nonfinite_points_are_dropped(server, tmp_path):
    client, mod = server
    xyz = torch.randn(N, 3)
    xyz[0, 0] = float("nan")
    torch.save({"xyz": xyz, "rgb": torch.rand(N, 3)}, tmp_path / "nan.pt")
    mod.DATASET_PATHS["nan"] = {"cloud": str(tmp_path / "nan.pt"), "splat": ""}
    meta = client.get("/api/v1/scene-meta", params={"mode": "dense", "dataset": "nan"}).json()
    assert meta["dropped_nonfinite"] == 1 and meta["count"] == N - 1


def test_scene_meta_contract(server):
    client, mod = server
    meta = client.get("/api/v1/scene-meta", params={"mode": "splat"}).json()
    assert meta["dataset"] == "t" and meta["stride_floats"] == mod.SPLAT_STRIDE
    assert meta["splat_layout"] == mod.SPLAT_LAYOUT
    assert meta["activations"] is not None
    assert meta["max_addressable_splats"] == 2 ** 22
    dense = client.get("/api/v1/scene-meta", params={"mode": "dense"}).json()
    assert dense["splat_layout"] is None and dense["center"] == meta["center"]


def test_status_and_reload(server):
    client, _ = server
    client.get("/api/v1/spatial-chunk", params={"mode": "dense"})
    st = client.get("/api/v1/status").json()
    assert st["datasets"]["t"] == {"splat": True, "dense": True}
    assert st["datasets"]["nosplat"]["splat"] is False
    assert len(st["cached_payloads"]) == 1
    assert client.post("/api/v1/reload", params={"dataset": "t"}).json()["dropped_payloads"] == 1
    assert client.get("/api/v1/status").json()["cached_payloads"] == []
