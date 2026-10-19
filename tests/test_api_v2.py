"""API v2 (aerosplat.server.api_v2): projects, sites, uploads, runs, jobs, streaming."""

import io
import json
import threading

import pytest
from PIL import Image as PILImage

from conftest import ES_JSON, write_jpeg

pytest.importorskip("httpx")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from aerosplat.jobs import JobQueue, JobStatus, Resource  # noqa: E402
from aerosplat.jobs.worker import Worker  # noqa: E402
from aerosplat.server import api_v2  # noqa: E402

ORIGIN = {"lat": 40.8942, "lon": -74.2173, "h": 233.0}


def _close_catalogs():
    for cat in api_v2._CATALOGS.values():
        cat.close()
    api_v2._CATALOGS.clear()


@pytest.fixture
def root(tmp_path, monkeypatch):
    r = tmp_path / "projects"
    monkeypatch.setenv("AEROSPLAT_PROJECTS", str(r))
    yield r
    _close_catalogs()


@pytest.fixture
def client(root):
    app = FastAPI()
    app.include_router(api_v2.router)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def site(client):
    assert client.post("/api/v2/projects", json={"id": "demo", "name": "Demo"}).status_code == 201
    r = client.post("/api/v2/projects/demo/sites",
                    json={"id": "pier", "name": "Pier 7", "kind": "depot", "origin": ORIGIN})
    assert r.status_code == 201, r.text
    return r.json()


def _jpeg_bytes(shade=90):
    buf = io.BytesIO()
    PILImage.new("RGB", (64, 48), (shade, 120, 60)).save(buf, "JPEG")
    return buf.getvalue()


def _upload(client, capture_id="phone-01", source="handheld", files=None):
    files = files or [("files", (f"img_{i}.jpg", _jpeg_bytes(40 * i), "image/jpeg"))
                      for i in range(2)]
    return client.post("/api/v2/sites/demo/pier/captures",
                       data={"capture_id": capture_id, "source": source, "license": "CC-BY-4.0"},
                       files=files)


def test_projects_and_sites(client, site):
    assert site["origin"] == ORIGIN and site["kind"] == "depot"
    assert [p["id"] for p in client.get("/api/v2/projects").json()] == ["demo"]
    sites = client.get("/api/v2/sites").json()
    assert len(sites) == 1 and sites[0]["site"]["id"] == "pier"
    assert sites[0]["captures"] == 0 and sites[0]["scene"] is None
    detail = client.get("/api/v2/sites/demo/pier").json()
    assert [e["id"] for e in detail["epochs"]] == ["baseline"]   # created with the site

    dup = client.post("/api/v2/projects/demo/sites", json={"id": "pier", "origin": ORIGIN})
    assert dup.status_code == 409
    assert client.post("/api/v2/projects/nope/sites",
                       json={"id": "x", "origin": ORIGIN}).status_code == 404
    assert client.post("/api/v2/projects/demo/sites",
                       json={"id": "Bad Slug", "origin": ORIGIN}).status_code == 422
    assert client.get("/api/v2/sites/demo/missing").status_code == 404


def _drain(root):
    """Run the queued CPU jobs (ingests) in-process, as the CPU worker's job process would."""
    from aerosplat.ingest.upload import run_ingest_job

    cat = api_v2.get_catalog()
    q = JobQueue(cat.engine)
    done = []
    while True:
        job = q.claim("test:cpu:1", Resource.cpu)
        if job is None:
            return done
        rc = run_ingest_job(q, job, cat)
        done.append(q.finish(job.id, JobStatus.succeeded if rc == 0 else JobStatus.failed,
                             error=None if rc == 0 else f"exit code {rc}"))


def test_upload_handheld_capture_and_thumbnail(client, site, root):
    r = _upload(client)
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["kind"] == "ingest" and job["resource"] == "cpu" and job["status"] == "queued"
    assert job["payload"]["capture_id"] == "phone-01" and job["payload"]["files"] == 2
    # not a capture until the job has finished
    assert client.get("/api/v2/sites/demo/pier").json()["captures"] == []
    assert client.get("/api/v2/sites/demo/pier/captures/phone-01/images").status_code == 404
    again = _upload(client)
    assert again.status_code == 409 and "being ingested" in again.json()["detail"]

    [done] = _drain(root)
    assert done.status == JobStatus.succeeded and done.id == job["id"]
    final = client.get(f"/api/v2/jobs/{job['id']}").json()
    assert final["result"] == {"site": "demo/pier", "capture": "phone-01", "images": 2,
                               "bytes": final["result"]["bytes"]}
    stages = [(e["data"]["stage"], e["data"]["state"])
              for e in client.get(f"/api/v2/jobs/{job['id']}/events").json() if e["type"] == "stage"]
    assert ("hash", "done") in stages and stages[-1] == ("register", "done")

    detail = client.get("/api/v2/sites/demo/pier").json()
    [cap] = detail["captures"]
    assert cap["capture"]["id"] == "phone-01" and cap["images"] == 2 and cap["with_prior"] == 0
    assert cap["capture"]["source"] == "handheld" and cap["capture"]["license"] == "CC-BY-4.0"
    images = client.get("/api/v2/sites/demo/pier/captures/phone-01/images").json()
    assert [i["name"] for i in images] == ["img_0.jpg", "img_1.jpg"]

    thumb = client.get("/api/v2/sites/demo/pier/captures/phone-01/images/img_1.jpg/thumb?size=32")
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"
    assert max(PILImage.open(io.BytesIO(thumb.content)).size) == 32
    for bad in ("nope.jpg", "..%2F..%2Fcatalog.db"):
        assert client.get(f"/api/v2/sites/demo/pier/captures/phone-01/images/{bad}/thumb"
                          ).status_code == 404
    assert _upload(client).status_code == 409          # same capture id
    # the staging directory is gone once the capture is registered
    captures_dir = root / "demo" / "sites" / "pier" / "captures"
    assert sorted(p.name for p in captures_dir.iterdir()) == ["phone-01"]


def test_upload_earthstudio_capture(client, site, tmp_path, root):
    data = json.loads(ES_JSON.read_text())
    data["name"] = "orbit"
    files = [("files", ("orbit.json", json.dumps(data).encode(), "application/json"))]
    for i in range(len(data["cameraFrames"])):
        p = tmp_path / f"orbit_{i:02d}.jpeg"
        write_jpeg(p, shade=30 * i)
        files.append(("files", (p.name, p.read_bytes(), "image/jpeg")))
    r = _upload(client, capture_id="es-orbit", source="earthstudio", files=files)
    assert r.status_code == 202, r.text
    assert [j.status for j in _drain(root)] == [JobStatus.succeeded]
    [cap] = client.get("/api/v2/sites/demo/pier").json()["captures"]
    assert cap["images"] == 3 and cap["with_prior"] == 3
    assert cap["capture"]["metadata_file"] == "source/orbit.json"

    # the cheap checks fail the request itself, and leave nothing behind
    no_json = _upload(client, capture_id="es-2", source="earthstudio", files=files[1:])
    assert no_json.status_code == 422
    missing_frame = _upload(client, capture_id="es-3", source="earthstudio", files=files[:-1])
    assert missing_frame.status_code == 422
    no_images = _upload(client, capture_id="es-4", source="handheld", files=files[:1])
    assert no_images.status_code == 422
    bad_name = _upload(client, capture_id="es-5",
                       files=[("files", ("ok.jpg", _jpeg_bytes(), "image/jpeg")),
                              ("files", ("bad|name.jpg", _jpeg_bytes(), "image/jpeg"))])
    assert bad_name.status_code == 422
    captures_dir = root / "demo" / "sites" / "pier" / "captures"
    assert sorted(p.name for p in captures_dir.iterdir()) == ["es-orbit"]
    assert client.get("/api/v2/jobs", params={"site": "demo/pier"}).json()[0]["id"] == r.json()["id"]


def test_start_run_validation_and_queue(client, site, root):
    assert client.post("/api/v2/sites/demo/pier/runs", json={}).status_code == 409   # no capture
    _upload(client)
    assert client.post("/api/v2/sites/demo/pier/runs", json={}).status_code == 409   # not ingested yet
    _drain(root)
    r = client.post("/api/v2/sites/demo/pier/runs",
                    json={"from_stage": "sfm", "overrides": {"iterations": 500}})
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["status"] == "queued" and job["resource"] == "gpu" and job["kind"] == "pipeline"
    assert job["site_key"] == "demo/pier" and job["payload"]["capture"] == "phone-01"
    assert job["payload"]["overrides"] == {"iterations": 500}

    for bad in ({"overrides": {"iterations": 0}}, {"overrides": {"nope": 1}},
                {"stages": ["nope"]}, {"capture": "missing"}):
        assert client.post("/api/v2/sites/demo/pier/runs", json=bad).status_code in (404, 422)
    _upload(client, capture_id="phone-02")
    _drain(root)
    assert client.post("/api/v2/sites/demo/pier/runs", json={}).status_code == 422  # ambiguous

    sites = client.get("/api/v2/sites").json()
    assert sites[0]["active_jobs"] == 1
    listed = client.get("/api/v2/jobs", params={"site": "demo/pier"}).json()
    assert [j["id"] for j in listed if j["kind"] == "pipeline"] == [job["id"]]
    assert client.get("/api/v2/jobs", params={"status": "running"}).json() == []

    cancelled = client.post(f"/api/v2/jobs/{job['id']}/cancel").json()
    assert cancelled["status"] == "cancelled"
    events = client.get(f"/api/v2/jobs/{job['id']}/events").json()
    assert [e["data"]["status"] for e in events] == ["queued", "cancelled"]
    assert client.get("/api/v2/jobs/nope").status_code == 404


def test_job_stream_follows_a_real_worker(client, root):
    job = client.post("/api/v2/jobs/diagnostic",
                      json={"stages": 2, "seconds_per_stage": 0.2, "lines_per_stage": 2}).json()
    assert job["resource"] == "cpu"
    worker = Worker(root, Resource.cpu, poll_s=0.05)
    t = threading.Thread(target=worker.run, kwargs={"once": True}, daemon=True)
    t.start()
    statuses, stages, lines = [], [], []
    with client.websocket_connect(f"/api/v2/jobs/{job['id']}/stream") as ws:
        while True:
            try:
                msg = ws.receive_json()
            except Exception:   # server closed the stream after the terminal status
                break
            if msg["type"] == "job":
                statuses.append(msg["job"]["status"])
            for e in msg.get("events", []):
                if e["type"] == "stage":
                    stages.append((e["data"]["stage"], e["data"]["state"]))
                elif e["type"] == "log":
                    lines += e["data"]["lines"]
    t.join(timeout=30)
    assert statuses[-1] == "succeeded"
    assert stages == [("step1", "started"), ("step1", "done"), ("step2", "started"),
                      ("step2", "done")]
    assert "   step1: line 1/2" in lines

    with client.websocket_connect("/api/v2/jobs/nope/stream") as ws:
        assert ws.receive_json()["type"] == "error"


def test_system_and_pipeline_info(client, root):
    q = JobQueue(api_v2.get_catalog().engine)
    q.register_worker("h:gpu:1", Resource.gpu, "h", 1)
    info = client.get("/api/v2/system").json()
    assert info["projects_root"] == str(root) and info["queue"]["queued"] == 0
    assert [w["id"] for w in info["workers"]] == ["h:gpu:1"] and info["workers"][0]["online"]
    assert info["gpu"] is None or isinstance(info["gpu"], list)
    meta = client.get("/api/v2/pipeline").json()
    assert [s["name"] for s in meta["stages"]][:2] == ["geo", "exif"]
    assert meta["defaults"]["iterations"] > 0 and "properties" in meta["config_schema"]


def test_run_detail_from_manifest(es_site, monkeypatch, client):
    from aerosplat import starter

    cat, site, cap, root = es_site
    monkeypatch.setenv("AEROSPLAT_PROJECTS", str(root))
    assert starter.main(["--site", site.key, "--only", "geo", "--run-id", "r1",
                         "--projects-root", str(root)]) == 0
    detail = client.get("/api/v2/sites/demo/es/runs/baseline/r1").json()
    assert detail["run"]["status"] == "ok" and detail["run"]["georeferenced"]
    assert [(s["stage"], s["status"]) for s in detail["stages"]] == [("geo", "ok")]
    assert detail["scene"] is None and detail["has_report"] is False   # no splat/cloud yet
    assert client.get("/api/v2/sites/demo/es/runs/baseline/r1/report").status_code == 404
    assert client.get("/api/v2/sites/demo/es/runs/baseline/nope").status_code == 404
    assert JobQueue(cat.engine).counts()[JobStatus.queued.value] == 0
