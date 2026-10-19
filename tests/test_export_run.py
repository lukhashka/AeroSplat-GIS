"""`aerosplat export` on a catalog run: refusals, products, manifest, the job and the API.

A synthetic georeferenced run (sphere-enu+affine, like Earth Studio) with a cloud, a
splat, a registration report and semantics -- everything an export reads, nothing else.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rasterio")
pytest.importorskip("laspy")

from aerosplat.export import run as export_run                         # noqa: E402
from aerosplat.export.georef import ExportRefused                      # noqa: E402
from aerosplat.export.run import ExportOptions, ExportTarget           # noqa: E402
from aerosplat.geo.crs import EnuFrame, GeoPoint                       # noqa: E402
from aerosplat.semantics.models import SemanticsResult, TerrainModel   # noqa: E402
from aerosplat.store import Catalog, Site                              # noqa: E402
from aerosplat.store.models import RunStatus                           # noqa: E402

ORIGIN = GeoPoint(lat=39.76003596749297, lon=-75.57385466041443, h=220.0)   # dataset_1910
FRAME = {"crs": "WGS84 Site ENU", "earth_model": "sphere-enu+affine",
         "residual_max_m": 0.00022}


def _make_run(cat: Catalog, site: Site, run_id: str, *, georeferenced: bool = True,
              earth_model: str = "sphere-enu+affine", semantics: bool = True) -> None:
    cat.create_run(site, "baseline", run_id, capture_ids=[], config={},
                   earth_model=earth_model, georeferenced=georeferenced)
    root = cat.run_dirs(site, "baseline", run_id).root
    (root / "gs").mkdir(parents=True)
    (root / "reports").mkdir()
    rng = np.random.default_rng(1)
    e, n = np.meshgrid(np.arange(-30.0, 30.0, 0.5), np.arange(-20.0, 20.0, 0.5))
    u = 0.1 * e + np.where((np.abs(e) < 6) & (np.abs(n) < 5), 9.0, 0.0)
    xyz = np.column_stack([e.ravel(), n.ravel(), u.ravel()])
    frame = FRAME if earth_model == "sphere-enu+affine" else None
    cloud = {"xyz": torch.tensor(xyz, dtype=torch.float32),
             "rgb": torch.tensor(rng.uniform(0, 1, xyz.shape), dtype=torch.float32),
             "meta": {"num_points": len(xyz), "source": "fused.ply"}}
    k = len(xyz)
    q = rng.normal(size=(k, 4))
    splat = {"xyz": torch.tensor(xyz, dtype=torch.float32),
             "scale": torch.full((k, 3), 0.3), "rotation": torch.tensor(q, dtype=torch.float32),
             "opacity": torch.full((k, 1), 0.9),
             "sh": torch.tensor(rng.normal(size=(k, 4, 3)) * 0.2, dtype=torch.float32),
             "rgb": torch.full((k, 3), 0.5),
             "activations": {"scale_is_log": False, "opacity_is_logit": False},
             "render": {"rasterizer": "gsplat", "rasterize_mode": "antialiased"}}
    if frame:
        cloud["frame"] = splat["frame"] = frame
    suffix = "_wgs84" if earth_model == "sphere-enu+affine" else ""
    torch.save(cloud, root / "gs" / f"dense{suffix}.pt")
    torch.save(splat, root / "gs" / f"splat{suffix}.pt")
    (root / "reports" / "registration.json").write_text(json.dumps({
        "earth_model": {"id": earth_model},
        "altitude_reference": "Earth Studio altitude taken as WGS84 ellipsoidal height (unverified)",
        "sphere_enu_to_site_enu": {"A": np.eye(3).tolist(), "b": [0, 0, 0],
                                   "residual_max_m": 0.00022},
        "sphere_to_wgs84": {"frames": {"max_m": 1.09}}}), encoding="utf-8")
    if semantics:
        model = TerrainModel(res_m=1.0, x0=-30.0, y0=-20.0, rows=40, cols=60, n_points=k,
                             filled_cells=2400)
        ce, cn = model.cell_centres()
        dsm = 0.1 * ce + np.where((np.abs(ce) < 6) & (np.abs(cn) < 5), 9.0, 0.0)
        sem = root / "semantics"
        sem.mkdir()
        (sem / "semantics.json").write_text(SemanticsResult(
            terrain=model, buildings=[], facades=[]).model_dump_json(), encoding="utf-8")
        np.savez_compressed(sem / "terrain.npz", dsm=dsm, dtm=0.1 * ce,
                            count=np.ones(dsm.shape, np.int64), roughness=np.zeros_like(dsm))
    cat.finish_run(site, "baseline", run_id, status=RunStatus.ok, georeferenced=georeferenced,
                   products={"splat": root / "gs" / f"splat{suffix}.pt",
                             "cloud": root / "gs" / f"dense{suffix}.pt",
                             "registration_report": root / "reports" / "registration.json",
                             "semantics": root / "semantics"})


@pytest.fixture
def cat(tmp_path, monkeypatch):
    monkeypatch.setenv("AEROSPLAT_PROJECTS", str(tmp_path / "projects"))
    c = Catalog.open(tmp_path / "projects")
    c.ensure_project("demo")
    site = c.create_site(Site(id="mill", project_id="demo", origin=ORIGIN))
    c.ensure_epoch(site)
    _make_run(c, site, "r1")
    yield c
    c.close()


def _site(c):
    return c.get_site("demo", "mill")


def test_ungeoreferenced_run_is_refused_and_nothing_is_written(cat):
    site = _site(cat)
    _make_run(cat, site, "synthetic", georeferenced=False, earth_model="wgs84")
    run = cat.get_run(site, "baseline", "synthetic")
    with pytest.raises(ExportRefused, match="no georeference"):
        ExportTarget.for_run(cat, site, run)
    rc = export_run.main(["--site", "demo/mill", "--run", "synthetic", "--kinds", "las,ply",
                          "--projects-root", str(cat.paths.root)])
    assert rc == 2
    assert not (cat.run_dirs(site, "baseline", "synthetic").root / "exports").exists()


def test_failed_run_is_refused(cat):
    site = _site(cat)
    _make_run(cat, site, "broken")
    cat.finish_run(site, "baseline", "broken", status=RunStatus.failed)
    with pytest.raises(ExportRefused, match="not ok"):
        ExportTarget.resolve(cat, "demo/mill", run_id="broken")


def test_sphere_frame_product_is_refused(cat):
    """A sphere-enu+affine run must export the georef stage's file, never the recon frame."""
    site = _site(cat)
    root = cat.run_dirs(site, "baseline", "r1").root
    data = torch.load(root / "gs" / "dense_wgs84.pt", weights_only=False)
    del data["frame"]
    torch.save(data, root / "gs" / "dense_wgs84.pt")
    target = ExportTarget.resolve(cat, "demo/mill", run_id="r1")
    with pytest.raises(ExportRefused, match="not WGS84 Site ENU"):
        export_run.export_run(target, ["las"], ExportOptions(), log=lambda m: None)


def test_export_writes_registers_and_keeps_the_served_run(cat):
    import laspy
    import rasterio

    from aerosplat.export.raster import read_tags
    from aerosplat.export.splat_ply import read_inria_ply

    site = _site(cat)
    before = cat.get_run(site, "baseline", "r1")
    target = ExportTarget.resolve(cat, "demo/mill")              # the served run
    assert target.run.id == "r1"
    events = []
    manifest = export_run.export_run(
        target, ["dsm", "dtm", "ndsm", "las", "ply"], ExportOptions(),
        on_event=lambda k, s, d: events.append((k, s)), log=lambda m: None)
    after = cat.get_run(site, "baseline", "r1")
    assert after.finished_at == before.finished_at and after.status == RunStatus.ok
    for k in ("dsm", "dtm", "ndsm", "las", "ply"):
        assert f"export_{k}" in after.products and k in manifest["files"]
        assert (k, "done") in events
    assert after.products["exports"] == "exports/exports.json"
    assert cat.latest_run(site).id == "r1"

    budget = manifest["georeference"]
    assert budget["frame_error_m"] == 0.00022 and budget["earth_model"] == "sphere-enu+affine"
    assert "unverified" in budget["altitude_reference"]
    out = cat.run_dirs(site, "baseline", "r1").root / "exports"
    assert json.loads((out / "exports.json").read_text(encoding="utf-8")) == manifest

    tags = read_tags(out / "dsm.tif")
    assert tags["AEROSPLAT_GEOREFERENCE"]["frame_error_m"] == 0.00022
    assert tags["AEROSPLAT_GEOREFERENCE"]["crs"]["epsg"] == 32618
    with rasterio.open(out / "ndsm.tif") as src:
        nd = src.read(1)
    assert nd.max() == pytest.approx(9.0, abs=0.05)                  # the block on the ground

    las = laspy.read(str(out / "cloud.laz"))
    assert len(las.points) == 80 * 120
    # the cloud point at ENU (0, 0, 0) is at the Site origin
    i = int(np.argmin(np.abs(np.asarray(las.x) - np.median(las.x))
                      + np.abs(np.asarray(las.y) - np.median(las.y))))
    from pyproj import Transformer
    lon, lat = Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True).transform(
        float(las.x[i]), float(las.y[i]))
    enu = EnuFrame(ORIGIN).from_geodetic(lat, lon, float(las.z[i]))
    assert abs(enu[0] - round(enu[0] * 2) / 2) < 0.002            # on the 0.5 m lattice
    assert abs(enu[1] - round(enu[1] * 2) / 2) < 0.002

    arrays, comments = read_inria_ply(out / "splat.ply")
    assert len(arrays["xyz"]) == 80 * 120 and arrays["f_rest"].shape[1] == 9
    assert any("origin_wgs84 lat 39.76003596749297" in c for c in comments)
    assert manifest["files"]["ply"]["details"]["rasterize_mode"] == "antialiased"


def test_ortho_through_a_renderer_factory(cat):
    """The orchestrator sizes the mosaic from the cloud and pins the renderer to the run."""
    import rasterio

    from test_export import FakeRenderer

    seen = {}

    def factory(target, splat):
        seen["splat"] = splat.name
        seen["run"] = target.run.id
        return FakeRenderer(run=target.run.id)

    target = ExportTarget.resolve(cat, "demo/mill", run_id="r1")
    manifest = export_run.export_run(target, ["ortho"], ExportOptions(gsd_m=0.5, tile_px=64),
                                     renderer_factory=factory, log=lambda m: None)
    assert seen == {"splat": "splat_wgs84.pt", "run": "r1"}
    d = manifest["files"]["ortho"]["details"]
    assert d["tiles"] >= 2 and d["registration_px_max"] < 0.01
    path = cat.run_dirs(_site(cat), "baseline", "r1").root / "exports" / "ortho.tif"
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 32618
        b = src.bounds
    assert (b.right - b.left) == pytest.approx(60.0, abs=1.5)        # the cloud is 60 x 40 m


def test_missing_semantics_explains_what_to_run(cat):
    site = _site(cat)
    _make_run(cat, site, "nosem", semantics=False)
    target = ExportTarget.resolve(cat, "demo/mill", run_id="nosem")
    with pytest.raises(ExportRefused, match="--from semantics"):
        export_run.export_run(target, ["dsm"], ExportOptions(), log=lambda m: None)


def test_concurrent_export_of_the_same_run_is_refused(cat):
    from aerosplat.store.locking import file_lock

    target = ExportTarget.resolve(cat, "demo/mill")
    out = target.run_dir / "exports"
    out.mkdir()
    old = export_run.LOCK_TIMEOUT_S
    export_run.LOCK_TIMEOUT_S = 0.2
    try:
        with file_lock(out / ".export.lock"):
            with pytest.raises(ExportRefused, match="another export"):
                export_run.export_run(target, ["ply"], ExportOptions(), log=lambda m: None)
    finally:
        export_run.LOCK_TIMEOUT_S = old


# ── job and API ──────────────────────────────────────────────────────────
def test_export_spec_resource_and_argv():
    from aerosplat.jobs.models import ExportSpec, Resource

    cpu = ExportSpec(site="demo/mill", epoch="baseline", run_id="r1", kinds=["dsm", "las"])
    assert cpu.resource == Resource.cpu
    gpu = ExportSpec(site="demo/mill", epoch="baseline", run_id="r1", kinds=["ortho"], laz=False)
    assert gpu.resource == Resource.gpu
    a = gpu.argv("/p")
    assert a[:6] == ["--site", "demo/mill", "--epoch", "baseline", "--run", "r1"]
    assert "--las" in a and a[-2:] == ["--projects-root", "/p"]
    args = export_run.build_parser().parse_args(cpu.argv())
    assert args.kinds == "dsm,las" and args.crs == "utm"
    with pytest.raises(ValueError):
        ExportSpec(site="s", epoch="e", run_id="r", kinds=["tiff"])
    with pytest.raises(ValueError):
        ExportSpec(site="s", epoch="e", run_id="r", crs="EPSG:32618; rm -rf")


def test_job_child_runs_the_export(cat):
    from aerosplat.jobs import child
    from aerosplat.jobs.models import EventType, ExportSpec, JobKind, Resource
    from aerosplat.jobs.queue import JobQueue

    q = JobQueue(cat.engine)
    spec = ExportSpec(site="demo/mill", epoch="baseline", run_id="r1", kinds=["ply"])
    job = q.enqueue(JobKind.export, spec.model_dump(), resource=Resource.cpu, site_key="demo/mill")
    assert child.main(["--projects-root", str(cat.paths.root), "--job", job.id]) == 0
    stages = [e.data for e in q.events(job.id) if e.type == EventType.stage]
    assert [s["state"] for s in stages] == ["started", "done"]
    assert q.get(job.id).result["run_id"] == "r1"


def test_api_refuses_queues_lists_and_downloads(cat):
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from aerosplat.server import api_export, api_v2

    app = FastAPI()
    app.include_router(api_v2.router)
    app.include_router(api_export.router)
    client = TestClient(app)
    site = _site(cat)
    _make_run(cat, site, "synthetic", georeferenced=False, earth_model="wgs84")
    base = "/api/v2/sites/demo/mill/runs/baseline"

    r = client.post(f"{base}/synthetic/exports", json={"kinds": ["las"]})
    assert r.status_code == 409 and "no georeference" in r.json()["detail"]
    listing = client.get(f"{base}/synthetic/exports").json()
    assert listing["exportable"] is False and "no georeference" in listing["refusal"]

    r = client.post(f"{base}/r1/exports", json={"kinds": ["las"], "crs": "EPSG:4326"})
    assert r.status_code == 422 and "not a projected" in r.json()["detail"]
    r = client.post(f"{base}/r1/exports", json={"kinds": ["ortho", "dsm"]})
    assert r.status_code == 202
    job = r.json()
    assert job["kind"] == "export" and job["resource"] == "gpu"
    assert client.post(f"{base}/r1/exports", json={"kinds": ["dsm"]}).json()["resource"] == "cpu"

    target = ExportTarget.resolve(cat, "demo/mill", run_id="r1")
    export_run.export_run(target, ["ply"], ExportOptions(), log=lambda m: None)
    listing = client.get(f"{base}/r1/exports").json()
    assert listing["exportable"] is True
    assert [f["kind"] for f in listing["files"]] == ["ply"]
    assert listing["georeference"]["frame_error_m"] == 0.00022
    d = client.get(listing["files"][0]["url"])
    assert d.status_code == 200 and d.content.startswith(b"ply\n")
    assert "mill_r1_splat.ply" in d.headers["content-disposition"]
    assert client.get(f"{base}/r1/exports/dsm").status_code == 404
    assert client.get(f"{base}/r1/exports/..%2Fmanifest.json").status_code == 404
    assert client.get(f"{base}/r1/exports/manifest").json()["run"] == "r1"


def test_cli_with_a_relative_projects_root_registers_the_files(cat, tmp_path, monkeypatch):
    """Regression: relative product paths are read relative to the run directory, so a
    relative --projects-root once wrote the files and silently registered none."""
    monkeypatch.chdir(tmp_path)
    assert export_run.main(["--site", "demo/mill", "--kinds", "ply",
                            "--projects-root", "projects"]) == 0
    run = cat.get_run(_site(cat), "baseline", "r1")
    assert run.products["export_ply"] == "exports/splat.ply"
