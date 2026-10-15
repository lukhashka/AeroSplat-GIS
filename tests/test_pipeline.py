import json

import pytest

from aerosplat.config import RunConfig
from aerosplat.pipeline import runner
from aerosplat.pipeline.layout import RunLayout
from aerosplat.pipeline.manifest import Manifest
from aerosplat.pipeline.stages import STAGES


@pytest.fixture
def layout(tmp_path):
    cfg = RunConfig(dataset="t", camera_source="synthetic").to_dict()
    return cfg, RunLayout.from_config(cfg, data_dir=tmp_path / "data", project_dir=tmp_path)


def _fake_stage(monkeypatch, calls, fail=None):
    for name, st in STAGES.items():
        def entry(cfg, name=name):
            def main(argv):
                calls.append(name)
                if name == fail:
                    raise ValueError("boom")
                return 0
            return main
        monkeypatch.setattr(type(st), "entry", lambda self, cfg, e=entry: e(cfg, self.name),
                            raising=False)


def test_layout_paths(layout):
    _, L = layout
    assert L.geotagged.name == "raw_images_t"
    assert L.manifest_path == L.data_dir / "t" / "run" / "manifest.json"
    assert L.cloud_in == L.dense_pt


def test_manifest_records_and_resume(layout, monkeypatch):
    cfg, L = layout
    calls: list[str] = []
    _fake_stage(monkeypatch, calls)
    L.sparse.mkdir(parents=True)
    L.fused_ply.parent.mkdir(parents=True)
    L.fused_ply.write_bytes(b"x")

    runner.run_stages(cfg, L, ["sfm", "dense"])
    assert calls == ["sfm", "dense"]
    data = json.loads(L.manifest_path.read_text(encoding="utf-8"))
    assert data["stages"]["sfm"]["status"] == "ok" and data["config"]["dataset"] == "t"

    runner.run_stages(cfg, L, ["sfm", "dense"], resume=True)
    assert calls == ["sfm", "dense"]  # both skipped

    runner.run_stages(dict(cfg, matcher="exhaustive"), L, ["sfm", "dense"], resume=True)
    assert calls == ["sfm", "dense", "sfm", "dense"]  # sfm changed → dense re-runs too


def test_failure_keeps_traceback_and_is_recorded(layout, monkeypatch):
    cfg, L = layout
    calls: list[str] = []
    _fake_stage(monkeypatch, calls, fail="sfm")
    with pytest.raises(ValueError, match="boom"):
        runner.run_stages(cfg, L, ["sfm"])
    rec = Manifest.load(L.manifest_path).data["stages"]["sfm"]
    assert rec["status"] == "failed" and "boom" in rec["error"]


def test_skipped_stage_not_run(layout, monkeypatch):
    cfg, L = layout
    calls: list[str] = []
    _fake_stage(monkeypatch, calls)
    runner.run_stages(dict(cfg, camera_source="earthstudio_json"), L, ["exif"])
    assert calls == []


@pytest.mark.parametrize("cloud_source", ["dense", "sparse"])
def test_every_stage_entry_resolves(cloud_source):
    """Real (unmocked) lookup: each stage's module exists and exposes main(argv)."""
    cfg = RunConfig(cloud_source=cloud_source).to_dict()
    for stage in STAGES.values():
        assert callable(stage.entry(cfg)), stage.name
        assert stage.script(cfg).is_file(), stage.name


# ── Site runs (ADR-0006): RunLayout as a view on the project tree ─────────
from conftest import write_es_export  # noqa: E402

from aerosplat import starter  # noqa: E402
from aerosplat.geo.crs import GeoPoint  # noqa: E402
from aerosplat.ingest.earthstudio import import_earthstudio  # noqa: E402
from aerosplat.pipeline.stages import argv_dense, argv_eval, argv_geo  # noqa: E402
from aerosplat.store import Catalog, RunStatus, Site  # noqa: E402

SITE_ORIGIN = GeoPoint(lat=40.8942, lon=-74.2173, h=233.0)


@pytest.fixture
def es_site(tmp_path):
    """Catalog with site demo/es holding one Earth Studio capture."""
    root = tmp_path / "projects"
    cat = Catalog.open(root)
    cat.ensure_project("demo")
    site = cat.create_site(Site(id="es", project_id="demo", origin=SITE_ORIGIN))
    js, footage = write_es_export(tmp_path / "export", "fixture")
    cap = import_earthstudio(cat, site, js, footage)
    yield cat, site, cap, root
    cat.close()


def test_site_layout_is_a_view_on_the_run_dir(es_site):
    cat, site, cap, _ = es_site
    cfg = RunConfig(camera_source="earthstudio_json").to_dict()
    L = RunLayout.for_site_run(cfg, site=site, site_dirs=cat.site_dirs(site), capture=cap,
                               epoch_id="baseline", run_id="r1")
    run = cat.run_dirs(site, "baseline", "r1")
    assert L.run_dir == run.root and L.is_site_run and L.origin == SITE_ORIGIN
    for p in (L.geotagged, L.colmap_out, L.dense, L.splat_pt, L.cloud_in, L.known_poses_dir,
              L.manifest_path, L.metrics_json, L.registration_report, L.log_path):
        assert run.root in p.parents, p
    assert L.known_poses_dir.parent == run.root          # outside sfm/ (wiped by --fresh)
    assert L.footage == cat.site_dirs(site).capture(cap.id).images
    assert L.json_path.is_file() and L.frame_prefix == "fixture"

    geo = argv_geo(cfg, L)
    assert geo[geo.index("--origin") + 1] == "40.8942,-74.2173,233.0"
    assert geo[geo.index("--report") + 1] == str(L.registration_report)
    dense = argv_dense(cfg, L)
    assert dense[dense.index("--dense-dir") + 1] == str(run.mvs)
    ev = argv_eval(cfg, L)
    assert ev[ev.index("--out-dir") + 1] == str(run.reports)


def test_legacy_layout_has_no_origin(layout):
    _, L = layout
    assert L.origin is None and not L.is_site_run
    assert "--origin" not in argv_geo(*layout)


def test_site_run_geo_stage_end_to_end(es_site):
    cat, site, cap, root = es_site
    rc = starter.main(["--site", "demo/es", "--only", "geo", "--run-id", "r1",
                       "--projects-root", str(root)])
    assert rc == 0
    run = cat.get_run(site, "baseline", "r1")
    assert run.status == RunStatus.ok and run.georeferenced and run.capture_ids == [cap.id]
    assert {"known_poses", "registration_report"} <= set(run.products)
    assert "splat" not in run.products
    rd = cat.run_dirs(site, "baseline", "r1")
    geo = json.loads((rd.known_poses / "geo_registration.json").read_text())
    assert geo["enu_origin"] == {"latitude": SITE_ORIGIN.lat, "longitude": SITE_ORIGIN.lon,
                                 "altitude": SITE_ORIGIN.h}
    assert len(list(rd.images.glob("fixture_*.jpeg"))) == 3
    manifest = json.loads(rd.manifest.read_text(encoding="utf-8"))
    assert manifest["site"] == "demo/es" and manifest["run_id"] == "r1"
    assert manifest["stages"]["geo"]["status"] == "ok"
    assert cat.get_site("demo", "es").origin == SITE_ORIGIN

    # --resume alone continues the latest run of the epoch
    assert starter.main(["--site", "es", "--only", "geo", "--resume",
                         "--projects-root", str(root)]) == 0
    assert [r.id for r in cat.list_runs(site)] == ["r1"]


def test_site_dry_run_writes_nothing(es_site, capsys):
    cat, site, _, root = es_site
    assert starter.main(["--site", "demo/es", "--dry-run", "--only", "geo",
                         "--projects-root", str(root)]) == 0
    assert "--origin 40.8942,-74.2173,233.0" in capsys.readouterr().out
    assert cat.list_runs(site) == [] and cat.list_epochs(site) == []


def test_site_and_dataset_are_exclusive(es_site):
    *_, root = es_site
    with pytest.raises(SystemExit):
        starter.main(["--site", "demo/es", "--dataset", "x", "--dry-run",
                      "--projects-root", str(root)])
    with pytest.raises(SystemExit):
        starter.main(["--site", "demo/missing", "--dry-run", "--projects-root", str(root)])
