import dataclasses
import json
import sys

import pytest
import torch

from aerosplat.config import RunConfig
from aerosplat.geo.sphere_enu import EARTH_MODEL
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
    assert L.cloud_wgs84 == L.data_dir / "t_dense_wgs84.pt"
    assert L.splat_wgs84 == L.data_dir / "t_splat_wgs84.pt"
    assert L.footage == L.data_dir / "t" / "footage"       # the dataset's own frames, not v6's


def test_explicit_footage_dir_wins(tmp_path):
    cfg = RunConfig(dataset="t", camera_source="synthetic", footage_dir="elsewhere/f").to_dict()
    L = RunLayout.from_config(cfg, data_dir=tmp_path / "data", project_dir=tmp_path)
    assert L.footage == tmp_path / "elsewhere" / "f"


def test_georef_skipped_without_known_poses(layout, monkeypatch):
    cfg, L = layout
    calls: list[str] = []
    _fake_stage(monkeypatch, calls)
    runner.run_stages(cfg, L, ["georef"])                                  # synthetic
    runner.run_stages(dict(cfg, camera_source="earthstudio_json", known_poses=False), L,
                      ["georef"])
    assert calls == []


def test_geo_fingerprint_carries_its_convention(layout):
    """Known poses written before sphere ENU must not be reused by --resume."""
    cfg, L = layout
    geo = STAGES["geo"]
    assert geo.version == EARTH_MODEL
    assert runner._fingerprint(geo, cfg, L) != runner._fingerprint(
        dataclasses.replace(geo, version=""), cfg, L)


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


# ── rasterizer=gsplat is checked before the first stage, not when training starts ──
@pytest.fixture
def no_gsplat_toolchain(monkeypatch):
    from aerosplat.gs import gsplat_backend

    asked: list[int] = []

    def problems():
        asked.append(1)
        return ["cl.exe (MSVC) не в PATH"]

    monkeypatch.setattr(gsplat_backend, "gsplat_problems", problems)
    return asked


def test_gsplat_run_without_toolchain_is_refused_in_preflight(layout, no_gsplat_toolchain):
    cfg, L = layout
    cfg = dict(cfg, rasterizer="gsplat")
    with pytest.raises(SystemExit) as e:
        runner.preflight(cfg, L, ["sfm", "dense", "parse", "train"])
    msg = str(e.value.code)
    assert "'train'" in msg and "cl.exe (MSVC) не в PATH" in msg and "rasterizer=auto" in msg
    assert not L.manifest_path.exists()                  # nothing ran, nothing was recorded


def test_gsplat_dry_run_only_warns(layout, no_gsplat_toolchain):
    cfg, L = layout
    runner.preflight(dict(cfg, rasterizer="gsplat"), L, ["train"], dry_run=True)
    assert no_gsplat_toolchain                           # it was checked


def test_gsplat_check_is_skipped_when_nothing_renders_with_gsplat(layout, no_gsplat_toolchain):
    cfg, L = layout
    for raster in ("auto", "cuda", "torch"):
        assert runner.check_rasterizer(dict(cfg, rasterizer=raster), L, ["train"]) is None
    gs = dict(cfg, rasterizer="gsplat")
    assert runner.check_rasterizer(gs, L, ["sfm", "dense", "parse"]) is None
    assert runner.check_rasterizer(dict(gs, eval_every=0), L, ["eval"]) is None   # eval skipped
    assert no_gsplat_toolchain == []                     # never even asked
    assert runner.check_rasterizer(gs, L, ["eval"]) is not None


def test_gsplat_check_does_not_refuse_a_resume_whose_training_is_done(layout, no_gsplat_toolchain,
                                                                      monkeypatch):
    cfg, L = layout
    cfg = dict(cfg, rasterizer="gsplat", eval_every=0)
    calls: list[str] = []
    _fake_stage(monkeypatch, calls)
    L.splat_pt.parent.mkdir(parents=True, exist_ok=True)
    L.splat_pt.write_bytes(b"x")
    runner.run_stages(cfg, L, ["train"])
    assert runner.stages_to_run(cfg, L, ["train", "georef"], resume=True) == []
    assert runner.check_rasterizer(cfg, L, ["train"], resume=True) is None
    assert runner.check_rasterizer(cfg, L, ["train"], resume=False) is not None
    # A changed training option makes train run again: then the check applies.
    assert runner.check_rasterizer(dict(cfg, iterations=10), L, ["train"], resume=True) is not None


def test_gsplat_problems_names_each_missing_piece(monkeypatch):
    from aerosplat.gs import gsplat_backend as gb

    monkeypatch.setitem(sys.modules, "gsplat", None)
    assert gb.gsplat_problems() == [gb.GSPLAT_MISSING]
    with pytest.raises(ImportError, match="pip install gsplat"):
        gb.require_gsplat()

    fake = type(sys)("gsplat")
    monkeypatch.setitem(sys.modules, "gsplat", fake)
    monkeypatch.setattr(gb.torch.cuda, "is_available", lambda: False)
    assert "CUDA" in gb.gsplat_problems()[0]

    monkeypatch.setattr(gb.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gb, "build_env_problems", lambda: ["ninja не знайдено в PATH"])
    (p,) = gb.gsplat_problems()
    assert "ninja не знайдено" in p and "gsplat_env.bat" in p and "aerosplat up" in p
    with pytest.raises(RuntimeError, match="ninja"):
        gb.require_gsplat()

    monkeypatch.setattr(gb, "build_env_problems", lambda: [])
    assert gb.gsplat_problems() == []                    # the toolchain is right: JIT will work
    monkeypatch.setattr(gb, "build_env_problems", lambda: ["x"])
    csrc = type(sys)("gsplat.csrc")
    fake.csrc = csrc
    monkeypatch.setitem(sys.modules, "gsplat.csrc", csrc)
    assert gb.gsplat_problems() == []                    # prebuilt: no toolchain needed


def test_stage_failure_reason_reaches_the_run_log(layout, monkeypatch):
    cfg, L = layout
    calls: list[str] = []
    _fake_stage(monkeypatch, calls, fail="sfm")
    with pytest.raises(ValueError):
        runner.run_stages(cfg, L, ["sfm"])
    assert "впала після" in L.log_path.read_text(encoding="utf-8")
    assert "ValueError: boom" in L.log_path.read_text(encoding="utf-8")


# ── Site runs (ADR-0006): RunLayout as a view on the project tree ─────────
from conftest import ES_ORIGIN as SITE_ORIGIN  # noqa: E402
from conftest import write_pt  # noqa: E402

from aerosplat import starter  # noqa: E402
from aerosplat.pipeline.stages import argv_dense, argv_eval, argv_geo  # noqa: E402
from aerosplat.store import RunStatus  # noqa: E402
from aerosplat.store.scenes import (CLOUD, CLOUD_RECON, SPLAT, SPLAT_RECON,  # noqa: E402
                                    catalog_scenes)


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
    assert run.earth_model == EARTH_MODEL
    assert {"known_poses", "registration_report"} <= set(run.products)
    assert "splat" not in run.products
    rd = cat.run_dirs(site, "baseline", "r1")
    geo = json.loads((rd.known_poses / "geo_registration.json").read_text())
    assert geo["enu_origin"] == {"latitude": SITE_ORIGIN.lat, "longitude": SITE_ORIGIN.lon,
                                 "altitude": SITE_ORIGIN.h}
    assert geo["earth_model"] == EARTH_MODEL
    assert len(list(rd.images.glob("fixture_*.jpeg"))) == 3
    manifest = json.loads(rd.manifest.read_text(encoding="utf-8"))
    assert manifest["site"] == "demo/es" and manifest["run_id"] == "r1"
    assert manifest["stages"]["geo"]["status"] == "ok"
    assert cat.get_site("demo", "es").origin == SITE_ORIGIN

    # --resume alone continues the latest run of the epoch
    assert starter.main(["--site", "es", "--only", "geo", "--resume",
                         "--projects-root", str(root)]) == 0
    assert [r.id for r in cat.list_runs(site)] == ["r1"]


def test_site_run_registers_and_serves_wgs84_products(es_site):
    """The catalog's splat/cloud of an Earth Studio run are the georef stage's WGS84
    Site ENU files; the sphere-ENU originals stay as *_recon and are never served."""
    cat, site, _, root = es_site
    args = ["--site", "demo/es", "--run-id", "r1", "--projects-root", str(root)]
    assert starter.main([*args, "--only", "geo"]) == 0
    rd = cat.run_dirs(site, "baseline", "r1")
    write_pt(rd.gs / "dense.pt", "dense", 50)          # as if dense/parse/train had run
    write_pt(rd.gs / "splat.pt", "splat", 50)
    assert starter.main([*args, "--only", "georef"]) == 0

    run = cat.get_run(site, "baseline", "r1")
    assert run.products[SPLAT] == "gs/splat_wgs84.pt"
    assert run.products[CLOUD] == "gs/dense_wgs84.pt"
    assert run.products[SPLAT_RECON] == "gs/splat.pt"
    assert run.products[CLOUD_RECON] == "gs/dense.pt"
    reg = json.loads(rd.registration_report.read_text(encoding="utf-8"))
    A = torch.tensor(reg["sphere_enu_to_site_enu"]["A"], dtype=torch.float64)
    b = torch.tensor(reg["sphere_enu_to_site_enu"]["b"], dtype=torch.float64)
    src = torch.load(rd.gs / "splat.pt")["xyz"].double()
    out = torch.load(rd.gs / "splat_wgs84.pt")["xyz"].double()
    assert torch.allclose(out, src @ A.T + b, atol=1e-4)

    scene = catalog_scenes(cat)["demo/es"]
    assert scene.splat.endswith("splat_wgs84.pt") and scene.cloud.endswith("dense_wgs84.pt")
    assert scene.metric["earth_model"] == EARTH_MODEL
    assert 0 < scene.metric["frame_error_m"] < 1e-3          # affine residual, sub-mm


def test_site_run_refuses_to_resume_other_earth_model(es_site):
    cat, site, cap, root = es_site
    cat.ensure_epoch(site, "baseline")
    cat.create_run(site, "baseline", "old", capture_ids=[cap.id], earth_model="wgs84",
                   georeferenced=True)
    with pytest.raises(SystemExit, match="earth_model=wgs84"):
        starter.main(["--site", "demo/es", "--run-id", "old", "--only", "geo",
                      "--projects-root", str(root)])


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


# ── doctor --site: resolved read-only through the catalog ────────────────
def test_doctor_site_without_a_run_says_so(es_site):
    from aerosplat import doctor

    cat, site, cap, root = es_site
    with pytest.raises(SystemExit, match="baseline"):          # no epoch, so no run
        doctor.main(["--site", site.key, "--projects-root", str(root), "--stage", "sparse"])
    assert cat.list_runs(site) == []                       # nothing was created


def test_doctor_site_and_dataset_are_exclusive(es_site):
    from aerosplat import doctor

    _, site, _, root = es_site
    with pytest.raises(SystemExit, match="взаємовиключні"):
        doctor.main(["--site", site.key, "--dataset", "x", "--projects-root", str(root)])
