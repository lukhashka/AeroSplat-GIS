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
