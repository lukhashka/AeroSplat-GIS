import json

import pytest
from pydantic import ValidationError

from aerosplat.config import RunConfig, build_config


def test_defaults_are_safe():
    cfg = RunConfig()
    assert cfg.camera_source == "auto"
    assert cfg.known_intrinsics is False
    assert cfg.recenter is False


def test_unknown_key_rejected():
    with pytest.raises(ValidationError):
        RunConfig(itterations=10)


@pytest.mark.parametrize("bad", [
    {"iterations": 0}, {"matcher": "bruteforce"}, {"lat": 123.0},
    {"init_headroom": 0}, {"warmup_iters": 10_000, "iterations": 100},
])
def test_invalid_values_rejected(bad):
    with pytest.raises(ValidationError):
        RunConfig(**bad)


def test_file_then_overrides(tmp_path):
    f = tmp_path / "run.yaml"
    f.write_text("dataset: foo\niterations: 100\nmatcher: spatial\n", encoding="utf-8")
    cfg = build_config(f, {"iterations": 200, "rasterizer": None})
    assert (cfg.dataset, cfg.iterations, cfg.matcher) == ("foo", 200, "spatial")
    assert cfg.rasterizer == "auto"  # None override ignored


def test_json_file(tmp_path):
    f = tmp_path / "run.json"
    f.write_text(json.dumps({"dataset": "bar"}), encoding="utf-8")
    assert build_config(f).dataset == "bar"
