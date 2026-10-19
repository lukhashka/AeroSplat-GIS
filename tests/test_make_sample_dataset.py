"""tools/make_sample_dataset.py: frames and camera frames stay in step; the source is untouched."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

TOOL = Path(__file__).resolve().parents[1] / "tools" / "make_sample_dataset.py"
spec = importlib.util.spec_from_file_location("make_sample_dataset", TOOL)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fake_dataset(root: Path, n: int = 7) -> Path:
    src = root / "ds"
    (src / "footage").mkdir(parents=True)
    for i in range(n):
        Image.new("RGB", (40, 20), (i * 30, 0, 0)).save(src / "footage" / f"ds_{i:02d}.jpeg")
    frames = [{"position": {"x": i}, "fovVertical": 20} for i in range(n)]
    (src / "ds.json").write_text(json.dumps({"name": "ds", "width": 40, "height": 20,
                                             "numFrames": n - 1, "cameraFrames": frames,
                                             "trackPoints": []}), encoding="utf-8")
    return src


def test_subset_keeps_frames_and_json_in_step(tmp_path):
    src = fake_dataset(tmp_path)
    before = sorted(p.name for p in (src / "footage").iterdir())
    dst = mod.make(src, "sample", every=2, max_frames=3, scale=0.5, out_root=tmp_path / "out")
    data = json.loads((dst / "sample.json").read_text(encoding="utf-8"))
    assert [f["position"]["x"] for f in data["cameraFrames"]] == [0, 2, 4]
    assert (data["width"], data["height"], data["name"]) == (20, 10, "sample")
    assert data["cameraFrames"][0]["fovVertical"] == 20          # intrinsics follow the scale
    out = sorted(p.name for p in (dst / "footage").iterdir())
    assert out == ["sample_00.jpeg", "sample_01.jpeg", "sample_02.jpeg"]
    assert Image.open(dst / "footage" / "sample_00.jpeg").size == (20, 10)
    assert sorted(p.name for p in (src / "footage").iterdir()) == before


def test_never_overwrites_and_refuses_mismatch(tmp_path):
    src = fake_dataset(tmp_path)
    mod.make(src, "s", 2, 3, 1.0, tmp_path / "out")
    with pytest.raises(SystemExit, match="exists"):
        mod.make(src, "s", 2, 3, 1.0, tmp_path / "out")
    (src / "footage" / "ds_06.jpeg").unlink()
    with pytest.raises(SystemExit, match="camera frames"):
        mod.make(src, "t", 2, 3, 1.0, tmp_path / "out")
