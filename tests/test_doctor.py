"""doctor: intrinsics of an Earth Studio dataset are checked against its JSON, not RunConfig."""

from __future__ import annotations

import json

from aerosplat import doctor


def _es(tmp_path, w, h):
    js = tmp_path / "ds.json"
    js.write_text(json.dumps({"width": w, "height": h,
                              "cameraFrames": [{"fovVertical": 20}, {"fovVertical": 20}]}))
    return js


def test_earthstudio_frames_matching_the_json_are_fine(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "_problems", [])
    # RunConfig's synthetic defaults (1920×1080, 60°) do not apply to earthstudio_json.
    doctor.check_intrinsics({"width": 960, "height": 540}, 1920, 1080, 60.0, _es(tmp_path, 960, 540))
    assert doctor._problems == []


def test_earthstudio_frames_not_matching_the_json_are_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "_problems", [])
    doctor.check_intrinsics({"width": 960, "height": 540}, 960, 540, 60.0, _es(tmp_path, 1920, 1080))
    assert any("НЕЗБІГ РОЗДІЛЬНОСТІ" in p for p in doctor._problems)


def test_without_json_the_config_is_checked(monkeypatch):
    monkeypatch.setattr(doctor, "_problems", [])
    doctor.check_intrinsics({"width": 960, "height": 540}, 1920, 1080, 60.0)
    assert any("НЕЗБІГ РОЗДІЛЬНОСТІ" in p for p in doctor._problems)
