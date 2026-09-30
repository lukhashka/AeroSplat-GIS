import pytest

import starter


def _args(argv):
    return starter.build_parser().parse_args(argv)


def test_config_defaults():
    assert starter.CONFIG["camera_source"] == "auto"


def test_resolve_stages_ranges():
    assert starter.resolve_stages(_args(["--from", "dense"])) == ["dense", "parse", "train"]
    assert starter.resolve_stages(_args(["--only", "sfm"])) == ["sfm"]
    assert starter.resolve_stages(_args(["--stages", "geo,sfm"])) == ["geo", "sfm"]
    assert "exif" not in starter.resolve_stages(_args(["--skip", "exif"]))


def test_resolve_stages_rejects_unknown():
    with pytest.raises(SystemExit):
        starter.resolve_stages(_args(["--stages", "geo,bogus"]))


def test_set_override_and_none():
    cfg = starter.resolve_config(_args(["--set", "iterations=10", "--set", "init_scale=none"]))
    assert cfg["iterations"] == 10 and cfg["init_scale"] is None


def test_set_unknown_key_and_bad_value():
    with pytest.raises(SystemExit):
        starter.resolve_config(_args(["--set", "nope=1"]))
    with pytest.raises(SystemExit):
        starter.resolve_config(_args(["--set", "iterations=abc"]))


def test_recenter_with_train_forbidden():
    with pytest.raises(SystemExit):
        starter.main(["--dry-run", "--set", "recenter=true", "--only", "train"])


def test_dry_run_builds_train_command(capsys):
    assert starter.main(["--dry-run", "--only", "train", "--camera-source", "synthetic"]) == 0
    out = capsys.readouterr().out
    assert "gs_trainer_v3.py" in out and "--colmap" in out
