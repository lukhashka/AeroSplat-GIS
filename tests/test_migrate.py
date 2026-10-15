"""Legacy data/<dataset> -> Site migration (aerosplat.store.legacy, tools/migrate_dataset.py)."""

import json

import pytest

from conftest import ROOT, make_legacy_dataset
from aerosplat.geo.crs import GeoPoint
from aerosplat.ingest.images import sha256_file
from aerosplat.store import RunStatus
from aerosplat.store import legacy
from aerosplat.store.legacy import LegacyDataset, migrate_dataset
from aerosplat.store.scenes import discover_scenes


def _quiet(_msg: str) -> None:
    pass


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    make_legacy_dataset(d, "fixture")
    return d


def test_copy_migration_builds_site_capture_and_run(catalog, data_dir):
    res = migrate_dataset(catalog, data_dir, "fixture", log=_quiet)
    site = catalog.get_site("default", "fixture")
    legacy_origin = LegacyDataset(data_dir, "fixture").legacy_enu_origin()
    assert site.origin == legacy_origin            # legacy run stays in the Site frame

    (cap,) = catalog.list_captures(site)
    assert cap.id == res.capture_id == "earthstudio-fixture"
    assert cap.frame_prefix == "fixture" and "Vexcel" in cap.attribution
    cdirs = catalog.site_dirs(site).capture(cap.id)
    imgs = catalog.list_images(site, cap.id)
    assert [i.name for i in imgs] == [f"fixture_{i:02d}.jpeg" for i in range(3)]
    assert all(i.sha256 == sha256_file(cdirs.images / i.name) for i in imgs)
    assert imgs[0].prior is not None and imgs[0].width == 16
    assert (cdirs.source / "fixture.json").is_file() and (cdirs.source / "fixture.esp").is_file()
    assert not list(catalog.paths.root.rglob("*.xlsx"))
    assert "private.xlsx" in res.left_behind

    assert res.run_id == "20260930-215021-legacy"
    run = catalog.get_run(site, "baseline", res.run_id)
    assert run.status == RunStatus.ok and run.georeferenced
    assert run.earth_model == "sphere-legacy" and run.git_commit == "abc1234"
    assert {"splat", "cloud", "sparse", "fused_ply", "registration_report"} <= set(run.products)
    rd = catalog.run_dirs(site, "baseline", run.id)
    for p in (rd.gs / "splat.pt", rd.gs / "dense.pt", rd.mvs / "fused.ply", rd.sfm / "database.db",
              rd.sfm / "sparse" / "0" / "cameras.bin", rd.known_poses / "geo_registration.json",
              rd.images / "fixture_00.jpeg", rd.reports / "metrics.json", rd.log):
        assert p.is_file(), p
    manifest = json.loads(rd.manifest.read_text(encoding="utf-8"))
    assert manifest["migrated_from"]["layout"] == "data/fixture"
    assert manifest["site"] == "default/fixture" and manifest["run_id"] == run.id
    reg = json.loads(rd.registration_report.read_text(encoding="utf-8"))
    assert reg["run_frame"]["earth_model"] == "sphere-legacy"
    assert 0 < reg["legacy_shift"]["max_m"] < 3

    # copy mode leaves the legacy layout usable for `aerosplat run --dataset`
    assert (data_dir / "fixture_splat.pt").is_file()
    assert (data_dir / "fixture" / "footage" / "fixture_00.jpeg").is_file()


def test_migration_is_idempotent(catalog, data_dir):
    first = migrate_dataset(catalog, data_dir, "fixture", log=_quiet)
    again = migrate_dataset(catalog, data_dir, "fixture", log=_quiet)
    assert again.run_id == first.run_id
    assert any("exists" in a for a in again.actions)
    site = catalog.get_site("default", "fixture")
    assert len(catalog.list_captures(site)) == 1 and len(catalog.list_runs(site)) == 1


def test_dry_run_changes_nothing(catalog, data_dir):
    res = migrate_dataset(catalog, data_dir, "fixture", dry_run=True, log=_quiet)
    assert res.site.key == "default/fixture" and res.run_id
    assert catalog.list_sites() == []
    assert not (catalog.paths.root / "default").exists()


def test_move_empties_legacy_layout(catalog, data_dir):
    res = migrate_dataset(catalog, data_dir, "fixture", move=True, log=_quiet)
    site = catalog.get_site("default", "fixture")
    rd = catalog.run_dirs(site, "baseline", res.run_id)
    assert (rd.gs / "splat.pt").is_file() and not (data_dir / "fixture_splat.pt").exists()
    assert not (data_dir / "colmap_output_fixture").exists()
    assert not list((data_dir / "fixture" / "footage").glob("*.jpeg"))
    assert (data_dir / "fixture" / "private.xlsx").is_file()      # never touched


def test_other_origin_means_not_georeferenced(catalog, data_dir):
    res = migrate_dataset(catalog, data_dir, "fixture", log=_quiet,
                          origin=GeoPoint(lat=40.9, lon=-74.2, h=200))
    run = catalog.get_run(catalog.get_site("default", "fixture"), "baseline", str(res.run_id))
    assert not run.georeferenced


def test_migrated_site_is_served_with_metric(catalog, data_dir):
    migrate_dataset(catalog, data_dir, "fixture", log=_quiet)
    site = catalog.get_site("default", "fixture")
    scenes = discover_scenes(catalog.paths.root, data_dir)
    src = scenes["default/fixture"]
    assert scenes["fixture"] is src                    # bare id wins over the legacy dataset
    assert src.kind == "catalog" and src.splat.endswith("splat.pt")
    m = src.metric
    assert m is not None and m["units_to_metres"] == 1.0 and m["crs"] == "ENU"
    assert m["origin"] == {"lat": site.origin.lat, "lon": site.origin.lon, "h": site.origin.h}
    assert m["earth_model"] == "sphere-legacy" and 0 < m["frame_error_m"] < 3


def test_cli_and_tool_script(tmp_path, data_dir, capsys):
    rc = legacy.main(["fixture", "--data-dir", str(data_dir),
                      "--projects-root", str(tmp_path / "p")])
    assert rc == 0 and "default/fixture" in capsys.readouterr().out
    assert (tmp_path / "p" / "catalog.db").is_file()
    assert (ROOT / "tools" / "migrate_dataset.py").is_file()


def test_missing_geo_reference_needs_origin(catalog, tmp_path):
    (tmp_path / "data" / "bare" / "footage").mkdir(parents=True)
    with pytest.raises(legacy.CatalogError, match="origin"):
        migrate_dataset(catalog, tmp_path / "data", "bare", log=_quiet)
