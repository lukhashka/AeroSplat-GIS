"""Catalog (aerosplat.store): schema/migration sync, Site origin immutability, captures, runs."""

import json
from datetime import datetime

import pytest
import yaml
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from pydantic import ValidationError
from sqlalchemy import exc, select, text
from sqlalchemy.orm import Session

from aerosplat.geo.crs import GeoPoint
from aerosplat.store import (BBox, Capture, CaptureSource, Catalog, CatalogError, Epoch, Image,
                             NotFoundError, RunStatus, Site)
from aerosplat.store.catalog import alembic_config
from aerosplat.store.db import Base, OriginImmutableError, SiteRow

ORIGIN = GeoPoint(lat=40.8942, lon=-74.2173, h=233.0)
SHA = "a" * 64


def _site(cat, sid="s1", project="demo", origin=ORIGIN):
    cat.ensure_project(project)
    return cat.create_site(Site(id=sid, project_id=project, origin=origin))


def test_migrations_match_models(catalog):
    head = ScriptDirectory.from_config(alembic_config()).get_current_head()
    with catalog.engine.connect() as conn:
        ctx = MigrationContext.configure(conn, opts={"compare_type": True})
        assert compare_metadata(ctx, Base.metadata) == []
        assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar() == head


def test_catalog_db_is_wal(catalog):
    # API, workers and job processes share the file (aerosplat.store.db.make_engine).
    with catalog.engine.connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"


def test_reopen_keeps_data(tmp_path):
    cat = Catalog.open(tmp_path)
    _site(cat)
    cat.close()
    again = Catalog.open(tmp_path)
    assert [s.key for s in again.list_sites()] == ["demo/s1"]
    again.close()


def test_create_site_writes_tree_and_yaml(catalog):
    site = _site(catalog)
    d = catalog.site_dirs(site)
    for p in (d.captures, d.epochs, d.authored, d.missions):
        assert p.is_dir()
    assert d.root == catalog.paths.root / "demo" / "sites" / "s1"
    on_disk = yaml.safe_load(d.site_yaml.read_text(encoding="utf-8"))
    assert GeoPoint(**on_disk["origin"]) == ORIGIN
    assert catalog.paths.project_yaml("demo").is_file()
    assert catalog.get_site("demo", "s1") == site


def test_site_origin_is_immutable(catalog, tmp_path):
    _site(catalog)
    with pytest.raises(CatalogError, match="already exists"):
        _site(catalog)
    with Session(catalog.engine) as s:          # ORM guard
        row = s.scalar(select(SiteRow))
        assert row is not None
        with pytest.raises(OriginImmutableError):
            row.origin_lat = 41.0
    with pytest.raises(exc.DatabaseError, match="immutable"):   # DB trigger, no ORM
        with catalog.engine.begin() as conn:
            conn.execute(text("UPDATE sites SET origin_h = 0"))
    assert catalog.get_site("demo", "s1").origin == ORIGIN
    # A lost/rebuilt index cannot silently re-create the site with another origin.
    other = Catalog.open(catalog.paths.root, db=tmp_path / "other.db")
    with pytest.raises(CatalogError, match="different origin"):
        _site(other, origin=GeoPoint(lat=40.9, lon=-74.2, h=0))
    other.close()


def test_adding_captures_never_moves_origin(catalog):
    site = _site(catalog)
    far = GeoPoint(lat=ORIGIN.lat + 0.05, lon=ORIGIN.lon - 0.05, h=900)
    first = [Image(name=f"a{i}.jpg", sha256=SHA, bytes=10, prior=far) for i in range(3)]
    catalog.add_capture(site, Capture(id="drone-1", source=CaptureSource.drone), first)
    catalog.add_capture(site, Capture(id="phone-1", source=CaptureSource.handheld),
                        [Image(name="b.jpg", sha256=SHA, bytes=5)])
    assert catalog.get_site("demo", "s1").origin == ORIGIN
    on_disk = yaml.safe_load(catalog.site_dirs(site).site_yaml.read_text(encoding="utf-8"))
    assert GeoPoint(**on_disk["origin"]) == ORIGIN
    assert [c.id for c in catalog.list_captures(site)] == ["drone-1", "phone-1"]
    imgs = catalog.list_images(site, "drone-1")
    assert [i.prior for i in imgs] == [far] * 3
    cdirs = catalog.site_dirs(site).capture("drone-1")
    lines = cdirs.images_jsonl.read_text(encoding="utf-8").splitlines()
    assert [json.loads(ln)["name"] for ln in lines] == ["a0.jpg", "a1.jpg", "a2.jpg"]
    assert yaml.safe_load(cdirs.capture_yaml.read_text(encoding="utf-8"))["num_images"] == 3
    with pytest.raises(CatalogError):
        catalog.add_capture(site, Capture(id="drone-1", source=CaptureSource.drone), [])


def test_run_lifecycle_and_latest(catalog):
    site = _site(catalog)
    catalog.ensure_epoch(site)
    catalog.add_capture(site, Capture(id="c1", source=CaptureSource.earthstudio), [])
    r1 = catalog.create_run(site, "baseline", "20260101-000000", capture_ids=["c1"],
                            config={"x": 1}, georeferenced=True)
    assert r1.status == RunStatus.running and r1.capture_ids == ["c1"]
    rd = catalog.run_dirs(site, "baseline", r1.id)
    (rd.gs).mkdir(parents=True)
    (rd.gs / "splat.pt").write_bytes(b"x")
    r1 = catalog.finish_run(site, "baseline", r1.id, status=RunStatus.ok,
                            products={"splat": rd.gs / "splat.pt", "cloud": rd.gs / "none.pt"})
    assert r1.products == {"splat": "gs/splat.pt"} and r1.finished_at is not None
    assert catalog.product_path(site, r1, "splat") == rd.gs / "splat.pt"

    r2 = catalog.create_run(site, "baseline", "20260102-000000")
    catalog.finish_run(site, "baseline", r2.id, status=RunStatus.failed)
    assert catalog.latest_run(site).id == r1.id
    assert catalog.latest_run(site, with_products=("cloud",)) is None
    assert catalog.latest_run(site, status=RunStatus.failed).id == r2.id
    with pytest.raises(CatalogError):
        catalog.create_run(site, "baseline", r1.id)
    with pytest.raises(NotFoundError):
        catalog.create_run(site, "no-such-epoch", "x")


def test_epochs(catalog):
    site = _site(catalog)
    e = catalog.ensure_epoch(site)
    assert e.id == "baseline" and catalog.ensure_epoch(site) == e
    post = catalog.create_epoch(site, Epoch(id="post-flood-2026-11-03", label="after the flood"))
    assert [x.id for x in catalog.list_epochs(site)] == ["baseline", post.id]
    assert catalog.site_dirs(site).epoch_yaml(post.id).is_file()
    with pytest.raises(ValidationError):          # catalog timestamps are UTC-aware
        Epoch(id="naive", valid_from=datetime(2026, 1, 1))


def test_find_site(catalog):
    _site(catalog, "pier", "a")
    _site(catalog, "pier", "b")
    _site(catalog, "depot", "a")
    assert catalog.find_site("depot").key == "a/depot"
    assert catalog.find_site("b/pier").key == "b/pier"
    with pytest.raises(CatalogError, match="ambiguous"):
        catalog.find_site("pier")
    with pytest.raises(NotFoundError):
        catalog.find_site("nope")


def test_model_validation():
    with pytest.raises(ValidationError):
        Site(id="Bad Id", project_id="demo", origin=ORIGIN)
    with pytest.raises(ValidationError):
        BBox(west=1, south=0, east=0, north=1)
    with pytest.raises(ValidationError):
        Image(name="x", sha256="nothex", bytes=1)
