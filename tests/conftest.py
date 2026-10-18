import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src",):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

FIXTURES = Path(__file__).parent / "fixtures"
ES_JSON = FIXTURES / "earthstudio_3frames.json"


# ── markers: `sitl` (opt-in) and `hazard("H-NNN")` (docs/HAZARDS.md) ─────────

def pytest_addoption(parser):
    parser.addoption("--sitl", action="store_true", default=False,
                     help="run tests marked `sitl` against ArduCopter SITL "
                          "(same as -m sitl or AEROSPLAT_SITL=1)")


def sitl_requested(config) -> bool:
    """SITL tests fly a vehicle, so a SITL that merely happens to be running is not enough:
    they run only on --sitl, AEROSPLAT_SITL=1, or a -m expression that selects `sitl`."""
    if config.getoption("--sitl") or os.environ.get("AEROSPLAT_SITL") == "1":
        return True
    return re.search(r"(?<!not )\bsitl\b", config.getoption("markexpr") or "") is not None


def pytest_collection_modifyitems(config, items):
    from aerosplat.hazards import HAZARDS_MD, load_hazards

    known = load_hazards()
    skip_sitl = pytest.mark.skip(reason="SITL tests are opt-in: pytest -m sitl (or --sitl)")
    run_sitl = sitl_requested(config)
    for item in items:
        if item.get_closest_marker("sitl") is not None and not run_sitl:
            item.add_marker(skip_sitl)
        for mark in item.iter_markers("hazard"):
            bad = [h for h in mark.args if h not in known] if mark.args else ["(no id)"]
            if bad:
                raise pytest.UsageError(f"{item.nodeid}: hazard {', '.join(map(str, bad))} "
                                        f"is not in {HAZARDS_MD.relative_to(ROOT)}")


# ── SITL fixtures (tests/sitl/; here, not in tests/sitl/conftest.py, because a second
#    conftest module would shadow `from conftest import ...` in the other tests) ─────────
#
# A SITL instance that is already running (`python -m aerosplat.sim.sitl start`, perhaps with
# Mission Planner attached) is used as is and left running. Otherwise the pinned build is started
# for the session and stopped at the end; AEROSPLAT_SITL_AUTOSTART=0 makes that a failure instead.
# AEROSPLAT_SITL_ENDPOINT=<pymavlink string> points the tests at some other SITL.
# Once SITL tests are requested nothing here skips: no pymavlink or no SITL is a failure.

@dataclass
class SitlRun:
    endpoint: str
    home: Optional[object]          # aerosplat.sim.sitl.Home the instance was launched with
    started_here: bool
    session: dict = field(default_factory=dict)


@pytest.fixture(scope="session")
def sitl_cfg():
    from aerosplat.sim.sitl import load_config

    return load_config()


@pytest.fixture(scope="session")
def sitl_run(sitl_cfg):
    try:
        import pymavlink  # noqa: F401
    except ImportError:
        pytest.fail("pymavlink is not installed: pip install -e .[sim]")
    from aerosplat.sim.sitl import ENDPOINT_ENV, Home, Sitl, SitlError, resolve_home

    if os.environ.get(ENDPOINT_ENV):
        yield SitlRun(os.environ[ENDPOINT_ENV], None, False)
        return
    sitl = Sitl(sitl_cfg)
    try:
        session = sitl.status()
        started = session is None
        if started:
            if os.environ.get("AEROSPLAT_SITL_AUTOSTART", "1") == "0":
                pytest.fail("SITL is not running and AEROSPLAT_SITL_AUTOSTART=0: "
                            "python -m aerosplat.sim.sitl start")
            session = sitl.start(resolve_home(sitl_cfg))
        endpoint = sitl.endpoint()
    except (SitlError, OSError, subprocess.SubprocessError) as e:
        pytest.fail(f"cannot start or reach SITL: {e}")
    home = Home.parse(session["home"], source="session") if session.get("home") else None
    try:
        yield SitlRun(endpoint, home, started, session)
    finally:
        if started:
            sitl.stop()


@pytest.fixture
def vehicle(sitl_run):
    from aerosplat.sim.vehicle import Vehicle

    with Vehicle.connect(sitl_run.endpoint, timeout=30) as v:
        yield v


@pytest.fixture
def catalog(tmp_path):
    from aerosplat.store import Catalog

    cat = Catalog.open(tmp_path / "projects")
    yield cat
    cat.close()


def _es_origin():
    from aerosplat.geo.crs import GeoPoint

    return GeoPoint(lat=40.8942, lon=-74.2173, h=233.0)


ES_ORIGIN = _es_origin()


@pytest.fixture
def es_site(tmp_path):
    """Catalog with site demo/es holding one Earth Studio capture (3 frames)."""
    from aerosplat.ingest.earthstudio import import_earthstudio
    from aerosplat.store import Catalog, Site

    root = tmp_path / "projects"
    cat = Catalog.open(root)
    cat.ensure_project("demo")
    site = cat.create_site(Site(id="es", project_id="demo", origin=ES_ORIGIN))
    js, footage = write_es_export(tmp_path / "export", "fixture")
    cap = import_earthstudio(cat, site, js, footage)
    yield cat, site, cap, root
    cat.close()


def write_jpeg(path: Path, shade: int = 100, size=(16, 12)) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (shade, 255 - shade, 60)).save(path, "JPEG")


def write_es_export(folder: Path, name: str = "fixture") -> tuple[Path, Path]:
    """An Earth Studio export: <name>.json (3 frames) + footage/<name>_NN.jpeg."""
    data = json.loads(ES_JSON.read_text())
    data["name"] = name
    js = folder / f"{name}.json"
    folder.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(data))
    for i in range(len(data["cameraFrames"])):
        write_jpeg(folder / "footage" / f"{name}_{i:02d}.jpeg", shade=40 * i)
    return js, folder / "footage"


def write_pt(path: Path, kind: str, n: int = 64, offset=(500.0, 200.0, 50.0)) -> None:
    import torch

    g = torch.Generator().manual_seed(0)
    xyz = torch.randn(n, 3, generator=g) * 10 + torch.tensor(offset)
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "splat":
        rot = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=1)
        torch.save({"xyz": xyz, "scale": torch.full((n, 3), 0.1), "rotation": rot,
                    "opacity": torch.full((n, 1), 0.8), "rgb": torch.rand(n, 3, generator=g),
                    "activations": {"scale_is_log": False, "opacity_is_logit": False}}, path)
    else:
        torch.save({"xyz": xyz, "rgb": torch.rand(n, 3, generator=g)}, path)


def make_legacy_dataset(data_dir: Path, name: str = "fixture") -> Path:
    """A finished `aerosplat run --dataset <name>` in the legacy data/ layout,
    as produced before ADR-0006 (spherical ENU around the mean of the frames)."""
    from aerosplat.geo.earthstudio_to_colmap import convert

    root = data_dir / name
    js, footage = write_es_export(root, name)
    (root / "ImagerySources.txt").write_text("Vexcel Imaging US, Inc.")
    (root / f"{name}.esp").write_text("{}")
    (root / "private.xlsx").write_bytes(b"not for the project tree")

    known = root / "sparse_known"
    convert(str(js), name, str(known), footage_dir=str(footage),
            images_out_dir=str(data_dir / f"raw_images_{name}"), write_exif=False)
    geo = json.loads((known / "geo_registration.json").read_text())
    geo["earth_model"] = "spherical, R=6371008.8m (IUGG mean radius) -- see module docstring"
    (known / "geo_registration.json").write_text(json.dumps(geo))

    colmap = data_dir / f"colmap_output_{name}"
    (colmap / "sparse" / "0").mkdir(parents=True)
    (colmap / "sparse" / "0" / "cameras.bin").write_bytes(b"cams")
    (colmap / "database.db").write_bytes(b"db")
    (colmap / "dense").mkdir()
    (colmap / "dense" / "fused.ply").write_bytes(b"ply")
    write_pt(data_dir / f"{name}_dense.pt", "dense")
    write_pt(data_dir / f"{name}_splat.pt", "splat")

    run = root / "run"
    run.mkdir()
    manifest = {
        "schema": 1, "dataset": name, "git_commit": "abc1234",
        "updated": "2026-10-01T00:50:21+0300",
        "config": {"dataset": name, "camera_source": "earthstudio_json", "known_poses": True,
                   "cloud_source": "dense"},
        "stages": {s: {"status": "ok"} for s in ("geo", "sfm", "dense", "parse", "train")},
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "run.log").write_text("log")
    (run / "metrics.json").write_text("{}")
    (run / "report.html").write_text("<html></html>")
    return root
