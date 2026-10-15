import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src",):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

FIXTURES = Path(__file__).parent / "fixtures"
ES_JSON = FIXTURES / "earthstudio_3frames.json"


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
