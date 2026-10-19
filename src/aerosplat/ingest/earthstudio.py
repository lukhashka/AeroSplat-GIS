"""Earth Studio adapter (ingest): an export (JSON + rendered frames) -> Capture in a Site.

Only files move here; nothing is converted. Poses are turned into Site ENU at
run time by `aerosplat.geo.earthstudio_to_colmap` with the Site's fixed origin,
so one capture can serve every run and epoch of the Site.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from aerosplat.geo.crs import GeoPoint
from aerosplat.geo.earthstudio_to_colmap import find_footage, frame_coordinates, load_frames
from aerosplat.ingest.images import ProgressFn, describe_images, transfer
from aerosplat.store.catalog import Catalog
from aerosplat.store.models import Capture, CaptureSource, Image, Site
from aerosplat.store.paths import CaptureDirs

LICENSE = "Google Earth Studio Terms of Service (Google Earth imagery)"
ATTRIBUTION_FILE = "ImagerySources.txt"


def frame_prefix(json_path: Path) -> str:
    data, _ = load_frames(json_path)
    return str(data.get("name") or json_path.stem)


def default_capture_id(json_path: Path) -> str:
    return f"earthstudio-{frame_prefix(json_path)}".lower()


def prepare_earthstudio(json_path: Path, footage_dir: Path, dest: CaptureDirs, *,
                        capture_id: Optional[str] = None, extra_files: Sequence[Path] = (),
                        author: str = "", move: bool = False,
                        on_progress: Optional[ProgressFn] = None) -> tuple[Capture, list[Image]]:
    """Lay the export out as a capture directory at `dest` and describe it; registers nothing.

    `dest` may be the final capture directory or a staging one that is renamed into
    place afterwards (the ingest job does that, so a half-built capture is never in
    its final location). `extra_files` (the .esp project, ImagerySources.txt, ...) go
    to source/ next to the JSON; anything else in the export folder is left behind.
    """
    json_path, footage_dir = Path(json_path), Path(footage_dir)
    _, frames = load_frames(json_path)
    prefix = frame_prefix(json_path)
    names = find_footage(str(footage_dir), prefix, len(frames))
    capture_id = capture_id or default_capture_id(json_path)

    if dest.root.exists() and any(dest.root.iterdir()):
        raise FileExistsError(f"capture directory already populated: {dest.root}")
    dest.images.mkdir(parents=True, exist_ok=True)
    for name in names:
        transfer(footage_dir / name, dest.images / name, move=move)
    transfer(json_path, dest.source / json_path.name, move=move)
    attribution = ["Google Earth"]
    for extra in extra_files:
        extra = Path(extra)
        if extra.is_file():
            transfer(extra, dest.source / extra.name, move=move)
            if extra.name == ATTRIBUTION_FILE:
                text = (dest.source / extra.name).read_text(encoding="utf-8", errors="replace")
                attribution += [ln.strip() for ln in text.splitlines() if ln.strip()]

    lat, lon, alt = frame_coordinates(frames)
    priors = {n: GeoPoint(lat=float(a), lon=float(b), h=float(c))
              for n, a, b, c in zip(names, lat, lon, alt)}
    images = describe_images(dest.images, names, priors, on_progress=on_progress)
    captured = datetime.fromtimestamp((dest.source / json_path.name).stat().st_mtime,
                                      tz=timezone.utc).replace(microsecond=0)
    capture = Capture(
        id=capture_id, source=CaptureSource.earthstudio, license=LICENSE,
        attribution="; ".join(attribution), author=author, captured_at=captured,
        metadata_file=f"source/{json_path.name}", frame_prefix=prefix,
        notes="Rendered by Google Earth Studio; poses are exact with respect to the "
              "Google globe, whose own absolute accuracy is not known here.")
    return capture, images


def import_earthstudio(catalog: Catalog, site: Site, json_path: Path, footage_dir: Path, *,
                       capture_id: Optional[str] = None, extra_files: Sequence[Path] = (),
                       author: str = "", move: bool = False) -> Capture:
    """Copy (or move) an Earth Studio export into a new capture of `site` and register it."""
    capture_id = capture_id or default_capture_id(Path(json_path))
    capture, images = prepare_earthstudio(
        json_path, footage_dir, catalog.site_dirs(site).capture(capture_id),
        capture_id=capture_id, extra_files=extra_files, author=author, move=move)
    return catalog.add_capture(site, capture, images)
