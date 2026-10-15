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
from aerosplat.ingest.images import describe_images, transfer
from aerosplat.store.catalog import Catalog
from aerosplat.store.models import Capture, CaptureSource, Site

LICENSE = "Google Earth Studio Terms of Service (Google Earth imagery)"
ATTRIBUTION_FILE = "ImagerySources.txt"


def frame_prefix(json_path: Path) -> str:
    data, _ = load_frames(json_path)
    return str(data.get("name") or json_path.stem)


def import_earthstudio(catalog: Catalog, site: Site, json_path: Path, footage_dir: Path, *,
                       capture_id: Optional[str] = None, extra_files: Sequence[Path] = (),
                       author: str = "", move: bool = False) -> Capture:
    """Copy (or move) an Earth Studio export into a new capture of `site` and register it.

    `extra_files` (the .esp project, ImagerySources.txt, ...) go to the
    capture's source/ directory next to the JSON; anything else in the export
    folder is left behind on purpose.
    """
    json_path, footage_dir = Path(json_path), Path(footage_dir)
    _, frames = load_frames(json_path)
    prefix = frame_prefix(json_path)
    names = find_footage(str(footage_dir), prefix, len(frames))
    capture_id = capture_id or f"earthstudio-{prefix}".lower()

    cdirs = catalog.site_dirs(site).capture(capture_id)
    if cdirs.root.exists() and any(cdirs.root.iterdir()):
        raise FileExistsError(f"capture directory already populated: {cdirs.root}")
    cdirs.images.mkdir(parents=True, exist_ok=True)
    for name in names:
        transfer(footage_dir / name, cdirs.images / name, move=move)
    transfer(json_path, cdirs.source / json_path.name, move=move)
    attribution = ["Google Earth"]
    for extra in extra_files:
        extra = Path(extra)
        if extra.is_file():
            transfer(extra, cdirs.source / extra.name, move=move)
            if extra.name == ATTRIBUTION_FILE:
                text = (cdirs.source / extra.name).read_text(encoding="utf-8", errors="replace")
                attribution += [ln.strip() for ln in text.splitlines() if ln.strip()]

    lat, lon, alt = frame_coordinates(frames)
    priors = {n: GeoPoint(lat=float(a), lon=float(b), h=float(c))
              for n, a, b, c in zip(names, lat, lon, alt)}
    images = describe_images(cdirs.images, names, priors)
    captured = datetime.fromtimestamp((cdirs.source / json_path.name).stat().st_mtime,
                                      tz=timezone.utc).replace(microsecond=0)
    capture = Capture(
        id=capture_id, source=CaptureSource.earthstudio, license=LICENSE,
        attribution="; ".join(attribution), author=author, captured_at=captured,
        metadata_file=f"source/{json_path.name}", frame_prefix=prefix,
        notes="Rendered by Google Earth Studio; poses are exact with respect to the "
              "Google globe, whose own absolute accuracy is not known here.")
    return catalog.add_capture(site, capture, images)
