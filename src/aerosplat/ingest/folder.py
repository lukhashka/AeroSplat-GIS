"""Generic adapter: a folder of images (no pose metadata) -> Capture in a Site."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from aerosplat.ingest.images import describe_images, list_images, transfer
from aerosplat.store.catalog import Catalog
from aerosplat.store.models import Capture, CaptureSource, Site


def import_folder(catalog: Catalog, site: Site, images_dir: Path, *, capture_id: str,
                  source: CaptureSource, license: str = "", author: str = "",
                  notes: str = "", move: bool = False,
                  pose_prior_sigma_m: Optional[float] = None) -> Capture:
    images_dir = Path(images_dir)
    names = list_images(images_dir)
    if not names:
        raise FileNotFoundError(f"no images in {images_dir}")
    cdirs = catalog.site_dirs(site).capture(capture_id)
    if cdirs.root.exists() and any(cdirs.root.iterdir()):
        raise FileExistsError(f"capture directory already populated: {cdirs.root}")
    cdirs.images.mkdir(parents=True, exist_ok=True)
    for name in names:
        transfer(images_dir / name, cdirs.images / name, move=move)
    capture = Capture(id=capture_id, source=source, license=license, author=author,
                      pose_prior_sigma_m=pose_prior_sigma_m, notes=notes)
    return catalog.add_capture(site, capture, describe_images(cdirs.images, names))
