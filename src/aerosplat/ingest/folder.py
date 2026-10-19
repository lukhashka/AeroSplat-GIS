"""Generic adapter: a folder of images (no pose metadata) -> Capture in a Site."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from aerosplat.ingest.images import ProgressFn, describe_images, list_images, transfer
from aerosplat.store.catalog import Catalog
from aerosplat.store.models import Capture, CaptureSource, Image, Site
from aerosplat.store.paths import CaptureDirs


def prepare_folder(images_dir: Path, dest: CaptureDirs, *, capture_id: str,
                   source: CaptureSource, license: str = "", author: str = "",
                   notes: str = "", move: bool = False,
                   pose_prior_sigma_m: Optional[float] = None,
                   on_progress: Optional[ProgressFn] = None) -> tuple[Capture, list[Image]]:
    """Lay the images out as a capture directory at `dest` and describe them; registers nothing."""
    images_dir = Path(images_dir)
    names = list_images(images_dir)
    if not names:
        raise FileNotFoundError(f"no images in {images_dir}")
    if dest.root.exists() and any(dest.root.iterdir()):
        raise FileExistsError(f"capture directory already populated: {dest.root}")
    dest.images.mkdir(parents=True, exist_ok=True)
    for name in names:
        transfer(images_dir / name, dest.images / name, move=move)
    capture = Capture(id=capture_id, source=source, license=license, author=author,
                      pose_prior_sigma_m=pose_prior_sigma_m, notes=notes)
    return capture, describe_images(dest.images, names, on_progress=on_progress)


def import_folder(catalog: Catalog, site: Site, images_dir: Path, *, capture_id: str,
                  source: CaptureSource, license: str = "", author: str = "",
                  notes: str = "", move: bool = False,
                  pose_prior_sigma_m: Optional[float] = None) -> Capture:
    capture, images = prepare_folder(
        images_dir, catalog.site_dirs(site).capture(capture_id), capture_id=capture_id,
        source=source, license=license, author=author, notes=notes, move=move,
        pose_prior_sigma_m=pose_prior_sigma_m)
    return catalog.add_capture(site, capture, images)
