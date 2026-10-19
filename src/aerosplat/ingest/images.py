"""Image records for a capture directory: content hash, size, pose prior."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Optional

from aerosplat.geo.crs import GeoPoint
from aerosplat.store.models import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def image_size(path: Path) -> tuple[Optional[int], Optional[int]]:
    try:
        from PIL import Image as PILImage

        with PILImage.open(path) as im:   # reads the header only
            return int(im.width), int(im.height)
    except Exception:
        return None, None


def list_images(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


# (done, total) after each image is hashed; the ingest job turns it into progress events.
ProgressFn = Callable[[int, int], None]


def describe_images(directory: Path, names: Iterable[str],
                    priors: Optional[Mapping[str, GeoPoint]] = None,
                    prior_sigma_m: Optional[float] = None,
                    on_progress: Optional[ProgressFn] = None) -> list[Image]:
    names = list(names)
    out = []
    for i, name in enumerate(names, 1):
        p = directory / name
        w, h = image_size(p)
        prior = (priors or {}).get(name)
        out.append(Image(name=name, sha256=sha256_file(p), bytes=p.stat().st_size,
                         width=w, height=h, prior=prior,
                         prior_sigma_m=prior_sigma_m if prior is not None else None))
        if on_progress is not None:
            on_progress(i, len(names))
    return out


def transfer(src: Path, dst: Path, *, move: bool = False) -> None:
    """Copy (or move) a file or directory tree; existing identical-size files are kept."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        if dst.exists():
            raise FileExistsError(f"refusing to move onto existing {dst}")
        shutil.move(str(src), str(dst))
        return
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        copy_function=_copy_if_changed)
    else:
        _copy_if_changed(src, dst)


def _copy_if_changed(src: str | Path, dst: str | Path) -> str:
    s, d = Path(src), Path(dst)
    if not (d.is_file() and d.stat().st_size == s.stat().st_size):
        shutil.copy2(s, d)
    return str(d)
