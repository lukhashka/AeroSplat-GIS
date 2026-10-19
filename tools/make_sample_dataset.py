"""Make the small sample dataset of the Phase 0 readiness criterion (ROADMAP P0, "sample").

Takes every `--every`-th frame of an Earth Studio dataset (footage + 3D tracking JSON), at most
`--max` frames, optionally downscaled by `--scale`, and writes them as a new dataset
`data/<name>/` in the layout the pipeline expects (`footage/<name>_<n>.jpeg`, `<name>.json`).
The source is only read. The JSON keeps every key; `cameraFrames` is subset in step with the
footage, `width`/`height` follow the scale, and `fovVertical` is unchanged, so the intrinsics
the converter derives stay right at any scale.

    python tools/make_sample_dataset.py --src data/dataset_v6 --name sample
    aerosplat run --dataset sample --iterations 500
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image


def frames(footage: Path, stem: str) -> list[tuple[int, Path]]:
    pat = re.compile(rf"^{re.escape(stem)}_(\d+)\.(jpe?g|png)$", re.I)
    found = sorted((int(m.group(1)), p) for p in footage.iterdir() if (m := pat.match(p.name)))
    if [i for i, _ in found] != list(range(len(found))):
        raise SystemExit(f"{footage}: frames {stem}_<n> are not numbered 0..{len(found) - 1}")
    return found


def make(src: Path, name: str, every: int, max_frames: int, scale: float, out_root: Path) -> Path:
    stem = src.name
    data = json.loads((src / f"{stem}.json").read_text(encoding="utf-8"))
    files = frames(src / "footage", stem)
    if len(files) != len(data["cameraFrames"]):
        raise SystemExit(f"{len(files)} footage files but {len(data['cameraFrames'])} camera "
                         f"frames in {stem}.json")
    pick = list(range(0, len(files), every))[:max_frames]
    if len(pick) < 3:
        raise SystemExit(f"only {len(pick)} frames picked; need at least 3")
    dst = out_root / name
    if dst.exists():
        raise SystemExit(f"{dst} exists; remove it first (this tool never overwrites)")
    (dst / "footage").mkdir(parents=True)
    for k, i in enumerate(pick):
        img = Image.open(files[i][1])
        if scale != 1.0:
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        img.convert("RGB").save(dst / "footage" / f"{name}_{k:02d}.jpeg", quality=95)
    out = dict(data, name=name, cameraFrames=[data["cameraFrames"][i] for i in pick],
               width=round(data["width"] * scale), height=round(data["height"] * scale))
    out["numFrames"] = len(pick) - 1
    (dst / f"{name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (dst / "SOURCE.txt").write_text(
        f"Sample of {src} (frames {pick}, scale {scale}); made by tools/make_sample_dataset.py\n",
        encoding="utf-8")
    return dst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, default=Path("data/dataset_v6"))
    ap.add_argument("--name", default="sample")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--max", dest="max_frames", type=int, default=16)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--out-root", type=Path, default=Path("data"))
    a = ap.parse_args(argv)
    if a.every < 1 or not (0 < a.scale <= 1):
        ap.error("--every must be >= 1 and 0 < --scale <= 1")
    dst = make(a.src, a.name, a.every, a.max_frames, a.scale, a.out_root)
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
