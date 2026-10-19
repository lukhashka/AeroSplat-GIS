"""Extra per-image supervision for the 3DGS trainer (ROADMAP Фаза 2.2): masks and depth.

Masks
-----
A mask excludes pixels (sky, moving objects, watermarks, invalid borders) from
the photometric loss and from the depth loss. Convention — COLMAP's own
(`ImageReader.mask_path`), so one set of files serves SfM and training:

* one directory of masks; the mask of image `<name>` (as COLMAP names it,
  e.g. `dataset_v6_00.jpeg`) is `<name>.png`; `<stem>.png` is accepted too;
* single channel (or anything PIL converts to "L"): **0 = ignore the pixel**,
  anything else = use it; any size with the image's aspect ratio (it is
  resampled to the training resolution);
* an image without a mask file is used whole.

In the pipeline the directory is `masks/` next to the capture's images
(`<capture>/masks/`, legacy: `data/<ds>/masks/`), or `masks_dir`. Producing
masks is a separate step: any segmenter that writes these files plugs in here.
An automatic sky heuristic is deliberately not built in — colour/position rules
mistake water, glass and blue roofs for sky on oblique aerial frames, and a
wrong mask silently deletes real geometry.

Depth
-----
Targets come from COLMAP's MVS depth maps (the `dense` stage,
`<dense>/stereo/depth_maps/<name>.geometric.bin`). They are z-depths in the
units of the very reconstruction the trainer's cameras come from, so the loss
compares them directly (no scale/shift fitting, unlike monocular priors):
metric whenever the poses are (Earth Studio known poses: metres). The loss is
the relative error |d̂ − d| / d, i.e. independent of the scene's absolute scale.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# A training pixel is valid only if (almost) all of its footprint in the mask is:
# conservative, so a mask edge never leaks sky into the loss after resampling.
MASK_KEEP_FRACTION = 0.999
# Pose / intrinsics of a depth map must match the training camera this closely.
DEPTH_ROT_TOL = 1e-4
DEPTH_REL_TOL = 1e-3
# Rendered pixels with less accumulated opacity have no meaningful expected depth.
DEPTH_MIN_ALPHA = 0.5


def _log(msg: str) -> None:
    print(msg, flush=True)


# ═══════════════════════════════════════════════════════════════════
#  Resampling
# ═══════════════════════════════════════════════════════════════════

def resize_mask(mask: Tensor, W: int, H: int) -> Tensor:
    """(h,w) bool -> (H,W) bool; a pixel stays valid only if its whole footprint is."""
    if mask.shape == (H, W):
        return mask.clone()
    frac = F.interpolate(mask.float()[None, None], size=(H, W), mode="area")[0, 0]
    return frac >= MASK_KEEP_FRACTION


def resize_depth(depth: Tensor, W: int, H: int) -> Tensor:
    """(h,w) -> (H,W) by nearest sampling: never blends depths across an edge or with holes."""
    if depth.shape == (H, W):
        return depth.clone()
    return F.interpolate(depth[None, None], size=(H, W), mode="nearest-exact")[0, 0]


# ═══════════════════════════════════════════════════════════════════
#  Masks
# ═══════════════════════════════════════════════════════════════════

def find_mask(masks_dir: str, image_name: str) -> Optional[str]:
    """`<name>.png` (COLMAP) or `<stem>.png` in `masks_dir`; None if there is none."""
    for cand in (image_name + ".png", os.path.splitext(image_name)[0] + ".png"):
        path = os.path.join(masks_dir, cand)
        if os.path.isfile(path):
            return path
    return None


def load_mask(path: str, W: int, H: int) -> Tensor:
    """Mask file -> (H,W) bool, True = use the pixel. The aspect ratio must match the image's."""
    from PIL import Image

    with Image.open(path) as img:
        arr = np.asarray(img.convert("L"))
    h, w = arr.shape
    if abs((w / h) / (W / H) - 1.0) > 0.01:
        raise ValueError(f"маска {path}: {w}×{h}, а кадр {W}×{H} — інше співвідношення сторін")
    return resize_mask(torch.from_numpy(arr > 0), W, H)


def attach_masks(cams: Sequence, masks_dir: str) -> int:
    """Set `cam.mask` from `masks_dir` (see the module docstring). Returns how many got one.

    Fails when the directory is missing or matches no image at all: masks were
    asked for, so silently training without them would be the worse outcome.
    """
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"Каталог масок не знайдено: {masks_dir}")
    n, empty = 0, []
    for cam in cams:
        path = find_mask(masks_dir, cam.name)
        if path is None:
            continue
        cam.mask = load_mask(path, cam.W, cam.H)
        n += 1
        if not bool(cam.mask.any()):
            empty.append(cam.name)
    if n == 0:
        raise FileNotFoundError(
            f"У {masks_dir} немає жодної маски для {len(cams)} кадрів "
            f"(очікується <ім'я кадру>.png, напр. {cams[0].name}.png)" if cams else
            f"У {masks_dir} немає масок")
    valid = [float(c.mask.float().mean()) for c in cams if c.mask is not None]
    _log(f"  Маски: {n} з {len(cams)} кадрів ({masks_dir}); у середньому валідно "
         f"{100 * sum(valid) / len(valid):.1f}% пікселів")
    if empty:
        _log(f"  ⚠️  {len(empty)} кадр(ів) замасковано повністю — у loss не беруть участі: "
             + ", ".join(empty[:5]) + (" …" if len(empty) > 5 else ""))
    return n


# ═══════════════════════════════════════════════════════════════════
#  COLMAP MVS depth maps
# ═══════════════════════════════════════════════════════════════════

def read_colmap_array(path: str) -> np.ndarray:
    """COLMAP dense array (`<w>&<h>&<c>&` + float32 column-major) -> (h, w) or (h, w, c)."""
    with open(path, "rb") as f:
        header = b""
        while header.count(b"&") < 3:
            ch = f.read(1)
            if not ch:
                raise ValueError(f"{path}: обірваний заголовок COLMAP-масиву")
            header += ch
            if len(header) > 64:
                raise ValueError(f"{path}: не схоже на COLMAP-масив")
        w, h, c = (int(v) for v in header.split(b"&")[:3])
        data = np.fromfile(f, dtype=np.float32)
    if data.size != w * h * c:
        raise ValueError(f"{path}: {data.size} значень замість {w}×{h}×{c}")
    arr = data.reshape((w, h, c), order="F").transpose(1, 0, 2)
    return arr[:, :, 0] if c == 1 else arr


def write_colmap_array(path: str, arr: np.ndarray) -> None:
    """Inverse of read_colmap_array (tests and tools)."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        a = a[:, :, None]
    h, w, c = a.shape
    with open(path, "wb") as f:
        f.write(f"{w}&{h}&{c}&".encode())
        a.transpose(1, 0, 2).reshape(-1, order="F").tofile(f)


def depth_map_path(dense_dir: str, image_name: str, kind: str = "geometric") -> str:
    return os.path.join(dense_dir, "stereo", "depth_maps", f"{image_name}.{kind}.bin")


def _check_same_view(cam, ref, path: str) -> None:
    """The depth map's camera (dense/sparse) must be the training camera, up to resolution."""
    dR = float((cam.R - ref.R).abs().max())
    dt = float((cam.t - ref.t).norm())
    t_tol = DEPTH_REL_TOL * max(1.0, float(cam.t.norm()))
    if dR > DEPTH_ROT_TOL or dt > t_tol:
        raise ValueError(f"{path}: поза карти глибини не збігається з камерою тренування "
                         f"(ΔR={dR:.2e}, Δt={dt:.3g}) — інша реконструкція?")
    sx, sy = cam.W / ref.W, cam.H / ref.H
    for what, got, want in (("fx", ref.fx * sx, cam.fx), ("fy", ref.fy * sy, cam.fy),
                            ("cx", ref.cx * sx, cam.cx), ("cy", ref.cy * sy, cam.cy)):
        # + 1 px: both resolutions are rounded to whole pixels independently
        if abs(got - want) > DEPTH_REL_TOL * abs(want) + 1.0:
            raise ValueError(f"{path}: {what} карти глибини після масштабування {got:.3f}, "
                             f"а в камери тренування {want:.3f} — інший кадр/дисторсія?")


def attach_mvs_depth(cams: Sequence, dense_dir: str, kind: str = "geometric",
                     train_sparse: Optional[str] = None) -> int:
    """Set `cam.depth` from COLMAP MVS depth maps of the `dense` stage. Returns how many got one.

    The depth maps belong to the undistorted model `<dense>/sparse`; each one's
    pose and (scaled) intrinsics are checked against the training camera, so a
    depth map from another reconstruction or resolution fails loudly instead of
    pulling the scene towards wrong geometry. `train_sparse` (the model the
    cameras were read from) must then be distortion-free too: the trainer
    ignores distortion, so a SIMPLE_RADIAL frame would match the pinhole check
    and still disagree with the undistorted depth map pixel by pixel.
    """
    from aerosplat.gs.trainer import load_colmap_cameras

    if train_sparse is not None:
        from aerosplat.io.colmap_model import read_model

        models = {p.intrinsics.model for p in read_model(train_sparse)}
        if not models <= {"PINHOLE", "SIMPLE_PINHOLE"}:
            raise ValueError(f"Глибина MVS: камери тренування ({train_sparse}) мають дисторсію "
                             f"({', '.join(sorted(models))}) — тренуй на undistorted-кадрах dense")
    sparse = os.path.join(dense_dir, "sparse")
    if not os.path.isdir(sparse):
        raise FileNotFoundError(f"Немає {sparse} — стадія dense не виконувалась?")
    refs = {c.name: c for c in load_colmap_cameras(sparse)}
    n, missing, coverage = 0, 0, []
    for cam in cams:
        path = depth_map_path(dense_dir, cam.name, kind)
        ref = refs.get(cam.name)
        if ref is None or not os.path.isfile(path):
            missing += 1
            continue
        _check_same_view(cam, ref, path)
        d = read_colmap_array(path)
        if d.ndim != 2 or d.shape != (ref.H, ref.W):
            raise ValueError(f"{path}: {d.shape}, а камера dense/sparse {ref.H}×{ref.W}")
        d = np.where(np.isfinite(d) & (d > 0), d, 0.0).astype(np.float32)
        cam.depth = resize_depth(torch.from_numpy(d), cam.W, cam.H)
        coverage.append(float((cam.depth > 0).float().mean()))
        n += 1
    if n == 0:
        raise FileNotFoundError(f"Жодної карти глибини ({kind}) для {len(cams)} кадрів у "
                                f"{os.path.join(dense_dir, 'stereo', 'depth_maps')}")
    _log(f"  Глибина MVS ({kind}): {n} з {len(cams)} кадрів, покриття у середньому "
         f"{100 * sum(coverage) / n:.1f}% пікселів" + (f"; без карти: {missing}" if missing else ""))
    return n


# ═══════════════════════════════════════════════════════════════════
#  Loss
# ═══════════════════════════════════════════════════════════════════

def depth_loss(pred: Tensor, target: Tensor, valid: Optional[Tensor] = None) -> Tensor:
    """Mean relative error |d̂ − d| / d over pixels with a target (d > 0) and `valid`.

    pred, target, valid: (H,W). L1 on the relative error: scale-free (the same
    weight works for a 50 m and a 500 m scene) and robust to MVS outliers.
    Returns an exact 0 (still attached to `pred`) when no pixel qualifies.
    """
    m = target > 0
    if valid is not None:
        m = m & valid
    if not bool(m.any()):
        return pred.sum() * 0.0
    t = target[m]
    return ((pred[m] - t).abs() / t).mean()


def depth_weight_at(iteration: int, total: int, start: float, end: float) -> float:
    """Log-linear decay from `start` (iteration 0) to `end` (the last one), like Inria's
    depth-L1 schedule: geometry early, when it decides where splats go, photometry late.
    `start` <= 0 switches the depth loss off."""
    if start <= 0:
        return 0.0
    if end <= 0:
        raise ValueError(f"depth_weight_final must be > 0, got {end}")
    t = min(1.0, max(0.0, iteration / max(1, total)))
    return math.exp((1.0 - t) * math.log(start) + t * math.log(end))


def supervision_summary(cams: Sequence) -> Tuple[List[str], List[str]]:
    """(names with a mask, names with depth), sorted: part of the checkpoint signature."""
    return (sorted(c.name for c in cams if c.mask is not None),
            sorted(c.name for c in cams if c.depth is not None))

