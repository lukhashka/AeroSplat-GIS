"""Training checkpoints of the gsplat trainer (ROADMAP Фаза 2.2): files, atomicity, discovery.

What goes into a checkpoint is decided by the training loop
(gsplat_backend.train_gsplat); this module only stores and finds them:

* `<output stem>_checkpoints/ckpt_<iteration>.pt` next to the output `.pt`
  (in a site run: `gs/splat_checkpoints/`);
* written to `<file>.tmp` and renamed (`os.replace` is atomic on one volume),
  so a crash mid-write leaves the previous checkpoints intact and never a
  truncated `ckpt_*.pt`;
* only the newest `keep` are kept, and none once the run has finished (the final
  `.pt` is the result; a finished run's checkpoints cannot be resumed into anything);
* loading walks from the newest down, skipping unreadable files with a
  warning, and uses `weights_only=True` (tensors and plain containers only).
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

CHECKPOINT_VERSION = 1
_NAME = re.compile(r"^ckpt_(\d+)\.pt$")


def _log(msg: str) -> None:
    print(msg, flush=True)


def checkpoint_dir(output_path: str) -> str:
    """`gs/splat.pt` -> `gs/splat_checkpoints` (one directory per output file)."""
    base, _ = os.path.splitext(os.path.abspath(output_path))
    return base + "_checkpoints"


def list_checkpoints(directory: str) -> List[Tuple[int, str]]:
    """[(iteration, path)] of complete checkpoints, oldest first."""
    if not os.path.isdir(directory):
        return []
    found = []
    for name in os.listdir(directory):
        m = _NAME.match(name)
        if m:
            found.append((int(m.group(1)), os.path.join(directory, name)))
    return sorted(found)


def write_checkpoint(directory: str, iteration: int, payload: Dict[str, Any], keep: int) -> str:
    """Atomically write `payload` as checkpoint `iteration`, then keep only the newest `keep`."""
    if keep < 1:
        raise ValueError(f"keep must be >= 1, got {keep}")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"ckpt_{iteration:07d}.pt")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(dict(payload, version=CHECKPOINT_VERSION, iteration=iteration), f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    for _, old in list_checkpoints(directory)[:-keep]:
        try:
            os.remove(old)
        except OSError as e:
            _log(f"  ⚠️  Не вдалося видалити старий чекпойнт {old}: {e}")
    return path


def clear_checkpoints(directory: str) -> int:
    """Delete every checkpoint (and leftover .tmp) in `directory`: a fresh run supersedes them."""
    n = 0
    if not os.path.isdir(directory):
        return 0
    for name in os.listdir(directory):
        if _NAME.match(name) or (name.startswith("ckpt_") and name.endswith(".pt.tmp")):
            os.remove(os.path.join(directory, name))
            n += 1
    if not os.listdir(directory):
        os.rmdir(directory)
    return n


def load_latest_checkpoint(directory: str, device) -> Optional[Tuple[str, Dict[str, Any]]]:
    """(path, payload) of the newest readable checkpoint, or None when there is none."""
    for iteration, path in reversed(list_checkpoints(directory)):
        try:
            payload = torch.load(path, map_location=device, weights_only=True)
        except Exception as e:  # truncated / corrupt file: fall back to an older one
            _log(f"  ⚠️  Чекпойнт {path} не читається ({type(e).__name__}: {e}) — беру старіший")
            continue
        if payload.get("version") != CHECKPOINT_VERSION or payload.get("iteration") != iteration:
            _log(f"  ⚠️  Чекпойнт {path}: версія {payload.get('version')} / ітерація "
                 f"{payload.get('iteration')} не та — беру старіший")
            continue
        return path, payload
    return None


def signature_diff(saved: Dict[str, Any], current: Dict[str, Any], prefix: str = "") -> List[str]:
    """Keys whose values differ between two (nested) signatures, for the error message."""
    out = []
    for key in sorted(set(saved) | set(current)):
        a, b = saved.get(key), current.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            out += signature_diff(a, b, f"{prefix}{key}.")
        elif a != b:
            if isinstance(a, list) or isinstance(b, list):
                out.append(f"{prefix}{key}")
            else:
                out.append(f"{prefix}{key}: {a!r} → {b!r}")
    return out
