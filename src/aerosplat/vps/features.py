"""Pluggable local features for VPS (OpenCV, CPU). numpy + cv2 only: runs on the companion.

    ex = make_extractor("orb", n_max=1000)
    f = ex.detect(gray)          # Features: xy (N,2) float32, desc (N,D) uint8, response (N,)

Pixel coordinates follow OpenCV: the centre of pixel (0, 0) is (0, 0). (The render service puts
it at (0.5, 0.5); `aerosplat.vps.build` converts its intrinsics.) Keypoints are bucketed on a
grid so that a textured corner of the image cannot take the whole budget: PnP needs points
spread over the frame, not only many of them.

Both built-in extractors are binary (Hamming distance). A float extractor (SIFT, a learned one)
would plug in with `norm = "l2"`; the atlas records which extractor and parameters built it and
the localizer refuses an atlas whose extractor it cannot reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import numpy as np

CV2_HINT = ("aerosplat.vps needs OpenCV 4.x: pip install \"opencv-python-headless>=4.8,<5\" "
            "(or the optional extra: pip install -e .[vps]). OpenCV 5 moved AKAZE to contrib.")


def require_cv2() -> Any:
    try:
        import cv2
    except ImportError as e:            # pragma: no cover - depends on the environment
        raise RuntimeError(CV2_HINT) from e
    return cv2


@dataclass(frozen=True)
class Features:
    xy: np.ndarray          # (N, 2) float32, OpenCV pixel coordinates
    desc: np.ndarray        # (N, D) uint8
    response: np.ndarray    # (N,) float32

    def __len__(self) -> int:
        return int(self.xy.shape[0])

    def take(self, idx: np.ndarray) -> "Features":
        return Features(self.xy[idx], self.desc[idx], self.response[idx])


class Extractor(Protocol):
    name: str
    norm: str               # "hamming" | "l2"

    @property
    def params(self) -> dict[str, Any]: ...

    def detect(self, gray: np.ndarray) -> Features: ...


def bucket(f: Features, width: int, height: int, n_max: int, grid: tuple[int, int]) -> Features:
    """At most `n_max` features, spread over a `grid` (cols, rows): each cell keeps its strongest
    ones up to an equal share, and the share left unused by empty cells goes to the strongest of
    the rest. Deterministic (stable sort on response)."""
    n = len(f)
    if n <= n_max:
        return f
    cols, rows = grid
    cx = np.clip((f.xy[:, 0] * cols / width).astype(int), 0, cols - 1)
    cy = np.clip((f.xy[:, 1] * rows / height).astype(int), 0, rows - 1)
    cell = cy * cols + cx
    order = np.argsort(-f.response, kind="stable")
    rank = np.empty(n, int)
    seen: dict[int, int] = {}
    for i in order:
        c = int(cell[i])
        rank[i] = seen.get(c, 0)
        seen[c] = rank[i] + 1
    share = max(1, n_max // (cols * rows))
    first = order[rank[order] < share]
    rest = order[rank[order] >= share]
    keep = np.concatenate([first, rest])[:n_max]
    return f.take(np.sort(keep))


def _from_cv(kps: Any, desc: Optional[np.ndarray], desc_bytes: int) -> Features:
    if desc is None or not kps:
        return Features(np.zeros((0, 2), np.float32), np.zeros((0, desc_bytes), np.uint8),
                        np.zeros(0, np.float32))
    xy = np.array([k.pt for k in kps], np.float32).reshape(-1, 2)
    resp = np.array([k.response for k in kps], np.float32)
    return Features(xy, np.ascontiguousarray(desc, dtype=np.uint8), resp)


def _gray(img: np.ndarray) -> np.ndarray:
    cv2 = require_cv2()
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        raise ValueError(f"expected an 8-bit image, got {img.dtype}")
    return img


class OrbExtractor:
    """ORB (FAST + rotated BRIEF, 32-byte descriptors). Over-detects 3x, then buckets."""

    name = "orb"
    norm = "hamming"
    DESC_BYTES = 32

    def __init__(self, n_max: int = 1000, scale_factor: float = 1.2, n_levels: int = 8,
                 fast_threshold: int = 12, grid: tuple[int, int] = (8, 6)):
        self.n_max, self.scale_factor, self.n_levels = int(n_max), float(scale_factor), int(n_levels)
        self.fast_threshold, self.grid = int(fast_threshold), (int(grid[0]), int(grid[1]))
        cv2 = require_cv2()
        self._orb = cv2.ORB_create(nfeatures=3 * self.n_max, scaleFactor=self.scale_factor,
                                   nlevels=self.n_levels, fastThreshold=self.fast_threshold)

    @property
    def params(self) -> dict[str, Any]:
        return {"n_max": self.n_max, "scale_factor": self.scale_factor, "n_levels": self.n_levels,
                "fast_threshold": self.fast_threshold, "grid": list(self.grid)}

    def detect(self, gray: np.ndarray) -> Features:
        gray = _gray(gray)
        kps, desc = self._orb.detectAndCompute(gray, None)
        f = _from_cv(kps, desc, self.DESC_BYTES)
        return bucket(f, gray.shape[1], gray.shape[0], self.n_max, self.grid)


class AkazeExtractor:
    """AKAZE (nonlinear scale space, 61-byte MLDB descriptors). Slower, more repeatable."""

    name = "akaze"
    norm = "hamming"
    DESC_BYTES = 61

    def __init__(self, n_max: int = 1000, threshold: float = 0.0008,
                 grid: tuple[int, int] = (8, 6)):
        self.n_max, self.threshold = int(n_max), float(threshold)
        self.grid = (int(grid[0]), int(grid[1]))
        cv2 = require_cv2()
        if not hasattr(cv2, "AKAZE_create"):
            raise RuntimeError(CV2_HINT)
        self._akaze = cv2.AKAZE_create(threshold=self.threshold)

    @property
    def params(self) -> dict[str, Any]:
        return {"n_max": self.n_max, "threshold": self.threshold, "grid": list(self.grid)}

    def detect(self, gray: np.ndarray) -> Features:
        gray = _gray(gray)
        kps, desc = self._akaze.detectAndCompute(gray, None)
        f = _from_cv(kps, desc, self.DESC_BYTES)
        return bucket(f, gray.shape[1], gray.shape[0], self.n_max, self.grid)


EXTRACTORS: dict[str, type] = {"orb": OrbExtractor, "akaze": AkazeExtractor}


def make_extractor(name: str, **params: Any) -> Extractor:
    if name not in EXTRACTORS:
        raise ValueError(f"unknown extractor '{name}' (known: {', '.join(EXTRACTORS)})")
    if "grid" in params and params["grid"] is not None:
        params["grid"] = tuple(params["grid"])
    return EXTRACTORS[name](**params)


def global_descriptor(gray: np.ndarray, size: tuple[int, int] = (32, 18)) -> np.ndarray:
    """A tiny whole-image descriptor for retrieval without a pose prior: the image shrunk to
    `size`, zero-mean, unit norm (float32). Weak by design (no invariance to yaw or light); the
    localizer uses it only when no prior is given, and the bench measures how weak it is."""
    cv2 = require_cv2()
    g = cv2.resize(_gray(gray), size, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    g -= g.mean()
    n = float(np.linalg.norm(g))
    return g / n if n > 0 else g
