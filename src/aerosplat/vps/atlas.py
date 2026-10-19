"""VPS atlas: keyframes of the twin with their features and the 3D points behind them.

One `.npz` (numpy only, no pickle) that the companion loads; built on the ground station by
`aerosplat.vps.build`. Arrays:

    kf_R        (K, 3, 3) float64   world -> camera rotation of each keyframe (OpenCV axes)
    kf_C        (K, 3)    float64   camera centre, Site ENU metres
    kf_look     (K, 3)    float64   where the optical axis meets the scene (median depth)
    kf_offsets  (K + 1,)  int64     keyframe k owns features [offsets[k], offsets[k + 1])
    kf_global   (K, G)    float16   whole-image descriptor (retrieval without a prior)
    xy          (N, 2)    float32   keypoint, OpenCV pixel coordinates of its keyframe
    desc        (N, D)    uint8     descriptor
    xyz         (N, 3)    float32   3D point relative to `anchor` (Site ENU = anchor + xyz)
    anchor      (3,)      float64

and `meta` (JSON): format, run / scene / splat hash, intrinsics, extractor and its parameters,
software versions, area, counts. `content_hash` (sha256 over every array and the metadata) is
stored with it and checked on load: the mission package names the atlas by this hash
(ADR-0015 §4: the twin is safety-critical, its version and integrity are audited).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import numpy as np

ATLAS_FORMAT = "aerosplat-vps-atlas/1"
ARRAYS = ("kf_R", "kf_C", "kf_look", "kf_offsets", "kf_global", "xy", "desc", "xyz", "anchor")


class AtlasError(ValueError):
    """An atlas that must not be used: wrong format, damaged, or inconsistent."""


def _canonical(meta: dict[str, Any]) -> bytes:
    return json.dumps(meta, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


@dataclass
class Atlas:
    kf_R: np.ndarray
    kf_C: np.ndarray
    kf_look: np.ndarray
    kf_offsets: np.ndarray
    kf_global: np.ndarray
    xy: np.ndarray
    desc: np.ndarray
    xyz: np.ndarray
    anchor: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    # ── shape ────────────────────────────────────────────────────────────
    @property
    def n_keyframes(self) -> int:
        return int(self.kf_C.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.xy.shape[0])

    def keyframe_slice(self, k: int) -> slice:
        return slice(int(self.kf_offsets[k]), int(self.kf_offsets[k + 1]))

    def points_enu(self, idx: Union[np.ndarray, slice]) -> np.ndarray:
        return self.xyz[idx].astype(np.float64) + self.anchor

    def nbytes(self) -> int:
        return int(sum(getattr(self, a).nbytes for a in ARRAYS))

    def validate(self) -> None:
        K, N = self.n_keyframes, self.n_features
        want = {"kf_R": (K, 3, 3), "kf_C": (K, 3), "kf_look": (K, 3), "kf_offsets": (K + 1,),
                "xy": (N, 2), "xyz": (N, 3), "anchor": (3,)}
        for name, shape in want.items():
            if getattr(self, name).shape != shape:
                raise AtlasError(f"atlas array {name} has shape {getattr(self, name).shape}, "
                                 f"expected {shape}")
        if self.desc.shape[0] != N or self.desc.dtype != np.uint8:
            raise AtlasError("atlas descriptors do not match the keypoints (or are not uint8)")
        if self.kf_global.shape[0] != K:
            raise AtlasError("atlas global descriptors do not match the keyframes")
        off = self.kf_offsets
        if K == 0 or off[0] != 0 or off[-1] != N or np.any(np.diff(off) < 0):
            raise AtlasError("atlas keyframe offsets are not a partition of the features")
        for name in ("kf_R", "kf_C", "kf_look", "xy", "xyz", "anchor"):
            if not np.all(np.isfinite(getattr(self, name))):
                raise AtlasError(f"atlas array {name} has non-finite values")
        if self.meta.get("format") != ATLAS_FORMAT:
            raise AtlasError(f"atlas format {self.meta.get('format')!r}, expected {ATLAS_FORMAT!r}")
        intr = self.meta.get("intrinsics") or {}
        if not {"width", "height", "fx", "fy", "cx", "cy"} <= set(intr):
            raise AtlasError("atlas metadata has no intrinsics")
        ex = self.meta.get("extractor") or {}
        if "name" not in ex or "params" not in ex:
            raise AtlasError("atlas metadata does not name its extractor")

    # ── integrity ────────────────────────────────────────────────────────
    def content_hash(self) -> str:
        h = hashlib.sha256()
        for name in ARRAYS:
            a = np.ascontiguousarray(getattr(self, name))
            h.update(name.encode())
            h.update(str(a.dtype).encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        h.update(_canonical({k: v for k, v in self.meta.items() if k != "content_hash"}))
        return h.hexdigest()

    # ── io ───────────────────────────────────────────────────────────────
    def save(self, path: Path) -> str:
        """Write the atlas; returns its content hash (also stored in the file)."""
        self.validate()
        meta = {k: v for k, v in self.meta.items() if k != "content_hash"}
        self.meta = meta
        digest = self.content_hash()
        self.meta = {**meta, "content_hash": digest}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(tmp, meta=np.frombuffer(_canonical(self.meta), np.uint8),
                            **{name: getattr(self, name) for name in ARRAYS})
        tmp.replace(path)
        return digest

    @classmethod
    def load(cls, path: Path, *, expect_hash: str | None = None) -> "Atlas":
        """Read and verify. Refuses (AtlasError) a damaged file, a hash that does not match its
        content, or one that is not `expect_hash` (the hash the mission package names)."""
        try:
            with np.load(Path(path), allow_pickle=False) as z:
                missing = set(ARRAYS + ("meta",)) - set(z.files)
                if missing:
                    raise AtlasError(f"{path}: not an atlas (missing {sorted(missing)})")
                arrays = {name: z[name] for name in ARRAYS}
                meta = json.loads(z["meta"].tobytes().decode())
        except (OSError, ValueError, json.JSONDecodeError) as e:
            if isinstance(e, AtlasError):
                raise
            raise AtlasError(f"{path}: cannot read atlas ({e})") from None
        atlas = cls(**arrays, meta=meta)
        atlas.validate()
        stored = meta.get("content_hash")
        actual = atlas.content_hash()
        if stored != actual:
            raise AtlasError(f"{path}: content hash {actual[:12]} does not match the stored "
                             f"{str(stored)[:12]} (damaged or edited)")
        if expect_hash is not None and actual != expect_hash:
            raise AtlasError(f"{path}: atlas {actual[:12]} is not the expected {expect_hash[:12]}")
        return atlas


def concat_keyframes(kfs: list[dict[str, np.ndarray]], anchor: np.ndarray, meta: dict[str, Any]
                     ) -> Atlas:
    """Assemble an atlas from per-keyframe dicts: R, C, look, global, xy, desc, xyz_enu."""
    if not kfs:
        raise AtlasError("no keyframes")
    counts = [len(k["xy"]) for k in kfs]
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    d = kfs[0]["desc"].shape[1]
    return Atlas(
        kf_R=np.stack([k["R"] for k in kfs]).astype(np.float64),
        kf_C=np.stack([k["C"] for k in kfs]).astype(np.float64),
        kf_look=np.stack([k["look"] for k in kfs]).astype(np.float64),
        kf_offsets=offsets,
        kf_global=np.stack([k["global"] for k in kfs]).astype(np.float16),
        xy=np.concatenate([k["xy"] for k in kfs]).astype(np.float32).reshape(-1, 2),
        desc=np.concatenate([k["desc"].reshape(-1, d) for k in kfs]).astype(np.uint8),
        xyz=(np.concatenate([k["xyz_enu"] for k in kfs]).reshape(-1, 3) - anchor).astype(np.float32),
        anchor=np.asarray(anchor, np.float64).reshape(3),
        meta={"format": ATLAS_FORMAT, **meta, "keyframes": len(kfs), "features": int(offsets[-1])},
    )
