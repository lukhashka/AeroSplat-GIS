"""Torch-free reader for a COLMAP sparse model (cameras + images, .bin or .txt).

`aerosplat.gs.trainer` has its own reader that builds torch Cameras for training;
this one returns plain numpy for the CPU-side consumers (semantics, coverage),
so they need neither torch nor a GPU.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]

# model id -> (name, number of parameters)
MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}
_ONE_FOCAL = {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL",
              "RADIAL_FISHEYE"}


@dataclass(frozen=True)
class Intrinsics:
    model: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_params(cls, model: str, width: int, height: int, params: list[float]
                    ) -> "Intrinsics":
        """Pinhole part of any COLMAP model; distortion is ignored."""
        if model in _ONE_FOCAL:
            f, cx, cy = params[0], params[1], params[2]
            return cls(model, width, height, f, f, cx, cy)
        if len(params) < 4:
            raise ValueError(f"camera model {model} needs 4+ parameters, got {len(params)}")
        return cls(model, width, height, params[0], params[1], params[2], params[3])


@dataclass(frozen=True)
class CameraPose:
    """World-to-camera rotation and translation: x_cam = R x_world + t."""

    name: str
    R: F64              # (3, 3)
    t: F64              # (3,)
    intrinsics: Intrinsics

    @property
    def centre(self) -> F64:
        return -self.R.T @ self.t

    def project(self, xyz) -> tuple[F64, F64]:
        """World points (..., 3) -> (pixels (..., 2), depth along the optical axis)."""
        p = np.asarray(xyz, float) @ self.R.T + self.t
        z = p[..., 2]
        k = self.intrinsics
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = np.stack([k.fx * p[..., 0] / z + k.cx, k.fy * p[..., 1] / z + k.cy], axis=-1)
        return uv, z

    def sees(self, xyz) -> NDArray[np.bool_]:
        """Whether world points fall in front of the camera and inside the image."""
        uv, z = self.project(xyz)
        k = self.intrinsics
        return ((z > 0) & (uv[..., 0] >= 0) & (uv[..., 0] < k.width)
                & (uv[..., 1] >= 0) & (uv[..., 1] < k.height))


def quat_to_rotmat(q) -> F64:
    """Single (w, x, y, z) unit quaternion -> 3x3 rotation."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _unpack(f, fmt: str):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def _read_cameras_bin(path: Path) -> dict[int, Intrinsics]:
    out: dict[int, Intrinsics] = {}
    with path.open("rb") as f:
        for _ in range(_unpack(f, "Q")[0]):
            cam_id, model_id, width, height = _unpack(f, "iiQQ")
            model, n = MODELS[model_id]
            out[cam_id] = Intrinsics.from_params(model, width, height,
                                                 list(_unpack(f, "d" * n)))
    return out


def _read_cameras_txt(path: Path) -> dict[int, Intrinsics]:
    out: dict[int, Intrinsics] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        out[int(p[0])] = Intrinsics.from_params(p[1], int(p[2]), int(p[3]),
                                                [float(v) for v in p[4:]])
    return out


def _read_images_bin(path: Path, cams: dict[int, Intrinsics]) -> list[tuple[int, CameraPose]]:
    out: list[tuple[int, CameraPose]] = []
    with path.open("rb") as f:
        for _ in range(_unpack(f, "Q")[0]):
            d = _unpack(f, "idddddddi")
            name = bytearray()
            while (ch := f.read(1)) != b"\x00":
                name += ch
            n_pts = _unpack(f, "Q")[0]
            f.seek(24 * n_pts, 1)                      # skip the 2-D observations
            out.append((d[0], CameraPose(name=name.decode(), R=quat_to_rotmat(d[1:5]),
                                         t=np.array(d[5:8], float), intrinsics=cams[d[8]])))
    return out


def _read_images_txt(path: Path, cams: dict[int, Intrinsics]) -> list[tuple[int, CameraPose]]:
    # Keep blank lines: an image with no 2-D observations has an empty second
    # line, and dropping it would shift every pose after it onto the wrong row.
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if not ln.lstrip().startswith("#")]
    out: list[tuple[int, CameraPose]] = []
    for i in range(0, len(lines), 2):                   # every second line is the observations
        p = lines[i].split()
        if len(p) < 10:
            continue
        out.append((int(p[0]), CameraPose(name=p[9], R=quat_to_rotmat([float(v) for v in p[1:5]]),
                                          t=np.array([float(v) for v in p[5:8]]),
                                          intrinsics=cams[int(p[8])])))
    return out


def read_model_ids(sparse_dir: Path | str) -> list[tuple[int, CameraPose]]:
    """(image_id, pose) of a COLMAP sparse model, in the order the images file stores them.

    Dense products number images by their position in the model (fused.ply.vis), so callers
    that need that numbering get the ids and the file order rather than a name sort.
    """
    d = Path(sparse_dir)
    if (d / "cameras.bin").is_file():
        cams = _read_cameras_bin(d / "cameras.bin")
    elif (d / "cameras.txt").is_file():
        cams = _read_cameras_txt(d / "cameras.txt")
    else:
        raise FileNotFoundError(f"no cameras.bin or cameras.txt in {d}")
    if (d / "images.bin").is_file():
        return _read_images_bin(d / "images.bin", cams)
    if (d / "images.txt").is_file():
        return _read_images_txt(d / "images.txt", cams)
    raise FileNotFoundError(f"no images.bin or images.txt in {d}")


def read_model(sparse_dir: Path | str) -> list[CameraPose]:
    """Camera poses of a COLMAP sparse model, sorted by image name."""
    return sorted((pose for _, pose in read_model_ids(sparse_dir)), key=lambda c: c.name)
