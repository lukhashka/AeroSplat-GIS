"""Camera maths for the render service. numpy only, no torch.

World = Site ENU (x east, y north, z up, metres). Camera = COLMAP/OpenCV
(x right, y down, z forward), `x_cam = R·x_world + t`.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

F64 = np.ndarray


def look_direction(yaw_deg: float, pitch_deg: float) -> F64:
    """Unit forward vector in ENU from compass heading and elevation."""
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    return np.array([math.sin(yaw) * math.cos(pitch),
                     math.cos(yaw) * math.cos(pitch),
                     math.sin(pitch)])


def yaw_pitch_to(eye: F64, target: F64) -> tuple[float, float]:
    """Compass heading and elevation of the ray eye -> target."""
    d = np.asarray(target, float) - np.asarray(eye, float)
    horiz = math.hypot(d[0], d[1])
    if horiz < 1e-9 and abs(d[2]) < 1e-9:
        raise ValueError("look_at coincides with the camera position")
    return math.degrees(math.atan2(d[0], d[1])), math.degrees(math.atan2(d[2], horiz))


def view_rotation(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> F64:
    """World -> camera rotation R (3x3). Rows are the camera's right, down, forward axes.

    The right axis is horizontal for any pitch, so looking straight down is not a
    singularity: yaw then says which way is "up" in the image (0 = north-up).
    """
    f = look_direction(yaw_deg, pitch_deg)
    yaw = math.radians(yaw_deg)
    r = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
    d = np.cross(f, r)
    roll = math.radians(roll_deg)
    r, d = (r * math.cos(roll) + d * math.sin(roll),
            -r * math.sin(roll) + d * math.cos(roll))
    return np.stack([r, d, f])


def view_matrix(position_enu: F64, yaw_deg: float, pitch_deg: float,
                roll_deg: float = 0.0) -> F64:
    """World -> camera 4x4 for a camera at `position_enu`."""
    R = view_rotation(yaw_deg, pitch_deg, roll_deg)
    V = np.eye(4)
    V[:3, :3] = R
    V[:3, 3] = -R @ np.asarray(position_enu, float)
    return V


def persp_intrinsics(width: int, height: int, fov_deg: float) -> tuple[float, float, float, float]:
    """(fx, fy, cx, cy) of a pinhole with the given vertical field of view and square pixels."""
    f = 0.5 * height / math.tan(math.radians(fov_deg) / 2)
    return f, f, width / 2, height / 2


def ortho_intrinsics(width: int, height: int, extent_m: float) -> tuple[float, float, float, float]:
    """Orthographic "intrinsics" as gsplat reads them: pixels per metre, principal point."""
    s = width / extent_m
    return s, s, width / 2, height / 2


def camera_centre(V: F64) -> F64:
    return -V[:3, :3].T @ V[:3, 3]


def project(V: F64, K: tuple[float, float, float, float], points: F64, *,
            ortho: bool = False) -> tuple[F64, F64]:
    """Pixels (N, 2) and camera-space depth (N,) of world points. Reference for tests."""
    fx, fy, cx, cy = K
    pc = np.asarray(points, float) @ V[:3, :3].T + V[:3, 3]
    z = pc[:, 2]
    if ortho:
        return np.stack([fx * pc[:, 0] + cx, fy * pc[:, 1] + cy], axis=1), z
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([fx * pc[:, 0] / z + cx, fy * pc[:, 1] / z + cy], axis=1), z


def ortho_ground_grid(V: F64, K: tuple[float, float, float, float], width: int, height: int
                      ) -> Optional[dict]:
    """Georeference of a straight-down orthographic render (north-up unless yawed).

    Returns the image-to-ENU affine (east, north of a pixel centre) and the metres per
    pixel; None when the camera is not looking straight down (the image is then a slanted
    parallel projection and has no single ground grid).
    """
    R = V[:3, :3]
    if abs(R[2, 2]) < 1.0 - 1e-6 or R[2, 2] > 0:         # forward axis must be (0, 0, -1)
        return None
    fx, fy, cx, cy = K
    C = camera_centre(V)
    # pixel (u, v) -> camera (xc, yc) = ((u - cx) / fx, (v - cy) / fy); ENU = C + Rᵀ·(xc, yc, 0)
    Rt = R.T
    ex = Rt[:2, 0] / fx            # ENU shift per +1 px in u
    ey = Rt[:2, 1] / fy            # ENU shift per +1 px in v
    origin = C[:2] + Rt[:2, 0] * (0.5 - cx) / fx + Rt[:2, 1] * (0.5 - cy) / fy   # centre of pixel (0,0)
    return {"m_per_px": 1.0 / fx, "pixel00_enu": [float(origin[0]), float(origin[1])],
            "du_enu": [float(ex[0]), float(ex[1])], "dv_enu": [float(ey[0]), float(ey[1])]}
