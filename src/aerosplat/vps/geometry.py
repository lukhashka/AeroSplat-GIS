"""Pose helpers shared by the atlas builder, the localizer and the bench. numpy only.

World = Site ENU (x east, y north, z up). Camera = OpenCV (x right, y down, z forward),
`x_cam = R (x_world - C)`. Angles as in the render service: yaw = compass heading of the optical
axis, pitch above the horizon (-90 = down), roll about the axis (positive = right side down).
Intrinsics are OpenCV's (centre of pixel (0, 0) at (0, 0)); the render service's principal point
is half a pixel further: `cv_intrinsics` converts.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from aerosplat.render.camera import view_rotation

Intrinsics = tuple[float, float, float, float]        # fx, fy, cx, cy (OpenCV convention)


def cv_intrinsics(render_K: Intrinsics) -> Intrinsics:
    """Render-service (fx, fy, cx, cy) -> OpenCV: pixel centres move from k + 0.5 to k."""
    fx, fy, cx, cy = render_K
    return fx, fy, cx - 0.5, cy - 0.5


def K_matrix(K: Intrinsics) -> np.ndarray:
    fx, fy, cx, cy = K
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def rotation(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    return view_rotation(yaw_deg, pitch_deg, roll_deg)


def angles(R: np.ndarray) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees of a world -> camera rotation (inverse of `rotation`)."""
    f = R[2]
    yaw = math.degrees(math.atan2(f[0], f[1]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, float(f[2])))))
    y = math.radians(yaw)
    r0 = np.array([math.cos(y), -math.sin(y), 0.0])
    d0 = np.cross(f, r0)
    roll = math.degrees(math.atan2(float(R[0] @ d0), float(R[0] @ r0)))
    return yaw, pitch, roll


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


# ── the camera on the vehicle ────────────────────────────────────────────
# Vehicle attitude: aerospace 3-2-1 Euler angles of the body (x forward, y right, z down)
# relative to NED, in radians, as MAVLink ATTITUDE / SIM_STATE give them. A camera mount: pitch
# above the body's horizontal plane (-90 = straight down) and yaw relative to the nose
# (positive = right), as in a vehicle profile (aerosplat.sim.airframe.CameraMount).
_ENU_FROM_NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def body_to_ned(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body (FRD) -> NED rotation of 3-2-1 Euler angles (radians)."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                              math.cos(yaw), math.sin(yaw))
    return np.array([[cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
                     [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
                     [-sp, sr * cp, cr * cp]])


def mount_axes(mount_pitch_deg: float, mount_yaw_deg: float = 0.0) -> np.ndarray:
    """Body (FRD) -> camera (OpenCV) rotation: rows are the camera's right, down and optical
    axes in the body frame."""
    p, y = math.radians(mount_pitch_deg), math.radians(mount_yaw_deg)
    fwd = np.array([math.cos(y), math.sin(y), 0.0])
    right = np.array([-math.sin(y), math.cos(y), 0.0])
    z = math.cos(p) * fwd - math.sin(p) * np.array([0.0, 0.0, 1.0])
    return np.stack([right, np.cross(z, right), z])


def camera_rotation(roll: float, pitch: float, yaw: float, mount_pitch_deg: float,
                    mount_yaw_deg: float = 0.0) -> np.ndarray:
    """World (Site ENU) -> camera rotation of a camera on a vehicle with this attitude."""
    R_enu_from_body = _ENU_FROM_NED @ body_to_ned(roll, pitch, yaw)
    return mount_axes(mount_pitch_deg, mount_yaw_deg) @ R_enu_from_body.T


def body_attitude(R_cam: np.ndarray, mount_pitch_deg: float, mount_yaw_deg: float = 0.0
                  ) -> tuple[float, float, float]:
    """(roll, pitch, yaw) in radians (3-2-1, NED) of the vehicle whose camera has the world ->
    camera rotation `R_cam` — the inverse of `camera_rotation`."""
    R_body_from_enu = mount_axes(mount_pitch_deg, mount_yaw_deg).T @ np.asarray(R_cam, float)
    Rbn = _ENU_FROM_NED.T @ R_body_from_enu.T          # body -> NED (_ENU_FROM_NED is its own inverse)
    yaw = math.atan2(Rbn[1, 0], Rbn[0, 0])
    pitch = -math.asin(max(-1.0, min(1.0, float(Rbn[2, 0]))))
    roll = math.atan2(Rbn[2, 1], Rbn[2, 2])
    return roll, pitch, yaw


def project(R: np.ndarray, C: np.ndarray, K: Intrinsics, X: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
    """Pixels (N, 2) and depths (N,) of world points X (N, 3)."""
    fx, fy, cx, cy = K
    pc = (np.asarray(X, float) - C) @ R.T
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([fx * pc[:, 0] / z + cx, fy * pc[:, 1] / z + cy], axis=1)
    return uv, z


def backproject(R: np.ndarray, C: np.ndarray, K: Intrinsics, uv: np.ndarray, depth: np.ndarray
                ) -> np.ndarray:
    """World points of pixels `uv` (N, 2) at z-depth `depth` (N,) along the optical axis."""
    fx, fy, cx, cy = K
    uv = np.asarray(uv, float)
    z = np.asarray(depth, float)
    pc = np.stack([(uv[:, 0] - cx) / fx * z, (uv[:, 1] - cy) / fy * z, z], axis=1)
    return pc @ R + C


def look_point(R: np.ndarray, C: np.ndarray, ground_z: float, max_range_m: float = 500.0
               ) -> np.ndarray:
    """Where the optical axis meets the plane z = ground_z (or `max_range_m` along it)."""
    f = R[2]
    if f[2] < -1e-6:
        t = (ground_z - C[2]) / f[2]
        if 0 < t <= max_range_m:
            return C + t * f
    return C + max_range_m * f


def rotvec_to_R(rv: np.ndarray) -> np.ndarray:
    """Rodrigues, numpy (no cv2 needed by callers that only do geometry)."""
    rv = np.asarray(rv, float).reshape(3)
    th = float(np.linalg.norm(rv))
    if th < 1e-12:
        k = np.array([[0, -rv[2], rv[1]], [rv[2], 0, -rv[0]], [-rv[1], rv[0], 0]])
        return np.eye(3) + k
    k = rv / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(c)))))


def pose_error(R_est: np.ndarray, C_est: np.ndarray, R_true: np.ndarray, C_true: np.ndarray,
               ) -> dict[str, float]:
    """Errors of an estimate: 3D and horizontal position (m), heading and full rotation (deg)."""
    d = np.asarray(C_est, float) - np.asarray(C_true, float)
    return {"pos_m": float(np.linalg.norm(d)), "horiz_m": float(np.hypot(d[0], d[1])),
            "vert_m": float(abs(d[2])),
            "yaw_deg": abs(wrap_deg(angles(R_est)[0] - angles(R_true)[0])),
            "rot_deg": rotation_angle_deg(R_est, R_true)}


def percentile(values: list[float], q: float) -> Optional[float]:
    return float(np.percentile(values, q)) if values else None
