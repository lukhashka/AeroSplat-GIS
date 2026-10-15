"""3-D affine maps x -> A x + b, applied to points and to 3D Gaussians.

A Gaussian with mean mu and covariance Sigma = R S^2 R^T maps exactly to mean
A mu + b and covariance A Sigma A^T; the new scales and rotation are re-derived
from the SVD of A R S. Used to carry Earth Studio reconstructions from sphere
ENU into WGS84 Site ENU (aerosplat.geo.sphere_enu, aerosplat.geo.georef).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

F64 = NDArray[np.float64]

# Gaussians per SVD batch: bounds peak memory (~10 (N,3,3) float64 arrays).
_CHUNK = 262_144
_PERMS = np.array(list(permutations(range(3))))      # (6, 3)


def quat_to_rotmat(q: ArrayLike) -> F64:
    """(N, 4) unit quaternions (w, x, y, z) -> (N, 3, 3) rotation matrices."""
    qa: F64 = np.asarray(q, dtype=np.float64)
    qa = qa / np.linalg.norm(qa, axis=-1, keepdims=True)
    w, x, y, z = qa[:, 0], qa[:, 1], qa[:, 2], qa[:, 3]
    R = np.empty((qa.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rotmat_to_quat(R: ArrayLike) -> F64:
    """(N, 3, 3) rotation matrices -> (N, 4) unit quaternions (w, x, y, z), w >= 0."""
    Ra: F64 = np.asarray(R, dtype=np.float64)
    m = {(i, j): Ra[:, i, j] for i in range(3) for j in range(3)}
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    # Row k is 4 q_k * q; use the k with the largest |q_k| for stability.
    cand = np.stack([
        np.stack([1 + tr, m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]], -1),
        np.stack([m[2, 1] - m[1, 2], 1 + m[0, 0] - m[1, 1] - m[2, 2], m[0, 1] + m[1, 0],
                  m[0, 2] + m[2, 0]], -1),
        np.stack([m[0, 2] - m[2, 0], m[0, 1] + m[1, 0], 1 - m[0, 0] + m[1, 1] - m[2, 2],
                  m[1, 2] + m[2, 1]], -1),
        np.stack([m[1, 0] - m[0, 1], m[0, 2] + m[2, 0], m[1, 2] + m[2, 1],
                  1 - m[0, 0] - m[1, 1] + m[2, 2]], -1),
    ], axis=1)                                                     # (N, 4, 4)
    k = np.argmax(np.stack([cand[:, i, i] for i in range(4)], -1), axis=1)
    q = cand[np.arange(Ra.shape[0]), k]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return np.where(q[:, :1] < 0, -q, q)


@dataclass(frozen=True)
class AffineMap:
    A: F64          # (3, 3)
    b: F64          # (3,)

    @classmethod
    def identity(cls) -> "AffineMap":
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AffineMap":
        A, b = np.asarray(d["A"], dtype=np.float64), np.asarray(d["b"], dtype=np.float64)
        if A.shape != (3, 3) or b.shape != (3,):
            raise ValueError(f"affine map needs A (3,3) and b (3,), got {A.shape}, {b.shape}")
        if not np.linalg.det(A) > 0:
            raise ValueError("affine map must preserve orientation (det A > 0)")
        return cls(A, b)

    def to_dict(self) -> dict[str, Any]:
        return {"A": self.A.tolist(), "b": self.b.tolist()}

    def apply(self, xyz: ArrayLike) -> F64:
        """Points (..., 3) -> A x + b."""
        return np.asarray(xyz, dtype=np.float64) @ self.A.T + self.b

    def apply_linear(self, v: ArrayLike) -> F64:
        """Vectors / offsets (..., 3) -> A v (no translation)."""
        return np.asarray(v, dtype=np.float64) @ self.A.T

    def rotation(self) -> F64:
        """Rotation part of A (polar decomposition A = R P, P symmetric positive definite).

        Directions map as d -> A d / |A d|; for a similarity that is exactly R d,
        otherwise R is the closest rotation (the stretch P only bends directions).
        """
        U, _, Vt = np.linalg.svd(self.A)
        return U @ Vt                  # det = +1: from_dict requires det A > 0

    def rotation_deg(self) -> float:
        R = self.rotation()
        return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))

    def apply_gaussians(self, xyz: ArrayLike, scale: ArrayLike, quat: ArrayLike
                        ) -> tuple[F64, F64, F64]:
        """Means, per-axis scales (N, 3, > 0) and rotations (N, 4, w x y z) of 3D Gaussians.

        Returns (A mu + b, scales', rotations') with R' S'^2 R'^T = A R S^2 R^T A^T.
        Axis order and signs follow the input wherever the Gaussian is anisotropic,
        so an identity map returns the input unchanged.
        """
        s: F64 = np.asarray(scale, dtype=np.float64)
        q: F64 = np.asarray(quat, dtype=np.float64)
        n = s.shape[0]
        out_s, out_q = np.empty((n, 3)), np.empty((n, 4))
        for lo in range(0, n, _CHUNK):
            sl = slice(lo, min(lo + _CHUNK, n))
            out_s[sl], out_q[sl] = self._gaussian_axes(s[sl], q[sl])
        return self.apply(xyz), out_s, out_q

    def _gaussian_axes(self, scale: F64, quat: F64) -> tuple[F64, F64]:
        M = (self.A @ quat_to_rotmat(quat)) * scale[:, None, :]       # A R S
        U, sv, Vt = np.linalg.svd(M)                                    # M M^T = U sv^2 U^T
        # |V[i, j]| ~ which input axis i became output axis j; pick the best
        # one-to-one assignment (6 permutations) per Gaussian.
        V = np.abs(np.swapaxes(Vt, 1, 2))
        score = V[:, np.arange(3), _PERMS].sum(axis=2)                  # (N, 6)
        perm = _PERMS[np.argmax(score, axis=1)]                         # (N, 3)
        rows = np.arange(len(sv))[:, None]
        new_s = sv[rows, perm]
        sign = np.sign(np.swapaxes(Vt, 1, 2)[rows, np.arange(3), perm])
        sign[sign == 0] = 1.0
        new_R = np.take_along_axis(U, perm[:, None, :], axis=2) * sign[:, None, :]
        # A column sign flip leaves the covariance unchanged; keep det = +1.
        flip = np.linalg.det(new_R) < 0
        new_R[flip, :, 2] *= -1
        return new_s, rotmat_to_quat(new_R)
