"""Real spherical harmonics in the Inria / gsplat convention (degree <= 3).

A Gaussian's colour seen along the unit direction d (camera centre -> mean) is

    colour(d) = max(0, 0.5 + sum_k sh[k] * Y_k(d))          sh: (K, 3), K = (degree + 1)^2

Y_0 = C0 is constant, so the view-independent part is rgb = 0.5 + C0 * sh[0]
(`sh_dc_to_rgb`): that is the `rgb` a gsplat-trained .pt exports for the
server and the viewer, which do not evaluate SH (ADR-0002, amended 2026-10-15).

`rotate_sh` re-expresses the coefficients in a rotated frame; the georef stage
needs it because a frame change rotates every viewing direction too.
The scalar helpers work on numpy arrays and torch tensors alike.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]

MAX_SH_DEGREE = 3
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
      -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
      -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def num_sh_bases(degree: int) -> int:
    return (degree + 1) ** 2


def sh_degree_of(n_bases: int) -> int:
    """K = (degree + 1)^2 -> degree; anything else is not an SH coefficient count."""
    degree = int(round(n_bases ** 0.5)) - 1
    if degree < 0 or num_sh_bases(degree) != n_bases or degree > MAX_SH_DEGREE:
        raise ValueError(f"{n_bases} SH coefficients do not form a degree 0..{MAX_SH_DEGREE} basis")
    return degree


def sh_dc_to_rgb(dc: Any) -> Any:
    """DC coefficient(s) -> view-independent colour in [0, 1] (numpy or torch)."""
    return (0.5 + C0 * dc).clip(0.0, 1.0)


def rgb_to_sh_dc(rgb: Any) -> Any:
    """Inverse of `sh_dc_to_rgb` for colours inside [0, 1]."""
    return (rgb - 0.5) / C0


def eval_sh_basis(degree: int, dirs: Any) -> F64:
    """(N, 3) unit directions -> (N, K) basis values Y_k(d), gsplat/Inria sign convention."""
    d: F64 = np.asarray(dirs, dtype=np.float64)
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    cols = [np.full_like(x, C0)]
    if degree > 0:
        cols += [-C1 * y, C1 * z, -C1 * x]
    if degree > 1:
        xx, yy, zz = x * x, y * y, z * z
        cols += [C2[0] * x * y, C2[1] * y * z, C2[2] * (2 * zz - xx - yy),
                 C2[3] * x * z, C2[4] * (xx - yy)]
    if degree > 2:
        cols += [C3[0] * y * (3 * xx - yy), C3[1] * x * y * z, C3[2] * y * (4 * zz - xx - yy),
                 C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy),
                 C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy)]
    return np.stack(cols, axis=1)


def eval_sh(sh: Any, dirs: Any) -> F64:
    """(N, K, 3) coefficients at (N, 3) unit directions -> (N, 3) colour (unclamped, + 0.5)."""
    c: F64 = np.asarray(sh, dtype=np.float64)
    Y = eval_sh_basis(sh_degree_of(c.shape[1]), dirs)
    return 0.5 + np.einsum("nk,nkc->nc", Y, c)


def _fibonacci_sphere(n: int) -> F64:
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    phi = np.pi * (3 - np.sqrt(5)) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def sh_rotation_matrix(R: Any, degree: int) -> F64:
    """(K, K) block-diagonal M with Y(R^T d) = Y(d) @ M for every unit d.

    Each band l is closed under rotation, so its block is the exact least-squares
    solution on a set of sample directions (no Wigner-D bookkeeping, and it
    follows whatever sign convention `eval_sh_basis` uses).
    """
    Rm: F64 = np.asarray(R, dtype=np.float64)
    dirs = _fibonacci_sphere(64)
    Y = eval_sh_basis(degree, dirs)
    Y_rot = eval_sh_basis(degree, dirs @ Rm)        # rows are (R^T d)^T = d^T R
    K = num_sh_bases(degree)
    M = np.zeros((K, K))
    for band in range(degree + 1):
        sl = slice(band * band, (band + 1) ** 2)
        M[sl, sl] = np.linalg.lstsq(Y[:, sl], Y_rot[:, sl], rcond=None)[0]
    return M


def rotate_sh(sh: Any, R: Any) -> F64:
    """Coefficients (N, K, 3) of colour(d) -> those of colour(R^T d).

    For a frame change in which directions map as d' = R d, the returned
    coefficients give the same colour along d' as the input does along d.
    """
    c: F64 = np.asarray(sh, dtype=np.float64)
    M = sh_rotation_matrix(R, sh_degree_of(c.shape[1]))
    return np.einsum("jk,nkc->njc", M, c)
