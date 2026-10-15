"""Earth Studio's spherical globe, its local frame around a Site origin ("sphere
ENU"), and the affine map from sphere ENU to WGS84 Site ENU.

Google Earth Studio renders the globe as a sphere: latitude/longitude are used
as spherical angles and altitude is added radially (radius EARTH_R + h). Its
renders are consistent with camera poses in that model and NOT with the same
coordinates in WGS84 (verified on dataset_v6, docs/ARCHITECTURE.md §4), so an
Earth Studio run reconstructs in sphere ENU and its products are mapped to
WGS84 Site ENU afterwards (ADR-0006, amended 2026-10-15).

Sphere ENU is built with ONE radius for the frames and for the origin, so a
constant error in that radius cancels. Its axes at the origin are the same as
WGS84 Site ENU's (same angles), and the origin maps to the origin; away from
it the two differ by the sphere/ellipsoid curvature, ~0.1-0.3 % horizontally
at mid latitudes. Over a site-sized box that difference is affine to well
below a millimetre (`fit_site_affine`), so Gaussians transform exactly:
mu -> A mu + b, Sigma -> A Sigma A^T (aerosplat.geo.affine).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aerosplat.geo.affine import AffineMap
from aerosplat.geo.crs import EnuFrame, GeoPoint, ecef_to_enu_rotation

F64 = NDArray[np.float64]

EARTH_R = 6371008.8  # IUGG mean radius, metres: Earth Studio's sphere (earthstudio_to_colmap)

# Run.earth_model (and geo_registration.json "earth_model") of a run
# reconstructed in sphere ENU whose products were mapped to WGS84 Site ENU by
# the affine below. Other values: "wgs84" (poses converted at ingest, S1) and
# "sphere-legacy" (pre-ADR-0006 frame from `position`, aerosplat.store.legacy).
EARTH_MODEL = "sphere-enu+affine"

# Default box for the affine fit, around the Site origin in sphere ENU:
# 1 km x 1 km horizontally, +-250 m vertically.
FIT_HALF_SIZE_M = 500.0
FIT_HALF_HEIGHT_M = 250.0
_FIT_STEPS = (11, 11, 5)


def sphere_to_ecef(lat: ArrayLike, lon: ArrayLike, h: ArrayLike, radius: float = EARTH_R) -> F64:
    """Geocentric xyz of lat/lon taken as spherical angles, at radius `radius` + h."""
    la = np.radians(np.asarray(lat, dtype=np.float64))
    lo = np.radians(np.asarray(lon, dtype=np.float64))
    r = radius + np.asarray(h, dtype=np.float64)
    return np.stack([r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo), r * np.sin(la)],
                    axis=-1)


def ecef_to_sphere(xyz: ArrayLike, radius: float = EARTH_R) -> tuple[F64, F64, F64]:
    """Inverse of sphere_to_ecef: (lat, lon, h)."""
    a = np.asarray(xyz, dtype=np.float64)
    r = np.linalg.norm(a, axis=-1)
    lat = np.degrees(np.arcsin(a[..., 2] / r))
    lon = np.degrees(np.arctan2(a[..., 1], a[..., 0]))
    return lat, lon, r - radius


class SphereEnuFrame:
    """East-north-up frame (metres) tangent to Earth Studio's sphere at `origin`.

    `origin` is the Site's WGS84 origin; its angles and height are read in the
    sphere model, exactly like Earth Studio's `coordinate`.
    """

    def __init__(self, origin: GeoPoint, radius: float = EARTH_R):
        self.origin = origin
        self.radius = radius
        self.rotation = ecef_to_enu_rotation(origin.lat, origin.lon)
        self.origin_ecef = sphere_to_ecef(origin.lat, origin.lon, origin.h, radius)

    def from_ecef(self, xyz: ArrayLike) -> F64:
        return (np.asarray(xyz, dtype=np.float64) - self.origin_ecef) @ self.rotation.T

    def to_ecef(self, enu: ArrayLike) -> F64:
        return np.asarray(enu, dtype=np.float64) @ self.rotation + self.origin_ecef

    def from_geodetic(self, lat: ArrayLike, lon: ArrayLike, h: ArrayLike) -> F64:
        return self.from_ecef(sphere_to_ecef(lat, lon, h, self.radius))

    def to_geodetic(self, enu: ArrayLike) -> tuple[F64, F64, F64]:
        return ecef_to_sphere(self.to_ecef(enu), self.radius)


def sphere_enu_to_site_enu(origin: GeoPoint, enu_sphere: ArrayLike) -> F64:
    """Exact map: a sphere-ENU point -> its lat/lon/h on the sphere -> the same
    lat/lon/h on WGS84 -> WGS84 Site ENU (pyproj)."""
    lat, lon, h = SphereEnuFrame(origin).to_geodetic(enu_sphere)
    return EnuFrame(origin).from_geodetic(lat, lon, h)


def fit_box(half_size_m: float = FIT_HALF_SIZE_M, half_height_m: float = FIT_HALF_HEIGHT_M
            ) -> F64:
    ne, nn, nu = _FIT_STEPS
    e = np.linspace(-half_size_m, half_size_m, ne)
    n = np.linspace(-half_size_m, half_size_m, nn)
    u = np.linspace(-half_height_m, half_height_m, nu)
    return np.stack(np.meshgrid(e, n, u, indexing="ij"), axis=-1).reshape(-1, 3)


def fit_site_affine(origin: GeoPoint, half_size_m: float = FIT_HALF_SIZE_M,
                    half_height_m: float = FIT_HALF_HEIGHT_M) -> tuple[AffineMap, dict[str, Any]]:
    """Least-squares affine sphere ENU -> WGS84 Site ENU over a box around the origin.

    Depends only on the Site origin and the box, so every run of a Site gets
    the same map. Returns (map, report block with the fit residuals).
    """
    src = fit_box(half_size_m, half_height_m)
    dst = sphere_enu_to_site_enu(origin, src)
    design = np.hstack([src, np.ones((len(src), 1))])
    sol, *_ = np.linalg.lstsq(design, dst, rcond=None)                # (4, 3)
    affine = AffineMap(A=sol[:3].T.copy(), b=sol[3].copy())
    res = np.linalg.norm(affine.apply(src) - dst, axis=1)
    block = {
        **affine.to_dict(),
        "fit": {"half_size_m": half_size_m, "half_height_m": half_height_m,
                "points": int(len(src)), "radius_m": EARTH_R},
        "residual_max_m": float(res.max()),
        "residual_rms_m": float(np.sqrt((res ** 2).mean())),
    }
    return affine, block
