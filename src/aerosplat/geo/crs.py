"""WGS84 <-> ECEF <-> local ENU conversions (pyproj).

The system's only earth model is WGS84: geodetic latitude/longitude in
degrees and ellipsoidal height in metres (EPSG:4979). Earth-centred
coordinates are EPSG:4978. A Site's metric frame is a topocentric
east-north-up frame tangent to the ellipsoid at an origin that is fixed when
the Site is created (ADR-0006, TARGET_ARCHITECTURE §5).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field
from pyproj import Transformer

WGS84_GEODETIC = "EPSG:4979"   # lat, lon (deg), ellipsoidal height (m)
WGS84_ECEF = "EPSG:4978"

F64 = NDArray[np.float64]


class GeoPoint(BaseModel):
    """A WGS84 position: degrees and ellipsoidal height in metres."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    h: float = 0.0

    @classmethod
    def parse(cls, text: str) -> "GeoPoint":
        """'LAT,LON[,H]' -> GeoPoint (CLI helper)."""
        parts = [p.strip() for p in text.split(",")]
        if len(parts) not in (2, 3):
            raise ValueError(f"expected LAT,LON[,H], got {text!r}")
        vals = [float(p) for p in parts]
        return cls(lat=vals[0], lon=vals[1], h=vals[2] if len(vals) == 3 else 0.0)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.lat, self.lon, self.h)


@lru_cache(maxsize=1)
def _geodetic_ecef() -> Transformer:
    return Transformer.from_crs(WGS84_GEODETIC, WGS84_ECEF, always_xy=True)


@lru_cache(maxsize=64)
def _topocentric(lat0: float, lon0: float, h0: float) -> Transformer:
    # ECEF -> ENU on the WGS84 ellipsoid; the inverse direction is ENU -> ECEF.
    return Transformer.from_pipeline(
        f"+proj=topocentric +ellps=WGS84 +lat_0={lat0!r} +lon_0={lon0!r} +h_0={h0!r}")


def _cols(xyz: ArrayLike) -> F64:
    a = np.asarray(xyz, dtype=np.float64)
    if a.shape[-1] != 3:
        raise ValueError(f"expected (..., 3) coordinates, got shape {a.shape}")
    return a


def geodetic_to_ecef(lat: ArrayLike, lon: ArrayLike, h: ArrayLike) -> F64:
    """WGS84 geodetic -> ECEF metres, shape (..., 3)."""
    x, y, z = _geodetic_ecef().transform(np.asarray(lon, dtype=np.float64),
                                         np.asarray(lat, dtype=np.float64),
                                         np.asarray(h, dtype=np.float64))
    return np.stack([np.asarray(x), np.asarray(y), np.asarray(z)], axis=-1)


def ecef_to_geodetic(xyz: ArrayLike) -> tuple[F64, F64, F64]:
    """ECEF metres (..., 3) -> (lat, lon, h)."""
    a = _cols(xyz)
    lon, lat, h = _geodetic_ecef().transform(a[..., 0], a[..., 1], a[..., 2],
                                             direction="INVERSE")
    return np.asarray(lat), np.asarray(lon), np.asarray(h)


def ecef_to_enu_rotation(lat_deg: float, lon_deg: float) -> F64:
    """Rotation taking ECEF direction vectors into the local ENU basis at (lat, lon).

    Rows are the east, north and up unit vectors. The same basis is valid for
    the ellipsoid (geodetic latitude) and for a sphere (spherical latitude) at
    equal angles, which is why Earth Studio orientations carry over unchanged.
    """
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return np.array([
        [-so, co, 0.0],
        [-sl * co, -sl * so, cl],
        [cl * co, cl * so, sl],
    ])


def utm_epsg(lat: float, lon: float) -> int:
    """EPSG code of the WGS84 / UTM zone that contains (lat, lon): 326zz north, 327zz south.

    The 6° zones, with the two standard exceptions (the EPSG areas of use do not encode
    them): south-west Norway is zone 32V, and Svalbard uses only zones 31/33/35/37.
    """
    zone = min(60, int((lon + 180.0) // 6.0) + 1)
    if 56.0 <= lat < 64.0 and 3.0 <= lon < 12.0:
        zone = 32
    elif 72.0 <= lat < 84.0 and 0.0 <= lon < 42.0:
        zone = 31 if lon < 9.0 else 33 if lon < 21.0 else 35 if lon < 33.0 else 37
    return (32600 if lat >= 0 else 32700) + zone


class SiteProjection:
    """Site ENU -> the Site's projected CRS (WGS84 / UTM of its origin), heights ellipsoidal.

    For exchange formats that want a projected CRS (CityJSON, §2.6). ENU is not simply
    shifted: the conversion goes through the ellipsoid, so grid convergence and the UTM
    scale factor are applied exactly. The height stays WGS84 ellipsoidal — there is no
    geoid model in the system, so nothing pretends to be AMSL.
    """

    def __init__(self, origin: GeoPoint):
        self.origin = origin
        self.epsg = utm_epsg(origin.lat, origin.lon)
        self.frame = EnuFrame(origin)
        self._tf = Transformer.from_crs(WGS84_GEODETIC, f"EPSG:{self.epsg}", always_xy=True)

    @property
    def crs_url(self) -> str:
        return f"https://www.opengis.net/def/crs/EPSG/0/{self.epsg}"

    def from_enu(self, enu: ArrayLike) -> F64:
        """(..., 3) ENU metres -> (..., 3) easting, northing (m), ellipsoidal height (m)."""
        lat, lon, h = self.frame.to_geodetic(_cols(enu))
        e, n = self._tf.transform(np.asarray(lon), np.asarray(lat))
        return np.stack([np.asarray(e), np.asarray(n), np.asarray(h)], axis=-1)


class EnuFrame:
    """Local east-north-up frame (metres) tangent to WGS84 at a fixed origin."""

    def __init__(self, origin: GeoPoint):
        self.origin = origin
        self.rotation = ecef_to_enu_rotation(origin.lat, origin.lon)
        self.origin_ecef = geodetic_to_ecef(origin.lat, origin.lon, origin.h)
        self._tf = _topocentric(origin.lat, origin.lon, origin.h)

    def from_ecef(self, xyz: ArrayLike) -> F64:
        a = _cols(xyz)
        e, n, u = self._tf.transform(a[..., 0], a[..., 1], a[..., 2])
        return np.stack([np.asarray(e), np.asarray(n), np.asarray(u)], axis=-1)

    def to_ecef(self, enu: ArrayLike) -> F64:
        a = _cols(enu)
        x, y, z = self._tf.transform(a[..., 0], a[..., 1], a[..., 2], direction="INVERSE")
        return np.stack([np.asarray(x), np.asarray(y), np.asarray(z)], axis=-1)

    def from_geodetic(self, lat: ArrayLike, lon: ArrayLike, h: ArrayLike) -> F64:
        return self.from_ecef(geodetic_to_ecef(lat, lon, h))

    def to_geodetic(self, enu: ArrayLike) -> tuple[F64, F64, F64]:
        return ecef_to_geodetic(self.to_ecef(enu))
