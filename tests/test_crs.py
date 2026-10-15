"""WGS84 <-> ECEF <-> ENU (aerosplat.geo.crs), checked against closed-form WGS84 formulas."""

import numpy as np
import pytest
from pydantic import ValidationError

from aerosplat.geo.crs import (EnuFrame, GeoPoint, ecef_to_enu_rotation, ecef_to_geodetic,
                               geodetic_to_ecef)

A = 6378137.0
F = 1 / 298.257223563
E2 = F * (2 - F)


def _ecef_closed_form(lat, lon, h):
    la, lo = np.radians(lat), np.radians(lon)
    n = A / np.sqrt(1 - E2 * np.sin(la) ** 2)
    return np.stack([(n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo),
                     (n * (1 - E2) + h) * np.sin(la)], axis=-1)


@pytest.fixture
def pts():
    rng = np.random.default_rng(7)
    return rng.uniform(-89, 89, 200), rng.uniform(-180, 180, 200), rng.uniform(-100, 9000, 200)


def test_geodetic_to_ecef_matches_closed_form(pts):
    lat, lon, h = pts
    np.testing.assert_allclose(geodetic_to_ecef(lat, lon, h), _ecef_closed_form(lat, lon, h),
                               rtol=0, atol=1e-6)


def test_wgs84_ecef_round_trip(pts):
    lat, lon, h = pts
    la, lo, hh = ecef_to_geodetic(geodetic_to_ecef(lat, lon, h))
    np.testing.assert_allclose(la, lat, atol=1e-10)
    np.testing.assert_allclose(lo, lon, atol=1e-10)
    np.testing.assert_allclose(hh, h, atol=1e-5)


def test_enu_round_trip_over_a_site():
    origin = GeoPoint(lat=40.8910459, lon=-74.2163420, h=233.0)
    frame = EnuFrame(origin)
    rng = np.random.default_rng(3)
    enu = rng.uniform([-2000, -2000, -50], [2000, 2000, 500], size=(300, 3))
    lat, lon, h = frame.to_geodetic(enu)
    np.testing.assert_allclose(frame.from_geodetic(lat, lon, h), enu, rtol=0, atol=1e-6)


def test_enu_origin_is_zero_and_matches_rotation_formula():
    origin = GeoPoint(lat=-33.86, lon=151.21, h=40.0)
    frame = EnuFrame(origin)
    np.testing.assert_allclose(frame.from_geodetic(origin.lat, origin.lon, origin.h), 0, atol=1e-6)
    R = ecef_to_enu_rotation(origin.lat, origin.lon)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    xyz = _ecef_closed_form(np.array([-33.861, -33.85]), np.array([151.2, 151.22]),
                            np.array([10.0, 300.0]))
    # pyproj's topocentric conversion == R (x - x0), independently computed
    expected = (xyz - _ecef_closed_form(origin.lat, origin.lon, origin.h)) @ R.T
    np.testing.assert_allclose(frame.from_ecef(xyz), expected, atol=1e-6)


def test_enu_axes_point_east_north_up():
    frame = EnuFrame(GeoPoint(lat=10.0, lon=20.0, h=0.0))
    e, n, u = frame.from_geodetic(10.0, 20.001, 0.0)
    assert e > 100 and abs(n) < 1 and u < 0       # east (and below the tangent plane)
    e, n, u = frame.from_geodetic(10.001, 20.0, 0.0)
    assert n > 100 and abs(e) < 1
    e, n, u = frame.from_geodetic(10.0, 20.0, 50.0)
    np.testing.assert_allclose([e, n, u], [0, 0, 50], atol=1e-6)


def test_geopoint_validation_and_parse():
    assert GeoPoint.parse("40.5, -74.2, 233") == GeoPoint(lat=40.5, lon=-74.2, h=233)
    assert GeoPoint.parse("1,2").h == 0.0
    with pytest.raises(ValueError):
        GeoPoint.parse("1")
    with pytest.raises(ValidationError):
        GeoPoint(lat=91, lon=0)
