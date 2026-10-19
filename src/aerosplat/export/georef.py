"""Site ENU -> a standard projected CRS, and the georeference error budget of an export.

Runs live in WGS84 Site ENU: a topocentric frame tangent to the ellipsoid at the Site's
fixed origin (aerosplat.geo.crs.EnuFrame, ADR-0006). No GIS reads that frame, so exports
leave it for a projected CRS:

* **Default: the WGS84 UTM zone of the Site origin** (EPSG:326zz / 327zz by the EPSG
  zone areas, i.e. by longitude; the MGRS Norway/Svalbard exceptions are not applied --
  both zones are valid projections there, and `--crs` picks the other). Metric and conformal
  -- a pixel is square and a metre is a metre to within the UTM scale factor
  (0.9996 ... ~1.001, recorded) -- on the same datum as Site ENU, so the projection adds
  no datum shift; EPSG-coded, so every GIS, COG viewer and LAS tool reads it. A site is
  ~1 km, far smaller than a 6° zone; the origin's zone is used for the whole site even if
  it touches a zone edge (the projection stays valid well beyond it).
* Any other projected, metre-based CRS can be asked for (`EPSG:xxxx`, e.g. a national
  grid). If its datum is not WGS84, pyproj picks a datum transformation; its stated
  accuracy goes into the error budget, and a CRS for which pyproj knows no accuracy, or
  whose area of use does not contain the Site, is refused.

Geographic WGS84 (EPSG:4326) is not offered for rasters: degree pixels are not square
in metres and a DSM in degrees x metres is a trap.

Heights everywhere are **WGS84 ellipsoidal** (h of EPSG:4979), converted exactly per point
from ENU (u is not h: the tangent plane leaves the ellipsoid by d^2 / 2R, 2 cm at 500 m,
8 cm at 1 km). Whether Earth Studio altitudes really are ellipsoidal is unverified
(ARCHITECTURE §4) and is recorded as such.

Rasters need the ENU <-> CRS map as an affine over the area (`PlaneFit`, at the ground
height): it is fitted to exact pyproj transforms of a control lattice and its residual is
checked, not assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from pyproj.enums import WktVersion

from aerosplat.geo.crs import EnuFrame, GeoPoint

F64 = NDArray[np.float64]

# The frame georeferenced products are in. Same string as aerosplat.geo.georef.SITE_FRAME
# (that module imports torch; tests check the two agree).
SITE_FRAME = "WGS84 Site ENU"
DEFAULT_CRS = "utm"
# A fitted ENU -> CRS affine must reproduce pyproj to this (metres) or the area is refused.
MAX_FIT_RESIDUAL_M = 0.005
HEIGHT_DATUM = "WGS84 ellipsoidal height (EPSG:4979 h), metres"
EARTH_RADIUS_M = 6371008.8

# What metric.frame_error_m means for each Run.earth_model.
FRAME_ERROR_MEANING = {
    "sphere-enu+affine": "max residual of the affine map sphere ENU -> WGS84 Site ENU that "
                         "placed the products (geo stage fit, reports/registration.json)",
    "sphere-legacy": "max shift of the legacy (pre-ADR-0006) frame against WGS84 Site ENU "
                     "(legacy_shift); the products are NOT corrected for it",
    "wgs84": "unknown: poses were converted to WGS84 at ingest; for Earth Studio captures "
             "they disagree with the frames by ~1 m (ARCHITECTURE §4)",
}

NOT_IN_BUDGET = [
    "absolute accuracy of the source imagery and poses (Earth Studio / Google 3D publishes "
    "none; the twin is assumed good to 2-3 m, docs/HAZARDS.md H-001)",
    "reconstruction error: bundle adjustment moves cameras against the known poses "
    "(compare_poses) and the MVS / 3DGS surface has its own error",
    "vertical datum: Earth Studio altitude is taken as WGS84 ellipsoidal height, "
    "unverified (ARCHITECTURE §4, altitude_reference)",
]


class ExportRefused(RuntimeError):
    """An export that would be wrong or ungeoreferenced. The message says why."""


# ── CRS choice ───────────────────────────────────────────────────────────
def utm_crs_for(origin: GeoPoint) -> CRS:
    """The WGS84 UTM zone CRS that contains the Site origin."""
    aoi = AreaOfInterest(origin.lon, origin.lat, origin.lon, origin.lat)
    found = [c for c in query_utm_crs_info(datum_name="WGS 84", area_of_interest=aoi)
             if c.auth_name == "EPSG" and c.code[:3] in ("326", "327")]
    if not found:
        raise ExportRefused(f"no WGS84 UTM zone contains {origin.lat}, {origin.lon} "
                            f"(polar site?): give a projected CRS explicitly")
    # Several only on a zone boundary; any of them is valid there, take the lowest code.
    return CRS.from_epsg(int(sorted(c.code for c in found)[0]))


def resolve_crs(origin: GeoPoint, spec: Optional[str] = DEFAULT_CRS) -> CRS:
    """'utm' (default) or any projected metre CRS pyproj understands ('EPSG:32618')."""
    if spec is None or str(spec).strip().lower() in ("", DEFAULT_CRS):
        return utm_crs_for(origin)
    try:
        crs = CRS.from_user_input(spec)
    except Exception as e:  # pyproj.exceptions.CRSError and friends
        raise ExportRefused(f"unknown CRS {spec!r}: {e}") from None
    if not crs.is_projected:
        raise ExportRefused(f"{crs.name} is not a projected CRS: exports need metric "
                            f"easting/northing (use 'utm' or a projected EPSG code)")
    units = {a.unit_name for a in crs.axis_info[:2]}
    if units != {"metre"}:
        raise ExportRefused(f"{crs.name} has axis units {sorted(units)}, not metres")
    aou = crs.area_of_use
    if aou is not None and not (aou.west <= origin.lon <= aou.east
                                and aou.south <= origin.lat <= aou.north):
        raise ExportRefused(f"{crs.name} is not defined at the Site origin "
                            f"({origin.lat:.5f}, {origin.lon:.5f}; its area of use is "
                            f"{aou.name})")
    return crs


# ── projection ───────────────────────────────────────────────────────────
class SiteProjection:
    """Exact conversions between Site ENU (metres) and a projected CRS + ellipsoidal h."""

    def __init__(self, origin: GeoPoint, crs: Optional[str | CRS] = DEFAULT_CRS):
        self.origin = origin
        self.frame = EnuFrame(origin)
        self.crs = crs if isinstance(crs, CRS) else resolve_crs(origin, crs)
        # Horizontal only: WGS84 lon/lat -> easting/northing; heights stay ellipsoidal.
        self._fwd = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        accuracy = self._fwd.accuracy
        if accuracy is None or accuracy < 0:
            raise ExportRefused(
                f"pyproj knows no accuracy for WGS84 -> {self.crs.name} "
                f"({self._fwd.description}); the error budget cannot be stated -- "
                f"use 'utm' or a CRS on the WGS84 datum")
        self.datum_accuracy_m = float(accuracy)

    @property
    def epsg(self) -> Optional[int]:
        return self.crs.to_epsg()

    @cached_property
    def wkt1(self) -> str:
        """WKT1 (GDAL flavour): what LAS 1.4 readers expect in the WKT VLR."""
        return self.crs.to_wkt(WktVersion.WKT1_GDAL)

    def enu_to_crs(self, enu: ArrayLike) -> F64:
        """(..., 3) Site ENU -> (..., 3) [easting, northing, ellipsoidal h], exact."""
        lat, lon, h = self.frame.to_geodetic(enu)
        x, y = self._fwd.transform(lon, lat)
        return np.stack([np.asarray(x, float), np.asarray(y, float), np.asarray(h, float)],
                        axis=-1)

    def crs_to_enu(self, x: ArrayLike, y: ArrayLike, h: ArrayLike) -> F64:
        """[easting, northing, ellipsoidal h] -> Site ENU, exact."""
        lon, lat = self._fwd.transform(np.asarray(x, float), np.asarray(y, float),
                                       direction="INVERSE")
        return self.frame.from_geodetic(np.asarray(lat), np.asarray(lon),
                                        np.broadcast_to(np.asarray(h, float), np.shape(lat)))

    def describe(self) -> dict[str, Any]:
        return {"name": self.crs.name, "epsg": self.epsg,
                "wkt": self.crs.to_wkt(WktVersion.WKT2_2019),
                "heights": HEIGHT_DATUM,
                "datum_transform": self._fwd.description,
                "datum_transform_accuracy_m": self.datum_accuracy_m}

    def fit_plane(self, e_range: tuple[float, float], n_range: tuple[float, float],
                  u_range: tuple[float, float], *, u_ref: float, steps: int = 9,
                  max_residual_m: float = MAX_FIT_RESIDUAL_M) -> "PlaneFit":
        """Affine (e, n) -> (easting, northing) at height u_ref over a box, fitted to exact
        transforms; refuses when the residual exceeds `max_residual_m`.

        Fitted at ONE height on purpose. A top-down orthographic render projects along the
        origin's up axis, not the local vertical, so a point h above u_ref lands
        ~ d / R * h away from where the local vertical would put it (d = distance from the
        origin). That displacement grows with e * u and is not affine; it is measured
        exactly over `u_range` and reported as `lean_m` (millimetres to centimetres on a
        site), not hidden in a fit.
        """
        e = np.linspace(*e_range, steps)
        n = np.linspace(*n_range, steps)
        en = np.stack(np.meshgrid(e, n, indexing="ij"), axis=-1).reshape(-1, 2)
        dst = self.enu_to_crs(np.column_stack([en, np.full(len(en), u_ref)]))[:, :2]
        design = np.hstack([en, np.ones((len(en), 1))])
        # Centre the target to keep lstsq well conditioned with UTM-sized numbers.
        c = dst.mean(axis=0)
        sol, *_ = np.linalg.lstsq(design, dst - c, rcond=None)            # (3, 2)
        L = sol[:2].T.copy()                                               # (2, 2)
        t = sol[2] + c
        res = float(np.linalg.norm(en @ L.T + t - dst, axis=1).max())
        if res > max_residual_m:
            raise ExportRefused(
                f"the area ({e_range[1] - e_range[0]:.0f} x {n_range[1] - n_range[0]:.0f} m) is "
                f"too large for one affine Site ENU -> {self.crs.name} map: residual "
                f"{res * 1000:.1f} mm > {max_residual_m * 1000:.1f} mm")
        # Lean: corners at the bottom and top of the height range vs. at u_ref.
        corners = np.array([[e_range[0], n_range[0]], [e_range[1], n_range[0]],
                            [e_range[0], n_range[1]], [e_range[1], n_range[1]]], float)
        ref = self.enu_to_crs(np.column_stack([corners, np.full(4, u_ref)]))[:, :2]
        lean = 0.0
        for u in u_range:
            at = self.enu_to_crs(np.column_stack([corners, np.full(4, u)]))[:, :2]
            lean = max(lean, float(np.linalg.norm(at - ref, axis=1).max()))
        return PlaneFit(L=L, t=t, u_ref=float(u_ref), residual_max_m=res, lean_m=lean,
                        e_range=(float(e_range[0]), float(e_range[1])),
                        n_range=(float(n_range[0]), float(n_range[1])),
                        u_range=(float(u_range[0]), float(u_range[1])))


@dataclass(frozen=True)
class PlaneFit:
    """(easting, northing) = L @ (e, n) + t for Site ENU points at height u_ref."""

    L: F64                  # (2, 2): d(easting, northing) / d(e, n), a scaled rotation
    t: F64                  # (2,)
    u_ref: float            # the height the ground is taken at
    residual_max_m: float
    lean_m: float           # worst horizontal shift of a point at either end of u_range
    e_range: tuple[float, float]
    n_range: tuple[float, float]
    u_range: tuple[float, float]

    def to_crs(self, e: ArrayLike, n: ArrayLike) -> tuple[F64, F64]:
        e, n = np.asarray(e, float), np.asarray(n, float)
        return (self.L[0, 0] * e + self.L[0, 1] * n + self.t[0],
                self.L[1, 0] * e + self.L[1, 1] * n + self.t[1])

    def to_enu(self, x: ArrayLike, y: ArrayLike) -> tuple[F64, F64]:
        """Horizontal ENU of a CRS position (for a point at u_ref)."""
        rx = np.asarray(x, float) - self.t[0]
        ry = np.asarray(y, float) - self.t[1]
        inv = np.linalg.inv(self.L)
        return inv[0, 0] * rx + inv[0, 1] * ry, inv[1, 0] * rx + inv[1, 1] * ry

    @property
    def scale_factor(self) -> float:
        """CRS metres per Site ENU metre (UTM point scale x height factor)."""
        return float(math.sqrt(abs(np.linalg.det(self.L))))

    @property
    def rotation_deg(self) -> float:
        """Angle, counter-clockwise, from the CRS easting axis to the image of ENU east.
        For a transverse Mercator this is the meridian convergence at the site (pyproj's sign:
        positive east of the central meridian in the northern hemisphere)."""
        return float(math.degrees(math.atan2(self.L[1, 0], self.L[0, 0])))

    @property
    def anisotropy(self) -> float:
        """Relative difference of the two scales of L (0 for an exact similarity)."""
        s = np.linalg.svd(self.L, compute_uv=False)
        return float(s[0] / s[1] - 1.0)

    def describe(self) -> dict[str, Any]:
        return {"scale_factor": self.scale_factor, "enu_to_grid_rotation_deg": self.rotation_deg,
                "anisotropy": self.anisotropy, "fit_residual_max_m": self.residual_max_m,
                "u_ref_m": self.u_ref, "lean_max_m": self.lean_m,
                "lean_meaning": "horizontal shift of a point at the top/bottom of the height "
                                "range against one at u_ref with the same Site ENU e, n "
                                "(projection along the origin's vertical)",
                "box_enu": {"e": list(self.e_range), "n": list(self.n_range),
                            "u": list(self.u_range)}}


# ── error budget ─────────────────────────────────────────────────────────
def georeference_budget(*, earth_model: str, metric: dict[str, Any],
                        report: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """What is known about how well a run's exports sit on the Earth (uncertainty is data).

    Only figures the run actually recorded are given; what nobody measured is listed under
    `not_included` instead of being silently treated as zero.
    """
    report = report or {}
    fe = metric.get("frame_error_m")
    out: dict[str, Any] = {
        "earth_model": earth_model,
        "frame": SITE_FRAME,
        "origin": metric.get("origin"),
        "frame_error_m": fe,
        "frame_error_meaning": FRAME_ERROR_MEANING.get(
            earth_model, "unknown earth model: frame error not characterised"),
        "altitude_reference": report.get("altitude_reference", "unknown"),
        "not_included": list(NOT_IN_BUDGET),
    }
    if "sphere_to_wgs84" in report:
        # The difference the sphere -> WGS84 modelling removed (corrected, not an error).
        out["sphere_to_wgs84_corrected"] = report["sphere_to_wgs84"]
    if earth_model == "sphere-legacy" and "legacy_shift" in report:
        out["legacy_shift"] = report["legacy_shift"]
    warnings = []
    if fe is None:
        warnings.append("frame error of this run is unknown")
    elif fe > 0.5:
        warnings.append(f"known frame error {fe:.2f} m: products are offset by up to that much")
    out["warnings"] = warnings
    return out
