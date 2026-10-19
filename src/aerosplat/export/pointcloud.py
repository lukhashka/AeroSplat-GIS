"""Point cloud of a run as LAS 1.4 / LAZ in the target CRS (laspy).

Source (stated in the file and the manifest):

* `cloud` (default) -- the run's registered cloud product: the dense MVS cloud
  (`gs/dense_wgs84.pt`) when the run used `cloud_source=dense`, else the sparse SfM
  cloud, after the parse stage's cleanup (and subsampling, if `max_points` was hit --
  the .pt's own stats are copied into the metadata). It is the cloud the splat was
  initialised from and the semantics stage measured.
* `splat` -- the 3DGS Gaussian centres with their DC colour. Not a survey product:
  centres of large, transparent or floater Gaussians are not surface points.

Coordinates are converted exactly per point: easting/northing in the CRS and Z as WGS84
ellipsoidal height (aerosplat.export.georef). Stored as int32 with 1 mm scale and an
offset near the minimum, i.e. no float32 rounding of UTM-sized numbers.

CRS: an OGC WKT1 VLR (LASF_Projection 2112), with the global-encoding WKT bit -- what
LAS 1.4 point formats 6-10 require; PDAL, LAStools, CloudCompare and QGIS read it. The
vertical datum (ellipsoidal heights) cannot be expressed in a WKT1 PROJCS, so it is
stated in the AeroSplat VLR next to the error budget.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from aerosplat.export.georef import HEIGHT_DATUM, ExportRefused, SiteProjection
from aerosplat.export.raster import GIS_EXTRA

SCALE_M = 0.001
VLR_USER_ID = "AeroSplat"          # <= 16 bytes
VLR_RECORD_ID = 1
SOURCES = ("cloud", "splat")


def require_laspy():
    try:
        import laspy
    except ImportError:
        raise ExportRefused(f"LAS export needs laspy: {GIS_EXTRA}") from None
    return laspy


def export_las(path: Path, *, enu: np.ndarray, rgb: Optional[np.ndarray],
               proj: SiteProjection, metadata: dict[str, Any]) -> dict[str, Any]:
    """Write `enu` (N, 3) Site ENU metres, `rgb` (N, 3) in 0..1 or None, as .las / .laz."""
    laspy = require_laspy()
    from laspy.vlrs.known import WktCoordinateSystemVlr

    enu = np.asarray(enu, np.float64)
    keep = np.isfinite(enu).all(axis=1)
    dropped = int((~keep).sum())
    enu = enu[keep]
    if len(enu) == 0:
        raise ExportRefused("the point cloud has no finite points")
    xyz = proj.enu_to_crs(enu)

    header = laspy.LasHeader(version="1.4", point_format=7)        # xyz + gps time + RGB
    header.scales = np.array([SCALE_M] * 3)
    header.offsets = np.floor(xyz.min(axis=0))
    header.global_encoding.wkt = True
    header.system_identifier = "AeroSplat-GIS export"
    header.generating_software = "aerosplat.export.pointcloud"
    header.vlrs.append(WktCoordinateSystemVlr(proj.wkt1))
    meta = {**metadata, "heights": HEIGHT_DATUM, "crs": proj.describe()["name"],
            "epsg": proj.epsg, "dropped_non_finite": dropped}
    header.vlrs.append(laspy.VLR(user_id=VLR_USER_ID, record_id=VLR_RECORD_ID,
                                 description="export metadata (JSON)",
                                 record_data=json.dumps(meta).encode("utf-8")))

    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if rgb is not None:
        c = np.clip(np.asarray(rgb, np.float64)[keep], 0.0, 1.0)
        c16 = np.round(c * 65535.0).astype(np.uint16)       # LAS colour is 16-bit
        las.red, las.green, las.blue = c16[:, 0], c16[:, 1], c16[:, 2]

    tmp = path.with_name(f".{path.stem}.partial{path.suffix}")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        las.write(str(tmp))
    except Exception as e:
        tmp.unlink(missing_ok=True)
        if path.suffix.lower() == ".laz":
            raise ExportRefused(f"cannot write LAZ ({e}); install a LAZ backend "
                                f"(`pip install lazrs`) or export .las") from None
        raise
    tmp.replace(path)
    return {"points": int(len(enu)), "dropped_non_finite": dropped,
            "bounds_crs": {"min": xyz.min(axis=0).tolist(), "max": xyz.max(axis=0).tolist()},
            "scale_m": SCALE_M, "format": "LAS 1.4, point format 7"}


def read_metadata(path: Path) -> dict[str, Any]:
    """The AeroSplat VLR of an exported LAS/LAZ (tests, audit)."""
    laspy = require_laspy()
    with laspy.open(str(path)) as f:
        for v in f.header.vlrs:
            if v.user_id == VLR_USER_ID and v.record_id == VLR_RECORD_ID:
                return json.loads(bytes(v.record_data).decode("utf-8"))
    return {}
