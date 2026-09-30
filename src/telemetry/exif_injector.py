#!/usr/bin/env python3
"""
exif_injector.py — AeroSplat-GIS

Writes synthetic UAV-style GPS EXIF into frames captured from a Google Earth
Studio "Orbit" render, so the frames behave like real geotagged drone photos
downstream (COLMAP SfM -> gs_trainer_v3 -> main.py -> appGLSL.js).

Two sources of pose are supported, in priority order:
  1. --telemetry data/flight_telemetry.csv  (ground truth from auto_photo.py
     or from Earth Studio's exported 3D tracking data). ALWAYS prefer this.
  2. An analytic circle, reconstructed from the orbit parameters below.
     Only correct if the Earth Studio orbit is a full 360 deg sweep with
     LINEAR keyframe interpolation.

Key differences vs. the original script:
  * EXIF is inserted with piexif.insert() -> no JPEG re-encode, no generational
    quality loss, existing EXIF preserved.
  * Writes to an output directory by default; originals are never touched.
  * WGS84 meters-per-degree instead of the flat 111111.0 constant.
  * Natural sort of filenames (frame_2 before frame_10).
  * Camera intrinsics priors (Make/Model + FocalLengthIn35mmFilm) so COLMAP
    gets a sane initial focal length and groups all frames under one camera.
  * Camera heading (GPSImgDirection) - the orbit camera looks at the centre.
  * Optional COLMAP model_aligner reference file export.
  * Read-back verification of every written file.

Comments are in English here for clarity; translate to match the rest of the
repo if you want to keep the Ukrainian convention.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import piexif

JPEG_SUFFIXES = {".jpg", ".jpeg"}


# --------------------------------------------------------------------------
# Geodesy
# --------------------------------------------------------------------------

def meters_per_degree(lat_deg: float) -> tuple[float, float]:
    """WGS84 local scale factors at a given latitude.

    Returns (metres per degree of latitude, metres per degree of longitude).
    Standard series approximation - accurate to well under a metre, versus
    ~0.2-0.5% error for the flat 111111.0 constant.
    """
    phi = math.radians(lat_deg)
    m_lat = (111132.92
             - 559.82 * math.cos(2 * phi)
             + 1.175 * math.cos(4 * phi)
             - 0.0023 * math.cos(6 * phi))
    m_lon = (111412.84 * math.cos(phi)
             - 93.5 * math.cos(3 * phi)
             + 0.118 * math.cos(5 * phi))
    return m_lat, m_lon


def offset_to_latlon(center_lat: float, center_lon: float,
                     north_m: float, east_m: float) -> tuple[float, float]:
    """Local tangent-plane offset (metres) -> WGS84 lat/lon."""
    m_lat, m_lon = meters_per_degree(center_lat)
    if abs(m_lon) < 1e-6:
        raise ValueError("Longitude scale collapses at the poles; "
                         "use a proper projection for polar captures.")
    return center_lat + north_m / m_lat, center_lon + east_m / m_lon


def focal35_from_hfov(hfov_deg: float) -> float:
    """Horizontal FOV (deg) -> 35mm-equivalent focal length (mm).

    36 mm is the width of a full-frame sensor. Read the FOV straight off the
    Earth Studio camera panel; it is constant for the whole render.
    """
    if not 1.0 < hfov_deg < 179.0:
        raise ValueError(f"Implausible horizontal FOV: {hfov_deg}")
    return 36.0 / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


# --------------------------------------------------------------------------
# EXIF helpers
# --------------------------------------------------------------------------

def deg_to_dms_rational(value: float, sec_den: int = 10000):
    """Signed decimal degrees -> EXIF (deg, min, sec) rational triple.

    sec_den=10000 gives ~3 mm of quantisation. The original script used 1000
    (~3 cm), which is coarse enough to be visible as jitter when the whole
    orbit is only 250 m across.
    """
    a = abs(value)
    d = int(a)
    m = int((a - d) * 60)
    s = (a - d - m / 60.0) * 3600.0
    s_num = round(s * sec_den)
    # Carry, in case rounding pushes seconds to exactly 60.
    if s_num >= 60 * sec_den:
        s_num -= 60 * sec_den
        m += 1
    if m >= 60:
        m -= 60
        d += 1
    return ((d, 1), (m, 1), (s_num, sec_den))


def dms_rational_to_deg(dms) -> float:
    d = dms[0][0] / dms[0][1]
    m = dms[1][0] / dms[1][1]
    s = dms[2][0] / dms[2][1]
    return d + m / 60.0 + s / 3600.0


def build_gps_ifd(lat: float, lon: float, alt_m: float,
                  heading_deg: float | None,
                  stamp: datetime | None) -> dict:
    gps = {
        piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0),
        piexif.GPSIFD.GPSMapDatum: "WGS-84",
        piexif.GPSIFD.GPSLatitudeRef: b"N" if lat >= 0 else b"S",
        piexif.GPSIFD.GPSLatitude: deg_to_dms_rational(lat),
        piexif.GPSIFD.GPSLongitudeRef: b"E" if lon >= 0 else b"W",
        piexif.GPSIFD.GPSLongitude: deg_to_dms_rational(lon),
        # Ref 0 = above sea level, 1 = below. The original hardcoded 0 and
        # would have written a nonsense positive value for negative altitudes.
        piexif.GPSIFD.GPSAltitudeRef: 0 if alt_m >= 0 else 1,
        piexif.GPSIFD.GPSAltitude: (round(abs(alt_m) * 1000), 1000),
    }
    if heading_deg is not None:
        h = heading_deg % 360.0
        gps[piexif.GPSIFD.GPSImgDirectionRef] = b"T"          # true north
        gps[piexif.GPSIFD.GPSImgDirection] = (round(h * 100), 100)
    if stamp is not None:
        utc = stamp.astimezone(timezone.utc)
        gps[piexif.GPSIFD.GPSDateStamp] = utc.strftime("%Y:%m:%d")
        gps[piexif.GPSIFD.GPSTimeStamp] = (
            (utc.hour, 1), (utc.minute, 1),
            (round((utc.second + utc.microsecond / 1e6) * 100), 100),
        )
    return gps


# --------------------------------------------------------------------------
# Orbit model
# --------------------------------------------------------------------------

@dataclass
class OrbitConfig:
    center_lat: float
    center_lon: float
    altitude_m: float          # metres above sea level (see --altitude-is-agl)
    radius_m: float
    arc_deg: float = 360.0     # partial orbits are common in Earth Studio
    start_bearing_deg: float = 0.0   # bearing of frame 0 from the centre
    clockwise: bool = True
    look_at_center: bool = True
    frame_interval_s: float = 1.0
    start_time: datetime | None = None

    def pose(self, i: int, n_frames: int) -> tuple[float, float, float, float | None]:
        """Pose of frame i. Returns (lat, lon, alt, heading_deg|None)."""
        if n_frames <= 0:
            raise ValueError("n_frames must be positive")
        # A full 360 orbit is periodic: the last frame must NOT duplicate the
        # first, so divide by n. A partial arc is not periodic: the last frame
        # should land exactly on arc_deg, so divide by (n - 1).
        closed = abs(self.arc_deg % 360.0) < 1e-9
        denom = n_frames if closed else max(n_frames - 1, 1)
        sweep = self.arc_deg * (i / denom)
        bearing = self.start_bearing_deg + (sweep if self.clockwise else -sweep)

        theta = math.radians(bearing)
        north = self.radius_m * math.cos(theta)
        east = self.radius_m * math.sin(theta)
        lat, lon = offset_to_latlon(self.center_lat, self.center_lon, north, east)

        heading = (bearing + 180.0) % 360.0 if self.look_at_center else None
        return lat, lon, self.altitude_m, heading


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------

def natural_key(name: str):
    """frame_2.jpg sorts before frame_10.jpg (plain sorted() gets this wrong)."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", name)]


def list_frames(images_dir: Path) -> list[Path]:
    frames = [p for p in images_dir.iterdir()
              if p.is_file() and p.suffix.lower() in JPEG_SUFFIXES]
    return sorted(frames, key=lambda p: natural_key(p.name))


def load_telemetry(csv_path: Path) -> list[dict]:
    """Load poses from flight_telemetry.csv.

    Expected columns (case-insensitive, extras ignored):
        lat, lon, alt   [, heading] [, frame or filename]
    Adjust the aliases below to match whatever auto_photo.py actually writes.
    """
    aliases = {
        "lat": {"lat", "latitude"},
        "lon": {"lon", "lng", "long", "longitude"},
        "alt": {"alt", "altitude", "height"},
        "heading": {"heading", "yaw", "bearing", "azimuth"},
        "name": {"filename", "file", "image", "frame_name"},
    }
    rows: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header row")
        lookup = {}
        for key, names in aliases.items():
            for col in reader.fieldnames:
                if col.strip().lower() in names:
                    lookup[key] = col
                    break
        for req in ("lat", "lon", "alt"):
            if req not in lookup:
                raise ValueError(
                    f"{csv_path} is missing a '{req}' column "
                    f"(found: {reader.fieldnames})")
        for row in reader:
            rows.append({
                "lat": float(row[lookup["lat"]]),
                "lon": float(row[lookup["lon"]]),
                "alt": float(row[lookup["alt"]]),
                "heading": (float(row[lookup["heading"]])
                            if "heading" in lookup and row[lookup["heading"]]
                            else None),
                "name": row[lookup["name"]] if "name" in lookup else None,
            })
    if not rows:
        raise ValueError(f"{csv_path} contains no data rows")
    return rows


# --------------------------------------------------------------------------
# Main injection
# --------------------------------------------------------------------------

def inject(frames: list[Path], poses: list[tuple], out_dir: Path | None,
           camera_make: str, camera_model: str, focal35: float | None,
           start_time: datetime | None, frame_interval_s: float,
           dry_run: bool) -> list[tuple[Path, tuple]]:
    written: list[tuple[Path, tuple]] = []

    for i, (src, (lat, lon, alt, heading)) in enumerate(zip(frames, poses)):
        dst = src if out_dir is None else out_dir / src.name
        stamp = (start_time + timedelta(seconds=i * frame_interval_s)
                 if start_time else None)

        if dry_run:
            written.append((dst, (lat, lon, alt, heading)))
            continue

        if dst != src:
            shutil.copy2(src, dst)

        # Preserve whatever EXIF the source already carries, then merge.
        try:
            exif = piexif.load(str(dst))
        except Exception:
            exif = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

        exif["GPS"] = build_gps_ifd(lat, lon, alt, heading, stamp)

        # Intrinsics priors: a constant Make/Model makes COLMAP treat every
        # frame as the SAME physical camera (correct for a fixed-FOV render),
        # and FocalLengthIn35mmFilm seeds the initial focal estimate instead
        # of COLMAP falling back to its 1.2 * max(w, h) guess.
        exif.setdefault("0th", {})
        exif["0th"][piexif.ImageIFD.Make] = camera_make
        exif["0th"][piexif.ImageIFD.Model] = camera_model
        if focal35 is not None:
            exif.setdefault("Exif", {})
            exif["Exif"][piexif.ExifIFD.FocalLengthIn35mmFilm] = round(focal35)
        if stamp is not None:
            exif.setdefault("Exif", {})
            ts = stamp.strftime("%Y:%m:%d %H:%M:%S")
            exif["Exif"][piexif.ExifIFD.DateTimeOriginal] = ts
            exif["Exif"][piexif.ExifIFD.DateTimeDigitized] = ts

        try:
            exif_bytes = piexif.dump(exif)
        except Exception:
            # Malformed thumbnails are the usual culprit; drop and retry.
            exif["thumbnail"] = None
            exif["1st"] = {}
            exif_bytes = piexif.dump(exif)

        # insert() rewrites only the APP1 segment - the JPEG scan data is
        # copied through byte-for-byte. No decode, no re-encode, no loss.
        piexif.insert(exif_bytes, str(dst))
        written.append((dst, (lat, lon, alt, heading)))

    return written


def verify(written: list[tuple[Path, tuple]], tol_deg: float = 1e-6) -> int:
    """Read every file back and confirm the GPS survived the round trip."""
    failures = 0
    for path, (lat, lon, _alt, _h) in written:
        try:
            gps = piexif.load(str(path))["GPS"]
            got_lat = dms_rational_to_deg(gps[piexif.GPSIFD.GPSLatitude])
            got_lon = dms_rational_to_deg(gps[piexif.GPSIFD.GPSLongitude])
            if gps[piexif.GPSIFD.GPSLatitudeRef] == b"S":
                got_lat = -got_lat
            if gps[piexif.GPSIFD.GPSLongitudeRef] == b"W":
                got_lon = -got_lon
            if abs(got_lat - lat) > tol_deg or abs(got_lon - lon) > tol_deg:
                print(f"  MISMATCH {path.name}: "
                      f"wrote {lat:.7f},{lon:.7f} read {got_lat:.7f},{got_lon:.7f}")
                failures += 1
        except Exception as exc:
            print(f"  UNREADABLE {path.name}: {exc}")
            failures += 1
    return failures


def write_colmap_geo(path: Path, written: list[tuple[Path, tuple]]) -> None:
    """Reference file for `colmap model_aligner`.

    Format: one line per image -> `image_name X Y Z`, here lat lon alt.
    Check `colmap model_aligner -h` for your build: the flag naming for
    lat/lon input has changed across versions (--ref_is_gps / --alignment_type).
    This is a far more reliable georegistration path than hoping the mapper
    picks up EXIF GPS on its own.
    """
    with path.open("w", encoding="utf-8") as fh:
        for p, (lat, lon, alt, _h) in written:
            fh.write(f"{p.name} {lat:.9f} {lon:.9f} {alt:.3f}\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    project_dir = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(
        description="Inject synthetic orbital GPS EXIF into Earth Studio frames.")
    p.add_argument("--images-dir", type=Path,
                   default=project_dir / "data" / "raw_images_ab")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (default: <images-dir>_geo). "
                        "Originals are never modified unless --in-place.")
    p.add_argument("--in-place", action="store_true",
                   help="Write EXIF back into the source files.")

    p.add_argument("--telemetry", type=Path, default=None,
                   help="CSV of per-frame poses. Overrides the analytic orbit.")

    p.add_argument("--lat", type=float, required=False,
                   help="Orbit centre latitude, decimal degrees.")
    p.add_argument("--lon", type=float, required=False,
                   help="Orbit centre longitude, decimal degrees.")
    p.add_argument("--altitude", type=float, default=441.0,
                   help="Camera altitude above sea level, metres.")
    p.add_argument("--radius", type=float, default=250.0,
                   help="Orbit radius, metres.")
    p.add_argument("--arc", type=float, default=360.0,
                   help="Total sweep in degrees (use e.g. 180 for a half orbit).")
    p.add_argument("--start-bearing", type=float, default=0.0,
                   help="Bearing of frame 0 from the centre, deg clockwise from N.")
    p.add_argument("--counter-clockwise", action="store_true")
    p.add_argument("--no-heading", action="store_true",
                   help="Skip GPSImgDirection (use if the camera is not "
                        "locked to the orbit centre).")

    p.add_argument("--hfov", type=float, default=None,
                   help="Horizontal field of view in degrees, from the Earth "
                        "Studio camera panel. Enables a focal-length prior.")
    p.add_argument("--camera-make", default="GoogleEarthStudio")
    p.add_argument("--camera-model", default="OrbitVirtualCam")

    p.add_argument("--frame-interval", type=float, default=1.0,
                   help="Seconds between frames, for synthetic timestamps.")
    p.add_argument("--no-timestamps", action="store_true")

    p.add_argument("--colmap-geo", type=Path, default=None,
                   help="Also write a model_aligner reference file here.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the computed track without touching any file.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    images_dir: Path = args.images_dir
    if not images_dir.is_dir():
        print(f"ERROR: {images_dir} does not exist", file=sys.stderr)
        return 1

    frames = list_frames(images_dir)
    if not frames:
        print(f"ERROR: no JPEG frames in {images_dir}", file=sys.stderr)
        return 1
    n = len(frames)

    # ---- poses -----------------------------------------------------------
    if args.telemetry:
        rows = load_telemetry(args.telemetry)
        if len(rows) != n:
            print(f"ERROR: telemetry has {len(rows)} rows but {n} frames were "
                  f"found. Refusing to guess the alignment.", file=sys.stderr)
            return 1
        if any(r["name"] for r in rows):
            by_name = {r["name"]: r for r in rows}
            missing = [f.name for f in frames if f.name not in by_name]
            if missing:
                print(f"ERROR: telemetry has no row for {missing[:3]}...",
                      file=sys.stderr)
                return 1
            rows = [by_name[f.name] for f in frames]
        poses = [(r["lat"], r["lon"], r["alt"], r["heading"]) for r in rows]
        source = f"telemetry {args.telemetry.name}"
    else:
        if args.lat is None or args.lon is None:
            print("ERROR: --lat and --lon are required without --telemetry",
                  file=sys.stderr)
            return 1
        if not -90 <= args.lat <= 90 or not -180 <= args.lon <= 180:
            print("ERROR: lat/lon out of range - did you swap them?",
                  file=sys.stderr)
            return 1
        cfg = OrbitConfig(
            center_lat=args.lat, center_lon=args.lon,
            altitude_m=args.altitude, radius_m=args.radius,
            arc_deg=args.arc, start_bearing_deg=args.start_bearing,
            clockwise=not args.counter_clockwise,
            look_at_center=not args.no_heading,
        )
        poses = [cfg.pose(i, n) for i in range(n)]
        source = (f"analytic orbit r={args.radius}m arc={args.arc}deg "
                  f"{'CW' if cfg.clockwise else 'CCW'}")

    # ---- output target ---------------------------------------------------
    if args.in_place:
        out_dir = None
    else:
        out_dir = args.out_dir or images_dir.parent / f"{images_dir.name}_geo"
        out_dir.mkdir(parents=True, exist_ok=True)

    focal35 = focal35_from_hfov(args.hfov) if args.hfov else None
    start_time = None if args.no_timestamps else datetime.now(timezone.utc)

    print(f"Frames      : {n} in {images_dir}")
    print(f"Pose source : {source}")
    print(f"Output      : {'IN PLACE' if out_dir is None else out_dir}")
    if focal35:
        print(f"Focal prior : {focal35:.2f} mm (35mm eq) from {args.hfov} deg HFOV")

    written = inject(frames, poses, out_dir,
                     args.camera_make, args.camera_model, focal35,
                     start_time, args.frame_interval, args.dry_run)

    step = max(n // 8, 1)
    for i, (path, (lat, lon, alt, h)) in enumerate(written):
        if i % step == 0 or i == n - 1:
            hd = f" hdg {h:6.2f}" if h is not None else ""
            print(f"  [{i+1:>4}/{n}] {path.name:<28} "
                  f"{lat:.7f}, {lon:.7f} @ {alt:.1f}m{hd}")

    if args.dry_run:
        print("Dry run - nothing written.")
        return 0

    print("Verifying round trip...")
    failures = verify(written)
    if failures:
        print(f"FAILED: {failures}/{n} frames did not verify")
        return 2
    print(f"OK: {n}/{n} frames verified.")

    if args.colmap_geo:
        write_colmap_geo(args.colmap_geo, written)
        print(f"Wrote COLMAP reference file: {args.colmap_geo}")

    print("next: python src/reconstruction/sfm_runner.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())