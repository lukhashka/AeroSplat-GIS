"""
earthstudio_to_colmap.py

Converts an Earth Studio 3D-tracking-data export (per-frame position,
rotation, geodetic coordinate, FOV) into a COLMAP known-pose seed:

  - cameras.txt   real intrinsics (SIMPLE_PINHOLE), derived from fovVertical
  - images.txt    known camera poses (QW QX QY QZ TX TY TZ) in a local ENU
                   frame -- one entry per REAL footage file, matched by name
  - points3D.txt  empty (populate via point_triangulator)
  - geo_registration.json   ENU origin + per-image lat/lon/alt, for
                   downstream georeferencing of the final reconstruction

Also copies (or symlinks) the matched footage into the `raw_images_<dataset>`
convention sfm_runner.py / dense_runner.py expect, and -- if piexif is
available -- writes real GPS EXIF into the copies. This bypasses
exif_injector.py entirely for this path: its telemetry-CSV format was
designed for synthetic circular flights and hasn't been confirmed against
this converter's output, so re-deriving raw_images_<ds> directly here avoids
depending on an unverified column format. exif_injector.py is still exactly
right for the old synthetic-orbit workflow; this script is the real-footage
equivalent, not a replacement for it there.

============================================================================
CONFIRMED MODEL (validated in-session against a real Earth Studio export --
see validate_convention.py for the residual tests that established this; if
Earth Studio ever changes its export convention, re-run that validation
before trusting this again):

  1. `position` is geocentric coordinates on a SPHERICAL earth model
     (R = 6371008.8 m, IUGG mean radius), computed directly from
     coordinate.{latitude,longitude,altitude} as if they were spherical
     angles. NOT WGS84 ellipsoidal ECEF (that model gives a ~21.1 km
     systematic error -- confirmed by direct residual test).

  2. `rotation` (rx, ry, rz, in degrees) composes as:
         R_cam2globalspherical = Rx(rx) @ Ry(ry) @ Rz(rz)
     applied in the SAME global spherical-geocentric frame as `position`
     (not a per-frame local-ENU tangent frame). Confirmed by a least-
     squares ray-convergence test: residual = 0.000 m (all forward rays
     intersect a single point exactly, across every frame tested).

  3. Camera forward axis in local camera space is +Z. Confirmed directly:
     dot(R @ [0,0,1], direction-to-converged-target) = 0.9999999999998.

  4. CONFIRMED (empirically, not by inspection): no axis flip at all is
     needed -- LOCAL_TO_COLMAP is the identity. Earth Studio's local
     camera axes, run through the composition above, already match
     COLMAP's (X-right, Y-down, Z-forward) convention directly. The
     originally-assumed glTF/Unity -> OpenCV flip (Y,Z -> -Y,-Z) was
     WRONG: tested by re-running point_triangulator against an
     already-built database (features/matches don't depend on pose, so
     this is a ~2s test per candidate, not a full SfM rerun) across all
     8 sign-flip combinations of X/Y/Z. Identity triangulated 13,977
     points; the next best candidate (X,Y flip) got 3,894; the original
     diag([1,-1,-1]) got 0. Re-run this sweep (see git history / session
     notes for the test script) if this ever needs re-confirming against
     a different Earth Studio export.

USAGE:
    python earthstudio_to_colmap.py data/Untitled/Untitled.json \\
        --dataset Untitled \\
        --footage-dir data/footage \\
        --images-out-dir data/raw_images_Untitled \\
        --out-dir data/colmap_output_Untitled/sparse_known

    (starter.py's new "geo" stage calls this for you with the right paths --
    see CONFIG["camera_source"] = "earthstudio_json".)

Recommended COLMAP usage downstream (the "hybrid" approach): seed with these
known poses/intrinsics via point_triangulator, then bundle_adjuster with
refinement ENABLED (not frozen), then compare refined poses back against
these known ones with compare_poses.py. Large divergence flags a bug in
this conversion; small divergence is expected BA noise.
"""
import argparse
import json
import os
import re
import shutil
import sys
import numpy as np

EARTH_R = 6371008.8  # IUGG mean radius, meters -- confirmed model, see header

# Identity -- see header note #4. Earth Studio's local camera axes, as
# produced by the rotation composition above, already match COLMAP's
# (X-right, Y-down, Z-forward) convention with no further flip. Confirmed
# empirically via a point_triangulator sweep across all 8 sign-flip
# candidates (identity triangulated 13,977 points vs. 0 for the
# originally-assumed diag([1,-1,-1]) glTF/Unity flip -- that assumption
# was wrong, not just unverified).
LOCAL_TO_COLMAP = np.diag([1.0, 1.0, 1.0])

IMG_EXTS = (".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG")


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def mat_to_quat_wxyz(R):
    """Rotation matrix -> unit quaternion (w, x, y, z), COLMAP's convention."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def find_footage(footage_dir, dataset, expected_count):
    """
    Match real frame files in `footage_dir` named like f"{dataset}_<n>.<ext>"
    (e.g. Untitled_00.jpeg .. Untitled_90.jpeg), sorted numerically by the
    trailing index -- NOT lexicographically, so this is safe even if the
    dataset ever has >=100 frames with inconsistent zero-padding.
    """
    pattern = re.compile(rf"^{re.escape(dataset)}_(\d+)(\.[^.]+)$")
    matches = []
    for name in os.listdir(footage_dir):
        m = pattern.match(name)
        if m and name.endswith(IMG_EXTS):
            matches.append((int(m.group(1)), name))
    matches.sort(key=lambda t: t[0])
    files = [name for _, name in matches]

    if len(files) != expected_count:
        raise SystemExit(
            f"ERROR: found {len(files)} footage files matching "
            f"'{dataset}_<n>.<ext>' in {footage_dir}, but the JSON has "
            f"{expected_count} camera frames. These must match 1:1 -- "
            f"fix the footage directory or re-check the export before "
            f"continuing (a silent mismatch here would misassign poses "
            f"to the wrong images)."
        )
    return files


def load_frames(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data, data["cameraFrames"]


def convert(json_path, dataset, out_dir, footage_dir=None, images_out_dir=None,
            copy_images=True, write_exif=True):
    data, frames = load_frames(json_path)
    n = len(frames)
    width, height = data["width"], data["height"]
    fov_v_deg = frames[0]["fovVertical"]
    for fr in frames:
        assert fr["fovVertical"] == fov_v_deg, \
            "fovVertical varies across frames -- this script assumes a fixed " \
            "focal length; extend cameras.txt to one entry per frame if not."

    # Detect (but do NOT drop) a duplicate closing frame -- with real
    # footage there is a real file for every JSON entry, so frame count
    # must match footage count exactly; dropping would break that 1:1 map.
    p0, pl = frames[0]["position"], frames[-1]["position"]
    if len(frames) > 1 and abs(p0["x"] - pl["x"]) < 1e-3 and abs(p0["y"] - pl["y"]) < 1e-3:
        print(f"Note: frame {n-1}'s pose duplicates frame 0's (orbit loop "
              f"closes). Keeping it -- there's a real image file for it and "
              f"COLMAP handles a near-duplicate view harmlessly. Not dropped.")

    # ---- Match real footage, or fall back to placeholder names ----
    if footage_dir:
        image_names = find_footage(footage_dir, dataset, n)
        print(f"Matched {n} real footage files in {footage_dir}")
    else:
        image_names = [f"frame_{i:04d}.jpeg" for i in range(n)]
        print(f"WARNING: no --footage-dir given -- using placeholder names "
              f"({image_names[0]} ... {image_names[-1]}). images.txt will "
              f"NOT match real files until you rename/copy footage to match.")

    pos_ecef = np.array([[fr["position"]["x"], fr["position"]["y"], fr["position"]["z"]]
                          for fr in frames])
    rot_deg = np.array([[fr["rotation"]["x"], fr["rotation"]["y"], fr["rotation"]["z"]]
                         for fr in frames])
    lat = np.array([fr["coordinate"]["latitude"] for fr in frames])
    lon = np.array([fr["coordinate"]["longitude"] for fr in frames])
    alt = np.array([fr["coordinate"]["altitude"] for fr in frames])

    # ---- ENU frame, origin at mean lat/lon/alt ----
    lat0, lon0, alt0 = lat.mean(), lon.mean(), alt.mean()
    lat0_r, lon0_r = np.radians(lat0), np.radians(lon0)
    origin_ecef = np.array([
        (EARTH_R + alt0) * np.cos(lat0_r) * np.cos(lon0_r),
        (EARTH_R + alt0) * np.cos(lat0_r) * np.sin(lon0_r),
        (EARTH_R + alt0) * np.sin(lat0_r),
    ])
    R_e2enu = np.array([
        [-np.sin(lon0_r), np.cos(lon0_r), 0],
        [-np.sin(lat0_r) * np.cos(lon0_r), -np.sin(lat0_r) * np.sin(lon0_r), np.cos(lat0_r)],
        [np.cos(lat0_r) * np.cos(lon0_r), np.cos(lat0_r) * np.sin(lon0_r), np.sin(lat0_r)],
    ])
    pos_enu = (pos_ecef - origin_ecef) @ R_e2enu.T

    # ---- Real intrinsics from fovVertical ----
    f_px = (height / 2) / np.tan(np.radians(fov_v_deg) / 2)
    cx, cy = width / 2.0, height / 2.0

    os.makedirs(out_dir, exist_ok=True)

    # cameras.txt -- one shared camera, real focal length
    with open(os.path.join(out_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Derived from fovVertical (Earth Studio export), NOT a "
                "generic FOV default. See module docstring.\n")
        f.write(f"1 SIMPLE_PINHOLE {width} {height} {f_px:.6f} {cx:.2f} {cy:.2f}\n")

    # images.txt -- known poses (world-to-camera), local ENU frame
    rx, ry, rz = np.radians(rot_deg[:, 0]), np.radians(rot_deg[:, 1]), np.radians(rot_deg[:, 2])
    lines = []
    lines.append("# Image list with two lines of data per image:\n")
    lines.append("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    lines.append("#   POINTS2D[] (empty here -- filled by feature extraction / matching)\n")
    lines.append("# Poses are KNOWN (from Earth Studio ground truth), local ENU frame.\n")
    lines.append(f"# ENU origin: lat={lat0:.8f} lon={lon0:.8f} alt={alt0:.3f}\n")

    geo_records = []
    R_world2cam_list = []
    for i in range(n):
        R_cam2global = rot_x(rx[i]) @ rot_y(ry[i]) @ rot_z(rz[i])
        R_cam2enu = R_e2enu @ R_cam2global               # camera-local -> ENU
        R_cam2enu_colmap = R_cam2enu @ LOCAL_TO_COLMAP.T  # apply axis flip
        R_world2cam = R_cam2enu_colmap.T                 # COLMAP wants world->cam
        t = -R_world2cam @ pos_enu[i]
        q = mat_to_quat_wxyz(R_world2cam)
        name = image_names[i]
        lines.append(
            f"{i+1} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} "
            f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} 1 {name}\n"
        )
        lines.append("\n")  # empty POINTS2D line
        R_world2cam_list.append(R_world2cam)
        geo_records.append({
            "image": name,
            "latitude": float(lat[i]),
            "longitude": float(lon[i]),
            "altitude": float(alt[i]),
            "enu_xyz": pos_enu[i].tolist(),
        })

    with open(os.path.join(out_dir, "images.txt"), "w") as f:
        f.writelines(lines)

    with open(os.path.join(out_dir, "points3D.txt"), "w") as f:
        f.write("# Empty -- populate via point_triangulator against these known poses.\n")

    geo = {
        "earth_model": "spherical, R=6371008.8m (IUGG mean radius) -- see module docstring",
        "enu_origin": {"latitude": lat0, "longitude": lon0, "altitude": alt0},
        "focal_length_px": f_px,
        "fov_vertical_deg": fov_v_deg,
        "principal_point": [cx, cy],
        "num_frames": n,
        "images": geo_records,
    }
    with open(os.path.join(out_dir, "geo_registration.json"), "w") as f:
        json.dump(geo, f, indent=2)

    print(f"Wrote {n} known poses to {out_dir}/")
    print(f"  focal length: {f_px:.2f}px (from fovVertical={fov_v_deg} deg)")
    print(f"  ENU origin: lat={lat0:.6f} lon={lon0:.6f} alt={alt0:.2f}")
    print(f"  orbit radius (ENU horiz extent): "
          f"~{np.abs(pos_enu[:, :2]).max():.1f} m")

    # ---- Copy footage into raw_images_<dataset>, optionally with real GPS EXIF ----
    if footage_dir and images_out_dir and copy_images:
        os.makedirs(images_out_dir, exist_ok=True)
        exif_ok = False
        if write_exif:
            try:
                import piexif
                exif_ok = True
            except ImportError:
                print("Note: piexif not installed -- copying footage without "
                      "GPS EXIF (not required for the known-pose pipeline, "
                      "just nice-to-have metadata). `pip install piexif` to enable.")

        failed = []
        for i, name in enumerate(image_names):
            src = os.path.join(footage_dir, name)
            dst = os.path.join(images_out_dir, name)
            try:
                shutil.copy2(src, dst)
                if exif_ok:
                    _write_gps_exif(dst, lat[i], lon[i], alt[i])
            except Exception as e:
                msg = str(e) or type(e).__name__
                failed.append((name, msg))

        copied = n - len(failed)
        print(f"Copied {copied}/{n} frames to {images_out_dir}/"
              f"{' (with real GPS EXIF)' if exif_ok else ''}")
        if failed:
            print(f"⚠️  {len(failed)} frame(s) failed to copy/tag and were "
                  f"SKIPPED (not fatal, but check these -- a missing image "
                  f"will make its images.txt pose orphaned):")
            for name, err in failed[:10]:
                print(f"    {name}: {err}")
            if len(failed) > 10:
                print(f"    ... and {len(failed) - 10} more")
    elif footage_dir and not images_out_dir:
        print("Note: --footage-dir given without --images-out-dir -- "
              "skipped copying. Point sfm_runner.py directly at "
              "--footage-dir if that's intentional, otherwise set "
              "--images-out-dir to the raw_images_<dataset> convention.")


def _write_gps_exif(path, lat, lon, alt):
    import piexif

    def _deg_to_dms_rational(deg):
        deg = abs(deg)
        d = int(deg)
        m_full = (deg - d) * 60
        m = int(m_full)
        s = round((m_full - m) * 60 * 100)
        return [(d, 1), (m, 1), (s, 100)]

    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef: "N" if lat >= 0 else "S",
        piexif.GPSIFD.GPSLatitude: _deg_to_dms_rational(lat),
        piexif.GPSIFD.GPSLongitudeRef: "E" if lon >= 0 else "W",
        piexif.GPSIFD.GPSLongitude: _deg_to_dms_rational(lon),
        piexif.GPSIFD.GPSAltitudeRef: 0 if alt >= 0 else 1,
        piexif.GPSIFD.GPSAltitude: (int(round(abs(alt) * 100)), 100),
    }
    try:
        exif_dict = piexif.load(path)
    except Exception:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    exif_dict["GPS"] = gps_ifd
    piexif.insert(piexif.dump(exif_dict), path)


def _fix_console_encoding() -> None:
    """Avoid UnicodeEncodeError on Windows consoles using legacy codepages
    (cp1252 etc) — this script's warning prints include a non-ASCII char."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


if __name__ == "__main__":
    _fix_console_encoding()
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--dataset", required=True,
                    help="Dataset name -- used to match footage files named "
                         "'<dataset>_<n>.<ext>' and for logging.")
    ap.add_argument("--out-dir", required=True,
                    help="Where to write cameras.txt/images.txt/points3D.txt/geo_registration.json")
    ap.add_argument("--footage-dir", default=None,
                    help="Directory with real frames named '<dataset>_<n>.<ext>'. "
                         "Omit to write placeholder names instead (no copying).")
    ap.add_argument("--images-out-dir", default=None,
                    help="Where to copy matched footage (the raw_images_<dataset> "
                         "convention sfm_runner.py/dense_runner.py expect).")
    ap.add_argument("--no-copy-images", dest="copy_images", action="store_false")
    ap.add_argument("--no-exif", dest="write_exif", action="store_false")
    args = ap.parse_args()
    convert(args.json_path, args.dataset, args.out_dir,
            footage_dir=args.footage_dir, images_out_dir=args.images_out_dir,
            copy_images=args.copy_images, write_exif=args.write_exif)