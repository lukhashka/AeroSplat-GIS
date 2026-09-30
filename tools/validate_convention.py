"""
Validation pass before committing to a converter.

Two things need to be established from data, not assumed:
  1. ECEF -> local ENU conversion is correct (sanity: reprojecting lat/lon/alt
     through the same ellipsoid should match the given ECEF positions).
  2. Which Euler order / intrinsic-vs-extrinsic / forward-axis convention
     Earth Studio used for `rotation`, found by searching for the convention
     under which every frame's forward vector actually points at the orbit
     center (which we know independently, from the position data alone).
"""
import argparse
import json
import itertools
import numpy as np

# CONFIRMED by direct residual test against this dataset (see session notes):
# Earth Studio's `position` field is NOT WGS84 ellipsoidal ECEF. It is
# geocentric coordinates on a SPHERICAL earth model, computed directly from
# geodetic lat/lon/alt as if they were spherical angles, using R = IUGG mean
# radius. True WGS84 ellipsoidal conversion gives a systematic ~21.1 km
# error here; this spherical model matches to ~1.2 m (residual float noise).
EARTH_R = 6371008.8

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("json_path", help="Earth Studio 3D-tracking JSON (e.g. data/dataset_v6/dataset_v6.json)")
_args = _ap.parse_args()

with open(_args.json_path, encoding="utf-8") as f:
    data = json.load(f)

frames = data["cameraFrames"]

# Drop the duplicate closing frame (orbit loop closes: frame[-1] == frame[0])
p0 = frames[0]["position"]
pl = frames[-1]["position"]
if abs(p0["x"] - pl["x"]) < 1e-3 and abs(p0["y"] - pl["y"]) < 1e-3:
    print(f"Dropping duplicate closing frame (frame {len(frames)-1} == frame 0)")
    frames = frames[:-1]

n = len(frames)
print(f"{n} usable frames, fovVertical={frames[0]['fovVertical']}, "
      f"w={data['width']} h={data['height']}")

pos_ecef = np.array([[fr["position"]["x"], fr["position"]["y"], fr["position"]["z"]] for fr in frames])
rot_deg = np.array([[fr["rotation"]["x"], fr["rotation"]["y"], fr["rotation"]["z"]] for fr in frames])
lat = np.array([fr["coordinate"]["latitude"] for fr in frames])
lon = np.array([fr["coordinate"]["longitude"] for fr in frames])
alt = np.array([fr["coordinate"]["altitude"] for fr in frames])

# ---- 1. Sanity check: recompute geocentric position from lat/lon/alt on a
#          SPHERE (confirmed model for this field), compare ----
lat_r, lon_r = np.radians(lat), np.radians(lon)
x_chk = (EARTH_R + alt) * np.cos(lat_r) * np.cos(lon_r)
y_chk = (EARTH_R + alt) * np.cos(lat_r) * np.sin(lon_r)
z_chk = (EARTH_R + alt) * np.sin(lat_r)
ecef_chk = np.stack([x_chk, y_chk, z_chk], axis=1)

err = np.linalg.norm(ecef_chk - pos_ecef, axis=1)
print(f"\nSpherical-geocentric reproduction from lat/lon/alt: mean err {err.mean():.3f} m, "
      f"max err {err.max():.3f} m")
if err.max() > 5.0:
    print("  -> WARNING: residual larger than expected, investigate before trusting ENU conversion.")
else:
    print("  -> Confirms `position` is spherical geocentric coords (R=IUGG mean radius), "
          "matching to float-noise precision.")

# ---- 2. Geocentric-spherical -> local ENU, origin at the mean lat/lon/alt ----
lat0, lon0, alt0 = lat.mean(), lon.mean(), alt.mean()
lat0_r, lon0_r = np.radians(lat0), np.radians(lon0)
origin_ecef = np.array([
    (EARTH_R + alt0) * np.cos(lat0_r) * np.cos(lon0_r),
    (EARTH_R + alt0) * np.cos(lat0_r) * np.sin(lon0_r),
    (EARTH_R + alt0) * np.sin(lat0_r),
])

R_ecef2enu = np.array([
    [-np.sin(lon0_r),                  np.cos(lon0_r),                 0],
    [-np.sin(lat0_r) * np.cos(lon0_r), -np.sin(lat0_r) * np.sin(lon0_r), np.cos(lat0_r)],
    [ np.cos(lat0_r) * np.cos(lon0_r),  np.cos(lat0_r) * np.sin(lon0_r), np.sin(lat0_r)],
])

pos_enu = (pos_ecef - origin_ecef) @ R_ecef2enu.T
print(f"\nENU origin: lat={lat0:.6f} lon={lon0:.6f} alt={alt0:.2f}")
print(f"ENU extent: E [{pos_enu[:,0].min():.1f}, {pos_enu[:,0].max():.1f}], "
      f"N [{pos_enu[:,1].min():.1f}, {pos_enu[:,1].max():.1f}], "
      f"U [{pos_enu[:,2].min():.1f}, {pos_enu[:,2].max():.1f}]")

def ray_convergence_residual(origins, dirs):
    """
    Least-squares intersection point of N 3D rays, and the mean perpendicular
    distance from that point back to each ray. For a camera truly orbiting a
    fixed target, the CORRECT rotation convention's rays should converge to
    a near-zero residual regardless of the (unknown) target elevation --
    this avoids needing to guess target height at all.
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for p, d in zip(origins, dirs):
        Pm = np.eye(3) - np.outer(d, d)
        A += Pm
        b += Pm @ p
    X = np.linalg.solve(A, b)
    perp = origins - X
    perp = perp - np.einsum("ni,ni->n", perp, dirs)[:, None] * dirs
    dist = np.linalg.norm(perp, axis=1)
    return X, dist.mean(), dist.max()

# ---- 3. Search Euler convention: order x intrinsic/extrinsic x forward axis ----
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
AX = {"X": rot_x, "Y": rot_y, "Z": rot_z}

rx, ry, rz = np.radians(rot_deg[:, 0]), np.radians(rot_deg[:, 1]), np.radians(rot_deg[:, 2])

best = None
orders = ["".join(p) for p in itertools.permutations("XYZ")]
forward_axes = {"+X": np.array([1,0,0]), "-X": np.array([-1,0,0]),
                 "+Y": np.array([0,1,0]), "-Y": np.array([0,-1,0]),
                 "+Z": np.array([0,0,1]), "-Z": np.array([0,0,-1])}

for order in orders:
    for mode in ("intrinsic_local2world", "extrinsic_world2local_T"):
        # Build per-frame rotation matrix applying rotations in `order`
        # sequence, each about the *world* axis (extrinsic composition,
        # left-multiplied in listed order) -- the common AE/Earth-Studio style.
        Rs = np.empty((n, 3, 3))
        for i in range(n):
            ang = {"X": rx[i], "Y": ry[i], "Z": rz[i]}
            R = np.eye(3)
            for axis in order:
                R = AX[axis](ang[axis]) @ R
            if mode == "extrinsic_world2local_T":
                R = R.T
            Rs[i] = R
        for fa_name, fa in forward_axes.items():
            fwd = np.einsum("nij,j->ni", Rs, fa)
            fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
            try:
                X, mean_res, max_res = ray_convergence_residual(pos_enu, fwd)
            except np.linalg.LinAlgError:
                continue
            if best is None or mean_res < best[0]:
                best = (mean_res, order, mode, fa_name, max_res, X)

print(f"\nBest convention found: order={best[1]}, mode={best[2]}, forward_axis={best[3]}")
print(f"  ray convergence residual: mean {best[0]:.4f} m, max {best[4]:.4f} m")
print(f"  converged target point (ENU): {best[5]}")
if best[0] < 1.0:
    print("  -> Excellent convergence. Convention is essentially certainly correct.")
elif best[0] < 20.0:
    print("  -> Good match, plausible correct convention.")
else:
    print("  -> Poor convergence even at the best candidate. Something else "
          "is wrong -- camera may not be a fixed-target orbit, or rotation "
          "isn't simple Euler composition (e.g. per-axis order differs from "
          "any of the 6 tested, or there's an additional coordinate-system "
          "flip between rotation and position).")

# Report the runner-up conventions too, for context
print("\nTop 5 candidates by ray-convergence residual:")
all_results = []
for order in orders:
    for mode in ("intrinsic_local2world", "extrinsic_world2local_T"):
        Rs = np.empty((n, 3, 3))
        for i in range(n):
            ang = {"X": rx[i], "Y": ry[i], "Z": rz[i]}
            R = np.eye(3)
            for axis in order:
                R = AX[axis](ang[axis]) @ R
            if mode == "extrinsic_world2local_T":
                R = R.T
            Rs[i] = R
        for fa_name, fa in forward_axes.items():
            fwd = np.einsum("nij,j->ni", Rs, fa)
            fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
            try:
                X, mean_res, max_res = ray_convergence_residual(pos_enu, fwd)
            except np.linalg.LinAlgError:
                continue
            all_results.append((mean_res, order, mode, fa_name))
all_results.sort(key=lambda r: r[0])
for r in all_results[:5]:
    print(f"  residual={r[0]:.3f}m  order={r[1]}  mode={r[2]}  forward={r[3]}")

# ---- 4. Real focal length from fovVertical ----
fov_v = np.radians(data["cameraFrames"][0]["fovVertical"])
f_px = (data["height"] / 2) / np.tan(fov_v / 2)
print(f"\nReal focal length from fovVertical={data['cameraFrames'][0]['fovVertical']} deg: "
      f"f = {f_px:.2f} px  (principal point assumed at ({data['width']/2}, {data['height']/2}))")