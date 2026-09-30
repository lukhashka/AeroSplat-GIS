"""
compare_poses.py

Run AFTER COLMAP has done a bundle-adjustment pass seeded from
earthstudio_to_colmap.py's known poses (refinement enabled, not frozen).

Compares COLMAP's refined images.txt against the known-pose images.txt to
answer the actual question that matters: did BA converge close to ground
truth (confirms the conversion is correct, and quantifies real residual
error), or did it diverge significantly (flags a bug upstream in the
conversion -- rotation convention, axis flip, or ENU math)?

Usage:
    python compare_poses.py colmap_init/images.txt path/to/colmap_output/sparse/0/images.txt
"""
import argparse
import numpy as np


def parse_images_txt(path):
    poses = {}
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if len(parts) < 10:
            continue
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = parts[9]
        poses[name] = {
            "q": np.array([qw, qx, qy, qz]),
            "t": np.array([tx, ty, tz]),
        }
    return poses


def quat_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def camera_center(q, t):
    # images.txt stores world->camera (R, t); camera center C = -R^T t
    R = quat_to_R(q)
    return -R.T @ t


def compare(known_path, refined_path):
    known = parse_images_txt(known_path)
    refined = parse_images_txt(refined_path)

    common = sorted(set(known) & set(refined))
    missing = sorted(set(known) - set(refined))
    if missing:
        print(f"WARNING: {len(missing)} images in known poses were not "
              f"registered by COLMAP: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if not common:
        print("No common images -- check file paths / naming.")
        return

    pos_errs, ang_errs = [], []
    for name in common:
        c_known = camera_center(known[name]["q"], known[name]["t"])
        c_refined = camera_center(refined[name]["q"], refined[name]["t"])
        pos_errs.append(np.linalg.norm(c_known - c_refined))

        R_known = quat_to_R(known[name]["q"])
        R_refined = quat_to_R(refined[name]["q"])
        R_diff = R_known.T @ R_refined
        cos_ang = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
        ang_errs.append(np.degrees(np.arccos(cos_ang)))

    pos_errs, ang_errs = np.array(pos_errs), np.array(ang_errs)
    print(f"\nCompared {len(common)} images (of {len(known)} known, "
          f"{len(refined)} refined).")
    print(f"Position residual   mean={pos_errs.mean():.4f} m   "
          f"median={np.median(pos_errs):.4f} m   max={pos_errs.max():.4f} m")
    print(f"Rotation residual   mean={ang_errs.mean():.4f} deg   "
          f"median={np.median(ang_errs):.4f} deg   max={ang_errs.max():.4f} deg")

    worst = np.argsort(-pos_errs)[:5]
    print("\nWorst 5 by position residual:")
    for i in worst:
        print(f"  {common[i]}: pos_err={pos_errs[i]:.4f}m  ang_err={ang_errs[i]:.4f}deg")

    print()
    if pos_errs.mean() < 0.5 and ang_errs.mean() < 0.5:
        print("-> Small residuals: conversion looks correct, BA is absorbing "
              "ordinary feature-matching noise. Safe to freeze known poses "
              "for a faster final run if desired.")
    elif pos_errs.mean() < 5.0 and ang_errs.mean() < 2.0:
        print("-> Moderate residuals: plausible, but worth checking the "
              "worst-offender frames individually before trusting the scene.")
    else:
        print("-> Large residuals: something in the conversion is likely "
              "wrong (rotation convention, axis flip, or ENU origin), OR "
              "COLMAP's own SfM solve is unreliable on this footage "
              "independent of the seed. Check whether errors correlate with "
              "specific frames (e.g. always near 0/180 deg in the orbit -- "
              "gimbal-lock-adjacent angles are a common place for an Euler "
              "convention bug to show up unevenly) before concluding either way.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("known_images_txt")
    ap.add_argument("refined_images_txt")
    args = ap.parse_args()
    compare(args.known_images_txt, args.refined_images_txt)