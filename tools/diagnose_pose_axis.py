"""
diagnose_pose_axis.py

The empirical test for the LOCAL_TO_COLMAP axis-flip question flagged (and
left unverified) in earthstudio_to_colmap.py. Matching already confirmed
healthy (inspect_matches.py) -- if triangulation still fails, the remaining
candidate is that our known poses are internally self-consistent (that's
what validate_convention.py checked) but WRONG relative to what the camera
actually saw, because the local-camera-axis-to-COLMAP-axis remap is off.

This pulls REAL matched keypoint pairs straight out of database.db (not
synthetic/guessed data), builds camera poses under a few candidate axis
conventions, and for each candidate does real two-view linear triangulation
on those real correspondences, checking:
  - cheirality: does the triangulated point land in FRONT of both cameras?
  - reprojection error: does it land back near the actual matched pixels?

The correct convention should show ~100% cheirality pass and low (sub- to
few-pixel) reprojection error. A wrong one typically fails almost totally
on one or both -- which is exactly what should distinguish them sharply
given how badly triangulation is currently failing.

Usage:
    python diagnose_pose_axis.py \\
        --db data/colmap_output_Untitled/database.db \\
        --json data/Untitled/Untitled.json \\
        --dataset Untitled
"""
import argparse
import sqlite3

import numpy as np

from aerosplat.geo.earthstudio_to_colmap import EARTH_R, load_frames, rot_x, rot_y, rot_z  # noqa: E402

# The only 4 axis remaps that keep a proper (determinant +1) rotation --
# i.e. an even number of sign flips. Odd-flip combinations are reflections,
# not valid camera-axis relabelings, so they're not physically meaningful
# candidates and are excluded.
CANDIDATES = {
    "identity (no flip)":            np.diag([1.0, 1.0, 1.0]),
    "flip Y,Z (original assumption)": np.diag([1.0, -1.0, -1.0]),
    "flip X,Z":                      np.diag([-1.0, 1.0, -1.0]),
    "flip X,Y":                      np.diag([-1.0, -1.0, 1.0]),
}


def pair_id_to_image_ids(pair_id):
    K = 2147483647
    image_id2 = pair_id % K
    image_id1 = (pair_id - image_id2) // K
    return int(image_id1), int(image_id2)


def read_keypoints(cur, image_id):
    cur.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?", (image_id,))
    row = cur.fetchone()
    if row is None or row[2] is None:
        return np.zeros((0, 2))
    rows, cols, data = row
    arr = np.frombuffer(data, dtype=np.float32).reshape(rows, cols)
    return arr[:, :2]  # x, y -- ignore scale/orientation/affine columns if present


def compute_enu_and_camera_poses(json_path):
    """Recompute, per frame: R_cam2global (validated Euler composition) and
    pos_enu (validated ECEF->ENU). Returns dict name -> (R_cam2global, pos_enu)."""
    data, frames = load_frames(json_path)
    lat = np.array([fr["coordinate"]["latitude"] for fr in frames])
    lon = np.array([fr["coordinate"]["longitude"] for fr in frames])
    alt = np.array([fr["coordinate"]["altitude"] for fr in frames])
    pos_ecef = np.array([[fr["position"]["x"], fr["position"]["y"], fr["position"]["z"]]
                          for fr in frames])
    rot_deg = np.array([[fr["rotation"]["x"], fr["rotation"]["y"], fr["rotation"]["z"]]
                         for fr in frames])

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

    rx, ry, rz = np.radians(rot_deg[:, 0]), np.radians(rot_deg[:, 1]), np.radians(rot_deg[:, 2])
    fov_v_deg = frames[0]["fovVertical"]
    f_px = (data["height"] / 2) / np.tan(np.radians(fov_v_deg) / 2)
    cx, cy = data["width"] / 2.0, data["height"] / 2.0

    result = {}
    n = len(frames)
    for i in range(n):
        R_cam2global = rot_x(rx[i]) @ rot_y(ry[i]) @ rot_z(rz[i])
        R_cam2enu = R_e2enu @ R_cam2global
        result[i] = (R_cam2enu, pos_enu[i])
    return result, f_px, cx, cy, n


def triangulate_dlt(P1, P2, pt1, pt2):
    """Standard linear (DLT) two-view triangulation for one correspondence."""
    A = np.array([
        pt1[0] * P1[2] - P1[0],
        pt1[1] * P1[2] - P1[1],
        pt2[0] * P2[2] - P2[0],
        pt2[1] * P2[2] - P2[1],
    ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def evaluate_candidate(name, M, poses_by_idx, f_px, cx, cy, pairs_data, max_pts=300):
    K = np.array([[f_px, 0, cx], [0, f_px, cy], [0, 0, 1]])
    cheirality_pass = 0
    total = 0
    reproj_errs = []

    for (idx1, idx2, kp1_all, kp2_all, matches) in pairs_data:
        R_enu1, t_enu1 = poses_by_idx[idx1]
        R_enu2, t_enu2 = poses_by_idx[idx2]

        R1 = (R_enu1 @ M.T).T
        t1 = -R1 @ t_enu1
        R2 = (R_enu2 @ M.T).T
        t2 = -R2 @ t_enu2

        P1 = K @ np.hstack([R1, t1.reshape(3, 1)])
        P2 = K @ np.hstack([R2, t2.reshape(3, 1)])

        n_sample = min(max_pts, len(matches))
        idxs = np.random.default_rng(0).choice(len(matches), n_sample, replace=False)
        for m in matches[idxs]:
            k1, k2 = int(m[0]), int(m[1])
            if k1 >= len(kp1_all) or k2 >= len(kp2_all):
                continue
            pt1, pt2 = kp1_all[k1], kp2_all[k2]
            X = triangulate_dlt(P1, P2, pt1, pt2)

            depth1 = (R1 @ X + t1)[2]
            depth2 = (R2 @ X + t2)[2]
            total += 1
            if depth1 > 0 and depth2 > 0:
                cheirality_pass += 1
                for P, pt in ((P1, pt1), (P2, pt2)):
                    proj = P @ np.append(X, 1.0)
                    proj = proj[:2] / proj[2]
                    reproj_errs.append(np.linalg.norm(proj - pt))

    pass_rate = 100 * cheirality_pass / total if total else 0.0
    med_err = np.median(reproj_errs) if reproj_errs else float("nan")
    return pass_rate, med_err, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--min-sep", type=int, default=8,
                     help="Minimum frame-index separation for a pair to be used "
                          "(avoids near-zero-baseline adjacent-frame pairs)")
    ap.add_argument("--max-sep", type=int, default=35)
    ap.add_argument("--num-pairs", type=int, default=6)
    args = ap.parse_args()

    poses_by_idx, f_px, cx, cy, n_frames = compute_enu_and_camera_poses(args.json)
    print(f"Loaded {n_frames} known poses. f={f_px:.2f}px cx={cx} cy={cy}\n")

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()
    cur.execute("SELECT image_id, name FROM images")
    id_to_name = {row[0]: row[1] for row in cur.fetchall()}
    name_to_idx = {}
    import re
    for iid, name in id_to_name.items():
        m = re.search(r"_(\d+)\.", name)
        if m:
            name_to_idx[iid] = int(m.group(1))

    cur.execute("SELECT pair_id, rows, data FROM two_view_geometries")
    candidates_pairs = []
    for pair_id, rows, data in cur.fetchall():
        if not rows or rows < 50:
            continue
        i1, i2 = pair_id_to_image_ids(pair_id)
        if i1 not in name_to_idx or i2 not in name_to_idx:
            continue
        sep = abs(name_to_idx[i1] - name_to_idx[i2])
        if args.min_sep <= sep <= args.max_sep:
            candidates_pairs.append((pair_id, i1, i2, rows, data))

    candidates_pairs.sort(key=lambda t: -t[3])
    chosen = candidates_pairs[:args.num_pairs]
    if not chosen:
        print("No suitable pairs found in the requested separation range -- "
              "widen --min-sep/--max-sep.")
        return
    print(f"Using {len(chosen)} pairs (separation {args.min_sep}-{args.max_sep} frames):")

    pairs_data = []
    for pair_id, i1, i2, rows, data in chosen:
        idx1, idx2 = name_to_idx[i1], name_to_idx[i2]
        print(f"  {id_to_name[i1]} <-> {id_to_name[i2]}  ({rows} matches)")
        matches = np.frombuffer(data, dtype=np.uint32).reshape(rows, 2)
        kp1 = read_keypoints(cur, i1)
        kp2 = read_keypoints(cur, i2)
        pairs_data.append((idx1, idx2, kp1, kp2, matches))
    print()

    print(f"{'Candidate':<32} {'cheirality pass':>16} {'median reproj err (px)':>24} {'N':>8}")
    print("-" * 84)
    results = []
    for name, M in CANDIDATES.items():
        pass_rate, med_err, total = evaluate_candidate(
            name, M, poses_by_idx, f_px, cx, cy, pairs_data)
        results.append((name, pass_rate, med_err))
        print(f"{name:<32} {pass_rate:>15.1f}% {med_err:>23.2f} {total:>8}")

    print()
    good = [(n, p, e) for (n, p, e) in results if p > 80 and not np.isnan(e) and e < 10]
    if good:
        best = min(good, key=lambda t: t[2])
        print(f"-> Winner: '{best[0]}' (cheirality {best[1]:.1f}%, "
              f"median reprojection error {best[2]:.2f}px). This is the "
              f"axis convention that's actually consistent with your real "
              f"footage -- update LOCAL_TO_COLMAP in earthstudio_to_colmap.py "
              f"to match and regenerate the known-poses seed.")
    else:
        print("-> No candidate cleared a reasonable bar (>80% cheirality, "
              "<10px median error). The problem may not be a pure axis-sign "
              "flip -- possible causes: an axis PERMUTATION rather than a "
              "sign flip (e.g. camera-local Y and X roles swapped), a wrong "
              "rotation order/composition, or an intrinsics mismatch. Share "
              "this table and we'll dig further.")


if __name__ == "__main__":
    main()