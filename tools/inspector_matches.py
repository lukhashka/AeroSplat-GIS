"""
inspect_matches.py

Reads verified two-view matches straight out of a COLMAP database.db and
reports their distribution -- run this BEFORE assuming the pose/axis
conversion is at fault. If matches themselves are thin or concentrated in
a handful of pairs, that's a matching problem (independent of pose
correctness) and needs fixing first; only if matches look healthy does a
near-zero triangulation result point at the known-pose conversion.

Usage:
    python inspect_matches.py data/colmap_output_Untitled/database.db
"""
import sqlite3
import sys
import numpy as np


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) // 2147483647
    return int(image_id1), int(image_id2)


def main(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # image_id -> name
    cur.execute("SELECT image_id, name FROM images")
    names = {row[0]: row[1] for row in cur.fetchall()}
    n_images = len(names)
    print(f"images table: {n_images} images")

    # Raw matches (pre-geometric-verification)
    cur.execute("SELECT pair_id, rows FROM matches")
    raw = cur.fetchall()
    raw_counts = [r for (_, r) in raw if r is not None and r > 0]
    print(f"\nraw matches (pre-verification): {len(raw_counts)} image pairs "
          f"had any matches at all, {sum(raw_counts)} matches total")

    # Geometrically verified two-view matches -- what triangulation actually uses
    cur.execute("SELECT pair_id, rows, config FROM two_view_geometries")
    verified = cur.fetchall()
    verified_counts = [(pid, r) for (pid, r, cfg) in verified if r is not None and r > 0]
    total_verified = sum(r for (_, r) in verified_counts)
    print(f"\nverified (geometric) matches: {len(verified_counts)} image "
          f"pairs passed geometric verification, {total_verified} matches total")

    if not verified_counts:
        print("\n No verified two-view geometries at all -- matching itself "
              "is failing before pose/triangulation even comes into it. "
              "Check image content, --matcher settings, and whether "
              "--loop-detection is silently misbehaving without a "
              "--vocab-tree path.")
        return

    expected_pairs = n_images * (n_images - 1) / 2
    counts = np.array([r for (_, r) in verified_counts])
    print(f"\nPair coverage: {len(verified_counts)} / {expected_pairs:.0f} "
          f"possible pairs matched ({100*len(verified_counts)/expected_pairs:.1f}%)")
    print(f"Matches per verified pair: min={counts.min()} "
          f"median={np.median(counts):.0f} mean={counts.mean():.1f} "
          f"max={counts.max()}")

    # Flag pathological concentration (a handful of pairs carrying almost
    # all the matches) vs. healthy spread across many neighbor pairs
    sorted_counts = np.sort(counts)[::-1]
    top5_share = sorted_counts[:5].sum() / total_verified if total_verified else 0
    print(f"Top 5 pairs carry {100*top5_share:.1f}% of all verified matches")

    print("\nWeakest 10 verified pairs (by match count):")
    verified_counts.sort(key=lambda t: t[1])
    for pid, r in verified_counts[:10]:
        i1, i2 = pair_id_to_image_ids(pid)
        n1, n2 = names.get(i1, f"id{i1}"), names.get(i2, f"id{i2}")
        print(f"  {n1} <-> {n2}: {r} matches")

    print("\nStrongest 10 verified pairs:")
    for pid, r in verified_counts[-10:]:
        i1, i2 = pair_id_to_image_ids(pid)
        n1, n2 = names.get(i1, f"id{i1}"), names.get(i2, f"id{i2}")
        print(f"  {n1} <-> {n2}: {r} matches")

    print("\n--- Verdict ---")
    if counts.mean() < 30 or len(verified_counts) < n_images * 0.5:
        print("Matches look thin and/or sparse in coverage. This alone can "
              "explain a near-empty triangulation, independent of whether "
              "the pose conversion is correct. Try: --matcher exhaustive "
              "(cheap at 91 images), check --loop-detection is not silently "
              "no-op'ing without --vocab-tree, and sanity-check the actual "
              "image content (resolution, compression, real texture) "
              "before touching pose conversion code.")
    else:
        print("Matches look reasonably healthy (decent count and coverage). "
              "If triangulation STILL produces almost nothing against "
              "known poses despite this, that shifts the leading suspect "
              "back to the pose/axis conversion (LOCAL_TO_COLMAP in "
              "earthstudio_to_colmap.py) -- the matches are geometrically "
              "consistent among themselves but inconsistent with the "
              "supplied camera poses.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_matches.py path/to/database.db")
        sys.exit(1)
    main(sys.argv[1])