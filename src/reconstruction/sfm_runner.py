#!/usr/bin/env python3
"""
sfm_runner.py — COLMAP SfM stage for AeroSplat-GIS (optimized).

Changes vs. the original:
  - Matcher strategy is selectable; default is `spatial`, which uses the
    exact GPS that exif_injector.py already writes into every frame —
    avoids O(N^2) exhaustive_matcher, which is pure waste on an orbital
    flight where opposite-side frames share no overlap.
  - Optional exact pinhole intrinsics for the Earth Studio virtual camera,
    frozen during bundle adjustment (--known-intrinsics, on by default).
  - Optional GLOMAP mapper as a drop-in for `colmap mapper` (same DB,
    same sparse-model format), reported to be much faster on large sets.
  - Auto-detects, PER OPTION, whether this COLMAP build uses the old
    (SiftExtraction/SiftMatching) or new (FeatureExtraction/FeatureMatching)
    flag prefix. The rename was not applied to the whole option group at
    once, so one build can need --FeatureExtraction.max_image_size and
    --SiftExtraction.domain_size_pooling simultaneously. Options this
    build has under neither prefix are dropped with a warning instead of
    crashing the run.
  - --fresh flag for a clean rerun (avoids silently appending to a stale DB).
  - Per-stage timing, output verification, and a real process exit code.
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import time


from aerosplat.logging_utils import fix_console_encoding as _fix_console_encoding


def log(msg):
    print(msg, flush=True)


def run(cmd, step_name):
    log(f"    $ {' '.join(cmd)}")
    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        log(f"❌ Помилка на етапі «{step_name}»: {e}")
        raise
    except FileNotFoundError:
        log(f"❌ Команда '{cmd[0]}' не знайдена в PATH.")
        raise
    dt = time.perf_counter() - t0
    log(f"✅ {step_name} завершено за {dt:.1f}s\n")
    return dt


_HELP_CACHE = {}


def colmap_help(colmap_cmd, subcommand):
    """Fetch (and cache) a subcommand's help text.

    NOTE: COLMAP prints help to stderr, not stdout — capturing only stdout
    yields an empty string and silently breaks every probe below.
    """
    if subcommand not in _HELP_CACHE:
        try:
            r = subprocess.run(
                [colmap_cmd, subcommand, "-h"],
                capture_output=True, text=True, check=False
            )
            _HELP_CACHE[subcommand] = (r.stdout or "") + (r.stderr or "")
        except FileNotFoundError:
            _HELP_CACHE[subcommand] = ""
    return _HELP_CACHE[subcommand]


EXT_GROUPS = ("FeatureExtraction", "SiftExtraction")
MATCH_GROUPS = ("FeatureMatching", "SiftMatching")


def opt(colmap_cmd, subcommand, option, groups):
    """Resolve --Group.option against what THIS build actually accepts.

    COLMAP's Sift* -> Feature* rename happened per-option, not per-group:
    a single build can expose --FeatureExtraction.max_image_size while
    still using --SiftExtraction.domain_size_pooling. So the group has to
    be resolved for each option separately, not once for the whole stage.

    Returns None if no group exposes the option (build doesn't support it).
    """
    help_text = colmap_help(colmap_cmd, subcommand)
    for g in groups:
        if f"--{g}.{option}" in help_text:
            return f"--{g}.{option}"
    return None


def ext_opt(colmap_cmd, option, value):
    """[flag, value] for a feature_extractor option, or [] if unsupported."""
    flag = opt(colmap_cmd, "feature_extractor", option, EXT_GROUPS)
    if flag is None:
        log(f"    ⚠️  Ця збірка COLMAP не має опції '{option}' — пропускаю")
        return []
    return [flag, str(value)]


def match_opt(colmap_cmd, subcommand, option, value):
    """[flag, value] for a matcher option, or [] if unsupported."""
    flag = opt(colmap_cmd, subcommand, option, MATCH_GROUPS)
    if flag is None:
        log(f"    ⚠️  Ця збірка COLMAP не має опції '{option}' — пропускаю")
        return []
    return [flag, str(value)]


def pinhole_params(width, height, fov_deg, fov_axis="horizontal"):
    """Exact intrinsics for an Earth Studio virtual pinhole camera — no need
    to let COLMAP *estimate* what the render settings already give exactly.

    NOTE ON fov_axis: Earth Studio (and this project's real JSON exports)
    report `fovVertical`, not horizontal FOV. This function originally
    always treated `fov_deg` as HORIZONTAL (computing fx from `width`),
    which silently gives the WRONG focal length if you pass a vertical FOV
    value straight through -- e.g. fovVertical=20deg gives fx=3062px via
    the height/vertical formula but would give a wrong ~5443px if run
    through the width/horizontal formula unchanged. Pass fov_axis="vertical"
    when your fov_deg came from a `fovVertical` field (as with
    earthstudio_to_colmap.py's output) to get the correct value.

    Prefer --camera-params over this entirely when you have a real,
    already-computed focal length (e.g. from earthstudio_to_colmap.py's
    cameras.txt) instead of round-tripping through a FOV value at all.
    """
    fov_rad = math.radians(fov_deg)
    if fov_axis == "vertical":
        f = (height / 2.0) / math.tan(fov_rad / 2.0)
    else:
        f = (width / 2.0) / math.tan(fov_rad / 2.0)
    cx, cy = width / 2.0, height / 2.0
    return f"{f},{cx},{cy}"


def read_known_camera(known_poses_dir):
    """Read CAMERA_MODEL/PARAMS from a known-poses cameras.txt (as written
    by earthstudio_to_colmap.py) so feature_extractor uses the real,
    already-derived intrinsics directly instead of re-deriving them from a
    FOV value (and the horizontal/vertical ambiguity that entails)."""
    path = os.path.join(known_poses_dir, "cameras.txt")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            model = parts[1]
            params = ",".join(parts[4:])
            return model, params
    raise ValueError(f"No camera found in {path}")


def remap_known_poses_ids(known_poses_dir, db_path, out_dir):
    """
    Rewrite a known-poses images.txt so IMAGE_ID matches what THIS database
    actually assigned to each filename, and copy cameras.txt/points3D.txt
    alongside it unchanged.

    feature_extractor assigns image_id by its own directory-scan order,
    which is NOT guaranteed to match the numeric filename order
    earthstudio_to_colmap.py assumes when it writes images.txt (it has to
    guess, since it runs before feature_extractor creates the database at
    all). point_triangulator attaches a known pose to a database image BY
    ID, not by name -- a wrong ID silently attaches the wrong pose to the
    wrong image, or, on builds that cross-check this internally, aborts
    with "Check failed: existing_frame.DataIds() == frame.DataIds()"
    instead of corrupting the reconstruction quietly. Confirmed empirically
    on this project: feature_extractor's scan order swapped IMAGE_ID 18/19
    and 26/27 relative to earthstudio_to_colmap.py's numeric assumption.
    """
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        name_to_id = {name: img_id for img_id, name in
                     conn.execute("SELECT image_id, name FROM images")}
    finally:
        conn.close()

    os.makedirs(out_dir, exist_ok=True)
    shutil.copy2(os.path.join(known_poses_dir, "cameras.txt"),
                os.path.join(out_dir, "cameras.txt"))
    shutil.copy2(os.path.join(known_poses_dir, "points3D.txt"),
                os.path.join(out_dir, "points3D.txt"))

    with open(os.path.join(known_poses_dir, "images.txt")) as f:
        lines = f.readlines()

    out_lines = []
    remapped = 0
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            out_lines.append(lines[i])
            i += 1
            continue
        parts = stripped.split()
        old_id, name = parts[0], parts[-1]
        if name not in name_to_id:
            raise ValueError(
                f"{name} (known-poses IMAGE_ID {old_id}) not found in "
                f"{db_path} -- feature_extractor did not register this "
                f"image. Check --images-dir matches the known-poses footage.")
        new_id = name_to_id[name]
        if str(new_id) != old_id:
            remapped += 1
        parts[0] = str(new_id)
        out_lines.append(" ".join(parts) + "\n")
        # POINTS2D line: earthstudio_to_colmap.py always writes it empty.
        out_lines.append(lines[i + 1] if i + 1 < len(lines) else "\n")
        i += 2

    with open(os.path.join(out_dir, "images.txt"), "w") as f:
        f.writelines(out_lines)

    if remapped:
        log(f"    ℹ️  Remapped {remapped}/{len(name_to_id)} IMAGE_ID(s) "
            f"(database scan order != numeric filename order) -> {out_dir}")
    return out_dir


def verify_sparse_output(sparse_dir):
    model0 = os.path.join(sparse_dir, "0")
    has_bin = os.path.isfile(os.path.join(model0, "cameras.bin"))
    has_txt = os.path.isfile(os.path.join(model0, "cameras.txt"))
    if not (has_bin or has_txt):
        log(f"❌ Не знайдено {model0}/cameras.bin(.txt) — реконструкція не створила модель.")
        return False
    extra_models = [d for d in os.listdir(sparse_dir)
                     if d != "0" and os.path.isdir(os.path.join(sparse_dir, d))]
    if extra_models:
        log(f"⚠️  Знайдено додаткові підмоделі {extra_models} — "
            f"реконструкція фрагментована (замало збігів між кадрами?).")
    return True


def run_colmap_sfm(args):
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    dataset = args.dataset or os.environ.get("AEROSPLAT_DATASET", "zaporizhzhia")
    images_dir = args.images_dir or os.path.join(project_dir, "data", f"raw_images_{dataset}")
    output_dir = args.output_dir or os.path.join(project_dir, "data", f"colmap_output_{dataset}")

    db_path = os.path.join(output_dir, "database.db")
    sparse_dir = os.path.join(output_dir, "sparse")

    if args.known_poses_dir:
        known_abs = os.path.abspath(args.known_poses_dir)
        output_abs = os.path.abspath(output_dir)
        if known_abs == output_abs or known_abs.startswith(output_abs + os.sep):
            log(f"❌ --known-poses-dir ({args.known_poses_dir}) знаходиться "
                f"всередині --output-dir ({output_dir}). --fresh видаляє "
                f"output-dir цілком, а відомі пози — це вхідні дані, які "
                f"мають пережити --fresh. Перенеси known-poses-dir за межі "
                f"output-dir.")
            return False

    if args.fresh and os.path.exists(output_dir):
        log(f"🧹 --fresh: видаляю попередній {output_dir}")
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    log("🚀 === ЗАПУСК АВТОМАТИЗОВАНОГО ПАЙПЛАЙНУ COLMAP ===")
    log(f"📂 Вхідні геореферовані кадри: {images_dir}")
    log(f"💾 Вихідна папка реконструкції: {output_dir}")
    log(f"🔀 Матчинг: {args.matcher}   🗺️  Mapper: {args.mapper}\n")

    if not os.path.isdir(images_dir) or len(os.listdir(images_dir)) == 0:
        log(f"❌ Помилка: Папка {images_dir} порожня або не існує. Спочатку запусти інжектор!")
        return False

    if os.path.exists(db_path) and not args.fresh:
        log(f"⚠️  {db_path} вже існує — повторний запуск без --fresh може дублювати "
            f"фічі/зображення. Використай --fresh для чистого прогону.\n")

    colmap_cmd = args.colmap_bin
    timings = {}

    try:
        # --- 1. Feature extraction ---
        log("🔍 [Крок 1/3] Екстракція фіч з кадрів БПЛА...")
        extract_args = [
            colmap_cmd, "feature_extractor",
            "--database_path", db_path,
            "--image_path", images_dir,
            "--ImageReader.single_camera", "1",
            *ext_opt(colmap_cmd, "use_gpu", 1),
            *ext_opt(colmap_cmd, "max_image_size", args.max_image_size),
            *ext_opt(colmap_cmd, "estimate_affine_shape", 0),
            *ext_opt(colmap_cmd, "domain_size_pooling", 0),
        ]
        if args.known_poses_dir:
            # Real, already-derived intrinsics from earthstudio_to_colmap.py's
            # cameras.txt -- skip the FOV round-trip (and its horizontal/
            # vertical ambiguity, see pinhole_params docstring) entirely.
            model, params = read_known_camera(args.known_poses_dir)
            log(f"    ℹ️  Інтринсики з known-poses cameras.txt: {model} {params}")
            extract_args += [
                "--ImageReader.camera_model", model,
                "--ImageReader.camera_params", params,
            ]
        elif args.known_intrinsics:
            extract_args += [
                "--ImageReader.camera_model", "SIMPLE_PINHOLE",
                "--ImageReader.camera_params",
                pinhole_params(args.image_width, args.image_height, args.fov_deg,
                               fov_axis=args.fov_axis),
            ]
        timings["extract"] = run(extract_args, "Екстракція фіч")

        # --- 2. Matching ---
        matcher_subcmd = {
            "exhaustive": "exhaustive_matcher",
            "sequential": "sequential_matcher",
            "spatial": "spatial_matcher",
        }[args.matcher]
        log(f"🔗 [Крок 2/3] Зіставлення фіч ({args.matcher})...")
        match_gpu = match_opt(colmap_cmd, matcher_subcmd, "use_gpu", 1)

        if args.matcher == "exhaustive":
            match_args = [colmap_cmd, matcher_subcmd,
                           "--database_path", db_path, *match_gpu]
        elif args.matcher == "sequential":
            # Loop detection needs a vocab tree; without one older COLMAP builds
            # abort (P0-1), so it is only enabled when a tree is supplied.
            if args.loop_detection and not args.vocab_tree:
                print("  [matcher] no --vocab-tree -> loop detection disabled")
                args.loop_detection = False
            match_args = [
                colmap_cmd, matcher_subcmd,
                "--database_path", db_path, *match_gpu,
                "--SequentialMatching.overlap", str(args.seq_overlap),
                "--SequentialMatching.loop_detection", "1" if args.loop_detection else "0",
            ]
            if args.loop_detection and args.vocab_tree:
                match_args += ["--SequentialMatching.vocab_tree_path", args.vocab_tree]
        else:  # spatial — uses the exact GPS exif_injector.py already wrote in
            match_args = [
                colmap_cmd, matcher_subcmd,
                "--database_path", db_path, *match_gpu,
                "--SpatialMatching.max_num_neighbors", str(args.spatial_neighbors),
                "--SpatialMatching.max_distance", str(args.spatial_max_distance),
            ]
        timings["match"] = run(match_args, "Матчинг фіч")

        # --- 3. Mapping ---
        if args.known_poses_dir:
            # Known-pose seeding: skip incremental pose-guessing entirely.
            # point_triangulator reads the poses from known_poses_dir and
            # ONLY triangulates 3D points against them -- it cannot fail to
            # "register" an image the way incremental mapping can, since
            # every pose is already given. bundle_adjuster then optionally
            # refines everything (poses included) against real reprojection
            # error -- compare its output back to known_poses_dir afterward
            # with compare_poses.py to check how far it moved.
            log("📐 [Крок 3/3] Тріангуляція за відомими позами (point_triangulator)...")
            remapped_poses_dir = os.path.join(output_dir, "known_poses_remapped")
            remap_known_poses_ids(args.known_poses_dir, db_path, remapped_poses_dir)
            triangulated_dir = os.path.join(sparse_dir, "0")
            os.makedirs(triangulated_dir, exist_ok=True)
            triang_args = [
                colmap_cmd, "point_triangulator",
                "--database_path", db_path,
                "--image_path", images_dir,
                "--input_path", remapped_poses_dir,
                "--output_path", triangulated_dir,
            ]
            timings["triangulate"] = run(triang_args, "Тріангуляція (known poses)")

            if args.refine_known_poses:
                log("🔧 [Крок 3b/3] Bundle adjustment (уточнення відомих поз)...")
                ba_args = [
                    colmap_cmd, "bundle_adjuster",
                    "--input_path", triangulated_dir,
                    "--output_path", triangulated_dir,
                ]
                timings["bundle_adjust"] = run(ba_args, "Bundle adjustment")
                log(f"    ℹ️  Порівняй результат з відомими позами: "
                    f"python compare_poses.py "
                    f"{os.path.join(args.known_poses_dir, 'images.txt')} "
                    f"{os.path.join(triangulated_dir, 'images.txt')}")
            else:
                log("    ℹ️  --no-refine-known-poses: пози заморожені як є "
                    "(тільки точки тріангульовані).")
        else:
            log(f"📐 [Крок 3/3] Побудова розрідженої 3D-хмари ({args.mapper})...")
            if args.mapper == "glomap":
                mapper_args = ["glomap", "mapper",
                               "--database_path", db_path,
                               "--image_path", images_dir,
                               "--output_path", sparse_dir]
            else:
                mapper_args = [colmap_cmd, "mapper",
                               "--database_path", db_path,
                               "--image_path", images_dir,
                               "--output_path", sparse_dir]
                if args.known_intrinsics:
                    mapper_args += [
                        "--Mapper.ba_refine_focal_length", "0",
                        "--Mapper.ba_refine_principal_point", "0",
                        "--Mapper.ba_refine_extra_params", "0",
                    ]
            timings["map"] = run(mapper_args, "Mapping")

        ok = verify_sparse_output(sparse_dir)
        total = sum(timings.values())
        log("🎉 === РЕКОНСТРУКЦІЮ COLMAP ЗАВЕРШЕНО! ===")
        stage_timings = "  ".join(f"{k}={v:.1f}s" for k, v in timings.items())
        log(f"⏱️  {stage_timings}  total={total:.1f}s")
        log(f"📂 Результат: {os.path.join(sparse_dir, '0')}")
        return ok

    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def build_arg_parser():
    p = argparse.ArgumentParser(description="AeroSplat-GIS COLMAP SfM stage (optimized)")
    p.add_argument("--dataset", default=None, help="Overrides AEROSPLAT_DATASET env var")
    p.add_argument("--images-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--colmap-bin", default="colmap")
    p.add_argument("--fresh", action="store_true", help="Wipe existing DB/output before running")

    p.add_argument("--matcher", choices=["exhaustive", "sequential", "spatial"],
                    default="spatial",
                    help="spatial (default) uses the exact GPS exif_injector.py wrote in; "
                         "avoids exhaustive_matcher's O(N^2) cost on an orbital flight")
    p.add_argument("--seq-overlap", type=int, default=10)
    p.add_argument("--loop-detection", dest="loop_detection", action="store_true", default=True)
    p.add_argument("--no-loop-detection", dest="loop_detection", action="store_false",
                    help="Disable SequentialMatching loop closure (was previously impossible to "
                         "turn off — --loop-detection defaulted to True with no negation flag)")
    p.add_argument("--vocab-tree", default=None, help="Path to a vocab tree .bin for loop closure")
    p.add_argument("--spatial-neighbors", type=int, default=30)
    p.add_argument("--spatial-max-distance", type=float, default=200.0,
                    help="meters; should exceed the orbital radius (~350m) so opposite-side "
                         "frames near the same point in the loop can still match")

    p.add_argument("--mapper", choices=["colmap", "glomap"], default="colmap",
                    help="glomap is a much faster global-SfM drop-in; benchmark before switching default")

    p.add_argument("--max-image-size", type=int, default=1600)
    p.add_argument("--known-intrinsics", action="store_true", default=True,
                    help="Pin exact Earth Studio pinhole intrinsics and freeze them in BA "
                         "(ignored if --known-poses-dir is set -- that path uses the real "
                         "intrinsics from the known-poses cameras.txt directly)")
    p.add_argument("--no-known-intrinsics", dest="known_intrinsics", action="store_false")
    p.add_argument("--image-width", type=int, default=1920,
                    help="Must match actual Earth Studio export resolution")
    p.add_argument("--image-height", type=int, default=1080,
                    help="Must match actual Earth Studio export resolution")
    p.add_argument("--fov-deg", type=float, default=60.0,
                    help="Must match actual Earth Studio camera FOV setting")
    p.add_argument("--fov-axis", choices=["horizontal", "vertical"], default="horizontal",
                    help="Which axis --fov-deg measures. Earth Studio's exported "
                         "`fovVertical` field is VERTICAL -- pass "
                         "--fov-axis vertical --fov-deg <that value> if you're "
                         "sourcing --fov-deg from it, or better, skip this whole "
                         "path and use --known-poses-dir with real intrinsics instead.")

    p.add_argument("--known-poses-dir", default=None,
                    help="Directory with cameras.txt/images.txt/points3D.txt "
                         "(known ground-truth poses, e.g. from "
                         "earthstudio_to_colmap.py). When set, replaces the "
                         "incremental `mapper` with point_triangulator "
                         "(+ optional bundle_adjuster) seeded from these poses "
                         "instead of solving them from scratch -- see "
                         "gs_trainer_v3_notes.md / AeroSplat-GIS session notes "
                         "for why this is more accurate than incremental SfM "
                         "when real ground-truth poses are available.")
    p.add_argument("--refine-known-poses", action="store_true", default=True,
                    help="Run bundle_adjuster after point_triangulator to let "
                         "COLMAP refine the seeded poses against real "
                         "reprojection error, rather than freezing them. "
                         "Recommended -- check the refinement with compare_poses.py.")
    p.add_argument("--no-refine-known-poses", dest="refine_known_poses", action="store_false")
    return p


if __name__ == "__main__":
    _fix_console_encoding()
    cli_args = build_arg_parser().parse_args()
    success = run_colmap_sfm(cli_args)
    sys.exit(0 if success else 1)