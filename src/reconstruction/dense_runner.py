import argparse
import os
import subprocess
import sys
import time


def _fix_console_encoding():
    """Avoid UnicodeEncodeError on Windows consoles using cp1251/cp866 locales."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def _run(cmd, step_label):
    """Run a subprocess step, time it, and propagate failure with a real exit code."""
    print(f"\n{step_label}")
    print("   " + " ".join(cmd))
    start = time.perf_counter()
    result = subprocess.run(cmd)
    elapsed = time.perf_counter() - start
    print(f"   ⏱  {elapsed:.1f}s")
    if result.returncode != 0:
        print(f"❌ Крок провалився (код {result.returncode}): {step_label}", file=sys.stderr)
        sys.exit(result.returncode)


def parse_args():
    parser = argparse.ArgumentParser(description="COLMAP dense reconstruction (MVS) runner")
    parser.add_argument(
        "--dataset",
        default=os.environ.get("AEROSPLAT_DATASET", "zaporizhzhia"),
        help="Dataset name, subdirectory suffix under data/ (default: $AEROSPLAT_DATASET or 'zaporizhzhia')",
    )
    parser.add_argument(
        "--colmap-bin",
        default=os.environ.get("COLMAP_BIN", "colmap"),
        help="Path to colmap executable (default: $COLMAP_BIN or 'colmap' on PATH)",
    )
    parser.add_argument("--gpu-index", default="0", help="GPU index for patch match stereo")
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=2000,
        help="Cap image resolution before stereo — biggest single speed lever (default: 2000)",
    )
    parser.add_argument(
        "--geom-consistency",
        choices=["true", "false"],
        default="true",
        help="'false' roughly halves stereo time at the cost of more outliers (default: true)",
    )
    parser.add_argument("--window-step", type=int, default=2, help="PatchMatchStereo.window_step (default 1 upstream; 2 is ~2x faster)")
    parser.add_argument("--num-samples", type=int, default=10, help="PatchMatchStereo.num_samples (default 15 upstream)")
    parser.add_argument("--num-iterations", type=int, default=3, help="PatchMatchStereo.num_iterations (default 5 upstream)")
    parser.add_argument("--cache-size", type=int, default=6, help="PatchMatchStereo.cache_size in GB")
    parser.add_argument("--min-num-pixels", type=int, default=8, help="StereoFusion.min_num_pixels (default 5 upstream) — quality filter, prefer this over post-hoc random subsampling")
    parser.add_argument("--max-reproj-error", type=float, default=1.5, help="StereoFusion.max_reproj_error (default 2 upstream)")
    parser.add_argument("--skip-undistort", action="store_true", help="Skip step 1 if already done")
    parser.add_argument("--skip-stereo", action="store_true", help="Skip step 2 if depth maps already exist")
    return parser.parse_args()


def run_colmap_dense():
    _fix_console_encoding()
    args = parse_args()

    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    images_dir = os.path.join(project_dir, "data", f"raw_images_{args.dataset}")
    sparse_dir = os.path.join(project_dir, "data", f"colmap_output_{args.dataset}", "sparse", "0")
    dense_dir = os.path.join(project_dir, "data", f"colmap_output_{args.dataset}", "dense")
    fused_ply_path = os.path.join(dense_dir, "fused.ply")

    # --- Pre-flight validation: fail fast with a clear message instead of a
    # cryptic COLMAP error after the subprocess has already spawned. ---
    if not args.skip_undistort:
        if not os.path.isdir(images_dir):
            print(f"❌ Не знайдено директорію зображень: {images_dir}", file=sys.stderr)
            sys.exit(1)
        if not os.path.isdir(sparse_dir):
            print(f"❌ Не знайдено sparse-реконструкцію: {sparse_dir}", file=sys.stderr)
            sys.exit(1)

    os.makedirs(dense_dir, exist_ok=True)

    print("🔥 === ЗАПУСК ЩІЛЬНОЇ 3D РЕКОНСТРУКЦІЇ (DENSE MVS) ===")
    print(f"📦 Датасет: {args.dataset}")

    total_start = time.perf_counter()

    # 1. Undistort
    if not args.skip_undistort:
        _run(
            [
                args.colmap_bin, "image_undistorter",
                "--image_path", images_dir,
                "--input_path", sparse_dir,
                "--output_path", dense_dir,
                "--output_type", "COLMAP",
                "--max_image_size", str(args.max_image_size),
            ],
            "📷 [Крок 1/3] Оптимізація та усунення дисторсії зображень (Undistorter)...",
        )
    else:
        print("\n⏭  [Крок 1/3] Пропущено (--skip-undistort)")

    # 2. Patch-Match Stereo (the ~85-90% of runtime step — flags below are the actual speed levers)
    if not args.skip_stereo:
        _run(
            [
                args.colmap_bin, "patch_match_stereo",
                "--workspace_path", dense_dir,
                "--PatchMatchStereo.gpu_index", args.gpu_index,
                "--PatchMatchStereo.max_image_size", str(args.max_image_size),
                "--PatchMatchStereo.window_step", str(args.window_step),
                "--PatchMatchStereo.num_samples", str(args.num_samples),
                "--PatchMatchStereo.num_iterations", str(args.num_iterations),
                "--PatchMatchStereo.geom_consistency", args.geom_consistency,
                "--PatchMatchStereo.cache_size", str(args.cache_size),
            ],
            "⚡ [Крок 2/3] Обчислення карт глибини на GPU (Patch-Match Stereo)...",
        )
    else:
        print("\n⏭  [Крок 2/3] Пропущено (--skip-stereo)")

    # 3. Fusion — filter here instead of randomly subsampling later in the trainer
    _run(
        [
            args.colmap_bin, "stereo_fusion",
            "--workspace_path", dense_dir,
            "--output_path", fused_ply_path,
            "--StereoFusion.min_num_pixels", str(args.min_num_pixels),
            "--StereoFusion.max_reproj_error", str(args.max_reproj_error),
            "--StereoFusion.max_image_size", str(args.max_image_size),
        ],
        "📐 [Крок 3/3] Ф'южн карт глибини в єдину щільну хмару (Stereo Fusion)...",
    )

    total_elapsed = time.perf_counter() - total_start
    print(f"\n🎉 === ЩІЛЬНУ РЕКОНСТРУКЦІЮ ЗАВЕРШЕНО за {total_elapsed/60:.1f} хв ===")
    print(f"💾 Суцільний 3D-скелет збережено в: {fused_ply_path}")


if __name__ == "__main__":
    run_colmap_dense()