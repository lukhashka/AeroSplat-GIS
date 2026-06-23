import os
import subprocess
import sys

def run_colmap_dense():
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    images_dir = os.path.join(project_dir, "data", "georef_images")
    sparse_dir = os.path.join(project_dir, "data", "colmap_output", "sparse", "0")
    dense_dir = os.path.join(project_dir, "data", "colmap_output", "dense")
    
    os.makedirs(dense_dir, exist_ok=True)
    fused_ply_path = os.path.join(dense_dir, "fused.ply")
    colmap_cmd = "colmap"

    print("🔥 === ЗАПУСК ЩІЛЬНОЇ 3D РЕКОНСТРУКЦІЇ (DENSE MVS) ===")
    
    try:
        # 1. Вирівнювання та усунення дисторсії кадрів
        print("\n📷 [Крок 1/3] Оптимізація та усунення дисторсії зображень (Undistorter)...")
        undistort_args = [
            colmap_cmd, "image_undistorter",
            "--image_path", images_dir,
            "--input_path", sparse_dir,
            "--output_path", dense_dir,
            "--output_type", "COLMAP"
        ]
        subprocess.run(undistort_args, check=True)

        # 2. Обчислення карт глибини на GPU (Patch Match Stereo)
        print("\n⚡ [Крок 2/3] Обчислення карт глибини на GPU (Patch-Match Stereo)...")
        print("💡 Примітка: Твоя RTX 4060 зараз увімкне вентилятори на повну. Це займе трохи часу.")
        stereo_args = [
            colmap_cmd, "patch_match_stereo",
            "--workspace_path", dense_dir,
            "--PatchMatchStereo.gpu_index", "0"  # Жорстко саджаємо на першу відеокарту
        ]
        subprocess.run(stereo_args, check=True)

        # 3. Ф'южн (об'єднання) карт глибини в суцільну хмару точок
        print("\n📐 [Крок 3/3] Ф'южн карт глибини в єдину щільну хмару (Stereo Fusion)...")
        fusion_args = [
            colmap_cmd, "stereo_fusion",
            "--workspace_path", dense_dir,
            "--output_path", fused_ply_path
        ]
        subprocess.run(fusion_args, check=True)

        print(f"\n🎉 === ЩІЛЬНУ РЕКОНСТРУКЦІЮ ЗАВЕРШЕНО! ===")
        print(f"💾 Суцільний 3D-скелет збережено в: {fused_ply_path}")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Помилка під час виконання щільної реконструкції: {e}", file=sys.stderr)

if __name__ == "__main__":
    run_colmap_dense()