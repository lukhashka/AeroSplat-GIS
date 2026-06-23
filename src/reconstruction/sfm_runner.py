import os
import subprocess
import sys

def run_colmap_sfm():
    # Базові шляхи проєкту
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    images_dir = os.path.join(project_dir, "data", "georef_images")
    output_dir = os.path.join(project_dir, "data", "colmap_output")
    
    os.makedirs(output_dir, exist_ok=True)
    
    db_path = os.path.join(output_dir, "database.db")
    sparse_dir = os.path.join(output_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    print("🚀 === ЗАПУСК АВТОМАТИЗОВАНОГО ПАЙПЛАЙНУ COLMAP ===")
    print(f"📂 Вхідні геореферовані кадри: {images_dir}")
    print(f"💾 Вихідна папка реконструкції: {output_dir}\n")

    # Перевіряємо, чи є взагалі кадри в папці
    if not os.path.exists(images_dir) or len(os.listdir(images_dir)) == 0:
        print(f"❌ Помилка: Папка {images_dir} порожня або не існує. Спочатку запусти інжектор!")
        return

    # Назва команди для виклику colmap (має бути в системному PATH)
    colmap_cmd = "colmap"

    try:
        # --- 1. Екстракція фіч (SIFT) ---
        # --- 1. Екстракція фіч ---
        print("🔍 [Крок 1/3] Екстракція фіч з кадрів БПЛА (COLMAP 4.0+)...")
        extract_args = [
            colmap_cmd, "feature_extractor",
            "--database_path", db_path,
            "--image_path", images_dir,
            "--ImageReader.single_camera", "1",
            "--FeatureExtraction.use_gpu", "1"      # 🔥 ВИПРАВЛЕНО під нову версію
        ]
        subprocess.run(extract_args, check=True)
        print("✅ Екстракція фіч завершена успішно.\n")

        # --- 2. Матчинг фіч ---
        print("🔗 [Крок 2/3] Зіставлення фіч (Exhaustive Matching) на GPU...")
        match_args = [
            colmap_cmd, "exhaustive_matcher",
            "--database_path", db_path,
            "--FeatureMatching.use_gpu", "1"        # 🔥 ВИПРАВЛЕНО під нову версію
        ]
        subprocess.run(match_args, check=True)
        print("✅ Матчинг фіч завершено успішно.\n")

        # --- 3. Інкрементальна реконструкція (Mapping) ---
        print("📐 [Крок 3/3] Тріангуляція та побудова розрідженої 3D-хмари (Sparse Mapper)...")
        mapper_args = [
            colmap_cmd, "mapper",
            "--database_path", db_path,
            "--image_path", images_dir,
            "--output_path", sparse_dir
        ]
        subprocess.run(mapper_args, check=True)
        
        print("\n🎉 === РЕКОНСТРУКЦІЮ COLMAP ЗАВЕРШЕНО УСПІШНО! ===")
        print(f"📂 Результати збережено в бінарники: {os.path.join(sparse_dir, '0')}")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Помилка під час виконання команди COLMAP: {e}", file=sys.stderr)
    except FileNotFoundError:
        print("\n❌ Помилка: Команда 'colmap' не знайдена в системному PATH.", file=sys.stderr)
        print("Перевірте, чи додано шлях до папки з colmap.exe в змінні оточення Windows.", file=sys.stderr)

if __name__ == "__main__":
    run_colmap_sfm()