import os
import math
import piexif
from PIL import Image

def convert_to_degrees(value):
    """Допоміжна функція для конвертації координат у формат EXIF (градуси, мінути, секунди)"""
    abs_value = abs(value)
    degrees = int(abs_value)
    minutes = int((abs_value - degrees) * 60)
    seconds = round((abs_value - degrees - minutes / 60) * 3600 * 1000)
    return ((degrees, 1), (minutes, 1), (seconds, 1000))

def inject_orbital_exif():
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    
    # 📂 Папка з твоїми 150 відібраними кадрами з Google Earth Studio
    images_dir = os.path.join(project_dir, "data", "raw_images")
    
    if not os.path.exists(images_dir):
        print(f"❌ Помилка: Папки {images_dir} не існує!")
        return

    # Отримуємо чітко відсортований список усіх кадрів (від 1 до 150)
    images = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg'))])
    TOTAL_FRAMES = len(images)
    
    if TOTAL_FRAMES == 0:
        print("⚠️ У папці немає потрібних картинок JPG!")
        return
        
    print(f"🚀 Знайдено {TOTAL_FRAMES} кадрів. Починаємо запікання орбітальних GPS-координат...")

    # --- 🛠️ НАЛАШТУВАННЯ ОРБІТИ (Твої параметри з Studio) ---
    CENTER_LAT = 74.004000    # Центр Запорізької локації
    CENTER_LON = 41.055000
    FLIGHT_ALT = 312.0        # Висота 312 метрів
    RADIUS_METERS = 300.0     # Розрахунковий радіус 300 метрів

    # --- 🔄 ГОЛОВНИЙ ЦИКЛ ОРБІТАЛЬНОГО ЗМІЩЕННЯ ---
    for i, img_name in enumerate(images):
        img_path = os.path.join(images_dir, img_name)
        
        # 1. Математика кола: розраховуємо унікальний кут для кожного кадру
        angle = 2 * math.pi * (i / TOTAL_FRAMES)
        
        # 2. Перераховуємо зміщення по радіусу 300м у координати
        delta_lat = (RADIUS_METERS * math.cos(angle)) / 111111.0
        delta_lon = (RADIUS_METERS * math.sin(angle)) / (111111.0 * math.cos(math.radians(CENTER_LAT)))
        
        # 3. Фінальні координати камери для конкретного кадру
        frame_lat = CENTER_LAT + delta_lat
        frame_lon = CENTER_LON + delta_lon

        # --- 💉 КРОК ІНЖЕКЦІЇ В EXIF МЕТАДАНІ ---
        try:
            # Конвертуємо наші флоти у специфічні раціональні числа для EXIF
            exif_lat = convert_to_degrees(frame_lat)
            exif_lon = convert_to_degrees(frame_lon)
            
            lat_ref = b"N" if frame_lat >= 0 else b"S"
            lon_ref = b"E" if frame_lon >= 0 else b"W"
            
            # Формуємо GPS-пакет метаданих
            gps_ifd = {
                piexif.GPSIFD.GPSLatitudeRef: lat_ref,
                piexif.GPSIFD.GPSLatitude: exif_lat,
                piexif.GPSIFD.GPSLongitudeRef: lon_ref,
                piexif.GPSIFD.GPSLongitude: exif_lon,
                piexif.GPSIFD.GPSAltitudeRef: 0, # 0 означає вище рівня моря
                piexif.GPSIFD.GPSAltitude: (int(FLIGHT_ALT * 100), 100)
            }
            
            exif_dict = {"GPS": gps_ifd}
            exif_bytes = piexif.dump(exif_dict)
            
            # Записуємо метадані назад у файл картинки
            img = Image.open(img_path)
            img.save(img_path, exif=exif_bytes, quality=100)
            img.close()
            
            if i % 20 == 0 or i == TOTAL_FRAMES - 1:
                print(f"📸 [{i+1}/{TOTAL_FRAMES}] Запечено: {img_name} -> LAT: {frame_lat:.6f}, LON: {frame_lon:.6f}")
                
        except Exception as e:
            print(f"❌ Збій інжекції у файл {img_name}: {e}")

    print(f"\n✅ [УСПІХ] Усі {TOTAL_FRAMES} кадрів отримали унікальні кругові GPS-мітки!")
    print("🦾 Тепер зі спокійною душею запускай: python src/reconstruction/sfm_runner.py")

if __name__ == "__main__":
    inject_orbital_exif()