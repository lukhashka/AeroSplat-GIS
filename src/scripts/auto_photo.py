import os
import time
import math
import csv
from PIL import ImageGrab
import pyautogui

def start_orbital_flight(duration_seconds=25, interval_seconds=0.5):
    # --- НАЛАШТУВАННЯ ЦЕНТРУ ЛОКАЦІЇ (Запорізька обл. для синхронізації) ---
    CENTER_LAT = 47.005200
    CENTER_LON = 34.916400
    FLIGHT_ALT = 150.0  # Висота польоту в метрах
    RADIUS_METERS = 350.0  # Радіус кола обльоту в метрах
    
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    output_dir = os.path.join(project_dir, "data", "raw_images_3d")
    log_filepath = os.path.join(project_dir, "data", "flight_telemetry.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("⏳ УВАГА! У тебе є 5 секунд, щоб відкрити вікно Google Earth і навести камеру на ціль...")
    for i in range(5, 0, -1):
        print(f"{i}...")
        time.sleep(1)
        
    print("\n🚀 АВТОПІЛОТ ТА ОРБІТАЛЬНИЙ ЗАПИС ЗАПУЩЕНО!")
    
    # Створюємо CSV-файл для запису унікальних GPS координат кожного кадру
    telemetry_file = open(log_filepath, mode='w', newline='', encoding='utf-8')
    telemetry_writer = csv.writer(telemetry_file)
    telemetry_writer.writerow(["filename", "latitude", "longitude", "altitude"])
    
    start_time = time.time()
    img_counter = 0
    
    # Затискаємо комбінацію клавіш для плавного обертання по колу в Google Earth Pro
    pyautogui.keyDown('alt')
    pyautogui.keyDown('shift')
    pyautogui.keyDown('right')
    
# === ЗНАЙДИ І ЗАМІНИ ЦЕЙ БЛОК У СВОЄМУ СКРИПТІ ===
    
    # Тепер затискаємо ТІЛЬКИ Shift + стрілку, ніякого Alt!
    pyautogui.keyDown('shift')
    pyautogui.keyDown('right') # або 'left' для обертання в інший бік
    
    try:
        while time.time() - start_time < duration_seconds:
            loop_start = time.time()
            current_elapsed = time.time() - start_time
            
            # ... (вся математика кола залишається без змін) ...
            
            # Робимо скріншот
            screenshot = ImageGrab.grab()
            filename = f"orbit_frame_{img_counter:04d}.jpg"
            filepath = os.path.join(output_dir, filename)
            screenshot.save(filepath, "JPEG", quality=95)
            
            # Запис у CSV
            telemetry_writer.writerow([filename, f"{frame_lat:.6f}", f"{frame_lon:.6f}", f"{FLIGHT_ALT:.1f}"])
            
            print(f"📸 Кадр {img_counter:04d} | LAT: {frame_lat:.6f} | LON: {frame_lon:.6f} | OK")
            img_counter += 1
            
            elapsed_loop = time.time() - loop_start
            sleep_time = max(0.05, interval_seconds - elapsed_loop)
            time.sleep(sleep_time)
            
    finally:
        # ВІДПУСКАЄМО клавіші (теж прибираємо Alt)
        pyautogui.keyUp('right')
        pyautogui.keyUp('shift')
        telemetry_file.close()
        
    print(f"\n✅ [УСПІХ] Орбітальний обліт завершено!")
    print(f"📂 {img_counter} кадрів збережено в: {output_dir}")
    print(f"📝 Лог динамічного автозміщення записано в: {log_filepath}")
    print("\n🦾 Далі запускай свій інжектор, щоб він прописав ці дані всередину EXIF!")

if __name__ == "__main__":
    # Запуск: крутимося 30 секунд, робимо знімок кожні 0.4 секунди (отримаємо ~75 кадрів з перекриттям 85%)
    start_orbital_flight(duration_seconds=30, interval_seconds=0.4)