import os
import struct
import torch

def parse_colmap_points3d():
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    bin_path = os.path.join(project_dir, "data", "colmap_output", "sparse", "0", "points3D.bin")
    output_pt_path = os.path.join(project_dir, "data", "zaporizhzhia_cloud.pt")

    if not os.path.exists(bin_path):
        print(f"❌ Помилка: Файл {bin_path} не знайдено!")
        return

    print(f"📖 Читання бінарного файлу COLMAP: {bin_path}")
    
    xyz_list = []
    rgb_list = []

    with open(bin_path, "rb") as fid:
        # Зчитуємо загальну кількість точок у файлі (uint64)
        num_points = struct.unpack("<Q", fid.read(8))[0]
        print(f"📊 Знайдено реальних 3D-точок у хмарі: {num_points:,}")

        for _ in range(num_points):
            # Кожна точка починається з: point3D_id(uint64), x(double), y(double), z(double)
            binary_point_data = fid.read(32)
            point_data = struct.unpack("<Q3d", binary_point_data)
            
            x, y, z = point_data[1], point_data[2], point_data[3]
            
            # Наступні 3 байти — це кольори R, G, B (uint8)
            r, g, b = struct.unpack("<3B", fid.read(3))
            
            # Наступні 8 байт — помилка репроекції (double)
            fid.seek(8, 1)
            
            # Наступні 8 байт — довжина треку видимості точки (uint64)
            track_len = struct.unpack("<Q", fid.read(8))[0]
            
            # Пропускаємо елементи треку (кожен елемент по 8 байт: image_id(uint32) + point2D_idx(uint32))
            fid.seek(track_len * 8, 1)

            # Зберігаємо дані (кольори нормалізуємо від 0 до 1 для Three.js)
            xyz_list.append([x, y, z])
            rgb_list.append([r / 255.0, g / 255.0, b / 255.0])

    # Конвертуємо у тензори PyTorch
    xyz_tensor = torch.tensor(xyz_list, dtype=torch.float32)
    rgb_tensor = torch.tensor(rgb_list, dtype=torch.float32)

    # Пакуємо в єдину структуру
    cloud_data = {
        'xyz': xyz_tensor,
        'rgb': rgb_tensor
    }

    # Зберігаємо на диск для сервера
    torch.save(cloud_data, output_pt_path)
    print(f"💾 Успішно експортовано {xyz_tensor.shape[0]} точок у файл {output_pt_path}")

if __name__ == "__main__":
    parse_colmap_points3d()