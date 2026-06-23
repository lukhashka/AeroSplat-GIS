import os
import struct  # 🔥 ВИПРАВЛЕНО: додали стандартний бінарний розпакувальник
import torch

def parse_dense_ply():
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    ply_path = os.path.join(project_dir, "data", "colmap_output", "dense", "fused.ply")
    output_pt_path = os.path.join(project_dir, "data", "zaporizhzhia_cloud.pt")

    if not os.path.exists(ply_path):
        print(f"❌ Помилка: Файл {ply_path} не знайдено! Спочатку запусти dense_runner.py")
        return

    print(f"📖 Читання щільної PLY-моделі: {ply_path}")
    
    xyz_list = []
    rgb_list = []
    
    with open(ply_path, "rb") as f:
        # Читаємо заголовок (header) файлу PLY
        header = ""
        while b"end_header" not in header.encode('utf-8', errors='ignore'):
            line = f.readline().decode('utf-8', errors='ignore')
            header += line
            if "element vertex" in line:
                num_vertices = int(line.split()[-1])
        
        print(f"📊 Знайдено реальних РЕКОНСТРУЙОВАНИХ точок: {num_vertices:,}")
        print("⏳ Парсинг масиву даних у тензори PyTorch...")

        has_normals = "nx" in header
        bytes_per_vertex = 27 if has_normals else 15

        for _ in range(num_vertices):
            data = f.read(bytes_per_vertex)
            if len(data) < bytes_per_vertex:
                break
            
            # 🔥 ВИПРАВЛЕНО: Читаємо 3 значення float32 (12 байт) безпечно через struct
            x, y, z = struct.unpack("<3f", data[:12])
            
            # Витягуємо кольори (останні 3 байти: uint8)
            r = data[-3] / 255.0
            g = data[-2] / 255.0
            b = data[-1] / 255.0
            
            xyz_list.append([x, y, z])
            rgb_list.append([r, g, b])

    # Зберігаємо у PyTorch тензори
    cloud_data = {
        'xyz': torch.tensor(xyz_list, dtype=torch.float32),
        'rgb': torch.tensor(rgb_list, dtype=torch.float32)
    }
    
    torch.save(cloud_data, output_pt_path)
    print(f"💾 Успішно експортовано {len(xyz_list):,} точок у {output_pt_path}!")

if __name__ == "__main__":
    parse_dense_ply()