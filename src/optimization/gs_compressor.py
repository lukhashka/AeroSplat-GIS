import torch
import time
import os

class GaussianSplatCompressor:
    def __init__(self, opacity_threshold=0.05):
        self.opacity_threshold = opacity_threshold
        # Перевіряємо, чи доступна твоя RTX 4060
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Compressor ініціалізовано на пристрої: {self.device}")

    def prune_transparent_gaussians(self, scene_dict):
        """
        Видаляє Гаусіани, прозорість яких нижча за поріг (Pruning).
        """
        start_time = time.time()
        opacities = scene_dict['opacity']
        
        # Створюємо маску для елементів, які залишаємо
        keep_mask = opacities >= self.opacity_threshold
        keep_mask = keep_mask.squeeze()
        
        # Фільтруємо всі тензори в сцені inline на GPU
        pruned_scene = {}
        for key, tensor in scene_dict.items():
            pruned_scene[key] = tensor[keep_mask]
            
        end_time = time.time()
        print(f"✂️  Pruning завершено за {(end_time - start_time)*1000:.2f} ms.")
        return pruned_scene

    def quantize_scene(self, scene_dict):
        """
        Стискає важкі сферичні гармоніки та геометрію з FP32 у FP16 (Quantization).
        """
        start_time = time.time()
        quantized_scene = {}
        
        for key, tensor in scene_dict.items():
            if key in ['features_dc', 'features_rest', 'rotation', 'scaling']:
                # Стискаємо колірні дані та параметри форми у напівточність (FP16)
                quantized_scene[key] = tensor.to(torch.float16)
            else:
                # Координати XYZ залишаємо в FP32, щоб не попливла геометрія карти
                quantized_scene[key] = tensor
                
        end_time = time.time()
        print(f"📦 Квантизація у FP16 завершена за {(end_time - start_time)*1000:.2f} ms.")
        return quantized_scene

    def run_compression_pipeline(self, num_gaussians=1000000):
        """
        Генерація симуляції та запуск повного циклу стиснення.
        """
        print(f"\n🔄 ГЕНЕРАЦІЯ СИМУЛЯЦІЇ: Створення {num_gaussians} важких Гаусіанів у VRAM...")
        
        # Симулюємо стандартну структуру даних навченої 3D-сцени
        torch.cuda.synchronize()
        
        # Ініціалізуємо випадкові тензори на твоїй відеокарті
        mock_scene = {
            'xyz': torch.randn(num_gaussians, 3, dtype=torch.float32, device=self.device),
            'rotation': torch.randn(num_gaussians, 4, dtype=torch.float32, device=self.device),
            'scaling': torch.randn(num_gaussians, 3, dtype=torch.float32, device=self.device),
            # Симулюємо прозорість від 0.0 до 1.0 (сигмоїда)
            'opacity': torch.rand(num_gaussians, 1, dtype=torch.float32, device=self.device),
            # Сферичні гармоніки (кольори під різними кутами) — найважча частина моделі
            'features_dc': torch.randn(num_gaussians, 3, dtype=torch.float32, device=self.device),
            'features_rest': torch.randn(num_gaussians, 45, dtype=torch.float32, device=self.device)
        }
        
        # Рахуємо початкову вагу в мегабайтах
        initial_bytes = sum(t.element_size() * t.nelement() for t in mock_scene.values())
        initial_mb = initial_bytes / (1024 * 1024)
        print(f"📊 Початковий розмір сцени у пам'яті GPU: {initial_mb:.2f} MB")
        
        print("\n🚀 Крок 1: Запуск алгоритму геометріометрії (Pruning)...")
        pruned_scene = self.prune_transparent_gaussians(mock_scene)
        
        print("\n🚀 Крок 2: Запуск бітового стиснення (Quantization)...")
        final_scene = self.quantize_scene(pruned_scene)
        
        # Рахуємо фінальну вагу
        final_bytes = sum(t.element_size() * t.nelement() for t in final_scene.values())
        final_mb = final_bytes / (1024 * 1024)
        
        print("\n🎯 === РЕЗУЛЬТАТИ ОПТИМІЗАЦІЇ СЦЕНИ ===")
        print(f"📉 Кількість точок після прунінгу: {final_scene['xyz'].shape[0]} / {num_gaussians}")
        print(f"💾 Фінальний розмір сцени: {final_mb:.2f} MB")
        print(f"⚡ Коефіцієнт стиснення: {initial_mb / final_mb:.2f}x")
        print("=======================================")

if __name__ == "__main__":
    # Створюємо компресор (все, де прозорість менша за 5% — під ніж)
    compressor = GaussianSplatCompressor(opacity_threshold=0.05)
    compressor.run_compression_pipeline(num_gaussians=1000000)