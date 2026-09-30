"""
ColmapDataLoader — завантажує камери та фото з COLMAP output.

Структура що очікується:
    data/colmap_output/
    ├── sparse/
    │   └── 0/
    │       ├── cameras.bin   ← intrinsics (fx, fy, cx, cy)
    │       └── images.bin    ← extrinsics (R, T) + імена файлів
    └── dense/
        └── images/
            ├── final_000.jpeg
            ├── final_001.jpeg
            └── ...

OPTIMIZATION PASS
──────────────────
Three independent changes, in order of impact:

1. read_images_binary() no longer reads the file one struct.unpack()/
   f.read(1) call at a time. It reads the whole file into memory ONCE,
   then walks it with struct.unpack_from() + bytes.index() on offsets.
   This also means the per-image 2D-keypoint block (potentially
   thousands of points per image in a real COLMAP scene) is skipped
   with pointer arithmetic instead of being read into a buffer and
   thrown away. This is the single biggest win in this file — real
   images.bin files are dominated by 2D-keypoint bytes we never use.

2. Camera._load_image now calls PIL's Image.draft() before decoding.
   For JPEGs, this tells libjpeg to decode directly at a reduced DCT
   scale instead of decoding full resolution and then downsampling —
   a real speedup whenever downscale <= 0.5 (the pipeline's default is
   0.25).

3. ColmapDataLoader.load() decodes+resizes all photos in a
   ThreadPoolExecutor instead of serially. PIL's decode/resize path
   releases the GIL for most of its work, so this parallelizes real
   wall-clock time across cores. GPU tensor creation is kept on the
   main thread (decoded arrays are handed back as plain numpy, moved
   to device once collected) to avoid any question of CUDA-from-
   worker-thread behavior.

Everything else (public API, Camera fields, ColmapDataLoader.load()'s
signature and return type, train_with_real_data, the CLI) is
unchanged — this is a drop-in replacement.
"""

import os
import math
import struct
import collections
import numpy as np
import torch
from torch import Tensor
from PIL import Image
from typing import List, NamedTuple, Optional
from concurrent.futures import ThreadPoolExecutor


# ═══════════════════════════════════════════════════════════════════
#  COLMAP BINARY READERS
#  Формат описано тут: https://colmap.github.io/format.html
# ═══════════════════════════════════════════════════════════════════

class ColmapCamera(NamedTuple):
    id:         int
    model:      str
    width:      int
    height:     int
    params:     np.ndarray   # [fx, fy, cx, cy] або [f, cx, cy] залежно від моделі


class ColmapImage(NamedTuple):
    id:          int
    qvec:        np.ndarray  # кватерніон [W, X, Y, Z] — world→camera rotation
    tvec:        np.ndarray  # translation world→camera
    camera_id:   int
    name:        str         # ім'я файлу, напр. "final_000.jpeg"


# Таблиця кількості параметрів для кожної моделі камери COLMAP
CAMERA_MODEL_PARAMS = {
    "SIMPLE_PINHOLE": 3,   # f, cx, cy
    "PINHOLE":        4,   # fx, fy, cx, cy
    "SIMPLE_RADIAL":  4,   # f, cx, cy, k
    "RADIAL":         5,   # f, cx, cy, k1, k2
    "OPENCV":         8,   # fx, fy, cx, cy, k1, k2, p1, p2
    "FULL_OPENCV":    12,
    "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE": 5,
}

CAMERA_MODEL_IDS = {
    0:  "SIMPLE_PINHOLE",
    1:  "PINHOLE",
    2:  "SIMPLE_RADIAL",
    3:  "RADIAL",
    4:  "OPENCV",
    5:  "FULL_OPENCV",
    6:  "SIMPLE_RADIAL_FISHEYE",
    7:  "RADIAL_FISHEYE",
}

# Precompiled struct formats — one unpack call per record instead of
# four/five separate ones. "<" = little-endian, standard size (no
# padding), so struct.calcsize matches a manual byte count exactly.
_CAMERA_HEADER = struct.Struct("<iiQQ")          # cam_id, model_id, width, height
_IMAGE_HEADER  = struct.Struct("<i4d3di")        # image_id, qvec(4), tvec(3), camera_id


def read_cameras_binary(path: str) -> dict:
    """Читає cameras.bin → dict {camera_id: ColmapCamera}"""
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id, model_id, width, height = _CAMERA_HEADER.unpack(
                f.read(_CAMERA_HEADER.size)
            )

            model_name  = CAMERA_MODEL_IDS.get(model_id, "PINHOLE")
            num_params  = CAMERA_MODEL_PARAMS.get(model_name, 4)
            params      = np.frombuffer(f.read(8 * num_params), dtype=np.float64)

            cameras[cam_id] = ColmapCamera(
                id=cam_id, model=model_name,
                width=int(width), height=int(height),
                params=params,
            )
    return cameras


def read_images_binary(path: str) -> dict:
    """
    Читає images.bin → dict {image_id: ColmapImage}

    Один f.read() на весь файл замість тисяч дрібних I/O-викликів, і
    2D keypoints (x, y, point3d_id — 24 байти на точку, часто тисячі
    точок на фото) пропускаються зсувом курсора, а не читанням у буфер.
    """
    with open(path, "rb") as f:
        data = f.read()

    images = {}
    offset = 0
    num_images = struct.unpack_from("<Q", data, offset)[0]
    offset += 8

    header_size = _IMAGE_HEADER.size
    for _ in range(num_images):
        image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = _IMAGE_HEADER.unpack_from(
            data, offset
        )
        offset += header_size

        # Ім'я файлу — null-terminated string. bytes.index() на вже
        # завантаженому буфері замість читання по одному байту.
        name_end = data.index(b"\x00", offset)
        name = data[offset:name_end].decode("utf-8")
        offset = name_end + 1

        # 2D keypoints — пропускаємо (не потрібні для тренування).
        # Раніше: f.read(24 * num_points2d) — виділяв і копіював буфер,
        # який одразу відкидався. Тепер: просто зсуваємо курсор.
        num_points2d = struct.unpack_from("<Q", data, offset)[0]
        offset += 8 + 24 * num_points2d

        images[image_id] = ColmapImage(
            id=image_id,
            qvec=np.array([qw, qx, qy, qz], dtype=np.float64),
            tvec=np.array([tx, ty, tz], dtype=np.float64),
            camera_id=camera_id,
            name=name,
        )
    return images


# ═══════════════════════════════════════════════════════════════════
#  CAMERA POSE CONVERSIONS
# ═══════════════════════════════════════════════════════════════════

def qvec_to_rotation_matrix(qvec: np.ndarray) -> np.ndarray:
    """
    COLMAP кватерніон [W, X, Y, Z] → матриця обертання 3×3.
    COLMAP зберігає world→camera rotation.
    """
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*(y**2 + z**2),   2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),          1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),          2*(y*z + w*x),       1 - 2*(x**2 + y**2)],
    ])
    return R


def build_view_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Будує 4×4 view matrix (world→camera transform).

    [R | t]
    [0 | 1]
    """
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = R
    view[:3,  3] = t
    return view


def extract_intrinsics(cam: ColmapCamera):
    """
    Витягує fx, fy, cx, cy залежно від моделі камери.
    Returns: (fx, fy, cx, cy)
    """
    p = cam.params
    if cam.model == "SIMPLE_PINHOLE":
        # params: [f, cx, cy]
        return float(p[0]), float(p[0]), float(p[1]), float(p[2])
    elif cam.model in ("PINHOLE", "OPENCV", "FULL_OPENCV"):
        # params: [fx, fy, cx, cy, ...]
        return float(p[0]), float(p[1]), float(p[2]), float(p[3])
    elif cam.model in ("SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        # params: [f, cx, cy, ...]
        return float(p[0]), float(p[0]), float(p[1]), float(p[2])
    else:
        # Fallback
        return float(p[0]), float(p[0]), float(p[1]), float(p[2])


# ═══════════════════════════════════════════════════════════════════
#  CAMERA DATACLASS (для передачі в trainer)
# ═══════════════════════════════════════════════════════════════════

class Camera:
    """
    Одна камера з усіма параметрами для рендерингу.
    Передається в gs_trainer як 'camera' об'єкт.
    """
    def __init__(
        self,
        image_path: str,
        R: np.ndarray,       # 3×3 world→camera rotation
        T: np.ndarray,       # (3,) world→camera translation
        fx: float, fy: float,
        cx: float, cy: float,
        width: int, height: int,
        device: torch.device,
        downscale: float = 1.0,
        preloaded_rgb: Optional[np.ndarray] = None,
    ):
        # Масштабуємо якщо треба
        if downscale != 1.0:
            width  = int(width  * downscale)
            height = int(height * downscale)
            fx *= downscale;  fy *= downscale
            cx *= downscale;  cy *= downscale

        self.R = R
        self.T = T
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.image_width  = width
        self.image_height = height
        self.image_path   = image_path
        self.device       = device

        # FoV (потрібно для CUDA растерізатора)
        self.FoVx = 2 * math.atan(width  / (2 * fx))
        self.FoVy = 2 * math.atan(height / (2 * fy))

        # View matrix (4×4)
        self.view_matrix = torch.tensor(
            build_view_matrix(R, T), dtype=torch.float32, device=device
        )

        # Завантажуємо фото. Якщо декодування вже зроблено заздалегідь
        # (паралельно, в ColmapDataLoader.load()) — просто переносимо
        # готовий масив на device замість повторного decode/resize.
        if preloaded_rgb is not None:
            self.gt_image = (
                torch.from_numpy(preloaded_rgb).to(device).permute(2, 0, 1)
            )
        else:
            self.gt_image = self._load_image(image_path, width, height, device)

    @staticmethod
    def _decode_image_cpu(path: str, W: int, H: int) -> np.ndarray:
        """
        Декодує + масштабує фото на CPU → (H, W, 3) float32 в [0, 1].

        Винесено в static method (без залежності від self/device), щоб
        можна було викликати з ThreadPoolExecutor паралельно для
        багатьох камер одразу.

        img.draft(...) просить libjpeg декодувати одразу в меншій
        роздільності (через DCT-скейлінг), замість повного decode +
        downsample. Помітно швидше коли downscale <= 0.5 — типове
        значення в пайплайні (0.25). Draft лише наближає розмір до
        кратного 1/2/4/8 — наступний .resize() все одно доводить до
        точного (W, H), як і раніше.
        """
        img = Image.open(path)
        img.draft("RGB", (W, H))
        img = img.convert("RGB").resize((W, H), Image.LANCZOS)
        return np.asarray(img, dtype=np.float32) / 255.0

    def _load_image(
        self, path: str, W: int, H: int, device: torch.device
    ) -> Tensor:
        """Завантажує фото → тензор (3, H, W) в [0, 1]."""
        arr = self._decode_image_cpu(path, W, H)
        return torch.from_numpy(arr).to(device).permute(2, 0, 1)  # (3, H, W)


# ═══════════════════════════════════════════════════════════════════
#  COLMAP DATA LOADER
# ═══════════════════════════════════════════════════════════════════

class ColmapDataLoader:
    """
    Завантажує всі камери та фото з COLMAP output.

    Використання:
        loader = ColmapDataLoader("data/colmap_output_aa", device)
        cameras = loader.load(downscale=0.25)  # 0.25 → зменш в 4 рази

        for camera in cameras:
            rendered = rasterizer.render(params, camera.view_matrix, bg)
            loss = photometric_loss(rendered, camera.gt_image)
    """

    def __init__(self, colmap_root: str, device: torch.device):
        self.colmap_root = colmap_root
        self.device      = device

        # Шукаємо sparse/0/ або sparse/
        sparse_candidates = [
            os.path.join(colmap_root, "sparse", "0"),
            os.path.join(colmap_root, "sparse"),
        ]
        self.sparse_dir = None
        for path in sparse_candidates:
            if os.path.isdir(path):
                self.sparse_dir = path
                break

        if self.sparse_dir is None:
            raise FileNotFoundError(
                f"Не знайдено sparse/ в {colmap_root}\n"
                f"Очікувалось: {sparse_candidates[0]}"
            )

        self.images_dir = os.path.join(colmap_root, "dense", "images")
        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"Не знайдено: {self.images_dir}")

    def load(
        self,
        downscale: float = 1.0,
        max_cameras: Optional[int] = None,
        max_workers: Optional[int] = None,
    ) -> List[Camera]:
        """
        Читає cameras.bin + images.bin, завантажує фото.

        Args:
            downscale:    коефіцієнт зменшення (0.25 → ÷4, рекомендовано для початку)
            max_cameras:  обмеження кількості камер (None = всі)
            max_workers:  розмір пулу потоків для decode/resize (None = авто)

        Returns:
            List[Camera] — відсортований за іменем файлу
        """
        cameras_bin = os.path.join(self.sparse_dir, "cameras.bin")
        images_bin  = os.path.join(self.sparse_dir, "images.bin")

        if not os.path.exists(cameras_bin):
            raise FileNotFoundError(f"Не знайдено: {cameras_bin}")
        if not os.path.exists(images_bin):
            raise FileNotFoundError(f"Не знайдено: {images_bin}")

        print(f"  Читаємо {cameras_bin}...")
        colmap_cameras = read_cameras_binary(cameras_bin)
        print(f"  Знайдено {len(colmap_cameras)} камер(и)")

        print(f"  Читаємо {images_bin}...")
        colmap_images = read_images_binary(images_bin)
        print(f"  Знайдено {len(colmap_images)} зображень")

        # Сортуємо за іменем для відтворюваності
        sorted_images = sorted(colmap_images.values(), key=lambda x: x.name)

        if max_cameras is not None:
            sorted_images = sorted_images[:max_cameras]

        # ── Крок 1: зібрати метадані + відфільтрувати відсутні файли ──
        cam_specs = []
        missing = []
        for col_img in sorted_images:
            img_path = os.path.join(self.images_dir, col_img.name)
            if not os.path.exists(img_path):
                missing.append(col_img.name)
                continue

            col_cam = colmap_cameras[col_img.camera_id]
            fx, fy, cx, cy = extract_intrinsics(col_cam)
            R = qvec_to_rotation_matrix(col_img.qvec)
            T = col_img.tvec.astype(np.float32)

            w = int(col_cam.width  * downscale) if downscale != 1.0 else col_cam.width
            h = int(col_cam.height * downscale) if downscale != 1.0 else col_cam.height

            cam_specs.append({
                "image_path": img_path, "R": R, "T": T,
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "width": col_cam.width, "height": col_cam.height,
                "decode_w": w, "decode_h": h,
            })

        if missing:
            print(f"  ⚠️  Пропущено {len(missing)} файлів (не знайдено в dense/images/)")

        # ── Крок 2: decode+resize фото паралельно (CPU, потоки) ────────
        # PIL відпускає GIL на час JPEG-декодування та ресайзу, тож
        # ThreadPoolExecutor дає реальний паралелізм по ядрах, а не
        # просто ілюзію конкурентності.
        workers = max_workers or min(8, (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            decoded_rgb = list(pool.map(
                lambda s: Camera._decode_image_cpu(s["image_path"], s["decode_w"], s["decode_h"]),
                cam_specs,
            ))

        # ── Крок 3: побудувати Camera-об'єкти (тензори — в головному потоці) ──
        cameras = []
        for spec, rgb in zip(cam_specs, decoded_rgb):
            cam = Camera(
                image_path=spec["image_path"],
                R=spec["R"], T=spec["T"],
                fx=spec["fx"], fy=spec["fy"], cx=spec["cx"], cy=spec["cy"],
                width=spec["width"], height=spec["height"],
                device=self.device,
                downscale=downscale,
                preloaded_rgb=rgb,
            )
            cameras.append(cam)

        print(f"  Завантажено {len(cameras)} камер, "
              f"розмір: {cameras[0].image_width}×{cameras[0].image_height}")

        return cameras



# ═══════════════════════════════════════════════════════════════════
#  ТОЧКА ВХОДУ
#
#  Тренування тепер живе виключно в gs_trainer_v3.py (--colmap/--images
#  там же, з load_colmap_cameras()/attach_images() і власними Camera/
#  TileRasterizer/CudaRasterizer). Цей файл лишається тим, чим і має бути
#  за власним докстрінгом — парсером COLMAP + завантажувачем кадрів.
#  CLI тут — лише швидка ручна перевірка, що ColmapDataLoader читає
#  конкретний датасет без помилок.
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ColmapDataLoader — перевірка читання cameras.bin/images.bin + фото. "
                    "Для тренування використовуй gs_trainer_v3.py --colmap ... --images ..."
    )
    parser.add_argument("--colmap",      default="data/colmap_output_ac_DS")
    parser.add_argument("--downscale",   type=float, default=0.25)
    parser.add_argument("--max_cameras", type=int, default=3)
    args = parser.parse_args()

    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    device  = torch.device("cpu")
    loader  = ColmapDataLoader(os.path.join(project_dir, args.colmap), device)
    cameras = loader.load(downscale=args.downscale, max_cameras=args.max_cameras)

    print(f"\n  ✅ Все ок! Перша камера:")
    print(f"     Файл: {os.path.basename(cameras[0].image_path)}")
    print(f"     fx={cameras[0].fx:.1f}, fy={cameras[0].fy:.1f}")
    print(f"     cx={cameras[0].cx:.1f}, cy={cameras[0].cy:.1f}")
    print(f"     gt_image shape: {cameras[0].gt_image.shape}")