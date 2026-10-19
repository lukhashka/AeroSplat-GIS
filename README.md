#  AeroSplat-GIS

## UAV-Based 3D Terrain Reconstruction and GIS Visualization Engine

AeroSplat-GIS is a geospatial reconstruction and visualization framework
designed for generating metrically accurate 3D terrain models from UAV
imagery and simulation environments.

The project combines traditional photogrammetry techniques (**Structure
from Motion** and **Multi-View Stereo**) with modern neural scene
representations based on **3D Gaussian Splatting (3DGS)**, enabling
interactive visualization of reconstructed environments.

The system is optimized for **interactive real-time visualization on
modern consumer GPUs**.

------------------------------------------------------------------------

## Reconstruction Examples

| Dense Point Cloud (MVS) | 3D Gaussian Splatting |
|---|---|
| <img src="assets/examples/dense_point_cloud.png" width="400"> | <img src="assets/examples/gaussian_splatting.png" width="400"> |
| Classical point-based reconstruction | Neural Gaussian representation |

------------------------------------------------------------------------

#  Features

## Dual Reconstruction Pipeline

AeroSplat-GIS supports two independent reconstruction approaches:

### Dense Point Cloud Reconstruction

Based on classical photogrammetry:

-   Structure-from-Motion (SfM)
-   Multi-View Stereo (MVS)
-   Dense point cloud generation

### Neural Scene Reconstruction

Based on:

-   3D Gaussian Splatting (3DGS)
-   Differentiable rendering
-   PyTorch optimization pipeline

Both approaches can be compared and visualized depending on required
accuracy and performance characteristics.

------------------------------------------------------------------------

##  Georeferenced Reconstruction

The pipeline supports geographic alignment through metadata processing
and coordinate injection.

Supported parameters:

-   Latitude
-   Longitude
-   Altitude

The reconstructed scene is scaled into real-world metric coordinates.

    1 unit in 3D space ≈ 1 meter in the real environment

------------------------------------------------------------------------

##  GPU Rendering Optimization

The visualization layer uses:

-   WebGL
-   Three.js
-   Custom GLSL shaders

Optimization techniques include:

-   Depth-based visibility filtering
-   Frustum culling
-   Efficient GPU-side rendering
-   Binary tensor streaming

------------------------------------------------------------------------

##  Binary Tensor Streaming

The backend supports direct streaming of serialized PyTorch tensors to
the WebGL frontend.

This reduces loading overhead compared to traditional text-based formats
and allows efficient transfer of large reconstructed scenes.

------------------------------------------------------------------------

# Architecture Pipeline

```text
Raw UAV Images
        │
        ▼
exif_injector.py
GPS metadata injection
        │
        ▼
sfm_runner.py
Sparse reconstruction (COLMAP SfM)
        │
        ▼
dense_runner.py
Dense reconstruction (MVS)
        │
        ▼
ply_parser.py
Point cloud processing
        │
        ├──────────────────────┐
        │                      │
        ▼                      ▼
Dense Point Cloud       3D Gaussian Splatting
                               │
                               ▼
                        gs_trainer.py
                        PyTorch optimization
                               │
                               ▼
                         FastAPI Backend
                               │
                               ▼
                      WebGL Viewer (Three.js)
```

------------------------------------------------------------------------

#  Repository Structure

    AeroSplat-GIS/
    ├── src/
    │   └── aerosplat/            # пакет
    │       ├── starter.py        # оркестратор (`aerosplat run`)
    │       ├── doctor.py         # діагностика (`aerosplat doctor`)
    │       ├── server/main.py    # FastAPI (`aerosplat serve`)
    │       ├── config.py, paths.py, logging_utils.py
    │       ├── pipeline/         # стадії, RunLayout, manifest.json
    │       ├── geo/              # earthstudio_to_colmap, exif_injector
    │       ├── sfm/              # sfm_runner, dense_runner
    │       ├── io/               # cloud_io, ply_parser, colmap_parser
    │       └── gs/               # trainer (3DGS)
    ├── web/                      # власний WebGL-в'ювер
    ├── tools/                    # дослідницькі/діагностичні скрипти
    ├── legacy/                   # застарілий код (лише для історії)
    ├── docs/                     # ARCHITECTURE, ROADMAP, DECISIONS, REAL_DRONES
    ├── data/                     # НЕ в git, див. data/README.md
    ├── pyproject.toml
    └── requirements.txt

------------------------------------------------------------------------

#  System Requirements

## Hardware

Recommended:

-   NVIDIA GPU with CUDA support
-   Minimum 6 GB VRAM for larger scenes

## Software

Required:

-   Python 3.x
-   CUDA-enabled PyTorch
-   COLMAP installed and available in system PATH

Supported:

-   Windows 10/11
-   Linux distributions

------------------------------------------------------------------------

#  Installation

``` bash
# PyTorch під вашу CUDA (приклад для CUDA 12.6)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e .
aerosplat doctor
```

COLMAP має бути в `PATH` (або `--colmap-bin`). Опційно: CUDA-растерізатор Inria
(`diff-gaussian-rasterization/`, **non-commercial**, збирається локально) —
без нього використовується повільніший PyTorch-растерізатор.

------------------------------------------------------------------------

#  Reconstruction Pipeline Usage

Датасет кладеться в `data/<ds>/` (див. [data/README.md](data/README.md)).

``` bash
aerosplat run --dataset dataset_v6            # весь пайплайн
aerosplat run --dataset dataset_v6 --dry-run  # показати команди
aerosplat run --dataset dataset_v6 --from sfm
```

Основний шлях — ділянки каталогу (кожен прогін — окремий незмінний run):

``` bash
aerosplat catalog --help                       # проєкти, ділянки, імпорт зйомок
aerosplat run --site default/dataset_v6        # новий run ділянки
aerosplat run --site default/dataset_v6 --resume --from semantics
aerosplat doctor --site default/dataset_v6     # діагностика останнього run-а (лише читає)
```

**Перевірка встановлення на малому семплі** (16 кадрів із `dataset_v6`, ≈1 хв на RTX 4060):

``` bash
python tools/make_sample_dataset.py            # data/dataset_v6 → data/sample (оригінал не змінюється)
aerosplat run --dataset sample --iterations 500
aerosplat doctor --dataset sample              # без ❌
```

`camera_source` за замовчуванням `auto`: Earth Studio JSON, якщо він є, інакше
synthetic-режим. Тренер без `--self-test` вимагає явні `--input/--colmap/--images`.

------------------------------------------------------------------------

#  Web UI and job queue

Веб-UI керує ділянками, завантаженням зйомок, запуском реконструкцій і показує
прогрес наживо. Одноразово зібрати UI (потрібен Node.js 20+):

``` bash
cd web/app && npm ci && npm run build
```

Сервер разом із GPU- і CPU-воркером та сервісом рендерингу:

``` bash
aerosplat up            # http://127.0.0.1:8000/  →  веб-UI на /app/; рендер на :8001
```

Задачі виконують воркери з черги в каталозі (`projects/catalog.db`), по одній
GPU-задачі за раз. Окремо: `aerosplat worker --resource gpu`, `aerosplat serve`.
Сервіс рендерингу (`aerosplat render-service`) малює навчену сцену з довільної пози,
перспективою чи ортографічно — сторінка «Вид із камери» в UI або
`POST /api/v2/sites/<project>/<site>/render`.
Для розробки UI з hot reload: `cd web/app && npm run dev` (проксіює API на :8000).
Завантажені кадри обробляє CPU-воркер (задача `ingest`): захоплення з'являється на ділянці,
коли задача завершилась.
Архітектура — [docs/ARCHITECTURE.md §3.10–3.12](docs/ARCHITECTURE.md).

### Доступ з інших машин (токени)

З цієї машини (`127.0.0.1`) все працює без налаштувань, як і раніше. Щоб відкрити сервер
у мережу, спершу створіть токен — без нього `aerosplat up` з не-loopback адресою не стартує:

``` bash
aerosplat token create alice      # токен показується один раз; зберігається лише його хеш
aerosplat up --host 0.0.0.0       # запити не з цієї машини вимагають токен
aerosplat token list              # хто має доступ
aerosplat token revoke alice      # діє з наступного запиту
```

У браузері UI сам попросить токен (далі — HttpOnly-cookie сесії); скрипти шлють заголовок
`Authorization: Bearer <токен>`. Зміни, зроблені з токеном, журнал аудиту підписує
`token:<ім'я>`. Це HTTP без TLS: лише довірена мережа або reverse proxy з TLS. Запити через
проксі чи тунель на цій машині (`X-Forwarded-For` тощо) теж вимагають токен;
`aerosplat up --require-auth` вимагає його навіть з localhost (тунелі без таких заголовків).

### E2E-тест UI (Playwright)

``` bash
cd web/app && npx playwright install chromium   # одноразово
npm run e2e        # збирає UI і проганяє smoke на тимчасових серверах :18765/:18766
```

Python береться з `AEROSPLAT_PYTHON` або `venv/` репозиторію; дані — тимчасовий каталог у
`%TEMP%/aerosplat-e2e`, не `projects/`. Без Chromium тест пропускається. У `npm test` не входить.

#  Launch Visualization Server

``` bash
aerosplat serve   # або: python -m aerosplat.server.main
```

Open:

    http://127.0.0.1:8000/web/

Recommended browsers:

-   Google Chrome
-   Microsoft Edge

------------------------------------------------------------------------

#  Simulation: ArduCopter SITL

SITL збирається й працює у WSL2 (або Linux), тести ганяються з Windows через pymavlink.
Версія ArduPilot зафіксована в [configs/sitl.yaml](configs/sitl.yaml) (Copter-4.7.2).

``` bash
pip install -e ".[sim]"
python -m aerosplat.sim.sitl setup     # раз: окремий worktree ArduPilot + збірка SITL
pytest -m sitl                         # смоук + 9 сценаріїв L2 (доставка, відмови), ~7 хв
python -m aerosplat.sim.runner scenario.yaml   # L2: місія з експорту -> SITL -> оракули -> report.html
python -m aerosplat.sim.sitl start     # вручну (Mission Planner: UDP 14550); ... stop
```

Звичайний `pytest` sitl-тести пропускає. Подробиці, мережа WSL і обмеження —
[docs/SIMULATION_AND_DELIVERY.md §4.8](docs/SIMULATION_AND_DELIVERY.md); раннер сценаріїв,
відмови, оракули й результати — [§4.10](docs/SIMULATION_AND_DELIVERY.md); журнал небезпек —
[docs/HAZARDS.md](docs/HAZARDS.md).

------------------------------------------------------------------------

## Reconstruction Modes Comparison

| Feature | Dense Point Cloud (MVS) | 3D Gaussian Splatting |
|---|---|---|
| Data Representation | XYZ coordinates with RGB attributes | Optimized Gaussian primitives with position, covariance, opacity, and color |
| Scene Model | Discrete sampled geometry | Continuous volumetric scene representation |
| Vegetation Handling | Fragmented and sparse reconstruction | Smooth volumetric reconstruction of complex structures |
| Rendering Pipeline | `THREE.Points` / point-based rendering | GPU-accelerated GLSL Gaussian splatting renderer |
| Memory Consumption | Lower storage requirements | Higher memory usage due to additional Gaussian parameters |
| Visual Quality | Moderate realism | High-fidelity photorealistic reconstruction |
| Real-Time Performance | Limited by point count | Optimized for interactive GPU rendering |

#  Development Status

Current version:

    v0.6.1 (Фаза 0 — стабілізація, див. docs/ROADMAP.md)

Implemented:

-   GPS metadata injection
-   COLMAP SfM reconstruction
-   Dense MVS reconstruction
-   Point cloud processing
-   3D Gaussian Splatting optimization
-   FastAPI backend
-   WebGL visualization frontend

------------------------------------------------------------------------

#  Future Improvements

-   Distance measurement tools
-   Elevation profile generation
-   GIS overlay support
-   Terrain analysis utilities
-   Additional export formats
-   Large-scale scene optimization

------------------------------------------------------------------------

#  License

Код проєкту — MIT. **Виняток:** `diff-gaussian-rasterization` (Inria) —
Gaussian-Splatting License, лише некомерційні дослідження; це опційна
залежність, не частина репозиторію. Докладніше: [docs/DECISIONS.md](docs/DECISIONS.md).
