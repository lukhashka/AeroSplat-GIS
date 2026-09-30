# AeroSplat-GIS — поточна архітектура (v0.6, стан робочого дерева на 2026-09-27)

> Документ описує **фактичний** стан коду, а не README. Там, де README
> розходиться з кодом, правдою вважається код (див. §8).

---

## 1. Призначення

Пайплайн, що перетворює аерозйомку (зараз — рендери Google Earth Studio,
що імітують БПЛА) на:

1. **щільну хмару точок** (COLMAP SfM + MVS),
2. **сцену 3D Gaussian Splatting** (власний PyTorch-тренер),

і віддає обидві репрезентації у **браузерний WebGL-в'ювер** через FastAPI.

---

## 2. Загальна схема

```mermaid
flowchart TD
    subgraph INPUT["Вхідні дані (data/<ds>/)"]
        F[footage/*.jpeg]
        J[<ds>.json<br/>Earth Studio 3D tracking]
        CSV[flight_telemetry.csv<br/>опційно]
    end

    subgraph ORCH["src/starter.py — оркестратор (in-process)"]
        direction TB
        G["geo<br/>earthstudio_to_colmap.py"]
        E["exif<br/>telemetry/exif_injector.py"]
        S["sfm<br/>sfm_runner.py → COLMAP"]
        D["dense<br/>dense_runner.py → COLMAP MVS"]
        P["parse<br/>ply_parser.py / colmap_parser.py<br/>(cloud_io.py)"]
        T["train<br/>gs_trainer_v3.py"]
    end

    J --> G
    F --> G
    F --> E
    CSV -.-> E
    G -->|sparse_known/ + raw_images_<ds>| S
    E -->|raw_images_<ds> з GPS EXIF| S
    S -->|colmap_output_<ds>/sparse/0| D
    D -->|dense/fused.ply| P
    S -->|points3D.bin| P
    P -->|data/<ds>_dense.pt / _cloud.pt| T
    S -->|пози камер| T
    T -->|data/<ds>_splat.pt| SRV

    subgraph SRV["src/server/main.py — FastAPI"]
        API["/api/v1/scene-meta<br/>/api/v1/spatial-chunk<br/>/api/v1/status<br/>/api/v1/reload"]
    end

    SRV -->|binary Float32 (SPLAT16 / dense6)| WEB

    subgraph WEB["web/ — Three.js r128 + GLSL"]
        V[appGLSL.js — рендерер]
        W[depth-sort_worker.js — сортування]
        DBG[aero-debug.js — діагностика F9]
    end

    DOC["src/doctor.py — read-only діагностика всіх стадій"] -.-> ORCH
```

---

## 3. Шари та модулі

### 3.1 Оркестрація — `src/starter.py`

* Єдина точка входу. Стадії: `geo → exif → sfm → dense → parse → train`.
* Стадії — реєстр у `aerosplat/pipeline/stages.py`: кожна будує argv і викликає
  `main(argv)` свого модуля **в тому ж процесі** (скрипти лишаються придатними
  для окремого запуску), тож збій дає звичайний traceback. Конфіг — валідований
  `RunConfig` (`aerosplat/config.py`), перекривається `--config` та `--set KEY=VALUE`.
* `RunLayout` (`aerosplat/pipeline/layout.py`) — єдине місце, де виводяться шляхи
  з імені датасету (`data/raw_images_<ds>`, `data/colmap_output_<ds>`, `data/<ds>_*.pt`).
* `data/<ds>/run/manifest.json` + `run.log`: конфіг, версія/коміт, статус, час,
  входи/виходи кожної стадії. `--resume` пропускає стадії, що вже успішно пройшли
  з тим самим конфігом і чиї виходи існують (зміна стадії інвалідує наступні).
* Два режими джерела камер (`camera_source`):
  * `synthetic` — пози невідомі, GPS синтезується по колу (exif-стадія),
    COLMAP сам шукає пози (incremental `mapper` / GLOMAP);
  * `earthstudio_json` — реальні пози з Earth Studio JSON →
    `point_triangulator` + `bundle_adjuster` (known-poses SfM).

### 3.2 Геометрія камер / телеметрія

| Модуль | Роль |
|---|---|
| `reconstruction/earthstudio_to_colmap.py` | Earth Studio JSON → `cameras.txt` / `images.txt` / `points3D.txt` у **локальній ENU-системі (метри)** + `geo_registration.json` (ENU-origin, lat/lon/alt кожного кадру). Модель Землі — сфера R=6371008.8 м (емпірично підтверджено). Копіює кадри в `raw_images_<ds>` і пише GPS EXIF. |
| `telemetry/exif_injector.py` | Синтетичний GPS EXIF (коло навколо точки або CSV-телеметрія), `piexif.insert` без перекодування JPEG, опційно файл для `model_aligner`. |
| `scripts/auto_photo.py` | Легасі: знімає скріншоти Google Earth через `pyautogui` + пише CSV. |

### 3.3 Фотограмметрія (COLMAP як зовнішній бінарник)

* `sfm_runner.py` — `feature_extractor` → matcher (`exhaustive|sequential|spatial`)
  → `mapper`/`glomap` **або** `point_triangulator` + `bundle_adjuster`.
  Автоматично визначає префікс опцій (`SiftExtraction` vs `FeatureExtraction`)
  за `colmap -h`. Перенумеровує IMAGE_ID known-poses за реальною БД.
* `dense_runner.py` — `image_undistorter` → `patch_match_stereo` → `stereo_fusion`.
  Шляхи виводить **сам** із `--dataset` (не приймає `--images-dir/--output-dir`).

### 3.4 Хмара точок — `cloud_io.py`, `ply_parser.py`, `colmap_parser.py`

* Векторизоване читання `points3D.bin` та PLY (NumPy).
* Фільтри: non-finite, reprojection error, track length, MAD-викиди, субсемплінг.
* Вихідний контракт `.pt`: `{"xyz": f32[N,3], "rgb": f32[N,3] ∈ [0,1], "meta": {...}}`.

### 3.5 3D Gaussian Splatting — `reconstruction/gs_trainer_v3.py`

* Модель: `xyz`, `log scale`, кватерніон `(w,x,y,z)`, `logit opacity`,
  `logit rgb` (**без сферичних гармонік** — колір не залежить від ракурсу).
* Ініціалізація масштабу — наближений KNN через Morton-код.
* Два растеризатори:
  * `CudaRasterizer` — обгортка над `diff-gaussian-rasterization`
    (Inria, збирається локально, лежить у корені репо, untracked);
  * `TileRasterizer` — власний диференційований tile-based растеризатор на
    чистому PyTorch (fallback).
* Loss `0.8·L1 + 0.2·(1−SSIM)`, Adam з lr за Kerbl et al., експоненційний
  decay lr позицій, densify (clone/split) за екранним градієнтом, prune,
  opacity reset, бюджет `max_gaussians`.
* Вихід: `{xyz, scale, rotation, opacity, rgb, activations}` — **вже
  активовані** значення + явний контракт `activations`.

Паралельно існують експериментальні/легасі тренери: `gs_trainer_v2.py`,
`trips_trainer.py` (TRIPS без нейро-реконструктора), `optimization/gs_trainer.py`
(v0.1, «рандомний GT»). У пайплайні не використовуються.

### 3.6 Сервер — `src/server/main.py` (FastAPI)

* `DATASET_PATHS` — жорстко прописаний реєстр датасетів
  (`zaporizhzhia`, `dataset_v6`, `mill19` → **усі три вказують на ті самі файли**).
* Лінива завантаження `.pt` + LRU-кеш payload-ів (бюджет 1.5 GB), ключ
  `(dataset, mode, step)`.
* Робастний «кадр сцени»: медіанний центр + 99.5-перцентиль радіуса,
  відсікання викидів > 3×extent, один кадр на обидва режими.
* Сервер **лише центрує** (трансляція), поворот осі Z-up→Y-up і масштаб
  застосовує в'ювер через `sceneRoot` (щоб не розсинхронізувати кватерніони).
* Wire-формат:
  * **SPLAT16** — 16 float32 на сплат = 4 RGBA32F-тексели
    (`xyz, opacity | scale, pad | quat wxyz | rgb, pad`), доповнено до
    цілого рядка текстури шириною 2048;
  * **dense** — 6 float32 (`xyz, rgb`).
* ETag + `Cache-Control`, `scene-meta` як JSON (центр, extent, scale,
  layout, `metric` — поки завжди `None`/«unparsed»).
* `mainG.py` — стара версія сервера (легасі).

### 3.7 В'ювер — `web/`

* `index.html` — Three.js **r128** та `OrbitControls` з CDN, класичні скрипти.
* `appGLSL.js`:
  * сплати — **інстансовані квади**, параметри читаються з float-текстури
    за `aSplatIndex`; вершинний шейдер будує Σ, проєктує через Якобіан,
    дилатація 0.3 px + компенсація непрозорості (Mip-Splatting), квад
    вирівняний по власних осях еліпса, cutoff 3.5σ;
  * dense — `THREE.Points` з розміром точки за оцінкою щільності;
  * NormalBlending, back-to-front, `depthWrite=false`.
* `depth-sort_worker.js` — counting sort по 65 536 бакетах глибини у Web
  Worker; сортування запускається лише при реальному русі камери.
* `aero-debug.js` — панель діагностики (F9): статистика реальних байтів,
  CPU-дзеркало вершинного шейдера, синтетична тест-сцена.
* Ліміт адресації: `aSplatIndex*4` у float32 → максимум **4 194 304** сплатів.

### 3.8 Діагностика

* `src/doctor.py` — read-only перевірка кожної стадії (кадри, інтринсики,
  EXIF, БД COLMAP, sparse, dense, `.pt`).
* `compare_poses.py`, `diagnose_pose_axis.py`, `validate_convention.py`,
  `inspector_matches.py`, `checker/checker_ply.py` — разові дослідницькі скрипти
  (частина з них має захардкоджені шляхи на кшталт `camera_track.json`).

---

## 4. Системи координат (ключове для GIS)

```
Earth Studio position  ── сферичний геоцентр (R=6371008.8) ──┐
                                                             ▼
                         earthstudio_to_colmap: ENU (метри), origin = mean(lat,lon,alt)
                                                             ▼
COLMAP world (known-poses)  = ENU, метри  ← метрика ЗБЕРЕЖЕНА
COLMAP world (synthetic)    = довільна подібність, одиниці НЕ метри
                                                             ▼
gs_trainer / .pt            = та сама система, що COLMAP (Z-up)
                                                             ▼
server                      = мінус median-центр (float32-безпечно)
                                                             ▼
viewer sceneRoot            = Rx(+90°) · (100 / extent)   → Y-up, радіус 100
```

Висновок: у режимі `earthstudio_json` сцена вже **метрична і
геоприв'язана** (ENU-origin лежить у `data/<ds>/sparse_known/geo_registration.json`),
але ця інформація **не доходить** до сервера і в'ювера.

---

## 5. Контракти даних

| Артефакт | Виробник | Споживач | Формат |
|---|---|---|---|
| `raw_images_<ds>/*.jpeg` | geo / exif | sfm, dense, train | JPEG + GPS EXIF |
| `<ds>/sparse_known/{cameras,images,points3D}.txt` | geo | sfm | COLMAP text |
| `colmap_output_<ds>/sparse/0/*.bin` | sfm | dense, parse, train | COLMAP binary |
| `colmap_output_<ds>/dense/fused.ply` | dense | parse | PLY |
| `<ds>_dense.pt`, `<ds>_cloud.pt` | parse | train, server | `torch.save` dict (pickle) |
| `<ds>_splat.pt` | train | server | `torch.save` dict (pickle) |
| HTTP `spatial-chunk` | server | viewer | raw Float32 LE |

---

## 6. Технологічний стек

* Python 3.11, PyTorch 2.6 + CUDA 12.6, NumPy, Pillow, piexif, FastAPI/uvicorn.
* COLMAP (зовнішній бінарник у PATH), опційно GLOMAP.
* `diff-gaussian-rasterization` (Inria, CUDA-розширення, збирається з MSVC).
* Three.js r128 (2021) з CDN, GLSL ES 1.0, Web Worker.
* Цільове залізо: RTX 4060 (8 GB), Windows 11.

---

## 7. Структура репозиторію (фактична)

```
AeroSplat-GIS/
├── src/
│   ├── starter.py                 # оркестратор
│   ├── doctor.py                  # діагностика
│   ├── telemetry/exif_injector.py
│   ├── reconstruction/
│   │   ├── earthstudio_to_colmap.py, sfm_runner.py, dense_runner.py
│   │   ├── cloud_io.py, ply_parser.py, colmap_parser.py
│   │   ├── gs_trainer_v3.py       # АКТИВНИЙ тренер
│   │   ├── gs_trainer_v2.py, trips_trainer.py, colmap_loader.py   # не в пайплайні
│   │   └── compare_poses.py, diagnose_pose_axis.py, validate_convention.py, inspector_matches.py
│   ├── optimization/gs_trainer.py, gs_compressor.py               # легасі v0.1
│   ├── server/main.py (активний), mainG.py (легасі)
│   ├── scripts/auto_photo.py      # легасі
│   ├── checker/checker_ply.py     # разовий скрипт
│   └── cloud_src_project.zip      # архів коду (untracked)
├── web/index.html, appGLSL.js, depth-sort_worker.js, aero-debug.js
├── diff-gaussian-rasterization/   # сторонній репо + build-артефакти (untracked)
├── data/                          # датасети, частково ЗАКОМІЧЕНІ в git
├── gis.zip                        # архів (untracked)
├── venv/                          # ігнорується
├── requirements.txt               # неповний
└── README.md                      # застарів
```

---

## 8. Розбіжності README ↔ код

| README каже | Насправді |
|---|---|
| `exif_injector.py` у `reconstruction/` | у `telemetry/` |
| тренер `optimization/gs_trainer.py` | активний — `reconstruction/gs_trainer_v3.py` |
| `web/app.js`, `orbit_controls.js` | `appGLSL.js`, OrbitControls з CDN; `app.js` видалено |
| кадри в `data/raw_images_3d/` | `data/<ds>/footage/` → `data/raw_images_<ds>/` |
| запуск 5 скриптів вручну | є `starter.py` |
| `pip install fastapi uvicorn pillow pyautogui piexif` | потрібні ще numpy, torch, COLMAP, збірка CUDA-растеризатора |
| «1 unit ≈ 1 m» | правда лише для `earthstudio_json`; в'ювер перемасштабовує сцену до радіуса 100 |
