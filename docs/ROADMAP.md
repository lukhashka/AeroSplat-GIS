# AeroSplat-GIS — Great Plan: від v0.6 до v1.0

> Мета документа — перетворити дослідницький прототип на **відтворюваний,
> вимірювальний, геоприв'язаний** інструмент 3D-реконструкції з БПЛА.
> Базується на повному review коду (2026-09-27); поточна архітектура —
> у [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 0. Підсумок review

### Що вже добре

* **Інженерна культура діагностики.** `doctor.py`, `aero-debug.js`
  (CPU-дзеркало шейдера), явні пост-умови в сервері, контракт `activations`
  у `.pt` — рідкісний рівень рефлексії для прототипу.
* **Коректна математика в критичних місцях**: Σ-проєкція в шейдері,
  column-major `quatToMat`, дилатація + компенсація непрозорості
  (Mip-Splatting), трансформація Σ разом з позиціями (`W·Σ·Wᵀ`),
  екранний градієнт у NDC для densification, фікс `cov3D_precomp.detach()`.
* **Known-poses SfM** з Earth Studio (`earthstudio_to_colmap.py` +
  `point_triangulator` + BA) — правильний підхід, емпірично перевірена
  конвенція осей і модель Землі.
* **Сервер**: робастний кадр сцени (медіана/перцентиль), єдиний кадр для
  обох режимів, zero-copy wire-формат, ETag, LRU-кеш.
* **В'ювер**: інстансовані квади замість `gl.POINTS`, сортування в
  Worker тільки при русі, захисти від TDZ/WebGL-помилок.

### Що заважає рухатись далі

| # | Проблема | Вага |
|---|---|---|
| 1 | Дефолтний `CONFIG` у `starter.py` не відповідає даним `dataset_v6` (див. P0-1) | 🔴 |
| 2 | Тренер мовчки тренує **синтетичну** сцену і перезаписує `_splat.pt`, якщо вхідного `.pt` немає (P0-2) | 🔴 |
| 3 | Ліцензія: `diff-gaussian-rasterization` (Inria) — **лише некомерційне використання**, проєкт заявлений як MIT (P0-5) | 🔴 |
| 4 | ~330 зображень і CSV закомічені в git, `.git` ≈ 2.5 GB loose objects; `data/` не в `.gitignore` | 🟠 |
| 5 | Метрика і геоприв'язка **є** (ENU у known-poses), але не доходять до сервера/в'ювера (`metric: None`) | 🟠 |
| 6 | Немає тестів, CI, lint, пакування; `requirements.txt` без torch/pillow | 🟠 |
| 7 | Немає оцінки якості (train/test split, PSNR/SSIM/LPIPS на відкладених кадрах) | 🟠 |
| 8 | Багато мертвого/легасі коду та дублікатів (v2, TRIPS, `optimization/*`, `mainG.py`, zip-архіви) | 🟡 |
| 9 | Реєстр датасетів захардкоджений; три назви → ті самі файли; «Ціль: Нью-Йорк» в HTML | 🟡 |
| 10 | 3DGS без SH, без анти-аліасингу в тренуванні, без resume; в'ювер на Three.js r128 з лімітом 4.19M сплатів і 64 байти/сплат | 🟡 |
| 11 | Коментарі-«журнали рев'ю» (`review-2 §S3…`) роздувають код і посилаються на відсутні документи (`AeroSplat-viewer-review.md`, `PIPELINE.md`, `DIAGNOSTICS-WIRING.md`) | 🟡 |
| 12 | Особисті файли в `data/dataset_v6/` (`*.xlsx` з розкладом/списками) — ризик випадково закомітити | 🟡 |

---

## 1. Фаза 0 — «Стабілізація» (1 тиждень)

> **Статус (2026-09-30):** P0-1…P0-4 і P0-6 виконано, окрім пунктів з ⏳;
> P0-5 вирішено як «опційна залежність» ([DECISIONS.md](DECISIONS.md)).
> ⏳ Залишилось: `git gc`/`filter-repo` (потребує окремого узгодження),
> видалення `gis.zip`, `src/cloud_src_project.zip`, `.xlsx` (зараз лише в .gitignore),
> семпл-датасет для критерію готовності, пакетні імпорти замість `sys.path`.

Мета: пайплайн запускається з чистого клону однією командою і дає
коректний результат на еталонному датасеті.

### P0-1. Виправити дефолтний конфіг

[starter.py:56](../src/starter.py) — `camera_source: "synthetic"`, орбіта
навколо Запоріжжя (47.00, 34.92), `fov_deg: 60`, `known_intrinsics: True`.
Але `data/dataset_v6/dataset_v6.json` — це Нью-Джерсі (40.89, −74.22),
`fovVertical: 20`. У synthetic-режимі це означає **заморожену в BA хибну
фокусну відстань** (fx≈1663 px замість ≈3062 px) + фейковий GPS.

* Дефолт → `camera_source: "earthstudio_json"` для датасетів, де є JSON;
  автоматичне визначення: якщо `data/<ds>/<ds>.json` існує — використовувати його.
* `synthetic` лишити лише з явним `--camera-source synthetic`, без
  `known_intrinsics` за замовчуванням.
* `sequential` matcher + `loop_detection=True` без vocab tree
  ([sfm_runner.py:445](../src/reconstruction/sfm_runner.py)) — на старих
  COLMAP падає; вимикати loop detection, якщо `vocab_tree` не задано.

### P0-2. Прибрати небезпечні fallback-и

* [gs_trainer_v3.py:1195](../src/reconstruction/gs_trainer_v3.py) —
  відсутній `--input` ⇒ self-test на випадковій хмарі і запис у `--output`.
  Зробити self-test **лише** за явним `--self-test`, інакше — помилка.
* Дефолти CLI тренера (`data/zaporizhzhia_cloud.pt`, `--colmap data/cloud_clo`)
  — видалити, зробити аргументи обов'язковими.
* `recenter=True` у parse-стадії зсуває хмару відносно поз COLMAP і
  тихо ламає тренування — заборонити комбінацію `recenter` + `train`.
* Тренер бере кадри з `raw_images_<ds>` і ігнорує дисторсію. Для
  `--no-known-intrinsics` (модель `SIMPLE_RADIAL`) треба тренуватись на
  `dense/images` + `dense/sparse` (undistorted) — додати це як дефолт.

### P0-3. Гігієна репозиторію

* `.gitignore`: `data/`, `*.pt`, `*.ply`, `*.zip`, `diff-gaussian-rasterization/`,
  `colmap_output_*`, `**/__pycache__`.
* `git rm --cached -r data/` (файли на диску лишаються), `git gc --prune=now`.
  Якщо історію треба схуднути на GitHub — `git filter-repo --path data --invert-paths`
  (переписує історію, force-push — **узгодити окремо**).
* Датасети — у DVC / Git LFS / зовнішнє сховище з `data/README.md`
  «звідки взяти».
* Видалити `gis.zip`, `src/cloud_src_project.zip`, `.xlsx` з `data/dataset_v6/`.

### P0-4. Архівувати легасі

Перенести в `legacy/` (або видалити — все є в git-історії):
`optimization/gs_trainer.py`, `optimization/gs_compressor.py`,
`server/mainG.py`, `scripts/auto_photo.py`, `reconstruction/gs_trainer_v2.py`,
`checker/checker_ply.py`. `trips_trainer.py` і `colmap_loader.py` — або
в `experiments/`, або видалити (`trips_trainer` імпортує з `gs_trainer_v3`).
Дослідницькі скрипти (`diagnose_pose_axis`, `validate_convention`,
`inspector_matches`, `compare_poses`) → `tools/` з нормальними CLI
(у `validate_convention.py` захардкоджено `camera_track.json`).

### P0-5. Ліцензія растеризатора

`graphdeco-inria/diff-gaussian-rasterization` поширюється під
Gaussian-Splatting License (non-commercial, research only). Варіанти:

1. **Рекомендовано:** перейти на [`gsplat`](https://github.com/nerfstudio-project/gsplat)
   (Apache-2.0) — pip-пакет, швидший, підтримує SH, anti-aliasing,
   absgrad, packed mode, depth, batch-рендер. Знімає потребу збирати CUDA
   з MSVC у корені репо.
2. Залишити Inria-растеризатор як опційну залежність і явно вказати
   обмеження в README/LICENSE.

### P0-6. Мінімальне пакування

* `pyproject.toml` (пакет `aerosplat`), `requirements.txt` → повний
  список з версіями (torch, torchvision, numpy, pillow, piexif, fastapi,
  uvicorn); `pyautogui`/`websockets` — прибрати.
* Замінити `sys.path`-імпорти між скриптами (`from cloud_io import ...`,
  `from gs_trainer_v3 import ...`) на нормальні пакетні.
* Оновити README під фактичну структуру (таблиця розбіжностей —
  [ARCHITECTURE.md §8](ARCHITECTURE.md)).

**Критерій готовності Фази 0:** `git clone && pip install -e . &&
aerosplat run --dataset sample` на малому семпл-датасеті (10–20 кадрів,
лежить поза git) проходить всі стадії; `doctor` — без ❌.

---

## 2. Фаза 1 — «Інженерний фундамент» (2–3 тижні)

### 1.1 Структура пакета

```
aerosplat/
├── config/          # pydantic-моделі конфігу + YAML-пресети
├── io/              # colmap (bin/txt), ply, pt/ply-3dgs, exif
├── geo/             # ENU/ECEF/WGS84, геореєстрація, CRS (pyproj)
├── sfm/             # обгортки COLMAP/GLOMAP
├── mvs/
├── gs/              # модель, тренер, densify, метрики, експорт
├── server/          # FastAPI app + роутери
├── pipeline/        # стадії, кешування, run-manifest
└── cli.py           # typer: `aerosplat run|train|serve|doctor|export`
web/                 # в'ювер (див. Фазу 4)
tests/
tools/
```

### 1.2 Конфігурація

* Один `run.yaml` (pydantic-валідація) замість `CONFIG`-словника + 60 CLI-флагів
  у 4 скриптах. Стадії отримують **об'єкт конфігу**, а не argv.
* Стадії викликаються **як функції**, а не `subprocess` Python→Python
  (subprocess лишається тільки для COLMAP). Це дає нормальні traceback-и,
  тести і повторне використання.
* `Paths` → `RunLayout`: `runs/<dataset>/<run_id>/{sfm,mvs,cloud,gs,logs,manifest.json}`.
  Кожен прогін незмінний і відтворюваний; `manifest.json` пише версію коду,
  конфіг, хеші входів, тривалість стадій, ключові метрики.
* Реєстр датасетів сервера читається з `runs/` (або `datasets.yaml`),
  а не з `DATASET_PATHS` у коді. Прибрати аліаси `zaporizhzhia`/`mill19`.

### 1.3 Інкрементальність

* Стадія пропускається, якщо хеш її входів + конфігу не змінився
  (замість ручних `--from/--skip/--skip-stereo`).
* `--fresh` не повинен видаляти нічого, крім директорії своєї стадії.

### 1.4 Якість коду

* `ruff` + `ruff format`, `mypy` (поступово), `eslint`/`prettier` для `web/`.
* Логування через `logging` (рівні, файл у run-директорії) замість `print`
  з emoji; одна функція `_fix_console_encoding` замість 7 копій.
* Коментарі-журнали (`review-2 §…`) — перенести в `docs/decisions/` (ADR),
  у коді лишити 1–2 рядки «чому». Мова коментарів — одна (рекомендую
  англійську в коді, українську в docs).

### 1.5 Тести та CI

* **Unit:** читання/запис COLMAP bin/txt (round-trip), ENU↔WGS84,
  `mat_to_quat_wxyz`, `filter_cloud`, `_activate_splat`, `robust_center_and_extent`,
  wire-формат SPLAT16 (layout/padding).
* **Числові:** `TileRasterizer` vs gsplat на випадковій сцені (max abs diff,
  градієнти) — це вже було перевірено вручну, треба зафіксувати тестом.
* **Інтеграційний smoke:** синтетична сцена → 200 ітерацій → PSNR росте;
  FastAPI `TestClient` для всіх ендпоінтів.
* **Golden-тест конвенції осей** Earth Studio: маленький JSON-фрагмент →
  очікувані `images.txt` (захищає емпірично знайдену конвенцію від регресій).
* GitHub Actions: lint + CPU-тести на кожен PR; GPU-тести — вручну/self-hosted.

---

## 3. Фаза 2 — «Якість реконструкції» (3–4 тижні)

### 2.1 Оцінка якості — спочатку вимірювання

* Train/test split (кожен 8-й кадр, як у Mip-NeRF 360-протоколі).
* Метрики на test: PSNR, SSIM, LPIPS; також кількість Гаусіанів, розмір
  файлу, час тренування, FPS у в'ювері.
* Для known-poses: `compare_poses` як метрика (ΔR°, Δt м після BA).
* Для геометрії: відхилення dense-хмари/центрів Гаусіанів від DSM
  (якщо є еталон) або від тріангульованих контрольних точок.
* Звіт `report.html` у run-директорії (рендер vs GT, diff-карти — як
  у `data/eval/`, але автоматично).

### 2.2 Тренер 3DGS (на базі gsplat)

| Покращення | Навіщо для аерозйомки |
|---|---|
| SH ступеня 3 (з поступовим вмиканням) | дзеркальні дахи, вода, скло |
| Anti-aliasing (Mip-Splatting 3D-фільтр / gsplat `antialiased`) | різні висоти польоту = різні масштаби |
| absgrad / MCMC-densification (3DGS-MCMC) | стабільний бюджет Гаусіанів замість евристик |
| Appearance embedding / per-image exposure | зміна освітлення між прольотами, авто-експозиція дрона |
| Маски неба/рухомих об'єктів | прибирає «флоатери» над сценою |
| Depth/normal-регуляризація з MVS | пласкі дахи і дороги, краща геометрія для вимірювань |
| Scale-регуляризація (обмеження анізотропії) | «голки» на краях сцени |
| Checkpoint/resume, TensorBoard/W&B | довгі прогони, порівняння |
| Епохи з перестановкою камер замість `randint` | рівномірне покриття кадрів |
| `max_screen_radius` ≈ 20 px (зараз = `max_res`, фактично вимкнено) | прибирає роздуті сплати |

Опційно пізніше: **2DGS** (surfel-и, краща поверхня для DSM/мешу),
**Scaffold-GS / Octree-GS** для великих територій.

### 2.3 SfM/MVS

* GLOMAP як дефолт для synthetic-режиму (після бенчмарку).
* Для реальних дронів: GPS-пріор у BA (`--Mapper.use_prior_position` у
  COLMAP 3.10+) або `model_aligner` на EXIF GPS — метрична шкала і для
  режиму без known poses.
* Опційно: сучасні фічі (SuperPoint+LightGlue через hloc) для текстурно
  бідних сцен — `doctor` вже діагностує «замало фіч».

---

## 4. Фаза 3 — «Справжній GIS» (3–4 тижні)

Це те, що відрізнятиме проєкт від чергового 3DGS-в'ювера.

### 3.1 Геореєстрація end-to-end

* `geo_registration.json` (ENU-origin) → `manifest.json` → сервер →
  `scene-meta.metric = {units_to_metres: 1.0, crs: "ENU", origin: {lat,lon,alt}}`.
  Прибрати заглушку `load_metric_scale()`.
* Бібліотека `aerosplat.geo`: WGS84 ↔ ECEF ↔ ENU ↔ UTM через `pyproj`.
  Окремо зафіксувати, що Earth Studio використовує **сферу**, а реальні
  дрони — WGS84 (еліпсоїд); висоти: ellipsoidal vs MSL (геоїд EGM96/2008).
* Для synthetic/реальних дронів без known poses — similarity-трансформація
  COLMAP→ENU за GPS EXIF (Umeyama/RANSAC), з метрикою похибки.

### 3.2 GIS-експорт

* **Ортофотоплан** (GeoTIFF, COG): рендер 3DGS надір-камерою з ортографічною
  проєкцією → растер з georeference.
* **DSM/DEM** (GeoTIFF): з dense-хмари (або depth-рендерів 3DGS) у сітку.
* **Хмара точок** LAS/LAZ з CRS (через `laspy`), EPSG-код.
* **3DGS**: стандартний `.ply` (сумісність з SuperSplat / gsplat / Postshot),
  `.spz` / SOG (стиснені); у перспективі — **OGC 3D Tiles** з Gaussian-
  splat розширенням для CesiumJS.
* **Mesh** (опційно): TSDF з depth-рендерів або 2DGS → glTF.

### 3.3 Вимірювальні інструменти у в'ювері

(з README «Future Improvements» — тепер це можливо, бо є метрика)

* Пікінг точки: ray-cast по dense-хмарі або depth-буфер 3DGS-рендера.
* Відстань, площа полігону, висота над землею, профіль рельєфу.
* Відображення lat/lon/alt під курсором, експорт позначок у GeoJSON.
* Явне маркування «відносні одиниці», якщо `metric == null`.
* Оцінка похибки вимірювань (з BA-ковариацій або контрольних точок).

### 3.4 Контекст карти

* Підкладка (OSM / супутник) під сценою, або вбудовування в **CesiumJS**
  (глобус, терен, інші шари). Хмара точок / сплати позиціонуються за ENU-origin.

---

## 5. Фаза 4 — «Масштаб і продуктивність» (4+ тижні)

### 4.1 Wire-формат і сервер

* Компактний формат: xyz як fp32 (або 16-біт квантовані відносно
  bbox тайла), scale як log-fp16, quat 4×int8, rgb+opacity 4×uint8
  ⇒ **~20–24 байти/сплат** замість 64 (×3 менше трафіку).
* Попередньо зібрані статичні файли (`.ply`/`.spz`/тайли) замість
  побудови payload-у на запит з pickle; сервер — лише каталог + статика.
* `torch.load(weights_only=False)` = виконання pickle; перейти на
  `safetensors`/`.ply`, або хоча б `weights_only=True` (dict тензорів
  це підтримує).
* Потокове/прогресивне завантаження (спочатку грубий LOD).

### 4.2 В'ювер

* Варіант A (швидкий): інтегрувати зрілий рендерер
  (`@mkkellogg/gaussian-splats-3d`, Spark, PlayCanvas/SuperSplat engine),
  зосередитись на GIS-функціях.
* Варіант B (власний, якщо це ціль дослідження): Three.js актуальної
  версії, WebGL2 (`texelFetch`, integer-атрибути — зняти ліміт 4.19M),
  GPU-сортування (WebGPU compute radix sort), SH у шейдері, LOD/тайли,
  frustum culling на рівні тайлів.
* Збірка через Vite + ES-модулі, TypeScript; прибрати CDN-залежність r128.

### 4.3 Великі території

* Поділ сцени на блоки (CityGS / VastGaussian-підхід): тренування по
  блоках з перекриттям, склеювання, LOD-ієрархія.
* Бюджет VRAM 8 GB (RTX 4060): gsplat packed mode, тренування блоками,
  зберігання кадрів у uint8 на GPU.

---

## 6. Фаза 5 — «Реальні дрони та продукт» (далі)

* Вхід: реальні DJI-знімки (XMP gimbal-кути, RTK), відео з вибором
  кадрів (різкість, перекриття), SRT-телеметрія.
* Черга задач (Celery/RQ або простий job-runner) + веб-UI: завантажив
  кадри → бачиш прогрес стадій → відкриваєш сцену.
* Docker-образ з COLMAP + CUDA; документація розгортання.
* Порівняння з еталонами (ODM/WebODM, Metashape, Pix4D) на відкритих
  датасетах (Mill-19, UrbanScene3D) — таблиця в README.

---

## 7. Пріоритети (якщо часу мало)

1. **P0-1, P0-2** — зараз пайплайн за замовчуванням дає некоректний результат.
2. **P0-3, P0-5** — git-репо і ліцензія; що довше чекати, то дорожче.
3. **Метрики якості (2.1)** — без них будь-яке «покращення» тренера — здогадка.
4. **Перехід на gsplat (P0-5 + 2.2)** — одним кроком: ліцензія, швидкість, SH, AA.
5. **Геометрика end-to-end (3.1) + вимірювання (3.3)** — головна цінність «GIS».
6. Усе інше.

---

## 8. Відкриті питання до власника проєкту — ВІДПОВІДІ

Зафіксовано в [DECISIONS.md](DECISIONS.md) (2026-09-30):

* Джерело даних: **Earth Studio зараз**, реальні БПЛА — окремий трек
  ([REAL_DRONES.md](REAL_DRONES.md)).
* Комерційне використання: **не планується** → P0-5 = опційний Inria-растеризатор;
  `gsplat` — у Фазі 2 заради функцій.
* Відображення: **власний в'ювер** (Фаза 4.2, варіант B).
* Територія: **≈1 км²** → Фаза 4.3 (блоки/LOD) не потрібна.
