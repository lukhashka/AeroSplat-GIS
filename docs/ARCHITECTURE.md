# AeroSplat-GIS — поточна архітектура (v0.6.1, гілка `phase-1-foundation`, стан на 2026-10-15, + S1)

> Документ описує **фактичний** стан коду, а не README. Там, де README
> розходиться з кодом, правдою вважається код (див. §8).
> Куди рухаємось — [TARGET_ARCHITECTURE.md](TARGET_ARCHITECTURE.md) і
> [ACCESS_POINTS.md](ACCESS_POINTS.md).

---

## 1. Призначення

Пайплайн, що перетворює аерозйомку (зараз — рендери Google Earth Studio,
що імітують БПЛА) на:

1. **щільну хмару точок** (COLMAP SfM + MVS),
2. **сцену 3D Gaussian Splatting** (власний PyTorch-тренер),

і віддає обидві репрезентації у **браузерний WebGL-в'ювер** через FastAPI.

---

## 2. Загальна схема

Етап S1 ([TARGET_ARCHITECTURE §13](TARGET_ARCHITECTURE.md#13-шлях-від-поточного-стану),
[ADR-0006](decisions/0006-project-site-epoch-capture.md)) додав каталог
Project → Site → Epoch → Capture і дерево `projects/` (§3.9). Схема нижче
показує стадії; звідки вони беруть і куди кладуть файли, визначає
`RunLayout`: або run ділянки (`aerosplat run --site`), або legacy `data/<ds>`
(`aerosplat run --dataset`, лишається до завершення міграції).

```mermaid
flowchart TD
    subgraph INPUT["Вхідні дані: captures/<id>/ ділянки (або legacy data/<ds>/)"]
        F[images/*.jpeg]
        J[source/<ds>.json<br/>Earth Studio 3D tracking]
        CSV[flight_telemetry.csv<br/>опційно]
    end

    subgraph ORCH["aerosplat/starter.py — оркестратор (in-process, pipeline/stages.py)"]
        direction TB
        G["geo<br/>earthstudio_to_colmap.py"]
        E["exif<br/>geo/exif_injector.py"]
        S["sfm<br/>sfm_runner.py → COLMAP"]
        D["dense<br/>dense_runner.py → COLMAP MVS"]
        P["parse<br/>ply_parser.py / colmap_parser.py<br/>(cloud_io.py)"]
        T["train<br/>gs/trainer.py"]
        EV["eval<br/>gs/evaluate.py + report.py"]
    end

    J --> G
    F --> G
    F --> E
    CSV -.-> E
    G -->|known_poses/ (Site ENU) + images/| S
    E -->|images/ з GPS EXIF| S
    S -->|sfm/sparse/0| D
    D -->|mvs/fused.ply| P
    S -->|points3D.bin| P
    P -->|gs/dense.pt / cloud.pt| T
    S -->|пози камер| T
    T -->|gs/splat.pt| RUN
    T --> EV
    EV -->|reports/metrics.json, report.html| RUN[(epochs/<e>/runs/<run_id>/<br/>+ запис у catalog.db)]
    RUN --> SRV

    subgraph SRV["aerosplat/server/main.py — FastAPI"]
        API["/api/v1/scene-meta (+ metric)<br/>/api/v1/spatial-chunk<br/>/api/v1/status<br/>/api/v1/reload"]
    end

    SRV -->|binary Float32 (SPLAT16 / dense6)| WEB

    subgraph WEB["web/ — Three.js r128 + GLSL"]
        V[appGLSL.js — рендерер]
        W[depth-sort_worker.js — сортування]
        DBG[aero-debug.js — діагностика F9]
    end

    DOC["aerosplat/doctor.py — read-only діагностика всіх стадій"] -.-> ORCH
```

---

## 3. Шари та модулі

### 3.1 Оркестрація — `aerosplat/starter.py`

* Єдина точка входу (`aerosplat run`). Стадії: `geo → exif → sfm → dense → parse → train → eval`.
* Стадії — реєстр у `aerosplat/pipeline/stages.py`: кожна будує argv і викликає
  `main(argv)` свого модуля **в тому ж процесі** (скрипти лишаються придатними
  для окремого запуску), тож збій дає звичайний traceback. Конфіг — валідований
  `RunConfig` (`aerosplat/config.py`), перекривається `--config` та `--set KEY=VALUE`.
* `RunLayout` (`aerosplat/pipeline/layout.py`) — єдине місце, де виводяться шляхи,
  і **вид** на одне з двох джерел; стадії бачать лише його поля:
  * `RunLayout.for_site_run` — `aerosplat run --site <project>/<site> [--epoch baseline]
    [--capture <id>] [--run-id <id>]`: кадри й JSON беруться із захоплення ділянки,
    усі виходи йдуть у `sites/<site>/epochs/<epoch>/runs/<run_id>/` (§3.9), ENU-початок —
    `Site.origin`. `aerosplat/pipeline/site_run.py` створює запис run-а в каталозі на
    старті й закриває його (`ok`/`failed` + продукти) наприкінці; `--dry-run` у каталог
    не пише; `--resume` без `--run-id` продовжує останній run епохи;
  * `RunLayout.from_config` — legacy `aerosplat run --dataset <ds>`: `data/raw_images_<ds>`,
    `data/colmap_output_<ds>`, `data/<ds>_*.pt`, як і раніше; ENU-початок = середнє кадрів.
* `manifest.json` + `run.log` у директорії run-а (legacy: `data/<ds>/run/`): конфіг,
  версія/коміт, статус, час, входи/виходи кожної стадії, для run ділянки — ще site, epoch,
  run_id, capture, origin. `--resume` пропускає стадії, що вже успішно пройшли
  з тим самим конфігом і чиї виходи існують (зміна стадії інвалідує наступні).
* Два режими джерела камер (`camera_source`):
  * `synthetic` — пози невідомі, GPS синтезується по колу (exif-стадія),
    COLMAP сам шукає пози (incremental `mapper` / GLOMAP);
  * `earthstudio_json` — реальні пози з Earth Studio JSON →
    `point_triangulator` + `bundle_adjuster` (known-poses SfM).

### 3.2 Геометрія камер / телеметрія

| Модуль | Роль |
|---|---|
| `aerosplat/geo/crs.py` | WGS84 (EPSG:4979) ↔ ECEF (EPSG:4978) ↔ ENU через **pyproj** (`+proj=topocentric`); `GeoPoint`, `EnuFrame` (фіксований початок), матриця ECEF→ENU для орієнтацій. |
| `aerosplat/geo/earthstudio_to_colmap.py` | Earth Studio JSON → `cameras.txt` / `images.txt` / `points3D.txt` у **Site ENU (метри, WGS84)** + `geo_registration.json` (ENU-origin, lat/lon/alt кожного кадру) + `registration_report.json`. ENU-початок приходить ззовні (`--origin`, від Site); без нього — legacy середнє кадрів із попередженням. Сфера Earth Studio (R=6371008.8 м, конвенція осей підтверджена емпірично) живе лише тут: `coordinate` читається як WGS84, орієнтації переносяться без змін (базис ENU при тих самих кутах однаковий на сфері й еліпсоїді). Звіт містить похибку переходу сфера→WGS84 на кадрах і на сітці 1 км² та перевірку експорту (див. §4). Копіює кадри в директорію для SfM і пише GPS EXIF. |
| `aerosplat/geo/exif_injector.py` | Синтетичний GPS EXIF (коло навколо точки або CSV-телеметрія), `piexif.insert` без перекодування JPEG, опційно файл для `model_aligner`. |
| `scripts/auto_photo.py` | Легасі: знімає скріншоти Google Earth через `pyautogui` + пише CSV. |

### 3.3 Фотограмметрія (COLMAP як зовнішній бінарник)

* `sfm_runner.py` — `feature_extractor` → matcher (`exhaustive|sequential|spatial`)
  → `mapper`/`glomap` **або** `point_triangulator` + `bundle_adjuster`.
  Автоматично визначає префікс опцій (`SiftExtraction` vs `FeatureExtraction`)
  за `colmap -h`. Перенумеровує IMAGE_ID known-poses за реальною БД.
* `dense_runner.py` — `image_undistorter` → `patch_match_stereo` → `stereo_fusion`.
  Шляхи приймає явно (`--images-dir/--sparse-dir/--dense-dir`, їх передає стадія з
  `RunLayout`); без них виводить legacy-шляхи з `--dataset`.

### 3.4 Хмара точок — `aerosplat/io/{cloud_io,ply_parser,colmap_parser}.py`

* Векторизоване читання `points3D.bin` та PLY (NumPy).
* Фільтри: non-finite, reprojection error, track length, MAD-викиди, субсемплінг.
* Вихідний контракт `.pt`: `{"xyz": f32[N,3], "rgb": f32[N,3] ∈ [0,1], "meta": {...}}`.

### 3.5 3D Gaussian Splatting — `aerosplat/gs/trainer.py`

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

* Тренування: перестановка камер по епохах, prune за екранним радіусом
  (`max_screen_radius`=20 px), регуляризація анізотропії (`scale_reg`,
  `max_aniso_ratio`); test-спліт кожен 8-й кадр (`eval_every`).
* `gs/evaluate.py` + `gs/report.py` — PSNR/SSIM/LPIPS, FPS, розмір →
  `run/metrics.json` + `report.html` (стадія `eval`, baseline — [BASELINE.md](BASELINE.md)).
* gsplat 1.5.3 збирається на Windows ([tools/gsplat_env.bat](../tools/gsplat_env.bat)),
  але в тренер **ще не інтегрований**.

Легасі-тренери (`gs_trainer_v2.py`, `trips_trainer.py`, `optimization/gs_trainer.py`)
перенесено в `legacy/`; у пайплайні не використовуються.

### 3.6 Сервер — `aerosplat/server/main.py` (FastAPI)

* Реєстр сцен читається з **каталогу** (`aerosplat/store/scenes.py`): одна сцена на
  Site — його останній успішний run із продуктом `splat` або `cloud`. Ключ
  `?dataset=<project>/<site>`; голий `<site>` — аліас, якщо id унікальний. Поки
  міграція не завершена, додатково підхоплюються legacy-виходи `data/<ds>_{dense,cloud,splat}.pt`
  (сайт каталогу з тим самим іменем має пріоритет). Жорстко прописаного `DATASET_PATHS`
  і аліасів `zaporizhzhia`/`mill19` більше немає. `AEROSPLAT_PROJECTS` / `AEROSPLAT_DATA`
  перевизначають корені, `AEROSPLAT_DATASET` — сцену за замовчуванням (інакше перша
  сцена каталогу). Сервер **не створює** каталог, якщо його ще немає.
  `POST /api/v1/reload` без параметрів перечитує каталог.
* Лінива завантаження `.pt` + LRU-кеш payload-ів (бюджет 1.5 GB), ключ
  `(dataset, mode, step)`, де `dataset` — канонічне ім'я (аліаси ділять кеш і ETag).
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
  layout, `metric`). `metric` заповнений, якщо run геоприв'язаний (SfM із відомими
  позами Earth Studio): `{units_to_metres: 1.0, crs: "ENU", datum: "WGS84 (EPSG:4979)",
  origin: {lat, lon, h}, served_offset: "enu = xyz + center", earth_model,
  frame_error_m, site, epoch, run}`. Для run-ів без відомих поз (synthetic GPS,
  вільний SfM) — чесний `None`: одиниці відносні. Заглушки `load_metric_scale()` більше немає.
* `legacy/server/mainG.py` — стара версія сервера.

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

* `aerosplat/doctor.py` — read-only перевірка кожної стадії (кадри, інтринсики,
  EXIF, БД COLMAP, sparse, dense, `.pt`).
* `tools/`: `compare_poses.py`, `diagnose_pose_axis.py`, `validate_convention.py`,
  `inspector_matches.py` — дослідницькі скрипти; `legacy/checker/checker_ply.py` — разовий.
  `doctor.py` поки знає лише legacy-шляхи `data/<ds>`.

### 3.9 Каталог і дерево проєктів — `aerosplat/store/`, `aerosplat/ingest/` (S1, ADR-0006)

* **Доменна модель** (`store/models.py`, pydantic): `Project`, `Site` (`origin: GeoPoint`
  — WGS84, фіксований; `kind`, `bbox`, `taxonomy_packs`), `Epoch`, `Capture` (джерело,
  ліцензія, атрибуція, `metadata_file`, `frame_prefix`), `Image` (sha256, розмір, пріор
  пози WGS84 + σ, поза в Site ENU — після реєстрації, статус, privacy-прапорці), `Run`
  (статус, `earth_model`, `georeferenced`, конфіг, коміт, продукти).
* **Каталог** (`store/db.py`, `store/catalog.py`): SQLite `projects/catalog.db` через
  SQLAlchemy 2 (лише Core-типи — під майбутній PostgreSQL/PostGIS); схема — Alembic
  (`store/migrations/`, застосовується автоматично в `Catalog.open`; тест звіряє міграції
  з моделями). `Catalog` повертає лише pydantic-моделі й поруч пише YAML-копії
  (`project.yaml`, `site.yaml`, `epoch.yaml`, `capture.yaml`, `images.jsonl`).
* **Фіксований початок Site** захищено тричі: API не має операції зміни; ORM-валідатор
  (`OriginImmutableError`); SQLite-тригер `site_origin_immutable`. Повторне створення
  ділянки з іншим origin поверх наявного `site.yaml` також відхиляється.
* **Дерево** (`store/paths.py`, корінь `$AEROSPLAT_PROJECTS` або `<repo>/projects`, у `.gitignore`):

  ```
  projects/catalog.db
  projects/<project>/project.yaml
  projects/<project>/sites/<site>/
  ├── site.yaml
  ├── captures/<capture>/{capture.yaml, images/, images_private/, images.jsonl, source/}
  ├── epochs/<epoch>/{epoch.yaml, runs/<run_id>/}
  │     runs/<run_id>/{manifest.json, run.log, images/, known_poses/, sfm/, mvs/, gs/, reports/}
  ├── authored/      # авторське, пайплайн сюди не пише
  └── missions/
  ```

  `known_poses/` лежить поза `sfm/`, бо `sfm --fresh` видаляє `sfm/` повністю.
* **Ingest** (`ingest/earthstudio.py`, `ingest/folder.py`): експорт Earth Studio (JSON +
  кадри + `.esp`/`ImagerySources.txt`) або тека зображень → `Capture` + `Image[]`. Пози тут
  **не** конвертуються: це робить стадія `geo` кожного run-а з origin ділянки, тож одне
  захоплення служить усім run-ам і епохам.
* **Міграція legacy** (`store/legacy.py`, CLI `tools/migrate_dataset.py` або
  `aerosplat catalog migrate-dataset`): `data/<ds>` → Site `default/<ds>`; origin = ENU-початок
  legacy run-а (`sparse_known/geo_registration.json`), тож перенесений run лишається в
  системі ділянки з точністю до переходу сфера→WGS84; run позначається
  `earth_model="sphere-legacy"`, а `reports/registration.json` фіксує його відхилення
  (`legacy_shift`). Копіює за замовчуванням (`--move` — переносить), ідемпотентна,
  `--dry-run` показує план; сторонні файли (`*.xlsx` тощо) лишаються в `data/<ds>`.
* **CLI** `aerosplat catalog list | create-site | import-earthstudio | import-folder |
  migrate-dataset | upgrade | revision`.

`dataset_v6` перенесено: `default/dataset_v6`, origin (40.89104597, −74.21634205, 233.0),
захоплення `earthstudio-dataset_v6` (31 кадр), run `20260930-215021-legacy`
(`sphere-legacy`, геоприв'язаний, `frame_error_m` ≈ 1.53 м). Legacy-копія в `data/` лишилась.

---

## 4. Системи координат (ключове для GIS)

```
Earth Studio coordinate (lat, lon, alt)  ── читається як WGS84 (EPSG:4979) ──┐
Earth Studio position (сфера R=6371008.8) ── лише для звіту/перевірки ────────┤
                                                                             ▼
             earthstudio_to_colmap + aerosplat.geo.crs (pyproj topocentric):
             Site ENU (метри), origin = Site.origin — фіксований при створенні Site
                                                                             ▼
COLMAP world (known-poses)  = Site ENU, метри  ← метрика ЗБЕРЕЖЕНА
COLMAP world (synthetic)    = довільна подібність, одиниці НЕ метри
                                                                             ▼
gs_trainer / .pt            = та сама система, що COLMAP (Z-up)
                                                                             ▼
server                      = мінус median-центр (float32-безпечно); scene-meta.metric:
                              enu = xyz + center, origin (WGS84) → будь-яка точка в lat/lon/h
                                                                             ▼
viewer sceneRoot            = Rx(+90°) · (100 / extent)   → Y-up, радіус 100
```

Метрика й геоприв'язка тепер **доходять** до сервера (`scene-meta.metric`); в'ювер
їх ще не використовує (Фаза 3.3). Legacy-run (`aerosplat run --dataset`) будується
так само, але з origin = середнє кадрів (воно зсувається з кожним новим захопленням).

**Перехід сфера → WGS84, виміряно на `dataset_v6`** (орбіта радіусом ≈386 м,
origin = legacy-центр): горизонтальне зміщення кадрів до 0.96 м (RMS 0.76 м), на
кутах ділянки 1×1 км — до 1.43 м; вертикальне — < 0.1 мм. Це різниця масштабів
сфери й еліпсоїда на 41° пн. ш. (≈0.1–0.3 %), а не шум. Окремо: `position` Earth Studio
лежить на **сталі 1.2 м** далі від центру Землі, ніж `R + altitude` (кутова розбіжність
≈1e-9 м) — сумісно з R≈6371010 м у Google; legacy-кадр через це був ще й піднятий на
1.2 м. Разом `legacy_shift` для `dataset_v6` — до 1.53 м. Яка модель правдива для
рендерів Google (і чи `altitude` — еліпсоїдальна висота), ще треба перевірити
емпірично (порівняти кількість тріангульованих точок, як у `validate_convention.py`);
до того припущення записано у звіті (`altitude_reference`).

---

## 5. Контракти даних

Шляхи — відносно директорії run-а ділянки (у дужках — legacy `data/`).

| Артефакт | Виробник | Споживач | Формат |
|---|---|---|---|
| `images/*.jpeg` (`raw_images_<ds>/`) | geo / exif | sfm, dense, train | JPEG + GPS EXIF |
| `known_poses/{cameras,images,points3D}.txt` (`<ds>/sparse_known/`) | geo | sfm | COLMAP text |
| `known_poses/geo_registration.json` | geo | migrate, server (legacy) | JSON: ENU-origin, lat/lon/alt кадрів |
| `reports/registration.json` (`sparse_known/registration_report.json`) | geo, migrate | server (`frame_error_m`), людина | JSON: перехід сфера→WGS84, перевірка експорту |
| `sfm/sparse/0/*.bin` (`colmap_output_<ds>/sparse/0/`) | sfm | dense, parse, train | COLMAP binary |
| `mvs/fused.ply` (`colmap_output_<ds>/dense/`) | dense | parse | PLY |
| `gs/dense.pt`, `gs/cloud.pt` (`<ds>_dense.pt`, `<ds>_cloud.pt`) | parse | train, server | `torch.save` dict (pickle) |
| `gs/splat.pt` (`<ds>_splat.pt`) | train | server | `torch.save` dict (pickle) |
| `reports/metrics.json`, `report.html` (`<ds>/run/`) | eval | людина | JSON / HTML |
| рядок `runs` + `products` у `catalog.db` | `run --site`, migrate | server | SQLite (шляхи відносно run-а) |
| HTTP `spatial-chunk` | server | viewer | raw Float32 LE |

---

## 6. Технологічний стек

* Python 3.11, PyTorch 2.6 + CUDA 12.6, NumPy, Pillow, piexif, FastAPI/uvicorn.
* pyproj (WGS84/ECEF/ENU), SQLAlchemy 2 + Alembic, SQLite (каталог).
* COLMAP (зовнішній бінарник у PATH), опційно GLOMAP.
* `diff-gaussian-rasterization` (Inria, CUDA-розширення, збирається з MSVC).
* Three.js r128 (2021) з CDN, GLSL ES 1.0, Web Worker.
* Цільове залізо: RTX 4060 (8 GB), Windows 11.

---

## 7. Структура репозиторію (фактична)

```
AeroSplat-GIS/
├── src/aerosplat/                 # пакет (pip install -e .)
│   ├── starter.py, doctor.py      # оркестратор (`aerosplat run`), діагностика
│   ├── config.py, paths.py, logging_utils.py
│   ├── pipeline/                  # stages, runner, layout (RunLayout), manifest, site_run
│   ├── geo/                       # crs (pyproj), earthstudio_to_colmap, exif_injector, compare_poses
│   ├── store/                     # models, db, catalog, paths, scenes, legacy, cli, migrations/ (Alembic)
│   ├── ingest/                    # earthstudio, folder, images
│   ├── sfm/                       # sfm_runner, dense_runner
│   ├── io/                        # cloud_io, ply_parser, colmap_parser
│   ├── gs/                        # trainer (активний), evaluate, report
│   └── server/main.py             # FastAPI
├── web/                           # index.html, appGLSL.js, depth-sort_worker.js, aero-debug.js
├── configs/                       # earthstudio.yaml, drone.yaml (пресети RunConfig)
├── tests/                         # CPU-тести + golden-фікстури Earth Studio
├── tools/                         # дослідницькі скрипти, gsplat_env.bat, migrate_dataset.py
├── legacy/                        # optimization/, reconstruction/ (v2, TRIPS), scripts/, server/mainG.py, checker/
├── docs/                          # ARCHITECTURE, TARGET_ARCHITECTURE, ACCESS_POINTS, ROADMAP, decisions/
├── diff-gaussian-rasterization/   # сторонній репо (ignored, non-commercial)
├── data/                          # legacy-датасети (ignored; у git лише data/README.md)
├── projects/                      # дерево проєктів + catalog.db (ignored, ADR-0006)
└── pyproject.toml, requirements.txt, package.json (eslint)
```

---

## 8. Розбіжності README ↔ код

| README каже | Насправді |
|---|---|
| `exif_injector.py` у `reconstruction/` | у `aerosplat/geo/` |
| тренер `optimization/gs_trainer.py` | активний — `aerosplat/gs/trainer.py` |
| `web/app.js`, `orbit_controls.js` | `appGLSL.js`, OrbitControls з CDN; `app.js` видалено |
| кадри в `data/raw_images_3d/` | `data/<ds>/footage/` → `data/raw_images_<ds>/` |
| запуск 5 скриптів вручну | є `starter.py` |
| `pip install fastapi uvicorn pillow pyautogui piexif` | потрібні ще numpy, torch, COLMAP, збірка CUDA-растеризатора |
| «1 unit ≈ 1 m» | правда лише для `earthstudio_json` (тоді `scene-meta.metric` заповнений); в'ювер перемасштабовує сцену до радіуса 100 |
| запуск через `--dataset` | основний шлях тепер `aerosplat run --site <project>/<site>` + `aerosplat catalog`; README цього ще не описує |
