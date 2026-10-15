# AeroSplat-GIS — поточна архітектура (v0.6.1, гілка `phase-1-foundation`, стан на 2026-10-15, + S1, + Фаза 2.2 gsplat)

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
        GR["georef<br/>geo/georef.py"]
    end

    J --> G
    F --> G
    F --> E
    CSV -.-> E
    G -->|known_poses/ (сферичний ENU) + images/| S
    G -->|reports/registration.json (A, b)| GR
    E -->|images/ з GPS EXIF| S
    S -->|sfm/sparse/0| D
    D -->|mvs/fused.ply| P
    S -->|points3D.bin| P
    P -->|gs/dense.pt / cloud.pt| T
    S -->|пози камер| T
    T --> EV
    P --> GR
    T -->|gs/splat.pt| GR
    GR -->|gs/*_wgs84.pt (WGS84 Site ENU)| RUN
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

* Єдина точка входу (`aerosplat run`). Стадії: `geo → exif → sfm → dense → parse → train → eval → georef`.
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
| `aerosplat/geo/earthstudio_to_colmap.py` | Earth Studio JSON → `cameras.txt` / `images.txt` / `points3D.txt` у **сферичному ENU ділянки** (метри) + `geo_registration.json` (ENU-origin, відображення сферичний ENU → WGS84 Site ENU, lat/lon/alt кожного кадру з обома ENU-позиціями) + `registration_report.json`. ENU-початок приходить ззовні (`--origin`, від Site); без нього — legacy середнє кадрів із попередженням. Центри камер — з `coordinate` як сферичних кутів на радіусі R + altitude (R=6371008.8 м, той самий для кадрів і origin), `position` для поз не використовується; орієнтації — без змін (конвенція осей підтверджена емпірично). Звіт: афінне відображення та його залишки, наскільки воно зсуває точки (кадри, сітка 1 км²), `legacy_shift`, перевірка експорту (див. §4). Копіює кадри в директорію для SfM і пише GPS EXIF. |
| `aerosplat/geo/sphere_enu.py` | Сфера Earth Studio: `SphereEnuFrame` (ENU на сфері навколо origin), точне відображення сферичний ENU → WGS84 Site ENU і його афінна апроксимація МНК на коробці 1×1 км × ±250 м (`fit_site_affine`, залежить лише від origin ділянки). `EARTH_MODEL = "sphere-enu+affine"`. |
| `aerosplat/geo/affine.py` | `AffineMap`: точки x → Ax + b; гаусіани точно: μ → Aμ + b, Σ → AΣAᵀ (масштаби й кватерніони з SVD від A·R·S, порядок осей зберігається). |
| `aerosplat/geo/georef.py` | Стадія `georef`: застосовує відображення зі звіту до `gs/{dense,cloud}.pt` і `gs/splat.pt` → `gs/*_wgs84.pt` (ключ `frame` у `.pt` каже, що файл уже в WGS84 Site ENU). Opacity й колір не змінюються. |
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

### 3.5 3D Gaussian Splatting — `aerosplat/gs/trainer.py`, `gs/gsplat_backend.py`

* Спільна підготовка: хмара `.pt`, камери COLMAP + кадри, test-спліт, радіус
  сцени, стартовий масштаб — наближений KNN через Morton-код.
* Три растеризатори (`rasterizer` у `RunConfig`):
  * `cuda` — `CudaRasterizer`, обгортка над `diff-gaussian-rasterization`
    (Inria, збирається локально, лежить у корені репо, untracked);
  * `torch` — `TileRasterizer`, власний диференційований tile-based растеризатор на
    чистому PyTorch (fallback; `auto` = `cuda`, якщо зібраний, інакше `torch`).
    Семплює Гаусіани в центрах пікселів (+0.5), як COLMAP/Inria/gsplat; збіг із
    gsplat `classic` перевіряє GPU-тест у `tests/test_gsplat.py`;
  * `gsplat` (Фаза 2.2) — gsplat 1.5.3, окремий цикл `train_gsplat` у
    `gs/gsplat_backend.py`. Обирається лише явно: JIT-збірка CUDA-розширення,
    на Windows — через [tools/gsplat_env.bat](../tools/gsplat_env.bat);
    `require_gsplat()` перевіряє середовище (CUDA toolkit = версія torch, MSVC ≠ VS 18)
    до першого виклику, бо невдала JIT-збірка псує кеш.

Шлях `cuda` / `torch` (`GaussianModel` + `GaussianDensifier`):

* Loss `0.8·L1 + 0.2·(1−SSIM)`, Adam з lr за Kerbl et al., експоненційний
  decay lr позицій, densify (clone/split) за екранним градієнтом, prune,
  opacity reset, бюджет `max_gaussians`.
* Вихід: `{xyz, scale, rotation, opacity, rgb, activations}` — **вже
  активовані** значення + явний контракт `activations`.

* Тренування: перестановка камер по епохах, prune за екранним радіусом
  (`max_screen_radius`=20 px), регуляризація анізотропії (`scale_reg`,
  `max_aniso_ratio`); test-спліт кожен 8-й кадр (`eval_every`); train-кадр
  з позою test-кадру (напр. замикальний кадр орбіти = кадр 0) виключається
  з тренування (`find_test_twins`), щоб test не протікав у train.

Шлях `gsplat` (`SplatModel` + `GsplatDensifier` / `McmcDensifier`):

* Параметри в `ParameterDict` у розкладці gsplat (`means`, `scales`, `quats`,
  `opacities`, `sh0`, `shN`), по одному Adam на ключ; хірургія параметрів при
  densify — `gsplat.strategy.ops`, тож стан Adam іде за гаусіанами.
* Колір — SH до 3-го ступеня (`sh_degree`), +1 ступінь кожні `sh_degree_interval`
  ітерацій. `rasterize_mode`: `antialiased` (за замовчуванням) — 2D-фільтр
  Mip-Splatting з компенсацією непрозорості, як у шейдері в'ювера (ADR-0004 §C2).
* `densify_strategy`: `mcmc` (за замовчуванням) — 3DGS-MCMC (`MCMCStrategy` gsplat:
  релокація «мертвих» гаусіанів, ріст до `max_gaussians`, шум позицій; шум і
  регуляризатори перераховано з нормалізованої сцени на метри через радіус сцени);
  `default` — правила гілки `cuda` (бюджет `max_gaussians`, prune за екранним
  радіусом, opacity reset), за бажання на абсолютних градієнтах (`absgrad`, поріг
  `absgrad_threshold`). Порівняння — [BASELINE.md](BASELINE.md).
* `appearance: exposure` — афінна корекція кольору 3×4 на кожен train-кадр, лише в
  тренуванні (спільний зсув віднімається); для змішаних джерел (TARGET_ARCHITECTURE §6.2).
* Перестановка камер, prune за екранним радіусом і регуляризація анізотропії —
  ті самі, що в гілці `cuda`.
* Вихід — той самий контракт плюс `sh (N, 16, 3)`, `sh_degree`, `render`; `rgb` =
  DC-складова SH (ADR-0002, доповнення 2026-10-15). SH3 збільшує `.pt` ≈3.7×
  (≈470 МБ на 2M гаусіанів).

Оцінка й геоприв'язка:

* `gs/evaluate.py` + `gs/report.py` — PSNR/SSIM/LPIPS, FPS, розмір →
  `run/metrics.json` + `report.html` (стадія `eval`, baseline — [BASELINE.md](BASELINE.md)).
  З `rasterizer=gsplat` test-кадри рендеряться з SH і `rasterize_mode` із `.pt`.
* `geo/georef.py` переносить сплат у WGS84 Site ENU: Σ → AΣAᵀ, а SH обертаються
  обертальною частиною A (полярний розклад, `gs/sh.py`). На dataset_v6 A діагональна,
  тож обертання нульове.

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
  Для Earth Studio run-ів (`sphere-enu+affine`) сервер віддає продукти стадії `georef`
  (`gs/*_wgs84.pt`), тож координати справді у WGS84 Site ENU, а `frame_error_m` — залишок
  афінного відображення (< 1 мм); для `sphere-legacy` — `legacy_shift` (≈1.5 м).
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
  `earth_model`: `sphere-enu+affine` — run Earth Studio з відомими позами (реконструкція у
  сферичному ENU, продукти переведено в WGS84 Site ENU, §4); `wgs84` — пози переводились у
  WGS84 на вході (run-и S1; так само позначаються run-и без геоприв'язки); `sphere-legacy` —
  перенесені з-до ADR-0006. Продукти `splat`/`cloud` геоприв'язаного run-а — це
  `gs/*_wgs84.pt`; файли в системі реконструкції лишаються як `splat_recon`/`cloud_recon`.
  Продовжити (`--resume`/`--run-id`) run з іншим `earth_model`, ніж дає поточний конвеєр,
  не можна — лише новий run.
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
Далі в ділянці: run `20261015-023347` (`wgs84`, пози переведено у WGS84 на вході — неузгоджені
з кадрами, §4) і run `20261015-031843` (`sphere-enu+affine`, поточний «latest», його показує сервер).

---

## 4. Системи координат (ключове для GIS)

```
Earth Studio coordinate (lat, lon, alt)  ── сферичні кути, радіус R + alt (R=6371008.8) ──┐
Earth Studio position (сфера, +1.2 м)    ── лише для звіту/перевірки ─────────────────────┤
                                                                                          ▼
             earthstudio_to_colmap + aerosplat.geo.sphere_enu:
             сферичний ENU ділянки (метри), origin = Site.origin (той самий R) — фіксований
                                                                                          ▼
COLMAP world (known-poses)  = сферичний ENU, метри  ← узгоджений із рендерами Earth Studio
COLMAP world (synthetic)    = довільна подібність, одиниці НЕ метри
                                                                                          ▼
gs_trainer / gs/*.pt        = та сама система, що COLMAP (Z-up); з нею працює eval
                                                                                          ▼
georef → gs/*_wgs84.pt      = A·x + b (Σ → AΣAᵀ): WGS84 Site ENU (pyproj topocentric),
                              A, b — з reports/registration.json, залишок < 1 мм на 1 км²
                                                                                          ▼
server                      = мінус median-центр (float32-безпечно); scene-meta.metric:
                              enu = xyz + center, origin (WGS84) → будь-яка точка в lat/lon/h
                                                                                          ▼
viewer sceneRoot            = Rx(+90°) · (100 / extent)   → Y-up, радіус 100
```

Метрика й геоприв'язка тепер **доходять** до сервера (`scene-meta.metric`); в'ювер
їх ще не використовує (Фаза 3.3). Legacy-run (`aerosplat run --dataset`) будується
так само, але з origin = середнє кадрів (воно зсувається з кожним новим захопленням);
його продукти в WGS84 — `data/<ds>_*_wgs84.pt`, і лише з ними сервер дає `metric`.

**Перехід сфера → WGS84, виміряно на `dataset_v6`** (орбіта радіусом ≈386 м,
origin = legacy-центр): горизонтальне зміщення кадрів до 0.96 м (RMS 0.76 м), на
кутах ділянки 1×1 км — до 1.43 м; вертикальне — < 0.1 мм. Це різниця масштабів
сфери й еліпсоїда на 41° пн. ш. (≈0.1–0.3 %), а не шум. Окремо: `position` Earth Studio
лежить на **сталі 1.2 м** далі від центру Землі, ніж `R + altitude` (кутова розбіжність
≈1e-9 м) — сумісно з R≈6371010 м у Google; legacy-кадр через це був ще й піднятий на
1.2 м. Разом `legacy_shift` для `dataset_v6` — до 1.53 м.

**Яка модель відповідає рендерам — перевірено 2026-10-15 на `dataset_v6`: сфера, а не WGS84.**
Тест: той самий `database.db` (фічі й матчі від поз не залежать), `point_triangulator` із
**замороженими** відомими позами; між варіантами змінюються лише центри камер, орієнтації
однакові. Мірило — середня похибка репроєкції при фіксованих позах. Кількість точок тут
оманлива: гірші пози рвуть треки на коротші, тож точок більше, а спостережень менше.

| Відомі пози | DB legacy-run-а: точки / спостереження / трек / репроєкція | DB run-а `20261015-023347` |
|---|---|---|
| сфера (`position`, legacy) | 13 148 / 50 782 / 3.86 / **0.407 px** | 13 950 / 54 319 / 3.89 / **0.402 px** |
| WGS84 (поточний конвертер) | 13 298 / 50 640 / 3.81 / 0.571 px | 14 112 / 54 157 / 3.84 / 0.567 px |
| WGS84, `altitude` + 1.2 м | ідентично WGS84 до всіх знаків | ідентично WGS84 |

Інтерполяція центрів `wgs84 + k·(сфера − wgs84)`, k = −2, −1, 0, 0.5, 1, 2, 3 → 0.974, 0.779,
0.571, 0.472, **0.407**, 0.565, 0.772 px (DB legacy; на другому DB так само). Мінімум при
k ≈ 1.0 (вершина параболи — 1.01), криві симетричні. Це узгоджується з власною камерою
Earth Studio, яка точна саме на сфері: промені «вперед» сходяться в одну точку з нев'язкою
0.0000 м для сферичних центрів і 0.54 м (max 0.74) для WGS84. BA у пайплайні маскує різницю
(після BA обидва варіанти ≈0.40 px), але платить зсувом камер. `compare_poses` known→refined:
сфера (legacy run) — 0.49 м середнє / 0.78 м max / 0.024°, WGS84 (run `20261015-023347`) —
1.14 м / 2.55 м / 0.055°.

Висновок: кадри узгоджені зі сферичною моделлю. Найімовірніше, рендерер Earth Studio кладе
світ на сферу: кути WGS84 стають сферичними кутами, висота відкладається радіально. Іншого
узгодженого пояснення не знайдено. Перерахунок **поз камер** у WGS84 на вході робить їх
неузгодженими з кадрами на ≈1 м (+0.16 px середньої репроєкції при f = 3062 px, дальність
≈410 м), і після BA реконструкція не лежить точно ні в тій, ні в іншій системі. Правильний
порядок інший: реконструювати в **сферичному ENU** ділянки (з `coordinate`, той самий R для
кадрів і origin, без +1.2 м), а сфера→WGS84 застосувати до результату. На ділянці 1 км²
(±250 м по висоті) це відображення афінне з точністю 0.23 мм — `diag(1.002558, 0.998712, 1)`
на 41° пн. ш. (0.87 мм на 2×2 км). Тож гаусіани переносяться точно: μ → Aμ + b, Σ → AΣAᵀ.
**Зроблено ([поправка ADR-0006](decisions/0006-project-site-epoch-capture.md#поправка-2026-10-15-earth-studio-реконструюється-у-сферичному-enu), 2026-10-15).**
Конвертер пише пози у сферичному ENU, а стадія `georef` переводить хмару й сплати в WGS84
Site ENU відображенням зі звіту; `earth_model = "sphere-enu+affine"`. Для origin `dataset_v6`
`A = diag(1.0025578, 0.9987115, 1.0)` (поза діагоналлю < 1e-13), `b = (0, 0.10, −0.01) мм`;
залишок — 0.22 мм на коробці 1×1 км × ±250 м і 0.07 мм на кадрах. Run `20261015-031843`
(новий «latest» ділянки): `compare_poses` known→BA — **0.49 м** середнє / 0.79 м max / 0.031°
(legacy — 0.49 / 0.78 / 0.024°, WGS84-run `20261015-023347` — 1.14 / 2.55 / 0.055°);
sparse — 13 734 точки, 0.404 px після BA. `georef` зсуває хмару й сплати в середньому на
0.28 / 0.36 м (до 1.2 м на краях сцени); `scene-meta.metric` цього run-а —
`earth_model="sphere-enu+affine"`, `frame_error_m` = 0.22 мм. Якість на 4 test-кадрах —
26.46 dB / 0.891 SSIM. Run тренувався вже без дубля 30≈00 у train, тож кадр 00 (26.88 dB)
з попередніми run-ами не порівнюється. Без кадру 00 (08/16/24): 26.32 dB проти 26.16 у
WGS84-run-а — +0.16 dB, як і в контролі A/B вище. Run `20261015-023347` лишається в
каталозі з `earth_model="wgs84"`.

Висота лишається неперевіреною. Стала вертикальна зміна всіх камер — гіпотеза +1.2 м або
`altitude` над геоїдом EGM96 замість еліпсоїда (тут ≈ −32 м) — на ділянці ~0.4 км є жорстким
зсувом, тож тріангуляція її не бачить (числа ідентичні). Радіус сфери скорочується, якщо
кадри й origin рахувати з одним R; тому `position` (R + 1.2 м) для поз не використовувати.
Абсолютну висоту (еліпсоїд чи геоїд) можна перевірити лише зовнішнім еталоном: DEM або
контрольні точки. Ціль орбіти `dataset_v6` — h = 70.56 м у конвенції Earth Studio. До такої
перевірки у звіті лишається `altitude_reference` «unverified».

Якість (PSNR/SSIM на 4 test-кадрах): WGS84-run — 25.96 dB / 0.8645, legacy — 27.53 / 0.9053
(legacy тренувався старим тренером, до 2.2). Майже весь розрив дає test-кадр 00: 30.62 →
25.35 dB, на кадрах 08/16/24 — лише −0.2…−0.4 dB. У кадру 00 є майже ідентичний
train-кадр 30 (орбіта замкнута; середня різниця пікселів ≈1/255). Тому PSNR кадру 00 міряє, наскільки BA розвів пару поз
00/30 (0.093° ≈ 5 px проти 0.020° ≈ 1 px), а не якість сцени.
Контроль A/B: той самий DB і код, змінено лише відомі пози (сфера); dense → train → eval
прогнано поза каталогом. Результат — 27.10 dB / 0.8993; кадр 00 — 29.34, пара 00/30 розведена
на 0.057°. Без кадру 00 (середнє 08/16/24): сфера — 26.35 dB / 0.8919, WGS84 — 26.16 / 0.8839,
тобто +0.19 dB / +0.008 SSIM на користь сфери. Напрям той самий, але різниця нижча за поріг
шуму (4 кадри); на якість сцени модель Землі майже не впливає. Решта розриву в середньому PSNR —
витік дубльованого кадру в спліті.
Витік усунено (2026-10-15): спліт виключає кадр 30 з train (`find_test_twins`). Без витоку
(4 кадри, поточний тренер): WGS84 — 25.89 dB / 0.8703, сфера — 26.34 / 0.8891, legacy — 26.36 /
0.8883. Кадр 00 тепер звичайний novel view (сфера 26.58, WGS84 25.22), тож WGS84 на 0.45 dB
нижче, майже весь розрив — кадр 00; на 08/16/24 — 0.15 dB. Деталі — [BASELINE.md](BASELINE.md).

---

## 5. Контракти даних

Шляхи — відносно директорії run-а ділянки (у дужках — legacy `data/`).

| Артефакт | Виробник | Споживач | Формат |
|---|---|---|---|
| `images/*.jpeg` (`raw_images_<ds>/`) | geo / exif | sfm, dense, train | JPEG + GPS EXIF |
| `known_poses/{cameras,images,points3D}.txt` (`<ds>/sparse_known/`) | geo | sfm | COLMAP text |
| `known_poses/geo_registration.json` | geo | migrate, server (legacy) | JSON: `earth_model`, ENU-origin, `sphere_enu_to_site_enu`, lat/lon/alt кадрів + `enu_xyz` (сферичний) і `site_enu_xyz` (WGS84) |
| `reports/registration.json` (`sparse_known/registration_report.json`) | geo, migrate | georef (A, b), server (`frame_error_m`), людина | JSON, schema 2: `sphere_enu_to_site_enu` {A, b, коробка, залишки}, зсув сфера→WGS84, `legacy_shift`, перевірка експорту |
| `sfm/sparse/0/*.bin` (`colmap_output_<ds>/sparse/0/`) | sfm | dense, parse, train | COLMAP binary |
| `mvs/fused.ply` (`colmap_output_<ds>/dense/`) | dense | parse | PLY |
| `gs/dense.pt`, `gs/cloud.pt` (`<ds>_dense.pt`, `<ds>_cloud.pt`) | parse | train, georef; server — лише без геоприв'язки | `torch.save` dict (pickle) |
| `gs/splat.pt` (`<ds>_splat.pt`) | train | eval, georef; server — лише без геоприв'язки | `torch.save` dict (pickle) |
| `gs/{dense,cloud,splat}_wgs84.pt` (`<ds>_*_wgs84.pt`) | georef | server | той самий формат + ключ `frame` (WGS84 Site ENU, A, b, джерело) |
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
│   ├── gs/                        # trainer (активний), gsplat_backend, sh, evaluate, report
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
