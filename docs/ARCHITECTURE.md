# AeroSplat-GIS — поточна архітектура (v0.6.1, гілка `main`, стан на 2026-10-15: S1 + S2 + Фаза 2.2 gsplat)

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

* Єдина точка входу (`aerosplat run`). Стадії: `geo → exif → sfm → dense → parse → train → eval → georef`,
  далі опційні `semantics → occupancy` (§3.13, §3.17): вони виконуються, лише коли їх названо
  (`--stages`, `--only`, `--to`, `--from semantics`), напр. `aerosplat run --site <p>/<s> --resume --from semantics`.
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
    дилатація 0.3 px (+ компенсація непрозорості Mip-Splatting лише для
    `scene-meta.rasterize_mode = antialiased`), квад вирівняний по власних осях
    еліпса, cutoff 3.5σ;
  * dense — `THREE.Points` з розміром точки за оцінкою щільності;
  * NormalBlending, back-to-front, `depthWrite=false`.
* `depth-sort_worker.js` — counting sort по 65 536 бакетах глибини у Web
  Worker; сортування запускається лише при реальному русі камери.
* `aero-debug.js` — панель діагностики (F9): статистика реальних байтів,
  CPU-дзеркало вершинного шейдера, синтетична тест-сцена, **Ref compare** —
  той самий ракурс від render service (gsplat) і попіксельний PSNR в'ювер↔еталон.
* Статичні файли `/web` віддаються з `Cache-Control: no-cache` (ревалідація за ETag):
  імена без хешу, тож інакше браузер тримав старий шейдер після оновлення.
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

### 3.10 Черга задач і воркери — `aerosplat/jobs/` (S2, ADR-0010)

* **Таблиці в каталозі** (міграція `0002`): `jobs` (kind, resource `gpu|cpu`, status
  `queued → running → succeeded|failed|cancelled`, priority, site_key, payload, result,
  error, cancel_requested, worker_id, часи), `job_events` (append-only; `pk` — курсор
  стріму; типи `status`, `stage`, `log`, `result`), `workers` (heartbeat). SQLite у режимі
  **WAL** з `busy_timeout`, бо в один файл пишуть API, воркери й процеси задач.
* **`JobQueue`** (`jobs/queue.py`): `claim` — один умовний `UPDATE … WHERE status='queued'`,
  тож задачу не забирають двічі; для `gpu` в тому ж операторі перевірка «жодної `gpu`-задачі
  в `running`» → одночасно **одна** GPU-задача, скільки б GPU-воркерів не було. FIFO в межах
  пріоритету — за автоінкрементним `pk`. `request_cancel`: `queued` → одразу `cancelled`,
  `running` → прапорець для воркера. `recover_orphans`: `running`-задачі зупиненого або
  мовчазного (> 45 с) воркера → `failed`.
* **Воркер** (`jobs/worker.py`, `aerosplat worker [--resource gpu|cpu] [--once]`): забирає
  задачу і запускає **окремий процес** `python -m aerosplat.jobs.child` у власній групі
  процесів. Читає його stdout/stderr і пише пачками (0.5 с / 500 рядків) як `log`-події,
  щосекунди перевіряє скасування (тоді вбиває все дерево — `taskkill /T` на Windows,
  `killpg` на POSIX, тобто й COLMAP), шле heartbeat. Код виходу 0 → `succeeded`, інакше
  `failed` з останніми 20 рядками виводу в `error`. Якщо процес задачі не закрив свій run у
  каталозі (вбитий, OOM), воркер ставить run у `failed`. На старті прибирає «мертвих»
  воркерів цього хоста (pid не живий) і їхні задачі.
* **Процес задачі** (`jobs/child.py`): `pipeline` → `starter.main(spec.argv())` з колбеками
  `on_run(site, run_id)` (→ `result`) і `on_event` (→ `stage`-події від `run_stages`:
  `started|skipped|done|failed`, index/total, секунди); `diagnostic` — фіктивні кроки для
  перевірки воркера і стріму.
* **`PipelineSpec`** (`jobs/models.py`): site, epoch, capture, run_id, stages або
  from/to, resume, overrides (ключі `RunConfig`, повна валідація при постановці; `dataset`,
  `footage_dir`, `json_path` заборонені — їх визначає ділянка).

### 3.11 API v2 і веб-UI — `aerosplat/server/api_v2.py`, `web/app/` (S2)

* **API v2** (без torch): `projects`, `sites` (створення з origin, зведення, деталі),
  `captures` (multipart-завантаження через ті самі адаптери ingest, що й CLI; мініатюри —
  лише для каталогізованих імен), `runs` (`POST` → `gpu`-задача; деталі зі `manifest.json`,
  `metrics.json`, звіт), `jobs` (список, деталі, скасування, події з курсора,
  **WebSocket** `/jobs/{id}/stream` — опитує `job_events` кожні 0.4 с і закривається після
  термінального статусу), `system` (воркери, черга, `nvidia-smi`), `pipeline` (стадії, схема
  і дефолти `RunConfig`), `scenes/refresh` — перечитує каталог у процесі сервера (run-и
  завершують воркери в інших процесах) і скидає кеш лише змінених сцен. OpenAPI — `/docs`.
* **Веб-UI** (`web/app`, Vite + React + TypeScript, TanStack Query, MapLibre GL;
  збірка `npm run build` → `web/app/dist`, сервер віддає її з `/app/` з SPA-fallback,
  `/` перенаправляє туди або на класичний в'ювер `/web/`): огляд ділянок на карті,
  створення ділянки кліком (origin), сторінка ділянки (позиції камер на карті, захоплення
  з мініатюрами, run-и з деталями й метриками), завантаження захоплень з прогресом,
  діалог запуску run-а (стадії, параметри), монітор задачі (стадії, живий лог, скасування),
  список задач, 3D — класичний в'ювер у рамці до Фази 4.2. Підкладки — лише безключові:
  OSM (у темній темі — інверсія яскравості растру) і супутник Esri World Imagery.
* **`aerosplat up`** (`server/up.py`): сервер + GPU- і CPU-воркер + сервіс рендерингу (§3.12)
  однією командою; процеси зупиняються разом із сервером. API слухає лише `127.0.0.1`
  (автентифікації немає).

### 3.12 Сервіс рендерингу — `aerosplat/render/` (S2, ADR-0010 п. 2)

Довгоживучий GPU-процес «сцена + поза → зображення». Окремий від API (той без torch) і від
воркерів (задача — це завершений процес, а тут потрібна затримка в мілісекундах і сцена в VRAM).

* **Процес** (`render/service.py`, `aerosplat render-service`, порт 8001, лише localhost):
  `render/app.py` — FastAPI: `POST /scenes/{project}/{site}/render`, `GET /health`, `GET /scenes`,
  `POST /unload`. Відповідь — PNG, а камера, інтринсики, `view_matrix`, геоприв'язка й час —
  JSON у заголовку `X-Render-Meta`. Помилки: 404 немає навченої сцени, 422 неможлива поза
  (напр. WGS84 для сцени без georef), 503 мало VRAM.
* **Рушій** (`render/engine.py`): сцену бере з каталогу так само, як сервер (`discover_scenes`:
  `splat_wgs84.pt` останнього run-а, тобто **Site ENU**, метри, z вгору). Тримає її на GPU,
  перечитує, якщо файл змінився; одна сцена одночасно, вивантаження після простою
  (`--idle-seconds`, 600 с), щоб тренування отримало VRAM назад. Один рендер за раз (lock).
  Растеризація — `gsplat.rasterization` для будь-якого `.pt` (SH, якщо є, інакше `rgb`;
  `rasterize_mode` — із файлу), `camera_model` `pinhole` або `ortho`. gsplat тут обов'язковий:
  Inria-растеризатор не вміє ортографіку.
* **Запит** (`render/models.py`, torch-free — його перевіряє й API): позиція в Site ENU
  (`e,n,u`) або WGS84 (`lat,lon,h`), курс/нахил/крен (`yaw` — компас, 0 = північ; `pitch` −90 =
  вниз; `roll` + = за годинниковою) або `look_at`; `fov_deg` (вертикальний) для перспективи,
  `extent_m` (ширина кадру в метрах) для ортографіки; `output` = `rgb` | `alpha` | `depth`
  (16-бітний PNG: `метри = значення / 65535 · max_m`, 0 = немає даних). Ортографічній камері
  без висоти `u` рушій ставить її над сценою; для прямого виду вниз у мета є `ground_grid`
  (піксель → E, N), тобто готова геоприв'язка ортофото.
* **Камера** (`render/camera.py`): COLMAP-конвенція (x вправо, y вниз, z вперед), світ — ENU.
  Права вісь завжди горизонтальна, тож вид униз не вироджений: `yaw` тоді задає, куди в
  кадрі «вгору» (0 = північ угорі).
* **API v2** проксіює: `POST /api/v2/sites/{p}/{s}/render` (валідує тіло, шле в сервіс, 503 з
  підказкою, якщо сервіс не працює) і `GET /api/v2/render/status`. Адреса —
  `AEROSPLAT_RENDER_URL`.
* **Windows і gsplat.** gsplat тут збирається JIT, тож `up` запускає сервіс через
  `tools/gsplat_env.bat`, якщо немає зібраного `gsplat.csrc` (`render_service_command`). Той самий
  скрипт потрібен для `aerosplat render-service` вручну і для GPU-тестів.
* **UI** — сторінка `/app/render/<project>/<site>` («Вид із камери»): перспектива/ортографіка,
  поля пози, колір/глибина/прозорість, розмір, «вписати в сцену» (межі сцени з мета першого
  запиту), координати E/N під курсором для виду вниз, PNG; у сайдбарі — індикатор сервісу.
* **Тести** (`tests/test_render.py`): математика камери, схема запиту, HTTP-шар на підставному
  рушії, проксі зі стаб-сервером; GPU-тести (синтетична сцена): положення гаусіана на
  перспективному й ортографічному зображенні збігається з еталонною проєкцією, глибина,
  WGS84-позиція = ENU-позиція, вивантаження після простою.

### 3.13 Семантика: рельєф, будівлі, фасади — `aerosplat/semantics/` (S3, Фаза 7.1)

Стадія `semantics` (остання, після `georef`) перетворює щільну хмару на шар, до якого
кріпляться отвори: рельєф → будівлі → фасади → карта покриття. Усе в Site ENU, лише numpy,
без torch і GPU (torch потрібен тільки щоб прочитати `.pt`), тож іде на CPU-воркері.

* **Рельєф** (`terrain.py`): растр із кроком `semantics_res` (1 м); DSM — найвища точка
  клітини, DTM — прогресивний морфологічний фільтр (Zhang та ін.): поверхню послідовно
  «відкривають» вікном, що росте, і клітину опускають лише тоді, коли перепад більший за той,
  який пояснює власний ухил рельєфу. Будівлі зникають, щойно вікно ширше за них, пагорби —
  ні. nDSM = DSM − DTM. Також рахується **шорсткість** (|DSM − медіана 3×3|).
* **Будівлі** (`buildings.py`): висоти замало — у передмісті майже все, що вище 2.5 м, це
  дерева. Розрізняє **шорсткість**: дах локально плаский, крона ні (на `dataset_v6` медіана
  по будівлі 0.04 м проти 0.29 м по рослинності). Два проходи: спочатку кожен високий згусток
  оцінюється за **медіанною** шорсткістю, виміряною **до** будь-якої фільтрації (відфільтруєш
  клітини спершу — і кожен згусток виглядає пласким за побудовою), потім ті, що пройшли,
  підчищаються поклітинно, що відрізає дерево, яке тулиться до стіни. Поріг
  самокалібрується: шорсткість голої землі — це власний шум хмари, і межа береться в кілька
  разів вище. Контур обводиться по ребрах клітин (точний для растру), далі прибирання
  колінеарних точок і Дуглас-Пекер.
* **Фасади** (`facades.py`): вертикальна площина в плані — це пряма, тож пошук іде **у 2D**
  (вертикальність виходить точною за побудовою, а не обмеженням). RANSAC прямої по точках
  стіни однієї будівлі (лише біля краю контуру — інакше точки даху тягнуть лінії через
  середину), тоді уточнення TLS, і так далі по залишку. Дві поправки, без яких результат
  непридатний: **суміжність** (лінія вздовж вулиці чіпляє стіну сусіднього будинку — лишається
  тільки неперервний відрізок) і **злиття компланарних** (точки розсіяні на пів метра обабіч
  справжньої поверхні, тож RANSAC знімає стіну паралельними шарами за метр один від одного;
  злиття вимагає ще й **перекриття** вздовж стіни, інакше ряд будинків уздовж вулиці зшивається
  в один 200-метровий «фасад»). Назва — `<будівля>/<октант><номер>`, номер зліва направо, якщо
  стояти зовні обличчям до стіни, — та сама конвенція, що й для колонок в адресації
  ([ACCESS_POINTS §4](ACCESS_POINTS.md)).
* **Покриття** (`coverage.py`): для кожної клітини фасаду — скільки кадрів дивляться на неї
  під кутом < 60° до нормалі, з клітиною в кадрі (**видимість**, геометрія зйомки) і з
  достатньою роздільністю (**покриття**, `coverage_max_gsd`). Розділення цих двох речей —
  суть: воно відрізняє «не знімали» від «знімали, але задалеко, щоб прочитати». Пози COLMAP
  живуть у кадрі реконструкції, тож переносяться тим самим афінним відображенням
  (`transform_pose`; R·A⁻¹ ортогоналізується полярним розкладом, для сфера→WGS84 поправка
  < 0.11°). Оклюзію v1 **не** перевіряє (`occlusion_tested: false`): стіна за деревом
  рахується як видима.
* **Вихід** у `runs/<run>/semantics/`: `semantics.json` (заголовок растру, будівлі, фасади,
  параметри, попередження), `terrain.npz` (dsm, dtm, count, roughness, labels),
  `coverage.npz` (сітка кадрів на фасад).
* **Читач COLMAP без torch** — `aerosplat/io/colmap_model.py` (.bin і .txt).
* **Якість на `dataset_v6`.** Перевірка алгоритму — на синтетиці з відомою відповіддю
  (`tests/test_semantics.py`: коробка → рівно 4 стіни з точними розмірами, дерево відкинуте,
  нормалі назовні, конвенція u). На самому `dataset_v6` результат **грубий і для розмітки
  отворів непридатний**: крок точок 3 м, меш Google із заокругленими ребрами, і стадія сама
  попереджає, що кадри дають у кращому разі 10 см/px замість потрібних 5. Це рівно та межа,
  заради якої в [ACCESS_POINTS §12](ACCESS_POINTS.md) існує M0 — власна зйомка зблизька.

### 3.14 Ортофото фасаду — `semantics/ortho.py` (S3, Фаза 7.1)

Щоб розмічати отвори, фасад треба бачити **фронтально й метрично**: тоді вікна прямокутні,
вирівняні, одного масштабу, а піксель переводиться в `(u, v)` константою
([ACCESS_POINTS §5.3](ACCESS_POINTS.md)). Це рівно ортографічний рендер уздовж −нормалі.

* **Сервіс рендерингу лишається універсальним** — він нічого не знає про фасади. Позу будує
  API з збереженого `Facade` (`facade_ortho`), і далі це звичайний ортографічний запит
  (§3.12). Чиста арифметика, без torch і GPU.
* **Чому поза виходить чистою:** за `yaw = азимут + 180` і `pitch = 0` права вісь камери —
  це **точно** вісь `u` фасаду, а вісь «вниз» — світова −z. Тож колонки зображення йдуть
  уздовж `u`, рядки — вниз по стіні, і жодного залишкового обертання немає
  (`test_facade_ortho.py` перевіряє це для довільної нормалі).
* **Розмір пікселя точний:** спершу обирається ширина в пікселях, `extent_m` виводиться з
  неї, тож см/px — це замовлене число, а не його заокруглення. Кадр = межі `(u, v)` фасаду
  плюс поле; занадто дрібний крок для одного зображення → 422 з підказкою, яка величина
  влізе.
* **`facade_grid`** у `X-Render-Meta` — відповідник `ground_grid` для стіни: афінне
  піксель → `(u, v)` плюс рамка фасаду в ENU, тож піксель ортофото однозначно лягає в
  метричну точку на стіні й назад.
* **Камера стоїть перед стіною** (`standoff_m`, 25 м): для ортографіки відстань не змінює
  зображення, але все позаду камери відсікається. Що стоїть між камерою і стіною — затуляє,
  і це правда, яку оператор має бачити.

### 3.15 Таксономія і авторські дані — `aerosplat/taxonomy/`, `aerosplat/authored/` (S3, M0)

* **Таксономія — дані, не код** ([ADR-0007](decisions/0007-access-point-taxonomy.md)):
  версіоновані YAML-пакети в `taxonomy/` (поки один, `core`). Ділянка вмикає потрібні
  (`Site.taxonomy_packs`). Завантажувач перевіряє те, чого не може перевірити окремий файл:
  що `parent` існує, що атрибути класу десь оголошені, що `requires` виконано, що два пакети
  не визначають один id, що в ланцюжку батьків немає циклу. Кожна Feature зберігає рядок
  версії (`core@1.0`, §2.5). Кеш у пам'яті стежить за mtime і розміром файлів: правка пакета
  діє без перезапуску сервера.
* **Авторські дані живуть поза run-ами** ([ADR-0006](decisions/0006-project-site-epoch-capture.md)):
  `sites/<site>/authored/access_points.json` (поточний стан, атомарний перезапис через
  тимчасовий файл) і `authored/audit.jsonl` (**тільки дозапис**: хто, коли, що змінив і яким
  воно було). Run-директорію можна видалити й перерахувати — розмітка лишається.
* **Запис точки доступу** (§3.1): id — ULID (`ap_…`, сортується за часом створення),
  геометрія — полігон у `(u, v)` фасаду, з якого виводяться ENU-центр, нормаль і метричний
  розмір. Feature несе **не лише id фасаду, а й сам кадр** (`host`: origin, normal, u_axis,
  v_ground): новий run може зсунути чи перейменувати фасад, і збережена площина — це те, з
  чим звірятиметься переякорювання (§5.9). Намальоване людиною одразу `verified` (§3.4);
  `rejected` **не видаляється** — це негативний приклад для навчання.
* **Адресація** (`semantics/addressing.py`, §4, §5.6): ряди й колонки — кластеризація `v`- і
  `u`-центрів намальованих отворів; `floor_index` рахується від `v_ground`. Зберігаються
  **обидві** нумерації (`column` — колонка сітки, `ordinal_on_floor` — n-й **наявний** отвір),
  бо людина каже «третє вікно», маючи на увазі другу. Модуль **нічого не вигадує і нічого не
  зсуває**: якщо ряду біля землі не намальовано або інтервали між поверхами нерегулярні,
  ставиться `floor_ambiguous`, і §6.3 блокує підтвердження цілі без явної перевірки людиною.
  Помилка поверху — найнебезпечніша помилка системи.
* **API v2** (`server/api_semantics.py`): `GET …/taxonomy`, `GET …/semantics`,
  `GET …/facades[/{id}]` (з сіткою покриття), `POST …/facades/{id}/ortho`,
  `POST …/facades/{id}/address`, `GET/POST/PATCH/DELETE …/access-points`, `GET …/audit`.
  Id фасаду містить `/`, тож маршрути з `{facade_id:path}` **жадібні** — конкретніші
  оголошені першими, і це закріплено тестом.

### 3.16 Facade editor — `web/app` (S3, M0)

Сторінки `/app/facades/<p>/<s>` (список фасадів з покриттям) і `/app/facade/<p>/<s>/<id>`
(редактор). Оператор тягне рамку по ортофото, обирає клас, зберігає; клавіатурний рев'ю за
[§10.4](ACCESS_POINTS.md): `1…9` клас, `↵` зберегти, `←`/`→` наступна точка, `V`/`X`
підтвердити/відхилити, `Del` видалити, `C` покриття, `A` адресація, `?` довідка.

**Карта покриття накладається поверх зображення** — не як число в звіті, а як червона заливка
там, де жоден кадр не роздільний. Без неї оператор розмічав би вікна, яких немає: на непокритій
ділянці 3DGS домальовує фасад із нічого. Фасад, який видно, але не різкіше за потрібне, отримує
окреме попередження вгорі сторінки. Поруч із намальованою рамкою показується її метричний
розмір проти `size_prior_m` класу і найгірше покриття під нею.

### 3.17 Зайнятість і ESDF — `aerosplat/surface/` (Фаза 8.1a, трек C4)

Стадія `occupancy` (опційна, після `semantics`) будує воксельну сітку в Site ENU зі станами
**вільно / зайнято / невідомо** і поле відстаней до найближчого зайнятого **або невідомого**
вокселя. Це вхід для перевірки коридорів (`aerosplat.mission`, 8.1b), оракулів колізій і
далекомірів `rng_1…6` у SITL ([SIMULATION_AND_DELIVERY §4.4, §4.6](SIMULATION_AND_DELIVERY.md)).
Невідоме блокує кожен запит так само, як зайняте
(внутрішні вимоги надійності §3 п. 3). Лише numpy/scipy, CPU.

* **Ґратка** (`models.py`): воксель `occupancy_res` (0.25–0.5 м) має **ділити** клітину рельєфу,
  тож кожна клітина `terrain.npz` — рівно k×k колонок, а масиви `(rows, cols, layers)` ідуть у
  тій самій конвенції рядків і стовпців, що й растр. По висоті сітка йде від вокселя під
  найнижчим ґрунтом до `headroom` (10 м) над «лінією дахів» — квантилем 0.9999 виміряних висот.
* **Модель v1 — 2.5D** (`occupancy.py`): у колонці вільне лише те, що **вище** її поверхні.
  Нижче все заблоковано: воксель із точками → `occupied`, ґрунт під виміряною колонкою →
  `occupied`, решта → `unknown`. Під мостом, балконом чи кроною вільного простору немає — це
  безпечно. Звідки взялася висота поверхні, записано для кожної колонки (`ColumnSource`):
  * `points` / `cell` — **виміряно**: найвища точка колонки або, якщо колонка порожня, DSM її клітини;
  * `filled` — дірка в хмарі в межах `fill_radius` (5 м): найвища поверхня навколо плюс
    `assumed_margin` (1 м); вільне над нею — **припущення**;
  * `skyline` — у радіусі даних немає: невідомо до лінії дахів плюс запас, вище —
    **припущення** вільного простору. З `occupancy_unknown=sky` такі колонки заблоковані на всю висоту;
  * `pillar` — поверхня вища за стелю сітки (шум або справді висока споруда): заблоковано до неба.

  Поверхню ще й **закривають** (grey closing, `close_radius` 1 м): ями, вужчі за ~2 м,
  піднімаються до краю, а стіни й уступи лишаються на місці. Без цього на розрідженій хмарі одна
  шумова точка під дахом відкриває «шахту» вільного простору в будівлю — тест це відтворює.
* **Поза сіткою** правила фіксовані й однакові для всіх запитів: збоку — невідомо (`outside`),
  знизу — ґрунт, зверху колонка продовжує стан свого верхнього шару.
* **ESDF** (`esdf.py`) — відстань від центру вокселя до центру найближчого заблокованого.
  Вона зберігається в **цілих сантиметрах, округлених униз** (`uint16`), і обрізана на
  `max_distance` (20 м). Обрізання робить розбиття на тайли **точним**: scipy EDT тримає
  ~50 байт на воксель, і Site цілком у пам'ять не влазить, але тайл з ореолом завширшки з
  обрізання дає те саме, що й цілий об'єм (тест порівнює значення один в один). Знакова версія
  (`--signed`, `int16`) рахується вдвічі довше. Споживача для неї поки немає: колізія в SIL —
  провал тесту, а не «відскок» ([SIMULATION_AND_DELIVERY §4.3](SIMULATION_AND_DELIVERY.md)).
* **Запити** повертають нижню межу — ніколи не більше за правду:
  * `clearance(p)` — відстань до найближчого заблокованого вокселя як **суцільного куба**.
    Біля перешкод (поле ≤ 8 вокселів) вона точна: перебір кубів у вікні. Далі це ліпшицева межа
    по 8 сусідніх центрах мінус ½√3·r (занижує до ~2 вокселів). Відносно справжньої поверхні
    похибка — до вокселя по осі (√3·r на ребрі), і завжди в бік меншого кліренсу.
  * `path_clearance(polyline)` — крок не більший за r/2 (інакше помилка). Мінімум мінус пів
    кроку — межа для **всього** шляху, а не лише для вибірок. Місце входу в заблоковане і що
    там (`occupied` / `unknown` / `outside`) визначаються точно, променем уздовж кожного відрізка.
    `assumed_free_s` — перша точка у вільному просторі, який лише **припущено** (`filled` /
    `skyline`). Для 8.1b це `feasible_with_risks`, а не доказ.
  * `raycast(o, d, max_range)` — точний обхід вокселів (Amanatides–Woo); відкритий простір
    перетинається стрибками, безпечність яких доводить ESDF.
* **Вихід** у `runs/<run>/occupancy/`: `occupancy.npz` (`state`, `esdf_cm`, `column_top`,
  `column_source`) і `occupancy.json` (заголовок сітки, межі, конвенція ESDF, статистика за
  станами й джерелами колонок, параметри, попередження, час). JSON пишеться останнім — це ознака
  завершеного запису. Планувальник читає продукт через `surface/products.py` (`find_occupancy`,
  як `semantics/products.py`).
* **Неправильні входи дають гучну відмову**, а не тихий результат: NaN/inf у хмарі чи запитах,
  порожня хмара, DTM з NaN, воксель, що не ділить клітину, сітка більша за `max_voxels`
  (перевіряється до виділення пам'яті), `bbox` поза растром, крок шляху > r/2, нульовий напрям
  променя.
* **`dataset_v6`.** Run-директорій ділянки локально немає, тож прогін ішов на legacy-хмарі
  `data/dataset_v6_dense.pt` (1.09 млн точок):

  | Прогін | Сітка | Вокселів | Час | Пік пам'яті | Колонки |
  |---|---|---|---|---|---|
  | уся площа 891×858 м, 0.5 м | 1716×1782×96 | 294 млн | 71 с | 2.5 ГБ | виміряно 24%, `skyline` 47% |
  | ядро 300×300 м, 0.25 м | 1200×1200×168 | 242 млн | 134 с | 2.5 ГБ | виміряно 77%, `filled` 17%, `skyline` 6% |

  Отже, на Earth Studio карта значною мірою складається з **припущень**, і запити про це
  повідомляють, а не приховують: транзит над лінією дахів проходить, але з `assumed_free_s`.
* **Межі v1:** модель 2.5D (немає вільного простору під нависаннями); дротів і тонких перешкод
  немає в самій хмарі. «Вільне» тут означає «вище поверхні», а не «бачили камерою». Наступний
  крок — промені від камер (space carving): вони зроблять `skyline` доказом, а не припущенням.

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
viewer sceneRoot            = Rx(−90°) · (100 / extent)   → Y-up, радіус 100
                              (Rx(+90°) лише для вільного кадру COLMAP без metric)
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
| `occupancy/occupancy.{npz,json}` | occupancy | mission (8.1b), SIL | npz (`state`, `esdf_cm`, колонки) + JSON `occupancy-v1` (§3.17) |
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
│   ├── jobs/                      # models, queue, worker, child, proctree (черга задач, S2)
│   ├── render/                    # models, camera, engine, app, service (сервіс рендерингу, S2)
│   ├── semantics/                 # terrain, buildings, facades, coverage (семантика, S3)
│   ├── surface/                   # occupancy, esdf, stage, products (зайнятість + ESDF, 8.1a)
│   └── server/                    # main.py (FastAPI, сцени v1), api_v2.py, up.py (`aerosplat up`)
├── web/                           # класичний в'ювер: index.html, appGLSL.js, depth-sort_worker.js, aero-debug.js
│   └── app/                       # веб-UI (Vite + React + TS); dist/ — збірка (ignored)
├── configs/                       # earthstudio.yaml, drone.yaml (пресети RunConfig)
├── tests/                         # CPU-тести + golden-фікстури Earth Studio
├── tools/                         # дослідницькі скрипти, gsplat_env.bat, migrate_dataset.py
├── legacy/                        # optimization/, reconstruction/ (v2, TRIPS), scripts/, server/mainG.py, checker/
├── docs/                          # ARCHITECTURE, TARGET_ARCHITECTURE, ACCESS_POINTS, ROADMAP, decisions/
├── diff-gaussian-rasterization/   # сторонній репо (ignored, non-commercial)
├── data/                          # legacy-датасети (ignored; у git лише data/README.md)
├── projects/                      # дерево проєктів + catalog.db (ignored, ADR-0006)
└── pyproject.toml, requirements.txt, package.json (eslint для web/)
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
