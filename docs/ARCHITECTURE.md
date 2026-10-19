# AeroSplat-GIS — поточна архітектура (v0.6.1, гілка `main`, стан на 2026-10-15: S1 + S2 + Фаза 2.2 gsplat; 2026-10-19: маски, глибина MVS, checkpoint/resume)

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

Додаткова supervision і довгі прогони (2026-10-19, `gs/supervision.py`, `gs/checkpoint.py`).
Усе вимкнено за замовчуванням; з вимкненими опціями цикл тренування той самий:

* **Маски** (`masks: true`, `--masks DIR`; будь-який растерізатор). Конвенція COLMAP
  (`ImageReader.mask_path`): маска кадру `<ім'я>` — `<ім'я>.png` (або `<stem>.png`) у
  каталозі масок, 0 = піксель ігнорується. У конвеєрі каталог — `masks/` поруч із кадрами
  захоплення (`<capture>/masks/`, legacy `data/<ds>/masks/`) або `masks_dir`. Маска
  ресемплиться консервативно (піксель валідний, лише якщо весь його відбиток валідний).
  У loss замасковані пікселі рендера замінюються на GT (`masked_photometric_loss`): нуль
  похибки й нуль градієнта, тож вони не впливають і на densify (екранні градієнти);
  L1 і SSIM усереднюються лише по валідних пікселях. Кадр без файлу маски — увесь валідний;
  каталог без жодної відповідної маски — помилка. Якщо тренування йде на undistorted
  `dense/images` (без known poses), маски з оригіналів не беруться — потрібен явний
  `masks_dir`. Авто-маски неба немає навмисно (див. docstring `supervision.py`): хук — будь-який
  сегментатор, що пише ці PNG. Стадія `eval` маски поки не враховує.
* **Регуляризація глибиною MVS** (`depth_reg: mvs`, `--depth-mvs <dense>`; лише gsplat).
  Цілі — `dense/stereo/depth_maps/<кадр>.geometric.bin` стадії `dense`; для кожного кадру
  перевіряється, що поза й інтринсики карти (модель `dense/sparse`) збігаються з камерою
  тренування, а камери тренування — без дисторсії; інакше помилка. Рендер — gsplat
  `RGB+ED` (очікувана глибина); loss — середня відносна похибка |d̂ − d|/d по пікселях із
  ціллю, непрозорістю рендера > 0.5 і поза маскою. Глибини MVS — у одиницях тієї самої
  реконструкції, що й пози, тож масштаб і зсув не підбираються (на відміну від монокулярних
  пріорів); у known-poses Earth Studio це метри. Вага `depth_weight` згасає лог-лінійно до
  `depth_weight_final` (за замовчуванням 1.0 → 0.01, як у Inria).
* **Checkpoint / resume** (`checkpoint_every`, `checkpoint_keep`, `train_resume`; лише gsplat).
  `gs/splat_checkpoints/ckpt_<ітерація>.pt` (legacy: `<ds>_splat_checkpoints/`): параметри,
  стани Adam, стан densify (MCMC/default), exposure, порядок камер семплера, RNG CPU і CUDA,
  історія loss і «підпис» запуску (опції, кадри, маски/глибина, стартова хмара). Запис —
  у `.tmp` + `os.replace`, тримаються останні N; при читанні пошкоджений найновіший
  пропускається. Resume з іншим підписом — помилка з переліком відмінностей; новий запуск без
  resume видаляє старі чекпойнти, а успішно завершений — свої (результат — `.pt`; на
  останній ітерації чекпойнт не пишеться). Продовження відтворює прямий прогін з тією ж точністю, з якою
  два прямі прогони відтворюють один одного (атомарні додавання CUDA в backward gsplat не
  детерміновані). Чекпойнт важкий: параметри + 2 моменти Adam ≈ 3× розміру `.pt`.
  Ці ключі не входять у відбиток стадії `train`.
* **TensorBoard** (`tensorboard: true` → `gs/splat_tensorboard/`, `--tensorboard DIR`; лише gsplat):
  loss, PSNR, кількість гаусіанів, lr, похибка глибини, test PSNR. Пакет `tensorboard` —
  опційний extra; без нього — зрозуміла помилка на старті.

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
  ├── captures/<capture>/{capture.yaml, images/, images_private/, images.jsonl, source/, masks/?}
  ├── epochs/<epoch>/{epoch.yaml, runs/<run_id>/}
  │     runs/<run_id>/{manifest.json, run.log, images/, known_poses/, sfm/, mvs/, gs/, reports/}
  ├── authored/      # авторське, пайплайн сюди не пише (access_points, targets, audit)
  └── missions/<id>/ # mission.json + експорт (waypoints, plan, parm), §3.18
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
  `started|skipped|done|failed`, index/total, секунди); `ingest` — завантажене захоплення
  (нижче); `diagnostic` — фіктивні кроки для перевірки воркера і стріму.
* **Ingest завантажень на CPU-воркері (2026-10-19)** (`ingest/upload.py`, задача `ingest`,
  `IngestSpec`). Запит API лише стрімить файли в `captures/.upload-<job id>/files/` (на тому ж
  диску, що й захоплення), робить дешеві перевірки (імена, є зображення; для Earth Studio —
  один JSON і кадри 1:1 з ним), **запечатує** теку (`upload.json`, атомарно) і ставить задачу
  (`202` + `Job`; id задачі генерується до печатки, бо воркер може забрати її одразу).
  Задача: `check` → `hash` (sha256 і розміри кадрів, `stage`-події з `progress: {done,
  total}` раз на секунду + рядки журналу) → `register`. Дерево захоплення будується **всередині
  staging** (`capture/`, адаптери розділено на `prepare_*` — розкласти й описати — і
  `import_*` = prepare + реєстрація), потім одним `os.rename` стає `captures/<id>/`, потім
  одна транзакція `add_capture`. Захоплення існує лише після коміту, тож напівготове ніколи
  не видно. Після вбивства в будь-якій точці (скасування, збій) повторний запуск тієї ж задачі
  доводить справу: напівзібране дерево повертається у `files/`; перейменоване, але не
  зареєстроване, реєструється з маркера `.ingest.json` (у ньому опис кадрів); готове —
  no-op (маркер лишається як `{"job", "registered": true}`). Теку `captures/<id>/`, яку
  створило не це завантаження, задача не чіпає — відмова з поясненням. Staging завершених
  задач прибирається при наступних завантаженнях на ділянку через годину, «осиротілий» (API
  впав посеред завантаження) — через 6 год без змін. Без CPU-воркера задача просто чекає в
  черзі; UI про це попереджає.
* **`PipelineSpec`** (`jobs/models.py`): site, epoch, capture, run_id, stages або
  from/to, resume, overrides (ключі `RunConfig`, повна валідація при постановці; `dataset`,
  `footage_dir`, `json_path` заборонені — їх визначає ділянка).
* **gsplat у GPU-воркері (2026-10-19).** Без зібраного `gsplat.csrc` gsplat компілює
  CUDA-розширення при першому виклику, а на Windows це працює лише в середовищі
  `tools/gsplat_env.bat` (VS 2022 + CUDA тієї версії, з якою зібрано torch). `aerosplat up`
  тепер запускає через нього не лише сервіс рендерингу, а й GPU-воркер (`up.worker_command`,
  `with_gsplat_env`), і процеси задач успадковують це середовище. CPU-воркер не обгортається.
  Крім того, конвеєр перевіряє gsplat **у preflight, до першої стадії**
  (`runner.check_rasterizer` → `gsplat_backend.gsplat_problems`: gsplat встановлено, CUDA є,
  а без `csrc` — ninja, cl.exe не з VS 18, toolkit = версія CUDA torch). Раніше задача падала
  на початку тренування, тобто вже після SfM і dense. Тепер вона одразу завершується з
  поясненням і підказкою (`gsplat_env.bat` або `rasterizer=auto`). Перевіряються лише стадії,
  що справді виконуватимуться і рендерять (`train`, `eval`; `stages_to_run` повторює правила
  `--resume`), тож відновлення з уже натренованим сплатом не відмовляється. `--dry-run` лише
  попереджає. Причина падіння стадії тепер є й у `run.log`, а не лише в маніфесті.

### 3.11 API v2 і веб-UI — `aerosplat/server/api_v2.py`, `web/app/` (S2)

* **API v2** (без torch): `projects`, `sites` (створення з origin, зведення, деталі),
  `captures` (multipart-завантаження → `ingest`-задача на CPU-воркері, §3.10; ті самі адаптери,
  що й CLI; мініатюри — лише для каталогізованих імен), `runs` (`POST` → `gpu`-задача; деталі зі `manifest.json`,
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
  однією командою; процеси зупиняються разом із сервером. За замовчуванням слухає `127.0.0.1`.
  `--host` з не-loopback адресою (напр. `0.0.0.0`) **не стартує без жодного токена** (нижче);
  `--require-auth` вимагає токен навіть з цієї машини. Сервіс рендерингу автентифікації не має
  і не-loopback адресу відхиляє.
* **Автентифікація (2026-10-19)** — `aerosplat/auth/` (токени, контекст), `server/auth.py`
  (middleware, `/api/v2/auth/*`). Один чистий ASGI-middleware перед **усім** застосунком —
  REST, WebSocket, сцени v1, статика (`/web`, `/app`) — а не залежність на кожному маршруті.
  * **Loopback відкритий без налаштувань**, але лише коли все разом: TCP-peer (`scope.client`,
    із сокета, не з заголовка) — loopback; **немає** проксі-заголовків (`X-Forwarded-For`,
    `Forwarded`, `X-Real-IP`, `CF-Connecting-IP`, `Via`, …) — проксі чи тунель на цій машині
    робить віддалений трафік «локальним»; `Host` — loopback-ім'я (DNS rebinding); `Origin`,
    якщо є, — локальний (інший сайт у браузері власника не керує API, хоч CORS і `*`).
    Заголовки можуть лише **забрати** довіру, ніколи не дати.
  * Інакше — **токен**: `Authorization: Bearer` (скрипти) або cookie `aerosplat_token`
    (HttpOnly, SameSite=Strict, 30 діб), яку ставить `POST /api/v2/auth/login`. Cookie — бо
    браузер не додає заголовок до `<img>` (мініатюри), iframe в'ювера і WebSocket, а токен в
    URL потрапив би в журнали й історію. Для cookie небезпечні методи й WebSocket мають бути
    same-origin (`Origin` = `Host`, не `Sec-Fetch-Site: cross-site`) — CSRF і cross-site
    WebSocket. Відмова: HTTP 401 JSON (`auth: token-required`, `WWW-Authenticate: Bearer`),
    WebSocket закривається до accept (403); 403 — cookie з чужого сайту.
  * **Публічне без токена**: оболонка UI і статика (`/`, `/app/…`, `/web/…` — код, не дані) та
    `/api/v2/auth/*` (`whoami`, `login`, `logout`), щоб UI міг попросити токен. Шлях із `..`
    публічним не вважається.
  * **Токени**: `aerosplat token create|list|revoke`; `<projects>/auth/tokens.json` — лише
    SHA-256 від 256-бітних випадкових секретів (`asg_…`), id, ім'я, хто й коли створив,
    відкликання. Порівняння `hmac.compare_digest` з усіма активними без раннього виходу. Сервер
    файл лише читає (перечитує при зміні mtime/розміру — відкликання діє з наступного запиту);
    пише CLI атомарно під файловим замком. Пошкоджений файл = жоден токен не дійсний (віддалені
    запити відхиляються, loopback працює) і `up --host` не стартує. Не таблиця каталогу: нова
    ревізія Alembic зробила б `catalog.db` нечитабельним для старших гілок, які власник запускає
    на тих самих `projects/`.
  * **Аудит**: middleware кладе принципала в `contextvar`; `authored.models.actor()` для запиту
    з токеном повертає `token:<ім'я>` і ігнорує заявлене клієнтом `by`. Без токена — як раніше
    (`by` або користувач ОС). Так підписуються точки доступу, цілі, місії, експорт.
  * **UI**: будь-яка відповідь 401 (`ApiError`) піднімає подію, `AuthGate` показує діалог
    токена, після входу перезапитує все; у сайдбарі — «токен <ім'я> · вийти». З цієї машини
    діалог не з'являється ніколи.
  * Токени логам не віддаються (uvicorn пише лише шлях; CLI друкує секрет один раз при
    створенні).
  * **TLS (2026-10-19, `server/tls.py`).** Не-loopback bind без TLS не стартує:
    `--tls-cert/--tls-key` (перевіряється, що ключ від сертифіката і він не прострочений) або
    `--tls-self-signed` (EC P-256 на 365 днів у `<projects>/auth/tls/`, SAN — імена й адреси
    машини, перевипуск за 30 днів до кінця; SHA-256 відбиток друкується при старті). Явне
    `--insecure-http` — лише за reverse proxy з TLS (він додає `X-Forwarded-For`, тож токен
    потрібен і через нього). Cookie сесії `Secure` на HTTPS. Loopback — без змін, HTTP.
* **E2E-тест UI** (`web/app/e2e`, Playwright, `npm run e2e` — збирає UI; `npm run e2e:only` —
  без збірки; у `npm test` не входить). `tools/e2e_server.py` піднімає `aerosplat up` (CPU-воркер,
  без GPU-воркера й рендера; `AEROSPLAT_RENDER_URL` — на закритий порт, щоб не зачепити рендер
  власника на :8001) на тимчасовому каталозі `%TEMP%/aerosplat-e2e/<port>` із синтетичними
  фікстурами: ділянка з фасадами (`demo/pier`) і місія з прольотом (`demo/street`). Сценарії:
  список ділянок; створення ділянки кліком на карті → завантаження кадрів → `ingest`-задача на
  CPU-воркері до «готово» → захоплення на ділянці; список задач; Facade editor; сторінка
  прольоту; другий сервер з `--require-auth` — діалог токена, хибний і правильний токен,
  HttpOnly-cookie, вихід. Без Chromium (`npx playwright install chromium`) пропускається.

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
  (піксель → E, N), тобто готова геоприв'язка ортофото. Необов'язкове `run` закріплює рендер
  за сплатом саме цього run-а (через каталог; невідомий run чи run без сплату — 404, без
  підміни найновішим); `run` у мета — що відрендерено насправді (§3.14).
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
  **Звіт стадії** (`coverage_report`, 2026-10-19) рахує фасади, а не бере медіану по всій
  ділянці: «з потрібною деталізацією N із M (з них повністю ≥95% клітин K); видно лише грубіше
  за поріг; не видно». Далі йде рядок для кожної з 5 будівель, найближчих до початку Site
  (центру зйомки): покрито фасадів / усього, медіана частки, найкраща GSD **саме цієї
  будівлі**. Попередження «задалеко» наводить GSD грубих фасадів, а не найкращу по ділянці.
  На dataset_1910 старий рядок казав «0%, найкраще 9 см/px (потрібно 16)», хоча вежа B08 мала
  12/12 фасадів зі 100% при 13 см/px. Новий звіт: «104 із 1833 фасадів (6%) … B08 (0 м від
  початку, h 44 м): 12/12 фасадів, медіана 100%, найкраще 13 см/px»; 1098 грубих фасадів мають
  найкраще 16.1 см/px, медіана 27.
* **Вихід** у `runs/<run>/semantics/`: `semantics.json` (заголовок растру, будівлі, фасади,
  параметри, попередження), `terrain.npz` (dsm, dtm, count, roughness, labels),
  `coverage.npz` (сітка кадрів на фасад). `terrain.npz` читають і pad-зони (§3.22), тож вони
  працюють на кожному run-і, що вже пройшов стадію, без перерахунку.
* **Id будівель і фасадів — у межах run-а.** Новий run нумерує все заново (вежа
  `dataset_1910`: `B08` у run-і 1, `B04` у run-і 2; `B04/SW1` run-а 1 — інша будівля). Тому
  все, що пов'язує авторські дані з фасадами, іде через кадр і геометрію, а не через id (§3.21).
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
* **Сцена — того самого run-а, що й фасад** (2026-10-19). Раніше ортофото й gridfit
  рендерились із найновішого сплату навіть із `?run=` (сесія 2026-10-18…19, помилка № 20):
  кадр фасаду run-а 1 поверх реконструкції run-а 2. Тепер `RenderRequest.run` закріплює сервіс
  рендерингу за run-ом (`store.scenes.pinned_scene` / `run_scene_source` через каталог, окремий
  ключ кешу `<site>@<run>`), API завжди передає run семантики і звіряє `run` у мета відповіді
  (502, якщо сервіс старіший за API й відрендерив інше). Run без власного сплату — 404, а не
  підміна. На копії `dataset_1910`: `B08/SW3?run=…101955` рендериться з run-а 1 (1.8 с), той
  самий кадр зі сплату run-а 2 відрізняється в середньому на 13.6/255 на піксель.

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
* **Свої й чужі точки фасаду.** Точка *рідна* фасаду, якщо несе рівно його кадр
  (`authored.carry.is_native`: той самий id **і** origin/normal/u_axis/v_ground). Лічильники
  `features_by_facade`, список `access-points?facade=…&run_id=…` і адресація беруть лише рідні:
  однойменний фасад іншого run-а більше не показує й не адресує чужі вікна. Точки інших run-ів
  на тій самій стіні — `…/carried` (§3.21).
* **Прив'язка до сітки фасаду** (§5.6 п. 3, `addressing.snap_to_grid`, `POST
  …/facades/{id}/regularize`): ряди й колонки — ті самі кластери, що в адресації, окремо для
  кожного класу (двері не задають ліній вікнам); лінія — медіана відповідного краю ≥ 2 членів.
  Край зсувається на лінію, лише якщо ближче за `tol_m` (0.15 м), тож **жоден край не рухається
  далі допуску**; дальші повертаються в `kept` і лишаються, як намальовано. Лише рідні,
  не відхилені, прямокутні по осях отвори; `dry_run` показує, що зміниться. Кожен зсув — запис в
  історії точки й рядок аудиту `op: regularize` з дельтами по краях (`replace_many(whats=…)`).
  Підтверджена ціль на зсунутій точці отримає `target_changed` (відбиток покриває геометрію).
* **API v2** (`server/api_semantics.py`): `GET …/taxonomy`, `GET …/semantics` (з переліком
  run-ів, `?run_id=`), `GET …/facades[/{id}]` (з сіткою покриття), `POST …/facades/{id}/ortho`,
  `POST …/facades/{id}/address`, `POST …/facades/{id}/regularize`,
  `POST …/facades/{id}/candidates/gridfit` (§3.20), `GET …/facades/{id}/carried`,
  `GET …/facade-matches` (§3.21), `POST …/pads/candidates` (§3.22), `GET …/export/cityjson`
  (§3.23), `GET/POST/PATCH/DELETE …/access-points`, `POST …/access-points/{id}/reanchor`,
  `GET …/audit`.
  Id фасаду містить `/`, тож маршрути з `{facade_id:path}` **жадібні** — конкретніші
  оголошені першими, і це закріплено тестом.

### 3.16 Facade editor — `web/app` (S3, M0)

Сторінки `/app/facades/<p>/<s>` (список фасадів з покриттям) і `/app/facade/<p>/<s>/<id>`
(редактор). Оператор тягне рамку по ортофото, обирає клас, зберігає; клавіатурний рев'ю за
[§10.4](ACCESS_POINTS.md): `1…9` клас, `↵` зберегти, `←`/`→` наступна точка, `V`/`X`
підтвердити/відхилити, `Del` видалити, `C` покриття, `A` адресація, `G` знайти вікна за
сіткою (§3.20), `R` вирівняти за сіткою (спершу показ, ще раз `R` — застосувати), `?` довідка.

* **Вибір run-а** (2026-10-19): у списку фасадів — перемикач run-ів із семантикою (`?run=`),
  посилання в редактор **завжди** несуть `?run=`, бо id означає цю стіну лише в цьому run-і.
* **Точки інших run-ів** на цій стіні — пунктиром; клік показує, звідки вона і з яким зсувом
  площини, і кнопку «Перенести на цей фасад» (лише для `placed`; для `ambiguous` — причина).
* **Pad-зони** — на сторінці списку фасадів: кнопка в кожної будівлі (пробний пошук → «Записати
  N кандидатів») і таблиця pad-точок ділянки з «Підтвердити»/«Відхилити». Карти зони немає:
  підтверджувати — лише після огляду на «Виді з камери» (підказка на кнопці).

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
(внутрішні вимоги надійності §3 п. 3). Лише numpy/scipy, CPU; torch
пакет не імпортує (тест перевіряє це в окремому процесі).

Сітка будується у два кроки: спершу 2.5D-модель (вільне — над поверхнею колонки), потім
**вирізання простору променями камер** (space carving, `carving.py`): невідомий воксель стає
вільним, лише якщо його перетнули промені від камер до точок, які ці камери бачили. Формат
файлів — `occupancy-v2`.

* **Ґратка** (`models.py`): воксель `occupancy_res` (0.25–0.5 м) має **ділити** клітину рельєфу,
  тож кожна клітина `terrain.npz` — рівно k×k колонок, а масиви `(rows, cols, layers)` ідуть у
  тій самій конвенції рядків і стовпців, що й растр. По висоті сітка йде від вокселя під
  найнижчим ґрунтом до `headroom` (10 м) над «лінією дахів» — квантилем 0.9999 виміряних висот.
* **Межі сітки** (`extent.py`, `occupancy_extent`, 2026-10-19). Поза сіткою все `outside`,
  тобто заблоковано, тож межі вирішують, які місії взагалі можна спланувати. Тому вони —
  явний вибір, а не наслідок того, куди сягнула хмара. Знімок Earth Studio однієї будівлі
  реконструює ~1.6 × 1.8 км: на dataset_1910 це 2.6 млрд вокселів, і стадія падала через 10–53 с.
  За замовчуванням (`auto`) порядок такий: `occupancy_bbox` (Site ENU, м), якщо задано;
  інакше **bbox ділянки** (`site.yaml`, WGS84 → охоплювальний прямокутник у Site ENU, ребра
  дискретизуються, бо в ENU вони трохи вигнуті); інакше **квадрат ±`occupancy_radius`**
  (150 м) навколо початку Site — центру зйомки. `radius` бере квадрат, навіть коли в
  ділянки є bbox; `raster` — увесь растр рельєфу (старий режим). Явний bbox разом із
  `radius`/`raster` — помилка конфігурації. Вибір пишеться в журнал («Межі сітки: …») і в
  `occupancy.json` (`extent`: джерело, запит, растр, сітка, сторони, де хмара закінчилась
  раніше). Сітку **ніколи не обрізають мовчки** під `occupancy_max_voxels`: вибір, що не
  вміщується, валить стадію з підказкою, які ключі змінити. Це перевіряється **до** читання
  видимості COLMAP: промені передаються функцією, яку викликають лише після перевірки бюджету.
  Також є попередження, коли хмара закінчується раніше за межі і коли межа сітки ріже будівлі
  (за растром міток семантики, а не за вершинами контуру). bbox ділянки входить у відбиток
  стадії (`Stage.layout_values`), тож новий bbox перебудовує сітку при `--resume`, а відбитки
  інших стадій не змінюються. Місія, що виходить за сітку, відмовляється (`outside_grid`, §3.18).
  Перевірка на dataset_1910 за замовчуванням, без `--set` (bbox у ділянки немає → ±150 м):
  прогін 20261019-101955 — 600×600×233 = 84 млн вокселів, 273 с з вирізанням; 20261019-105253 —
  600×600×123 = 44 млн, 87 с. Обидва раніше падали (2.6 і 1.2 млрд). Межу сітки ріжуть
  12 і 20 будівель, про це є попередження.
* **Крок 1 — 2.5D** (`occupancy.py`): у колонці вільне лише те, що **вище** її поверхні
  (з шару `free_from`, він зберігається для кожної колонки). Нижче все заблоковано: воксель із
  точками → `occupied`, ґрунт під виміряною колонкою → `occupied`, решта → `unknown`. «Ґрунт» —
  це нижче **і** DTM, **і** найнижчої точки колонки (для колонки `cell` — її клітини рельєфу):
  DTM часто йде поверх настилу мосту чи навісу, і тоді повітря під ним ставало б «матерією»,
  яку жоден промінь не відкриє. Між найнижчою точкою і DTM колонка `unknown` — так само
  заблокована, як і раніше, але вже не «доведено зайнята». Звідки взялася висота поверхні,
  записано для кожної колонки (`ColumnSource`):
  * `points` / `cell` — **виміряно**: найвища точка колонки або, якщо колонка порожня, DSM її клітини;
  * `filled` — дірка в хмарі в межах `fill_radius` (5 м): найвища поверхня навколо плюс
    `assumed_margin` (1 м); вільне над нею — **припущення**;
  * `skyline` — у радіусі даних немає: невідомо до лінії дахів плюс запас, вище —
    **припущення** вільного простору. З `occupancy_unknown=sky` такі колонки заблоковані на всю висоту;
  * `pillar` — поверхня вища за стелю сітки (шум або справді висока споруда): заблоковано до неба;
  * `carved` — була `filled` / `skyline` / `sky`, але промені вирізали її **верхній заблокований
    воксель** (крок 2): вільне над ним — доказ, а не припущення.

  Поверхню ще й **закривають** (grey closing, `close_radius` 1 м): ями, вужчі за ~2 м,
  піднімаються до краю, а стіни й уступи лишаються на місці. Без цього на розрідженій хмарі одна
  шумова точка під дахом відкриває «шахту» вільного простору в будівлю — тест це відтворює.
* **Крок 2 — вирізання променями** (`carving.py`). Точку `fused.ply` бачили камери, чиї карти
  глибини її утворили, тож відрізок від кожної з них до точки не перетнув нічого непрозорого.
  Воксель, який перетнув такий відрізок, — доказ вільного простору, і єдиний, що може відкрити
  заблоковане 2.5D-моделлю: повітря під навісом, мостом, балконом і колонки, де хмари немає.
  * **Звідки видимість.** У `.pt`-хмарах її немає. COLMAP `stereo_fusion` пише поруч із
    `fused.ply` файл `fused.ply.vis`: для кожної точки — індекси кадрів, з яких її злито. Індекс —
    це позиція кадру в моделі робочої теки `dense/sparse` (а не `image_id`): у сучасному COLMAP —
    за зростанням `image_id`, у старих — у порядку файлу. Пробуються обидва порядки; прийнятий
    має покласти в кадр (з допуском 2 px) щонайменше 99.5% пар і бути **єдиним** таким, інакше
    видимості немає (`VisibilityError`, стадія падає). Перевірено на `dataset_v6` картами
    глибини: правильний порядок — 99.99% пар у кадрі, 99.2% узгоджені з глибиною (1%), а порядок
    за назвою (тут хибний: `image_id` 8 — кадр 09) — однаково 96% / 92%. Тобто «у кадрі» розрізняє
    порядки слабко, звідси суворий поріг і відмова при неоднозначності. Пари поза кадром
    відкидаються й рахуються. Розріджені треки (`points3D`) — запасний варіант, не потрібний:
    сітку будують саме зі злитих точок.
  * **Ніколи — «камера бачить, бо точка в її кадрі».** Камера, що кадрує точку, може дивитися на
    стіну перед нею. Тест: будівля без точок на стінах і така «видимість» — промені прорізають її
    наскрізь; з видимістю злиття всередині не відкрито нічого.
  * **Кадр.** Точки й камери COLMAP переводяться в Site ENU тим самим афінним перетворенням
    (`sphere_enu_to_site_enu` зі звіту реєстрації), що й камери семантики; нормалі — оберненою
    транспонованою. Хмара, з якої будується сітка, має бути підмножиною злитих точок у цьому
    кадрі: медіана відстані до найближчої злитої точки ≤ 1 см, інакше — відмова (не той прогін
    чи не та реєстрація), а не косо вирізаний простір.
  * **Правила**, кожне з тестом, що без нього падає:
    1. відрізок закінчується у вокселі, де є точка хмари сітки (викиди, які прибрав `parse`,
       і точки поза `bbox` не вирізають);
    2. останні `occupancy_carve_margin` (1 м) відрізка не вирізаються — шум глибини вздовж променя;
    3. відрізок, що перетнув **зайнятий** воксель раніше, ніж за цей запас, відкидається
       **цілком**: або його точка, або зайнятий воксель хибні, а хибна глибина нічого не доводить.
       Виняток — власна поверхня точки: пологий промінь до точки на землі метрами йде в шарі
       вокселів ґрунту. Тому зайняті вокселі в межах діагоналі вокселя від дотичної площини точки
       (довжина √3·r/|d·n|, не більше 10 вокселів, n — нормаль із `fused.ply`) не рахуються. На
       `dataset_v6` це зменшило відкинуті промені з 17% до 6%; без нормалей лишається лише запас;
    4. воксель мають перетнути промені від `occupancy_carve_min_views` (2) **різних** камер;
    5. **ерозія на один воксель:** перетнутий воксель, у якого серед 26 сусідів є простір, що
       лишився заблокованим, повертається. Відрізок доводить вільною лише свою лінію, а матерія
       поруч (стіна, дах, ґрунт) може заходити у воксель, який він перетнув. Тонкі сліди — шумова
       точка під дахом, поодинокий пологий промінь — зникають разом із цим шаром (тест: без
       ерозії шумова точка відкриває шахту в будівлю). Поза сіткою — ті самі правила: збоку
       невідомо, знизу ґрунт, зверху продовжується верхній шар;
    6. вирізаються лише `unknown`-вокселі: `occupied` (точки, ґрунт) — ніколи, колонки `pillar` —
       ніколи (над сіткою в них точка, і вільний простір туди не дістає).
  * **Вибірка.** Воксель «перетнуто», якщо в нього потрапила вибірка відрізка; вибірки йдуть
    через r/2, тож воксель, кут якого відрізок лише зачепив, може й не врахуватися — завжди в бік
    меншого вирізання. Понад 80% кожного променя — вільне повітря над виміряною поверхнею, де
    нічого вирізати; його пропускає ієрархічна перевірка (інтервали 8 м, далі 1 м): інтервал
    відкидається, лише якщо його найнижча вибірка вища за найвищий `free_from` у вікні, що
    покриває інтервал. Пропуск **точний** — тест порівнює з повним перебором вибірок на
    випадкових сітках. Обчислення — блоками (сегмент × інтервал), float32/int32: робота
    впирається в пропускну здатність пам'яті, і блоки розміром із кеш дають ~2×.
  * **Походження вільного** тепер по вокселях: вільний воксель **нижче** `free_from` колонки може
    бути лише вирізаним — це спостереження; **вище** — як скаже `ColumnSource` колонки.
    Колонка `filled` / `skyline` / `sky`, у якої вирізано верхній заблокований воксель, стає
    `carved`: жодна матерія, що стоїть на землі, не сягає вище за вирізаний воксель (те саме
    2.5D-припущення, на якому тримається «вільне над виміряною поверхнею»), тож повітря над ним —
    доказ. Для `sky` це верхній шар сітки, і тоді колонка відкривається до неба; без такого
    променя `sky` лишається заблокованою на всю висоту. Невирізані вокселі нижче лишаються
    `unknown`. `grid.assumed_at(p)` і `PathClearance.assumed_free_s` спираються саме на це.
  * **Під виміряною поверхнею** (навіси, мости; `occupancy_carve_under`, за замовчуванням увімк.)
    вирізання відкриває те, що v1 тримав закритим. Відомий ризик — **дзеркальні фасади**: MVS
    зливає узгоджені «віртуальні» точки за склом (відображення вулиці), і промені до них
    прорізають будівлю, дах якої виміряно. v1 тут був у безпеці (усе під дахом заблоковано), а
    без даху вразливий так само (віртуальні точки на рівні ґрунту стають поверхнею колонки).
    Правила 3–5 звужують, але не закривають цей випадок. Earth Studio відображень не рендерить,
    тож `dataset_v6` не зачеплено; для реальних зйомок міст зі склом — рішення власника, чи
    залишати `occupancy_carve_under` увімкненим.
  * **Вхідні файли** — робоча тека `dense` COLMAP: `--fused dense/fused.ply` (поруч
    `fused.ply.vis`), `--sparse dense/sparse`, `--registration` для відомих поз Earth Studio
    (пайплайн передає їх сам; `occupancy_carve=false` вимикає крок). Якщо файлів немає — сітка
    лишається 2.5D з попередженням у звіті; якщо файли є, але суперечать одне одному чи хмарі —
    стадія падає. Попередження й тоді, коли понад 25% променів перетинають зайняте перед своєю
    точкою: видимість або пози камер суперечать хмарі.
* **Поза сіткою** правила фіксовані й однакові для всіх запитів: збоку — невідомо (`outside`),
  знизу — ґрунт, зверху колонка продовжує стан свого верхнього шару.
* **ESDF** (`esdf.py`) — відстань від центру вокселя до центру найближчого заблокованого.
  Вона зберігається в **цілих сантиметрах, округлених униз** (`uint16`), і обрізана на
  `max_distance` (20 м). Обрізання робить розбиття на тайли **точним**: scipy EDT тримає
  ~50 байт на воксель, і Site цілком у пам'ять не влазить, але тайл з ореолом завширшки з
  обрізання дає те саме, що й цілий об'єм (тест порівнює значення один в один). Поле читає сам
  масив станів (`FREE == 0`, ненульове — заблоковане) без булевої копії на всю сітку: це
  звільнило місце для променів у тому ж бюджеті пам'яті. Знакова версія
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
    `assumed_free_s` — перша точка у вільному просторі, який лише **припущено**: над 2.5D-поверхнею
    колонки `filled` / `skyline` (`assumed_at`). Вирізаний простір і колонки `carved` — доказ.
    Для 8.1b `assumed_free_s` — це `feasible_with_risks`, а не доказ.
  * `raycast(o, d, max_range)` — точний обхід вокселів (Amanatides–Woo); відкритий простір
    перетинається стрибками, безпечність яких доводить ESDF.
* **Вихід** у `runs/<run>/occupancy/`: `occupancy.npz` (`state`, `esdf_cm`, `column_top`,
  `column_source`, `free_from`) і `occupancy.json` (заголовок сітки, межі, конвенція ESDF,
  статистика за станами й джерелами колонок, `stats.carving` — промені використані / без опори /
  відкинуті, вирізані й повернуті ерозією вокселі, скільки колонок стало `carved`, порядок кадрів;
  параметри, попередження, час). JSON пишеться останнім — це ознака
  завершеного запису. Планувальник читає продукт через `surface/products.py` (`find_occupancy`,
  як `semantics/products.py`).
* **Неправильні входи дають гучну відмову**, а не тихий результат: NaN/inf у хмарі чи запитах,
  порожня хмара, DTM з NaN, воксель, що не ділить клітину, сітка більша за `max_voxels`
  (перевіряється до виділення пам'яті), `bbox` поза растром, крок шляху > r/2, нульовий напрям
  променя; для вирізання — обрізаний чи зайвий `fused.ply.vis`, індекс кадру поза моделлю,
  неоднозначний порядок кадрів, понад 0.5% пар поза кадром, хмара не з цих злитих точок.
* **`dataset_v6`.** Run-директорій ділянки локально немає, тож прогін ішов на legacy-хмарі
  `data/dataset_v6_dense.pt` (1.09 млн точок; це рівно `fused.ply` робочої теки
  `data/colmap_output_dataset_v6/dense`, у тому ж кадрі) з видимістю з тієї ж теки
  (10.1 млн пар точка–камера, 31 кадр; орбіта ~190 м над ґрунтом, промені під 11–34°):

  | Прогін | Сітка | Вокселів | Час | Пік пам'яті | Колонки з припущенням |
  |---|---|---|---|---|---|
  | уся площа 891×858 м, 0.5 м, лише 2.5D | 1716×1782×96 | 294 млн | 68 с | 2.36 ГБ | 76.3% (`filled` 29%, `skyline` 47%) |
  | те саме з вирізанням | | | 109 с (+39 с) | 2.53 ГБ | 67.9% (`filled` 24%, `skyline` 44%) |
  | ядро 300×300 м, 0.25 м, лише 2.5D | 1200×1200×168 | 242 млн | 117–129 с | 2.35 ГБ | 23.2% (`filled` 17%, `skyline` 6%) |
  | те саме з вирізанням | | | 131 с (вирізання 21 с) | 2.52 ГБ | 17.1% (`filled` 12%, `skyline` 5%) |

  Час — збірка в одному процесі без читання файлів (видимість читається ще ~3 с); між прогонами
  він гуляє на ±10 с, тож для ядра вказано і власний час кроку вирізання. На ядрі 2 млн променів
  закінчуються поза кропом і не вирізають; відкрито 3.64 млн вокселів, з них 1.06 млн під
  виміряною поверхнею (щільна частина міста, навіси й дерева), 87 тис. колонок стали `carved`.
  Пік пам'яті з вирізанням — як у v1 до цієї зміни (2.5 ГБ): поле відстаней більше не тримає
  булеву копію сітки, і промені вміщуються у звільнене місце. На всій площі
  вирізання використало 9.64 млн променів (відкинуто як такі, що перетинають зайняте, 4.5%),
  відкрило 5.52 млн вокселів (з них 0.29 млн — під виміряною поверхнею) і перевело в `carved`
  256 тис. колонок; ерозія повернула 7.8 млн із 13.4 млн перетнутих вокселів. Зразків оглянуто
  314 млн замість ~3.7 млрд при суцільному кроці r/2. Отже, на Earth Studio карта й далі значною
  мірою складається з **припущень** — переважно на периферії, куди промені не дійшли, — і запити
  про це повідомляють, а не приховують: транзит над лінією дахів там проходить з `assumed_free_s`.
* **Межі.** Дротів і тонких перешкод немає в самій хмарі, і промені їх теж не «бачать»: предмет,
  менший за воксель і не торкнутий жодною точкою, може стояти між променями. Доказ є лише там,
  де пройшли промені: на периферії ділянки (куди дивилося мало камер) і під широкими навісами
  вирізано мало. Над сіткою окремого доказу немає: стан і походження продовжують верхній шар.
  Дзеркальні фасади — див. «Під виміряною поверхнею» вище.

### 3.18 Місії v0 — `aerosplat/mission/` (Фаза 8.1b)

Підтверджена ціль → геометрія підльоту → перевірка коридору по зайнятості (§3.17) → місія для
ArduPilot. Це вхід для SITL-треку (SIM-1). Лише numpy, CPU.

* **Ціль** (`targets.py`, [ACCESS_POINTS §6](ACCESS_POINTS.md), [ADR-0009](decisions/0009-human-in-the-loop-planning-grade.md)).
  `MissionTarget` створюється **лише** з `verified` AP і лише названим оператором. Ціль
  зберігається в `authored/targets.json` (авторські дані), а підтвердження дописується в
  `authored/audit.jsonl` разом із ціллю в тому вигляді, в якому її підтвердили. Підтвердження
  **блокується** з усіма причинами одразу: `not_verified`, `stale`, `not_redetected` (новий
  прапорець AP), `floor_ambiguous`, `sigma_unknown`, `sigma_too_large`, `mode_not_allowed`,
  `no_threshold`, `override_relaxes_taxonomy`. Перевизначення чисел класу дозволене **лише
  в бік обережності**: більший standoff і кліренс, вужчий конус. Ціль несе відбиток AP; якщо AP
  змінили після підтвердження, планування й експорт відмовляють (`target_changed`).
* **σ.** `σ_AP` береться з `uncertainty` AP (нове поле за §3.1), а якщо його немає — зі слів
  оператора, і це так і записується. `σ_georef` поки задає оператор, бо жодна стадія його не
  вимірює. **Невідома σ — відмова, а не нуль.** Поріг σ цілі для способу передачі —
  `handover_modes.<mode>.max_sigma_m` у таксономії (`core` 1.1, плейсхолдери).
* **Геометрія** (`planner.py`, §7.2). Для отвору: `H = c + a·standoff + up·Δh`,
  `E = c + a·D_entry + up·Δh`, курс `−a`. Для майданчика: `H = c + up·standoff`, вхід згори
  з висоти транзиту. `a = n`; якщо коридор заблоковано, перебираються напрямки в конусі
  (половина `approach_cone_deg` на бік, 3 нахили × 8 азимутів) і береться найчистіший.
  Напрямки, чиї E чи H надто близько до краю сітки, не пробуються (див. `outside_grid` нижче).
  `D_entry` і `Δh` — нові поля `handover` у таксономії.
* **Коридор.** Шлях: дім → вертикально вгору → транзит на висоті `z_t` → над E → E → H (зависання)
  → назад тим самим шляхом → посадка вдома. Кожна ділянка має свій радіус
  `r = r_дрона + m + k·σ`: біля цілі `σ_total² = σ_AP² + σ_georef² + σ_GNSS²` і `m` класу,
  на транзиті `σ² = σ_georef² + σ_GNSS²`. Транзит прокладається над найвищою зайнятою
  колонкою в смузі `r` уздовж шляху. Перевірка дає `feasible` / `feasible_with_risks` /
  `infeasible` з причинами: `unknown_space`, `obstacle_in_corridor`, `overhang_above`,
  `standoff_too_small`, `transit_too_high`, `home_unknown`, `home_blocked`,
  `home_not_on_surface` (блокуючі), а також `assumed_free_air`, `wire_risk` і ризики RTL
  (ризики). **Поверхня, на якій ділянка закінчується за задумом** (стіна цілі для підльоту,
  дах для майданчика, ґрунт під домом), в `r`-перевірці не рахується: відстань до неї
  тримає бортове наведення. Рахується все перед її площиною далі ніж
  `surface_relief_m + √3/2·воксель`. Межа сітки — це невідоме.
* **Місія за межами сітки — відмова `outside_grid`** (2026-10-19). Раніше дім чи E біля краю
  сітки давали `unknown_space`, ніби там перешкода, і план виходив `infeasible`. Але за краєм
  нічого не перевірено, і рятує тут не інший маршрут, а більша сітка. Тепер `MissionRefused`
  (API: 409) з координатами, бракованим запасом і готовою підказкою
  `--set occupancy_bbox=…`: прямокутник, що вміщує сітку й місію, округлений до 10 м.
  Перевіряються дім і ціль (до геометрії), потім E і H напрямків конуса. Кожна точка — з
  радіусом свого коридору плюс 2 вокселі (стільки ще дивиться пошук висоти транзиту); для E
  і H це `max(r_транзиту, r_підльоту)`. Сітка прямокутна, ділянки прямі, тож кінців ділянок
  достатньо. E і H кожного напрямку конуса відомі ще до перевірки коридорів, тому
  **пошук конуса пробує лише напрямки, чиї E і H вміщуються в сітку**
  (`directions_inside_grid`, 2026-10-19). Раніше він брав найчистіший коридор, а E і H
  перевіряв уже після вибору. Біля краю E найчистішого часто не вміщувався, і місію
  відмовляли, хоча інший напрямок давав перевірений план. Вулиця тестів, вхід 50 м: найчистіший
  E (−8.7, −34.2) за 5.8 м від краю при потрібних 6.6, а нахил 15° тримає 6.7 — тепер план
  `feasible`. Так само діагональні вікна біля західного краю. Скільки напрямків не пробували,
  каже `approach.candidates_outside_grid`. Що буває біля краю:
  * **Жоден напрямок не вміщується** — відмова `outside_grid` одразу, до перевірки
    коридорів. Вона називає прямий підліт і додає, що решта напрямків конуса теж не
    вміщуються (вулиця тестів, вхід 45 м, r 13.6 м: 71 с перевірки широких коридорів за
    краєм → миттєво).
  * **Деякі вміщуються, але жоден із них не чистий** — план `infeasible`, не відмова. Він
    показує перешкоди першого з тих, що вміщуються: прямого підльоту, якщо вміщується він,
    інакше найменш нахиленого. Примітка до знахідок підльоту каже, скільки напрямків не
    пробували через край сітки. Цей напрямок перевірено, і перешкода на ньому справжня;
    відмова `outside_grid` для напрямку, якого ніхто не перевіряв, сказала б неправду.
  * **Обраний напрямок** проходить усі перевірки ділянок, як і раніше. Його E і H ще раз
    перевіряє `require_inside_grid` (E майданчика ставиться на висоту транзиту вдруге).
* **Дім = origin ділянки** (або задана точка), висота — **виміряна поверхня** під ним.
  Від нього рахуються всі відносні висоти, тому дім «у повітрі» чи в будівлі — відмова.
* **Пошук місця зльоту** (`takeoff.py`, 2026-10-19). Origin — центр зйомки, тобто зазвичай біля
  цілі або на ній (dataset_1910: поруч із дахом вежі). Місце вручну шукали кількома скриптами.
  `find_takeoff_spots(grid, ground, target, sigma_georef_m, profile)` пропонує місця, ранжовані
  за відстанню до точки, де закінчується транзит планувальника (`transit_point(target, ap, grid,
  profile)`, див. нижче). Місце придатне, лише якщо:
  1. це **ґрунт**: виміряна колонка (`points`/`cell`), поза контурами будівель (мітки семантики
     того самого прогону), не далі `ground_tol_m` (1 м) від DTM — не дах, не настил, не фура;
  2. воно **відкрите в радіусі зльоту**: ближче за `r = r_дрона + m + k·hypot(σ_georef, σ_GNSS)`
     немає жодного заблокованого вокселя (зайнятого, невідомого чи поза сіткою), вищого за
     поверхню + рельєф. Це радіус і правило площини дому самого планувальника. Перевіряється
     для всіх колонок одразу за 2D-верхівками (EDT на кожен поріг висоти);
  3. **вертикальний стовп вільний** за точно тими самими перевірками, що й у плані:
     `check_home`, ділянка зльоту до висоти транзиту й сам транзит (`check_leg`). Тому
     запропонований дім ніколи не провалює перевірок дому й зльоту в плані (тест проганяє
     планувальник з кожним кандидатом).

  Кандидати стоять щонайменше `max(2r, 10 м)` один від одного, щоб список називав різні місця,
  а не сусідні вокселі однієї парковки. Точних перевірок — не більше `max_checks`. Якщо місця
  немає, `TakeoffSearch.message` пояснює чому: скільки колонок не пройшло яку перевірку,
  найвідкритіший ґрунт, який бачив пошук, і `σ_GNSS`, з яким той пройшов би (тест перевіряє,
  що з цим σ місце справді знаходиться). Пошук **ніколи сам не послаблює** критерій.
  Інтерфейси: `POST …/targets/{id}/takeoff` (профіль, `max_distance_m`, `n`, `max_checks`
  1…400; порожній список з поясненням — це відповідь, а не помилка), `MissionCreate.home_search`
  (дім = найближчий кандидат, `home.source` каже звідки; немає місця → 409 `no_takeoff_spot`),
  `MissionCreate.home_from_search` (разом з `home_enu`: `{rank, max_distance_m, n, ground_tol_m,
  max_checks}`. Сервер повторює пошук з тими самими run і профілем. Якщо під цим номером уже
  інше місце (далі 1 см) або місця немає, відповідь — 409 `takeoff_spot_changed`; інакше
  `home.source` = «takeoff search: #k (the operator's choice) of n spots within D m, …»), CLI
  `python -m aerosplat.mission.takeoff --occupancy … --target x,y,z --sigma-georef …`.
  Якщо бюджет точних перевірок вичерпано, відповідь — «не підтверджено, перевірок замало»,
  а не «місця немає»; найвідкритіший ґрунт при цьому не називається.
  * **Кінець транзиту — той самий E, що в плані** (2026-10-19). Раніше `transit_point` завжди
    давав E прямого підльоту, а планувальник після `directions_inside_grid` біля краю сітки
    нахиляє підліт. Вулиця тестів, вхід 50 м: прямий E (0, −35) за 5 м від краю, а потрібно
    6.56. Транзит кожного кандидата туди заходив за край, тож кожна точна перевірка падала з
    `transit_blocked`, і пошук закінчувався «не підтверджено … raise max_checks» навіть із 400
    перевірками. Планувальник із дому (9.75, 7.75) при цьому дає `feasible` з нахилом 15°.
    Для отвору вибір напрямку **не залежить від дому**: E і H висять на отворі, а коридор
    E → H, яким ранжуються напрямки, перевіряється з `r_підльоту` до площини цілі. Тому пошук
    один раз запускає власний пошук конуса планувальника (`planner.opening_approach`, спільний
    з `plan_mission` код `choose_direction`) і перевіряє транзит кожного кандидата до його E.
    Тепер там 5 місць, і план з кожного `feasible` з E (−12.94, −33.30). Так само, коли прямий
    підліт заблоковано (стовп перед вікном), транзит іде до нахиленого E. Як E опинився там,
    каже `TakeoffSearch.approach_note`; це ж речення додано до `message`.
  * **Ціль або весь конус за краєм** — відмова `outside_grid` планувальника дослівно (з
    `occupancy_bbox=`), ще до перевірки колонок: жоден дім тут не допоможе, лише більша
    сітка. `POST …/targets/{id}/takeoff` відповідає 409, `home_search` / `home_from_search` —
    тим самим 409, що й план. Тому профіль із `σ_GNSS` 15 м на вулиці тестів (`r_підльоту`
    32 м, а ціль за 25 м від краю) тепер дає відмову для цілі. Раніше він давав «місця немає,
    підійшло б із σ_GNSS ≤ …», хоча план з будь-якого дому відмовили б.
  * **Жоден напрямок, що вміщується, не чистий** — пошук іде до E того напрямку, який покаже
    план (прямого або найменш нахиленого). У `approach_note` сказано, що план до цієї цілі
    буде `infeasible` з будь-якого дому: перешкода на підльоті, а не біля дому.
  * **Майданчик**: E над ним на висоті транзиту, а нахил конуса залежить від висоти транзиту
    з кожного дому. Пошук його не моделює й перевіряє транзит до самого майданчика. Якщо E і H
    прямого підльоту не вміщуються в сітку (буває лише з профілем, де `transit_margin_m`
    більший за кліренс класу), то це `outside_grid` одразу, з поясненням, а не «не
    підтверджено» після вичерпаних перевірок. План із заданим `home_enu` при цьому ще може
    знайти нахил.
  * **Ціль біля краю сітки, задана напряму** (`--target`, 4faad07). Якщо точка, де закінчується
    транзит, ближче до краю сітки, ніж `r + 2 вокселі`, пошук одразу відповідає «місця немає»
    з причиною `target_near_grid_edge` для всіх колонок, 0 точних перевірок, і тією самою
    підказкою `occupancy_bbox=` (спільна `rebuild_hint`). Для цілей через `transit_point` ця
    перевірка ніколи не спрацьовує: там уже пройдено більшу вимогу планувальника до E
    (`max(r_tr, r_app) + 2 вокселі`). Під час злиття з 1b20c66 (2026-10-19) лишено обидва:
    конус — головний шлях, ця перевірка — запобіжник для сирої точки.
* **dataset_1910 (2026-10-19)**, вікно B08/SW3 (σ_georef 0.5 м), сітка за замовчуванням:
  * σ_GNSS 0.1 м (`r` 3.52 м): найближче місце — за 2 м від E. Дім, обраний вручну
    (−32.2, −50.2), і далі дає `feasible_with_risks` (лише `wire_risk`), як і місія, що пройшла SITL;
  * σ_GNSS 1.5 м (`r` 5.66 м): місця **є**. Найближче — (−5.2, −37.8), за 11 м від E і 23 м від
    вікна, з ≈6.2 м відкритого ґрунту. Планувальник із цим домом дає `feasible_with_risks`
    (лише `wire_risk`), кліренс зльоту 6.08 м. Повний перебір вокселів підтверджує: жодного
    заблокованого вокселя вище поверхні + рельєф ближче 5.79 м. Отже, висновок ручного пошуку
    «у радіусі 115 м немає» за правилами самого планувальника не підтверджується. Непридатним
    при σ 1.5 є саме дім, обраний вручну: заблоковане за 4.74 м від нього.
* **Експорт** (`export.py`): `missions/<id>/mission.json` (повний план), а також
  `mission.waypoints` (QGC WPL 110), `mission.plan` (QGC: місія + полігон геофенсу +
  rally) і `mission.parm` (FENCE_*, RTL_*_M, WP_SPD*, FS_*, BATT_FS_*, AVOID_*; імена,
  одиниці й значення — для Copter-4.7.2, перевіряються тестом за таблицею параметрів самої
  зафіксованої збірки, SIM-1). **Висоти лише відносно дому**: AMSL
  відмовляється, бо абсолютна висота Earth Studio не перевірена, а моделі геоїда немає.
  Рядок дому у WPL і `plannedHomePosition` містять еліпсоїдну висоту з позначкою «не AMSL».
  Кожен файл перед видачею читається назад і має повернутися в Site ENU **в межах 1 см**;
  з квантуванням `MISSION_ITEM_INT` (1e-7°) похибка лишається < 1 см. Змішаний кадр висот
  (абсолютна висота у відносному пункті, пункт в іншому `frame`, `AltitudeMode` ≠ 1)
  ловиться тут, а не в польоті. Геофенс — опукла оболонка кіл радіуса `r + запас` навколо
  вершин шляху. Тест перевіряє, що весь коридор лежить усередині.
* **Ворота експорту:** `infeasible` не експортується; `feasible_with_risks` — лише з
  `accept_risks` (у журнал пишеться, хто прийняв які ризики); відкликана ціль або змінена AP
  зупиняють експорт навіть уже порахованої місії.
* **API v2** (`server/api_mission.py`): `POST/GET …/targets`, `POST …/targets/{id}/confirm`,
  `POST/GET …/missions`, `POST …/missions/{id}/export/{waypoints|plan|parm}`. Відмови
  повертаються як 409 з `detail.reasons`.
* **AP-майданчики.** `AccessPoint` тепер буває або на фасаді (`host` + `geometry`), або
  майданчиком (`pad`: контур у Site ENU, висота, центроїд площі). Старі файли читаються без
  змін. UI для малювання майданчиків ще немає (залишок 7.1).
* **`dataset_v6` (2026-10-19).** Run-директорій каталожної ділянки локально немає, тож
  використано legacy-хмару `data/dataset_v6_dense.pt` і семантику з прогону 8.1a. Зайнятість
  v2 побудовано заново: ядро 300×300 м по 0.5 м, з вирізанням, 34 с. Origin — середнє кадрів
  Earth Studio, тож lat/lon наближені. AP **розставлено скриптом**: тестове вікно посередині
  фасаду на 6–7.5 м над `v_ground` і майданчик 6×6 м у центрі даху. Це не справжні вікна.
  Результати:
  * з `σ_georef` = 2.5 м (задокументована точність Earth Studio) ціль на вікно **відмовлена**
    (`sigma_too_large`, поріг 1.5 м);
  * з типовим профілем (`σ_GNSS` 1.5 м, `r` транзиту 5.66 м) у радіусі 80 м від origin
    **немає жодної колонки, з якої дозволено зліт**: дерева, авто й нерівності ближче за `r`;
  * з гіпотетичними `σ_georef` 0.5 м і RTK-подібним `σ_GNSS` 0.1 м вікна зі standoff 3 м дали
    **0 з 39** придатних. Головна причина: у 2.5D-моделі стіна «товща» за площину фасаду на
    1–2 м (невідоме/зайняте), плюс дерева вздовж вулиць. Зі standoff 6 м / `D_entry` 18 м
    вийшло **7 з 39** `feasible_with_risks` (лише `wire_risk`). Майданчик на даху: 1 з 1
    придатний; ще 8 дахів відсіяно як занадто нерівні під квадратом 6×6 м. Обидві місії
    експортовано, найгірший зворотний перехід — 0.75 мм.

  Висновок збігається з [SIMULATION §4.9](SIMULATION_AND_DELIVERY.md): двійник Earth Studio
  годиться для перевірки коридорів із великими запасами, але не для вікна на 3 м.
  Потрібен M0.

### 3.19 Перегляд місії й прольоту — `server/api_sim.py`, `sim/replay.py`, `web/app` (ідея 12)

Сторінка **«Місія / проліт»** (`/app/flight/<project>/<site>`, кнопка на сторінці ділянки)
показує місію та її проліт у SITL (SIM-1) у 3D над 3DGS-сценою двійника. На ній:
шлях і точки місії, коридор (труби радіуса `r` кожної ділянки), геофенс (призма від дому до
`FENCE_ALT_MAX`), контур AP, H, E і стрілка курсу в H. Істинна траєкторія розфарбована за
фазою, failsafe (не AUTO) — червоним. Таймлайн відтворює проліт: дрон, висота над домом,
кліренс, режим і остання подія. Поруч — панель вердиктів оракулів (клік переносить у момент
події) і вид згори на карті MapLibre: коридор там має справжню ширину `2r`. Це основа для
реальних логів (ідея 13): `replay.py` бере лише `Recording` і план.

* **API v2, лише читання:** `GET …/sims` (усі прогони всіх місій), `GET …/missions/{id}/geometry`,
  `GET …/missions/{id}/sims`, `GET …/missions/{id}/sims/{scenario}/report` і `…/trajectory`
  (`?seed=n` для `seed-<n>/`). Траєкторія приходить разом із фазою кожного семпла,
  ознакою failsafe, сегментами, висотою над домом, lon/lat і кліренсом. Нічого не
  перераховується. Фаза береться з пункту місії, до якого летить апарат, і з його місця в
  плані: ділянка одразу після H — це відхід, утримання в H відрізняється від підходу допуском
  оракула. Під час failsafe фаза лишається тією, з якої апарат вийшов з AUTO.
* **Кліренс** — це ряд самого оракула (`clearance_series`). Раннер тепер пише його в
  `clearance.json` поруч із `trajectory.json`. Для старих прогонів:
  `python -m aerosplat.sim.report clearance <run> --occupancy <dir>`. Команда відмовляє, якщо
  мінімум не збігається з вердиктом, тобто якщо сітка не та, на якій прогін судили.
* **Кадри.** WGS84 ⇄ Site ENU рахує сервер тим самим `EnuFrame`, що й експорт, тож точка в API
  — це рядок `mission.waypoints`. 3D: `served = ENU сцени − scene-meta.center`. Накладка
  (`web/mission-overlay.js` у `web/mission.html`) — дочірня група `sceneRoot`, тому поворот
  `Rx(−90°)` і масштаб до неї доходять ті самі, що й до сплатів. Другого перетворення, у якому
  можна помилитися, немає. Сцена з іншим origin перераховується через ECEF
  (`web/app/src/lib/frames.ts`). Сцену без `metric` (вільний кадр COLMAP) місія не приймає.
  Тести: `tests/test_api_sim.py` і `web/app/tests/frames.test.ts` (`npm test` у `web/app`).
* **Яка сцена під місією** — двійник, на якому її планували (`twin.scene`): legacy-датасет
  (`legacy:<ds>_dense.pt` → `<ds>`) або сцена ділянки. Якщо сцени на сервері немає, місія
  малюється над сіткою на висоті дому (перемикач «3DGS / Сітка»).
* **Перевірено на `dataset_v6` (2026-10-19, вікно B12/NE4).** Legacy-сцена `dataset_v6` має той
  самий origin, що й місія. Серед сплатів, які в'ювер реально завантажив, у межах ±3 м від
  вікна є пік на площині фасаду (|d| < 0.5 м — 168 сплатів). Між 3 і 10 м перед нею немає
  жодного. H стоїть на 6.0 м, E — на 18 м, траєкторія проходить через обидві. H, спроєктована
  камерою в'ювера, лягає на піксель свого маркера. Lon/lat на карті наближені, бо origin
  Earth Studio приблизний (§3.18): супутникова підкладка зсунута на кілька метрів.
* Сервер тепер віддає `.mjs` як `text/javascript`. На Windows реєстр давав `text/plain`,
  і воркер MapLibre не стартував: шари GeoJSON на картах лишались порожніми.
* **Планування з UI** (панель «Нова місія», 2026-10-19). Ціль (`GET …/targets`) → «Знайти
  місце зльоту» (`POST …/targets/{id}/takeoff`, §3.18) → план. На карті: кандидати з номерами й
  коридором зльоту `r` навколо кожного, кінець транзиту, початок ділянки. Якщо місця немає,
  показано найвідкритіше місце («?», відкрито ≈ vs потрібне `r`) і `σ_GNSS`, з яким воно
  підійшло б. Поруч — повідомлення сервера дослівно й лічильники `rejected`. План — з
  вибраним кандидатом (`home_enu` + `home_from_search` з його номером і параметрами пошуку, з
  тими самими профілем і run; після зміни параметрів результат позначено застарілим) або з
  `home_search` (сервер шукає сам: до 150 м, 5 місць). Поле «Точних перевірок» (`max_checks`,
  ціле 1…400); якщо пошук зупинився на ліміті, панель підказує його збільшити. Відмови 409
  (`outside_grid` з підказкою `occupancy_bbox=`, `no_takeoff_spot`, `takeoff_spot_changed`,
  `target_not_confirmed` …) виводяться дослівно, кожна причина окремо. Поля профілю: порожнє
  поле означає «типово сервера», а нечислове значення — помилку, а не тихе значення за
  замовчуванням. Логіка — `web/app/src/lib/takeoff.ts`, тести — `web/app/tests/takeoff.test.ts`.
* **Ще немає:** оцінки EKF поруч з істиною, еліпсоїдів σ, імпорту DataFlash (ідея 13),
  підкладки DSM/зайнятості в режимі без сцени.

### 3.20 Кандидати вікон за сіткою — `semantics/gridfit.py` (Фаза 7.3, крок 1)

Перший крок автоматичних кандидатів (ROADMAP 7.3) — **без ML і без нових залежностей**: лише
numpy, детерміновано. Регулярний фасад повторюється: поверхи з кроком `h_f`, колонки з кроком
бею. На метричному ортофото фасаду (§3.14) ця регулярність — сильний пріор, і модуль
перетворює її на **кандидатів**, ніколи на `verified` ([ADR-0009](decisions/0009-human-in-the-loop-planning-grade.md)).

* **Маски.** Відрендерено (alpha), покрито (карта покриття фасаду: де жоден кадр не роздільний,
  3DGS домальовує текстуру, §5.3), рослинність (зелений індекс із закриттям прогалин крони) і
  — якщо є рендер глибини — усе, що стоїть **> 2 м перед площиною стіни** (§5.5): дерево, стовп,
  темний фургон, який за кольором не відрізнити від скла. Клітинка, на яку припадає більше
  2 % листя чи оклюдера, кандидатом не стає.
* **Ознаки** (`CUES`): темнота і синява, кожна нормована робастно (медіана/MAD по придатних
  пікселях). Нова ознака — ще один запис у словнику.
* **Колонки.** Крок бею — автокореляція профілю колонок з маскою (дерево скорочує перекриття,
  а не прикидається стіною), основна гармоніка, уточнення по кратних піках. Згорнутий бей
  ділиться на «отвір/стіна» методом Оцу; ознака вважається періодичною, лише якщо контраст
  згортки ≥ 0.3 (слабка брижа з кроком імпостів теж повторюється).
* **Згода ознак.** Якщо обидві ознаки бачать колонки, кожна мусить **підтвердити** колонки
  іншої (вкладення, а не IoU: синява позначає все скло, темнота — лише його затінену частину).
  Ознака, що повторює смугу ширшу за вікно (стовп балконів), не погоджується ні з чим.
  Незгода → **відмова**, а не здогадка. Погоджені ознаки усереднюються: клітинка має бути
  і темною, і синьою, тінь отримує лише половину.
* **Поверхи** — з профілю, обмеженого колонками отворів; крок перевіряється пріором
  2.4–4.2 м (ті самі константи, що й в адресації, §4); пошук ширший за пріор, щоб половинний
  чи подвоєний крок було видно як причину відмови.
* **Рішення по клітинці** — проти опорного рівня **стіни свого поверху**: медіана ознаки
  неотворових пікселів того ж поверху в межах ±1 бею (сонце й AO затіняють довгу стіну
  нерівно). `score = (медіана внутрішньої частини клітинки − опора) / розкид`, поріг 1.0.
  Відсутнє вікно — це стіна, і кандидатом не стає (§5.6 п. 5: при доброму покритті порожня
  клітинка — просто стіна).
* **Відмова — це відповідь**, з кодом і поясненням українською: `not_rendered`, `no_coverage`,
  `no_columns`, `cues_disagree`, `no_strips`, `no_floors`, `floor_out_of_prior`, `no_band`,
  `sparse_grid` (видимих клітинок, схожих на отвори, < 50 %: сітка не описує фасад).
* **Кандидати** (`authored/candidates.py`, спільне для будь-якого майбутнього детектора):
  `lifecycle: candidate`, `origin: pipeline`, `provenance.model = gridfit@<версія>`,
  `confidence`, у `notes` — поверх, крок колонок, ознака, згода, `z`. Пропозиція, що
  перекриває (≥ 30 % меншої з двох фігур) **будь-яку** наявну Feature, відкидається:
  verified — вже є, rejected — не пропонуємо вдруге (§3.4), candidate/stale — повторне
  натискання нічого не додає. Перекриття рахується в 3D через збережений кадр `host`, тож
  відхилене на `B08/SW3` старого run-а блокує ту саму стіну, що в новому run-і зветься
  `B04/SW1` (§5.9). Перевірка і запис — під блокуванням авторського сховища,
  `create_many`: один запис стану, рядок аудиту на кожну Feature.
* **API:** `POST …/facades/{id}/candidates/gridfit` (`gsd_cm` 8, `depth`, `dry_run`, `by`)
  сам рендерить rgb + alpha + depth і повертає результат, кандидатів і `skipped` за
  lifecycle. **Editor:** кнопка «Знайти вікна за сіткою» (`G`): кандидати жовті, перший
  виділено, далі звичайний рев'ю `V`/`X`/`←`/`→`. `VERSION` у модулі піднімається з кожною
  зміною, що може змінити вихід.
* **Тести:** `tests/test_gridfit.py`, `tests/test_api_gridfit.py` на синтетиці з відомою
  відповіддю (`tests/gridfit_scenes.py`); негативних випадків більше, ніж позитивних.

**До/після на `dataset_1910` (2026-10-19, ≈ 8 см/px).** «До» — прототип `detect_v2`
(кільце навколо клітинки, одна ознака). Істина — на рівні колонок: засклені колонки між
пілонами розмічено вручну по лінійці; темні щілини між стовпами лоджій — «неоднозначно» (не
рахуються ні в TP, ні в FP); оклюдери — грубі прямокутники плюс нижній поверх; пояси поверхів
— з яскравих ліній перекриттів у самому зображенні (виміряно лише верхні 9 поверхів). TP —
центр рамки в заскленій колонці, у поясі поверху, не на оклюдері. Розмітка одного оператора,
тож числа орієнтовні.

| Фасад | Прототип z ≥ 0.25 | gridfit 1.0.0 |
|---|---|---|
| B08/SW3, зображення експерименту (старий сплат, без глибини) | 109 рамок, P 0.85, R 0.71 (45/63) | 175 рамок, P 0.84, R 0.98 (62/63) |
| B04/SW1 — та сама стіна, новий run і сплат, з глибиною | 176 рамок, P 0.81, R 0.72 (69/96) | 269 рамок, P 0.86, R 1.00 (96/96) |
| NE1 (обидва сплати) | 122 (z ≥ 0.8) / 293 рамки; за брифом — на тінях балконів, не на склі | **відмова** (`no_strips` / `cues_disagree`) |
| SW2 (зображення експерименту) | відмова | відмова (`no_columns`: контраст 0.13 на кроці 1.33 м) |
| SE1, NW1 (торці) | відмова | відмова (`no_strips`: повторюється смуга 7.1 м) |

* Прототип із порогом z ≥ 0.8 дає 11 і 5 рамок (recall ≈ 0.02); «шумне рішення по клітинці»
  з брифу — це саме вибір між 0.8 і 0.25. Опора по стіні поверху прибрала цей вибір: точність
  та сама, recall 0.98–1.00.
* Помилки gridfit на SW3 без глибини: 8 рамок на стіні/пілоні, 4 на темному не зеленому об'єкті внизу ліворуч, 5 на
  рівні землі. На B04/SW1 глибина знижує рамки на оклюдерах з 29 до 16, і 14 із цих 16 — це
  видимі вікна над кронами, що потрапили в грубі прямокутники розмітки.
* **Ціна відмови:** на NE1 справжні засклені бею є, але ознаки там не погоджуються — їх
  розмічають вручну (recall 0 на цьому фасаді). Це свідомий вибір «відмовити, а не вгадувати».
* 24 випадкові добре покриті фасади інших будівель ділянки: усі 24 відхилено (15 —
  `no_coverage`: глибина показує, що стіну затуляють дерева чи сусіди; 8 — `no_columns`;
  1 — `no_strips`), жодної рамки. Ендпоінт із трьома рендерами — 0.2–0.7 с на фасад.
* Наскрізний `dry_run` на живому каталозі для B08/SW3 пропустив 102 кандидати й 1 підтверджене
  вікно з експерименту, а для `B04/SW1` (та сама стіна в новому run-і) — ті самі 103.

**Ще немає:** порівняння вигляду клітинок між собою (колонки лоджій проходять, якщо там темне
скло), сіток із кількома типами бею, вікон на нерегулярних фасадах і дверей (усе пропонується
як `opening.window`, клас змінює рецензент).

**Після закріплення сцени за run-ом (§3.14, 2026-10-19).** Наявні 169 кандидатів на `B08/SW3`
рахувались зі сплату run-а 2 (помилка № 20). На копії `dataset_1910` пробний gridfit на власному
сплаті run-а 1 з глибиною — **відмова** `floor_out_of_prior` (крок 7.85 м — подвоєння); без
глибини — 176 клітинок, 165 збігаються з наявними кандидатами, 10 нових. На `B04/SW1` (run 2) —
як і раніше: 269 клітинок, 169 пропущено, 99 нових. Тобто сітка залежить від сплату, і кандидати
`B08/SW3` треба переглядати як отримані на іншій реконструкції.

### 3.21 Фасади між run-ами — `semantics/matching.py`, `authored/carry.py` (§5.9)

Новий run нумерує будівлі й фасади заново, тож відповідність шукається геометрично в Site ENU
(спільному для всіх run-ів ділянки, ADR-0006), без переякорювання.

* **Фасад ↔ фасад** (`match_facades`): той самий бік стіни, якщо нормалі розходяться ≤ 10°,
  площини в місці перекриття розходяться ≤ `2·√(rms_a² + rms_b²)` (обмежено 1…3 м — розкид
  площин на `dataset_1910` 0.6 м у медіані), а спільна ділянка вздовж стіни ≥ 2 м (або половина
  коротшого) і по висоті ≥ 30 %. Векторизований префільтр — 1833 × 1276 фасадів за 0.12 с.
* **Відповідь — завжди одна з чотирьох, з причиною:** `match` (`same`; `inside` — фасад
  частина довшого, що в іншому run-і злив кілька, з переліком `shared_with`; `part`;
  `partial`), `split` (кілька фасадів поруч, кожен свою ділянку), `ambiguous` (два фасади
  покривають **ту саму** ділянку — паралельні припасування в межах допуску: нічого не
  вибираємо), `none`.
* **Точка ↔ фасад** (`place_feature`): кути полігона йдуть у ENU через збережений кадр і
  проєктуються на кожен фасад run-а з тими самими порогами; центр у межах фасаду (+0.5 м) і
  ≥ 50 % форми всередині. Рівно один — `placed`, кілька — `ambiguous`, жодного — `none`.
* **Нічого не переноситься саме.** `GET …/facades/{id}/carried` показує чужі точки на стіні;
  `POST …/access-points/{id}/reanchor` — лише на запит людини і лише для `placed` саме на цей
  фасад (інакше 409): висоти зберігаються, форма лягає на нову площину (зсув = відстань між
  площинами, пишеться в аудит `op: reanchor`), стан і історія лишаються, адреса скидається
  (вона називала старий фасад). Підтверджена ціль на цій точці отримає `target_changed`.
* **`dataset_1910` (копія):** `B08/SW3 → B04/SW1` (кут 0.6°, зсув 0.03 м); `B08/SW2` теж
  `inside` `B04/SW1`; `NE1→NE1`, `NW1→NW1`, `SE1→SE1`. Усі 170 точок `B08/SW3` — `placed` на
  `B04/SW1`, зсув площини 0.02–0.30 м (медіана 0.16). Загалом run 1 → run 2: 294 `match`, 10
  `split`, 11 `ambiguous`, 1518 `none` (у межах 100 м від початку: 20 / 1 / 0 / 13). `none` біля
  вежі — низькі стіни подіуму, які run 2 припасував під кутом 10–29°: свідомо не зіставлено.
  Зворотно `B04/NE1` — `ambiguous` (`B08/NE1` і паралельний `B08/NE2` за 1.2 м).

### 3.22 Pad-зони дахів і дворів — `semantics/pads.py` (§5.7)

Із растрів стадії (`terrain.npz`, 1 м) — без нових даних. Для кожної будівлі:

* **Дах:** клітини мітки будівлі. Ухил — локальна площина МНК у вікні 5 м (точна й на краю
  даху), опора — NaN-медіана, тож комин чи кондиціонер > 0.3 м над нею — перешкода; клітини,
  які маска будівлі викинула всередині контуру (стадія відсіює шорсткі — комини, антени), і
  неспостережені — теж перешкоди (невідоме = зайняте). Сходинки > 0.3 м між сусідами ділять
  дах на рівні.
* **Двір:** спостережена гола земля (`DSM − DTM < 0.3 м`, не будівля) у радіусі 30 м; крона —
  перешкода. **Відкритість неба** рахується для кожної придатної клітини (горизонт у 16
  напрямках до 40 м): частина зони в щілині між високими стінами відрізається з причиною.
* Кожна зв'язна рівна ділянка: площина МНК (ухил ≤ 5°, залишок ≤ 5 см — числа §5.7), ерозія на
  `clearance_m` 2 м (0.5 м дрон + 1 м запас + 2σ, σ = пів клітини), площа центру ≥ 4 м². Зона —
  місце, куди може стати **центр** дрона. Вигнута чи з дірою (комин усередині) зона стає
  квадратом навколо точки з найбільшим запасом: контур ніколи не охоплює перешкоду.
* Кожна відмова — з кодом (`slope`, `rough`, `no_room`, `sky`, `no_data`, `no_roof`) і текстом.
* `POST …/pads/candidates` (`building_ids`, пороги, `dry_run`): `candidate` класу
  `pad.roof_flat` / `pad.ground` (`core`), `origin: pipeline`, `surface/hazard: unknown`,
  метрики в `notes`, ніколи не `verified`; пад, що перекриває наявний (будь-який стан, ≥ 30 % і
  |Δz| < 2 м), не пропонується. Схему `PadGeometry` не змінено (сумісність старих читачів).
  Планувальник місій розуміє `ap.pad` як і раніше (центр, `building_id`).
* **Межі:** 1-м растр губить перешкоди менші за клітину; дороги, вода, авто не розпізнаються
  (у `notes` кожної зони двору); дротів не видно (§5.8).
* **`dataset_1910` (копія, 10 будівель біля початку):** за порогами §5.7 run 2 дає 1 зону двору й
  жодної на дахах — дах вежі `B04` рівний (ухил 0.1°), але залишок 7.5–9.3 см на 1-м растрі
  Earth Studio; із `max_rms_m = 0.10` — 3 зони на даху вежі (центр може бути на 20, 40 і
  136 м², z ≈ −125.2 м) і 22 у дворах. Run 1 за порогами §5.7 — 0 зон. Пороги лишено за
  специфікацією; ослаблювати — рішення власника (§6 журналу).

### 3.23 Експорт CityJSON — `semantics/cityjson.py` (§2.6)

* **Що:** кожна будівля run-а — `Building` з **LoD1** `Solid` (контур, витягнутий від `ground_z`
  до `roof_z`; Ground/Roof/WallSurface). Будівлі з експортованими точками — ще **LoD3**
  `MultiSurface`: стіни з отворами (`WallSurface`, отвори вирізані дірками, якщо лежать строго
  всередині й не перекриваються; інакше `aerosplat_not_cut`), `Window`/`Door` з парою
  parent ↔ children, pad-и дахів як `RoofSurface`. На кожній поверхні — `aerosplat_id`, клас,
  стан, адреса. Класи без відповідника CityGML, pad-и землі й отвори, чию стіну цей run не має
  однозначно, — `GenericCityObject` з `aerosplat_class` і причиною. Отвори з інших run-ів
  кладуться на стіну лише як `placed` (§3.21). LoD2 немає: дахових площин стадія не будує.
* **Координати:** UTM-зона початку Site (`geo.crs.utm_epsg`, з винятками Норвегії/Свальбарду;
  `SiteProjection`: ENU → еліпсоїд → UTM точно, зі збіжністю меридіанів і масштабом), **висоти
  WGS84 еліпсоїдні** (геоїда в системі немає — так і сказано в `metadata.title`), цілі вершини з
  `transform` 1 мм. Контури з растру, що торкаються себе в куті клітини, беруть найбільшу петлю
  (у звіті й атрибуті); контур, що перетинає себе, — без LoD1, а не зламана геометрія.
* **Перевірка без cjio/cjval** (`validate`): корінь, transform, індекси, кільця (≥ 3 різні,
  незамкнені, без повторів, прості, пласкі до 1 см), дірки всередині й навпаки орієнтовані, типи
  семантики й parent/children, оболонки `Solid` замкнені й назовні, невикористані вершини,
  `geographicalExtent`. Тест ламає документ по одному правилу. CLI не пише файл, API повертає
  500, якщо експорт не пройшов.
* **Як:** `aerosplat semantics cityjson --site P/S [--run R] [--lifecycle verified candidate]
  -o out.city.json`; `GET …/export/cityjson?run_id=&lifecycle=` (`application/city+json`,
  звіт у `X-Export-Report`). За замовчуванням — лише `verified`.
* **`dataset_1910` (копія, run 2, verified + candidate):** 360 будівель, LoD1 у 341 (103 контури
  виправлено, 19 пропущено як самоперетинні), вежа `B04` з LoD3-стіною `SW1` і 170 отворами,
  намальованими в run-і 1; EPSG:32618; перевірка чиста; 1.4 МБ.

### 3.24 SIL-раннер і навігація без GNSS — `aerosplat/sim/` (SIM-0, SIM-1, SIM-2n)

Місії з `aerosplat.mission` літають у зафіксованому ArduCopter SITL (WSL2, `configs/sitl.yaml`)
і судяться за **істиною** `SIM_STATE` проти двійника. Модулі й прогони —
[SIMULATION_AND_DELIVERY §4.8–§4.11](SIMULATION_AND_DELIVERY.md); тут — лише будова.

* `sitl` — запуск зафіксованої збірки (`tools/sitl/*.sh` через `wsl.exe --exec`); `start(...,
  boot_defaults=)` додає файл параметрів, з яким SITL **завантажується** (SIM-2n).
* `params` — таблиця параметрів самої збірки: знімок `Copter-4.7.2.params.json` і варіант
  `Copter-4.7.2.extnav.params.json` (знято з увімкненими ExtNav, потоком і далекоміром:
  ArduPilot показує `VISO_*`, `RNGFND1_*`, `FLOW_*` лише тоді).
* `scenario`, `faults`, `mavmission`, `runner`, `record`, `oracles`, `report` — SIM-1 (§4.10).
  Оракули: `clearance`, `fence`, `hover`, `heading`, `altitude_frame`, `failsafe`, `completion`,
  **`wall_descent`** (жодного зниження ближче X м від стіни, H-011; для всіх прогонів) і для
  прогонів без GNSS **`navigation`** (жодного фіксу GNSS, origin = Site origin, оцінка EKF ↔
  істина з фіксацією VPS, час виявлення спуфінгу).
* `extnav` — **companion-емулятор** (ADR-0015 етап 1): `SET_GPS_GLOBAL_ORIGIN` = Site origin із
  читанням назад, `ODOMETRY` з істини + модель похибки VPS (шум, зсув Гаусса–Маркова, затримка,
  пропуски, зони без фіксації), реакції на втрату VPS (`MAV_CMD_SET_EKF_SOURCE_SET`, відхід від
  стіни за «далекомірами» — променями в зайнятість двійника, посадка), монітор GNSS.
* `camera` — профілі камер (`configs/sim/cameras/imx290.yaml`: інтринсики для 2.8/4/6 мм,
  rolling shutter); дані для SIM-3 і карти локалізовності.
* Профілі апаратів — `configs/sim/vehicles/` (`sitl-quad`; `sitl-quad-extnav` з потоком і
  далекоміром SITL, `boot_params` — драйвери, що діють лише з увімкнення; `x500-pi5-imx290` —
  тестовий апарат, ADR-0015 §6). Розділи про справжній апарат (`frame` з r_drone, `cameras` з
  кріпленням, `rangefinders`, `flow`, `companion`, `nav`, `hardware_params` — лише для заліза,
  у SITL не надсилаються) описує `sim.airframe` і перехресно перевіряє `VehicleProfile`
  (r_drone не менший за кінці пропелерів і не більший на > 0.15 м; камера профілю = камера VPS;
  далекомір потоку дивиться вниз). Усі розділи необов'язкові — старі профілі не змінились.
* Профіль навігації місії — `mission.models.NavProfile` у `PlanningProfile.nav`; експорт
  (`export.nav_parameters`, `nav_problems`) пише й перевіряє навігаційну частину `.parm`.

---

### 3.25 GIS-експорт — `aerosplat/export/` (Фаза 3.2)

Стандартні GIS-файли одного завершеного run-а: ортофото, DSM/DTM/nDSM, хмара точок, 3DGS
`.ply`. Тут продукти вперше залишають WGS84 Site ENU (§4) для стандартної проєкційної СК.

* **СК** (`export/georef.py`). За замовчуванням — **UTM-зона WGS84, у якій лежить origin
  ділянки** (EPSG:326zz/327zz, за EPSG-зонами, тобто за довготою; MGRS-винятки Норвегії/Шпіцбергена
  не застосовуються). Чому UTM: метрична й конформна (піксель квадратний, масштаб 0.9996…≈1.001
  записується), той самий датум, що й Site ENU (жодного перерахунку датуму), EPSG-код читає
  будь-яка GIS, COG-в'ювер і LAS-утиліта; ділянка ~1 км, зона — 6°. Можна будь-яку іншу
  проєкційну СК у метрах (`--crs EPSG:xxxx`); для іншого датуму pyproj вибирає перетворення, і
  його точність потрапляє в бюджет; СК без відомої точності, з одиницями не в метрах, географічна
  (EPSG:4326 — пікселі не квадратні в метрах) або така, що не визначена в origin, — відмова.
  **Висоти — еліпсоїдні WGS84** (h з EPSG:4979), перераховуються точно для кожної точки/клітини:
  u ≠ h, дотична площина відходить від еліпсоїда на d²/2R (2 см на 500 м). Чи справді висоти Earth
  Studio еліпсоїдні — не перевірено (§4), і це записано у файлі.
* **ENU → СК для растрів** (`SiteProjection.fit_plane`): афінне (e, n) → (E, N) на висоті землі,
  підігнане до точних pyproj-перетворень решітки 9×9; залишок перевіряється (> 5 мм — відмова; на
  `dataset_1910` 1.6×1.2 км — 0.6 мм). Висота в підгонку **не** входить: ортографічний вид униз
  проєктує вздовж вертикалі origin, а не місцевої, і точка на висоті Δh зсувається на ≈ d/R·Δh —
  це не афінне (росте з e·u), тому міряється точно й пишеться як `lean_max_m` (на `dataset_1910`
  7.6 мм для всього діапазону висот).
* **Ортофото** (`export/ortho.py`): рендеримо **одразу в сітку СК**, а не перевибірковуємо після.
  На ділянці ENU → UTM — подібність (масштаб k і поворот на зближення меридіанів), тож ортокамера
  з `yaw` = поворот сітки і k/px пікселів на ENU-метр ставить центри пікселів точно на сітку UTM.
  Мозаїка: кожна плитка (`--tile`, 2048 px) — окремий рендер, чий блок пікселів є вікном
  глобальної сітки; плитки вставляються без змішування й ресемплінгу, шви не можуть зсунутись.
  Кожна плитка перевіряється двічі: план (ground grid камери через **точну** проєкцію має лягти на
  центри пікселів, ≤ 0.05 px; на `dataset_1910` — 0.0025 px) і рендер (`ground_grid` у мета
  рендерера = запланований, інакше відмова — рендерер, що тихо змінив extent, yaw чи конвенцію
  пікселя, не пройде). Колір: рендерер компонує на чорне (premultiplied), ділимо на alpha-рендер →
  прямий RGBA, alpha — четвертий канал (маска GDAL): де сплат не непрозорий, у GIS прозоро, а не
  «чорна земля». Рендерер — `RenderEngine` у процесі експорту, **закріплений за конкретним run-ом**
  (`RenderEngine.pin_scene`; сервіс рендерингу показує лише останній run ділянки) і з перевіркою
  `meta.run`. Площа — охоплення хмари run-а (викиди вже прибрала стадія parse) або `--area`.
* **DSM/DTM/nDSM** (`export/dem.py`) — **ті самі** растри `semantics/terrain.npz`, з яких виведено
  будівлі й фасади, перевибрані в північно-орієнтовану сітку СК. Ресемплінг ніколи не перекидає
  значення через дірку: білінійно, якщо всі 4 клітини мають дані, інакше — власне значення клітини,
  інакше nodata (суворий білінійний варіант відкидав ~40 % клітин розрідженої хмари: 13 % → 8 %).
  DTM — оцінка: під будівлями інтерпольована за задумом, але далі ніж `--max-gap` (25 м) від
  виміряної клітини — nodata. nDSM — nodata там, де DSM, а не 0 (0 означав би «тут земля»).
  float32, nodata −9999.
* **LAS/LAZ** (`export/pointcloud.py`, laspy): LAS 1.4, формат точок 7 (RGB 16 біт), int32 з
  кроком 1 мм і зсувом біля мінімуму; CRS — WKT1-VLR (LASF_Projection 2112) з бітом WKT у global
  encoding; Z — еліпсоїдна h (WKT1 PROJCS вертикаль не виражає — сказано в нашому VLR). Джерело —
  `cloud` (за замовчуванням: зареєстрована хмара run-а, тобто dense MVS після очищення parse,
  з її статистикою) або `splat` (центри гаусіанів з DC-кольором — не точки поверхні).
* **3DGS `.ply`** (`export/splat_ply.py`): INRIA-розкладка (x y z, nx ny nz, f_dc, f_rest
  **channel-major**, opacity — logit, scale — log, rot — w x y z) з активованого `.pt` за контрактом
  ADR-0002 (без контракту — відмова, не вгадуємо). Без SH — ступінь 0, f_dc = (rgb − 0.5)/C0.
  Координати лишаються в **Site ENU** (PLY не має CRS, а UTM-числа ~4·10⁶ у float32 — кроки по
  0.5 м); origin, кадр, earth model і `rasterize_mode` — у коментарях заголовка. Розбіжні гаусіани
  (NaN, нульовий кватерніон) відкидаються й рахуються. Є читач (`read_inria_ply`) для перевірки.
* **Відмова, а не тихий файл.** Run не `ok`; немає геоприв'язки (`run_metric` → `metric` null —
  та сама умова, з якою сервер не дає `scene-meta.metric`); для `sphere-enu+affine` файл не
  `*_wgs84.pt` або `.pt` без `frame = WGS84 Site ENU`; семантика не в Site ENU; немає семантики
  (повідомлення каже, яку стадію запустити); забагато пікселів; паралельний експорт того самого
  run-а (файловий замок). Нічого не пишеться під іменем продукту, доки файл не готовий
  (тимчасове ім'я → `os.replace`).
* **Бюджет похибки геоприв'язки** (`georeference_budget`) — у кожному файлі (теги GeoTIFF
  `AEROSPLAT_GEOREFERENCE`, VLR `AeroSplat` у LAS, коментарі PLY) і в `exports.json`: `earth_model`,
  `frame_error_m` і що він означає для цієї моделі (залишок афінного сфера→WGS84; legacy-зсув;
  «невідомо» для `wgs84`), `altitude_reference`, виправлене сфера→WGS84 (довідково), параметри
  проєкції (масштаб, поворот, залишок, lean, точність перетворення датуму) і **`not_included`** —
  чого ніхто не міряв: абсолютна точність Earth Studio, похибка реконструкції, вертикальний датум.
* **Продукти run-а.** Вихід — `runs/<run>/exports/` (`ortho.tif`, `dsm.tif`, `dtm.tif`,
  `ndsm.tif` — COG; `cloud.laz`; `splat.ply`; `exports.json` з sha256, СК і деталями кожного
  файлу); у каталозі — продукти `export_<kind>` і `exports` через новий
  `Catalog.register_products`, який **не чіпає** статус і `finished_at` run-а (повторне
  `finish_run` пересунуло б, який run ділянка показує). Шлях реєструється абсолютним, а
  незареєстрований файл — помилка, не мовчання.
* **CLI / задача / API / UI.** `aerosplat export --site … [--run …] [--kinds …] [--crs …] [--gsd …]`
  (код 2 — відмова). Задача `JobKind.export` (`ExportSpec`): GPU-воркер, якщо є `ortho`, інакше
  CPU; кожен вид — стадія в моніторі. API (`server/api_export.py`): `POST
  /api/v2/sites/{p}/{s}/runs/{e}/{r}/exports` (409 з поясненням без геоприв'язки, 422 погана СК),
  `GET …/exports` (файли й бюджет), `GET …/exports/{kind}` (завантаження). UI — розділ «Експорт
  GIS» у деталях run-а на сторінці ділянки.
* **Залежності** — extra `gis` (`pip install -e .[gis]`: rasterio, laspy, lazrs); без них
  відповідний експорт відмовляє з підказкою.
* **Перевірено** (`tests/test_export.py`, `tests/test_export_run.py`): маркер у відомій ENU-точці
  на ортофото лягає на свої lon/lat (< 0.1 px), мозаїка без швів (кожен піксель має колір своєї
  ENU-позиції), рендерер зі зсувом на 0.5 px — відмова без файлу; висоти DSM = точна h у
  lon/lat клітини (< 5 мм), дірка — nodata; точки LAS на своїх lon/lat (< 2 мм) і h (< 1 мм);
  `.ply` туди-назад, порядок властивостей і channel-major; відмови й API. На копії
  `dataset_1910` (run `20261019-105253`): ортофото 6431×5004 px по 0.25 м, 12 плиток, ~1 хв
  на RTX 4060; DSM/DTM/nDSM 1610×1253 по 1 м; LAZ 2.59 млн точок; `.ply` 130 МБ; межі будівель
  з nDSM лягають на ортофото (узгодженість між продуктами, не абсолютна перевірка).

### 3.26 Візуальна локалізація відносно двійника — `aerosplat/vps/` (ADR-0015, шар A)

Прототип VPS: кадр камери апарата → поза в Site ENU з коваріацією **або відмова з причиною**.
Дві половини з жорсткою межею залежностей: те, що летить на companion (Raspberry Pi 5), імпортує
лише numpy + OpenCV (тест перевіряє в окремому інтерпретаторі, що `torch`, `gsplat`, `pydantic`
не завантажуються); те, що будує атлас, живе на наземній станції з GPU. Бенч і числа —
[SIMULATION_AND_DELIVERY §4.12](SIMULATION_AND_DELIVERY.md).

* `features` — екстрактори, що підключаються: ORB (32 Б) і AKAZE (61 Б), обидва бінарні
  (Hamming); ознаки розкладаються по сітці кадру (`bucket`), бо PnP потрібні точки по всьому
  кадру, а не лише багато. Координати пікселів — OpenCV (центр пікселя (0,0) у (0,0)); render
  service має центр у (0.5, 0.5) — `geometry.cv_intrinsics` переводить. Глобальний дескриптор
  (кадр 32×18) — лише для пошуку без апріорної пози.
* `atlas` — один `.npz` без pickle: пози ключових кадрів, точка, куди дивиться вісь, ознаки
  (піксель, дескриптор, 3D-точка в Site ENU відносно `anchor`), метадані (run, sha256 сплату,
  інтринсики, екстрактор і параметри, версії ПЗ, область). **`content_hash`** (sha256 усіх
  масивів і метаданих) перевіряється при завантаженні; `expect_hash` — хеш, який назве пакет
  місії (ADR-0015 §4). Пошкоджений, змінений чи «не той» атлас — `AtlasError`, не робота.
* `build` — збирач атласу з будь-якого `Renderer` (rgb + z-глибина): глибина в точці ознаки —
  білінійна; точки на силуетах (розкид глибини у вікні 5×5 > 6 %), без глибини й далі 250 м
  відкидаються — очікувана глибина 3DGS змішує дах і вулицю на краю.
* `twin` — run ділянки як `Renderer`: `RenderEngine` у процесі, сцена закріплена за шляхом
  (`gs/splat_wgs84.pt`), **без каталогу** (нічого під `projects/` не пишеться); DTM/DSM з
  `semantics/terrain.npz` для висот над землею й перевірки, що поза не в будівлі.
* `localize` — `Localizer(atlas).localize(image, K, prior)`: кандидати (найближчі за точкою
  погляду до апріорної пози з EKF/AHRS; без неї — глобальний дескриптор) → зіставлення з
  ratio-тестом **проти найближчої іншої 3D-точки** (той самий об'єкт з сусідніх кадрів не
  вбиває правильний збіг) → PnP-RANSAC (AP3P) + LM → перевірки. Причини відмови:
  `bad_input`, `no_features`, `outside_atlas`, `too_few_matches`, `pnp_failed`,
  `too_few_inliers` (< 40), `low_inlier_ratio` (< 0.30), `poor_geometry` (inlier-и < 5 з 16
  клітин або вироджена інформаційна матриця), `high_residual`, `implausible_pose`,
  `sigma_too_large` (σ_h > 1.5 м або курс > 3°), `ambiguous` (інша поза пояснює ≥ половину
  inlier-ів серед решти збігів), `prior_mismatch` (далі за 3σ від апріорної — HOLD, ADR §2.5).
  Коваріація — інформація Гаусса–Ньютона репроєкцій (чисельний якобіан), масштаб залишку з
  підлогою 0.7 px × 2, плюс σ_map = 0.3 м самого двійника.
* `perturb` — фотометричні збурення, розмиття, шум, JPEG (грубе наближення доменного розриву).
* `bench` — `run` (GPU: атласи, набір запитів, експорт) і `localize` (CPU: те, що запускається на
  Pi з копії папки).

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
| `gs/splat_checkpoints/ckpt_*.pt` (`<ds>_splat_checkpoints/`) | train (`checkpoint_every`) | train (`train_resume`) | `torch.save` dict, лише тензори й прості типи (`weights_only`) |
| `masks/<кадр>.png` поруч із кадрами захоплення | людина / сегментатор | train (`masks`) | PNG, 0 = ігнорувати піксель |
| `mvs/stereo/depth_maps/*.geometric.bin` | dense | train (`depth_reg: mvs`) | COLMAP dense array (float32) |
| `gs/{dense,cloud,splat}_wgs84.pt` (`<ds>_*_wgs84.pt`) | georef | server | той самий формат + ключ `frame` (WGS84 Site ENU, A, b, джерело) |
| `reports/metrics.json`, `report.html` (`<ds>/run/`) | eval | людина | JSON / HTML |
| `occupancy/occupancy.{npz,json}` | occupancy | mission (8.1b), SIL | npz (`state`, `esdf_cm`, колонки) + JSON `occupancy-v1` (§3.17) |
| рядок `runs` + `products` у `catalog.db` | `run --site`, migrate | server | SQLite (шляхи відносно run-а) |
| `exports/{ortho,dsm,dtm,ndsm}.tif`, `cloud.laz`, `splat.ply`, `exports.json` | `aerosplat export` | GIS, людина, API (завантаження) | COG у UTM-зоні ділянки (або `--crs`), LAS 1.4 + WKT, INRIA PLY у Site ENU; бюджет геоприв'язки в кожному (§3.25) |
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
│   ├── ingest/                    # earthstudio, folder, images, upload (ingest-задача)
│   ├── sfm/                       # sfm_runner, dense_runner
│   ├── io/                        # cloud_io, ply_parser, colmap_parser
│   ├── gs/                        # trainer (активний), gsplat_backend, sh, evaluate, report
│   ├── jobs/                      # models, queue, worker, child, proctree (черга задач, S2)
│   ├── auth/                      # tokens, context, cli (`aerosplat token`; доступ не з localhost)
│   ├── render/                    # models, camera, engine, app, service (сервіс рендерингу, S2)
│   ├── semantics/                 # terrain, buildings, facades, coverage, ortho, gridfit (семантика, S3)
│   ├── surface/                   # occupancy, esdf, stage, products (зайнятість + ESDF, 8.1a)
│   ├── mission/                   # targets, planner, corridor, export, store (місії v0, 8.1b)
│   └── server/                    # main.py (FastAPI, сцени v1), auth.py, api_v2.py, api_semantics.py, api_mission.py, up.py

│   ├── sim/                       # sitl, params, runner, oracles, extnav, camera, airframe… (SIL, SIM-0/1/2n)
│   ├── vps/                       # atlas, features, localize (companion, CPU), build, twin, bench (ADR-0015 A)
│   └── server/                    # main.py (FastAPI, сцени v1), api_v2.py, api_semantics.py, api_mission.py, up.py
├── web/                           # класичний в'ювер: index.html, appGLSL.js, depth-sort_worker.js, aero-debug.js
│   └── app/                       # веб-UI (Vite + React + TS); dist/ — збірка (ignored); e2e/ — Playwright
├── configs/                       # earthstudio.yaml, drone.yaml (пресети RunConfig)
├── tests/                         # CPU-тести + golden-фікстури Earth Studio
├── tools/                         # дослідницькі скрипти, gsplat_env.bat, migrate_dataset.py, e2e_server.py
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

---

## 9. Процес: журнал планів і рішень

Після кожного плану (фаза, крок, сесія з підсесіями) окремо записуємо **помилки, рішення, чому
саме так, альтернативи, припущення й відкрите** — один файл на план. Цей документ описує систему
«як є», ADR — окремі довгі рішення; журнал зберігає шлях до них, щоб не повторювати відкинутого й
не губити уроків. Правило власника від 2026-10-19.

Репозиторій публічний, тому журнал, звіти сесій, внутрішні рішення й вимоги домену лежать **поза
git**, у приватній теці власника (див. `CLAUDE.md`); у публічних документах на них лише
текстові посилання «внутрішній журнал» / «внутрішні вимоги надійності».
