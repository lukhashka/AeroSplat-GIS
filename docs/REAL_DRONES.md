# Підтримка реальних БПЛА (окремий трек)

Earth Studio — поточне джерело, але пайплайн має працювати й на реальних
знімках. Ключова відмінність: в Earth Studio **точні пози і внутрішні
параметри відомі** (known-poses), у дрона — **ні** (є лише наближені GPS/IMU).

## Порівняння режимів

| | Earth Studio (зараз) | Реальний дрон |
|---|---|---|
| Пози камер | точні, з JSON | наближені (EXIF GPS + XMP gimbal): метри (GNSS) … см (RTK) |
| Інтринсики | відомі (`fovVertical`) | невідомі; дисторсія (SIMPLE_RADIAL/OPENCV), rolling shutter |
| Модель Землі | **сфера** R=6371008.8 | **WGS84 еліпсоїд**; висоти ellipsoidal vs MSL |
| SfM | `point_triangulator` + BA | повний mapper (COLMAP/GLOMAP) з GPS-пріором |
| Метрика | ENU з JSON | Umeyama/RANSAC COLMAP→ENU по EXIF GPS (або `model_aligner`) |
| Освітлення | однакове | змінне; авто-експозиція → appearance embedding |

## Що вже працює

* `synthetic`-режим + `exif_injector.py` (інжекція GPS/FOV), `sfm_runner.py`
  з `--matcher spatial` (використовує GPS з EXIF), `--mapper colmap|glomap`.
* З Фази 0: `camera_source=auto`; `known_intrinsics=False` за замовчуванням;
  тренування на undistorted-кадрах (`dense/images`) — необхідно для реальної оптики.

## Що треба зробити (за пріоритетом)

1. **Адаптер входу** `inputs/drone`: EXIF GPS + DJI XMP (`GimbalYawDegree`,
   `RelativeAltitude`), SRT-телеметрія для відео; вивід у тій самій структурі,
   що `earthstudio_to_colmap` (кадри + початкові пози).
2. **Геореєстрація без known poses:** similarity COLMAP→ENU за GPS
   (Umeyama + RANSAC), звіт RMSE; запис `geo_registration.json` у тому ж
   форматі, що й для Earth Studio (Фаза 3.1).
3. **GPS-пріор у BA:** `--Mapper.use_prior_position` / `model_aligner`.
4. **Камера:** self-calibration (не заморожувати fx), `OPENCV` для широкого кута;
   вибір кадрів з відео (різкість, перекриття).
5. **WGS84 vs сфера:** в `aerosplat.geo` (Фаза 3.1) явна опція
   `earth_model: sphere|wgs84` (Earth Studio = `sphere`, дрони = `wgs84` + геоїд для MSL).
6. **Appearance embedding / exposure** у тренері (Фаза 2.2).
7. **Тестовий реальний датасет** (10–20 кадрів) для регресій; відкриті: Mill-19, UrbanScene3D.

## Принцип

`starter.py` обирає адаптер лише на стадіях `geo`/`exif`. Усе після `sfm` не
повинно містити `if earthstudio`. Перевіряти при рев'ю змін.
