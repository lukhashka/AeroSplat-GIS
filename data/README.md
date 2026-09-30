# data/

Тека **не відстежується git** (див. `.gitignore`): зображення, `.pt`, `.ply`,
COLMAP-вихід і `.xlsx` тут лише локально.

## Очікувана структура датасету `<ds>`

```
data/<ds>/
├── footage/            # кадри (jpeg), напр. <ds>_00.jpeg …
├── <ds>.json           # Earth Studio «3D tracking data» (для camera_source=earthstudio_json)
└── sparse_known/       # генерується стадією geo (відомі пози COLMAP)
```

Решта (`raw_images_<ds>/`, `colmap_output_<ds>/`, `<ds>_dense.pt`,
`<ds>_splat.pt`) створюється пайплайном.

## Звідки взяти дані

* **Earth Studio:** проєкт → Render (JPEG-кадри) + Export → *3D Tracking Data (JSON)*.
  Еталонний датасет `dataset_v6` зберігається у зовнішньому сховищі власника
  (додай посилання сюди: `TODO`).
* **Реальні дрони:** див. [docs/REAL_DRONES.md](../docs/REAL_DRONES.md).

Рішення про зовнішнє сховище великих датасетів (DVC / Git LFS / хмарна тека)
ще не прийнято — див. ROADMAP P0-3.
