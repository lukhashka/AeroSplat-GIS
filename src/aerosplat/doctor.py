#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
doctor.py — діагностика даних пайплайну AeroSplat-GIS.

Нічого не запускає і нічого не змінює. Читає те, що вже є на диску,
і каже, де саме пайплайн зламався (або зламається).

    aerosplat doctor
    aerosplat doctor --dataset dataset_v6
    aerosplat doctor --stage sparse      # лише один розділ
    aerosplat doctor --verbose           # + список кадрів/пар

Перевіряє:
  1. footage    — скільки кадрів, роздільність, чи однакова, реальний FOV
  2. exif       — чи записано GPS, чи не вироджений трек
  3. database   — фічі на кадр, пари з матчами, geometric verification
  4. sparse     — скільки кадрів зареєстровано, track length, покриття
  5. dense      — depth maps, fused.ply, кількість вершин
  6. cloud/pt   — що реально лежить у .pt (bbox, кількість точок)
"""

from __future__ import annotations

import argparse
import math
import os
import sqlite3
import struct
from collections import Counter
from pathlib import Path

from aerosplat.paths import DATA_DIR, PROJECT_DIR

OK, WARN, BAD, INFO = "✅", "⚠️ ", "❌", "  "
_problems: list[str] = []


def say(mark, msg):
    print(f"{mark} {msg}")


def bad(msg):
    _problems.append(msg)
    say(BAD, msg)


def warn(msg):
    _problems.append(msg)
    say(WARN, msg)


def header(title):
    print(f"\n{'─' * 66}\n{title}\n{'─' * 66}")


# ══════════════════════════════════════════════════════════════════════
# 1. FOOTAGE
# ══════════════════════════════════════════════════════════════════════
def check_footage(footage_dir: Path, verbose=False) -> dict:
    header(f"1. FOOTAGE  —  {footage_dir}")
    info: dict = {}
    if not footage_dir.is_dir():
        bad(f"теки не існує: {footage_dir}")
        return info

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    frames = sorted(p for p in footage_dir.iterdir()
                    if p.suffix.lower() in exts)
    info["count"] = len(frames)
    if not frames:
        bad("жодного кадру не знайдено")
        return info

    say(OK if len(frames) >= 60 else WARN, f"кадрів: {len(frames)}")
    if len(frames) < 40:
        warn(f"{len(frames)} кадрів для орбіти 360° — це ~{360/len(frames):.1f}° "
             f"між сусідніми. Для SfM бажано <6° (60+ кадрів).")

    try:
        from PIL import Image
    except ImportError:
        warn("Pillow не встановлено — пропускаю аналіз зображень")
        return info

    sizes: Counter = Counter()
    total_bytes = 0
    for p in frames:
        try:
            with Image.open(p) as im:
                sizes[im.size] += 1
        except Exception as e:
            bad(f"не читається {p.name}: {e}")
        total_bytes += p.stat().st_size

    if len(sizes) == 1:
        (w, h), _ = sizes.most_common(1)[0]
        info["width"], info["height"] = w, h
        say(OK, f"роздільність: {w}×{h} (однакова у всіх кадрів)")
    else:
        bad(f"РІЗНІ роздільності: {dict(sizes)}")
        bad("  --known-intrinsics з однією камерою тут некоректний "
            "(single_camera=1 передбачає ідентичні кадри)")
        (w, h), _ = sizes.most_common(1)[0]
        info["width"], info["height"] = w, h

    say(INFO, f"середній розмір файлу: {total_bytes/len(frames)/1024:.0f} KB")

    if verbose:
        say(INFO, f"перший: {frames[0].name}   останній: {frames[-1].name}")
    return info


# ══════════════════════════════════════════════════════════════════════
# 2. INTRINSICS SANITY
# ══════════════════════════════════════════════════════════════════════
def check_intrinsics(info: dict, cfg_w, cfg_h, cfg_fov):
    header("2. INTRINSICS  —  що COLMAP отримає з --known-intrinsics")
    if "width" not in info:
        say(INFO, "немає даних про кадри — пропускаю")
        return

    w, h = info["width"], info["height"]
    say(INFO, f"реальні кадри:  {w}×{h}")
    say(INFO, f"конфіг RunConfig: {cfg_w}×{cfg_h}, fov={cfg_fov}°")

    if (w, h) != (cfg_w, cfg_h):
        bad(f"НЕЗБІГ РОЗДІЛЬНОСТІ: кадри {w}×{h}, а в конфізі {cfg_w}×{cfg_h}")
        bad("  focal length рахується з конфігу і ЗАМОРОЖУЄТЬСЯ в BA "
            "(--Mapper.ba_refine_focal_length 0)")
        bad("  → це найімовірніша причина провалу реєстрації")
        say(INFO, f"  виправ:  --set image_width={w} --set image_height={h}")
        say(INFO, "  або взагалі:  --set known_intrinsics=false")
    else:
        say(OK, "роздільність збігається з конфігом")

    f_cfg = (cfg_w / 2.0) / math.tan(math.radians(cfg_fov) / 2.0)
    say(INFO, f"f з конфігу = {f_cfg:.2f} px  (буде заморожено)")
    say(INFO, "правдоподібні f для інших FOV на цих кадрах:")
    for fov in (30, 40, 50, 60, 70, 90):
        say(INFO, f"    fov={fov:2d}° → f={(w/2.0)/math.tan(math.radians(fov)/2.0):8.1f}")
    say(INFO, "  FOV Earth Studio треба звірити з реальним експортом —"
              " якщо не знаєш точно, надійніше known_intrinsics=false")


# ══════════════════════════════════════════════════════════════════════
# 3. EXIF / GPS
# ══════════════════════════════════════════════════════════════════════
def _dms_to_deg(dms, ref):
    def r(x):
        return x[0] / x[1] if isinstance(x, tuple) else float(x)
    deg = r(dms[0]) + r(dms[1]) / 60.0 + r(dms[2]) / 3600.0
    if ref in (b"S", b"W", "S", "W"):
        deg = -deg
    return deg


def check_exif(geotagged: Path, verbose=False):
    header(f"3. EXIF / GPS  —  {geotagged}")
    if not geotagged.is_dir():
        bad(f"теки не існує: {geotagged} (стадія exif не виконана?)")
        return

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    frames = sorted(p for p in geotagged.iterdir() if p.suffix.lower() in exts)
    say(INFO, f"геотегованих кадрів: {len(frames)}")
    if not frames:
        bad("тека порожня")
        return

    try:
        import piexif
    except ImportError:
        warn("piexif не встановлено — пропускаю перевірку GPS")
        return

    coords, missing = [], 0
    for p in frames:
        try:
            ex = piexif.load(str(p))
            g = ex.get("GPS") or {}
            if piexif.GPSIFD.GPSLatitude not in g:
                missing += 1
                continue
            lat = _dms_to_deg(g[piexif.GPSIFD.GPSLatitude],
                              g.get(piexif.GPSIFD.GPSLatitudeRef, b"N"))
            lon = _dms_to_deg(g[piexif.GPSIFD.GPSLongitude],
                              g.get(piexif.GPSIFD.GPSLongitudeRef, b"E"))
            alt = g.get(piexif.GPSIFD.GPSAltitude)
            alt = alt[0] / alt[1] if alt else None
            coords.append((p.name, lat, lon, alt))
        except Exception:
            missing += 1

    if missing:
        bad(f"{missing} кадрів без GPS EXIF")
    if not coords:
        bad("GPS не записано в жоден кадр — spatial matcher не працюватиме")
        return
    say(OK, f"GPS присутній у {len(coords)}/{len(frames)} кадрів")

    lats = [c[1] for c in coords]
    lons = [c[2] for c in coords]
    lat0 = sum(lats) / len(lats)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    span_x = (max(lons) - min(lons)) * m_per_deg_lon
    span_y = (max(lats) - min(lats)) * m_per_deg_lat
    say(INFO, f"центр треку: {lat0:.6f}, {sum(lons)/len(lons):.6f}")
    say(INFO, f"розмах треку: {span_x:.1f} м × {span_y:.1f} м")

    if span_x < 1.0 and span_y < 1.0:
        bad("трек вироджений (усі кадри в одній точці) — "
            "spatial matching безглуздий, використовуй exhaustive")
    alts = [c[3] for c in coords if c[3] is not None]
    if alts:
        say(INFO, f"висота: {min(alts):.1f}–{max(alts):.1f} м")

    say(INFO, "ПРИМІТКА: цей GPS синтетичний (коло навколо точки), "
              "якщо не підключено реальну телеметрію —")
    say(INFO, "  на реконструкцію він не впливає, лише на spatial matcher "
              "і геореєстрацію.")

    if verbose:
        for name, la, lo, al in coords[:5]:
            say(INFO, f"    {name}: {la:.6f}, {lo:.6f}, {al}")


# ══════════════════════════════════════════════════════════════════════
# 4. COLMAP DATABASE — фічі та матчі
# ══════════════════════════════════════════════════════════════════════
def check_database(db_path: Path, verbose=False):
    header(f"4. COLMAP DATABASE  —  {db_path}")
    if not db_path.is_file():
        bad("database.db не існує (SfM не запускався?)")
        return

    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    def tables():
        return {r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}

    tbls = tables()
    n_images = cur.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    say(INFO, f"зображень у БД: {n_images}")

    # --- keypoints per image ---
    if "keypoints" in tbls:
        rows = cur.execute("SELECT image_id, rows FROM keypoints").fetchall()
        counts = [r[1] for r in rows]
        if counts:
            counts_sorted = sorted(counts)
            med = counts_sorted[len(counts_sorted) // 2]
            say(OK if med > 1000 else WARN,
                f"фічі на кадр: min={min(counts)}  median={med}  max={max(counts)}")
            if med < 1000:
                bad("замало фіч — типово для синтетичних/гладких текстур "
                    "(дахи, вода, розмита рослинність)")
                say(INFO, "  спробуй: --set sfm_max_image_size=-1 (нативна роздільність)")
            weak = [r[0] for r in rows if r[1] < 200]
            if weak:
                warn(f"{len(weak)} кадрів мають <200 фіч — вони майже напевно "
                     f"не зареєструються")
        else:
            bad("таблиця keypoints порожня — feature_extractor не відпрацював")

    # --- raw matches ---
    if "matches" in tbls:
        pairs = cur.execute(
            "SELECT COUNT(*) FROM matches WHERE rows > 0").fetchone()[0]
        say(INFO, f"пар із сирими матчами: {pairs}")

    # --- geometrically verified — ЦЕ головне число ---
    if "two_view_geometries" in tbls:
        rows = cur.execute(
            "SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0"
        ).fetchall()
        say(OK if len(rows) > 0 else BAD,
            f"пар після geometric verification: {len(rows)}")

        if not rows:
            bad("ЖОДНОЇ верифікованої пари — реконструкція неможлива в принципі")
            bad("  причина майже завжди: замало фіч, або кадри не перетинаються")
        else:
            inliers = [r[1] for r in rows]
            s = sorted(inliers)
            say(INFO, f"inliers на пару: min={min(s)}  median={s[len(s)//2]}  max={max(s)}")

            # скільки кадрів узагалі бере участь хоч в одній парі
            seen = set()
            for pair_id, _ in rows:
                seen.add(pair_id // 2147483647)
                seen.add(pair_id % 2147483647)
            say(OK if len(seen) >= n_images * 0.9 else BAD,
                f"кадрів, задіяних хоч в одній верифікованій парі: "
                f"{len(seen)}/{n_images}")
            if len(seen) < n_images:
                orphan = n_images - len(seen)
                bad(f"{orphan} кадрів не мають ЖОДНОЇ верифікованої пари — "
                    f"вони випадуть із реконструкції")

            # чи є ланцюг (потрібно для track length > 2)
            deg: Counter = Counter()
            for pair_id, _ in rows:
                deg[pair_id // 2147483647] += 1
                deg[pair_id % 2147483647] += 1
            if deg:
                d = sorted(deg.values())
                say(INFO, f"пар на кадр: min={d[0]}  median={d[len(d)//2]}  max={d[-1]}")
                if d[len(d) // 2] < 2:
                    bad("медіана <2 пар на кадр → track length не перевищить 2 "
                        "→ stereo_fusion дасть порожній PLY")
    con.close()


# ══════════════════════════════════════════════════════════════════════
# 5. SPARSE MODEL
# ══════════════════════════════════════════════════════════════════════
def _read_points3d_count(path: Path):
    """Мінімальний парсер points3D.bin: кількість точок + довжини треків."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        track_lens = []
        for _ in range(n):
            f.read(8 + 24 + 3 + 8)          # id, xyz, rgb, error
            tl = struct.unpack("<Q", f.read(8))[0]
            f.read(tl * 8)                   # track elements
            track_lens.append(tl)
    return n, track_lens


def check_sparse(sparse_dir: Path, n_input_images: int | None):
    header(f"5. SPARSE MODEL  —  {sparse_dir}")
    if not sparse_dir.is_dir():
        bad("теки sparse/ не існує")
        return

    submodels = sorted(d for d in sparse_dir.iterdir() if d.is_dir())
    if not submodels:
        bad("жодної підмоделі — mapper не побудував реконструкцію")
        return
    if len(submodels) > 1:
        bad(f"РЕКОНСТРУКЦІЯ ФРАГМЕНТОВАНА: {len(submodels)} підмоделей "
            f"({[d.name for d in submodels]})")
        bad("  далі по пайплайну піде ЛИШЕ sparse/0 — решта кадрів втрачені")

    for sub in submodels:
        pts = sub / "points3D.bin"
        imgs = sub / "images.bin"
        if not pts.is_file():
            bad(f"{sub.name}: немає points3D.bin")
            continue

        try:
            n_pts, tracks = _read_points3d_count(pts)
        except Exception as e:
            bad(f"{sub.name}: не читається points3D.bin: {e}")
            continue

        # кількість зареєстрованих кадрів
        try:
            with open(imgs, "rb") as f:
                n_reg = struct.unpack("<Q", f.read(8))[0]
        except Exception:
            n_reg = None

        mean_track = sum(tracks) / len(tracks) if tracks else 0.0
        print()
        say(INFO, f"── {sub.name} ──")
        if n_reg is not None:
            frac = (n_reg / n_input_images) if n_input_images else 1.0
            say(OK if frac >= 0.9 else BAD,
                f"зареєстровано кадрів: {n_reg}"
                + (f" / {n_input_images} ({frac*100:.0f}%)" if n_input_images else ""))
            if n_input_images and frac < 0.6:
                bad("  <60% — модель непридатна для dense MVS")
        say(OK if n_pts > 5000 else WARN, f"3D-точок: {n_pts:,}")
        say(OK if mean_track >= 3.0 else BAD,
            f"середня довжина треку: {mean_track:.3f}")

        if mean_track < 3.0:
            bad("  track length < 3 → stereo_fusion не матиме що зливати")
            bad("  (fusion потребує підтвердження точки з ≥3 ракурсів)")
        if tracks:
            c = Counter(tracks)
            say(INFO, "  розподіл треків: "
                      + "  ".join(f"{k}:{c[k]}" for k in sorted(c)[:6]))


# ══════════════════════════════════════════════════════════════════════
# 6. DENSE
# ══════════════════════════════════════════════════════════════════════
def _ply_vertex_count(path: Path):
    """Читає лише header PLY — не вантажить файл у пам'ять."""
    with open(path, "rb") as f:
        n = None
        for _ in range(60):
            line = f.readline()
            if not line:
                break
            s = line.decode("ascii", "replace").strip()
            if s.startswith("element vertex"):
                n = int(s.split()[-1])
            if s == "end_header":
                break
    return n


def check_dense(dense_dir: Path, n_reg_hint=None):
    header(f"6. DENSE MVS  —  {dense_dir}")
    if not dense_dir.is_dir():
        say(INFO, "теки dense/ немає (стадія dense не запускалась)")
        return

    imgs = dense_dir / "images"
    if imgs.is_dir():
        n = len(list(imgs.iterdir()))
        say(INFO, f"undistorted images: {n}")
    dm = dense_dir / "stereo" / "depth_maps"
    if dm.is_dir():
        geo = list(dm.glob("*.geometric.bin"))
        pho = list(dm.glob("*.photometric.bin"))
        say(INFO, f"depth maps: {len(geo)} geometric, {len(pho)} photometric")
        if not geo and pho:
            bad("є лише photometric, а stereo_fusion за замовчуванням "
                "читає geometric → PLY буде порожній")
            say(INFO, "  причина: --PatchMatchStereo.geom_consistency false")

    ply = dense_dir / "fused.ply"
    if not ply.is_file():
        say(INFO, "fused.ply ще не створено")
        return
    n = _ply_vertex_count(ply)
    size_mb = ply.stat().st_size / 1e6
    say(OK if (n or 0) > 1000 else BAD,
        f"fused.ply: {n:,} вершин, {size_mb:.1f} MB" if n is not None
        else f"fused.ply: header не розпізнано, {size_mb:.1f} MB")
    if n == 0:
        bad("ПОРОЖНІЙ PLY — stereo_fusion не знайшов узгоджених точок")
        bad("  дивись розділ 5: track length та кількість зареєстрованих кадрів")


# ══════════════════════════════════════════════════════════════════════
# 7. .pt CLOUDS
# ══════════════════════════════════════════════════════════════════════
def check_pt(paths: list[Path]):
    header("7. .pt ФАЙЛИ")
    try:
        import torch
    except ImportError:
        warn("torch не встановлено — пропускаю")
        return
    any_found = False
    for p in paths:
        if not p.is_file():
            continue
        any_found = True
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            bad(f"{p.name}: не читається ({e})")
            continue
        if not isinstance(d, dict):
            say(INFO, f"{p.name}: {type(d).__name__}")
            continue
        keys = ", ".join(sorted(d.keys()))
        say(INFO, f"── {p.name} ──  ключі: {keys}")
        xyz = d.get("xyz")
        if xyz is not None and hasattr(xyz, "shape"):
            n = xyz.shape[0]
            say(OK if n > 1000 else BAD, f"   точок: {n:,}")
            if n:
                mn = xyz.min(0).values.tolist()
                mx = xyz.max(0).values.tolist()
                ext = [b - a for a, b in zip(mn, mx)]
                say(INFO, f"   bbox min: [{', '.join(f'{v:.2f}' for v in mn)}]")
                say(INFO, f"   bbox max: [{', '.join(f'{v:.2f}' for v in mx)}]")
                say(INFO, f"   розмір:   [{', '.join(f'{v:.2f}' for v in ext)}]")
                if max(ext) > 0 and min(ext) / max(ext) < 0.01:
                    warn("   сцена майже пласка/вироджена по одній осі")
    if not any_found:
        say(INFO, "жодного .pt ще не створено")


# ══════════════════════════════════════════════════════════════════════
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Діагностика даних пайплайну AeroSplat-GIS")
    ap.add_argument("--dataset", default="dataset_v6")
    ap.add_argument("--footage-dir", default=None,
                    help="за замовчуванням data/<dataset>/footage")
    ap.add_argument("--stage", default="all",
                    choices=["all", "footage", "intrinsics", "exif",
                             "database", "sparse", "dense", "pt"])
    ap.add_argument("--image-width", type=int, default=1920,
                    help="що зараз стоїть у RunConfig (aerosplat/config.py)")
    ap.add_argument("--image-height", type=int, default=1080)
    ap.add_argument("--fov-deg", type=float, default=60.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    ds = args.dataset
    footage = Path(args.footage_dir) if args.footage_dir \
        else DATA_DIR / ds / "footage"
    if not footage.is_absolute():
        footage = PROJECT_DIR / footage
    geotagged = DATA_DIR / f"raw_images_{ds}"
    colmap_out = DATA_DIR / f"colmap_output_{ds}"

    print(f"AeroSplat-GIS doctor — датасет '{ds}'")
    print(f"проєкт: {PROJECT_DIR}")

    want = args.stage
    info = {}
    n_frames = None

    if want in ("all", "footage", "intrinsics"):
        info = check_footage(footage, args.verbose)
        n_frames = info.get("count")
    if want in ("all", "intrinsics"):
        check_intrinsics(info, args.image_width, args.image_height, args.fov_deg)
    if want in ("all", "exif"):
        check_exif(geotagged, args.verbose)
    if want in ("all", "database"):
        check_database(colmap_out / "database.db", args.verbose)
    if want in ("all", "sparse"):
        check_sparse(colmap_out / "sparse", n_frames)
    if want in ("all", "dense"):
        check_dense(colmap_out / "dense")
    if want in ("all", "pt"):
        check_pt([DATA_DIR / f"{ds}_cloud.pt",
                  DATA_DIR / f"{ds}_dense.pt",
                  DATA_DIR / f"{ds}_splat.pt"])

    header("ПІДСУМОК")
    if not _problems:
        say(OK, "проблем не виявлено")
    else:
        print(f"знайдено проблем: {len(_problems)}\n")
        for i, p in enumerate(_problems, 1):
            print(f"  {i}. {p}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())