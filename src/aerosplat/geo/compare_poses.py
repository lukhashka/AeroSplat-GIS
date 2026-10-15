"""Порівняння поз: відомі (Earth Studio) vs уточнені COLMAP BA (images.txt).

Логіка з tools/compare_poses.py, винесена в бібліотеку, щоб її міг викликати
evaluate (ΔR°, Δt м у metrics.json). tools/compare_poses.py лишається CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def parse_images_txt(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    poses: Dict[str, Dict[str, np.ndarray]] = {}
    lines = [l for l in Path(path).read_text(encoding="utf-8").splitlines()
             if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if len(parts) < 10:
            continue
        poses[parts[9]] = {"q": np.array(parts[1:5], dtype=float),
                           "t": np.array(parts[5:8], dtype=float)}
    return poses


def quat_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def camera_center(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """images.txt зберігає world→camera (R, t); центр камери C = -Rᵀ t."""
    return -quat_to_R(q).T @ t


def pose_errors(known: Dict[str, Any], refined: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Похибки за спільними кадрами: позиція (м) і обертання (°). None якщо немає спільних."""
    common = sorted(set(known) & set(refined))
    if not common:
        return None
    pos, ang = [], []
    for name in common:
        pos.append(np.linalg.norm(camera_center(known[name]["q"], known[name]["t"])
                                  - camera_center(refined[name]["q"], refined[name]["t"])))
        r_diff = quat_to_R(known[name]["q"]).T @ quat_to_R(refined[name]["q"])
        ang.append(np.degrees(np.arccos(np.clip((np.trace(r_diff) - 1) / 2, -1, 1))))
    pos_a, ang_a = np.array(pos), np.array(ang)
    return {
        "n_compared": len(common), "n_known": len(known), "n_refined": len(refined),
        "pos_mean_m": float(pos_a.mean()), "pos_median_m": float(np.median(pos_a)),
        "pos_max_m": float(pos_a.max()),
        "rot_mean_deg": float(ang_a.mean()), "rot_median_deg": float(np.median(ang_a)),
        "rot_max_deg": float(ang_a.max()),
        "worst": [{"name": common[i], "pos_m": float(pos_a[i]), "rot_deg": float(ang_a[i])}
                  for i in np.argsort(-pos_a)[:5]],
    }


def compare_pose_files(known_path: Path, refined_path: Path) -> Optional[Dict[str, Any]]:
    """None, якщо якийсь із файлів відсутній (напр. COLMAP зберіг лише .bin) або немає спільних кадрів."""
    if not Path(known_path).is_file() or not Path(refined_path).is_file():
        return None
    return pose_errors(parse_images_txt(known_path), parse_images_txt(refined_path))
