"""Оцінка якості 3DGS на test-кадрах: PSNR / SSIM / LPIPS, FPS, розмір, пози.

CLI (для вже готового splat.pt; той самий test-спліт, що в тренері):
    python -m aerosplat.gs.evaluate --splat data/ds_splat.pt --colmap <sparse> \
        --images <dir> --out-dir data/ds/run --eval-every 8
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from aerosplat.gs.sh import sh_degree_of
from aerosplat.gs.trainer import (
    CudaRasterizer, TileRasterizer, attach_images, build_covariance_3d,
    load_colmap_cameras, log, psnr, split_cameras, ssim_loss, Camera,
)


def load_scene(path: str, device) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
    """splat.pt (активовані значення, ADR-0002) → (params для растерізатора, `render` з файлу).

    `sh` (gsplat-тренер) потрапляє в params; Inria/torch-растерізатори його
    ігнорують і малюють `rgb`. `render` — як сплат рендерили під час тренування
    ({} для файлів Inria/torch-тренера).
    """
    scene = torch.load(path, map_location="cpu", weights_only=False)
    act = scene.get("activations", {})
    if act.get("scale_is_log") or act.get("opacity_is_logit"):
        raise ValueError("splat.pt з неактивованими значеннями не підтримується цим оцінювачем")
    keys = ("xyz", "scale", "rotation", "opacity", "rgb") + (("sh",) if "sh" in scene else ())
    p = {k: scene[k].to(device).float() for k in keys}
    p["cov3d"] = build_covariance_3d(torch.log(p["scale"]), p["rotation"])
    return p, dict(scene.get("render") or {})


def load_scene_params(path: str, device) -> Dict[str, Tensor]:
    return load_scene(path, device)[0]


def _make_lpips(device):
    """LPIPS опційний: без пакета (або ваг) повертаємо None, метрика = null."""
    try:
        import lpips
        return lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    except Exception as e:  # ImportError, відсутні ваги, немає мережі
        log(f"  ⚠️  LPIPS недоступний ({type(e).__name__}) — метрика буде null "
            f"(pip install lpips)")
        return None


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def mean_test_psnr(params: Dict[str, Tensor], rasterizer, cams: List[Camera], device) -> float:
    """Швидка метрика для логів тренування: лише середній PSNR."""
    bg = torch.zeros(3, device=device)
    vals = []
    for cam in cams:
        img, _ = rasterizer.render(params, cam, cam.view_matrix(device), bg)
        assert cam.image is not None
        vals.append(psnr(img, cam.image.to(device)))
    return sum(vals) / len(vals)


@torch.no_grad()
def evaluate_views(params: Dict[str, Tensor], rasterizer, cams: List[Camera], device,
                   lpips_fn=None, keep_images: bool = False) -> Dict[str, Any]:
    """Рендерить кожен test-кадр і рахує метрики. Повертає словник у форматі metrics.json."""
    if not cams:
        raise ValueError("Порожній test-набір")
    bg = torch.zeros(3, device=device)
    per_view: List[Dict[str, Any]] = []
    images: Dict[str, Dict[str, Tensor]] = {}
    render_time = 0.0

    # прогрів (CUDA-ядра / алокатор), щоб не псував FPS
    rasterizer.render(params, cams[0], cams[0].view_matrix(device), bg)

    for cam in cams:
        assert cam.image is not None
        gt = cam.image.to(device)
        _sync(device)
        t0 = time.perf_counter()
        pred, _ = rasterizer.render(params, cam, cam.view_matrix(device), bg)
        _sync(device)
        render_time += time.perf_counter() - t0

        pred_c, gt_c = pred.permute(2, 0, 1), gt.permute(2, 0, 1)
        row: Dict[str, Any] = {
            "name": cam.name,
            "psnr": round(psnr(pred, gt), 3),
            "ssim": round(1.0 - float(ssim_loss(pred_c, gt_c)), 4),
            "lpips": None,
        }
        if lpips_fn is not None:
            row["lpips"] = round(float(lpips_fn(pred_c.unsqueeze(0) * 2 - 1,
                                                gt_c.unsqueeze(0) * 2 - 1)), 4)
        per_view.append(row)
        if keep_images:
            images[cam.name] = {"render": pred.cpu(), "gt": gt.cpu()}

    def mean(key: str) -> Optional[float]:
        vals = [r[key] for r in per_view if r[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    best = max(per_view, key=lambda r: r["psnr"])
    worst = min(per_view, key=lambda r: r["psnr"])
    out: Dict[str, Any] = {
        "mean_psnr": mean("psnr"), "mean_ssim": mean("ssim"), "mean_lpips": mean("lpips"),
        "best_view": best, "worst_view": worst, "per_view": per_view,
        "n_test_views": len(per_view),
        "render_fps": round(len(cams) / max(render_time, 1e-9), 2),
    }
    if keep_images:
        out["_images"] = images
    return out


def _make_rasterizer(kind: str, device, render: Optional[Dict[str, Any]] = None):
    """Растерізатор оцінки. gsplat бере rasterize_mode із файлу (`render`), щоб test-кадри
    рендерились так само, як під час тренування; файл без `render` — classic (Inria)."""
    render = render or {}
    if kind == "gsplat":
        from aerosplat.gs.gsplat_backend import GsplatRasterizer
        return GsplatRasterizer(render.get("rasterize_mode", "classic"))
    if render.get("rasterizer") == "gsplat":
        log(f"  ⚠️  Сплат натреновано gsplat ({render.get('rasterize_mode')}), а оцінка йде "
            f"rasterizer={kind}: без SH і antialiasing метрики занижені.")
    if kind in ("auto", "cuda") and device.type == "cuda":
        try:
            return CudaRasterizer()
        except ImportError:
            if kind == "cuda":
                raise
    return TileRasterizer()


def run_evaluation(splat: str, colmap: str, images: str, out_dir: str, *,
                   eval_every: int = 8, max_res: int = 1024, rasterizer: str = "auto",
                   use_lpips: bool = True, manifest: Optional[str] = None,
                   known_poses: Optional[str] = None, n_report_views: int = 5) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cams = attach_images(load_colmap_cameras(colmap), images, max_res=max_res)
    _, test_cams = split_cameras(cams, eval_every)
    if not test_cams:
        raise SystemExit(f"❌ Замало кадрів ({len(cams)}) для test-спліту кожен {eval_every}-й")
    log(f"  Оцінка: {len(test_cams)} test-кадрів з {len(cams)}")

    params, render = load_scene(splat, device)
    raster = _make_rasterizer(rasterizer, device, render)
    lp = _make_lpips(device) if use_lpips else None
    result = evaluate_views(params, raster, test_cams, device, lpips_fn=lp, keep_images=True)
    imgs = result.pop("_images")

    system: Dict[str, Any] = {
        "n_gaussians": int(params["xyz"].shape[0]),
        "splat_size_mb": round(os.path.getsize(splat) / 1024 ** 2, 2),
        "rasterizer": type(raster).__name__,
        "rasterize_mode": getattr(raster, "rasterize_mode", "classic"),
        # SH is evaluated only by gsplat; the other rasterizers draw `rgb`.
        "sh_degree": (sh_degree_of(params["sh"].shape[1])
                      if rasterizer == "gsplat" and "sh" in params else 0),
        "device": str(device),
        "train_seconds": None,
    }
    if manifest and Path(manifest).is_file():
        rec = json.loads(Path(manifest).read_text(encoding="utf-8")).get("stages", {}).get("train")
        if rec:
            system["train_seconds"] = rec.get("seconds")
    result["system"] = system

    if known_poses:
        from aerosplat.geo.compare_poses import compare_pose_files
        pose_stats = compare_pose_files(Path(known_poses) / "images.txt",
                                        Path(colmap) / "images.txt")
        if pose_stats:
            result["poses"] = pose_stats

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    from aerosplat.gs.report import write_report
    write_report(out / "report.html", result, imgs, n_views=n_report_views)
    log(f"  PSNR {result['mean_psnr']} | SSIM {result['mean_ssim']} | "
        f"LPIPS {result['mean_lpips']} | {result['render_fps']} FPS")
    log(f"  Збережено: {out / 'metrics.json'}, {out / 'report.html'}")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Оцінка якості 3DGS на test-кадрах")
    ap.add_argument("--splat", required=True)
    ap.add_argument("--colmap", required=True, help="каталог COLMAP sparse")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eval-every", type=int, default=8)
    ap.add_argument("--max-res", type=int, default=1024)
    ap.add_argument("--rasterizer", choices=["auto", "cuda", "torch", "gsplat"], default="auto")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--manifest", default=None, help="manifest.json (час тренування)")
    ap.add_argument("--known-poses-dir", default=None,
                    help="каталог з відомими images.txt → ΔR°/Δt після BA")
    args = ap.parse_args(argv)
    run_evaluation(args.splat, args.colmap, args.images, args.out_dir,
                   eval_every=args.eval_every, max_res=args.max_res,
                   rasterizer=args.rasterizer, use_lpips=not args.no_lpips,
                   manifest=args.manifest, known_poses=args.known_poses_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
