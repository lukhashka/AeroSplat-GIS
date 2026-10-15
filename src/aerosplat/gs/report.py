"""Самодостатній report.html: таблиця метрик + render / GT / diff для обраних кадрів."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import Tensor


def _png_b64(img: Tensor) -> str:
    """(H,W,3) float [0,1] → base64 PNG."""
    from PIL import Image
    arr = (img.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _diff(render: Tensor, gt: Tensor, gain: float = 4.0) -> Tensor:
    return ((render - gt).abs() * gain).clamp(0, 1)


def pick_views(per_view: List[Dict[str, Any]], n: int) -> List[str]:
    """best + worst + рівномірно розставлені між ними (за PSNR), без дублікатів."""
    ordered = sorted(per_view, key=lambda r: r["psnr"])
    names = [ordered[0]["name"], ordered[-1]["name"]]
    if n > 2 and len(ordered) > 2:
        step = len(ordered) / (n - 1)
        names += [ordered[min(len(ordered) - 1, int(i * step))]["name"] for i in range(1, n - 1)]
    seen: List[str] = []
    for nm in names:
        if nm not in seen:
            seen.append(nm)
    return seen[:n]


def _fmt(v: Any, nd: int = 3) -> str:
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def write_report(path: Path, metrics: Dict[str, Any], images: Dict[str, Dict[str, Tensor]],
                 n_views: int = 5) -> None:
    sysm = metrics.get("system", {})
    rows = "".join(
        f"<tr><td>{html.escape(r['name'])}</td><td>{_fmt(r['psnr'], 2)}</td>"
        f"<td>{_fmt(r['ssim'], 4)}</td><td>{_fmt(r['lpips'], 4)}</td></tr>"
        for r in metrics["per_view"])
    cards = []
    for name in pick_views(metrics["per_view"], n_views):
        im = images.get(name)
        if im is None:
            continue
        cells = "".join(
            f"<figure><img src='data:image/png;base64,{_png_b64(t)}'><figcaption>{cap}"
            f"</figcaption></figure>"
            for cap, t in (("render", im["render"]), ("ground truth", im["gt"]),
                           ("|diff| ×4", _diff(im["render"], im["gt"]))))
        cards.append(f"<h3>{html.escape(name)}</h3><div class='row'>{cells}</div>")

    poses = metrics.get("poses")
    pose_html = ""
    if poses:
        pose_html = (f"<h2>Пози після BA</h2><p>ΔR mean {poses['rot_mean_deg']:.3f}°, "
                     f"max {poses['rot_max_deg']:.3f}° · Δt mean {poses['pos_mean_m']:.3f} м, "
                     f"max {poses['pos_max_m']:.3f} м ({poses['n_compared']} кадрів)</p>")

    doc = f"""<!doctype html><html lang="uk"><meta charset="utf-8">
<title>AeroSplat — звіт якості</title>
<style>body{{font:14px system-ui;margin:24px;max-width:1300px}}
table{{border-collapse:collapse}}td,th{{border:1px solid #8884;padding:4px 10px;text-align:right}}
td:first-child,th:first-child{{text-align:left}}.row{{display:flex;gap:8px;flex-wrap:wrap}}
figure{{margin:0;flex:1 1 30%}}img{{width:100%}}</style>
<h1>Звіт якості реконструкції</h1>
<table><tr><th>PSNR ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th><th>Test-кадрів</th><th>FPS</th>
<th>Гаусіанів</th><th>Файл, MB</th><th>Тренування, с</th></tr>
<tr><td>{_fmt(metrics['mean_psnr'], 2)}</td><td>{_fmt(metrics['mean_ssim'], 4)}</td>
<td>{_fmt(metrics['mean_lpips'], 4)}</td><td>{metrics['n_test_views']}</td>
<td>{_fmt(metrics['render_fps'], 1)}</td><td>{_fmt(sysm.get('n_gaussians'))}</td>
<td>{_fmt(sysm.get('splat_size_mb'))}</td><td>{_fmt(sysm.get('train_seconds'), 0)}</td></tr></table>
{pose_html}
<h2>Приклади (найгірший, найкращий, проміжні)</h2>{''.join(cards)}
<h2>Усі test-кадри</h2>
<table><tr><th>Кадр</th><th>PSNR</th><th>SSIM</th><th>LPIPS</th></tr>{rows}</table></html>"""
    Path(path).write_text(doc, encoding="utf-8")
