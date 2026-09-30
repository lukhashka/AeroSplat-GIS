"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      TRIPS — TRILINEAR POINT SPLATTING — DROP-IN ЗАМІНА gs_trainer_v01      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  Franke et al. 2024, https://arxiv.org/abs/2401.06003                        ║
║                                                                                ║
║  КЛЮЧОВА ВІДМІННІСТЬ ВІД 3DGS (gs_trainer_v01.py):                          ║
║                                                                                ║
║  3DGS:  точка = анізотропний 3D-гаусіан з коваріацією Σ = R·S·Sᵀ·Rᵀ        ║
║         (6 параметрів форми: 3 масштаби + кватерніон),                       ║
║         рендер = exp(-½·(x-μ)ᵀΣ'⁻¹(x-μ))  — м'яке радіальне спадання       ║
║                                                                                ║
║  TRIPS: точка = ІЗОТРОПНА сфера з одним скалярним розміром s_world,          ║
║         рендер = проєкція в screen space + вибір рівня піраміди +           ║
║         трилінійний (білінійний-в-XY × лінійний-між-рівнями) запис          ║
║                                                                                ║
║  МАТЕМАТИКА (та сама, що в trips_core.py, але інтегрована в             ║
║  ту саму структуру: params-dict, view_matrix, GaussianDensifier-подібний    ║
║  клас, create_optimizer, photometric_loss):                                  ║
║                                                                                ║
║    s = focal · s_world / z                                                   ║
║    L = log2(s),  L_lo = ⌊L⌋,  L_hi = ⌈L⌉,  ι = frac(L)                    ║
║    γ = β · ι · α       (β — білінійна вага в межах шару)                    ║
║                                                                                ║
║  ЧОГО НЕМАЄ В ЦІЙ РЕАЛІЗАЦІЇ (порівняно з оригінальною статтею):            ║
║    - Немає нейромережі-реконструктора (gated conv U-Net), яка в TRIPS       ║
║      прибирає дірки й шум після растеризації.                               ║
║    - Немає custom CUDA-кернелів — чистий PyTorch, O(N · H · W) як і         ║
║      PureTorchRasterizer в оригінальному gs_trainer_v01.                    ║
║    - Alpha-compositing між рівнями піраміди спрощений (без честивого        ║
║      depth-sort між рівнями, тільки в межах кожного рівня) — позначено      ║
║      коментарем нижче, це прийнятний компроміс для навчальної/дослідної     ║
║      версії, але не production-точний.                                       ║
║                                                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple, Optional

# Representation-agnostic частини переїжджають без змін з gs_trainer_v01:
# photometric_loss, l1_loss, ssim_loss не залежать від того, як точка
# рендериться, тому імпортуємо їх напряму, а не дублюємо.
from gs_trainer_v3 import l1_loss, ssim_loss, photometric_loss

# ═══════════════════════════════════════════════════════════════════
#  TRILINEAR POINT MODEL  (заміна GaussianModel)
# ═══════════════════════════════════════════════════════════════════

class TrilinearPointModel(nn.Module):
    """
    Параметрична модель сцени для TRIPS.

    На відміну від GaussianModel, тут НЕМАЄ _scaling (3 числа) та
    _rotation (кватерніон) — замість анізотропної форми кожна точка
    має один скалярний world-space розмір _size. Це прямий наслідок
    того, що TRIPS не моделює форму гаусіана: розмір визначає лише
    те, у який шар піраміди потрапить точка (s = focal·s_world/z),
    а форма спалту в екранному просторі задається кернелем
    растерізатора, не параметрами точки.
    """

    def __init__(self, init_xyz: Tensor, init_rgb: Tensor, init_size: float = 0.02):
        super().__init__()
        N = init_xyz.shape[0]
        print(f"  Ініціалізація {N:,} TRIPS-точок (trilinear splats)...")

        # μ — позиції центрів (те саме, що в GaussianModel)
        self._xyz = nn.Parameter(init_xyz.clone().float())

        # log(s_world) — скалярний розмір точки в лог-просторі для стабільності
        # (аналог _scaling, але 1 число замість 3 + кватерніон)
        init_log_size = torch.log(torch.ones(N, 1) * init_size)
        self._size = nn.Parameter(init_log_size)

        # α — opacity через logit, як в GaussianModel
        self._opacity = nn.Parameter(torch.ones(N, 1) * 0.5)

        # c — колір RGB через logit
        rgb = init_rgb.clone().float()
        rgb = rgb / 255.0 if rgb.max() > 1.0 else rgb
        rgb = rgb.clamp(1e-4, 1 - 1e-4)
        self._rgb = nn.Parameter(torch.logit(rgb))

    def get_params(self) -> Dict[str, Tensor]:
        """Повертає активовані параметри, готові для рендерингу."""
        return {
            "xyz":        self._xyz,
            "size_world": torch.exp(self._size).squeeze(-1),   # (N,) — заміна "scale"+"rotation"+"cov3d"
            "opacity":    torch.sigmoid(self._opacity).squeeze(-1),
            "rgb":        torch.sigmoid(self._rgb),
        }


# ═══════════════════════════════════════════════════════════════════
#  TRILINEAR POINT RASTERIZER  (заміна PureTorchRasterizer)
# ═══════════════════════════════════════════════════════════════════

class TrilinearRasterizer:
    """
    Диференційований растерізатор TRIPS на чистому PyTorch.

    Алгоритм (див. заголовок файлу для формул):
      1. Трансформуємо точки в camera space (ідентично GS-версії)
      2. Проєктуємо центри в 2D: u,v (ідентично GS-версії)
      3. Рахуємо s = focal·s_world/z → неперервний рівень піраміди L
      4. Точки групуються по цілому рівню L_lo/L_hi, кожна група
         скейлиться до відповідної роздільної здатності і пишеться
         білінійно (β) через index_add (scatter), вага множиться на
         інтер-рівневу вагу ι = frac(L) (чи 1-ι для L_lo)
      5. Кожен рівень апсемплюється назад до full-res і комбінується
         front-to-back по накопиченій прозорості (те саме T, як у GS)

    ⚠️ Складність приблизно O(N + Σ_levels H_l·W_l) — це ЗНАЧНО дешевше
       за O(N·H·W) в PureTorchRasterizer, бо тут немає циклу "кожна
       точка проти кожного пікселя": скейл точки одразу визначає, в
       який (маленький) буфер вона пишеться. Це головна практична
       перевага портування на TRIPS у цьому проєкті — навіть без
       CUDA, чистий PyTorch варіант тут швидший за GS-варіант з
       тим самим числом точок.
    """

    def __init__(self, H: int, W: int, fx: float, fy: float,
                 cx: float, cy: float, num_levels: int = 5):
        self.H, self.W = H, W
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.focal = 0.5 * (fx + fy)
        self.num_levels = num_levels
        self.level_sizes = [
            (max(4, H >> l), max(4, W >> l)) for l in range(num_levels)
        ]

    def render(
        self,
        params: Dict[str, Tensor],
        view_matrix: Tensor,
        bg_color: Tensor,
    ) -> Tensor:
        device = params["xyz"].device
        H, W = self.H, self.W

        xyz_world = params["xyz"]          # (N, 3)
        rgb       = params["rgb"]          # (N, 3)
        opacity   = params["opacity"]      # (N,)
        size_w    = params["size_world"]   # (N,)

        # ── 1. World → Camera (ідентично GS-версії) ──────────────────
        N = xyz_world.shape[0]
        ones = torch.ones(N, 1, device=device)
        xyz_h = torch.cat([xyz_world, ones], dim=-1)
        xyz_cam = (view_matrix @ xyz_h.T).T[:, :3]

        mask = xyz_cam[:, 2] > 0.1
        xyz_cam  = xyz_cam[mask]
        rgb_m    = rgb[mask]
        opacity_m = opacity[mask]
        size_m   = size_w[mask]

        if xyz_cam.shape[0] == 0:
            return bg_color.expand(H, W, 3).clone()

        # ── 2. Проєкція центрів у 2D (ідентично GS-версії) ───────────
        z = xyz_cam[:, 2].clamp(min=1e-4)
        u = self.fx * xyz_cam[:, 0] / z + self.cx
        v = self.fy * xyz_cam[:, 1] / z + self.cy

        # ── 3. TRIPS-специфіка: розмір на екрані → рівень піраміди ──
        s = (self.focal * size_m / z).clamp(min=1.0, max=float(min(H, W)))
        L = torch.log2(s).clamp(0, self.num_levels - 1 - 1e-4)
        L_lo = torch.floor(L).long()
        L_hi = (L_lo + 1).clamp(max=self.num_levels - 1)
        iota = L - L_lo.float()

        # ── 4. Сортування за глибиною front-to-back ───────────────────
        sort_idx = torch.argsort(z)
        u, v       = u[sort_idx], v[sort_idx]
        rgb_m      = rgb_m[sort_idx]
        opacity_m  = opacity_m[sort_idx]
        L_lo, L_hi = L_lo[sort_idx], L_hi[sort_idx]
        iota       = iota[sort_idx]

        final_color = bg_color.expand(H, W, 3).contiguous().clone()
        final_alpha = torch.zeros(H, W, device=device)

        for lvl in range(self.num_levels):
            w_lo = torch.where(L_lo == lvl, 1 - iota, torch.zeros_like(iota))
            w_hi = torch.where(L_hi == lvl, iota, torch.zeros_like(iota))
            level_weight = w_lo + w_hi
            active = level_weight > 1e-6
            if active.sum() == 0:
                continue

            h, w = self.level_sizes[lvl]
            scale = h / H
            xi = (u[active] * scale)
            yi = (v[active] * scale)
            a_active = (opacity_m[active] * level_weight[active]).clamp(0, 1)
            c_active = rgb_m[active]

            x0 = torch.floor(xi).long()
            y0 = torch.floor(yi).long()
            fx_ = xi - x0.float()
            fy_ = yi - y0.float()

            color_buf = torch.zeros(h, w, 3, device=device)
            alpha_buf = torch.zeros(h, w, device=device)
            weight_buf = torch.zeros(h, w, device=device) + 1e-6

            for dx, dy, beta in [
                (0, 0, (1 - fx_) * (1 - fy_)),
                (1, 0, fx_ * (1 - fy_)),
                (0, 1, (1 - fx_) * fy_),
                (1, 1, fx_ * fy_),
            ]:
                xci = (x0 + dx).clamp(0, w - 1)
                yci = (y0 + dy).clamp(0, h - 1)
                idx = yci * w + xci
                gamma = beta * a_active   # γ = β · ι(вже в a_active) · α(вже в a_active)

                flat_color = color_buf.view(-1, 3)
                flat_alpha = alpha_buf.view(-1)
                flat_weight = weight_buf.view(-1)
                flat_color.index_add_(0, idx, c_active * gamma.unsqueeze(-1))
                flat_alpha.index_add_(0, idx, gamma)
                flat_weight.index_add_(0, idx, gamma)

            color_buf = color_buf / weight_buf.unsqueeze(-1).clamp(min=1e-6)
            alpha_buf = alpha_buf.clamp(0, 1)

            # ── 5. Апсемпл рівня і front-to-back комбінування ─────────
            c_up = F.interpolate(color_buf.permute(2, 0, 1).unsqueeze(0),
                                  size=(H, W), mode='bilinear', align_corners=False)[0].permute(1, 2, 0)
            a_up = F.interpolate(alpha_buf.unsqueeze(0).unsqueeze(0),
                                  size=(H, W), mode='bilinear', align_corners=False)[0, 0]

            T = (1 - final_alpha).clamp(min=0)
            final_color = final_color * 1.0  # no-op, keeps graph clean
            final_color = final_color + (c_up - bg_color) * a_up.unsqueeze(-1) * T.unsqueeze(-1)
            final_alpha = (final_alpha + a_up * T).clamp(0, 1)

        return final_color.clamp(0, 1)


# ═══════════════════════════════════════════════════════════════════
#  DENSIFICATION  (заміна GaussianDensifier)
# ═══════════════════════════════════════════════════════════════════

class TrilinearDensifier:
    """
    Той самий принцип, що GaussianDensifier: клонуємо точки з великим
    градієнтом позиції в "недореконструйованих" зонах, ріжемо прозорі.

    Відмінність від GS-версії: немає split-логіки за розміром гаусіана
    (бо форма ізотропна) — замість clone/split є тільки clone/prune,
    і замість перевірки max_scale перевіряємо просто розмір s_world.
    Це трохи простіша евристика, ніж в оригіналі TRIPS (де densification
    керується щільністю початкової MVS-хмари), але зберігає той самий
    градієнт-based сигнал "куди бракує точок".
    """

    def __init__(self, model: TrilinearPointModel, grad_threshold: float = 0.0002):
        self.model = model
        self.grad_threshold = grad_threshold
        self._reset_stats()

    def _reset_stats(self):
        N = self.model._xyz.shape[0]
        device = self.model._xyz.device
        self.xyz_grad_accum = torch.zeros(N, device=device)
        self.denom = torch.zeros(N, device=device)

    def update_stats(self):
        if self.model._xyz.grad is not None:
            grad_norm = self.model._xyz.grad.norm(dim=-1)
            self.xyz_grad_accum += grad_norm
            self.denom += 1

    def densify_and_prune(
        self,
        optimizer: torch.optim.Optimizer,
        opacity_threshold: float = 0.005,
        max_points: int = 1_000_000,
    ):
        avg_grad = self.xyz_grad_accum / (self.denom + 1e-8)
        N_current = self.model._xyz.shape[0]
        if avg_grad.shape[0] != N_current:
            avg_grad = avg_grad[:N_current]

        clone_mask = (avg_grad > self.grad_threshold)
        clone_mask = clone_mask[:N_current]

        with torch.no_grad():
            if clone_mask.any() and self.model._xyz.shape[0] < max_points:
                self._clone_points(clone_mask, optimizer)

            keep_mask = ~(torch.sigmoid(self.model._opacity).squeeze() < opacity_threshold)
            keep_mask = keep_mask[: self.model._xyz.shape[0]]
            if not keep_mask.all():
                self._prune_points(keep_mask, optimizer)

            print(f"  Densification (TRIPS): clone={int(clone_mask.sum())}, "
                  f"prune={int((~keep_mask).sum())}, total={self.model._xyz.shape[0]:,}")

        self._reset_stats()

    def _clone_points(self, mask: Tensor, optimizer: torch.optim.Optimizer):
        new_xyz     = self.model._xyz[mask].detach()
        new_size    = self.model._size[mask].detach()
        new_opacity = self.model._opacity[mask].detach()
        new_rgb     = self.model._rgb[mask].detach()
        self._concat_params(new_xyz, new_size, new_opacity, new_rgb, optimizer)

    def _concat_params(self, xyz, size, opacity, rgb, optimizer):
        param_dict = {"_xyz": xyz, "_size": size, "_opacity": opacity, "_rgb": rgb}
        for name, new_tensor in param_dict.items():
            old_param = getattr(self.model, name)
            new_full = torch.cat([old_param.data, new_tensor], dim=0)
            setattr(self.model, name, nn.Parameter(new_full))

            for group in optimizer.param_groups:
                if group.get("name") == name:
                    old_state = optimizer.state.get(group["params"][0], {})
                    new_param = getattr(self.model, name)
                    group["params"][0] = new_param
                    if "exp_avg" in old_state:
                        zeros = torch.zeros_like(new_tensor)
                        old_state["exp_avg"] = torch.cat([old_state["exp_avg"], zeros])
                        old_state["exp_avg_sq"] = torch.cat([old_state["exp_avg_sq"], zeros])
                    optimizer.state[new_param] = old_state

        n_new = xyz.shape[0]
        device = xyz.device
        self.xyz_grad_accum = torch.cat([self.xyz_grad_accum, torch.zeros(n_new, device=device)])
        self.denom = torch.cat([self.denom, torch.zeros(n_new, device=device)])

    def _prune_points(self, keep_mask: Tensor, optimizer: torch.optim.Optimizer):
        for name in ["_xyz", "_size", "_opacity", "_rgb"]:
            old_param = getattr(self.model, name)
            new_data = old_param.data[keep_mask]
            setattr(self.model, name, nn.Parameter(new_data))
            for group in optimizer.param_groups:
                if group.get("name") == name:
                    old_state = optimizer.state.get(group["params"][0], {})
                    new_param = getattr(self.model, name)
                    group["params"][0] = new_param
                    if "exp_avg" in old_state:
                        old_state["exp_avg"] = old_state["exp_avg"][keep_mask]
                        old_state["exp_avg_sq"] = old_state["exp_avg_sq"][keep_mask]
                    optimizer.state[new_param] = old_state

        self.xyz_grad_accum = self.xyz_grad_accum[keep_mask]
        self.denom = self.denom[keep_mask]


# ═══════════════════════════════════════════════════════════════════
#  OPTIMIZER  (заміна create_optimizer)
# ═══════════════════════════════════════════════════════════════════

def create_optimizer(model: TrilinearPointModel) -> torch.optim.Adam:
    """
    Ті самі lr, що в GS-версії для xyz/opacity/rgb (щоб порівняння було
    чесним — ми не тюнимо гіперпараметри на користь одного методу).
    lr для _size замість _scaling — трохи вищий, бо тут це один скаляр,
    а не 3-вимірний масштаб + кватерніон, тож градієнт "гучніший".
    """
    return torch.optim.Adam([
        {"params": [model._xyz],     "lr": 0.00016, "name": "_xyz"},
        {"params": [model._size],    "lr": 0.008,   "name": "_size"},
        {"params": [model._opacity], "lr": 0.05,    "name": "_opacity"},
        {"params": [model._rgb],     "lr": 0.0025,  "name": "_rgb"},
    ], eps=1e-15)


# Аліаси для сумісності з ім'юваннями в colmap_loader_v01.py, який
# імпортує "GaussianModel, PureTorchRasterizer, GaussianDensifier".
# Якщо не хочеш змінювати train_with_real_data взагалі — розкоментуй:
#
GaussianModel      = TrilinearPointModel
PureTorchRasterizer = TrilinearRasterizer
GaussianDensifier   = TrilinearDensifier