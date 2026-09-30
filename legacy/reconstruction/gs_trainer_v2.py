"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              3D GAUSSIAN SPLATTING — ТЕОРІЯ + РЕАЛІЗАЦІЯ                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ТЕОРІЯ (коротко)                                                            ║
║  ─────────────────                                                           ║
║                                                                              ║
║  3DGS представляє сцену як набір N 3D-Гаусіанів.                            ║
║  Кожен Гаусіан i має параметри:                                              ║
║                                                                              ║
║    μᵢ  ∈ ℝ³        — центр (позиція в 3D)                                  ║
║    Σᵢ  ∈ ℝ³ˣ³      — коваріаційна матриця (форма/орієнтація)               ║
║    αᵢ  ∈ (0,1)     — непрозорість (opacity)                                 ║
║    cᵢ  ∈ ℝ³        — колір RGB (або SH-коефіцієнти)                        ║
║                                                                              ║
║  Коваріаційна матриця параметризується як:                                   ║
║    Σ = R · S · Sᵀ · Rᵀ                                                     ║
║  де:                                                                         ║
║    S = diag(s₁, s₂, s₃)  — масштаб (scale), зберігається як log(s)        ║
║    R                      — матриця обертання з кватерніона q               ║
║                                                                              ║
║  ПРОЄКЦІЯ В 2D (Splatting)                                                   ║
║  ──────────────────────────                                                  ║
║  Для камери з матрицею Якобіана J проєкції:                                 ║
║    Σ' = J · W · Σ · Wᵀ · Jᵀ   (2x2 матриця в image space)                 ║
║  де W — матриця view transform.                                              ║
║                                                                              ║
║  Кожен Гаусіан у 2D має вигляд:                                             ║
║    G(x) = exp( -½ · (x - μ')ᵀ · Σ'⁻¹ · (x - μ') )                       ║
║                                                                              ║
║  РЕНДЕРИНГ (Alpha Compositing, front-to-back)                               ║
║  ─────────────────────────────────────────────                               ║
║  Пікселі рендеряться через alpha-compositing по глибині:                    ║
║                                                                              ║
║    C = Σᵢ cᵢ · αᵢ · Gᵢ(x) · Πⱼ<ᵢ (1 - αⱼ · Gⱼ(x))                     ║
║                                                                              ║
║  LOSS FUNCTION                                                               ║
║  ─────────────                                                               ║
║    L = (1 - λ) · L₁ + λ · L_SSIM                                          ║
║  де:                                                                         ║
║    L₁    = |C_render - C_gt|  (пікселева різниця)                          ║
║    L_SSIM = структурна подібність (враховує локальні патерни)               ║
║    λ = 0.2  (стандартне значення з оригінальної статті)                     ║
║                                                                              ║
║  Градієнти backprop'аються через весь pipeline до μ, Σ, α, c               ║
║                                                                              ║
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


# ═══════════════════════════════════════════════════════════════════
#  ДОПОМІЖНІ ФУНКЦІЇ
# ═══════════════════════════════════════════════════════════════════

def quaternion_to_rotation_matrix(q: Tensor) -> Tensor:
    """
    Перетворює кватерніон [W, X, Y, Z] → матрицю обертання 3x3.
    
    q має бути нормалізований: |q| = 1
    Shape: (N, 4) → (N, 3, 3)
    """
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.stack([
        1 - 2*(y**2 + z**2),   2*(x*y - w*z),       2*(x*z + w*y),
        2*(x*y + w*z),          1 - 2*(x**2 + z**2), 2*(y*z - w*x),
        2*(x*z - w*y),          2*(y*z + w*x),       1 - 2*(x**2 + y**2)
    ], dim=-1).reshape(-1, 3, 3)

    return R


def build_covariance_3d(scaling: Tensor, rotation: Tensor) -> Tensor:
    """
    Будує 3D коваріаційну матрицю: Σ = R · S · Sᵀ · Rᵀ
    
    scaling:  (N, 3) — логарифм масштабу, exp() перед використанням
    rotation: (N, 4) — кватерніон
    Returns:  (N, 3, 3)
    """
    S = torch.diag_embed(torch.exp(scaling))   # (N, 3, 3)
    R = quaternion_to_rotation_matrix(rotation) # (N, 3, 3)
    # Σ = R @ S @ Sᵀ @ Rᵀ = R @ S² @ Rᵀ (якщо S діагональна)
    cov3d = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)
    return cov3d


def project_covariance_2d(
    cov3d: Tensor,
    xyz_cam: Tensor,
    fx: float, fy: float
) -> Tensor:
    """
    Проєктує 3D коваріацію в 2D image space через Якобіан проєкції.
    
    Якобіан перспективної проєкції (pinhole camera):
        J = [[fx/z,  0,   -fx·x/z²],
             [0,    fy/z, -fy·y/z²]]
    
    Σ' = J · Σ · Jᵀ  (результат: N, 2, 2)
    """
    x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
    z = z.clamp(min=1e-4)

    # Якобіан: (N, 2, 3)
    zeros = torch.zeros_like(x)
    J = torch.stack([
        fx / z,    zeros,    -fx * x / (z ** 2),
        zeros,     fy / z,   -fy * y / (z ** 2),
    ], dim=-1).reshape(-1, 2, 3)

    # Σ' = J @ Σ3d @ Jᵀ
    cov2d = J @ cov3d @ J.transpose(-1, -2)  # (N, 2, 2)

    # Додаємо мінімальний шум для числової стабільності
    cov2d[:, 0, 0] += 0.3
    cov2d[:, 1, 1] += 0.3

    return cov2d


# ═══════════════════════════════════════════════════════════════════
#  GAUSSIAN MODEL
# ═══════════════════════════════════════════════════════════════════

class GaussianModel(nn.Module):
    """
    Параметрична модель 3D Gaussian Splatting сцени.
    
    Всі параметри — диференційовані через PyTorch autograd,
    що дозволяє backprop від photometric loss до кожного Гаусіана.
    """

    def __init__(self, init_xyz: Tensor, init_rgb: Tensor):
        super().__init__()
        N = init_xyz.shape[0]
        print(f"  Ініціалізація {N:,} Гаусіанів...")

        # μ — позиції центрів
        self._xyz = nn.Parameter(init_xyz.clone().float())

        # log(s) — масштаб в логарифмічному просторі (для стабільності)
        # Початковий масштаб: маленькі сферичні Гаусіани
        init_scale = torch.log(torch.ones(N, 3) * 0.01)
        self._scaling = nn.Parameter(init_scale)

        # q — кватерніон обертання [W, X, Y, Z], старт = одиничний (без повороту)
        init_rot = torch.zeros(N, 4)
        init_rot[:, 0] = 1.0
        self._rotation = nn.Parameter(init_rot)

        # α — opacity через logit (sigmoid(2.19) ≈ 0.9 → старт з непрозорих)
        self._opacity = nn.Parameter(torch.ones(N, 1) * 0.5)

        # c — колір RGB, нормалізований в [0, 1]
        rgb = init_rgb.clone().float()
        rgb = rgb / 255.0 if rgb.max() > 1.0 else rgb
        self._rgb = nn.Parameter(rgb)

    def get_params(self) -> Dict[str, Tensor]:
        """Повертає активовані параметри, готові для рендерингу."""
        return {
            "xyz":      self._xyz,
            "scale":    torch.exp(self._scaling),
            "rotation": F.normalize(self._rotation, dim=-1),
            "opacity":  torch.sigmoid(self._opacity).squeeze(-1),
            "rgb":      torch.sigmoid(self._rgb),
            # 3D covariance для проєкції
            "cov3d":    build_covariance_3d(self._scaling, self._rotation),
        }


# ═══════════════════════════════════════════════════════════════════
#  ВАРІАНТ 1: PyTorch CPU/GPU Растерізатор (без CUDA extensions)
# ═══════════════════════════════════════════════════════════════════

class PureTorchRasterizer:
    """
    Диференційований растерізатор на чистому PyTorch.
    
    Алгоритм:
      1. Трансформуємо Гаусіани в camera space
      2. Проєктуємо коваріацію 3D → 2D (через Якобіан)
      3. Для кожного пікселя рахуємо вплив кожного Гаусіана G(x)
      4. Alpha-compositing front-to-back по глибині
    
    ⚠️  Складність O(N · H · W) — повільно для великих сцен.
        Для продакшену використовуй diff-gaussian-rasterization (CUDA).
        Цей варіант — для розуміння та малих тестових сцен (<5k Гаусіанів).
    """

    def __init__(self, H: int, W: int, fx: float, fy: float,
                 cx: float, cy: float):
        self.H, self.W = H, W
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy

    def render(
        self,
        params: Dict[str, Tensor],
        view_matrix: Tensor,         # (4, 4) world→camera
        bg_color: Tensor,            # (3,)
    ) -> Tensor:
        """
        Рендерить сцену з заданої камери.
        Returns: (H, W, 3) rendered image
        """
        device = params["xyz"].device
        H, W = self.H, self.W

        xyz_world = params["xyz"]          # (N, 3)
        rgb       = params["rgb"]          # (N, 3)
        opacity   = params["opacity"]      # (N,)
        cov3d     = params["cov3d"]        # (N, 3, 3)

        # ── 1. World → Camera transform ──────────────────────────────
        N = xyz_world.shape[0]
        ones = torch.ones(N, 1, device=device)
        xyz_h = torch.cat([xyz_world, ones], dim=-1)  # (N, 4)
        xyz_cam = (view_matrix @ xyz_h.T).T[:, :3]    # (N, 3)

        # Відкидаємо Гаусіани позаду камери
        mask = xyz_cam[:, 2] > 0.1
        xyz_cam  = xyz_cam[mask]
        rgb_m    = rgb[mask]
        opacity_m = opacity[mask]
        cov3d_m  = cov3d[mask]

        if xyz_cam.shape[0] == 0:
            return bg_color.expand(H, W, 3).clone()

        # ── 2. Проєкція центрів у 2D ─────────────────────────────────
        z = xyz_cam[:, 2]
        u = self.fx * xyz_cam[:, 0] / z + self.cx  # (M,)
        v = self.fy * xyz_cam[:, 1] / z + self.cy  # (M,)

        # ── 3. Проєкція коваріацій у 2D ──────────────────────────────
        # Rotation частина view matrix (3x3)
        R_view = view_matrix[:3, :3]  # (3, 3)
        cov3d_cam = R_view @ cov3d_m @ R_view.T  # (M, 3, 3)
        cov2d = project_covariance_2d(cov3d_cam, xyz_cam, self.fx, self.fy)
        # cov2d: (M, 2, 2)

        # ── 4. Інверсія 2x2 матриці аналітично ──────────────────────
        # [a b; c d]⁻¹ = 1/(ad-bc) · [d -b; -c a]
        a = cov2d[:, 0, 0]
        b = cov2d[:, 0, 1]
        c = cov2d[:, 1, 0]
        d = cov2d[:, 1, 1]
        det = a * d - b * c
        det = det.clamp(min=1e-8)
        inv = torch.stack([d, -b, -c, a], dim=-1).reshape(-1, 2, 2) / det.unsqueeze(-1).unsqueeze(-1)

        # ── 5. Сортування за глибиною (front-to-back) ────────────────
        sort_idx = torch.argsort(z)
        u, v     = u[sort_idx], v[sort_idx]
        rgb_m    = rgb_m[sort_idx]
        opacity_m = opacity_m[sort_idx]
        inv      = inv[sort_idx]

        # ── 6. Растеризація: обчислення впливу Гаусіанів на пікселі ──
        # Генеруємо сітку пікселів
        ys = torch.arange(H, device=device, dtype=torch.float32)
        xs = torch.arange(W, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)

        # Ініціалізуємо вихідне зображення та accumulated alpha
        image   = bg_color.clone().expand(H, W, 3).contiguous().clone()
        T       = torch.ones(H, W, device=device)  # transmittance

        M = u.shape[0]
        # Обробляємо по батчах щоб не OOM
        BATCH = 8
        for start in range(0, M, BATCH):
            end = min(start + BATCH, M)
            u_b   = u[start:end]       # (B,)
            v_b   = v[start:end]
            rgb_b = rgb_m[start:end]   # (B, 3)
            a_b   = opacity_m[start:end]  # (B,)
            inv_b = inv[start:end]     # (B, 2, 2)

            # dx, dy від центру кожного Гаусіана до кожного пікселя
            # (H, W, B)
            dx = grid_x.unsqueeze(-1) - u_b.unsqueeze(0).unsqueeze(0)
            dy = grid_y.unsqueeze(-1) - v_b.unsqueeze(0).unsqueeze(0)

            # Маланобіс відстань: d = [dx, dy] · Σ'⁻¹ · [dx, dy]ᵀ
            # (H, W, B, 2)
            d = torch.stack([dx, dy], dim=-1)
            # d @ inv_b: (H, W, B, 2) @ (B, 2, 2) → (H, W, B, 2)
            mahal = (d.unsqueeze(-2) @ inv_b).squeeze(-2)  # (H, W, B, 2)
            mahal = (mahal * d).sum(-1)                    # (H, W, B)

            # G(x) = exp(-½ · mahal)
            G = torch.exp(-0.5 * mahal.clamp(max=20.0))   # (H, W, B)

            # Вклад кожного Гаусіана
            alpha = (a_b * G).clamp(0, 0.99)              # (H, W, B)

            # Alpha compositing front-to-back:
            # C += T · α · c;   T *= (1 - α)
            for i in range(end - start):
                contrib = (T * alpha[..., i]).unsqueeze(-1) * rgb_b[i]
                image   = image + contrib
                T       = T * (1.0 - alpha[..., i])

        return image.clamp(0, 1)


# ═══════════════════════════════════════════════════════════════════
#  ВАРІАНТ 2: CUDA Растерізатор (diff-gaussian-rasterization)
# ═══════════════════════════════════════════════════════════════════

def render_with_cuda_rasterizer(
    params: Dict[str, Tensor],
    camera,                       # об'єкт з полями: R, T, FoVx, FoVy, image_width, image_height
    bg_color: Tensor,
) -> Dict[str, Tensor]:
    """
    Рендеринг через офіційний CUDA растерізатор.
    
    Потребує:
        pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization
    
    Переваги перед PyTorch варіантом:
      - Tile-based растеризація (ділить екран на 16×16 тайли)
      - Повністю на GPU, O(N) замість O(N·H·W)
      - В 100-1000x швидше для великих сцен
      - Градієнти реалізовані в CUDA backward pass
    """
    try:
        from diff_gaussian_rasterization import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )
    except ImportError:
        raise ImportError(
            "Встанови CUDA растерізатор:\n"
            "  pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization\n"
            "Потрібен: CUDA toolkit, компілятор C++ (MSVC або GCC)"
        )

    import math

    # Будуємо матриці проєкції (OpenGL конвенція)
    def get_projection_matrix(fov_x, fov_y, znear=0.01, zfar=100.0):
        t = math.tan(fov_y / 2) * znear
        r = math.tan(fov_x / 2) * znear
        P = torch.zeros(4, 4)
        P[0, 0] = znear / r
        P[1, 1] = znear / t
        P[2, 2] = (zfar + znear) / (zfar - znear)
        P[3, 2] = 1.0
        P[2, 3] = -(2 * zfar * znear) / (zfar - znear)
        return P

    device = params["xyz"].device
    H, W   = camera.image_height, camera.image_width

    # World-to-camera: [R | T]
    R = torch.tensor(camera.R, dtype=torch.float32, device=device)
    T_vec = torch.tensor(camera.T, dtype=torch.float32, device=device)
    view_matrix = torch.eye(4, device=device)
    view_matrix[:3, :3] = R
    view_matrix[:3,  3] = T_vec
    view_matrix = view_matrix.T  # Column-major для OpenGL

    proj_matrix = get_projection_matrix(
        camera.FoVx, camera.FoVy
    ).to(device).T  # Column-major

    full_proj = (view_matrix.unsqueeze(0) @ proj_matrix.unsqueeze(0)).squeeze(0)

    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=math.tan(camera.FoVx / 2),
        tanfovy=math.tan(camera.FoVy / 2),
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=view_matrix,
        projmatrix=full_proj,
        sh_degree=0,          # 0 = простий RGB (без Spherical Harmonics)
        campos=torch.zeros(3, device=device),
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # CUDA растерізатор очікує SH-коефіцієнти замість простого RGB
    # При sh_degree=0 достатньо одного коефіцієнта (DC term)
    # SH DC = color / C0, де C0 = 0.28209...
    C0 = 0.28209479177387814
    shs = (params["rgb"] - 0.5) / C0  # (N, 3)
    shs = shs.unsqueeze(1)            # (N, 1, 3) — 1 band

    rendered_image, radii = rasterizer(
        means3D=params["xyz"],
        means2D=torch.zeros_like(params["xyz"][:, :2], requires_grad=True),
        shs=shs,
        colors_precomp=None,
        opacities=params["opacity"].unsqueeze(-1),
        scales=params["scale"],
        rotations=params["rotation"],
        cov3D_precomp=None,
    )

    return {
        "render":  rendered_image,   # (3, H, W)
        "radii":   radii,            # (N,) — видимі Гаусіани
    }


# ═══════════════════════════════════════════════════════════════════
#  LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def l1_loss(pred: Tensor, gt: Tensor) -> Tensor:
    """MAE між рендером та ground truth."""
    return torch.abs(pred - gt).mean()


def ssim_loss(pred: Tensor, gt: Tensor, window_size: int = 11) -> Tensor:
    """
    Structural Similarity Index (SSIM).
    
    Порівнює локальні патерни яскравості, контрасту та структури.
    Краще L1 відображає перцептивну схожість зображень.
    
    SSIM(x,y) = (2μₓμᵧ + C₁)(2σₓᵧ + C₂) / ((μₓ² + μᵧ² + C₁)(σₓ² + σᵧ² + C₂))
    
    pred, gt: (C, H, W) або (3, H, W)
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)  # → (1, C, H, W)
        gt   = gt.unsqueeze(0)

    C1, C2 = 0.01**2, 0.03**2

    # Гаусове ядро для локального усереднення
    def gaussian_kernel(size: int, sigma: float = 1.5) -> Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        kernel = g.outer(g)
        kernel = kernel / kernel.sum()
        return kernel.expand(pred.shape[1], 1, size, size).to(pred.device)

    kernel = gaussian_kernel(window_size)
    pad = window_size // 2

    mu_x  = F.conv2d(pred, kernel, padding=pad, groups=pred.shape[1])
    mu_y  = F.conv2d(gt,   kernel, padding=pad, groups=gt.shape[1])
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sig_x2  = F.conv2d(pred*pred, kernel, padding=pad, groups=pred.shape[1]) - mu_x2
    sig_y2  = F.conv2d(gt*gt,     kernel, padding=pad, groups=gt.shape[1])   - mu_y2
    sig_xy  = F.conv2d(pred*gt,   kernel, padding=pad, groups=pred.shape[1]) - mu_xy

    ssim_map = ((2*mu_xy + C1) * (2*sig_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2))

    return 1.0 - ssim_map.mean()


def photometric_loss(pred: Tensor, gt: Tensor, lambda_ssim: float = 0.2) -> Tensor:
    """
    L = (1 - λ) · L₁ + λ · L_SSIM
    
    λ = 0.2 — значення з оригінальної статті (Kerbl et al., 2023)
    """
    return (1.0 - lambda_ssim) * l1_loss(pred, gt) + lambda_ssim * ssim_loss(pred, gt)


# ═══════════════════════════════════════════════════════════════════
#  DENSIFICATION (Adaptive Control of Gaussians)
# ═══════════════════════════════════════════════════════════════════

class GaussianDensifier:
    """
    Адаптивний контроль Гаусіанів — ключова частина 3DGS.
    
    Логіка:
      - Клонуємо малі Гаусіани з великим градієнтом (under-reconstruction)
      - Розбиваємо великі Гаусіани з великим градієнтом (over-reconstruction)
      - Видаляємо прозорі Гаусіани (α < threshold)
    
    Виконується кожні ~100 ітерацій.
    """

    def __init__(self, model: GaussianModel, grad_threshold: float = 0.0002):
        self.model          = model
        self.grad_threshold = grad_threshold
        self.xyz_grad_accum = None
        self.denom          = None
        self._reset_stats()

    def _reset_stats(self):
        N = self.model._xyz.shape[0]
        device = self.model._xyz.device
        self.xyz_grad_accum = torch.zeros(N, device=device)
        self.denom          = torch.zeros(N, device=device)

    def update_stats(self):
        """Накопичує градієнти по xyz після кожного backward."""
        if self.model._xyz.grad is not None:
            grad_norm = self.model._xyz.grad.norm(dim=-1)
            self.xyz_grad_accum += grad_norm
            self.denom          += 1

    def densify_and_prune(
        self,
        optimizer: torch.optim.Optimizer,
        opacity_threshold: float = 0.005,
        max_gaussians: int = 1_000_000,
    ):
        avg_grad = self.xyz_grad_accum / (self.denom + 1e-8)

        # Маска Гаусіанів з великим градієнтом
        densify_mask = avg_grad > self.grad_threshold

        with torch.no_grad():
            scales = torch.exp(self.model._scaling)
            max_scale = scales.max(dim=-1).values

            # Малі → клонуємо
            clone_mask  = densify_mask & (max_scale <= 0.01)
            # Великі → розбиваємо на 2
            split_mask  = densify_mask & (max_scale > 0.01)
            # Прозорі → видаляємо
            prune_mask  = torch.sigmoid(self.model._opacity).squeeze() < opacity_threshold

            print(f"  Densification: clone={clone_mask.sum()}, "
                  f"split={split_mask.sum()}, prune={prune_mask.sum()}")

            # Клонування
            if clone_mask.any():
                self._clone_gaussians(clone_mask, optimizer)

            # Видалення (після клонування)
            keep_mask = ~prune_mask
            if not keep_mask.all():
                self._prune_gaussians(keep_mask, optimizer)

        self._reset_stats()

    def _clone_gaussians(self, mask: Tensor, optimizer: torch.optim.Optimizer):
        """Дублюємо Гаусіани з малим масштабом та великим градієнтом."""
        new_xyz      = self.model._xyz[mask].detach()
        new_scaling  = self.model._scaling[mask].detach()
        new_rotation = self.model._rotation[mask].detach()
        new_opacity  = self.model._opacity[mask].detach()
        new_rgb      = self.model._rgb[mask].detach()

        self._concat_params(
            new_xyz, new_scaling, new_rotation, new_opacity, new_rgb, optimizer
        )

    def _concat_params(self, xyz, scaling, rotation, opacity, rgb, optimizer):
        """Додає нові Гаусіани до параметрів моделі та оновлює оптимізатор."""
        param_dict = {
            "_xyz":      xyz,
            "_scaling":  scaling,
            "_rotation": rotation,
            "_opacity":  opacity,
            "_rgb":      rgb,
        }
        for name, new_tensor in param_dict.items():
            old_param = getattr(self.model, name)
            new_full  = torch.cat([old_param.data, new_tensor], dim=0)

            # Оновлюємо параметр моделі
            setattr(self.model, name, nn.Parameter(new_full))

            # Оновлюємо стан оптимізатора (momentum буферами)
            for group in optimizer.param_groups:
                if group.get("name") == name:
                    old_state = optimizer.state[group["params"][0]]
                    new_param = getattr(self.model, name)
                    group["params"][0] = new_param

                    if "exp_avg" in old_state:
                        zeros = torch.zeros_like(new_tensor)
                        old_state["exp_avg"]    = torch.cat([old_state["exp_avg"], zeros])
                        old_state["exp_avg_sq"] = torch.cat([old_state["exp_avg_sq"], zeros])
                    optimizer.state[new_param] = old_state

        # Розширюємо накопичувачі градієнтів
        n_new = xyz.shape[0]
        device = xyz.device
        self.xyz_grad_accum = torch.cat([self.xyz_grad_accum,
                                          torch.zeros(n_new, device=device)])
        self.denom = torch.cat([self.denom,
                                 torch.zeros(n_new, device=device)])

    def _prune_gaussians(self, keep_mask: Tensor, optimizer: torch.optim.Optimizer):
        """Видаляємо 'мертві' Гаусіани (занадто прозорі)."""
        for name in ["_xyz", "_scaling", "_rotation", "_opacity", "_rgb"]:
            old_param = getattr(self.model, name)
            new_data  = old_param.data[keep_mask]
            setattr(self.model, name, nn.Parameter(new_data))

            for group in optimizer.param_groups:
                if group.get("name") == name:
                    old_state = optimizer.state.get(group["params"][0], {})
                    new_param = getattr(self.model, name)
                    group["params"][0] = new_param
                    if "exp_avg" in old_state:
                        old_state["exp_avg"]    = old_state["exp_avg"][keep_mask]
                        old_state["exp_avg_sq"] = old_state["exp_avg_sq"][keep_mask]
                    optimizer.state[new_param] = old_state

        self.xyz_grad_accum = self.xyz_grad_accum[keep_mask]
        self.denom          = self.denom[keep_mask]


# ═══════════════════════════════════════════════════════════════════
#  TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════

def create_optimizer(model: GaussianModel) -> torch.optim.Adam:
    """
    Adam з різними learning rates для різних параметрів.
    Значення взяті з оригінальної статті Kerbl et al. 2023.
    """
    return torch.optim.Adam([
        {"params": [model._xyz],      "lr": 0.00016, "name": "_xyz"},
        {"params": [model._scaling],  "lr": 0.005,   "name": "_scaling"},
        {"params": [model._rotation], "lr": 0.001,   "name": "_rotation"},
        {"params": [model._opacity],  "lr": 0.05,    "name": "_opacity"},
        {"params": [model._rgb],      "lr": 0.0025,  "name": "_rgb"},
    ], eps=1e-15)


def train(
    pt_path: str,
    output_path: str,
    num_iterations: int = 7000,
    use_cuda_rasterizer: bool = False,
    H: int = 256,
    W: int = 256,
):
    """
    Головний training loop 3DGS.
    
    Args:
        pt_path:             шлях до .pt файлу з xyz та rgb точковою хмарою
        output_path:         куди зберегти навчену сцену
        num_iterations:      кількість ітерацій (7000 — швидко, 30000 — якісно)
        use_cuda_rasterizer: True → diff-gaussian-rasterization, False → PyTorch
        H, W:                розмір зображення для рендерингу
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Пристрій: {device}")

    # ── Завантаження даних ──────────────────────────────────────────
    if not os.path.exists(pt_path):
        print(f"  Файл не знайдено: {pt_path}")
        print("  Генеруємо синтетичну хмару точок для демонстрації...")
        N = 2048
        data = {
            "xyz": torch.randn(N, 3) * 2.0,
            "rgb": torch.rand(N, 3),
        }
    else:
        data = torch.load(pt_path, map_location=device)
        print(f"  Завантажено {data['xyz'].shape[0]:,} точок")

    # ── Ініціалізація моделі ────────────────────────────────────────
    model = GaussianModel(data["xyz"].to(device), data["rgb"].to(device)).to(device)
    optimizer = create_optimizer(model)
    densifier = GaussianDensifier(model, grad_threshold=0.0002)

    # ── Налаштування камери (синтетична, для демонстрації) ──────────
    # У реальному проєкті: завантажуй з COLMAP output
    fx = fy = W * 1.2  # наближена фокусна відстань
    cx, cy  = W / 2, H / 2

    # View matrix: камера дивиться з (0,0,5) в сторону початку координат
    view_matrix = torch.eye(4, device=device)
    view_matrix[2, 3] = -5.0  # translate z = -5

    bg_color = torch.zeros(3, device=device)  # чорний фон

    rasterizer = PureTorchRasterizer(H, W, fx, fy, cx, cy)

    # ── Training Loop ───────────────────────────────────────────────
    print(f"\n  Тренування: {num_iterations} ітерацій")
    print(f"  Растерізатор: {'CUDA (diff-gaussian-rasterization)' if use_cuda_rasterizer else 'PyTorch'}\n")

    losses = []

    for iteration in range(1, num_iterations + 1):
        t0 = time.time()
        optimizer.zero_grad()

        # ── Forward pass ────────────────────────────────────────────
        params = model.get_params()

        if use_cuda_rasterizer:
            # CUDA шлях
            out = render_with_cuda_rasterizer(params, camera=None, bg_color=bg_color)
            rendered = out["render"]  # (3, H, W)
            # Ground truth: у реальному проєкті завантажуй реальне фото
            gt = torch.rand_like(rendered)
            loss = photometric_loss(rendered, gt)
        else:
            # PyTorch шлях
            rendered = rasterizer.render(params, view_matrix, bg_color)  # (H, W, 3)
            # Ground truth: у реальному проєкті завантажуй реальне фото
            gt = torch.rand(H, W, 3, device=device)
            loss = photometric_loss(
                rendered.permute(2, 0, 1),  # → (3, H, W)
                gt.permute(2, 0, 1)
            )

        # ── Backward pass ────────────────────────────────────────────
        loss.backward()
        densifier.update_stats()
        optimizer.step()

        losses.append(loss.item())
        elapsed = (time.time() - t0) * 1000

        # ── Densification кожні 100 ітерацій ────────────────────────
        if iteration % 100 == 0 and iteration < num_iterations * 0.75:
            densifier.densify_and_prune(optimizer)

        # ── Логування ───────────────────────────────────────────────
        if iteration % 50 == 0 or iteration == 1:
            n = model._xyz.shape[0]
            print(f"  Ітерація [{iteration:>5}/{num_iterations}] | "
                  f"Loss: {loss.item():.4f} | "
                  f"Гаусіанів: {n:,} | "
                  f"Час: {elapsed:.1f}ms")

    # ── Збереження ──────────────────────────────────────────────────
    final = model.get_params()
    scene = {k: v.detach().cpu() for k, v in final.items() if k != "cov3d"}
    torch.save(scene, output_path)

    avg_loss = sum(losses) / len(losses)
    print(f"\n  Середній loss: {avg_loss:.4f}")
    print(f"  Збережено: {output_path}")
    print(f"  Фінальна кількість Гаусіанів: {model._xyz.shape[0]:,}")


# ═══════════════════════════════════════════════════════════════════
#  ТОЧКА ВХОДУ
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="3D Gaussian Splatting Trainer")
    parser.add_argument("--input",       default="data/zaporizhzhia_cloud.pt")
    parser.add_argument("--output",      default="data/zaporizhzhia_splat_v2.pt")
    parser.add_argument("--iterations",  type=int,  default=7000)
    parser.add_argument("--cuda",        action="store_true",
                        help="Використовувати diff-gaussian-rasterization (потребує CUDA build)")
    parser.add_argument("--height",      type=int,  default=256)
    parser.add_argument("--width",       type=int,  default=256)
    args = parser.parse_args()

    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    pt_path     = os.path.join(project_dir, args.input)
    output_path = os.path.join(project_dir, args.output)

    train(
        pt_path=pt_path,
        output_path=output_path,
        num_iterations=args.iterations,
        use_cuda_rasterizer=args.cuda,
        H=args.height,
        W=args.width,
    )