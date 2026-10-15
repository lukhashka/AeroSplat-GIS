"""gsplat backend of the 3DGS trainer (ROADMAP Фаза 2.2): `rasterizer: gsplat`.

What it adds over the Inria (`cuda`) and pure-PyTorch (`torch`) paths of trainer.py:

* view-dependent colour: SH up to degree 3. All coefficients exist from the
  start; the renderer uses one more band every `sh_degree_interval` iterations;
* `rasterize_mode="antialiased"`: the Mip-Splatting 2D filter with opacity
  compensation. The viewer's shader does the same since ADR-0004 §C2, so the
  trained opacities are the ones the viewer needs;
* densification `mcmc` (the default; +0.56 dB on dataset_v6, docs/BASELINE.md):
  3DGS-MCMC (dead splats relocated onto live ones, growth to a fixed cap,
  position noise); `default`: Kerbl et al. with this trainer's rules (hard
  `max_gaussians` budget, screen-radius prune, opacity reset), optionally on
  absolute screen gradients (AbsGS, `absgrad`);
* `appearance="exposure"`: a per-train-image colour affine applied to the render
  during training only (auto-exposure, lighting differences between sources,
  TARGET_ARCHITECTURE §6.2). Test views and the exported splat have none.

Parameters live in a ParameterDict keyed the way gsplat's densification ops
expect (means, scales, quats, opacities, sh0, shN), one Adam per key. The saved
.pt keeps the trainer's contract (ADR-0002): activated xyz/scale/rotation/opacity
and `rgb` (the SH DC term as a colour), plus `sh`, `sh_degree` and `render`.

Without a prebuilt wheel gsplat JIT-compiles its CUDA extension on first use;
on Windows run through tools/gsplat_env.bat (docs/BASELINE.md, «gsplat на Windows»).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from aerosplat.gs.sh import MAX_SH_DEGREE, num_sh_bases, rgb_to_sh_dc, sh_dc_to_rgb, sh_degree_of
from aerosplat.gs.trainer import (
    Camera, CameraSampler, anisotropy_penalty, inverse_sigmoid, log, photometric_loss, psnr,
    xyz_lr_at,
)

RASTERIZE_MODES = ("classic", "antialiased")
DENSIFY_STRATEGIES = ("default", "mcmc")
APPEARANCE_MODES = ("none", "exposure")

# Same near cull as the Inria rasterizer (it drops z <= 0.2).
NEAR_PLANE = 0.2
# Adam learning rates of Kerbl et al. / gsplat's simple_trainer; means: see xyz_lr_at.
LR = {"scales": 0.005, "quats": 0.001, "opacities": 0.05, "sh0": 0.0025, "shN": 0.0025 / 20}
EXPOSURE_LR = 0.0005
PRUNE_OPACITY = 0.005
OPACITY_RESET_VALUE = 0.01


def _toolkit_version(cuda_home: Optional[str]) -> Optional[str]:
    """"12.6" for the CUDA toolkit at `cuda_home` (version.json or version.txt)."""
    if not cuda_home:
        return None
    try:
        with open(os.path.join(cuda_home, "version.json"), encoding="utf-8") as f:
            full = str(json.load(f)["cuda"]["version"])
    except (OSError, KeyError, ValueError):
        try:
            with open(os.path.join(cuda_home, "version.txt"), encoding="utf-8") as f:
                full = f.read().split()[-1]
        except (OSError, IndexError):
            return None
    return ".".join(full.split(".")[:2])


def build_env_problems() -> List[str]:
    """Why a gsplat JIT build would fail here ([] = the environment looks right).

    torch rebuilds the extension whenever the toolchain differs from the cached
    build, and a failed build leaves the cache unusable, so this must be checked
    before gsplat's first call rather than diagnosed from ninja's output.
    """
    from torch.utils.cpp_extension import CUDA_HOME

    problems = []
    have, want = _toolkit_version(CUDA_HOME), torch.version.cuda or "?"
    if have != ".".join(want.split(".")[:2]):
        problems.append(f"CUDA toolkit {have} ({CUDA_HOME}), а torch зібрано з CUDA {want}")
    if os.name == "nt":
        cl = shutil.which("cl")
        toolset = re.search(r"MSVC[\\/](\d+)\.(\d+)", cl or "")
        if not cl:
            problems.append("cl.exe (MSVC) не в PATH")
        elif (toolset and (int(toolset[1]), int(toolset[2])) >= (14, 50)
              and tuple(int(v) for v in re.findall(r"\d+", want)[:2]) < (12, 9)):
            # nvcc <= 12.8 supports MSVC 14.3x/14.4x (VS 2022); VS 18's cudafe++ crashes.
            problems.append(f"{cl}: MSVC {toolset[1]}.{toolset[2]} (VS 18) не підтримується nvcc {want}")
    return problems


def require_gsplat() -> None:
    """Fail early with a usable message when gsplat or its build environment is missing."""
    try:
        import gsplat  # noqa: F401
    except ImportError as e:
        raise ImportError("gsplat не встановлено: pip install gsplat==1.5.3 "
                          "(див. docs/BASELINE.md, «gsplat на Windows»)") from e
    if not torch.cuda.is_available():
        raise RuntimeError("rasterizer=gsplat потребує CUDA")
    try:
        from gsplat import csrc  # noqa: F401  (prebuilt extension: no JIT, no toolchain)
        return
    except ImportError:
        pass
    problems = build_env_problems()
    if problems:
        raise RuntimeError(
            "gsplat компілює CUDA-розширення при першому виклику, а середовище збірки не те: "
            + "; ".join(problems) + ". Запускай через tools\\gsplat_env.bat, напр.: "
            "tools\\gsplat_env.bat venv\\Scripts\\aerosplat run ... --set rasterizer=gsplat")


# ═══════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════

class SplatModel(nn.Module):
    """Raw (pre-activation) splat parameters in gsplat's layout.

    means (N,3) · scales (N,3) log · quats (N,4) w,x,y,z unnormalised ·
    opacities (N,) logit · sh0 (N,1,3) DC · shN (N,K-1,3) higher bands
    (absent for sh_degree 0).
    """

    def __init__(self, xyz: Tensor, rgb: Tensor, init_scale: Tensor, sh_degree: int = 3,
                 init_opacity: float = 0.1):
        super().__init__()
        if not 0 <= sh_degree <= MAX_SH_DEGREE:
            raise ValueError(f"sh_degree must be 0..{MAX_SH_DEGREE}, got {sh_degree}")
        n, device = xyz.shape[0], xyz.device
        log(f"  Ініціалізація {n:,} Гаусіанів (gsplat, SH до ступеня {sh_degree})...")
        colour = rgb.float()
        if colour.max() > 1.0:
            colour = colour / 255.0
        quats = torch.zeros(n, 4, device=device)
        quats[:, 0] = 1.0
        params = {
            "means": xyz.float(),
            "scales": torch.log(init_scale.float().clamp(min=1e-7)).unsqueeze(-1).repeat(1, 3),
            "quats": quats,
            "opacities": torch.full((n,), inverse_sigmoid(init_opacity), device=device),
            "sh0": rgb_to_sh_dc(colour.clamp(0.0, 1.0)).unsqueeze(1),
        }
        if sh_degree > 0:
            params["shN"] = torch.zeros(n, num_sh_bases(sh_degree) - 1, 3, device=device)
        self.splats = nn.ParameterDict({k: nn.Parameter(v.contiguous()) for k, v in params.items()})
        self.sh_degree = sh_degree

    @property
    def n_gaussians(self) -> int:
        return self.splats["means"].shape[0]

    def sh(self) -> Tensor:
        if "shN" in self.splats:
            return torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)
        return self.splats["sh0"]

    def get_params(self) -> Dict[str, Tensor]:
        """Activated values for GsplatRasterizer (the keys a saved .pt has)."""
        s = self.splats
        return {"xyz": s["means"], "scale": torch.exp(s["scales"]),
                "rotation": F.normalize(s["quats"], dim=-1),
                "opacity": torch.sigmoid(s["opacities"]), "sh": self.sh()}

    @torch.no_grad()
    def export(self) -> Dict[str, Any]:
        """CPU float32 tensors in the .pt layout, `rgb` = view-independent DC colour."""
        out: Dict[str, Any] = {k: v.detach().float().cpu() for k, v in self.get_params().items()}
        out["rgb"] = sh_dc_to_rgb(out["sh"][:, 0])
        out["sh_degree"] = self.sh_degree
        return out


def create_optimizers(model: SplatModel, extent: float) -> Dict[str, torch.optim.Optimizer]:
    """One Adam per parameter (gsplat's ops replace parameters key by key)."""
    lrs = dict(LR, means=xyz_lr_at(0, 1, extent))
    return {name: torch.optim.Adam([{"params": [p], "lr": lrs[name], "name": name}], eps=1e-15)
            for name, p in model.splats.items()}


def save_splat(model: SplatModel, path: str, rasterize_mode: str,
               extra: Optional[Dict[str, Any]] = None) -> None:
    """The trainer's .pt contract (ADR-0002) plus SH.

    xyz/scale/rotation/opacity activated, `rgb` = clamp(0.5 + C0·sh[:, 0]) for the
    server and viewer (no SH there), `sh` (N, K, 3) raw coefficients including
    the DC term, `sh_degree`, and `render` = how to reproduce the training renders.
    """
    scene = model.export()
    scene["activations"] = {"scale_is_log": False, "opacity_is_logit": False}
    scene["render"] = {"rasterizer": "gsplat", "rasterize_mode": rasterize_mode}
    if extra:
        scene.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(scene, path)


# ═══════════════════════════════════════════════════════════════════
#  Rasterizer
# ═══════════════════════════════════════════════════════════════════

class GsplatRasterizer:
    """gsplat's CUDA rasterizer behind the trainer's `render()` interface.

    `params` are activated values (SplatModel.get_params() or a loaded .pt):
    with an `sh` key the colour is evaluated from SH, otherwise `rgb` is used
    as is. `sh_degree` None = every band the coefficients have.
    """

    def __init__(self, rasterize_mode: str = "antialiased", sh_degree: Optional[int] = None,
                 absgrad: bool = False, packed: bool = False):
        if rasterize_mode not in RASTERIZE_MODES:
            raise ValueError(f"rasterize_mode must be one of {RASTERIZE_MODES}, got {rasterize_mode!r}")
        require_gsplat()
        from gsplat import rasterization

        self._rasterization = rasterization
        self.rasterize_mode = rasterize_mode
        self.sh_degree = sh_degree
        self.absgrad = absgrad
        self.packed = packed

    def rasterize(self, params: Dict[str, Tensor], cam: Camera, view: Tensor, bg_color: Tensor,
                  sh_degree: Optional[int] = None) -> Tuple[Tensor, Dict[str, Any]]:
        """(H, W, 3) unclamped render and gsplat's meta (means2d, radii, ...)."""
        device = params["xyz"].device
        K = torch.tensor([[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]],
                         device=device)
        sh = params.get("sh")
        degree: Optional[int]
        if sh is not None:
            available = sh_degree_of(sh.shape[1])
            colors, degree = sh, available if sh_degree is None else min(sh_degree, available)
        else:
            colors, degree = params["rgb"], None
        renders, _, meta = self._rasterization(
            means=params["xyz"], quats=params["rotation"], scales=params["scale"],
            opacities=params["opacity"].reshape(-1), colors=colors,
            viewmats=view[None], Ks=K[None], width=int(cam.W), height=int(cam.H),
            near_plane=NEAR_PLANE, sh_degree=degree, packed=self.packed, absgrad=self.absgrad,
            rasterize_mode=self.rasterize_mode, backgrounds=bg_color[None])
        return renders[0], meta

    def render(self, params: Dict[str, Tensor], cam: Camera, view: Tensor, bg_color: Tensor,
               screen_pts: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, Any]]:
        img, meta = self.rasterize(params, cam, view, bg_color, self.sh_degree)
        return img.clamp(0, 1), meta


# ═══════════════════════════════════════════════════════════════════
#  Densification
# ═══════════════════════════════════════════════════════════════════

class GsplatDensifier:
    """Adaptive density control (Kerbl et al. §5.2) on SplatModel, with the rules
    of trainer.GaussianDensifier: clone small / split large splats whose mean
    screen-space (NDC) gradient exceeds the threshold, a hard `max_gaussians`
    budget (highest gradients win), prune of transparent, world-oversized
    (> 0.1·extent) and screen-wide splats, opacity reset.

    absgrad=True accumulates per-pixel |grad| (AbsGS); it needs a ~4x higher
    threshold. The parameter surgery is gsplat's (strategy.ops), which keeps the
    Adam state and `self.state` aligned with the splats.
    """

    def __init__(self, model: SplatModel, grad_threshold: float = 0.0002, absgrad: bool = False,
                 percent_dense: float = 0.01):
        self.model = model
        self.grad_threshold = grad_threshold
        self.absgrad = absgrad
        self.percent_dense = percent_dense
        self.state: Dict[str, Tensor] = {}
        self._reset_stats()

    def _reset_stats(self) -> None:
        n, device = self.model.n_gaussians, self.model.splats["means"].device
        self.state = {key: torch.zeros(n, device=device)
                      for key in ("grad2d", "count", "max_radius")}

    @torch.no_grad()
    def update_stats(self, meta: Dict[str, Any]) -> None:
        """Accumulate the NDC screen gradient of every splat visible in this render."""
        means2d = meta["means2d"]
        grad = means2d.absgrad if self.absgrad else means2d.grad
        if grad is None:
            return
        grad = grad.clone()
        grad[..., 0] *= meta["width"] / 2.0          # px -> NDC, as in the paper
        grad[..., 1] *= meta["height"] / 2.0
        radii = meta["radii"]
        if radii.dim() == grad.dim():                 # gsplat >= 1.5: per-axis radii
            radii = radii.max(dim=-1).values
        if "gaussian_ids" in meta and grad.dim() == 2:   # packed
            ids = meta["gaussian_ids"]
        else:
            sel = radii > 0                           # (C, N)
            ids = torch.where(sel)[1]
            grad, radii = grad[sel], radii[sel]
        st = self.state
        st["grad2d"].index_add_(0, ids, grad.norm(dim=-1))
        st["count"].index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))
        st["max_radius"][ids] = torch.maximum(st["max_radius"][ids], radii.float())

    @torch.no_grad()
    def densify_and_prune(self, optimizers: Dict[str, torch.optim.Optimizer], extent: float,
                          max_gaussians: int, max_screen_radius: Optional[float] = None,
                          opacity_threshold: float = PRUNE_OPACITY) -> None:
        from gsplat.strategy import ops

        splats, st = self.model.splats, self.state
        n = self.model.n_gaussians
        avg_grad = st["grad2d"] / st["count"].clamp(min=1.0)
        big = torch.exp(splats["scales"]).max(dim=-1).values > self.percent_dense * extent
        cand = avg_grad > self.grad_threshold
        clone_mask, split_mask = cand & ~big, cand & big

        budget = max_gaussians - n
        cand_idx = torch.nonzero(cand, as_tuple=True)[0]
        if budget <= 0:
            clone_mask = torch.zeros_like(clone_mask)
            split_mask = torch.zeros_like(split_mask)
            log(f"    max_gaussians={max_gaussians:,} досягнуто — тільки prune.")
        elif cand_idx.numel() > budget:
            keep = cand_idx[torch.argsort(avg_grad[cand_idx], descending=True)[:budget]]
            sel = torch.zeros(n, dtype=torch.bool, device=cand.device)
            sel[keep] = True
            clone_mask &= sel
            split_mask &= sel

        # New splats start without screen-radius history (the original trainer's
        # rule): children of a screen-wide parent must not be pruned for it.
        st["max_radius"][clone_mask | split_mask] = 0.0
        n_clone, n_split = int(clone_mask.sum()), int(split_mask.sum())
        if n_clone:
            ops.duplicate(splats, optimizers, st, clone_mask)
        if n_split:
            split_mask = torch.cat([split_mask, torch.zeros(n_clone, dtype=torch.bool,
                                                            device=split_mask.device)])
            ops.split(splats, optimizers, st, split_mask)

        prune = torch.sigmoid(splats["opacities"]) < opacity_threshold
        prune |= torch.exp(splats["scales"]).max(dim=-1).values > 0.1 * extent
        if max_screen_radius is not None:
            prune |= st["max_radius"] > max_screen_radius
        n_prune = int(prune.sum())
        if n_prune and n_prune < self.model.n_gaussians:
            ops.remove(splats, optimizers, st, prune)

        log(f"    Densify: clone={n_clone:,} split={n_split:,} "
            f"prune={n_prune:,} → {self.model.n_gaussians:,} Гаусіанів")
        self._reset_stats()

    @torch.no_grad()
    def reset_opacity(self, optimizers: Dict[str, torch.optim.Optimizer],
                      value: float = OPACITY_RESET_VALUE) -> None:
        from gsplat.strategy import ops

        ops.reset_opa(self.model.splats, optimizers, self.state, value)


class McmcDensifier:
    """3DGS-MCMC (Kheradmand et al. 2024) through gsplat's MCMCStrategy.

    Dead splats (opacity <= 0.005) are relocated onto live ones, the count grows
    by 5% per refine step up to `cap` and stays there, and low-opacity splats get
    covariance-shaped position noise. Its opacity/scale regularisers are added
    to the loss by the training loop (mcmc_regularizer).
    """

    def __init__(self, model: SplatModel, cap: int, refine_start: int, refine_stop: int,
                 refine_every: int, noise_lr: float, extent: float):
        from gsplat.strategy import MCMCStrategy

        self.model = model
        self.extent = extent
        self.strategy = MCMCStrategy(cap_max=cap, noise_lr=noise_lr, refine_start_iter=refine_start,
                                     refine_stop_iter=refine_stop, refine_every=refine_every,
                                     min_opacity=PRUNE_OPACITY)
        self.state = self.strategy.initialize_state()

    def step(self, optimizers: Dict[str, torch.optim.Optimizer], iteration: int,
             meta: Dict[str, Any], lr_xyz: float) -> None:
        # gsplat's noise is covariance · lr · noise_lr, tuned for a world normalised
        # to a unit scene radius; in metres that is covariance · lr_m · noise_lr / extent².
        self.strategy.step_post_backward(self.model.splats, optimizers, self.state, iteration,
                                         meta, lr=lr_xyz / self.extent ** 2)


def mcmc_regularizer(params: Dict[str, Tensor], extent: float, opacity_reg: float,
                     scale_reg: float) -> Tensor:
    """MCMC's L1 pull of opacities and scales towards zero (scales relative to the scene)."""
    return (opacity_reg * params["opacity"].mean()
            + scale_reg * params["scale"].mean() / extent)


# ═══════════════════════════════════════════════════════════════════
#  Appearance
# ═══════════════════════════════════════════════════════════════════

class ExposureCorrection(nn.Module):
    """Per-train-image colour affine c' = (I + D_i) c + b_i, starting at identity.

    Absorbs exposure, white-balance and lighting differences between images so
    they are not baked into the splats' colours. The mean correction over all
    images is subtracted, so a shared colour shift stays in the splats (test
    views and the export render without any correction).
    """

    def __init__(self, n_images: int):
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(n_images, 3, 4))

    def affine(self) -> Tensor:
        """(n_images, 3, 4) effective [M | b] per image."""
        d = self.delta - self.delta.mean(dim=0, keepdim=True)
        eye = torch.eye(3, 4, device=d.device, dtype=d.dtype)
        return eye + d

    def forward(self, img: Tensor, index: int) -> Tensor:
        a = self.affine()[index]
        return img @ a[:, :3].T + a[:, 3]


# ═══════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GsplatOptions:
    num_iterations: int = 7000
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    rasterize_mode: str = "antialiased"
    densify_strategy: str = "mcmc"
    absgrad: bool = False
    grad_threshold: float = 0.0002
    absgrad_threshold: float = 0.0008
    densify_from: int = 500
    densify_until: int = 3500
    densify_interval: int = 100
    opacity_reset_interval: int = 3000
    max_gaussians: int = 1_000_000
    max_screen_radius: float = 20.0
    scale_reg: float = 0.01
    max_aniso_ratio: float = 10.0
    mcmc_opacity_reg: float = 0.01
    mcmc_scale_reg: float = 0.01
    mcmc_noise_lr: float = 5e5
    # MCMC keeps relocating dead splats until here. It used to stop at densify_until
    # (50% of the run); gsplat's reference schedule stops at 25k of 30k.
    mcmc_refine_stop: Optional[int] = None
    appearance: str = "none"
    warmup_iters: int = 0
    eval_interval: int = 1000
    save_every: int = 0

    def __post_init__(self) -> None:
        for name, value, allowed in (("rasterize_mode", self.rasterize_mode, RASTERIZE_MODES),
                                     ("densify_strategy", self.densify_strategy, DENSIFY_STRATEGIES),
                                     ("appearance", self.appearance, APPEARANCE_MODES)):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
        if not 0 <= self.sh_degree <= MAX_SH_DEGREE:
            raise ValueError(f"sh_degree must be 0..{MAX_SH_DEGREE}, got {self.sh_degree}")

    def active_sh_degree(self, iteration: int) -> int:
        """Bands switched on so far: one more every `sh_degree_interval` iterations."""
        if self.sh_degree_interval <= 0:
            return self.sh_degree
        return min(iteration // self.sh_degree_interval, self.sh_degree)

    def describe(self) -> str:
        dens = self.densify_strategy + (" + absgrad" if self.absgrad and self.densify_strategy == "default"
                                        else "")
        sh = (f"SH ≤ {self.sh_degree}, +1 ступінь / {self.sh_degree_interval} іт."
              if self.sh_degree and self.sh_degree_interval else f"SH {self.sh_degree}")
        return (f"gsplat ({self.rasterize_mode}, {sh}, densify {dens}, "
                f"appearance {self.appearance})")


def train_gsplat(opts: GsplatOptions, *, xyz0: Tensor, rgb0: Tensor, init_scale: Tensor,
                 train_cams: List[Camera], test_cams: List[Camera], extent: float,
                 device: torch.device, output_path: str,
                 cams_warm: Optional[List[Camera]] = None) -> SplatModel:
    """Train on `train_cams` and save the .pt to `output_path` (see save_splat)."""
    from aerosplat.gs.evaluate import mean_test_psnr

    mcmc = opts.densify_strategy == "mcmc"
    absgrad = opts.absgrad and not mcmc
    raster = GsplatRasterizer(opts.rasterize_mode, absgrad=absgrad)
    log(f"  Растерізатор: {opts.describe()}")
    if opts.absgrad and mcmc:
        log("  ⚠️  absgrad ігнорується з densify_strategy=mcmc (MCMC не дивиться на градієнти).")

    model = SplatModel(xyz0, rgb0, init_scale, opts.sh_degree,
                       init_opacity=0.5 if mcmc else 0.1).to(device)
    optimizers = create_optimizers(model, extent)
    mcmc_dens: Optional[McmcDensifier] = None
    densifier: Optional[GsplatDensifier] = None
    if mcmc:
        refine_stop = opts.mcmc_refine_stop if opts.mcmc_refine_stop is not None else opts.densify_until
        mcmc_dens = McmcDensifier(model, opts.max_gaussians, opts.densify_from, refine_stop,
                                  opts.densify_interval, opts.mcmc_noise_lr, extent)
        log(f"  MCMC: релокація з {opts.densify_from} до {refine_stop} ітерації")
    else:
        densifier = GsplatDensifier(
            model, opts.absgrad_threshold if absgrad else opts.grad_threshold, absgrad=absgrad)
    exposure: Optional[ExposureCorrection] = None
    exposure_opt: Optional[torch.optim.Optimizer] = None
    if opts.appearance == "exposure":
        exposure = ExposureCorrection(len(train_cams)).to(device)
        exposure_opt = torch.optim.Adam(exposure.parameters(), lr=EXPOSURE_LR, eps=1e-15)

    bg_color = torch.zeros(3, device=device)
    sampler = CameraSampler(len(train_cams))
    losses: List[float] = []
    t_loop = time.time()
    n_iter = opts.num_iterations

    for iteration in range(1, n_iter + 1):
        lr_xyz = xyz_lr_at(iteration, n_iter, extent)
        optimizers["means"].param_groups[0]["lr"] = lr_xyz
        sh_deg = opts.active_sh_degree(iteration)

        pick = sampler.next()
        cam = (cams_warm[pick] if (cams_warm is not None and iteration <= opts.warmup_iters)
               else train_cams[pick])
        assert cam.image is not None
        gt = cam.image.to(device, non_blocking=True)

        params = model.get_params()
        rendered, meta = raster.rasterize(params, cam, cam.view_matrix(device), bg_color,
                                          sh_degree=sh_deg)
        if densifier is not None:
            meta["means2d"].retain_grad()
        if exposure is not None:
            rendered = exposure(rendered, pick)

        loss = photometric_loss(rendered.permute(2, 0, 1), gt.permute(2, 0, 1))
        if opts.scale_reg:
            loss = loss + opts.scale_reg * anisotropy_penalty(model.splats["scales"],
                                                              opts.max_aniso_ratio)
        if mcmc:
            loss = loss + mcmc_regularizer(params, extent, opts.mcmc_opacity_reg,
                                           opts.mcmc_scale_reg)
        loss.backward()

        if densifier is not None:
            densifier.update_stats(meta)
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if exposure_opt is not None:
            exposure_opt.step()
            exposure_opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))

        # ── Densification ───────────────────────────────────────────
        if mcmc_dens is not None:
            mcmc_dens.step(optimizers, iteration, meta, lr_xyz)
        elif densifier is not None:
            if (opts.densify_from <= iteration <= opts.densify_until
                    and iteration % opts.densify_interval == 0):
                densifier.densify_and_prune(
                    optimizers, extent=extent, max_gaussians=opts.max_gaussians,
                    max_screen_radius=(float(opts.max_screen_radius)
                                       if opts.max_screen_radius and iteration >= 1000 else None))
                torch.cuda.empty_cache()
            if (opts.opacity_reset_interval and iteration % opts.opacity_reset_interval == 0
                    and iteration < opts.densify_until):
                log(f"    Скидання opacity на ітерації {iteration}")
                densifier.reset_opacity(optimizers)

        # ── Logging ─────────────────────────────────────────────────
        if iteration % 50 == 0 or iteration == 1:
            with torch.no_grad():
                psnr_val = psnr(rendered.clamp(0, 1), gt)
            per_it = (time.time() - t_loop) / iteration
            eta = per_it * (n_iter - iteration)
            log(f"  [{iteration:>6}/{n_iter}] loss {loss.item():.4f} | "
                f"PSNR {psnr_val:5.2f} | N {model.n_gaussians:,} | SH {sh_deg} | "
                f"lr_xyz {lr_xyz:.2e} | {per_it*1000:6.1f} ms/it | ETA {eta/60:.1f} min")

        if test_cams and opts.eval_interval and iteration % opts.eval_interval == 0:
            raster.sh_degree = sh_deg
            test_psnr = mean_test_psnr(model.get_params(), raster, test_cams, device)
            log(f"    Test PSNR @ {iteration}: {test_psnr:.2f} dB ({len(test_cams)} кадрів)")

        if opts.save_every and iteration % opts.save_every == 0:
            save_splat(model, output_path, opts.rasterize_mode)

    extra: Dict[str, Any] = {}
    if exposure is not None:
        extra["appearance"] = {"kind": "exposure", "images": [c.name for c in train_cams],
                               "affine": exposure.affine().detach().cpu()}
    save_splat(model, output_path, opts.rasterize_mode, extra)
    log(f"\n  Середній loss: {sum(losses)/len(losses):.4f}")
    return model
