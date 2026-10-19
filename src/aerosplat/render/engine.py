"""3DGS render engine: keeps scenes on the GPU and renders them from a pose (persp / ortho).

One process, one GPU, one render at a time. A scene stays loaded until it has been idle
for `idle_s` (so a training job gets its VRAM back) or another scene displaces it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

from aerosplat.geo.crs import EnuFrame, GeoPoint
from aerosplat.gs.gsplat_backend import NEAR_PLANE, RASTERIZE_MODES, require_gsplat
from aerosplat.gs.sh import sh_degree_of
from aerosplat.render import camera as cam
from aerosplat.render.models import RenderRequest
from aerosplat.store.scenes import SceneLookupError, SceneSource, discover_scenes, pinned_scene

SCENE_LIST_TTL_S = 3.0
DEPTH_UNSET = 0


class RenderError(ValueError):
    """A request the engine cannot honour (unknown scene, no georeference, ...)."""


class SceneNotFound(RenderError):
    pass


class GpuOutOfMemory(RuntimeError):
    pass


@dataclass
class LoadedScene:
    source: SceneSource
    path: str
    mtime: float
    params: dict[str, Tensor]
    rasterize_mode: str
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    frame: Optional[EnuFrame]
    loaded_s: float
    last_used: float = field(default_factory=time.monotonic)

    @property
    def n(self) -> int:
        return int(self.params["xyz"].shape[0])

    @property
    def sh_bands(self) -> int:
        return sh_degree_of(self.params["sh"].shape[1]) if "sh" in self.params else 0


@dataclass
class Rendered:
    image: np.ndarray            # rgb: (H,W,3) uint8 · alpha: (H,W) uint8 · depth: (H,W) uint16
    meta: dict[str, Any]


def _load_params(path: str, device: torch.device) -> tuple[dict[str, Tensor], str]:
    scene = torch.load(path, map_location="cpu", weights_only=False)
    act = scene.get("activations", {})
    if act.get("scale_is_log") or act.get("opacity_is_logit"):
        raise RenderError("splat.pt with raw (pre-activation) values is not supported")
    keys = ("xyz", "scale", "rotation", "opacity") + (("sh",) if "sh" in scene else ("rgb",))
    cpu = {k: scene[k].float() for k in keys}
    # Same row filter as the viewer's server: one NaN mean poisons the scene bounds
    # (ortho default height) and a zero quaternion has no rotation.
    keep = cpu["rotation"].norm(dim=1) > 1e-8
    for t in cpu.values():
        keep &= torch.isfinite(t.reshape(t.shape[0], -1)).all(dim=1)
    params = {k: t[keep].to(device).contiguous() for k, t in cpu.items()}
    mode = (scene.get("render") or {}).get("rasterize_mode", "classic")
    return params, mode if mode in RASTERIZE_MODES else "classic"


class RenderEngine:
    def __init__(self, projects_root: Optional[Path], data_dir: Optional[Path] = None, *,
                 idle_s: float = 600.0, max_scenes: int = 1,
                 device: Optional[torch.device] = None):
        require_gsplat()
        from gsplat import rasterization
        self._rasterization = rasterization
        self.projects_root, self.data_dir = projects_root, data_dir
        self.idle_s, self.max_scenes = idle_s, max_scenes
        self.device = device or torch.device("cuda")
        self._scenes: dict[str, LoadedScene] = {}
        self._lock = threading.RLock()
        self._listing: tuple[float, dict[str, SceneSource]] = (-1e9, {})

    # ── scenes ───────────────────────────────────────────────────────────
    def scenes(self, fresh: bool = False) -> dict[str, SceneSource]:
        with self._lock:
            now = time.monotonic()
            if fresh or now - self._listing[0] > SCENE_LIST_TTL_S:
                self._listing = (now, discover_scenes(self.projects_root, self.data_dir))
            return self._listing[1]

    def _resolve(self, name: str, run: Optional[str] = None) -> SceneSource:
        src = self.scenes().get(name) or self.scenes(fresh=True).get(name)
        if run is not None and (src is None or src.run != run):
            # Pinned to a run that is not the one the listing serves: look that run up in
            # the catalog, and refuse rather than render some other run (§5.9).
            site_key = src.site if src is not None and src.site else name
            try:
                src = pinned_scene(self.projects_root, site_key, run)
            except SceneLookupError as e:
                raise SceneNotFound(str(e)) from None
        if src is None or not src.splat or not Path(src.splat).is_file():
            raise SceneNotFound(f"no trained scene '{name}' (run the pipeline first)")
        return src

    def get_scene(self, name: str, run: Optional[str] = None) -> LoadedScene:
        with self._lock:
            src = self._resolve(name, run)
            # A pinned run is cached under its own key, so it never answers for the newest one.
            key = src.name if src.run is None else f"{src.name}@{src.run}"
            mtime = Path(src.splat).stat().st_mtime
            have = self._scenes.get(key)
            if have is not None and have.path == src.splat and have.mtime == mtime:
                have.last_used = time.monotonic()
                return have
            self._scenes.pop(key, None)
            while len(self._scenes) >= self.max_scenes:
                self._scenes.pop(min(self._scenes, key=lambda k: self._scenes[k].last_used))
            t0 = time.perf_counter()
            try:
                params, mode = _load_params(src.splat, self.device)
            except torch.cuda.OutOfMemoryError as e:
                self._scenes.clear()
                torch.cuda.empty_cache()
                raise GpuOutOfMemory("not enough GPU memory to load the scene") from e
            xyz = params["xyz"]
            origin = (src.metric or {}).get("origin")
            frame = (EnuFrame(GeoPoint(lat=origin["lat"], lon=origin["lon"], h=origin["h"]))
                     if origin else None)
            scene = LoadedScene(
                source=src, path=src.splat, mtime=mtime, params=params, rasterize_mode=mode,
                bounds_min=xyz.min(0).values.cpu().numpy(), bounds_max=xyz.max(0).values.cpu().numpy(),
                frame=frame, loaded_s=time.perf_counter() - t0)
            self._scenes[key] = scene
            return scene

    def unload_idle(self) -> list[str]:
        """Drop scenes unused for `idle_s`; returns their names."""
        with self._lock:
            now = time.monotonic()
            gone = [k for k, s in self._scenes.items() if now - s.last_used > self.idle_s]
            for k in gone:
                del self._scenes[k]
            if gone:
                torch.cuda.empty_cache()
            return gone

    def unload_all(self) -> None:
        with self._lock:
            self._scenes.clear()
            torch.cuda.empty_cache()

    def status(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            return {
                "device": torch.cuda.get_device_name(self.device),
                "vram_allocated_mb": round(torch.cuda.memory_allocated(self.device) / 2 ** 20, 1),
                "idle_unload_s": self.idle_s,
                "loaded": [{"scene": k, "gaussians": s.n, "sh_bands": s.sh_bands,
                            "rasterize_mode": s.rasterize_mode, "idle_s": round(now - s.last_used, 1),
                            "load_s": round(s.loaded_s, 2)} for k, s in self._scenes.items()],
            }

    # ── pose ─────────────────────────────────────────────────────────────
    @staticmethod
    def _to_enu(scene: LoadedScene, p, default_u: Optional[float] = None) -> np.ndarray:
        if p.is_geodetic:
            if scene.frame is None:
                raise RenderError("this scene has no georeference: use e/n/u in scene coordinates")
            h = p.h
            if h is None:
                if default_u is None:
                    raise RenderError("height (h) is required")
                return np.array([*np.ravel(scene.frame.from_geodetic(p.lat, p.lon, 0.0))[:2],
                                 default_u])
            return np.asarray(scene.frame.from_geodetic(p.lat, p.lon, h), float).reshape(3)
        u = p.u if p.u is not None else default_u
        if u is None:
            raise RenderError("height (u) is required")
        return np.array([p.e, p.n, u], float)

    def _pose(self, scene: LoadedScene, req: RenderRequest) -> tuple[np.ndarray, float, float, float]:
        top = float(scene.bounds_max[2]) + 1.0
        pitch = req.pitch
        if req.mode == "ortho" and req.position.vertical is None and abs(pitch + 90.0) > 1e-6:
            raise RenderError("a tilted orthographic camera needs a height (u or h)")
        eye = self._to_enu(scene, req.position, default_u=top if req.mode == "ortho" else None)
        yaw = req.yaw_deg
        if req.look_at is not None:
            yaw, pitch = cam.yaw_pitch_to(eye, self._to_enu(scene, req.look_at))
        return eye, yaw, pitch, req.roll_deg

    # ── render ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def render(self, scene_name: str, req: RenderRequest) -> Rendered:
        t_start = time.perf_counter()
        with self._lock:
            scene = self.get_scene(scene_name, req.run)
            eye, yaw, pitch, roll = self._pose(scene, req)
            ortho = req.mode == "ortho"
            V = cam.view_matrix(eye, yaw, pitch, roll)
            K = (cam.ortho_intrinsics(req.width, req.height, req.extent_m) if ortho
                 else cam.persp_intrinsics(req.width, req.height, req.fov_deg))
            dev = self.device
            Kt = torch.tensor([[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1]], device=dev,
                              dtype=torch.float32)
            Vt = torch.tensor(V, device=dev, dtype=torch.float32)
            p = scene.params
            if "sh" in p:
                colors = p["sh"]
                degree: Optional[int] = (scene.sh_bands if req.sh_degree is None
                                         else min(req.sh_degree, scene.sh_bands))
            else:
                colors, degree = p["rgb"], None
            want_depth = req.output == "depth"
            try:
                renders, alphas, _ = self._rasterization(
                    means=p["xyz"], quats=p["rotation"], scales=p["scale"],
                    opacities=p["opacity"].reshape(-1), colors=colors,
                    viewmats=Vt[None], Ks=Kt[None], width=req.width, height=req.height,
                    near_plane=NEAR_PLANE, sh_degree=degree, packed=False,
                    rasterize_mode=scene.rasterize_mode,
                    camera_model="ortho" if ortho else "pinhole",
                    render_mode="ED" if want_depth else "RGB")
            except torch.cuda.OutOfMemoryError as e:
                self._scenes.clear()
                torch.cuda.empty_cache()
                raise GpuOutOfMemory("not enough GPU memory to render at this size") from e
            alpha = alphas[0, ..., 0].clamp(0, 1)
            meta: dict[str, Any] = {}
            if req.output == "rgb":
                bg = torch.tensor(req.background, device=dev, dtype=torch.float32)
                rgb = (renders[0] + (1.0 - alpha)[..., None] * bg).clamp(0, 1)
                image = (rgb * 255.0 + 0.5).to(torch.uint8).cpu().numpy()
            elif req.output == "alpha":
                image = (alpha * 255.0 + 0.5).to(torch.uint8).cpu().numpy()
            else:
                depth = renders[0, ..., 0]
                valid = (alpha > 0.5) & (depth > 0)
                if bool(valid.any()):
                    d_max = float(depth[valid].max())
                    q = (depth / d_max * 65535.0).round().clamp(1, 65535)
                    image = torch.where(valid, q, torch.zeros_like(q)).to(torch.int32) \
                        .cpu().numpy().astype(np.uint16)
                else:
                    d_max = 0.0
                    image = np.zeros((req.height, req.width), np.uint16)
                meta["depth"] = {"max_m": d_max, "scale": "metres = value / 65535 * max_m",
                                 "no_data": DEPTH_UNSET,
                                 "kind": "distance along the optical axis"
                                         if not ortho else "distance from the camera plane"}
            scene.last_used = time.monotonic()
        meta.update(self._describe(scene, req, V, K, eye, yaw, pitch, roll, degree))
        meta["render_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
        return Rendered(image=image, meta=meta)

    @staticmethod
    def _describe(scene: LoadedScene, req: RenderRequest, V: np.ndarray, K, eye, yaw, pitch,
                  roll, degree) -> dict[str, Any]:
        out: dict[str, Any] = {
            "scene": scene.source.name, "run": scene.source.run, "mode": req.mode,
            "output": req.output, "width": req.width, "height": req.height,
            "frame": ("Site ENU, metres" if scene.frame is not None
                      else "scene coordinates (not georeferenced)"),
            "camera_enu": [float(x) for x in eye],
            "yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll,
            "intrinsics": {"fx": K[0], "fy": K[1], "cx": K[2], "cy": K[3]},
            "view_matrix": V.tolist(),
            "gaussians": scene.n, "sh_degree": degree, "rasterize_mode": scene.rasterize_mode,
            "scene_bounds_enu": {"min": scene.bounds_min.tolist(), "max": scene.bounds_max.tolist()},
        }
        if scene.frame is not None:
            lat, lon, h = (float(np.ravel(v)[0]) for v in scene.frame.to_geodetic(eye))
            out["camera_geo"] = {"lat": lat, "lon": lon, "h": h}
        if req.mode == "ortho":
            out["extent_m"] = req.extent_m
            grid = cam.ortho_ground_grid(V, K, req.width, req.height)
            if grid is not None:
                out["ground_grid"] = grid
        else:
            out["fov_deg"] = req.fov_deg
        return out
