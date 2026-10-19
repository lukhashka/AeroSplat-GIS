"""The twin side of VPS (ground station, GPU): a site run as a `Renderer` and its terrain.

Read-only: a run folder is opened without the project catalog (nothing under `projects/` is
written, not even SQLite's WAL); the splat is the run's WGS84 Site ENU product
(`gs/splat_wgs84.pt` for Earth Studio runs, ADR-0006), served through the render service's
`RenderEngine` in-process with the scene pinned by path.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from aerosplat.geo.crs import GeoPoint
from aerosplat.vps.build import View
from aerosplat.vps.geometry import Intrinsics

SPLAT_CANDIDATES = ("gs/splat_wgs84.pt", "gs/splat.pt")


@dataclass
class SiteRun:
    run_dir: Path
    run_id: str
    site_id: str
    origin: GeoPoint
    earth_model: str
    splat: Path
    dtm: np.ndarray
    dsm: np.ndarray
    x0: float
    y0: float
    res_m: float
    tallest: Optional[dict[str, Any]]      # building with the largest height (semantics)

    @classmethod
    def open(cls, run_dir: Path) -> "SiteRun":
        run_dir = Path(run_dir)
        reg = json.loads((run_dir / "reports" / "registration.json").read_text(encoding="utf-8"))
        o = reg["enu_origin"]
        model = (reg.get("earth_model") or {}).get("id", "wgs84")
        splat = next((run_dir / p for p in SPLAT_CANDIDATES if (run_dir / p).is_file()), None)
        if splat is None:
            raise FileNotFoundError(f"{run_dir}: no trained splat ({', '.join(SPLAT_CANDIDATES)})")
        if model != "wgs84" and splat.name != "splat_wgs84.pt":
            raise ValueError(f"{run_dir}: run in {model} without the WGS84 Site ENU splat")
        sem = json.loads((run_dir / "semantics" / "semantics.json").read_text(encoding="utf-8"))
        t = sem["terrain"]
        with np.load(run_dir / "semantics" / "terrain.npz") as z:
            dtm, dsm = z["dtm"].astype(np.float64), z["dsm"].astype(np.float64)
        b = max(sem.get("buildings") or [{}], key=lambda x: x.get("height_m", -1.0)) or None
        return cls(run_dir=run_dir, run_id=run_dir.name, site_id=run_dir.parents[3].name,
                   origin=GeoPoint(lat=o["lat"], lon=o["lon"], h=o["h"]), earth_model=model,
                   splat=splat, dtm=dtm, dsm=dsm, x0=float(t["x0"]), y0=float(t["y0"]),
                   res_m=float(t["res_m"]), tallest=b if b and "footprint" in b else None)

    def _cell(self, e: float, n: float) -> tuple[int, int]:
        return int((n - self.y0) / self.res_m), int((e - self.x0) / self.res_m)

    def ground_z(self, e: float, n: float, r_m: float = 5.0) -> float:
        """Median DTM within `r_m` (falls back to the site median where the DTM has holes)."""
        i, j = self._cell(e, n)
        k = max(1, int(r_m / self.res_m))
        win = self.dtm[max(i - k, 0):i + k + 1, max(j - k, 0):j + k + 1]
        win = win[np.isfinite(win)]
        return float(np.median(win)) if win.size else float(np.nanmedian(self.dtm))

    def top_z(self, e: float, n: float, r_m: float) -> float:
        """Highest DSM within `r_m` (NaN-free; -inf where nothing is known)."""
        i, j = self._cell(e, n)
        k = max(1, int(math.ceil(r_m / self.res_m)))
        win = self.dsm[max(i - k, 0):i + k + 1, max(j - k, 0):j + k + 1]
        win = win[np.isfinite(win)]
        return float(win.max()) if win.size else -math.inf

    def clear(self, e: float, n: float, u: float, r_m: float = 10.0, margin_m: float = 5.0) -> bool:
        return u >= self.top_z(e, n, r_m) + margin_m

    def tallest_centre(self) -> Optional[tuple[float, float]]:
        if not self.tallest:
            return None
        fp = np.asarray(self.tallest["footprint"], float).reshape(-1, 2)
        return float(fp[:, 0].mean()), float(fp[:, 1].mean())

    def splat_sha256(self) -> str:
        h = hashlib.sha256()
        with open(self.splat, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        return h.hexdigest()


class EngineRenderer:
    """`Renderer` over the render service's engine (gsplat, CUDA). rgb + z-depth per view."""

    def __init__(self, run: SiteRun):
        from aerosplat.render.engine import RenderEngine
        from aerosplat.store.scenes import SceneSource, metric_block
        self.engine = RenderEngine(None, None)
        self.name = f"vps-{run.site_id}-{run.run_id}"
        self.engine.pin_scene(SceneSource(
            name=self.name, cloud="", splat=str(run.splat), kind="catalog",
            metric=metric_block(run.origin, earth_model=run.earth_model, source="vps-bench",
                                run=run.run_id)))
        self.frames = 0

    def _request(self, view: View, width: int, height: int, K: Intrinsics) -> dict[str, Any]:
        from aerosplat.render.models import Position
        fx, fy, cx, cy = K
        if abs(fx - fy) > 1e-6 * fx or abs(cx - (width / 2 - 0.5)) > 1e-6 \
                or abs(cy - (height / 2 - 0.5)) > 1e-6:
            raise ValueError("the render service draws centred square-pixel pinholes only")
        fov = math.degrees(2.0 * math.atan(height / 2.0 / fy))
        return dict(position=Position(e=view.C[0], n=view.C[1], u=view.C[2]), yaw_deg=view.yaw_deg,
                    pitch_deg=view.pitch_deg, roll_deg=view.roll_deg, width=width, height=height,
                    fov_deg=fov)

    def render_rgb(self, view: View, width: int, height: int, K: Intrinsics) -> np.ndarray:
        """The colour frame only (a camera frame: SIM-3 needs no depth)."""
        from aerosplat.render.models import RenderRequest
        rgb = self.engine.render(self.name, RenderRequest(**self._request(view, width, height, K),
                                                          output="rgb")).image
        self.frames += 1
        return rgb

    def render(self, view: View, width: int, height: int, K: Intrinsics
               ) -> tuple[np.ndarray, np.ndarray]:
        from aerosplat.render.models import RenderRequest
        base = self._request(view, width, height, K)
        rgb = self.engine.render(self.name, RenderRequest(**base, output="rgb")).image
        d = self.engine.render(self.name, RenderRequest(**base, output="depth"))
        max_m = float(d.meta["depth"]["max_m"])
        depth = d.image.astype(np.float32) * np.float32(max_m / 65535.0)
        self.frames += 1
        return rgb, depth

    def close(self) -> None:
        self.engine.unload_all()


def software_versions() -> dict[str, Any]:
    import aerosplat
    out: dict[str, Any] = {"aerosplat": aerosplat.__version__, "numpy": np.__version__}
    try:
        import torch
        out["torch"] = torch.__version__
        import gsplat
        out["gsplat"] = getattr(gsplat, "__version__", "?")
    except ImportError:
        pass
    return out
