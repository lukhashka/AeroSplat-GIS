"""Atlas builder (ground station): render keyframes of the twin, extract features, lift them to
3D with the rendered depth, store everything in an `Atlas`.

The builder does not know about gsplat: it takes any `Renderer` (rgb + z-depth for a view). The
twin renderer is `aerosplat.vps.twin.EngineRenderer` (GPU); the tests use a synthetic one.

Depth comes from the splat's expected depth, which blends foreground and background at a
silhouette: a keypoint on a roof edge would get a depth between the roof and the street. Such
points are dropped (relative depth spread in a window around the keypoint), as are points
without depth (sky, holes) and points farther than `max_depth_m` (the twin is coarse there).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Protocol

import numpy as np

from aerosplat.vps.atlas import Atlas, concat_keyframes
from aerosplat.vps.features import Extractor, global_descriptor, require_cv2
from aerosplat.vps.geometry import Intrinsics, backproject, look_point, rotation


@dataclass(frozen=True)
class View:
    C: tuple[float, float, float]     # camera centre, Site ENU
    yaw_deg: float
    pitch_deg: float
    roll_deg: float = 0.0

    @property
    def R(self) -> np.ndarray:
        return rotation(self.yaw_deg, self.pitch_deg, self.roll_deg)

    def as_dict(self) -> dict[str, Any]:
        return {"C": [float(c) for c in self.C], "yaw_deg": self.yaw_deg,
                "pitch_deg": self.pitch_deg, "roll_deg": self.roll_deg}


class Renderer(Protocol):
    def render(self, view: View, width: int, height: int, K: Intrinsics
               ) -> tuple[np.ndarray, np.ndarray]:
        """(rgb (H, W, 3) uint8, z-depth (H, W) float32 metres, 0 where nothing was hit).
        K is in OpenCV convention."""
        ...


@dataclass(frozen=True)
class DepthFilter:
    window: int = 2                  # half size of the window (5 x 5)
    max_rel_spread: float = 0.06     # (max - min) / median depth inside the window
    max_depth_m: float = 250.0
    min_depth_m: float = 1.0


def keypoint_depths(depth: np.ndarray, xy: np.ndarray, flt: DepthFilter) -> np.ndarray:
    """z-depth at each keypoint (bilinear), NaN where the filter rejects the window around it."""
    h, w = depth.shape
    out = np.full(len(xy), np.nan)
    if len(xy) == 0:
        return out
    ix = np.rint(xy[:, 0]).astype(int)
    iy = np.rint(xy[:, 1]).astype(int)
    r = flt.window
    inside = (ix >= r + 1) & (iy >= r + 1) & (ix < w - r - 1) & (iy < h - r - 1)
    for i in np.flatnonzero(inside):
        win = depth[iy[i] - r:iy[i] + r + 1, ix[i] - r:ix[i] + r + 1]
        if np.any(win <= 0):
            continue
        med = float(np.median(win))
        if not flt.min_depth_m <= med <= flt.max_depth_m:
            continue
        if (float(win.max()) - float(win.min())) / med > flt.max_rel_spread:
            continue
        # bilinear at the sub-pixel keypoint: the nearest pixel is up to half a pixel of
        # depth gradient away (decimetres on an oblique view of the ground)
        x, y = float(xy[i, 0]), float(xy[i, 1])
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        q = depth[y0:y0 + 2, x0:x0 + 2]
        out[i] = float((q[0, 0] * (1 - fx) + q[0, 1] * fx) * (1 - fy)
                       + (q[1, 0] * (1 - fx) + q[1, 1] * fx) * fy)
    return out


def grid_views(center_en: tuple[float, float], half_m: float, spacing_m: float,
               alts_agl: Iterable[float], yaws_deg: Iterable[float], pitch_deg: float,
               ground_z: Callable[[float, float], float],
               clear: Optional[Callable[[float, float, float], bool]] = None) -> list[View]:
    """Keyframe views over a square area: every grid node x height above ground x heading.
    `clear(e, n, u)` may veto a node inside an obstacle (a tower taller than the height)."""
    n = int(np.floor(half_m / spacing_m))
    offs = np.arange(-n, n + 1) * spacing_m
    out = []
    for de in offs:
        for dn in offs:
            e, nn = center_en[0] + float(de), center_en[1] + float(dn)
            g = ground_z(e, nn)
            for alt in alts_agl:
                u = g + float(alt)
                if clear is not None and not clear(e, nn, u):
                    continue
                for yaw in yaws_deg:
                    out.append(View((e, nn, u), float(yaw), float(pitch_deg)))
    return out


def build_atlas(renderer: Renderer, views: list[View], width: int, height: int, K: Intrinsics,
                extractor: Extractor, meta: dict[str, Any], depth_filter: DepthFilter = DepthFilter(),
                min_points: int = 30, log: Optional[Callable[[str], None]] = None
                ) -> tuple[Atlas, dict[str, Any]]:
    """Render, extract and lift every view; keyframes with fewer than `min_points` usable 3D
    points are left out (counted in the stats)."""
    cv2 = require_cv2()
    kfs: list[dict[str, np.ndarray]] = []
    t_render = t_extract = t_lift = 0.0
    dropped = 0
    n_kp = n_kept = 0
    for i, v in enumerate(views):
        t0 = time.perf_counter()
        rgb, depth = renderer.render(v, width, height, K)
        t1 = time.perf_counter()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        f = extractor.detect(gray)
        t2 = time.perf_counter()
        d = keypoint_depths(depth, f.xy, depth_filter)
        ok = np.isfinite(d)
        R, C = v.R, np.asarray(v.C, float)
        xyz = backproject(R, C, K, f.xy[ok].astype(float), d[ok])
        t3 = time.perf_counter()
        t_render += t1 - t0
        t_extract += t2 - t1
        t_lift += t3 - t2
        n_kp += len(f)
        n_kept += int(ok.sum())
        if ok.sum() < min_points:
            dropped += 1
            continue
        med = float(np.median(d[ok]))
        kfs.append({"R": R, "C": C, "look": C + med * R[2], "global": global_descriptor(gray),
                    "xy": f.xy[ok], "desc": f.desc[ok], "xyz_enu": xyz})
        if log is not None and (i + 1) % 100 == 0:
            log(f"   keyframes {i + 1}/{len(views)}: {n_kept} points kept of {n_kp}")
    if not kfs:
        raise ValueError("no keyframe had enough points with depth")
    allxyz = np.concatenate([k["xyz_enu"] for k in kfs])
    anchor = np.round(np.median(allxyz, axis=0), 1)
    ground_z = float(np.percentile(allxyz[:, 2], 10))
    for k in kfs:   # look points on a common ground plane for retrieval
        k["look"] = look_point(k["R"], k["C"], ground_z)
    full = {**meta,
            "intrinsics": {"width": width, "height": height, "fx": K[0], "fy": K[1], "cx": K[2],
                           "cy": K[3], "convention": "opencv (pixel (0,0) centre at 0,0)"},
            "extractor": {"name": extractor.name, "params": extractor.params, "norm": extractor.norm},
            "depth_filter": depth_filter.__dict__.copy(), "ground_z": ground_z,
            "opencv": cv2.__version__}
    atlas = concat_keyframes(kfs, anchor, full)
    stats = {"views": len(views), "keyframes": atlas.n_keyframes, "dropped_views": dropped,
             "keypoints": n_kp, "points_kept": n_kept, "render_s": round(t_render, 2),
             "extract_s": round(t_extract, 2), "lift_s": round(t_lift, 2),
             "atlas_mb": round(atlas.nbytes() / 2 ** 20, 2)}
    return atlas, stats
