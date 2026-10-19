"""Orthophoto of a run: straight-down orthographic 3DGS renders mosaicked into a north-up
GeoTIFF / COG in the target CRS.

The renders are made **in the target grid**, not resampled into it afterwards. Over a
site the map Site ENU -> projected CRS is a similarity to well below a millimetre (a scale
k ~ 0.9996-1.001 and a rotation by the meridian convergence, `PlaneFit`), so an ortho
camera yawed by that rotation, with k / px pixels per ENU metre, puts its pixel centres
exactly on the CRS grid. Each tile is one such render whose pixel block is a window of
the global grid; tiles are pasted, never blended or resampled, so the mosaic is as sharp
as a single render and seams cannot shift.

Each tile is verified twice before it is written (fail-safe, not fail-silent):

* **plan**: the camera's own ground grid (aerosplat.render.camera), pushed through the
  *exact* projection, must land on the CRS pixel centres to `max_registration_px`;
* **render**: the `ground_grid` the renderer reports must be the planned one -- a renderer
  that silently changed the extent, the yaw or the pixel convention is refused.

Colour: the renderer composites onto a black background, i.e. returns premultiplied
colour; it is divided by the alpha render to give straight RGBA, and alpha is written as
the fourth band (GDAL's alpha mask). Alpha < 1 means the splat is not opaque there:
nothing was reconstructed, which GIS shows as transparent, not as black ground.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from aerosplat.export.georef import ExportRefused, PlaneFit, SiteProjection
from aerosplat.export.raster import NorthUpGrid, finish_raster, open_raster
from aerosplat.render import camera as cam
from aerosplat.render.models import MAX_SIDE, Position, RenderRequest

MIN_SIDE = 16                      # RenderRequest's smallest image side
DEFAULT_TILE_PX = 2048
MAX_REGISTRATION_PX = 0.05

# (request) -> (image, meta): image is (H, W, 3) uint8 for rgb, (H, W) uint8 for alpha.
Renderer = Callable[[RenderRequest], tuple[np.ndarray, dict[str, Any]]]


@dataclass(frozen=True)
class OrthoTile:
    """One render: the grid window it covers (c0, r0, w, h) and the request that draws it.

    The request may be larger than the window (edge tiles under MIN_SIDE); the extra
    pixels fall outside the grid and are cropped.
    """

    c0: int
    r0: int
    w: int
    h: int
    yaw_deg: float
    centre_enu: tuple[float, float]
    extent_m: float
    req_w: int
    req_h: int

    def request(self, output: str, sh_degree: Optional[int] = None) -> RenderRequest:
        return RenderRequest(
            mode="ortho", position=Position(e=self.centre_enu[0], n=self.centre_enu[1]),
            yaw_deg=self.yaw_deg, pitch_deg=-90.0, roll_deg=0.0, width=self.req_w,
            height=self.req_h, extent_m=self.extent_m, output=output,
            background=(0.0, 0.0, 0.0), sh_degree=sh_degree)

    def planned_ground_grid(self) -> dict[str, Any]:
        """The ground grid the renderer must report for this tile (height-independent)."""
        V = cam.view_matrix(np.array([self.centre_enu[0], self.centre_enu[1], 0.0]),
                            self.yaw_deg, -90.0, 0.0)
        K = cam.ortho_intrinsics(self.req_w, self.req_h, self.extent_m)
        grid = cam.ortho_ground_grid(V, K, self.req_w, self.req_h)
        assert grid is not None          # pitch -90 by construction
        return grid


def grid_yaw_and_scale(fit: PlaneFit) -> tuple[float, float]:
    """(yaw_deg, enu_m_per_crs_m): the ortho camera heading whose image +x is CRS +easting,
    and the ENU length of one CRS metre along it."""
    dir_c = np.linalg.solve(fit.L, np.array([1.0, 0.0]))       # ENU step per CRS metre east
    yaw = float(np.degrees(np.arctan2(-dir_c[1], dir_c[0])))  # camera right = (cos, -sin)
    return yaw, float(np.linalg.norm(dir_c))


def plan_tiles(grid: NorthUpGrid, fit: PlaneFit, tile_px: int = DEFAULT_TILE_PX
               ) -> list[OrthoTile]:
    if not MIN_SIDE <= tile_px <= MAX_SIDE:
        raise ExportRefused(f"tile size must be {MIN_SIDE}..{MAX_SIDE} px, got {tile_px}")
    yaw, enu_per_m = grid_yaw_and_scale(fit)
    enu_per_px = enu_per_m * grid.px
    tiles = []
    for r0 in range(0, grid.height, tile_px):
        for c0 in range(0, grid.width, tile_px):
            w, h = min(tile_px, grid.width - c0), min(tile_px, grid.height - r0)
            rw, rh = max(w, MIN_SIDE), max(h, MIN_SIDE)
            # Window centre in pixel-edge coordinates -> CRS -> ENU (at the ground height).
            x, y = grid.edge_xy(c0 + rw / 2.0, r0 + rh / 2.0)
            e, n = fit.to_enu(x, y)
            tiles.append(OrthoTile(c0=c0, r0=r0, w=w, h=h, yaw_deg=yaw,
                                   centre_enu=(float(e), float(n)), extent_m=rw * enu_per_px,
                                   req_w=rw, req_h=rh))
    return tiles


def registration_error_px(tile: OrthoTile, grid: NorthUpGrid, proj: SiteProjection,
                          u_ref: float) -> float:
    """Worst distance, in pixels, between where the tile camera puts a pixel centre
    (exact projection) and where the grid says that pixel is."""
    g = tile.planned_ground_grid()
    p00, du, dv = (np.asarray(g[k], float) for k in ("pixel00_enu", "du_enu", "dv_enu"))
    cols = np.array([0, tile.req_w - 1, 0, tile.req_w - 1, tile.req_w // 2], float)
    rows = np.array([0, 0, tile.req_h - 1, tile.req_h - 1, tile.req_h // 2], float)
    en = p00 + cols[:, None] * du + rows[:, None] * dv
    xyz = proj.enu_to_crs(np.column_stack([en, np.full(len(en), u_ref)]))
    dx = xyz[:, 0] - grid.centre_x(tile.c0 + cols)
    dy = xyz[:, 1] - grid.centre_y(tile.r0 + rows)
    return float(np.hypot(dx, dy).max() / grid.px)


def _check_reported(tile: OrthoTile, meta: dict[str, Any], grid_px: float) -> None:
    got = meta.get("ground_grid")
    if not got:
        raise ExportRefused("the renderer returned no ground_grid for a straight-down "
                            "ortho render: cannot georeference the tile")
    want = tile.planned_ground_grid()
    m = want["m_per_px"]
    for key in ("pixel00_enu", "du_enu", "dv_enu"):
        err = float(np.abs(np.asarray(got[key], float) - np.asarray(want[key], float)).max())
        if err / m > 1e-3:
            raise ExportRefused(f"tile at ({tile.c0}, {tile.r0}): the renderer's {key} "
                                f"differs from the plan by {err / m:.4f} px -- refusing "
                                f"to write a misregistered orthophoto")
    shape = (meta.get("height"), meta.get("width"))
    if shape != (tile.req_h, tile.req_w):
        raise ExportRefused(f"tile at ({tile.c0}, {tile.r0}): rendered {shape}, "
                            f"asked {(tile.req_h, tile.req_w)}")


def straight_rgba(rgb_premult: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """(H,W,3) colour composited on black + (H,W) alpha, both uint8 -> (H,W,4) straight RGBA."""
    a = alpha.astype(np.float32)
    c = rgb_premult.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        straight = np.where(a[..., None] > 0, c * 255.0 / a[..., None], 0.0)
    out = np.empty(alpha.shape + (4,), np.uint8)
    out[..., :3] = np.clip(np.round(straight), 0, 255).astype(np.uint8)
    out[..., 3] = alpha
    return out


def export_ortho(path: Path, *, renderer: Renderer, proj: SiteProjection, fit: PlaneFit,
                 grid: NorthUpGrid, tile_px: int = DEFAULT_TILE_PX, cog: bool = True,
                 sh_degree: Optional[int] = None,
                 max_registration_px: float = MAX_REGISTRATION_PX,
                 tags: Optional[dict[str, Any]] = None,
                 on_tile: Optional[Callable[[int, int], None]] = None) -> dict[str, Any]:
    """Render, verify and mosaic the orthophoto. Returns what was done (for the manifest)."""
    tiles = plan_tiles(grid, fit, tile_px)
    worst = max(registration_error_px(t, grid, proj, fit.u_ref) for t in tiles)
    if worst > max_registration_px:
        raise ExportRefused(f"planned tiles miss the CRS grid by {worst:.3f} px "
                            f"(limit {max_registration_px}): area too large for one grid")
    opaque = 0
    covered = 0
    with open_raster(path, grid, proj, count=4, dtype="uint8", rgba=True, tags=tags) as dst:
        from rasterio.windows import Window

        for i, t in enumerate(tiles):
            rgb, meta_rgb = renderer(t.request("rgb", sh_degree))
            alpha, meta_a = renderer(t.request("alpha", sh_degree))
            _check_reported(t, meta_rgb, grid.px)
            _check_reported(t, meta_a, grid.px)
            if rgb.shape != (t.req_h, t.req_w, 3) or alpha.shape != (t.req_h, t.req_w):
                raise ExportRefused(f"tile at ({t.c0}, {t.r0}): unexpected image shapes "
                                    f"{rgb.shape} / {alpha.shape}")
            rgba = straight_rgba(rgb, alpha)[:t.h, :t.w]
            opaque += int((rgba[..., 3] == 255).sum())
            covered += int((rgba[..., 3] >= 128).sum())
            dst.write(np.moveaxis(rgba, -1, 0), window=Window(t.c0, t.r0, t.w, t.h))
            if on_tile is not None:
                on_tile(i + 1, len(tiles))
    finish_raster(path, cog=cog)
    total = grid.width * grid.height
    return {"tiles": len(tiles), "tile_px": tile_px, "registration_px_max": worst,
            "yaw_deg": tiles[0].yaw_deg, "coverage_alpha_ge_0.5": covered / total,
            "opaque_fraction": opaque / total, "sh_degree": sh_degree}
