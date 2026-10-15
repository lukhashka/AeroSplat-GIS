"""What the server can show: one scene per Site (its latest successful run), plus
legacy `data/<ds>_*.pt` outputs until every dataset is migrated (ADR-0006).

No torch here: this only resolves paths and the `scene-meta.metric` block.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from aerosplat.geo.crs import GeoPoint
from aerosplat.store.catalog import Catalog
from aerosplat.store.models import Run, Site
from aerosplat.store.paths import CATALOG_FILE

# Product kinds a run registers (Catalog.finish_run) and the server reads.
SPLAT = "splat"            # gs/splat.pt
CLOUD = "cloud"            # gs/dense.pt or gs/cloud.pt, whichever the run trained from
REGISTRATION = "registration_report"


@dataclass(frozen=True)
class SceneSource:
    name: str
    cloud: str
    splat: str
    metric: Optional[dict[str, Any]]
    kind: str                         # "catalog" | "legacy"
    site: Optional[str] = None
    run: Optional[str] = None
    aliases: tuple[str, ...] = field(default=())


def is_georeferenced(cfg: dict[str, Any]) -> bool:
    """A run's geometry is metric Site ENU only when SfM was seeded with known georeferenced poses."""
    return cfg.get("camera_source") == "earthstudio_json" and bool(cfg.get("known_poses", True))


def metric_block(origin: GeoPoint, *, earth_model: str, source: str,
                 frame_error_m: Optional[float] = None, **extra: Any) -> dict[str, Any]:
    """`scene-meta.metric`: how served coordinates map to the Earth.

    Served xyz are ENU metres minus `scene-meta.center`, i.e.
    enu = xyz_served + center, then WGS84 via the origin below (aerosplat.geo.crs.EnuFrame).
    """
    block: dict[str, Any] = {
        "units_to_metres": 1.0,
        "crs": "ENU",
        "datum": "WGS84 (EPSG:4979)",
        "axes": "x=east y=north z=up (before the viewer's up-axis rotation)",
        "origin": {"lat": origin.lat, "lon": origin.lon, "h": origin.h},
        "served_offset": "enu = xyz + center",
        "earth_model": earth_model,
        # Known discrepancy between this run's frame and true WGS84 Site ENU
        # (sphere-legacy runs only); None when the run was built in WGS84.
        "frame_error_m": frame_error_m,
        "source": source,
    }
    block.update(extra)
    return block


def _read_json(path: Optional[Path]) -> Optional[dict[str, Any]]:
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def run_metric(catalog: Catalog, site: Site, run: Run) -> Optional[dict[str, Any]]:
    if not run.georeferenced:
        return None
    frame_error = None
    if run.earth_model != "wgs84":
        report = _read_json(catalog.product_path(site, run, REGISTRATION)) or {}
        frame_error = (report.get("legacy_shift") or {}).get("max_m")
    return metric_block(site.origin, earth_model=run.earth_model, source="catalog",
                        frame_error_m=frame_error, site=site.key, epoch=run.epoch_id, run=run.id)


def catalog_scenes(catalog: Catalog) -> dict[str, SceneSource]:
    out: dict[str, SceneSource] = {}
    sites = catalog.list_sites()
    bare_counts: dict[str, int] = {}
    for s in sites:
        bare_counts[s.id] = bare_counts.get(s.id, 0) + 1
    for site in sites:
        run = catalog.latest_run(site, with_products=(SPLAT, CLOUD))
        if run is None:
            continue
        splat, cloud = catalog.product_path(site, run, SPLAT), catalog.product_path(site, run, CLOUD)
        aliases = (site.id,) if bare_counts[site.id] == 1 else ()
        out[site.key] = SceneSource(
            name=site.key, cloud=str(cloud or ""), splat=str(splat or ""),
            metric=run_metric(catalog, site, run), kind="catalog", site=site.key, run=run.id,
            aliases=aliases)
    return out


def legacy_scenes(data_dir: Path) -> dict[str, SceneSource]:
    """`aerosplat run --dataset X` outputs (data/X_{dense,cloud,splat}.pt), not yet migrated."""
    out: dict[str, SceneSource] = {}
    if not data_dir.is_dir():
        return out
    names = {p.name.rsplit("_", 1)[0] for pattern in ("*_splat.pt", "*_dense.pt", "*_cloud.pt")
             for p in data_dir.glob(pattern)}
    for ds in sorted(names):
        cfg = (_read_json(data_dir / ds / "run" / "manifest.json") or {}).get("config") or {}
        cloud_name = "cloud" if cfg.get("cloud_source") == "sparse" else "dense"
        cloud = data_dir / f"{ds}_{cloud_name}.pt"
        if not cloud.is_file():
            cloud = data_dir / f"{ds}_{'dense' if cloud_name == 'cloud' else 'cloud'}.pt"
        metric = None
        geo = _read_json(data_dir / ds / "sparse_known" / "geo_registration.json")
        if geo and is_georeferenced(cfg):
            o = geo.get("enu_origin") or {}
            metric = metric_block(
                GeoPoint(lat=o["latitude"], lon=o["longitude"], h=o["altitude"]),
                earth_model=("wgs84" if str(geo.get("earth_model", "")).startswith("WGS84")
                             else "sphere-legacy"),
                source="legacy", origin_source=geo.get("origin_source", "frames_mean (legacy)"))
        out[ds] = SceneSource(name=ds, cloud=str(cloud), splat=str(data_dir / f"{ds}_splat.pt"),
                              metric=metric, kind="legacy")
    return out


def discover_scenes(projects_root: Optional[Path], data_dir: Optional[Path]
                    ) -> dict[str, SceneSource]:
    """name -> scene. Catalog sites win over legacy datasets of the same name.

    The catalog is only opened if it already exists: the server never
    creates a project tree as a side effect of starting.
    """
    scenes: dict[str, SceneSource] = {}
    if data_dir is not None:
        scenes.update(legacy_scenes(data_dir))
    if projects_root is not None and (projects_root / CATALOG_FILE).is_file():
        catalog = Catalog.open(projects_root)
        try:
            found = catalog_scenes(catalog)
        finally:
            catalog.close()
        for src in found.values():
            for alias in src.aliases:
                scenes[alias] = src
        scenes.update(found)
    return scenes
