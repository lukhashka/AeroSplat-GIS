"""What the server can show: one scene per Site (its latest successful run), plus
legacy `data/<ds>_*.pt` outputs until every dataset is migrated (ADR-0006).

No torch here: this only resolves paths and the `scene-meta.metric` block.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from aerosplat.geo.crs import GeoPoint
from aerosplat.geo.sphere_enu import EARTH_MODEL as SPHERE_ENU_AFFINE
from aerosplat.store.catalog import Catalog
from aerosplat.store.models import Run, RunStatus, Site
from aerosplat.store.paths import CATALOG_FILE

log = logging.getLogger(__name__)

# Product kinds a run registers (Catalog.finish_run) and the server reads.
# For georeferenced Earth Studio runs SPLAT/CLOUD are the georef stage's
# WGS84 Site ENU files (gs/*_wgs84.pt); the *_RECON kinds keep the files in
# the reconstruction frame (sphere ENU), which the server never serves.
SPLAT = "splat"            # gs/splat.pt, or gs/splat_wgs84.pt
CLOUD = "cloud"            # gs/{dense,cloud}.pt (whichever the run trained from), or *_wgs84.pt
SPLAT_RECON = "splat_recon"
CLOUD_RECON = "cloud_recon"
REGISTRATION = "registration_report"
SEMANTICS = "semantics"    # semantics/ of the run: terrain, buildings, facades, coverage
OCCUPANCY = "occupancy"    # occupancy/ of the run: voxel grid + distance field
WGS84 = "wgs84"


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


def run_earth_model(cfg: dict[str, Any]) -> str:
    """Run.earth_model of a new run: Earth Studio known poses reconstruct in sphere
    ENU and the georef stage maps the products (ADR-0006, amended 2026-10-15)."""
    return SPHERE_ENU_AFFINE if is_georeferenced(cfg) else WGS84


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
        # Known discrepancy between the served coordinates and WGS84 Site ENU:
        # the affine map's residual (sphere-enu+affine, sub-millimetre), the
        # legacy shift (sphere-legacy, ~1.5 m), None for runs built in WGS84.
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


def frame_error(report: dict[str, Any], earth_model: str) -> Optional[float]:
    """metric_block.frame_error_m from a registration report."""
    if earth_model == WGS84:
        return None
    if earth_model == SPHERE_ENU_AFFINE:
        return (report.get("sphere_enu_to_site_enu") or {}).get("residual_max_m")
    return (report.get("legacy_shift") or {}).get("max_m")


def run_metric(catalog: Catalog, site: Site, run: Run) -> Optional[dict[str, Any]]:
    if not run.georeferenced:
        return None
    report = _read_json(catalog.product_path(site, run, REGISTRATION)) or {}
    return metric_block(site.origin, earth_model=run.earth_model, source="catalog",
                        frame_error_m=frame_error(report, run.earth_model),
                        site=site.key, epoch=run.epoch_id, run=run.id)


# Scene discovery reruns every few seconds (render service, /scenes/refresh);
# each skipped or replaced run is reported once per process.
_reported: set[str] = set()


def _report_once(msg: str) -> None:
    if msg not in _reported:
        _reported.add(msg)
        log.warning(msg)


def _missing_products(catalog: Catalog, site: Site, run: Run) -> list[Path]:
    """The run's registered splat/cloud files that are no longer on disk."""
    paths = (catalog.product_path(site, run, kind) for kind in (SPLAT, CLOUD))
    return [p for p in paths if p is not None and not p.is_file()]


def scene_run(catalog: Catalog, site: Site) -> Optional[Run]:
    """The run a Site's scene is served from: its latest successful run whose
    registered splat/cloud files all still exist.

    A run whose folder was deleted would otherwise resolve to missing files and
    shadow a legacy `data/<ds>_*.pt` scene of the same name; such runs are
    skipped (with a warning) in favour of an older complete run, or no scene.
    """
    runs = catalog.finished_runs(site, with_products=(SPLAT, CLOUD))
    for i, run in enumerate(runs):
        missing = _missing_products(catalog, site, run)
        if not missing:
            return run
        older = (f"беру старіший run {runs[i + 1].id}" if i + 1 < len(runs)
                 else "сцену каталогу пропущено")
        _report_once(f"   ⚠️  [Scenes] {site.key}: run {run.id} — немає файлів "
                     f"{', '.join(str(p) for p in missing)}; {older}")
    return None


class SceneLookupError(LookupError):
    """A scene pinned to one run that cannot be served (unknown run, no splat, files gone)."""


def run_scene_source(catalog: Catalog, site: Site, run: Run) -> SceneSource:
    """The scene of one particular run, for a caller that must not mix runs.

    `scene_run` serves a Site's newest complete run, which is right for a viewer but wrong
    for anything drawn in another run's frame: a facade of an older run rendered from the
    newest splat is a picture of a different reconstruction (ACCESS_POINTS §5.9). Raises
    `SceneLookupError` instead of falling back to some other run.
    """
    if run.status != RunStatus.ok:
        raise SceneLookupError(f"run {run.id} of {site.key} is {run.status.value}, not ok")
    splat = catalog.product_path(site, run, SPLAT)
    if splat is None:
        raise SceneLookupError(f"run {run.id} of {site.key} has no trained scene (no splat product)")
    if not splat.is_file():
        raise SceneLookupError(f"run {run.id} of {site.key}: splat file {splat} is missing")
    cloud = catalog.product_path(site, run, CLOUD)
    return SceneSource(name=site.key, cloud=str(cloud or ""), splat=str(splat),
                       metric=run_metric(catalog, site, run), kind="catalog", site=site.key,
                       run=run.id)


def find_run(catalog: Catalog, site: Site, run_id: str) -> Run:
    """A run of the Site by id, in any epoch. Raises `SceneLookupError` if absent or ambiguous."""
    runs = [r for r in catalog.list_runs(site) if r.id == run_id]
    if not runs:
        raise SceneLookupError(f"site {site.key} has no run '{run_id}'")
    if len(runs) > 1:
        raise SceneLookupError(f"run id '{run_id}' exists in several epochs of {site.key} "
                               f"({', '.join(r.epoch_id for r in runs)})")
    return runs[0]


def pinned_scene(projects_root: Optional[Path], site_key: str, run_id: str) -> SceneSource:
    """`run_scene_source` by name, opening the catalog for the call (the render service's path)."""
    if projects_root is None or not (projects_root / CATALOG_FILE).is_file():
        raise SceneLookupError(f"no project catalog: cannot pin '{site_key}' to run {run_id}")
    if site_key.count("/") != 1:
        raise SceneLookupError(f"a scene pinned to a run must be named <project>/<site>, "
                               f"not '{site_key}'")
    project_id, site_id = site_key.split("/")
    catalog = Catalog.open(projects_root)
    try:
        try:
            site = catalog.get_site(project_id, site_id)
        except LookupError as e:
            raise SceneLookupError(str(e)) from None
        return run_scene_source(catalog, site, find_run(catalog, site, run_id))
    finally:
        catalog.close()


def catalog_scenes(catalog: Catalog) -> dict[str, SceneSource]:
    out: dict[str, SceneSource] = {}
    sites = catalog.list_sites()
    bare_counts: dict[str, int] = {}
    for s in sites:
        bare_counts[s.id] = bare_counts.get(s.id, 0) + 1
    for site in sites:
        run = scene_run(catalog, site)
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
        splat = data_dir / f"{ds}_splat.pt"
        metric = None
        geo = _read_json(data_dir / ds / "sparse_known" / "geo_registration.json")
        if geo and is_georeferenced(cfg):
            o = geo.get("enu_origin") or {}
            origin = GeoPoint(lat=o["latitude"], lon=o["longitude"], h=o["altitude"])
            label = str(geo.get("earth_model", ""))
            extra = {"origin_source": geo.get("origin_source", "frames_mean (legacy)")}
            if label == SPHERE_ENU_AFFINE:
                # Only the georef stage's outputs are in WGS84 ENU; without them the
                # scene is in sphere ENU and gets no metric block.
                cloud_w = cloud.with_name(f"{cloud.stem}_wgs84.pt")
                if cloud_w.is_file():
                    cloud, splat = cloud_w, splat.with_name(f"{splat.stem}_wgs84.pt")
                    metric = metric_block(origin, earth_model=SPHERE_ENU_AFFINE, source="legacy",
                                          frame_error_m=frame_error(geo, SPHERE_ENU_AFFINE),
                                          **extra)
            else:
                metric = metric_block(origin, earth_model=(WGS84 if label.startswith("WGS84")
                                                           else "sphere-legacy"),
                                      source="legacy", **extra)
        out[ds] = SceneSource(name=ds, cloud=str(cloud), splat=str(splat),
                              metric=metric, kind="legacy")
    return out


def discover_scenes(projects_root: Optional[Path], data_dir: Optional[Path]
                    ) -> dict[str, SceneSource]:
    """name -> scene. Catalog sites win over legacy datasets of the same name,
    unless none of the site's runs still has its splat/cloud files (scene_run).

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
