"""Finding and reading what the `occupancy` stage left behind, for planning and simulation.

Same rules as `aerosplat.semantics.products`: a run directory is derived data, so nothing
holds on to a path — the occupancy of a Site means "whatever its newest successful run
produced", resolved freshly. A run that predates the registered product is still found by
looking for the file on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from aerosplat.store.catalog import Catalog
from aerosplat.store.models import Run, RunStatus, Site
from aerosplat.store.scenes import OCCUPANCY
from aerosplat.surface.models import FORMAT_VERSION, OCCUPANCY_JSON, OccupancyReport
from aerosplat.surface.occupancy import OccupancyGrid, read_occupancy

OCCUPANCY_DIR = "occupancy"         # the directory inside a run, when the product is not registered


class OccupancyMissing(LookupError):
    """No run of this Site has been through the `occupancy` stage yet."""


@dataclass(frozen=True)
class OccupancyProduct:
    """One run's occupancy output. The arrays are large, so they load only on request."""

    run: Run
    dir: Path
    report: OccupancyReport

    def load_grid(self) -> OccupancyGrid:
        grid, _ = read_occupancy(self.dir)
        return grid


def _occupancy_dir(catalog: Catalog, site: Site, run: Run) -> Optional[Path]:
    path = catalog.product_path(site, run, OCCUPANCY)
    if path is None:
        path = catalog.run_dirs(site, run.epoch_id, run.id).root / OCCUPANCY_DIR
    return path if (path / OCCUPANCY_JSON).is_file() else None


def find_occupancy(catalog: Catalog, site: Site, *, run_id: Optional[str] = None,
                   epoch_id: Optional[str] = None) -> Optional[OccupancyProduct]:
    """The newest successful run with occupancy in the current format, or a named one."""
    runs = [r for r in catalog.list_runs(site, epoch_id) if r.status == RunStatus.ok]
    if run_id is not None:
        runs = [r for r in runs if r.id == run_id]
    runs.sort(key=lambda r: (r.finished_at or r.created_at, r.id), reverse=True)
    for run in runs:
        folder = _occupancy_dir(catalog, site, run)
        if folder is None:
            continue
        try:
            report = OccupancyReport(
                **json.loads((folder / OCCUPANCY_JSON).read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
        if report.format != FORMAT_VERSION:
            continue
        return OccupancyProduct(run=run, dir=folder, report=report)
    return None


def require_occupancy(catalog: Catalog, site: Site, **kw) -> OccupancyProduct:
    found = find_occupancy(catalog, site, **kw)
    if found is None:
        raise OccupancyMissing(
            f"site {site.key} has no occupancy yet — run the pipeline through the "
            f"`occupancy` stage (it needs `semantics`)")
    return found
