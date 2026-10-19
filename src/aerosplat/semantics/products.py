"""Finding and reading what the `semantics` stage left behind, for the API and the editor.

The stage writes into a run directory, which is derived data and may be deleted and
recomputed (ADR-0006). So nothing holds on to a path: the semantics of a Site means
"whatever its newest successful run produced", resolved freshly on every request.

Older runs finished before `semantics` was a registered product, so the lookup falls
back to looking for the file on disk — a dataset that has already been through the
stage stays usable without recomputing it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from aerosplat.semantics.models import COVERAGE_NPZ, SEMANTICS_JSON, Facade, SemanticsResult
from aerosplat.store.catalog import Catalog
from aerosplat.store.models import Run, RunStatus, Site
from aerosplat.store.scenes import SEMANTICS

SEMANTICS_DIR = "semantics"         # the directory inside a run, when the product is not registered


class SemanticsMissing(LookupError):
    """No run of this Site has been through the `semantics` stage yet."""


@dataclass(frozen=True)
class SemanticsProduct:
    """One run's semantics output: where it is, and what it holds."""

    run: Run
    dir: Path
    result: SemanticsResult

    @property
    def coverage_npz(self) -> Path:
        return self.dir / COVERAGE_NPZ

    def facade(self, facade_id: str) -> Facade:
        return self.result.facade(facade_id)

    def coverage_grid(self, facade_id: str) -> Optional[NDArray[np.int32]]:
        """Per-cell view count of one facade, shaped (nv, nu) with v growing up.

        This is the map that says where the orthophoto is evidence and where the splat
        is inventing texture (§5.3), so the editor must show it. None when the run had
        no cameras to build it from.
        """
        if not self.coverage_npz.is_file():
            return None
        with np.load(self.coverage_npz) as z:
            if facade_id not in z.files:
                return None
            return z[facade_id].astype(np.int32)


def _semantics_dir(catalog: Catalog, site: Site, run: Run) -> Optional[Path]:
    path = catalog.product_path(site, run, SEMANTICS)
    if path is None:
        path = catalog.run_dirs(site, run.epoch_id, run.id).root / SEMANTICS_DIR
    return path if (path / SEMANTICS_JSON).is_file() else None


def semantics_runs(catalog: Catalog, site: Site) -> list[Run]:
    """Successful runs whose semantics output is on disk, newest first (for a run picker)."""
    runs = [r for r in catalog.list_runs(site) if r.status == RunStatus.ok
            and _semantics_dir(catalog, site, r) is not None]
    runs.sort(key=lambda r: (r.finished_at or r.created_at, r.id), reverse=True)
    return runs


def find_semantics(catalog: Catalog, site: Site, *, run_id: Optional[str] = None,
                   epoch_id: Optional[str] = None) -> Optional[SemanticsProduct]:
    """The newest successful run with semantics, or a named one. None when there is none."""
    runs = [r for r in catalog.list_runs(site, epoch_id) if r.status == RunStatus.ok]
    if run_id is not None:
        runs = [r for r in runs if r.id == run_id]
    runs.sort(key=lambda r: (r.finished_at or r.created_at, r.id), reverse=True)
    for run in runs:
        folder = _semantics_dir(catalog, site, run)
        if folder is None:
            continue
        try:
            result = SemanticsResult(
                **json.loads((folder / SEMANTICS_JSON).read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
        return SemanticsProduct(run=run, dir=folder, result=result)
    return None


def require_semantics(catalog: Catalog, site: Site, **kw) -> SemanticsProduct:
    found = find_semantics(catalog, site, **kw)
    if found is None:
        raise SemanticsMissing(
            f"site {site.key} has no semantics yet — run the pipeline through the "
            f"`semantics` stage (it needs a dense cloud)")
    return found
