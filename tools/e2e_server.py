"""Throwaway AeroSplat server for the Playwright e2e smoke (web/app/e2e, `npm run e2e`).

    python tools/e2e_server.py --port 8765
    python tools/e2e_server.py --port 8766 --require-auth --token-file %TEMP%/e2e-token

Seeds a fresh projects root in the system temp directory (never the repository's
`projects/`) with synthetic fixtures, then runs `aerosplat up` on it with a CPU worker
and no GPU worker / render service:

* `demo/pier`   — a run with semantics: one building, two facades (Facade editor);
* `demo/street` — the synthetic window mission of tests/sim_scenarios.py with one
                  simulated flight (the flight page).

The render service URL points at a closed port, so nothing here can reach a render
service the owner may be running on :8001. `--require-auth` turns loopback trust off and
creates a token (written to `--token-file`) for the token-entry test.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))        # mission_scenes / sim_scenarios fixtures


def seed_facades(root: Path) -> None:
    import numpy as np

    from aerosplat.geo.crs import GeoPoint
    from aerosplat.semantics.models import (COVERAGE_NPZ, SEMANTICS_JSON, Building, Coverage, Facade,
                                            SemanticsResult, SurfaceKind, TerrainModel, azimuth_of,
                                            u_axis_for)
    from aerosplat.store import Catalog, Site
    from aerosplat.store.models import RunStatus

    def facade(fid, normal, origin, u_min=-10.0, u_max=10.0, height=12.0, coverage=None):
        n = np.asarray(normal, float)
        ux, uy = u_axis_for((float(n[0]), float(n[1])))
        return Facade(id=fid, building_id=fid.split("/")[0], kind=SurfaceKind.wall, origin=origin,
                      normal=(float(n[0]), float(n[1]), 0.0), u_axis=(ux, uy, 0.0),
                      azimuth_deg=azimuth_of((float(n[0]), float(n[1]))), u_min=u_min,
                      u_max=u_max, v_ground=origin[2], v_max=origin[2] + height,
                      plane_rms_m=0.09, n_points=4200, coverage=coverage)

    cat = Catalog.open(root)
    cat.ensure_project("demo", "E2E demo")
    s = cat.create_site(Site(id="pier", project_id="demo", name="Pier (facades)",
                             origin=GeoPoint(lat=40.8942, lon=-74.2173, h=233.0)))
    cat.ensure_epoch(s)
    cat.create_run(s, "baseline", "r1", capture_ids=[], config={}, earth_model="wgs84",
                   georeferenced=True)
    dirs = cat.run_dirs(s, "baseline", "r1")
    sem = dirs.root / "semantics"
    sem.mkdir(parents=True)
    cov = Coverage(cell_m=0.5, cells=960, visible=900, covered=700, mean_views=8.0,
                   max_gsd_cm=5.0, best_gsd_cm=2.1, median_angle_deg=28.0)
    north = facade("B01/N1", (0.0, 1.0), (0.0, 0.0, 0.0), coverage=cov)
    east = facade("B01/E1", (1.0, 0.0), (10.0, 0.0, 0.0), u_min=-6.0, u_max=6.0, coverage=cov)
    building = Building(id="B01", footprint=[(-10, -12), (10, -12), (10, 0), (-10, 0)],
                        area_m2=240.0, ground_z=0.0, roof_z=12.0, height_m=12.0,
                        facade_ids=["B01/E1", "B01/N1"])
    result = SemanticsResult(
        terrain=TerrainModel(res_m=1.0, x0=-40.0, y0=-40.0, rows=80, cols=80, n_points=900_000,
                             filled_cells=5400),
        buildings=[building], facades=[east, north], warnings=[])
    (sem / SEMANTICS_JSON).write_text(result.model_dump_json(), encoding="utf-8")
    np.savez_compressed(sem / COVERAGE_NPZ, **{"B01/N1": np.full((24, 40), 7, np.int32)})
    # A placeholder "trained scene": enables the editor (orthophotos then fail with 503 — no
    # render service here — which the page shows; drawing needs a real scene).
    (dirs.gs).mkdir(parents=True)
    (dirs.gs / "splat_wgs84.pt").write_bytes(b"e2e placeholder, not a torch file")
    cat.finish_run(s, "baseline", "r1", status=RunStatus.ok,
                   products={"semantics": sem, "splat": dirs.gs / "splat_wgs84.pt"})
    cat.close()


def seed_mission(root: Path, work: Path) -> None:
    import mission_scenes as ms
    import sim_scenarios as ss

    from aerosplat.store import Catalog, Site

    cat = Catalog.open(root)
    cat.ensure_project("demo", "E2E demo")
    site = cat.create_site(Site(id="street", project_id="demo", name="Street (mission)",
                                origin=ms.ORIGIN))
    cat.ensure_epoch(site)
    missions = cat.site_dirs(site).missions
    cat.close()
    plan = ss.window_plan(work / "plan", ms.street_grid())
    folder = ss.export(plan, missions / plan.id)
    rec = ss.fly(plan)
    out = folder / "sim" / "street-window-delivered"
    out.mkdir(parents=True)
    rec.meta = {"scenario": "street-window-delivered", "speedup": 3.0,
                "origin": ms.ORIGIN.model_dump(), "home_enu": list(plan.home.enu),
                "mission": plan.id}
    rec.save(out / "trajectory.json")
    delivered = {"outcome": "complete", "reason": "delivered"}
    report = {"schema": 1,
              "scenario": ss.scenario("street-window-delivered", delivered).model_dump(mode="json"),
              "seed": 0, "passed": True, "summary": "delivered: all invariants hold",
              "outcome": "complete", "reason": None, "detail": "",
              "expected": {"outcome": "complete", "reason": "delivered",
                           "invariants": {"clearance": "pass"}},
              "verdicts": [{"invariant": "clearance", "status": "pass", "detail": "ok",
                            "expected": "pass", "metrics": {}}],
              "timing": {"mission_sim_s": float(rec.t[-1] - rec.t[0])},
              "firmware": {"tag": "Copter-4.7.2"}, "warnings": [], "faults": [], "events": [],
              "samples": len(rec)}
    (out / "report.json").write_text(json.dumps(report), encoding="utf-8")


def write_samples(folder: Path, n: int = 3) -> None:
    """Small JPEGs the browser test uploads as a capture (<tmp>/aerosplat-e2e/<port>/samples)."""
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (64, 48), (40 + 60 * i, 120, 200 - 50 * i)).save(folder / f"e2e_{i}.jpg", "JPEG")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--require-auth", action="store_true")
    ap.add_argument("--token-file", type=Path, default=None)
    args = ap.parse_args()
    if args.port in (8000, 8001):
        raise SystemExit("ports 8000/8001 belong to the owner's servers; pick another")

    # A fixed, recognisable directory per port: wiped here, because a hard-killed run
    # (Playwright stops the server with taskkill /F) never gets to clean up after itself.
    base = Path(tempfile.gettempdir()) / "aerosplat-e2e" / str(args.port)
    shutil.rmtree(base, ignore_errors=True)
    root, work, data = base / "projects", base / "work", base / "data"
    for d in (root, work, data):
        d.mkdir(parents=True)

    seed_facades(root)
    seed_mission(root, work)
    write_samples(base / "samples")
    if args.token_file is not None:
        from aerosplat.auth.tokens import TokenStore

        _, secret = TokenStore.for_root(root).create("e2e")
        args.token_file.parent.mkdir(parents=True, exist_ok=True)
        args.token_file.write_text(secret, encoding="utf-8")

    os.environ["AEROSPLAT_DATA"] = str(data)                     # no legacy scenes from data/
    os.environ["AEROSPLAT_RENDER_URL"] = "http://127.0.0.1:9"    # never the owner's :8001
    os.environ.pop("AEROSPLAT_DATASET", None)
    print(f"e2e projects root: {root}", flush=True)
    from aerosplat.server import up

    argv = ["--port", str(args.port), "--projects-root", str(root), "--no-gpu-worker", "--no-render"]
    if args.require_auth:
        argv.append("--require-auth")
    return up.main(argv)


if __name__ == "__main__":
    sys.exit(main())
