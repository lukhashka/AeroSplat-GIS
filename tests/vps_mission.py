"""The SIM-3 mission on dataset_1910 (ADR-0015 §5 stage 2): the window B08/SW3/07/04 planned for
the GNSS-free test bed (x500-pi5-imx290), its corridor atlas and localizability from the site
run's splat, exported to a temporary folder. Read-only on the project: the target and access
point are copied out of `authored/` first; nothing under projects/ is written.

The take-off spot (-5.2, -37.8) is the takeoff search's nearest spot for this window (mission
msn_01M5A4JN41G1P7KTXMMEQHETTW, run 20261019-101955); the plan is checked again here against the
occupancy of run 20261019-105253, whose splat is the world the camera sees.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

from aerosplat.store.paths import default_projects_root

# AEROSPLAT_PROJECTS points a checkout without the data (a worktree) at the projects folder.
SITE_DIR = default_projects_root() / "default" / "sites" / "dataset_1910"
RUN_ID = "20261019-105253"
RUN_DIR = SITE_DIR / "epochs" / "baseline" / "runs" / RUN_ID
TARGET_ID = "tgt_01M59XNQ141FPZ1940F7ECB8MY"
HOME_ENU = (-5.2, -37.8, -166.7)
VEHICLE = "x500-pi5-imx290"


def data_missing() -> Optional[str]:
    """Why the dataset_1910 mission cannot be built here (None: it can)."""
    need = [SITE_DIR / "authored" / "targets.json", SITE_DIR / "authored" / "access_points.json",
            RUN_DIR / "occupancy" / "occupancy.npz", RUN_DIR / "gs" / "splat_wgs84.pt",
            RUN_DIR / "reports" / "registration.json", RUN_DIR / "semantics" / "terrain.npz"]
    missing = [str(p) for p in need if not p.is_file()]
    return f"dataset_1910 run {RUN_ID} not on this machine: {missing[0]}" if missing else None


def gpu_missing() -> Optional[str]:
    try:
        import torch

        from aerosplat.gs.gsplat_backend import require_gsplat
        if not torch.cuda.is_available():
            return "no CUDA device"
        require_gsplat()
    except Exception as e:          # noqa: BLE001 - any reason means: no GPU renderer here
        return f"gsplat + CUDA unavailable ({e}); run under tools\\gsplat_env.bat"
    return None


def plan(tmp: Path) -> Any:
    """The GNSS-free plan (VPS declared at the profile's rate, pose only; optical flow)."""
    from aerosplat.authored.store import AuthoredStore
    from aerosplat.mission.models import ExtNavSpec, NavProfile, OccupancyRef, PlanningProfile
    from aerosplat.mission.planner import plan_mission
    from aerosplat.mission.targets import TargetStore
    from aerosplat.sim.runner import load_vehicle
    from aerosplat.surface.occupancy import read_occupancy
    from aerosplat.vps.twin import SiteRun

    authored = tmp / "authored"
    authored.mkdir(parents=True, exist_ok=True)
    for name in ("targets.json", "access_points.json"):
        shutil.copy(SITE_DIR / "authored" / name, authored / name)
    store = TargetStore(AuthoredStore(authored, "default/dataset_1910"))
    target = store.get(TARGET_ID)
    ap = store.authored.get(target.ap_id)
    grid, report = read_occupancy(RUN_DIR / "occupancy")
    ref = OccupancyRef(run_id=RUN_ID, epoch_id="baseline", format=report.format,
                       res_m=report.grid.res_m, unknown_mode=report.params.get("unknown"),
                       carved=report.stats.carving is not None)
    veh = load_vehicle(VEHICLE)
    mount = veh.airframe.vps_mount()
    assert veh.frame is not None and mount is not None and mount.vps is not None
    ext = ExtNavSpec(rate_hz=mount.vps.rate_hz, latency_ms=100.0, camera=mount.camera, velocity=False)
    prof = PlanningProfile(r_drone_m=veh.frame.r_drone_m,
                           nav=NavProfile(gnss="none", extnav=ext, flow=True))
    run = SiteRun.open(RUN_DIR)
    return plan_mission(target=target, ap=ap, grid=grid, occupancy=ref, origin=run.origin,
                        by="op:sim-3-test", profile=prof, home_enu=HOME_ENU), grid, run


def package(tmp: Path, world: Any, log=print) -> tuple[Path, Any, Any, Any]:
    """Plan, atlas + localizability (`world`: the run's EngineRenderer), export. Returns
    (package folder, plan, localizability, occupancy grid)."""
    from aerosplat.mission.export import render
    from aerosplat.sim.camera import load_camera
    from aerosplat.sim.runner import load_vehicle
    from aerosplat.vps.features import make_extractor
    from aerosplat.vps.route import CameraModel, prepare

    p, grid, run = plan(tmp)
    mount = load_vehicle(VEHICLE).airframe.vps_mount()
    assert mount is not None and mount.vps is not None
    cam = CameraModel.of(load_camera(mount.camera), mount, tuple(mount.vps.resolution))
    folder = tmp / "missions" / p.id
    folder.mkdir(parents=True)
    p, _, rep = prepare(p, world, cam, make_extractor(mount.vps.extractor), folder, run_id=RUN_ID,
                        meta={"site": "dataset_1910", "splat": run.splat.name,
                              "splat_sha256": run.splat_sha256()}, log=log)
    (folder / "mission.json").write_text(p.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
    if not p.blocking:
        for fmt in ("waypoints", "plan", "parm"):
            name, _, text = render(p, run.origin, fmt, accept_risks=True)
            (folder / name).write_text(text, encoding="utf-8")
    log(json.dumps({k: v for k, v in rep.summary().items() if k not in ("segments",)}, default=float))
    return folder, p, rep, grid
