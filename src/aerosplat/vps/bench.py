"""VPS feasibility bench (ADR-0015 layer A, measured on the twin).

    # ground station (GPU): build atlases, render and export the query set, localize, report
    tools\\gsplat_env.bat venv\\Scripts\\python -m aerosplat.vps.bench run --run-dir <site run> --out <dir>

    # companion (CPU only: numpy + OpenCV): localizer-only, from a copied <dir>
    python3 -m aerosplat.vps.bench localize --data <dir> --threads 4,1

`run` renders keyframes of the twin around an area (default: the site's tallest building, the
takeoff region of dataset_1910) with the VPS camera of a vehicle profile (x500-pi5-imx290: IMX290
4 mm, mount pitch), builds one atlas per resolution and extractor, renders query frames from
random poses (heights 20-60 m above the DTM, any heading, ±5° attitude) and exports
`<dir>/<W>x<H>/{atlas_<extractor>.npz, queries/*.jpg, queries.json}`. Then it runs exactly what
`localize` runs.

Query groups: `clean`, `mild`, `strong` (appearance perturbations of the same frames,
aerosplat.vps.perturb), `noprior` (clean, retrieval by global descriptor), `off_atlas` (frames
from outside the atlas with a prior that claims to be inside: every accepted fix is false).
The prior is the truth plus noise (σ 3 m horizontal, 1 m vertical, 5° heading).

**What this measures and what it does not.** Queries and atlas come from the same splat: this is
the method's ceiling (geometry, matching, refusal logic, timing), not the domain gap. The
perturbations are photometric and codec noise only. Real IMX290 frames vs. an Earth Studio twin
(season, shadows, cars, construction, lens distortion, rolling shutter, motion blur) are not
represented; only flights over the golden site (L4/L5) answer that. Timings here are of this
PC; `--threads 1` approximates a single slow core, not a Raspberry Pi 5.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from aerosplat.vps import perturb
from aerosplat.vps.atlas import Atlas
from aerosplat.vps.features import require_cv2
from aerosplat.vps.geometry import angles, percentile, pose_error, rotation
from aerosplat.vps.localize import Localizer, PosePrior

FALSE_FIX_M = 2.0
LEVELS = ("clean", "mild", "strong")
JPEG_QUALITY = 97              # stored "clean" frames (visually lossless; stated in the report)
DEFAULT_VEHICLE = "x500-pi5-imx290"
PRIOR_SIGMA = {"pos_m": 3.0, "z_m": 1.0, "yaw_deg": 5.0}


def log(msg: str) -> None:
    print(msg, flush=True)


# ── ground station: build and export ────────────────────────────────────
def _parse_res(s: str) -> list[tuple[int, int]]:
    out = []
    for part in s.split(","):
        w, h = part.lower().split("x")
        out.append((int(w), int(h)))
    return out


def _query_views(run: Any, rng: np.random.Generator, centre: tuple[float, float], half_m: float,
                 n: int, pitch: float, alt: tuple[float, float]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    tries = 0
    while len(out) < n:
        tries += 1
        if tries > 100 * n:
            raise RuntimeError("cannot place query poses clear of the buildings")
        e = centre[0] + rng.uniform(-half_m, half_m)
        nn = centre[1] + rng.uniform(-half_m, half_m)
        u = run.ground_z(e, nn) + rng.uniform(*alt)
        if not run.clear(e, nn, u):
            continue
        out.append({"C": [e, nn, u], "yaw_deg": float(rng.uniform(0, 360)),
                    "pitch_deg": float(pitch + rng.uniform(-5, 5)),
                    "roll_deg": float(rng.uniform(-5, 5))})
    return out


def _prior_for(view: dict[str, Any], rng: np.random.Generator, centre: Optional[tuple[float, float]] = None,
               half_m: float = 0.0) -> dict[str, Any]:
    if centre is None:
        C = np.asarray(view["C"], float) + np.array([rng.normal(0, PRIOR_SIGMA["pos_m"]),
                                                     rng.normal(0, PRIOR_SIGMA["pos_m"]),
                                                     rng.normal(0, PRIOR_SIGMA["z_m"])])
    else:   # a prior that claims to be inside the atlas while the camera is elsewhere
        C = np.array([centre[0] + rng.uniform(-half_m, half_m),
                      centre[1] + rng.uniform(-half_m, half_m), view["C"][2]])
    return {"C": C.tolist(), "yaw_deg": float(view["yaw_deg"] + rng.normal(0, PRIOR_SIGMA["yaw_deg"])),
            "pitch_deg": view["pitch_deg"], "roll_deg": view["roll_deg"],
            "sigma_pos_m": PRIOR_SIGMA["pos_m"], "sigma_yaw_deg": PRIOR_SIGMA["yaw_deg"]}


def cmd_run(a: argparse.Namespace) -> int:
    cv2 = require_cv2()
    from aerosplat.sim.camera import load_camera
    from aerosplat.sim.runner import load_vehicle
    from aerosplat.vps.build import DepthFilter, View, build_atlas, grid_views
    from aerosplat.vps.features import make_extractor
    from aerosplat.vps.twin import EngineRenderer, SiteRun, software_versions

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run = SiteRun.open(Path(a.run_dir))
    veh = load_vehicle(a.vehicle)
    mount = veh.airframe.vps_mount()
    if mount is None:
        raise SystemExit(f"vehicle {a.vehicle} has no VPS camera")
    cam = load_camera(mount.camera)
    pitch = float(mount.pitch_deg)
    if a.center:
        cxy = [float(v) for v in a.center.split(",")]
        centre = (cxy[0], cxy[1])
    else:
        centre = run.tallest_centre() or (0.0, 0.0)
    rng = np.random.default_rng(a.seed)
    log(f"VPS bench: site {run.site_id} run {run.run_id}, camera {cam.name} pitch {pitch:g}°, "
        f"area {2 * a.half:g} m square around ({centre[0]:.1f}, {centre[1]:.1f})")
    t0 = time.perf_counter()
    splat_hash = run.splat_sha256()
    log(f"   splat {run.splat.name} sha256 {splat_hash[:16]} ({time.perf_counter() - t0:.1f} s)")

    views = grid_views(centre, a.half, a.spacing, a.kf_alts, [i * 360.0 / a.kf_yaws for i in range(a.kf_yaws)],
                       pitch, run.ground_z, run.clear)
    qviews = _query_views(run, rng, centre, a.half - 10.0, a.queries, pitch, (20.0, 60.0))
    priors = [_prior_for(v, rng) for v in qviews]
    levels = {lvl: [perturb.draw(lvl, rng) for _ in qviews] for lvl in LEVELS}
    renderer = EngineRenderer(run)
    # off-atlas frames: 150-300 m from the centre, where the twin still has content
    off: list[dict[str, Any]] = []
    tries = 0
    while len(off) < a.off_atlas and tries < 50 * max(a.off_atlas, 1):
        tries += 1
        ang, dist = rng.uniform(0, 2 * math.pi), rng.uniform(150, 300)
        e, nn = centre[0] + dist * math.sin(ang), centre[1] + dist * math.cos(ang)
        u = run.ground_z(e, nn) + rng.uniform(20, 60)
        if not run.clear(e, nn, u):
            continue
        v: dict[str, Any] = {"C": [e, nn, u], "yaw_deg": float(rng.uniform(0, 360)),
                             "pitch_deg": float(pitch + rng.uniform(-5, 5)),
                             "roll_deg": float(rng.uniform(-5, 5))}
        _, depth = renderer.render(View(tuple(v["C"]), v["yaw_deg"], v["pitch_deg"], v["roll_deg"]),
                                   320, 180, (cam.fx / 6, cam.fy / 6, 159.5, 89.5))
        if float((depth > 0).mean()) < 0.9:
            continue
        off.append(v)
    off_priors = [_prior_for(v, rng, centre, a.half - 10.0) for v in off]

    summary: dict[str, Any] = {"site": run.site_id, "run": run.run_id, "splat": run.splat.name,
                               "splat_sha256": splat_hash, "camera": cam.name, "mount_pitch_deg": pitch,
                               "vehicle": a.vehicle, "centre_en": list(centre), "half_m": a.half,
                               "keyframe_views": len(views), "queries": len(qviews),
                               "off_atlas": len(off), "seed": a.seed, "builds": {},
                               "software": {**software_versions(), "opencv": cv2.__version__},
                               "jpeg_quality": JPEG_QUALITY, "prior_sigma": PRIOR_SIGMA}
    for (W, H) in _parse_res(a.resolutions):
        s = W / cam.width
        K = (cam.fx * s, cam.fy * s, W / 2 - 0.5, H / 2 - 0.5)
        rdir = out / f"{W}x{H}"
        (rdir / "queries").mkdir(parents=True, exist_ok=True)
        log(f"== {W}x{H}: fx {K[0]:.1f} px; rendering {len(qviews) + len(off)} query frames")
        records = []
        for i, (v, p) in enumerate(zip(qviews + off, priors + off_priors)):
            rgb, _ = renderer.render(View(tuple(v["C"]), v["yaw_deg"], v["pitch_deg"], v["roll_deg"]),
                                     W, H, K)
            name = f"q{i:04d}.jpg"
            cv2.imwrite(str(rdir / "queries" / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            in_atlas = i < len(qviews)
            records.append({"id": i, "file": f"queries/{name}", "truth": v, "prior": p,
                            "in_atlas": in_atlas,
                            "perturb": {lvl: levels[lvl][i] for lvl in LEVELS} if in_atlas else {}})
        (rdir / "queries.json").write_text(json.dumps(
            {"intrinsics": list(K), "width": W, "height": H, "camera": cam.name,
             "false_fix_m": FALSE_FIX_M, "noprior_count": a.noprior, "queries": records},
            indent=1), encoding="utf-8")
        for ex_name in a.extractors.split(","):
            n_max = int(a.features * (W / 960.0) ** 0.5)
            ex = make_extractor(ex_name, n_max=n_max)
            log(f"   atlas {ex_name} (n_max {n_max}): {len(views)} views")
            t0 = time.perf_counter()
            atlas, st = build_atlas(renderer, views, W, H, K, ex,
                                    {"site": run.site_id, "run": run.run_id, "splat": run.splat.name,
                                     "splat_sha256": splat_hash, "frame": "WGS84 Site ENU",
                                     "origin": {"lat": run.origin.lat, "lon": run.origin.lon,
                                                "h": run.origin.h},
                                     "camera": cam.name, "mount_pitch_deg": pitch,
                                     "area": {"centre_en": list(centre), "half_m": a.half,
                                              "spacing_m": a.spacing, "alts_agl_m": list(a.kf_alts),
                                              "yaws": a.kf_yaws},
                                     "software": summary["software"],
                                     "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                                    DepthFilter(), log=log)
            path = rdir / f"atlas_{ex_name}.npz"
            digest = atlas.save(path)
            st.update(build_s=round(time.perf_counter() - t0, 1), file_mb=round(path.stat().st_size / 2 ** 20, 2),
                      content_hash=digest)
            summary["builds"][f"{W}x{H}/{ex_name}"] = st
            log(f"   -> {atlas.n_keyframes} keyframes, {atlas.n_features} points, "
                f"{st['file_mb']} MB on disk, {st['build_s']} s")
    renderer.close()
    (out / "build.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    a.data = str(out)
    return cmd_localize(a)


# ── companion: localize the exported query set ──────────────────────────
def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    ok = [r for r in rows if r["ok"]]
    false = [r for r in ok if r["err"]["pos_m"] > FALSE_FIX_M]
    good = [r for r in ok if r["err"]["pos_m"] <= FALSE_FIX_M]
    reasons: dict[str, int] = {}
    for r in rows:
        if not r["ok"]:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1

    def _round(v: Optional[float], digits: int) -> Optional[float]:
        return None if v is None else round(v, digits)

    def pct(vals: list[float]) -> dict[str, Optional[float]]:
        return {f"p{q}": _round(percentile(vals, q), 3) for q in (50, 90, 95)} | \
               {"max": round(max(vals), 3) if vals else None}

    def t(stage: str) -> dict[str, Optional[float]]:
        vals = [r["t"][stage] for r in rows if r["t"].get(stage) is not None]
        return {"p50": _round(percentile(vals, 50), 1),
                "p95": _round(percentile(vals, 95), 1)}

    ratio = [r["err"]["horiz_m"] / r["sigma_h_m"] for r in ok if r["sigma_h_m"]]
    return {
        "n": n, "fix": len(ok), "fix_rate": round(len(ok) / n, 3) if n else None,
        "good_fix_rate": round(len(good) / n, 3) if n else None,
        "refused": n - len(ok), "refusal_rate": round((n - len(ok)) / n, 3) if n else None,
        "reasons": dict(sorted(reasons.items())),
        "false_fix": len(false), "false_fix_rate": round(len(false) / n, 4) if n else None,
        "pos_err_m": pct([r["err"]["pos_m"] for r in ok]),
        "horiz_err_m": pct([r["err"]["horiz_m"] for r in ok]),
        "yaw_err_deg": pct([r["err"]["yaw_deg"] for r in ok]),
        "sigma_h_m": pct([r["sigma_h_m"] for r in ok]),
        "within_2sigma_h": round(float(np.mean(np.array(ratio) <= 2.0)), 3) if ratio else None,
        "err_over_sigma_p95": _round(percentile(ratio, 95), 2) if ratio else None,
        "timing_ms": {s: t(s) for s in ("extract", "retrieve", "match", "pnp", "verify", "total")},
    }


def _localize_set(rdir: Path, ex_name: str, threads: int, accuracy: bool) -> dict[str, Any]:
    cv2 = require_cv2()
    q = json.loads((rdir / "queries.json").read_text(encoding="utf-8"))
    K = tuple(q["intrinsics"])
    atlas = Atlas.load(rdir / f"atlas_{ex_name}.npz")
    loc = Localizer(atlas)
    cv2.setNumThreads(threads)
    groups: dict[str, list[dict[str, Any]]] = {g: [] for g in (*LEVELS, "noprior", "off_atlas")}
    for rec in q["queries"]:
        bgr = cv2.imread(str(rdir / rec["file"]), cv2.IMREAD_COLOR)
        tv = rec["truth"]
        R_true = rotation(tv["yaw_deg"], tv["pitch_deg"], tv["roll_deg"])
        C_true = np.asarray(tv["C"], float)
        p = rec["prior"]
        prior = PosePrior(C=tuple(p["C"]), yaw_deg=p["yaw_deg"], pitch_deg=p["pitch_deg"],
                          roll_deg=p["roll_deg"], sigma_pos_m=p["sigma_pos_m"],
                          sigma_yaw_deg=p["sigma_yaw_deg"])
        runs: list[tuple[str, np.ndarray, Optional[PosePrior]]] = []
        if rec["in_atlas"]:
            for lvl in LEVELS if accuracy else ("clean",):
                runs.append((lvl, perturb.apply(bgr, rec["perturb"][lvl]), prior))
            if accuracy and rec["id"] < q.get("noprior_count", 0):
                runs.append(("noprior", bgr, None))
        elif accuracy:
            runs.append(("off_atlas", bgr, prior))
        for group, img, pr in runs:
            fix = loc.localize(img, K, pr)
            row: dict[str, Any] = {"id": rec["id"], "ok": fix.ok, "reason": fix.reason,
                                   "t": fix.timings_ms, "sigma_h_m": fix.sigma_h_m,
                                   "inliers": fix.stats.get("inliers")}
            if fix.ok:
                assert fix.R is not None and fix.C is not None
                row["err"] = pose_error(fix.R, fix.C, R_true, C_true)
                row["yaw_est"] = angles(fix.R)[0]
            groups[group].append(row)
    return {"atlas_hash": atlas.meta["content_hash"], "atlas_keyframes": atlas.n_keyframes,
            "atlas_points": atlas.n_features, "atlas_mb": round(atlas.nbytes() / 2 ** 20, 2),
            "threads": threads, "opencv_threads": cv2.getNumThreads(),
            "groups": {g: _stats(rows) for g, rows in groups.items() if rows}}


def _table(report: dict[str, Any]) -> str:
    lines = ["| res | extractor | threads | group | n | fix | refused | false fix (>2 m) | "
             "horiz err p50/p95/max m | yaw err p50/p95 ° | ≤2σ_h | total ms p50/p95 | "
             "extract / match / pnp ms p50 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, res in report["results"].items():
        r, ex, th = key.split("|")
        for g, s in res["groups"].items():
            he, ye, tm = s["horiz_err_m"], s["yaw_err_deg"], s["timing_ms"]

            def f(v: Optional[float], d: int = 2) -> str:
                return "—" if v is None else f"{v:.{d}f}"
            lines.append(
                f"| {r} | {ex} | {th} | {g} | {s['n']} | {s['fix_rate']:.0%} | {s['refusal_rate']:.0%} "
                f"| {s['false_fix']} ({s['false_fix_rate']:.1%}) | {f(he['p50'])}/{f(he['p95'])}/{f(he['max'])} "
                f"| {f(ye['p50'])}/{f(ye['p95'])} | {f(s['within_2sigma_h'])} "
                f"| {f(tm['total']['p50'], 0)}/{f(tm['total']['p95'], 0)} "
                f"| {f(tm['extract']['p50'], 0)} / {f(tm['match']['p50'], 0)} / {f(tm['pnp']['p50'], 0)} |")
    return "\n".join(lines)


def cmd_localize(a: argparse.Namespace) -> int:
    cv2 = require_cv2()
    data = Path(a.data)
    resdirs = sorted(p for p in data.iterdir() if p.is_dir() and (p / "queries.json").is_file())
    if a.resolutions_filter:
        keep = set(a.resolutions_filter.split(","))
        resdirs = [p for p in resdirs if p.name in keep]
    if not resdirs:
        raise SystemExit(f"{data}: no <W>x<H>/queries.json (export with `run` first)")
    threads = [int(t) for t in str(a.threads).split(",")]
    host = {"machine": platform.machine(), "processor": platform.processor() or platform.platform(),
            "python": sys.version.split()[0], "opencv": cv2.__version__,
            "cpus": cv2.getNumberOfCPUs(), "numpy": np.__version__}
    log(f"localize: {data} on {host['machine']} ({host['processor']}), OpenCV {host['opencv']}, "
        f"{host['cpus']} CPUs")
    report: dict[str, Any] = {"host": host, "false_fix_m": FALSE_FIX_M, "results": {},
                              "note": ("rendered twin vs rendered twin (same splat): method ceiling, "
                                       "not the real-camera domain gap; timings are of this host")}
    for rdir in resdirs:
        exs = a.extractors.split(",") if a.extractors else \
            sorted(p.stem.removeprefix("atlas_") for p in rdir.glob("atlas_*.npz"))
        for ex in exs:
            for i, th in enumerate(threads):
                t0 = time.perf_counter()
                res = _localize_set(rdir, ex, th, accuracy=(i == 0 or a.full_each))
                key = f"{rdir.name}|{ex}|{th}"
                report["results"][key] = res
                c = res["groups"].get("clean", {})
                log(f"   {key}: clean fix {c.get('fix_rate')}, false {c.get('false_fix')}, "
                    f"horiz p95 {c.get('horiz_err_m', {}).get('p95')} m, total p50 "
                    f"{c.get('timing_ms', {}).get('total', {}).get('p50')} ms "
                    f"({time.perf_counter() - t0:.0f} s)")
    name = a.report or f"report_{host['machine'].lower()}"
    (data / f"{name}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    md = _table(report)
    (data / f"{name}.md").write_text(md + "\n", encoding="utf-8")
    log(md)
    bad = {k: s["false_fix"] for k, r in report["results"].items() for g, s in r["groups"].items()
           if s["false_fix"]}
    if bad:
        log(f"FALSE FIXES: {bad}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aerosplat.vps.bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="ground station: build atlases + query set, then localize (GPU)")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--vehicle", default=DEFAULT_VEHICLE)
    r.add_argument("--center", default=None, help="E,N of the area centre (default: tallest building)")
    r.add_argument("--half", type=float, default=60.0, help="half size of the atlas area, m")
    r.add_argument("--spacing", type=float, default=30.0, help="keyframe grid spacing, m")
    r.add_argument("--kf-alts", type=float, nargs="+", default=[25.0, 40.0, 55.0])
    r.add_argument("--kf-yaws", type=int, default=8)
    r.add_argument("--queries", type=int, default=100)
    r.add_argument("--noprior", type=int, default=30, help="how many queries also run without a prior")
    r.add_argument("--off-atlas", type=int, default=30)
    r.add_argument("--features", type=int, default=1000, help="features per frame at 960 px width")
    r.add_argument("--resolutions", default="960x540,1920x1080")
    r.add_argument("--extractors", default="orb,akaze")
    r.add_argument("--seed", type=int, default=1910)
    for p in (r, sub.add_parser("localize", help="companion: localize an exported query set (CPU)")):
        p.add_argument("--threads", default="4,1", help="OpenCV threads, comma-separated "
                                                         "(accuracy groups run with the first)")
        p.add_argument("--full-each", action="store_true", help="every group for every thread count")
        p.add_argument("--report", default=None, help="report file stem (default: report_<machine>)")
        p.add_argument("--resolutions-filter", default=None, help="e.g. 960x540")
        if p is not r:
            p.add_argument("--data", required=True)
            p.add_argument("--extractors", default=None)
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_localize(a)


if __name__ == "__main__":
    sys.exit(main())
