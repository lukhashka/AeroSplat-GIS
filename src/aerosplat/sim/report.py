"""Report of an L2 run: `report.json` (everything, machine-readable), `trajectory.json` (the
recording), `clearance.json` (the clearance oracle's series, for the flight view of the web UI)
and a self-contained `report.html` (SIMULATION §4.1 "звіт"):

* verdicts of the oracles, expected vs got, and the outcome;
* the true trajectory over the twin (top view: column heights of the occupancy grid, unknown
  columns marked), the planned path, the uploaded fence, home / E / H, faults and failsafes;
* height above home vs time (truth, and the planned height of the item being flown to);
* clearance vs time, with the required r_drone + m;
* events (modes, mission progress, faults, STATUSTEXT) and the parameters that were applied.

No plotting library: SVG written here, colours as CSS tokens with a dark variant.

A run recorded before `clearance.json` existed gets one after the fact, against the occupancy
it was judged on: `python -m aerosplat.sim.report clearance <run dir> --occupancy <dir>`.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from aerosplat.surface.models import FREE, UNKNOWN

H_MAX_DISPLAY_M = 40.0          # the grey ramp of the top view saturates here (above home)
CLEARANCE_CAP_M = 12.0          # the clearance chart is cut here (the field saturates anyway)
REPORT_JSON = "report.json"
TRAJECTORY_JSON = "trajectory.json"
CLEARANCE_JSON = "clearance.json"


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o).__name__)


def report_dict(result) -> dict:
    sc = result.scenario
    rec = result.recording
    pkg = result.package
    d: dict[str, Any] = {
        "schema": 1,
        "scenario": sc.model_dump(mode="json"),
        "seed": result.seed,
        "passed": result.passed,
        "summary": result.summary,
        "outcome": result.outcome,
        "reason": result.reason,
        "detail": result.detail,
        "expected": {"outcome": sc.expect.outcome.value, "reason": sc.expect.reason,
                     "invariants": result.expected},
        "verdicts": [{"invariant": v.invariant, "status": v.status, "detail": v.detail,
                      "expected": result.expected.get(v.invariant), "metrics": v.metrics}
                     for v in result.verdicts],
        "preflight": result.preflight,
        "warnings": result.warnings,
        "firmware": result.firmware,
        "timing": result.timing,
        "faults": result.injected,
        "applied_params": result.applied,
        "response": None if result.response is None else {
            "trigger": result.response.trigger, "modes": list(result.response.modes),
            "by": list(result.response.by), "delay_s": result.response.delay_s},
        "mission": None if pkg is None else {"id": pkg.plan.id, "dir": str(pkg.dir),
                                             "site": pkg.plan.site_key,
                                             "verdict": pkg.plan.verdict.value,
                                             "origin": pkg.origin.model_dump()},
        "events": [] if rec is None else [{"t": e.t, "kind": e.kind, "detail": e.detail, **e.data}
                                          for e in rec.events],
        "samples": 0 if rec is None else len(rec),
    }
    return d


def write_report(result, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = report_dict(result)
    (out_dir / REPORT_JSON).write_text(json.dumps(d, indent=1, ensure_ascii=False,
                                                  default=_json_default) + "\n", encoding="utf-8")
    if result.recording is not None:
        result.recording.save(out_dir / TRAJECTORY_JSON)
    if result.clearance is not None:
        write_clearance(result.clearance, out_dir)
    (out_dir / "report.html").write_text(render_html(result, d), encoding="utf-8")
    return out_dir / "report.html"


def _finite_or_none(values) -> list[Optional[float]]:
    return [round(float(v), 3) if math.isfinite(float(v)) else None for v in values]


def clearance_dict(series) -> dict:
    """The clearance oracle's series as JSON: airborne samples only, on the recording's clock.

    `target_distance_m` is null outside the zone where the target surface is left out; values
    at `cap_m` mean "at least that" (the exact search stops there, oracles.clearance_series).
    """
    return {"schema": 1,
            "t": [round(float(v), 3) for v in series.t],
            "clearance_m": _finite_or_none(series.clearance),
            "target_distance_m": _finite_or_none(series.target_distance),
            "state": list(series.state),
            "required_m": float(series.required),
            "cap_m": float(series.cap) if math.isfinite(series.cap) else None}


def write_clearance(series, out_dir: Path) -> Path:
    path = Path(out_dir) / CLEARANCE_JSON
    path.write_text(json.dumps(clearance_dict(series), separators=(",", ":")), encoding="utf-8")
    return path


def mission_dir_of(run_dir: Path) -> Path:
    """`<mission>/sim/<scenario>[/seed-<n>]` -> `<mission>`."""
    run_dir = Path(run_dir)
    up = run_dir.parent if run_dir.name.startswith("seed-") else run_dir
    if up.parent.name != "sim":
        raise ValueError(f"{run_dir} is not <mission>/sim/<scenario>[/seed-<n>]: pass --mission")
    return up.parent.parent


def backfill_clearance(run_dir: Path, occupancy_dir: Path,
                       mission_dir: Optional[Path] = None) -> Path:
    """Write `clearance.json` for a run recorded without one, from its own files.

    The occupancy must be the one the run was judged on (report.json `scenario.occupancy`):
    the minimum of the series is then the clearance verdict's `min_clearance_m`, which is
    checked here so that a wrong grid does not pass for the right one.
    """
    from aerosplat.mission.models import MISSION_JSON, MissionPlan
    from aerosplat.sim.oracles import Context, clearance_series
    from aerosplat.sim.record import Recording
    from aerosplat.sim.scenario import Expect
    from aerosplat.surface.occupancy import read_occupancy

    run_dir = Path(run_dir)
    rec = Recording.load(run_dir / TRAJECTORY_JSON)
    report = json.loads((run_dir / REPORT_JSON).read_text(encoding="utf-8"))
    mdir = Path(mission_dir) if mission_dir is not None else mission_dir_of(run_dir)
    plan = MissionPlan(**json.loads((mdir / MISSION_JSON).read_text(encoding="utf-8")))
    if rec.meta.get("mission") not in (None, plan.id):
        raise ValueError(f"{run_dir} flew {rec.meta.get('mission')}, not {plan.id}")
    grid, _ = read_occupancy(Path(occupancy_dir))
    ctx = Context(plan=plan, items=[], planned_enu={}, fence_enu=[], params={}, grid=grid,
                  expect=Expect(**report["scenario"]["expect"]))
    series = clearance_series(rec, ctx)
    judged = next((v.get("metrics", {}).get("min_clearance_m") for v in report.get("verdicts", [])
                   if v.get("invariant") == "clearance"), None)
    if judged is not None and len(series.t) and abs(float(np.min(series.clearance)) - judged) > 0.01:
        raise ValueError(f"minimum clearance {float(np.min(series.clearance)):.3f} m against "
                         f"{occupancy_dir}, but the run was judged at {judged:.3f} m: "
                         f"not the occupancy it flew against")
    return write_clearance(series, run_dir)


# ── HTML ─────────────────────────────────────────────────────────────────
STYLE = """
:root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
--s1:#2a78d6;--s2:#eb6834;--s7:#6250d6;--good:#0ca30c;--bad:#d03b3b;--na:#898781;
--plan:#52514e;--fence:#52514e}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;
--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s7:#9085e9;
--plan:#c3c2b7;--fence:#c3c2b7}}
:root[data-theme="dark"]{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s7:#9085e9;--plan:#c3c2b7;--fence:#c3c2b7}
body{background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;
margin:0;padding:24px 16px}
main{max-width:1100px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px}
.sub{color:var(--ink2);margin:0 0 16px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 16px;margin:8px 0}
.verdict{font-size:18px;font-weight:600}
.badge{display:inline-block;border-radius:4px;padding:0 6px;font-weight:600;font-size:12px;
border:1px solid currentColor}
.pass{color:var(--good)}.fail{color:var(--bad)}.na{color:var(--na)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
.tbl{overflow-x:auto}td{overflow-wrap:anywhere}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--ink2);font-weight:600}
.scroll{max-height:420px;overflow:auto}
svg{display:block;width:100%;height:auto}
svg text{fill:var(--ink2);font-size:12px}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;color:var(--ink2);font-size:12px;margin:6px 0}
.legend i{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:6px;border-radius:2px}
.tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--border);
border-radius:6px;padding:4px 8px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.15);display:none;z-index:9}
code{font-size:12px}
"""

SCRIPT = """
document.querySelectorAll('svg[data-series]').forEach(function(svg){
  var d=JSON.parse(svg.getAttribute('data-series'));var tip=document.getElementById('tip');
  var cross=svg.querySelector('.cross'),dot=svg.querySelector('.dot');
  svg.addEventListener('mousemove',function(ev){
    var r=svg.getBoundingClientRect(),vb=svg.viewBox.baseVal,x=(ev.clientX-r.left)*vb.width/r.width;
    if(x<d.x0||x>d.x1){tip.style.display='none';cross.style.display='none';dot.style.display='none';return;}
    var t=d.t0+(x-d.x0)/(d.x1-d.x0)*(d.t1-d.t0),lo=0,hi=d.t.length-1;
    while(hi-lo>1){var m=(lo+hi)>>1;if(d.t[m]<t)lo=m;else hi=m;}
    var k=Math.abs(d.t[lo]-t)<Math.abs(d.t[hi]-t)?lo:hi,px=d.px[k],py=d.py[k];
    cross.setAttribute('x1',px);cross.setAttribute('x2',px);cross.style.display='';
    dot.setAttribute('cx',px);dot.setAttribute('cy',py);dot.style.display='';
    tip.innerHTML='t = '+d.t[k].toFixed(1)+' с<br>'+d.label+': <b>'+d.y[k].toFixed(2)+' м</b>'+
      (d.mode?'<br>режим '+d.mode[k]:'');
    tip.style.display='block';tip.style.left=(ev.clientX+12)+'px';tip.style.top=(ev.clientY+12)+'px';});
  svg.addEventListener('mouseleave',function(){tip.style.display='none';cross.style.display='none';dot.style.display='none';});
});
"""

STATUS_MARK = {"pass": ("✓", "pass"), "fail": ("✗", "fail"), "n/a": ("–", "na")}


def _esc(s: Any) -> str:
    return html.escape(str(s))


def _num(v: Optional[float], fmt: str = ".1f") -> str:
    return "—" if v is None else format(v, fmt)


def _changes(f: dict) -> str:
    return ", ".join(f"{n} = {v:g}" for n, v in f["params"]) or f["note"]


def _badge(status: Optional[str]) -> str:
    if status is None:
        return ""
    mark, cls = STATUS_MARK.get(status, ("?", "na"))
    return f"<span class='badge {cls}'>{mark} {_esc(status)}</span>"


def render_html(result, d: dict) -> str:
    sc = result.scenario
    ok = result.passed
    head = (f"<p class='verdict {'pass' if ok else 'fail'}'>{'✓ PASS' if ok else '✗ FAIL'} — "
            f"{_esc(result.summary)}</p>"
            f"<p>Очікувано: <b>{_esc(sc.expect.outcome.value)}</b> ({_esc(sc.expect.reason)}) · "
            f"отримано: <b>{_esc(result.outcome)}</b>"
            + (f" ({_esc(result.reason)})" if result.reason else "") + "</p>")
    if result.outcome in ("refused", "error") and result.detail:
        head += f"<p>Причина: <code>{_esc(result.detail)}</code></p>"
    rows = "".join(
        f"<tr><td>{_esc(v.invariant)}</td><td>{_badge(result.expected.get(v.invariant))}</td>"
        f"<td>{_badge(v.status)}</td><td>{_esc(v.detail)}</td></tr>" for v in result.verdicts)
    verdicts = ("<table><tr><th>Інваріант</th><th>Очікувано</th><th>Отримано</th><th>Деталі</th></tr>"
                f"{rows}</table>") if rows else "<p>Оракули не запускались (політ не відбувся).</p>"
    warn = "".join(f"<li>{_esc(w)}</li>" for w in result.warnings)
    fw = result.firmware
    meta = [("Сценарій", sc.scenario), ("Опис", sc.description or "—"),
            ("Ділянка / сцена", f"{sc.site or '—'} / {sc.scene or '—'}"
                                + (f" (варіант {sc.variant})" if sc.variant else "")),
            ("Місія", d["mission"]["id"] if d.get("mission") else "—"),
            ("Апарат", sc.vehicle), ("Прошивка", f"{fw.get('tag', '—')} {fw.get('git', '')}"),
            ("Seed", result.seed),
            ("Час", ", ".join(f"{k} {v:.1f}" for k, v in result.timing.items()))]
    meta_html = "".join(f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in meta)
    parts = [f"<!doctype html><html lang='uk'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>SITL: {_esc(sc.scenario)}</title><style>{STYLE}</style></head><body><main>",
             f"<h1>L2-прогін: {_esc(sc.scenario)}</h1>",
             "<p class='sub'>ArduCopter SITL (вбудована фізика); оракули — за істинним станом "
             "<code>SIM_STATE</code>, не за EKF.</p>",
             f"<div class='card'>{head}</div>", "<h2>Вердикти оракулів</h2>", verdicts]
    if warn:
        parts.append(f"<h2>Попередження</h2><ul>{warn}</ul>")
    rec = result.recording
    if rec is not None and len(rec) and result.package is not None:
        parts += ["<h2>Траєкторія над двійником (вид згори)</h2>", _top_view(result),
                  "<h2>Висота над домом</h2>", _height_chart(result),
                  "<h2>Кліренс за часом</h2>", _clearance_chart(result)]
    parts += ["<h2>Сценарій</h2>", f"<table>{meta_html}</table>"]
    if result.injected:
        frows = "".join(
            f"<tr><td>{_esc(f['fault'])}</td><td>{_esc(f['after'])} + {f['at_s']:g} с</td>"
            f"<td>{_num(f['on'])}</td><td>{_num(f['off'])}</td>"
            f"<td>{_esc(_changes(f))}</td></tr>" for f in result.injected)
        parts.append("<h2>Відмови</h2><table><tr><th>Відмова</th><th>Коли</th><th>Увімкнено, с</th>"
                     f"<th>Вимкнено, с</th><th>Що саме</th></tr>{frows}</table>")
    if rec is not None and rec.events:
        erows = "".join(f"<tr><td>{e.t:.2f}</td><td>{_esc(e.kind)}</td><td>{_esc(e.detail)}</td></tr>"
                        for e in rec.events)
        parts.append("<h2>Події (час симуляції, с)</h2><div class='scroll'><table><tr><th>t</th>"
                     f"<th>Подія</th><th>Деталі</th></tr>{erows}</table></div>")
    if result.applied:
        prow = "".join(f"<tr><td><code>{_esc(a['name'])}</code></td><td>{_esc(a['source'])}</td>"
                       f"<td>{_num(a['before'], 'g')}</td><td>{a['value']:g}</td></tr>"
                       for a in result.applied)
        parts.append("<h2>Застосовані параметри</h2><div class='scroll'><table><tr><th>Параметр</th>"
                     f"<th>Джерело</th><th>Було</th><th>Стало</th></tr>{prow}</table></div>")
    parts.append(f"<div id='tip' class='tip'></div><script>{SCRIPT}</script></main></body></html>")
    # Tables scroll inside their own box on a narrow screen instead of widening the page.
    out = "".join(parts).replace("<table>", "<div class='tbl'><table>").replace("</table>", "</table></div>")
    return out


# ── top view ─────────────────────────────────────────────────────────────
def _scene_png(grid, x0: float, x1: float, y0: float, y1: float, home_z: float) -> tuple[str, tuple]:
    """Column heights (above home) of the grid inside the box as a grey PNG; unknown tops tinted.
    Returns (base64, (x0, x1, y0, y1) actually covered)."""
    from PIL import Image

    s = grid.spec
    j0 = max(0, int(math.floor((x0 - s.x0) / s.res_m)))
    j1 = min(s.cols, int(math.ceil((x1 - s.x0) / s.res_m)))
    i0 = max(0, int(math.floor((y0 - s.y0) / s.res_m)))
    i1 = min(s.rows, int(math.ceil((y1 - s.y0) / s.res_m)))
    if j1 <= j0 or i1 <= i0:
        return "", (x0, x1, y0, y1)
    st = grid.state[i0:i1, j0:j1, :]
    blocked = st != FREE
    any_b = blocked.any(axis=2)
    top = st.shape[2] - 1 - np.argmax(blocked[:, :, ::-1], axis=2)
    h = np.where(any_b, s.z0 + (top + 1) * s.res_m - home_z, np.nan)
    top_state = np.take_along_axis(st, top[..., None], axis=2)[..., 0]
    v = np.clip(np.nan_to_num(h, nan=0.0) / H_MAX_DISPLAY_M, 0, 1)
    g = (236 - v * 150).astype(np.uint8)              # light grey ground -> dark grey high
    rgb = np.stack([g, g, g], axis=-1)
    unk = any_b & (top_state == UNKNOWN)
    rgb[unk] = (rgb[unk] * 0.55 + np.array([98, 80, 214]) * 0.45).astype(np.uint8)
    rgb = rgb[::-1]                                  # rows grow north; image rows grow down
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, format="PNG")
    cov = (s.x0 + j0 * s.res_m, s.x0 + j1 * s.res_m, s.y0 + i0 * s.res_m, s.y0 + i1 * s.res_m)
    return base64.b64encode(buf.getvalue()).decode("ascii"), cov


def _poly(points, fx, fy) -> str:
    return " ".join(f"{fx(x):.1f},{fy(y):.1f}" for x, y in points)


def _top_view(result) -> str:
    rec, pkg, grid = result.recording, result.package, result.grid
    p = rec.positions
    plan = pkg.plan
    fence = np.asarray(pkg.fence_enu, float)
    planned = np.asarray([v for _, v in sorted(pkg.planned_enu.items())], float)
    allxy = np.vstack([p[:, :2], fence, planned[:, :2]])
    lo, hi = allxy.min(axis=0) - 6.0, allxy.max(axis=0) + 6.0
    span = float(max(hi - lo))
    cx, cy = (lo + hi) / 2
    x0, x1, y0, y1 = cx - span / 2, cx + span / 2, cy - span / 2, cy + span / 2
    W = 640
    pad = 36
    k = (W - 2 * pad) / span

    def fx(x: float) -> float:
        return pad + (x - x0) * k

    def fy(y: float) -> float:
        return W - pad - (y - y0) * k

    img = ""
    if grid is not None:
        b64, (ax0, ax1, ay0, ay1) = _scene_png(grid, x0, x1, y0, y1, plan.home.enu[2])
        if b64:
            img = (f"<image href='data:image/png;base64,{b64}' x='{fx(ax0):.1f}' y='{fy(ay1):.1f}' "
                   f"width='{(ax1 - ax0) * k:.1f}' height='{(ay1 - ay0) * k:.1f}' "
                   f"preserveAspectRatio='none' style='image-rendering:pixelated'/>")
    # Split the true path where the vehicle left AUTO (a failsafe flies the rest).
    modes = np.asarray(rec.mode)
    armed = np.asarray(rec.armed, bool)
    start = result.timing.get("mission_start_sim_s", -math.inf)
    t = rec.times
    off = (modes != "AUTO") & (t >= start) & armed
    cut = int(np.argmax(off)) if off.any() else len(p)
    seg_a = p[:cut + 1, :2] if cut < len(p) else p[:, :2]
    seg_b = p[cut:, :2] if cut < len(p) else np.zeros((0, 2))
    svg = [f"<svg viewBox='0 0 {W} {W}' role='img' aria-label='вид згори'>"
           f"<rect x='0' y='0' width='{W}' height='{W}' fill='var(--surface)'/>", img,
           f"<polygon points='{_poly(fence, fx, fy)}' fill='none' stroke='var(--fence)' "
           f"stroke-width='1.5' stroke-dasharray='6 4'/>",
           f"<polyline points='{_poly(planned[:, :2], fx, fy)}' fill='none' stroke='var(--plan)' "
           f"stroke-width='1.5' stroke-dasharray='2 3'/>",
           f"<polyline points='{_poly(seg_a, fx, fy)}' fill='none' stroke='var(--s1)' stroke-width='2' "
           f"stroke-linejoin='round'/>"]
    if len(seg_b) > 1:
        svg.append(f"<polyline points='{_poly(seg_b, fx, fy)}' fill='none' stroke='var(--s2)' "
                   f"stroke-width='2' stroke-linejoin='round'/>")
    marks = [("дім", plan.home.enu), ("E", plan.approach.entry), ("H", plan.approach.hover)]
    for name, q in marks:
        svg.append(f"<circle cx='{fx(q[0]):.1f}' cy='{fy(q[1]):.1f}' r='4.5' fill='var(--surface)' "
                   f"stroke='var(--ink)' stroke-width='1.5'/><text x='{fx(q[0]) + 7:.1f}' "
                   f"y='{fy(q[1]) - 7:.1f}' style='fill:var(--ink)'>{name}</text>")
    for e in rec.events_of("fault_on"):
        i = int(np.clip(np.searchsorted(t, e.t), 0, len(t) - 1))
        q = p[i]
        svg.append(f"<rect x='{fx(q[0]) - 4:.1f}' y='{fy(q[1]) - 4:.1f}' width='8' height='8' "
                   f"fill='var(--s2)' stroke='var(--surface)' stroke-width='2'/>"
                   f"<text x='{fx(q[0]) + 8:.1f}' y='{fy(q[1]) + 14:.1f}'>{_esc(e.detail)}</text>")
    # Scale bar
    bar = 10 ** math.floor(math.log10(max(span / 4, 1)))
    svg.append(f"<line x1='{pad}' y1='{W - 14}' x2='{pad + bar * k:.1f}' y2='{W - 14}' "
               f"stroke='var(--ink2)' stroke-width='2'/><text x='{pad}' y='{W - 20}'>{bar:g} м</text>"
               f"<text x='{W - pad - 8}' y='{pad - 10}'>N ↑</text></svg>")
    legend = ("<div class='legend'><span><i style='background:var(--s1)'></i>істинна траєкторія (AUTO)</span>"
              "<span><i style='background:var(--s2)'></i>після виходу з AUTO (failsafe)</span>"
              "<span><i style='background:var(--plan);height:1px'></i>план (точки місії)</span>"
              "<span><i style='background:var(--fence);height:1px'></i>геофенс (завантажений)</span>"
              "<span>сірий — висота колонок над домом (до 40 м); фіолетовий відтінок — "
              "верх колонки невідомий (= зайнято)</span></div>")
    return f"<div class='card'>{''.join(svg)}{legend}</div>"


# ── time series ──────────────────────────────────────────────────────────
def _series_chart(t, y, *, label: str, ymin: float, ymax: float, refs=(), steps=None,
                  events=(), modes=None, below: Optional[float] = None) -> str:
    W, Hh, L, R, T, B = 720, 280, 46, 14, 24, 30
    t0, t1 = float(t[0]), float(max(t[-1], t[0] + 1e-3))

    def fx(v: float) -> float:
        return L + (v - t0) / (t1 - t0) * (W - L - R)

    def fy(v: float) -> float:
        return T + (1 - (min(max(v, ymin), ymax) - ymin) / (ymax - ymin)) * (Hh - T - B)

    out = [f"<rect x='0' y='0' width='{W}' height='{Hh}' fill='var(--surface)'/>"]
    nt = 6
    for i in range(nt + 1):
        v = ymin + (ymax - ymin) * i / nt
        out.append(f"<line x1='{L}' x2='{W - R}' y1='{fy(v):.1f}' y2='{fy(v):.1f}' stroke='var(--grid)'/>"
                   f"<text x='{L - 6}' y='{fy(v) + 4:.1f}' text-anchor='end'>{v:.0f}</text>")
    step = 10 ** math.floor(math.log10(max((t1 - t0) / 6, 1e-3)))
    for m in (1, 2, 5, 10):
        if (t1 - t0) / (step * m) <= 8:
            step *= m
            break
    v = math.ceil(t0 / step) * step
    while v <= t1:
        out.append(f"<text x='{fx(v):.1f}' y='{Hh - 10}' text-anchor='middle'>{v:.0f}</text>")
        v += step
    out.append(f"<line x1='{L}' x2='{W - R}' y1='{Hh - B}' y2='{Hh - B}' stroke='var(--axis)'/>"
               f"<text x='{W - R}' y='{Hh - 10}' text-anchor='end'>с</text>"
               f"<text x='{L}' y='12'>{_esc(label)}, м</text>")
    for ev_t, ev_label in events:
        x = fx(ev_t)
        out.append(f"<line x1='{x:.1f}' x2='{x:.1f}' y1='{T}' y2='{Hh - B}' stroke='var(--muted)' "
                   f"stroke-dasharray='2 3'/><text x='{x + 3:.1f}' y='{T + 10}'>{_esc(ev_label)}</text>")
    for val, lab in refs:
        out.append(f"<line x1='{L}' x2='{W - R}' y1='{fy(val):.1f}' y2='{fy(val):.1f}' "
                   f"stroke='var(--bad)' stroke-width='1.5' stroke-dasharray='6 4'/>"
                   f"<text x='{W - R - 4}' y='{fy(val) - 5:.1f}' text-anchor='end'>{_esc(lab)}</text>")
    if steps is not None:
        out.append(f"<polyline points='{' '.join(f'{fx(a):.1f},{fy(b):.1f}' for a, b in steps)}' "
                   f"fill='none' stroke='var(--plan)' stroke-width='1.5' stroke-dasharray='4 3'/>")
    px = [fx(float(a)) for a in t]
    py = [fy(float(b)) for b in y]
    out.append(f"<polyline points='{' '.join(f'{a:.1f},{b:.1f}' for a, b in zip(px, py))}' fill='none' "
               f"stroke='var(--s1)' stroke-width='2' stroke-linejoin='round'/>")
    if below is not None:
        for a, b, yy in zip(px, py, y):
            if yy < below:
                out.append(f"<circle cx='{a:.1f}' cy='{b:.1f}' r='3' fill='var(--bad)'/>")
    out.append("<line class='cross' y1='%d' y2='%d' stroke='var(--ink2)' style='display:none'/>"
               "<circle class='dot' r='4' fill='var(--s1)' stroke='var(--surface)' stroke-width='2' "
               "style='display:none'/>" % (T, Hh - B))
    data = {"t": [round(float(a), 2) for a in t], "y": [round(float(b), 3) for b in y],
            "px": [round(a, 1) for a in px], "py": [round(b, 1) for b in py], "x0": L, "x1": W - R,
            "t0": t0, "t1": t1, "label": label, "mode": modes}
    return (f"<svg viewBox='0 0 {W} {Hh}' data-series='{html.escape(json.dumps(data))}' role='img' "
            f"aria-label='{_esc(label)}'>{''.join(out)}</svg>")


def _events_for_chart(rec, start: float) -> list[tuple[float, str]]:
    out = [(e.t, e.detail) for e in rec.events_of("mode") if e.t >= start]
    out += [(e.t, f"⚡ {e.detail}") for e in rec.events_of("fault_on")]
    return sorted(out)


def _height_chart(result) -> str:
    rec, pkg = result.recording, result.package
    hz = pkg.plan.home.enu[2]
    t = rec.times
    z = rec.positions[:, 2] - hz
    alt = {it.seq: it.alt for it in pkg.items}
    # The planned height of the item being flown to; not the home row (seq 0, absolute) nor LAND.
    steps = [(float(a), float(alt[int(s)])) for a, s in zip(t, rec.seq)
             if 0 < int(s) < len(pkg.items) and pkg.items[int(s)].command != 21]
    start = result.timing.get("mission_start_sim_s", float(t[0]))
    ymax = max(5.0, float(np.nanmax(z)) * 1.15 + 1)
    chart = _series_chart(t, z, label="істинна висота над домом", ymin=min(0.0, float(np.nanmin(z))),
                          ymax=ymax, steps=steps or None, events=_events_for_chart(rec, start),
                          modes=list(rec.mode))
    legend = ("<div class='legend'><span><i style='background:var(--s1)'></i>істина (SIM_STATE)</span>"
              "<span><i style='background:var(--plan);height:1px'></i>запланована висота поточної "
              "точки (.waypoints, над домом)</span></div>")
    return f"<div class='card'>{chart}{legend}</div>"


def _clearance_chart(result) -> str:
    s = result.clearance
    if s is None or len(s.t) == 0:
        return "<p>Немає даних кліренсу.</p>"
    y = np.minimum(s.clearance, CLEARANCE_CAP_M)
    ymax = max(CLEARANCE_CAP_M if float(np.max(s.clearance)) >= CLEARANCE_CAP_M else float(np.max(y)) + 1,
               s.required + 1)
    start = result.timing.get("mission_start_sim_s", float(s.t[0]))
    sl_modes = None
    rec = result.recording
    if rec is not None:
        armed = np.flatnonzero(np.asarray(rec.armed, bool))
        if len(armed):
            sl_modes = list(rec.mode[int(armed[0]):int(armed[-1]) + 1])
    chart = _series_chart(s.t, y, label="кліренс до зайнятого/невідомого", ymin=0.0, ymax=ymax,
                          refs=[(s.required, f"потрібно r_дрона + m = {s.required:.2f} м")],
                          events=_events_for_chart(rec, start) if rec is not None else (),
                          modes=sl_modes, below=s.required)
    legend = ("<div class='legend'><span><i style='background:var(--s1)'></i>кліренс істинної позиції "
              f"(обрізано на {CLEARANCE_CAP_M:g} м)</span><span><i style='background:var(--bad)'></i>"
              "поріг; червоні точки — нижче порогу</span><span>поверхні, яких місія торкається навмисно "
              "(земля під домом і під точкою посадки, площина цілі на підході), не враховано</span></div>")
    return f"<div class='card'>{chart}{legend}</div>"


def main(argv: Optional[list[str]] = None) -> int:
    from aerosplat.logging_utils import fix_console_encoding

    fix_console_encoding()
    ap = argparse.ArgumentParser(prog="python -m aerosplat.sim.report",
                                 description="after-the-fact products of an L2 run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    cl = sub.add_parser("clearance", help="write clearance.json for a run recorded without it")
    cl.add_argument("run_dir", type=Path, help="<mission>/sim/<scenario>[/seed-<n>]")
    cl.add_argument("--occupancy", type=Path, required=True,
                    help="the occupancy folder the run was judged on")
    cl.add_argument("--mission", type=Path, help="mission folder (default: from run_dir)")
    args = ap.parse_args(argv)
    print(f"wrote {backfill_clearance(args.run_dir, args.occupancy, args.mission)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
