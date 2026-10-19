import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";

import { refreshScenes, useMissionGeometry, useMissions, useSimReport, useSimTrajectory, useSiteSims } from "../api/hooks";
import type { FlightTrajectory, MissionGeometry, SimReport, SimRunSummary } from "../api/types";
import { FlightTimeline } from "../components/FlightTimeline";
import { Icon } from "../components/Icon";
import { MissionMap } from "../components/MissionMap";
import { MissionPlanner } from "../components/MissionPlanner";
import { MissionScene, type SceneLayer, type SceneStatus, type SceneView } from "../components/MissionScene";
import { ErrorBox } from "../components/Modal";
import {
  atCap,
  eventText,
  FAILSAFE_COLOR,
  feedEvents,
  indexAt,
  INVARIANT_LABEL,
  LAYER_COLORS,
  PHASE_GROUPS,
  PHASE_LABEL,
  phaseColor,
} from "../lib/flight";

const SPEEDS = [1, 4, 10, 30] as const;
const VERDICT_LABEL: Record<string, string> = {
  feasible: "можна летіти",
  feasible_with_risks: "можна, з ризиками",
  infeasible: "не можна",
};
const LAYER_LABEL: Record<SceneLayer, string> = {
  corridor: "Коридор",
  fence: "Геофенс",
  target: "Ціль",
  waypoints: "Точки",
  trajectory: "Траєкторія",
  drone: "Дрон",
};
// Where in the flight a verdict happened, from its metrics (aerosplat.sim.oracles).
const SEEK_KEYS = ["at_t", "first_breach_t", "trigger_t", "arrived_t", "faced_from_t"];

// The replay opens where a delivery is judged: the arrival at H (else the end of the flight).
function openingTime(tr: FlightTrajectory): number {
  const k = tr.phase.indexOf("hold");
  return tr.t[k >= 0 ? k : tr.t.length - 1] ?? 0;
}

const runKey = (r: { scenario: string; seed: number | null }) => (r.seed === null ? r.scenario : `${r.scenario}@${r.seed}`);
const fmtClock = (s: number) => `${s < 0 ? "−" : ""}${Math.floor(Math.abs(s) / 60)}:${(Math.abs(s) % 60).toFixed(1).padStart(4, "0")}`;

function StatusMark({ status }: { status: string }) {
  if (status === "pass") return <span className="badge ok">✓ pass</span>;
  if (status === "fail") return <span className="badge bad">✗ fail</span>;
  return <span className="badge mute">– {status}</span>;
}

/** "Місія / проліт": a mission and one of its simulated flights, in 3D over the twin and on the map. */
export function FlightPage() {
  const { project = "", site = "" } = useParams();
  const [params, setParams] = useSearchParams();
  const missions = useMissions(project, site);
  const sims = useSiteSims(project, site);
  const [planning, setPlanning] = useState(false);
  // Open by default while the site has no missions, and kept open once shown so: planning the
  // first mission refetches the list, which must not unmount the panel before it shows the verdict.
  const noMissions = !!missions.data && !missions.data.length;
  useEffect(() => {
    if (noMissions) setPlanning(true);
  }, [noMissions]);
  const showPlanner = planning || noMissions;

  // mission and run: from the URL, else the newest mission that has flown, else the first one
  const flown = useMemo(() => new Set((sims.data ?? []).map((r) => r.mission_id)), [sims.data]);
  const missionId =
    params.get("mission") ??
    (missions.data ?? []).find((m) => flown.has(m.id))?.id ??
    missions.data?.[0]?.id ??
    null;
  const runs = useMemo(() => (sims.data ?? []).filter((r) => r.mission_id === missionId), [sims.data, missionId]);
  const run: SimRunSummary | null = runs.find((r) => runKey(r) === params.get("run")) ?? runs[0] ?? null;

  const geometry = useMissionGeometry(project, site, missionId);
  const report = useSimReport(project, site, missionId, run?.scenario ?? null, run?.seed ?? null);
  const trajectory = useSimTrajectory(project, site, missionId, run?.has_trajectory ? run.scenario : null, run?.seed ?? null);

  const select = (mission: string, r?: string) => {
    const next = new URLSearchParams(params);
    next.set("mission", mission);
    if (r) next.set("run", r);
    else next.delete("run");
    setParams(next, { replace: true });
  };

  return (
    <div className="page" style={{ maxWidth: "none", gap: 16 }}>
      <div className="page-head">
        <div style={{ minWidth: 0 }}>
          <div className="crumbs">
            <Link to="/">Ділянки</Link> / <Link to={`/sites/${project}/${site}`}>{`${project}/${site}`}</Link> / Місія
          </div>
          <h1>Місія / проліт</h1>
        </div>
        <div className="row">
          <button
            className={`btn${showPlanner ? " primary" : ""}`}
            title="Спланувати місію до підтвердженої цілі: місце зльоту, план, вердикт"
            onClick={() => setPlanning((p) => !p)}
          >
            <Icon name="plus" /> Нова місія
          </button>
          <label className="field inline">
            <span>Місія</span>
            <select value={missionId ?? ""} onChange={(e) => select(e.target.value)} disabled={!missions.data?.length}>
              {(missions.data ?? []).map((m) => (
                <option key={m.id} value={m.id}>
                  {m.id.slice(-8)} · {m.ap_id.slice(-6)} · {VERDICT_LABEL[m.verdict] ?? m.verdict}
                  {flown.has(m.id) ? "" : " · без прольоту"}
                </option>
              ))}
            </select>
          </label>
          <label className="field inline">
            <span>Проліт</span>
            <select
              value={run ? runKey(run) : ""}
              onChange={(e) => missionId && select(missionId, e.target.value)}
              disabled={!runs.length}
            >
              {runs.map((r) => (
                <option key={runKey(r)} value={runKey(r)}>
                  {r.passed ? "✓" : "✗"} {r.scenario}
                  {r.seed !== null ? ` · seed ${r.seed}` : ""}
                </option>
              ))}
              {!runs.length && <option value="">немає прольотів SITL</option>}
            </select>
          </label>
        </div>
      </div>

      <ErrorBox error={missions.error ?? sims.error ?? geometry.error ?? report.error ?? trajectory.error} />
      {showPlanner && (
        <MissionPlanner
          project={project}
          site={site}
          onPlanned={(id) => select(id)}
        />
      )}
      {missions.data && !missions.data.length && (
        <div className="panel">
          <div className="empty">
            <div className="big">На ділянці ще немає місій</div>
            Сплануйте першу з підтвердженої цілі вище; проліт — раннером SITL:{" "}
            <span className="mono">python -m aerosplat.sim.runner scenario.yaml</span>.
          </div>
        </div>
      )}
      {geometry.data && (
        <Flight
          key={`${geometry.data.id}/${run ? runKey(run) : ""}`}
          geometry={geometry.data}
          trajectory={trajectory.data ?? null}
          report={report.data ?? null}
          run={run}
        />
      )}
    </div>
  );
}

function Flight({
  geometry,
  trajectory,
  report,
  run,
}: {
  geometry: MissionGeometry;
  trajectory: FlightTrajectory | null;
  report: SimReport | null;
  run: SimRunSummary | null;
}) {
  // which scene: the twin the plan was checked against, if the server can show it
  const [scenes, setScenes] = useState<string[] | null>(null);
  const [useScene, setUseScene] = useState(true);
  useEffect(() => {
    let live = true;
    refreshScenes()
      .then((r) => live && setScenes(r.scenes))
      .catch(() => live && setScenes([]));
    return () => {
      live = false;
    };
  }, []);
  const sceneAvailable = scenes?.includes(geometry.twin.scene) ?? false;
  const scene = useScene && sceneAvailable ? geometry.twin.scene : null;
  const [status, setStatus] = useState<SceneStatus>({ state: "loading" });

  const [layers, setLayers] = useState<Record<SceneLayer, boolean>>({
    corridor: true,
    fence: true,
    target: true,
    waypoints: true,
    trajectory: true,
    drone: true,
  });
  const [xray, setXray] = useState(false);
  const [follow, setFollow] = useState(false);
  const [view, setView] = useState<{ name: SceneView; n: number }>({ name: "mission", n: 0 });

  // replay clock, in simulation seconds
  const tr = trajectory;
  const start = tr?.mission_start_t ?? tr?.t[0] ?? 0;
  const [time, setTime] = useState<number>(() => (tr ? openingTime(tr) : 0));
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(10);
  const index = tr ? indexAt(tr.t, time) : 0;
  useEffect(() => {
    if (tr) setTime(openingTime(tr));
  }, [tr]);
  const raf = useRef(0);
  useEffect(() => {
    if (!playing || !tr) return;
    let last = performance.now();
    const end = tr.t[tr.t.length - 1] ?? 0;
    const step = (now: number) => {
      const dt = ((now - last) / 1000) * speed;
      last = now;
      setTime((t) => {
        const next = t + dt;
        if (next >= end) {
          setPlaying(false);
          return end;
        }
        return next;
      });
      raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf.current);
  }, [playing, speed, tr]);
  const seek = (k: number) => tr && setTime(tr.t[Math.max(0, Math.min(k, tr.t.length - 1))]!);
  const seekTime = (t: number) => tr && setTime(t);

  const events = useMemo(() => (tr ? feedEvents(tr.events) : []), [tr]);
  const lastEvent = [...events].reverse().find((e) => e.t <= time + 1e-6);

  return (
    <>
      <FrameNote geometry={geometry} trajectory={tr} status={status} scenes={scenes} sceneAvailable={sceneAvailable} useScene={useScene} />

      <div className="flight-layout">
        <div className="panel flight-3d">
          <div className="panel-head">
            <div className="row">
              <div className="segmented">
                {(
                  [
                    ["mission", "Місія"],
                    ["target", "Ціль"],
                    ["side", "Профіль"],
                    ["top", "Зверху"],
                  ] as const
                ).map(([v, label]) => (
                  <button key={v} onClick={() => setView((s) => ({ name: v, n: s.n + 1 }))}>
                    {label}
                  </button>
                ))}
              </div>
              <div className="segmented" title="Що під місією">
                <button className={scene ? "on" : ""} disabled={!sceneAvailable} onClick={() => setUseScene(true)}>
                  3DGS
                </button>
                <button className={!scene ? "on" : ""} onClick={() => setUseScene(false)}>
                  Сітка
                </button>
              </div>
            </div>
            <div className="row chips">
              {(Object.keys(LAYER_LABEL) as SceneLayer[]).map((k) => (
                <button key={k} className={`chip${layers[k] ? " on" : ""}`} onClick={() => setLayers((l) => ({ ...l, [k]: !l[k] }))}>
                  {LAYER_LABEL[k]}
                </button>
              ))}
              <button className={`chip${xray ? " on" : ""}`} title="Траєкторію й точки — поверх сплатів" onClick={() => setXray((x) => !x)}>
                Поверх сцени
              </button>
              <button className={`chip${follow ? " on" : ""}`} title="Камера йде за дроном" onClick={() => setFollow((x) => !x)}>
                Стежити
              </button>
            </div>
          </div>
          <div className="flight-stage">
            <MissionScene
              scene={scene}
              geometry={geometry}
              trajectory={tr}
              index={index}
              layers={layers}
              xray={xray}
              follow={follow}
              view={view}
              onStatus={setStatus}
            />
            {status.state === "loading" && <div className="render-busy">{scene ? `Завантаження сцени ${scene}…` : "Підготовка…"}</div>}
            <Legend />
          </div>
        </div>

        <div className="flight-side">
          <Readout geometry={geometry} trajectory={tr} index={index} start={start} lastEvent={lastEvent ? eventText(lastEvent) : null} />
          <VerdictPanel report={report} run={run} onSeek={seekTime} />
        </div>
      </div>

      {tr ? (
        <div className="panel">
          <div className="panel-head">
            <div className="row">
              <button className="btn sm primary" onClick={() => {
                if (!playing && index >= tr.t.length - 1) setTime(start);
                setPlaying((p) => !p);
              }}>
                <Icon name={playing ? "stop" : "play"} size={14} /> {playing ? "Пауза" : "Відтворити"}
              </button>
              <div className="segmented">
                {SPEEDS.map((s) => (
                  <button key={s} className={s === speed ? "on" : ""} onClick={() => setSpeed(s)}>
                    {s}×
                  </button>
                ))}
              </div>
              <span className="mono muted">
                {fmtClock(time - start)} / {fmtClock((tr.t[tr.t.length - 1] ?? 0) - start)}
              </span>
            </div>
            <input
              className="grow scrub"
              type="range"
              min={tr.t[0]}
              max={tr.t[tr.t.length - 1]}
              step={0.05}
              value={time}
              onChange={(e) => {
                setPlaying(false);
                setTime(Number(e.target.value));
              }}
              aria-label="Час прольоту"
            />
          </div>
          <div className="panel-body">
            <FlightTimeline trajectory={tr} index={index} events={events} onSeek={(k) => { setPlaying(false); seek(k); }} />
          </div>
        </div>
      ) : (
        run && !run.has_trajectory && <div className="note">Проліт «{run.scenario}» без траєкторії: раннер відмовив до зльоту ({run.summary}).</div>
      )}

      <div className="grid-2">
        <div className="panel">
          <div className="panel-head">
            <h2>Вид згори</h2>
            <span className="faint">клік біля треку — перейти до моменту</span>
          </div>
          <div className="panel-body flush">
            <MissionMap geometry={geometry} trajectory={tr} index={index} onSeek={(k) => { setPlaying(false); seek(k); }} />
          </div>
        </div>
        <div className="panel" style={{ display: "flex", flexDirection: "column" }}>
          <div className="panel-head">
            <h2>Події</h2>
            <span className="faint">{events.length}</span>
          </div>
          <div className="event-feed">
            {events.map((e, i) => (
              <button key={i} className={`event-row${e.t <= time + 1e-6 ? " past" : ""}${lastEvent === e ? " current" : ""}`} onClick={() => seekTime(e.t)}>
                <span className="mono">{fmtClock(e.t - start)}</span>
                <span className={`ev-kind ${e.kind}`}>{e.kind}</span>
                <span className="grow">{eventText(e)}</span>
              </button>
            ))}
            {!events.length && <div className="empty">Немає подій.</div>}
          </div>
          <MissionFacts geometry={geometry} />
        </div>
      </div>
    </>
  );
}

function FrameNote({
  geometry,
  trajectory,
  status,
  scenes,
  sceneAvailable,
  useScene,
}: {
  geometry: MissionGeometry;
  trajectory: FlightTrajectory | null;
  status: SceneStatus;
  scenes: string[] | null;
  sceneAvailable: boolean;
  useScene: boolean;
}) {
  const o = geometry.origin;
  const items: { tone: "ok" | "bad" | "mute"; text: string }[] = [
    { tone: "mute", text: `Site ENU · початок ${o.lat.toFixed(7)}, ${o.lon.toFixed(7)}, h ${o.h.toFixed(1)} м` },
  ];
  if (status.state === "ready" && status.originOffsetM !== null) {
    items.push(
      status.originOffsetM < 0.001
        ? { tone: "ok", text: status.scene ? `сцена ${status.scene}: той самий ENU-кадр` : "без сцени: сітка на висоті дому" }
        : { tone: "bad", text: `сцена в іншому ENU (${status.originOffsetM.toFixed(2)} м між початками): перераховано через ECEF` },
    );
  }
  if (trajectory && trajectory.origin_offset_m > 0.01)
    items.push({ tone: "bad", text: `траєкторію записано з іншим початком (${trajectory.origin_offset_m.toFixed(2)} м)` });
  if (status.state === "error") items.push({ tone: "bad", text: status.message });
  if (scenes && !sceneAvailable && useScene)
    items.push({ tone: "bad", text: `3DGS-сцени «${geometry.twin.scene}» на сервері немає: місію показано над сіткою` });
  if (geometry.twin.legacy) items.push({ tone: "mute", text: `двійник: legacy ${geometry.twin.run_id.replace(/^legacy:/, "")}` });
  return (
    <div className="row frame-note">
      {items.map((x, i) => (
        <span key={i} className={`badge ${x.tone}`}>
          {x.text}
        </span>
      ))}
    </div>
  );
}

function Legend() {
  return (
    <div className="flight-legend">
      {PHASE_GROUPS.map((g) => (
        <span key={g.id}>
          <i style={{ background: g.color }} />
          {g.label}
        </span>
      ))}
      <span>
        <i style={{ background: FAILSAFE_COLOR }} />
        failsafe (не AUTO)
      </span>
      <span>
        <i className="ring" style={{ borderColor: LAYER_COLORS.corridor }} />
        коридор (радіус плану)
      </span>
      <span>
        <i className="dash" style={{ borderColor: LAYER_COLORS.fence }} />
        геофенс
      </span>
    </div>
  );
}

function Readout({
  geometry,
  trajectory: tr,
  index,
  start,
  lastEvent,
}: {
  geometry: MissionGeometry;
  trajectory: FlightTrajectory | null;
  index: number;
  start: number;
  lastEvent: string | null;
}) {
  if (!tr || !tr.t.length)
    return (
      <div className="panel">
        <div className="panel-body muted">Немає траєкторії для відтворення.</div>
      </div>
    );
  const k = Math.min(index, tr.t.length - 1);
  const p = tr.enu[k]!;
  const h = geometry.target.hover.enu;
  const dH = Math.hypot(p[0] - h[0], p[1] - h[1], p[2] - h[2]);
  const clr = tr.clearance;
  const c = clr?.clearance_m[k] ?? null;
  const phase = tr.phase[k]!;
  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Зараз</h2>
        <span className="mono muted">{fmtClock(tr.t[k]! - start)}</span>
      </div>
      <div className="panel-body readouts">
        <div>
          <span className="lbl">Фаза</span>
          <span className="val">
            <i className="swatch" style={{ background: phaseColor(phase, tr.failsafe[k]) }} />
            {PHASE_LABEL[phase] ?? phase}
          </span>
        </div>
        <div>
          <span className="lbl">Режим</span>
          <span className="val">
            {tr.failsafe[k] ? <span className="badge bad">{tr.mode[k]} · failsafe</span> : <span className="badge mute">{tr.mode[k]}</span>}
          </span>
        </div>
        <div>
          <span className="lbl">Висота над домом</span>
          <span className="val mono">{tr.alt_rel_m[k]!.toFixed(2)} м</span>
        </div>
        <div>
          <span className="lbl">Кліренс</span>
          <span className="val mono">
            {clr === null ? (
              <span className="faint" title="Прогін без clearance.json: python -m aerosplat.sim.report clearance …">
                не записано
              </span>
            ) : c === null ? (
              "—"
            ) : (
              <span style={{ color: c < clr.required_m ? "var(--red)" : undefined }}>
                {atCap(c, clr.cap_m) ? "≥ " : ""}
                {c.toFixed(2)} м
                <span className="faint"> / {clr.required_m.toFixed(1)}</span>
              </span>
            )}
          </span>
        </div>
        <div>
          <span className="lbl">Швидкість</span>
          <span className="val mono">{tr.speed_mps[k]!.toFixed(2)} м/с</span>
        </div>
        <div>
          <span className="lbl">До H</span>
          <span className="val mono">{dH.toFixed(2)} м</span>
        </div>
        <div className="wide">
          <span className="lbl">Подія</span>
          <span className="val">{lastEvent ?? "—"}</span>
        </div>
      </div>
    </div>
  );
}

function VerdictPanel({ report, run, onSeek }: { report: SimReport | null; run: SimRunSummary | null; onSeek: (t: number) => void }) {
  if (!run)
    return (
      <div className="panel">
        <div className="panel-body muted">Цю місію ще не літали в SITL.</div>
      </div>
    );
  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Вердикт оракулів</h2>
        {run.passed ? <span className="badge ok">✓ PASS</span> : <span className="badge bad">✗ FAIL</span>}
      </div>
      <div className="panel-body" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        <div>
          <div style={{ fontWeight: 600 }}>{run.summary}</div>
          <div className="faint" style={{ fontSize: 12 }}>
            {run.scenario} · {run.outcome}
            {run.reason ? ` (${run.reason})` : ""} · {run.firmware ?? "?"}
            {run.mission_sim_s ? ` · ${run.mission_sim_s.toFixed(0)} с польоту` : ""}
          </div>
          {run.description && <div className="muted" style={{ fontSize: 12.5, marginTop: 4 }}>{run.description}</div>}
        </div>
        <div className="verdicts">
          {(report?.verdicts ?? run.verdicts.map((v) => ({ ...v, metrics: {} as Record<string, unknown> }))).map((v) => {
            const at = SEEK_KEYS.map((key) => v.metrics[key]).find((x): x is number => typeof x === "number");
            const unexpected = v.expected && v.status !== v.expected && !(v.status === "n/a" && v.expected === "pass");
            return (
              <button key={v.invariant} className={`verdict-row${unexpected ? " unexpected" : ""}`} disabled={at === undefined} onClick={() => at !== undefined && onSeek(at)} title={at !== undefined ? "Перейти до цього моменту" : undefined}>
                <StatusMark status={v.status} />
                <span className="name">{INVARIANT_LABEL[v.invariant] ?? v.invariant}</span>
                <span className="detail">
                  {v.detail}
                  {unexpected ? ` — очікувано ${v.expected}` : ""}
                </span>
              </button>
            );
          })}
        </div>
        {report && report.faults.length > 0 && (
          <div className="muted" style={{ fontSize: 12.5 }}>
            Відмови: {report.faults.map((f) => `${f.fault} (${f.reason})${f.on !== null ? ` @ ${f.on.toFixed(1)} с` : ""}`).join("; ")}
          </div>
        )}
        {report && report.warnings.length > 0 && (
          <div className="note" style={{ fontSize: 12.5 }}>
            {report.warnings.join(" · ")}
          </div>
        )}
      </div>
    </div>
  );
}

function MissionFacts({ geometry: g }: { geometry: MissionGeometry }) {
  return (
    <div className="panel-body" style={{ borderTop: "1px solid var(--border)" }}>
      <dl className="kv">
        <dt>Ціль</dt>
        <dd>
          {g.target.ap_class}
          {g.target.facade_id ? ` · ${g.target.facade_id}` : ""} · <span className="mono">{g.target.ap_id}</span>
        </dd>
        <dt>План</dt>
        <dd>
          {VERDICT_LABEL[g.verdict] ?? g.verdict}
          {g.findings.length ? ` · ${g.findings.map((f) => f.code).join(", ")}` : ""}
        </dd>
        <dt>Курс у H</dt>
        <dd>{g.target.heading_deg === null ? "вільний" : `${g.target.heading_deg.toFixed(1)}°`}</dd>
        <dt>Коридор</dt>
        <dd>
          {g.legs.map((l) => `${l.name} ${l.radius_m.toFixed(2)} м`).join(" · ")}
        </dd>
        <dt>Геофенс</dt>
        <dd>
          {g.fence.polygon_enu.length} вершин, до {g.fence.alt_max_rel_m.toFixed(0)} м над домом
        </dd>
      </dl>
    </div>
  );
}
