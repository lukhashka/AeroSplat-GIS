import { useState } from "react";

import { usePlanMission, useSite, useTakeoffSearch, useTargets } from "../api/hooks";
import type { GeoPoint, MissionFinding, MissionPlan, MissionRefusal, MissionTarget, TakeoffQuery, TakeoffSearch } from "../api/types";
import {
  fmtEnu,
  type HomeChoice,
  MAX_CHECKS_LIMIT,
  parseCount,
  parseOptional,
  planBody,
  profileFromForm,
  refusalOf,
  rejectedRows,
  SEARCH_DEFAULTS,
  searchKey,
} from "../lib/takeoff";
import { Icon } from "./Icon";
import { ErrorBox } from "./Modal";
import { TakeoffMap } from "./TakeoffMap";

const TARGET_STATE: Record<string, string> = { draft: "чернетка", confirmed: "підтверджена", revoked: "відкликана" };
const VERDICT_LABEL: Record<string, string> = {
  feasible: "можна летіти",
  feasible_with_risks: "можна, з ризиками",
  infeasible: "не можна",
};
const VERDICT_TONE: Record<string, string> = { feasible: "ok", feasible_with_risks: "run", infeasible: "bad" };

interface Found {
  key: string;
  targetId: string;
  query: TakeoffQuery;
  result: TakeoffSearch;
}

/** Plan a mission for a confirmed target: find a takeoff spot near it, then plan from the spot. */
export function MissionPlanner({ project, site, onPlanned }: { project: string; site: string; onPlanned: (missionId: string) => void }) {
  const siteQ = useSite(project, site);
  const targets = useTargets(project, site);
  const search = useTakeoffSearch(project, site);
  const plan = usePlanMission(project, site);

  const [targetId, setTargetId] = useState("");
  const [runId, setRunId] = useState("");
  const [form, setForm] = useState({ sigma_gnss_m: "", r_drone_m: "" });
  const [reach, setReach] = useState("");
  const [checks, setChecks] = useState("");
  const [found, setFound] = useState<Found | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [planned, setPlanned] = useState<MissionPlan | null>(null);

  const list = targets.data ?? [];
  const tid = targetId || list.find((t) => t.state === "confirmed")?.id || "";
  const target = list.find((t) => t.id === tid) ?? null;
  const runs = (siteQ.data?.runs ?? []).filter((r) => r.status === "ok" && "occupancy" in r.products);
  const origin = siteQ.data?.site.origin ?? null;

  const prof = profileFromForm(form);
  const dist = parseOptional(reach);
  const budget = parseCount(checks, MAX_CHECKS_LIMIT);
  const formError =
    "error" in prof
      ? prof.error
      : "error" in dist
        ? `max_distance_m: ${dist.error}`
        : "error" in budget
          ? `max_checks: ${budget.error}`
          : null;
  const profile = "profile" in prof ? prof.profile : undefined;
  const query: TakeoffQuery | null = formError
    ? null
    : {
        ...(runId ? { run_id: runId } : {}),
        ...(profile ? { profile } : {}),
        ...("value" in dist && dist.value !== undefined ? { max_distance_m: dist.value } : {}),
        ...("value" in budget && budget.value !== undefined ? { max_checks: budget.value } : {}),
      };
  const key = query && tid ? searchKey(tid, query) : null;
  const stale = found !== null && found.key !== key;
  const chosen = found && !stale ? (found.result.candidates.find((c) => c.rank === selected) ?? null) : null;
  const busy = search.isPending || plan.isPending;
  const pendingSearchHome = plan.isPending && !!plan.variables?.home_search;

  const find = () => {
    if (!query || !key) return;
    plan.reset();
    setPlanned(null);
    search.mutate(
      { targetId: tid, query },
      {
        onSuccess: (result) => {
          setFound({ key, targetId: tid, query, result });
          setSelected(result.candidates[0]?.rank ?? null);
        },
      },
    );
  };

  const planWith = (home: HomeChoice) => {
    // a chosen candidate is planned with the parameters it was found with (they equal the form's: not stale)
    const q = home.kind === "candidate" && found ? found.query : query;
    if (!q || !tid) return;
    setPlanned(null);
    plan.mutate(planBody(tid, home, { run_id: q.run_id, profile: q.profile }), {
      onSuccess: (m) => {
        setPlanned(m);
        onPlanned(m.id);
      },
    });
  };

  const refusal = refusalOf(plan.error);
  const reachUsed = query?.max_distance_m ?? SEARCH_DEFAULTS.max_distance_m;

  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Нова місія</h2>
        <span className="faint">ціль → місце зльоту → план</span>
      </div>
      <div className="panel-body planner">
        <div className="row planner-form">
          <label className="field">
            <span>Ціль</span>
            <select value={tid} onChange={(e) => setTargetId(e.target.value)} disabled={!list.length}>
              {list.map((t) => (
                <option key={t.id} value={t.id} disabled={t.state === "revoked"}>
                  {t.ap_class} · т. доступу …{t.ap_id.slice(-6)} · ціль …{t.id.slice(-6)} · {TARGET_STATE[t.state] ?? t.state}
                  {t.handover.overridden.length ? ` · змінено: ${t.handover.overridden.join(", ")}` : ""}
                </option>
              ))}
              {!list.length && <option value="">немає цілей</option>}
            </select>
          </label>
          <label className="field">
            <span>Двійник (run)</span>
            <select value={runId} onChange={(e) => setRunId(e.target.value)}>
              <option value="">найновіший з occupancy</option>
              {runs.map((r) => (
                <option key={`${r.epoch_id}/${r.id}`} value={r.id}>
                  {r.epoch_id}/{r.id}
                </option>
              ))}
            </select>
          </label>
          <NumberField label="σ_GNSS, м" hint="типово сервера: 1.5" value={form.sigma_gnss_m} onChange={(v) => setForm((f) => ({ ...f, sigma_gnss_m: v }))} />
          <NumberField label="r_drone, м" hint="типово: 0.5" value={form.r_drone_m} onChange={(v) => setForm((f) => ({ ...f, r_drone_m: v }))} />
          <NumberField label="Пошук до, м" hint={`типово: ${SEARCH_DEFAULTS.max_distance_m}`} value={reach} onChange={setReach} />
          <NumberField
            label="Точних перевірок"
            hint={`типово: ${SEARCH_DEFAULTS.max_checks}`}
            title={`max_checks, 1…${MAX_CHECKS_LIMIT}: скільки місць пошук перевіряє точно, перш ніж сказати «не підтверджено»`}
            value={checks}
            onChange={setChecks}
          />
          <button className="btn primary" disabled={!tid || !query || busy} onClick={find}>
            <Icon name="pin" /> {search.isPending ? "Шукаю…" : "Знайти місце зльоту"}
          </button>
        </div>

        {target && <TargetFacts target={target} />}
        {formError && <div className="error-box">{formError}</div>}
        {targets.data && !list.length && (
          <div className="note">
            На ділянці немає цілей. Ціль створюють з перевіреної точки доступу й підтверджують (API v2{" "}
            <span className="mono">POST /targets</span>, <span className="mono">…/confirm</span>).
          </div>
        )}
        <ErrorBox error={targets.error ?? siteQ.error ?? search.error} />

        {found && (
          <>
            {stale && <div className="note warn">Параметри змінились після пошуку: результат нижче — для попередніх. Шукайте знову, щоб планувати з місцем.</div>}
            <TakeoffResult search={found.result} origin={origin} selected={selected} onSelect={setSelected} />
          </>
        )}

        <div className="row planner-actions">
          <button
            className="btn primary"
            disabled={!chosen || busy}
            title={
              chosen
                ? `Дім = місце #${chosen.rank} (E/N ${fmtEnu(chosen.enu)} м). Сервер повторить пошук з тими самими параметрами; якщо #${chosen.rank} вже інше місце — 409 takeoff_spot_changed`
                : "Спершу знайдіть і виберіть місце зльоту"
            }
            onClick={() => chosen && found && planWith({ kind: "candidate", rank: chosen.rank, enu: chosen.enu, search: found.query })}
          >
            <Icon name="play" />{" "}
            {plan.isPending && !pendingSearchHome
              ? "Планую…"
              : chosen
                ? `Спланувати з місцем #${chosen.rank}`
                : "Спланувати з вибраним місцем"}
          </button>
          <button
            className="btn"
            disabled={!tid || !query || busy}
            title="home_search: сервер сам шукає місця зльоту з тим самим профілем і бере найближче; 409 no_takeoff_spot, якщо немає жодного"
            onClick={() => planWith({ kind: "search" })}
          >
            {pendingSearchHome ? "Планую…" : "Спланувати з автопошуком дому"}
          </button>
          <span className="faint grow">
            Автопошук на сервері — до {SEARCH_DEFAULTS.max_distance_m} м, {SEARCH_DEFAULTS.n} місць
            {reachUsed !== SEARCH_DEFAULTS.max_distance_m ? ` (не ${reachUsed} м, як у пошуку вище)` : ""}; бере найближче.
          </span>
        </div>

        {refusal ? <RefusalBox refusal={refusal} status={(plan.error as { status?: number }).status ?? null} /> : <ErrorBox error={plan.error} />}
        {planned && <PlannedBox plan={planned} />}
      </div>
    </div>
  );
}

function NumberField({
  label,
  hint,
  title,
  value,
  onChange,
}: {
  label: string;
  hint: string;
  title?: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <label className="field planner-num" title={title}>
      <span>{label}</span>
      <input type="text" inputMode="decimal" value={value} placeholder={hint} onChange={(e) => onChange(e.target.value)} />
    </label>
  );
}

function TargetFacts({ target: t }: { target: MissionTarget }) {
  return (
    <div className="row frame-note">
      <span className={`badge ${t.state === "confirmed" ? "ok" : t.state === "revoked" ? "bad" : "wait"}`}>{TARGET_STATE[t.state] ?? t.state}</span>
      <span className="badge mute">
        {t.handover.mode} · відступ {t.handover.standoff_m} м{t.handover.entry_distance_m ? ` · вхід ${t.handover.entry_distance_m} м` : ""}
      </span>
      <span className="badge mute" title={t.sigma.georef.source}>
        σ_georef {t.sigma.georef.value_m} м · σ_ap {t.sigma.ap.value_m} м
      </span>
      <span className="mono faint">{t.id}</span>
    </div>
  );
}

function TakeoffResult({
  search: s,
  origin,
  selected,
  onSelect,
}: {
  search: TakeoffSearch;
  origin: GeoPoint | null;
  selected: number | null;
  onSelect: (rank: number) => void;
}) {
  const rows = rejectedRows(s.rejected);
  const none = !s.candidates.length;
  return (
    <div className="takeoff-layout">
      <div className="takeoff-map">
        {origin ? <TakeoffMap origin={origin} search={s} selected={selected} onSelect={onSelect} /> : <div className="map empty">Немає початку ділянки для карти.</div>}
      </div>
      <div className="takeoff-side">
        <div className={`note${none ? " warn" : ""}`}>
          <div className="faint" style={{ fontSize: 11.5, marginBottom: 4 }}>
            Відповідь сервера
          </div>
          <span className="server-text">{s.message}</span>
        </div>
        <div className="row chips">
          <span className="badge mute">r = {s.radius_m.toFixed(2)} м</span>
          <span className="badge mute">σ_transit {s.sigma_transit_m.toFixed(2)} м</span>
          <span className="badge mute">
            {s.searched_columns} колонок · {s.checked} точних перевірок
          </span>
          <span className="badge mute">місця ≥ {s.spacing_m.toFixed(0)} м одне від одного</span>
          {s.grid_limited && <span className="badge run">пошук обрізано краєм сітки</span>}
        </div>

        {s.candidates.length > 0 && (
          <div className="table-wrap">
            <table className="list compact">
              <thead>
                <tr>
                  <th>#</th>
                  <th>E, N, U (м)</th>
                  <th className="num">До цілі</th>
                  <th className="num">Відкрито</th>
                  <th className="num">Підйом</th>
                  <th className="num">Над DTM</th>
                </tr>
              </thead>
              <tbody>
                {s.candidates.map((c) => (
                  <tr key={c.rank} className={`click${c.rank === selected ? " on" : ""}`} onClick={() => onSelect(c.rank)}>
                    <td>
                      <input type="radio" name="takeoff" checked={c.rank === selected} onChange={() => onSelect(c.rank)} aria-label={`Місце #${c.rank}`} /> {c.rank}
                    </td>
                    <td className="mono nowrap">
                      {fmtEnu(c.enu)}, {c.enu[2].toFixed(1)}
                      {c.notes.map((n) => (
                        <div key={n} className="server-text faint" style={{ fontFamily: "var(--sans)", whiteSpace: "normal" }}>
                          {n}
                        </div>
                      ))}
                    </td>
                    <td className="num">{c.distance_m.toFixed(0)} м</td>
                    <td className="num">≈{c.open_radius_m.toFixed(1)} м</td>
                    <td className="num">{c.climb_m !== null ? `${c.climb_m.toFixed(1)} м` : "—"}</td>
                    <td className="num">{c.above_dtm_m.toFixed(2)} м</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {s.rejected.not_checked ? (
          <div className="note warn">
            Пошук зупинився на ліміті точних перевірок ({s.checked}): {s.rejected.not_checked} колонок не перевірено. Збільште «Точних перевірок» (до{" "}
            {MAX_CHECKS_LIMIT}) і шукайте знову.
          </div>
        ) : null}

        {s.best_open && (
          <div className="note warn">
            <b>Найвідкритіше місце</b> (не підходить, «?» на карті): E/N {fmtEnu(s.best_open.enu)} м, {s.best_open.distance_m.toFixed(0)} м до цілі, відкрито ≈
            {s.best_open.open_radius_m.toFixed(1)} м при потрібних r = {s.radius_m.toFixed(2)} м.{" "}
            {s.sigma_gnss_would_fit_m !== null
              ? `Підійшло б із σ_GNSS ≤ ${s.sigma_gnss_would_fit_m.toFixed(2)} м (якщо більше ніщо не завадить).`
              : "Жодне σ_GNSS не зробить його придатним з цими r_drone і запасом."}
          </div>
        )}

        {none && rows.length > 0 && (
          <div>
            <div className="faint" style={{ fontSize: 12, marginBottom: 4 }}>
              Чому відкинуто колонки (за першою невдалою перевіркою)
            </div>
            <dl className="kv">
              {rows.map((r) => (
                <div key={r.code} style={{ display: "contents" }}>
                  <dt>
                    {r.label} <span className="mono faint">{r.code}</span>
                  </dt>
                  <dd className="num">{r.count}</dd>
                </div>
              ))}
            </dl>
          </div>
        )}
      </div>
    </div>
  );
}

function FindingList({ findings }: { findings: MissionFinding[] }) {
  return (
    <ul className="findings">
      {findings.map((f, i) => (
        <li key={i}>
          <span className={`badge ${f.severity === "blocking" ? "bad" : "run"}`}>{f.code}</span>
          {f.leg && <span className="faint"> {f.leg}</span>} <span className="server-text">{f.detail}</span>
        </li>
      ))}
    </ul>
  );
}

/** The planner said no: every reason, verbatim (outside_grid carries its occupancy_bbox hint). */
function RefusalBox({ refusal, status }: { refusal: MissionRefusal; status: number | null }) {
  return (
    <div className="error-box refusal">
      <div style={{ fontWeight: 600, marginBottom: 6 }}>Місію не сплановано{status ? ` (HTTP ${status})` : ""}</div>
      <FindingList findings={refusal.reasons} />
    </div>
  );
}

function PlannedBox({ plan: m }: { plan: MissionPlan }) {
  return (
    <div className="note">
      <div className="row" style={{ gap: 8, marginBottom: m.findings.length ? 6 : 0 }}>
        <b>
          Місію <span className="mono">{m.id.slice(-8)}</span> сплановано
        </b>
        <span className={`badge ${VERDICT_TONE[m.verdict] ?? "mute"}`}>{VERDICT_LABEL[m.verdict] ?? m.verdict}</span>
        <span className="muted">
          дім E/N {fmtEnu(m.home.enu)} м · транзит {(m.transit_z - m.home.enu[2]).toFixed(1)} м над домом · двійник {m.occupancy.run_id}
        </span>
      </div>
      <div className="faint server-text" style={{ fontSize: 12 }}>
        Дім: {m.home.source}
      </div>
      {m.findings.length > 0 && <FindingList findings={m.findings} />}
    </div>
  );
}
