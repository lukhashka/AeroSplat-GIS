import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";

import { ApiError, renderFacadeOrtho } from "../api/client";
import { useAccessPointEdits, useAccessPoints, useCarried, useFacade, useTaxonomy } from "../api/hooks";
import type {
  AccessPoint,
  CarriedOut,
  CoverageGrid,
  FacadeGrid,
  FacadeOrthoMeta,
  GridfitOut,
  Lifecycle,
  TaxonomyClass,
} from "../api/types";
import { Icon } from "../components/Icon";
import { ErrorBox } from "../components/Modal";

/** ACCESS_POINTS §5.3 wants 1–2 cm/px to read openings; coarser is for looking, not drawing. */
const GSD_CHOICES = [2, 3, 5, 10, 20];
const LIFECYCLE_LABEL: Record<Lifecycle, string> = {
  candidate: "кандидат",
  verified: "підтверджено",
  rejected: "відхилено",
  stale: "перевірити",
};
const LIFECYCLE_TONE: Record<Lifecycle, string> = {
  candidate: "var(--amber)",
  verified: "var(--green)",
  rejected: "var(--red)",
  stale: "var(--blue)",
};

type Draft = { u0: number; v0: number; u1: number; v1: number };

const SKIPPED_LABEL: Record<Lifecycle, string> = {
  candidate: "вже кандидати",
  verified: "вже підтверджені",
  rejected: "раніше відхилені",
  stale: "на перевірці",
};

/** One line on what the grid fit did: how many candidates, from what grid, or why not. */
function gridfitSummary(g: GridfitOut): { text: string; warn: boolean } {
  const r = g.result;
  if (g.refused)
    return { text: `Сітку не знайдено: ${r.refusals.map((x) => x.message).join("; ")}`, warn: true };
  const grid = [
    r.floor && `поверх ${r.floor.floor_m.toFixed(2)} м`,
    r.bay_m != null && `крок колонок ${r.bay_m.toFixed(2)} м`,
  ].filter(Boolean);
  const skipped = (Object.entries(g.skipped) as [Lifecycle, number][])
    .filter(([, n]) => n > 0)
    .map(([lc, n]) => `${n} ${SKIPPED_LABEL[lc]}`);
  return {
    text:
      `${g.candidates.length} нових кандидатів` +
      (grid.length ? ` (${grid.join(", ")})` : "") +
      (skipped.length ? ` · пропущено: ${skipped.join(", ")}` : "") +
      (g.candidates.length ? " · перевірте кожен: V підтвердити, X відхилити" : ""),
    warn: false,
  };
}

/** Pixel ↔ (u, v) for the orthophoto, in *image* pixels (0…width, 0…height at the edges). */
function mapping(g: FacadeGrid) {
  const m = g.m_per_px;
  const [u0, v0] = g.pixel00_uv;
  return {
    m,
    xOf: (u: number) => (u - u0) / m + 0.5,
    yOf: (v: number) => (v0 - v) / m + 0.5,
    uOf: (x: number) => u0 + (x - 0.5) * m,
    vOf: (y: number) => v0 - (y - 0.5) * m,
  };
}

/** Smallest and mean view count over the cells a rectangle covers (0 = the splat invented it). */
function coverageUnder(grid: CoverageGrid | null, r: Draft): { min: number; mean: number } | null {
  if (!grid || !grid.views.length) return null;
  const cell = (x: number) => grid.cell_m && Math.floor(x / grid.cell_m);
  const i0 = Math.max(0, cell(Math.min(r.u0, r.u1) - grid.u_min));
  const i1 = Math.min(grid.nu - 1, cell(Math.max(r.u0, r.u1) - grid.u_min));
  const j0 = Math.max(0, cell(Math.min(r.v0, r.v1) - grid.v_min));
  const j1 = Math.min(grid.nv - 1, cell(Math.max(r.v0, r.v1) - grid.v_min));
  let min = Infinity;
  let sum = 0;
  let n = 0;
  for (let j = j0; j <= j1; j++) {
    const row = grid.views[j];
    if (!row) continue;
    for (let i = i0; i <= i1; i++) {
      const value = row[i] ?? 0;
      min = Math.min(min, value);
      sum += value;
      n += 1;
    }
  }
  return n ? { min, mean: sum / n } : null;
}

/**
 * The coverage map, washed over the orthophoto. Where no capture resolved the wall
 * finely enough, 3DGS fills the facade in from nothing and a detector — or a person —
 * will happily find windows that are not there (§5.3). So the uncovered part has to be
 * visibly marked, not merely recorded in a report.
 */
function CoverageWash({
  grid,
  box,
}: {
  grid: CoverageGrid;
  box: { left: number; top: number; width: number; height: number };
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    const ctx = cv?.getContext("2d");
    if (!cv || !ctx) return;
    ctx.clearRect(0, 0, cv.width, cv.height);
    for (let j = 0; j < grid.nv; j++) {
      const row = grid.views[j];
      for (let i = 0; i < grid.nu; i++) {
        const views = row?.[i] ?? 0;
        // Canvas row 0 is the top; the grid's row 0 is the bottom of the wall.
        const y = grid.nv - 1 - j;
        if (views === 0) ctx.fillStyle = "rgba(255, 70, 70, 0.42)";
        else if (views <= 2) ctx.fillStyle = "rgba(255, 184, 77, 0.22)";
        else continue;
        ctx.fillRect(i, y, 1, 1);
      }
    }
  }, [grid]);
  return (
    <canvas
      ref={ref}
      className="facade-cov"
      width={grid.nu}
      height={grid.nv}
      style={{
        left: `${box.left}%`,
        top: `${box.top}%`,
        width: `${box.width}%`,
        height: `${box.height}%`,
      }}
    />
  );
}

function ClassList({
  classes,
  value,
  onPick,
}: {
  classes: TaxonomyClass[];
  value: string;
  onPick: (id: string) => void;
}) {
  return (
    <div className="class-list">
      {classes.map((c, i) => (
        <button
          key={c.id}
          className={`class-row${c.id === value ? " on" : ""}`}
          onClick={() => onPick(c.id)}
          title={c.id}
        >
          <kbd>{i + 1}</kbd>
          <span className="grow">{c.labels.uk}</span>
          {c.size_prior_m && (
            <small className="faint num">
              {c.size_prior_m.width[0]}–{c.size_prior_m.width[1]} м
            </small>
          )}
        </button>
      ))}
    </div>
  );
}

export function FacadePage() {
  // The facade id has a slash in it ("B01/N1"), so the route carries it as a splat.
  const { project = "", site = "", "*": facadeKey = "" } = useParams();
  // `?run=<id>` pins the editor to one run (facade ids differ between runs); default: newest.
  const [params] = useSearchParams();
  const runId = params.get("run");

  const facade = useFacade(project, site, facadeKey || null, runId);
  const taxonomy = useTaxonomy(project, site);
  // The run this facade belongs to, once known: features are listed by frame, not by id,
  // because the same id names another wall in another run (§5.9).
  const facadeRun = facade.data?.run_id ?? null;
  const features = useAccessPoints(project, site, facadeKey || null, facadeRun, !!facadeRun);
  const carried = useCarried(project, site, facadeKey || null, facadeRun);
  const edits = useAccessPointEdits(project, site, facadeKey || null);

  const [gsd, setGsd] = useState(5);
  const [showCoverage, setShowCoverage] = useState(true);
  const [shot, setShot] = useState<{ url: string; meta: FacadeOrthoMeta } | null>(null);
  const [busy, setBusy] = useState(false);
  const [renderError, setRenderError] = useState<unknown>(null);
  const [classId, setClassId] = useState("opening.window");
  const [draft, setDraft] = useState<Draft | null>(null);
  const [dragging, setDragging] = useState<Draft | null>(null);
  // The authoritative drag lives in a ref, not in state: when the browser coalesces
  // move and up into one frame, a handler closure still holds the pre-drag state and
  // the rectangle is lost.
  const dragRef = useRef<Draft | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [hover, setHover] = useState<{ u: number; v: number } | null>(null);
  const [help, setHelp] = useState(false);
  const [addressNote, setAddressNote] = useState<string | null>(null);
  const [gridNote, setGridNote] = useState<{ text: string; warn: boolean } | null>(null);
  // The note is about one facade; moving to another must not leave it behind.
  useEffect(() => setGridNote(null), [facadeKey]);
  const stageRef = useRef<HTMLDivElement>(null);

  // The classes an operator can put on a wall: openings first, then the constraints
  // that matter for approach. Pads live on roofs and the ground, not here.
  const classes = useMemo(() => {
    const all = taxonomy.data?.classes ?? [];
    const wall = all.filter((c) => c.kind === "opening" || c.kind === "constraint");
    return [...wall].sort((a, b) => (a.kind === b.kind ? a.id.localeCompare(b.id) : a.kind === "opening" ? -1 : 1));
  }, [taxonomy.data]);

  useEffect(() => {
    if (classes.length && !classes.some((c) => c.id === classId)) setClassId(classes[0]!.id);
  }, [classes, classId]);

  // Render the wall head-on. One request per (facade, cm/px); the GPU is behind a lock.
  useEffect(() => {
    if (!facadeKey) return;
    const ctl = new AbortController();
    setBusy(true);
    renderFacadeOrtho(project, site, facadeKey, { gsd_cm: gsd, run_id: runId }, ctl.signal)
      .then((r) => {
        setShot((old) => {
          if (old) URL.revokeObjectURL(old.url);
          return r;
        });
        setRenderError(null);
      })
      .catch((e) => {
        if ((e as Error).name !== "AbortError") setRenderError(e);
      })
      .finally(() => {
        if (!ctl.signal.aborted) setBusy(false);
      });
    return () => ctl.abort();
  }, [project, site, facadeKey, gsd, runId]);

  const grid = shot?.meta.facade_grid ?? null;
  const map = useMemo(() => (grid ? mapping(grid) : null), [grid]);
  const coverage = facade.data?.coverage_grid ?? null;
  const list = features.data ?? [];
  const current = list.find((f) => f.id === selected) ?? null;
  // Features of other runs projected onto this wall: shown dashed, carried only on request.
  const foreign = carried.data ?? [];
  const currentForeign = foreign.find((c) => c.feature.id === selected) ?? null;
  const pickedClass = classes.find((c) => c.id === classId) ?? null;

  // Where the coverage grid sits on the image, as a percentage box, so CSS scales it.
  const covBox = useMemo(() => {
    if (!coverage || !map || !shot) return null;
    const { width, height } = shot.meta;
    const left = map.xOf(coverage.u_min);
    const right = map.xOf(coverage.u_min + coverage.nu * coverage.cell_m);
    const bottom = map.yOf(coverage.v_min);
    const top = map.yOf(coverage.v_min + coverage.nv * coverage.cell_m);
    return {
      left: (100 * left) / width,
      top: (100 * top) / height,
      width: (100 * (right - left)) / width,
      height: (100 * (bottom - top)) / height,
    };
  }, [coverage, map, shot]);

  const uvAt = (ev: { clientX: number; clientY: number }): { u: number; v: number } | null => {
    const box = stageRef.current?.getBoundingClientRect();
    if (!box || !map || !shot) return null;
    const x = ((ev.clientX - box.left) / box.width) * shot.meta.width;
    const y = ((ev.clientY - box.top) / box.height) * shot.meta.height;
    return { u: map.uOf(x), v: map.vOf(y) };
  };

  const save = () => {
    // `isPending` as well as the draft: two Enters in one frame would otherwise both
    // see the draft still set and store the opening twice.
    if (!draft || !facadeKey || edits.create.isPending) return;
    edits.create.mutate(
      {
        facade_id: facadeKey,
        run_id: runId,
        class: classId,
        polygon_uv: [
          [Math.min(draft.u0, draft.u1), Math.min(draft.v0, draft.v1)],
          [Math.max(draft.u0, draft.u1), Math.min(draft.v0, draft.v1)],
          [Math.max(draft.u0, draft.u1), Math.max(draft.v0, draft.v1)],
          [Math.min(draft.u0, draft.u1), Math.max(draft.v0, draft.v1)],
        ],
      },
      // Deliberately not selected afterwards: in the draw-many flow the next class key
      // would then silently reclassify the opening just saved instead of arming the
      // next one. Reclassifying happens on a feature the operator picked on purpose.
      { onSuccess: () => { setDraft(null); setSelected(null); } },
    );
  };

  const setLifecycle = (id: string, lifecycle: Lifecycle) => edits.update.mutate({ id, lifecycle });

  /** Floors and columns from what has been drawn (§5.6). Never moves or invents an opening. */
  const runAddress = () => {
    if (!list.length || edits.address.isPending) return;
    edits.address.mutate(
      { run_id: runId },
      {
        onSuccess: (a) =>
          setAddressNote(
            a.rows.length === 0
              ? "Отворів немає"
              : `${a.rows.length} ряд(ів), ${a.column_centres.length} колонок` +
                (a.floor_height_m ? `, поверх ≈ ${a.floor_height_m.toFixed(2)} м` : "") +
                (a.ambiguous ? ` · неоднозначно: ${a.reasons.join("; ")}` : ""),
          ),
      },
    );
  };

  /**
   * Window candidates from the facade's periodic grid (gridfit). They arrive as
   * `candidate`, never verified (ADR-0009): the first one is selected so the review
   * can start straight away with V / X and the arrows.
   */
  const runGridfit = () => {
    if (!facadeKey || edits.gridfit.isPending) return;
    edits.gridfit.mutate(
      { run_id: runId },
      {
        onSuccess: (g) => {
          setGridNote(gridfitSummary(g));
          const first = g.candidates[0];
          if (first && !g.dry_run) {
            setDraft(null);
            setSelected(first.id);
          }
        },
      },
    );
  };

  const step = (delta: number) => {
    if (!list.length) return;
    const at = list.findIndex((f) => f.id === selected);
    const next = at < 0 ? (delta > 0 ? 0 : list.length - 1) : (at + delta + list.length) % list.length;
    setSelected(list[next]!.id);
  };

  // Keyboard-driven review (§10.4): the operator should not have to reach for the mouse
  // except to draw.
  const onKey = (e: KeyboardEvent) => {
    const el = e.target as HTMLElement | null;
    if (el && (el.tagName === "INPUT" || el.tagName === "SELECT" || el.tagName === "TEXTAREA")) return;
    if (e.key === "?" || (e.key === "/" && e.shiftKey)) return setHelp((h) => !h);
    if (e.key === "Escape") {
      if (help) return setHelp(false);
      if (draft) return setDraft(null);
      return setSelected(null);
    }
    if (e.key === "Enter" && draft) {
      e.preventDefault();
      return save();
    }
    if (e.key >= "1" && e.key <= "9") {
      const pick = classes[Number(e.key) - 1];
      if (!pick) return;
      setClassId(pick.id);
      if (!draft && current) edits.update.mutate({ id: current.id, class: pick.id });
      return;
    }
    if (e.key === "ArrowRight" || e.key === "j") return step(1);
    if (e.key === "ArrowLeft" || e.key === "k") return step(-1);
    if (e.key === "c" || e.key === "C") return setShowCoverage((v) => !v);
    if (e.key === "a" || e.key === "A") return void runAddress();
    if (e.key === "g" || e.key === "G") return void runGridfit();
    if (!current) return;
    if (e.key === "v" || e.key === "V") return setLifecycle(current.id, "verified");
    if (e.key === "x" || e.key === "X") return setLifecycle(current.id, "rejected");
    if (e.key === "Delete" || e.key === "Backspace") {
      e.preventDefault();
      edits.remove.mutate(current.id, { onSuccess: () => setSelected(null) });
    }
  };

  // The listener is attached once and dispatches through a ref that every render
  // refreshes. A dependency array here would have to name every value the handler
  // reads, and the one it missed -- the picked class -- silently saved openings under
  // whichever class was current when the effect last ran.
  const keyHandler = useRef(onKey);
  keyHandler.current = onKey;
  useEffect(() => {
    const listener = (e: KeyboardEvent) => keyHandler.current(e);
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, []);

  if (!facadeKey) return <div className="page faint">Фасад не вказано.</div>;
  if (facade.isLoading) return <div className="page faint">Завантаження фасаду…</div>;
  if (facade.error || !facade.data)
    return (
      <div className="page">
        <ErrorBox error={facade.error ?? "фасад не знайдено"} />
        <Link to={`/facades/${project}/${site}`}>← До фасадів</Link>
      </div>
    );

  const f = facade.data.facade;
  const tooCoarse = f.coverage && f.coverage.visible > 0 && f.coverage.covered === 0;
  const draftSize = draft
    ? { w: Math.abs(draft.u1 - draft.u0), h: Math.abs(draft.v1 - draft.v0) }
    : null;
  const draftCov = draft ? coverageUnder(coverage, draft) : null;
  const sizeOdd =
    draftSize && pickedClass?.size_prior_m
      ? draftSize.w < pickedClass.size_prior_m.width[0] ||
        draftSize.w > pickedClass.size_prior_m.width[1] ||
        draftSize.h < pickedClass.size_prior_m.height[0] ||
        draftSize.h > pickedClass.size_prior_m.height[1]
      : false;

  const rectOf = (poly: [number, number][]) => {
    const us = poly.map((p) => p[0]);
    const vs = poly.map((p) => p[1]);
    return { u0: Math.min(...us), u1: Math.max(...us), v0: Math.min(...vs), v1: Math.max(...vs) };
  };

  const svgRect = (r: Draft) => {
    if (!map) return null;
    const x = Math.min(map.xOf(r.u0), map.xOf(r.u1));
    const y = Math.min(map.yOf(r.v0), map.yOf(r.v1));
    return {
      x,
      y,
      width: Math.abs(map.xOf(r.u1) - map.xOf(r.u0)),
      height: Math.abs(map.yOf(r.v1) - map.yOf(r.v0)),
    };
  };

  return (
    <div className="page" style={{ maxWidth: "none", gap: 14 }}>
      <div className="page-head">
        <div style={{ minWidth: 0 }}>
          <div className="crumbs">
            <Link to="/">Ділянки</Link> / <Link to={`/sites/${project}/${site}`}>{project}/{site}</Link> /{" "}
            <Link to={`/facades/${project}/${site}?run=${encodeURIComponent(facade.data.run_id)}`}>Фасади</Link> / {f.id}
          </div>
          <h1>Редактор фасаду {f.id}</h1>
          <div className="sub row" style={{ gap: 14 }}>
            <span className="num">
              {(f.u_max - f.u_min).toFixed(1)} × {(f.v_max - f.v_ground).toFixed(1)} м · азимут{" "}
              {f.azimuth_deg.toFixed(0)}°
            </span>
            <span className="mono faint">run {facade.data.run_id}</span>
            <span>{list.length} точок доступу</span>
          </div>
        </div>
        <div className="row">
          <label className="field" style={{ width: 130 }}>
            <span>Деталізація</span>
            <select value={gsd} onChange={(e) => setGsd(Number(e.target.value))}>
              {GSD_CHOICES.map((g) => (
                <option key={g} value={g}>
                  {g} см/px
                </option>
              ))}
            </select>
          </label>
          <button className={`btn${showCoverage ? " primary" : ""}`} onClick={() => setShowCoverage((v) => !v)}>
            Покриття <kbd>C</kbd>
          </button>
          <button className="btn" onClick={() => setHelp((h) => !h)}>
            Клавіші <kbd>?</kbd>
          </button>
        </div>
      </div>

      {tooCoarse && (
        <div className="error-box" style={{ borderColor: "color-mix(in srgb, var(--amber) 45%, var(--border))" }}>
          ⚠️ Зйомка бачить цей фасад, але найкраще лише {f.coverage?.best_gsd_cm?.toFixed(0)} см/px — щоб читати
          отвори, потрібно ≤ {f.coverage?.max_gsd_cm.toFixed(0)}. Текстура на ортофото — здогадка 3DGS, а не
          свідчення. Малювати тут можна, але це розмітка для перевірки, а не підтверджені дані.
        </div>
      )}
      {renderError instanceof ApiError && renderError.status === 503 && (
        <div className="error-box">
          Сервіс рендерингу не працює: <code>aerosplat up</code> (або <code>aerosplat render-service</code>).
        </div>
      )}
      <ErrorBox error={renderError instanceof ApiError && renderError.status === 503 ? null : renderError} />
      <ErrorBox
        error={
          edits.create.error ?? edits.update.error ?? edits.remove.error ?? edits.address.error ??
          edits.gridfit.error ?? edits.reanchor.error
        }
      />

      <div className="render-layout">
        <div className="panel render-controls">
          <div className="panel-body" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div>
              <div className="panel-sub">Клас</div>
              <ClassList classes={classes} value={classId} onPick={setClassId} />
            </div>

            {draft && draftSize && (
              <div className="draft-box">
                <div className="row">
                  <b className="grow">{pickedClass?.labels.uk ?? classId}</b>
                  <span className="num">
                    {draftSize.w.toFixed(2)} × {draftSize.h.toFixed(2)} м
                  </span>
                </div>
                {sizeOdd && pickedClass?.size_prior_m && (
                  <small style={{ color: "var(--amber)" }}>
                    Поза типовим розміром класу ({pickedClass.size_prior_m.width.join("–")} ×{" "}
                    {pickedClass.size_prior_m.height.join("–")} м)
                  </small>
                )}
                {draftCov && draftCov.min === 0 && (
                  <small style={{ color: "var(--red)" }}>
                    Частина форми на непокритій ділянці: там текстуру домалював 3DGS.
                  </small>
                )}
                {draftCov && draftCov.min > 0 && (
                  <small className="faint">Покриття: від {draftCov.min} кадрів, у середньому {draftCov.mean.toFixed(1)}</small>
                )}
                <div className="row">
                  <button className="btn primary sm grow" onClick={save} disabled={edits.create.isPending}>
                    Зберегти <kbd>↵</kbd>
                  </button>
                  <button className="btn sm" onClick={() => setDraft(null)}>
                    <kbd>Esc</kbd>
                  </button>
                </div>
              </div>
            )}

            <div>
              <div className="row panel-sub">
                <span className="grow">Точки доступу</span>
                <button
                  className="btn sm"
                  disabled={!list.length || edits.address.isPending}
                  title="Порахувати поверхи й колонки за намальованими отворами (§5.6)"
                  onClick={runAddress}
                >
                  Адресація <kbd>A</kbd>
                </button>
              </div>
              <button
                className="btn sm"
                style={{ width: "100%", marginBottom: 8 }}
                disabled={edits.gridfit.isPending}
                title="Знайти повторювану сітку поверхів і колонок на ортофото і запропонувати її клітинки як кандидати (не підтверджені). Фасад без чіткої сітки буде відхилено з причиною."
                onClick={runGridfit}
              >
                {edits.gridfit.isPending ? "Шукаю вікна…" : "Знайти вікна за сіткою"} <kbd>G</kbd>
              </button>
              {gridNote && (
                <div className="draft-box" style={{ marginBottom: 8 }}>
                  <small className={gridNote.warn ? "" : "faint"} style={gridNote.warn ? { color: "var(--amber)" } : undefined}>
                    {gridNote.text}
                  </small>
                </div>
              )}
              {addressNote && (
                <div className="draft-box" style={{ marginBottom: 8 }}>
                  <small className={addressNote.includes("неоднозначно") ? "" : "faint"}
                         style={addressNote.includes("неоднозначно") ? { color: "var(--amber)" } : undefined}>
                    {addressNote}
                  </small>
                </div>
              )}
              <div className="ap-list">
                {list.map((ap) => (
                  <ApRow
                    key={ap.id}
                    ap={ap}
                    classes={classes}
                    on={ap.id === selected}
                    onSelect={() => setSelected(ap.id)}
                  />
                ))}
                {!list.length && <div className="faint">Ще нічого не намальовано. Протягни рамку по ортофото.</div>}
              </div>
            </div>

            {foreign.length > 0 && (
              <div>
                <div className="panel-sub" title="Точки, намальовані на цій самій стіні в іншому run-і (там у фасаду інший id). Пунктир — їхнє місце тут. Самі вони не переносяться.">
                  З інших run-ів ({foreign.length})
                </div>
                <div className="ap-list">
                  {foreign.map((c) => (
                    <CarriedRow
                      key={c.feature.id}
                      c={c}
                      classes={classes}
                      on={c.feature.id === selected}
                      onSelect={() => setSelected(c.feature.id)}
                    />
                  ))}
                </div>
              </div>
            )}

            {currentForeign && (
              <div className="draft-box">
                <div className="row">
                  <b className="grow">
                    {classes.find((x) => x.id === currentForeign.feature.class)?.labels.uk ?? currentForeign.feature.class}
                  </b>
                  <span className="faint">{LIFECYCLE_LABEL[currentForeign.feature.lifecycle]}</span>
                </div>
                <small className="faint">
                  Намальовано на {currentForeign.feature.host?.facade_id}
                  {currentForeign.feature.provenance.run_id && ` (run ${currentForeign.feature.provenance.run_id})`}; тут —
                  зсув площини {currentForeign.here.offset_m.toFixed(2)} м, кут {currentForeign.here.angle_deg.toFixed(1)}°
                </small>
                {currentForeign.placement.status === "placed" ? (
                  <>
                    <button
                      className="btn sm"
                      disabled={edits.reanchor.isPending || !facadeRun}
                      onClick={() =>
                        edits.reanchor.mutate(
                          { id: currentForeign.feature.id, facade_id: facadeKey, run_id: facadeRun },
                          { onSuccess: () => setSelected(currentForeign.feature.id) },
                        )
                      }
                    >
                      Перенести на цей фасад
                    </button>
                    <small className="faint">
                      Висота зберігається, форма лягає на площину цього run-а; стан і історія лишаються, адресу
                      треба порахувати знову (<kbd>A</kbd>). Ціль на цій точці доведеться підтвердити ще раз.
                    </small>
                  </>
                ) : (
                  <small style={{ color: "var(--amber)" }}>Не переноситься: {currentForeign.placement.reason}</small>
                )}
              </div>
            )}

            {current && (
              <div className="draft-box">
                <div className="row">
                  <b className="grow">{classes.find((c) => c.id === current.class)?.labels.uk ?? current.class}</b>
                  <span className="num faint">
                    {current.geometry.width.toFixed(2)} × {current.geometry.height.toFixed(2)} м
                  </span>
                </div>
                {current.address && (
                  <small className="faint num">
                    поверх {current.address.floor_label ?? "?"} · колонка {current.address.column ?? "?"} · {current.address.ordinal_on_floor ?? "?"}-й на поверсі
                    {current.address.floor_ambiguous && <span style={{ color: "var(--amber)" }}> · неоднозначно</span>}
                  </small>
                )}
                <div className="row">
                  <button className="btn sm grow" onClick={() => setLifecycle(current.id, "verified")}>
                    Підтвердити <kbd>V</kbd>
                  </button>
                  <button className="btn sm" onClick={() => setLifecycle(current.id, "rejected")}>
                    Відхилити <kbd>X</kbd>
                  </button>
                  <button
                    className="btn danger sm"
                    onClick={() => edits.remove.mutate(current.id, { onSuccess: () => setSelected(null) })}
                  >
                    <Icon name="close" size={13} />
                  </button>
                </div>
                <small className="faint">
                  Відхилення не видаляє: відхилена точка лишається як негативний приклад для навчання (§3.4).
                </small>
              </div>
            )}

            <div className="cov-legend">
              <span>
                <i style={{ background: "rgba(255, 70, 70, 0.42)" }} /> не роздільно
              </span>
              <span>
                <i style={{ background: "rgba(255, 184, 77, 0.22)" }} /> 1–2 кадри
              </span>
              <span>
                <i style={{ background: "transparent", borderColor: "var(--border-strong)" }} /> видно добре
              </span>
            </div>
          </div>
        </div>

        <div className="render-stage">
          {shot ? (
            <div
              className="render-img facade-img"
              ref={stageRef}
              // Capped by WIDTH, from the image's own aspect, so a tall facade still
              // fits the viewport. Capping the height instead would letterbox the
              // image inside this box and put every overlay rectangle off the wall.
              style={{ maxWidth: `calc((100vh - 240px) * ${shot.meta.width / shot.meta.height})` }}
              onMouseDown={(e) => {
                if (e.button !== 0) return;
                const at = uvAt(e);
                if (!at) return;
                e.preventDefault();
                setDraft(null);
                dragRef.current = { u0: at.u, v0: at.v, u1: at.u, v1: at.v };
                setDragging(dragRef.current);
              }}
              onMouseMove={(e) => {
                const at = uvAt(e);
                setHover(at);
                if (!at || !dragRef.current) return;
                dragRef.current = { ...dragRef.current, u1: at.u, v1: at.v };
                setDragging(dragRef.current);
              }}
              onMouseUp={() => {
                const r = dragRef.current;
                dragRef.current = null;
                setDragging(null);
                // A click rather than a drag: that deselects, it is not a 0 x 0 shape.
                if (r && Math.abs(r.u1 - r.u0) > 0.1 && Math.abs(r.v1 - r.v0) > 0.1) setDraft(r);
              }}
              onMouseLeave={() => {
                setHover(null);
                dragRef.current = null;
                setDragging(null);
              }}
            >
              <img src={shot.url} alt={`Ортофото фасаду ${f.id}`} draggable={false} />
              {showCoverage && coverage && covBox && <CoverageWash grid={coverage} box={covBox} />}
              <svg
                className="facade-overlay"
                viewBox={`0 0 ${shot.meta.width} ${shot.meta.height}`}
                preserveAspectRatio="none"
              >
                {list.map((ap) => {
                  const r = svgRect(rectOf(ap.geometry.polygon_uv));
                  if (!r) return null;
                  return (
                    <rect
                      key={ap.id}
                      {...r}
                      className={`ap${ap.id === selected ? " on" : ""}`}
                      stroke={LIFECYCLE_TONE[ap.lifecycle]}
                      strokeWidth={ap.id === selected ? 3 : 1.5}
                      vectorEffect="non-scaling-stroke"
                      onMouseDown={(e) => {
                        e.stopPropagation();
                        setSelected(ap.id);
                        setDraft(null);
                      }}
                    />
                  );
                })}
                {foreign.map((c) => {
                  const r = svgRect(rectOf(c.here.polygon_uv));
                  if (!r) return null;
                  return (
                    <rect
                      key={`carried-${c.feature.id}`}
                      {...r}
                      className={`ap carried${c.feature.id === selected ? " on" : ""}`}
                      stroke={LIFECYCLE_TONE[c.feature.lifecycle]}
                      strokeWidth={c.feature.id === selected ? 2.5 : 1.2}
                      vectorEffect="non-scaling-stroke"
                      onMouseDown={(e) => {
                        e.stopPropagation();
                        setSelected(c.feature.id);
                        setDraft(null);
                      }}
                    />
                  );
                })}
                {(() => {
                  const live = dragging ?? draft;
                  const r = live && svgRect(live);
                  return r ? <rect {...r} className="ap draft" vectorEffect="non-scaling-stroke" /> : null;
                })()}
              </svg>
              {busy && <div className="render-busy">рендер…</div>}
              {hover && (
                <div className="render-readout num">
                  u {hover.u.toFixed(2)} · v {hover.v.toFixed(2)} м
                  {coverage &&
                    (() => {
                      const c = coverageUnder(coverage, { u0: hover.u, v0: hover.v, u1: hover.u, v1: hover.v });
                      return c ? ` · ${c.min} кадр.` : "";
                    })()}
                </div>
              )}
            </div>
          ) : (
            <div className="faint">{busy ? "Рендер ортофото…" : "Немає зображення."}</div>
          )}
          {shot && (
            <div className="row faint" style={{ fontSize: 12, flexWrap: "wrap", gap: 14 }}>
              <span className="num">
                {shot.meta.width} × {shot.meta.height} px · {(100 * shot.meta.facade_grid.m_per_px).toFixed(1)} см/px
              </span>
              <span className="num">
                u {shot.meta.facade_grid.u_min.toFixed(1)}…{shot.meta.facade_grid.u_max.toFixed(1)} · v{" "}
                {shot.meta.facade_grid.v_min.toFixed(1)}…{shot.meta.facade_grid.v_max.toFixed(1)} м
              </span>
              <span title="u зростає праворуч, якщо стояти зовні обличчям до стіни — так само лічаться колонки (§4)">
                → u праворуч (вид зовні)
              </span>
              {shot.meta.render_ms != null && <span className="num">{shot.meta.render_ms} мс</span>}
            </div>
          )}
        </div>
      </div>

      {help && (
        <div className="panel">
          <div className="panel-head">
            <b>Клавіші</b>
            <button className="btn ghost sm" onClick={() => setHelp(false)}>
              <Icon name="close" />
            </button>
          </div>
          <div className="panel-body kbd-help">
            <div><kbd>протягнути</kbd> намалювати рамку</div>
            <div><kbd>↵</kbd> зберегти рамку</div>
            <div><kbd>Esc</kbd> скасувати / зняти вибір</div>
            <div><kbd>1</kbd>…<kbd>9</kbd> клас (і зміна класу вибраної точки)</div>
            <div><kbd>←</kbd> <kbd>→</kbd> або <kbd>k</kbd> <kbd>j</kbd> наступна точка</div>
            <div><kbd>V</kbd> підтвердити · <kbd>X</kbd> відхилити</div>
            <div><kbd>Del</kbd> видалити</div>
            <div><kbd>C</kbd> карта покриття</div>
            <div><kbd>A</kbd> адресація (поверхи й колонки)</div>
            <div><kbd>G</kbd> знайти вікна за сіткою (кандидати на перевірку)</div>
          </div>
        </div>
      )}
    </div>
  );
}

function CarriedRow({
  c,
  classes,
  on,
  onSelect,
}: {
  c: CarriedOut;
  classes: TaxonomyClass[];
  on: boolean;
  onSelect: () => void;
}) {
  const label = classes.find((x) => x.id === c.feature.class)?.labels.uk ?? c.feature.class;
  const ok = c.placement.status === "placed";
  return (
    <button className={`ap-row${on ? " on" : ""}`} onClick={onSelect} title={c.placement.reason}>
      <span className="dot" style={{ background: "transparent", border: `1.5px dashed ${LIFECYCLE_TONE[c.feature.lifecycle]}` }} />
      <span className="grow">{label}</span>
      <small className="num faint">{c.feature.host?.facade_id}</small>
      <small style={ok ? undefined : { color: "var(--amber)" }} className={ok ? "faint" : ""}>
        {ok ? "можна" : "неоднозн."}
      </small>
    </button>
  );
}

function ApRow({
  ap,
  classes,
  on,
  onSelect,
}: {
  ap: AccessPoint;
  classes: TaxonomyClass[];
  on: boolean;
  onSelect: () => void;
}) {
  const label = classes.find((c) => c.id === ap.class)?.labels.uk ?? ap.class;
  return (
    <button className={`ap-row${on ? " on" : ""}`} onClick={onSelect}>
      <span className="dot" style={{ background: LIFECYCLE_TONE[ap.lifecycle] }} />
      <span className="grow">{label}</span>
      {ap.address?.floor_label && (
        <small className="num faint">
          {ap.address.floor_label}·{ap.address.column}
        </small>
      )}
      <small className="num faint">
        {ap.geometry.width.toFixed(1)}×{ap.geometry.height.toFixed(1)}
      </small>
      <small className="faint" title={LIFECYCLE_LABEL[ap.lifecycle]}>
        {LIFECYCLE_LABEL[ap.lifecycle].slice(0, 4)}
      </small>
    </button>
  );
}
