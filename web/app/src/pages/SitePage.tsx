import { useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { thumbUrl } from "../api/client";
import { useCaptureImages, useRunDetail, useRunExports, useSite, useStartExport } from "../api/hooks";
import type { ExportKind, Run } from "../api/types";
import { Icon } from "../components/Icon";
import { JobsTable } from "../components/JobsTable";
import { ErrorBox } from "../components/Modal";
import { RunDialog } from "../components/RunDialog";
import { SiteMap } from "../components/SiteMap";
import { KIND_LABEL, RunBadge, SOURCE_LABEL } from "../components/StatusBadge";
import { UploadDialog } from "../components/UploadDialog";
import { between, bytes, coord, dateTime, duration, timeAgo } from "../lib/format";

const THUMBS = 24;

export function SitePage() {
  const { project = "", site = "" } = useParams();
  const navigate = useNavigate();
  const { data, error, isLoading } = useSite(project, site);
  const [captureSel, setCaptureSel] = useState<string | null>(null);
  const [runSel, setRunSel] = useState<Run | null>(null);
  const [dialog, setDialog] = useState<"upload" | "run" | null>(null);

  const capture = captureSel ?? data?.captures[0]?.capture.id ?? null;
  const { data: images = [] } = useCaptureImages(project, site, capture);
  const priors = useMemo(
    () => images.filter((i) => i.prior).map((i) => ({ lat: i.prior!.lat, lon: i.prior!.lon })),
    [images],
  );

  if (isLoading) return <div className="page faint">Завантаження…</div>;
  if (error || !data)
    return (
      <div className="page">
        <ErrorBox error={error ?? "ділянку не знайдено"} />
        <Link to="/">← До ділянок</Link>
      </div>
    );

  const s = data.site;
  const active = data.jobs.filter((j) => j.status === "queued" || j.status === "running");

  return (
    <div className="page">
      <div className="page-head">
        <div style={{ minWidth: 0 }}>
          <div className="crumbs">
            <Link to="/">Ділянки</Link> / {s.project_id}
          </div>
          <h1>{s.name || s.id}</h1>
          <div className="sub row" style={{ gap: 14 }}>
            <span className="mono">
              {s.project_id}/{s.id}
            </span>
            <span>{KIND_LABEL[s.kind]}</span>
            <span title="ENU-початок ділянки, WGS84 (фіксований)">
              <Icon name="pin" size={13} style={{ verticalAlign: -2 }} /> {coord(s.origin.lat, s.origin.lon)} · h {s.origin.h.toFixed(1)} м
            </span>
          </div>
        </div>
        <div className="row">
          <button className="btn" onClick={() => setDialog("upload")}>
            <Icon name="upload" /> Завантажити кадри
          </button>
          <button className="btn" disabled={!data.captures.length} onClick={() => setDialog("run")}>
            <Icon name="play" /> Запустити run
          </button>
          <button className="btn" disabled={!data.scene} onClick={() => navigate(`/render/${s.project_id}/${s.id}`)}>
            <Icon name="camera" /> Вид із камери
          </button>
          <button
            className="btn"
            disabled={!data.scene}
            title="Фасади, ортофото й точки доступу (вікна, двері)"
            onClick={() => navigate(`/facades/${s.project_id}/${s.id}`)}
          >
            <Icon name="map" /> Фасади
          </button>
          <button
            className="btn"
            title="Місії до точок доступу і їхні прольоти в SITL: 3D над сценою, вид згори, вердикти"
            onClick={() => navigate(`/flight/${s.project_id}/${s.id}`)}
          >
            <Icon name="pulse" /> Місія / проліт
          </button>
          <button className="btn primary" disabled={!data.scene} onClick={() => navigate(`/viewer/${s.project_id}/${s.id}`)}>
            <Icon name="cube" /> Відкрити 3D
          </button>
        </div>
      </div>

      {active.length > 0 && (
        <div className="panel" style={{ borderColor: "color-mix(in srgb, var(--amber) 40%, var(--border))" }}>
          <div className="panel-body row">
            <span className="badge run">
              <span className="dot" />
              {active.length === 1 ? "Задача виконується" : `${active.length} активних задач`}
            </span>
            <span className="grow muted">{active[0]!.title}</span>
            <Link className="btn sm" to={`/jobs/${active[0]!.id}`}>
              Монітор →
            </Link>
          </div>
        </div>
      )}

      <div className="grid-2">
        <SiteMap
          tall
          origin={{ lat: s.origin.lat, lon: s.origin.lon }}
          points={priors}
          hint={priors.length ? `${priors.length} позицій камер із «${capture}»` : undefined}
        />
        <div className="panel" style={{ display: "flex", flexDirection: "column" }}>
          <div className="panel-head">
            <h2>Захоплення</h2>
            <button className="btn sm" onClick={() => setDialog("upload")}>
              <Icon name="plus" size={14} /> Додати
            </button>
          </div>
          {!data.captures.length ? (
            <div className="empty">
              <div className="big">Ще немає кадрів</div>
              Завантаж експорт Earth Studio, фото з дрона, ручну зйомку чи фото з інтернету.
            </div>
          ) : (
            <>
              <table className="list">
                <tbody>
                  {data.captures.map((c) => (
                    <tr
                      key={c.capture.id}
                      className="click"
                      onClick={() => setCaptureSel(c.capture.id)}
                      style={c.capture.id === capture ? { boxShadow: "inset 3px 0 0 var(--accent)" } : undefined}
                    >
                      <td>
                        <div style={{ fontWeight: 600 }}>{c.capture.id}</div>
                        <div className="faint" style={{ fontSize: 12 }}>
                          {SOURCE_LABEL[c.capture.source]} · {c.capture.license || "ліцензію не вказано"}
                        </div>
                      </td>
                      <td className="num muted" style={{ textAlign: "right", whiteSpace: "nowrap" }}>
                        {c.images} кадрів
                        <div className="faint" style={{ fontSize: 12 }}>
                          {bytes(c.bytes)}
                          {c.with_prior ? ` · ${c.with_prior} з GPS` : ""}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {capture && images.length > 0 && (
                <div className="panel-body" style={{ borderTop: "1px solid var(--border)" }}>
                  <div className="thumbs">
                    {images.slice(0, THUMBS).map((im) => (
                      <img key={im.name} loading="lazy" alt={im.name} title={im.name} src={thumbUrl(project, site, capture, im.name, 200)} />
                    ))}
                  </div>
                  {images.length > THUMBS && (
                    <div className="faint" style={{ fontSize: 12, marginTop: 8 }}>
                      і ще {images.length - THUMBS}
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">
          <h2>Реконструкції</h2>
          <span className="faint" style={{ fontSize: 12 }}>
            епохи: {data.epochs.map((e) => e.label || e.id).join(", ") || "—"}
          </span>
        </div>
        {!data.runs.length ? (
          <div className="empty">Run-ів ще не було. Завантаж кадри й натисни «Запустити run».</div>
        ) : (
          <div className="table-wrap">
            <table className="list">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Статус</th>
                  <th>Система координат</th>
                  <th>Продукти</th>
                  <th>Коли</th>
                  <th>Тривалість</th>
                </tr>
              </thead>
              <tbody>
                {data.runs.map((r) => (
                  <tr
                    key={r.epoch_id + r.id}
                    className="click"
                    onClick={() => setRunSel(runSel?.id === r.id ? null : r)}
                    style={runSel?.id === r.id ? { boxShadow: "inset 3px 0 0 var(--accent)" } : undefined}
                  >
                    <td>
                      <div className="mono nowrap" style={{ fontWeight: 600 }}>
                        {r.id}
                      </div>
                      <div className="faint" style={{ fontSize: 12 }}>
                        {r.epoch_id} · {r.capture_ids.join(", ")}
                      </div>
                    </td>
                    <td className="nowrap">
                      <RunBadge status={r.status} />
                    </td>
                    <td className="nowrap">
                      {r.georeferenced ? <span className="badge accent">метрична</span> : <span className="badge mute">відносна</span>}
                      <div className="faint mono" style={{ fontSize: 11.5, marginTop: 3 }}>
                        {r.earth_model}
                      </div>
                    </td>
                    <td className="muted" style={{ fontSize: 12.5 }}>
                      {Object.keys(r.products).length ? Object.keys(r.products).sort().join(", ") : "—"}
                    </td>
                    <td className="muted nowrap" title={dateTime(r.created_at)}>
                      {timeAgo(r.created_at)}
                    </td>
                    <td className="num muted nowrap">{duration(between(r.created_at, r.finished_at))}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {runSel && <RunDetailPanel project={project} site={site} run={runSel} />}
      </div>

      <div className="panel">
        <div className="panel-head">
          <h2>Задачі ділянки</h2>
        </div>
        <div className="panel-body flush">
          <JobsTable jobs={data.jobs} />
        </div>
      </div>

      {dialog === "upload" && (
        <UploadDialog project={project} site={site} existing={data.captures.map((c) => c.capture.id)} onClose={() => setDialog(null)} />
      )}
      {dialog === "run" && (
        <RunDialog project={project} site={site} captures={data.captures} epochs={data.epochs} onClose={() => setDialog(null)} />
      )}
    </div>
  );
}

function RunDetailPanel({ project, site, run }: { project: string; site: string; run: Run }) {
  const navigate = useNavigate();
  const { data, error } = useRunDetail(project, site, run.epoch_id, run.id);
  // metrics.json (aerosplat.gs.evaluate): top-level numbers are the summary.
  const numbers = Object.entries(data?.metrics ?? {}).filter(
    (e): e is [string, number] => typeof e[1] === "number" && Number.isFinite(e[1]),
  );
  return (
    <div className="panel-body" style={{ borderTop: "1px solid var(--border)", background: "var(--panel-2)" }}>
      <ErrorBox error={error} />
      {data && (
        <div className="grid-2">
          <div>
            <h3 style={{ marginBottom: 8 }}>Стадії (manifest.json)</h3>
            {data.stages.length ? (
              <table className="list" style={{ background: "var(--panel)", borderRadius: 8 }}>
                <tbody>
                  {data.stages.map((st) => (
                    <tr key={st.stage}>
                      <td className="mono">{st.stage}</td>
                      <td>{st.status === "ok" ? <span className="badge ok">ok</span> : <span className="badge bad">{st.status}</span>}</td>
                      <td className="num muted">{duration(st.seconds)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="faint">Маніфест ще порожній.</div>
            )}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <h3>Якість і продукти</h3>
            <dl className="kv">
              {numbers.map(([k, v]) => (
                <FragmentKV key={k} k={METRIC_LABEL[k] ?? k} v={formatMetric(k, v)} />
              ))}
              <FragmentKV k="Earth model" v={run.earth_model} />
              <FragmentKV k="Commit" v={run.git_commit?.slice(0, 10) ?? "—"} />
              <FragmentKV k="Захоплення" v={run.capture_ids.join(", ")} />
            </dl>
            <div className="row">
              {data.scene && (
                <button className="btn primary sm" onClick={() => navigate(`/viewer/${project}/${site}`)}>
                  <Icon name="cube" size={14} /> Відкрити 3D
                </button>
              )}
              {data.has_report && (
                <a className="btn sm" href={`/api/v2/sites/${project}/${site}/runs/${run.epoch_id}/${run.id}/report`} target="_blank" rel="noreferrer">
                  <Icon name="report" size={14} /> Звіт якості
                </a>
              )}
            </div>
          </div>
        </div>
      )}
      {data && run.status === "ok" && <ExportPanel project={project} site={site} run={run} />}
    </div>
  );
}

const EXPORT_KINDS: { kind: ExportKind; label: string; title: string }[] = [
  { kind: "ortho", label: "Ортофото", title: "GeoTIFF/COG, рендер 3DGS згори (GPU)" },
  { kind: "dsm", label: "DSM", title: "GeoTIFF, найвища точка клітини (зі стадії semantics)" },
  { kind: "dtm", label: "DTM", title: "GeoTIFF, оцінка голої землі" },
  { kind: "ndsm", label: "nDSM", title: "GeoTIFF, висота над DTM" },
  { kind: "las", label: "LAZ", title: "Хмара точок run-а, LAS 1.4 з CRS" },
  { kind: "ply", label: "3DGS .ply", title: "INRIA-формат (SuperSplat, gsplat, Postshot), Site ENU" },
];

/** GIS exports of a finished run: what exists (with downloads) and a form to make more. */
function ExportPanel({ project, site, run }: { project: string; site: string; run: Run }) {
  const navigate = useNavigate();
  const { data } = useRunExports(project, site, run.epoch_id, run.id);
  const start = useStartExport(project, site, run.epoch_id, run.id);
  const [kinds, setKinds] = useState<ExportKind[]>(["dsm", "dtm", "ndsm", "las", "ply"]);
  const [gsd, setGsd] = useState(0.25);
  if (!data) return null;
  const geo = data.georeference;
  const toggle = (k: ExportKind) => setKinds((ks) => (ks.includes(k) ? ks.filter((x) => x !== k) : [...ks, k]));
  return (
    <div style={{ marginTop: 16 }}>
      <h3 style={{ marginBottom: 8 }}>Експорт GIS</h3>
      {!data.exportable ? (
        <div className="faint">{data.refusal}</div>
      ) : (
        <>
          <div className="row" style={{ flexWrap: "wrap", gap: 12 }}>
            {EXPORT_KINDS.map((k) => (
              <label key={k.kind} title={k.title} className="row" style={{ gap: 4 }}>
                <input type="checkbox" checked={kinds.includes(k.kind)} onChange={() => toggle(k.kind)} />
                {k.label}
              </label>
            ))}
            {kinds.includes("ortho") && (
              <label className="row" style={{ gap: 4 }} title="Розмір пікселя ортофото, м (UTM)">
                GSD, м
                <input type="number" min={0.01} max={10} step={0.05} value={gsd} style={{ width: 70 }}
                  onChange={(e) => setGsd(Number(e.target.value))} />
              </label>
            )}
            <button className="btn primary sm" disabled={!kinds.length || start.isPending}
              onClick={() => start.mutate({ kinds, gsd_m: gsd }, { onSuccess: (job) => navigate(`/jobs/${job.id}`) })}>
              <Icon name="download" size={14} /> Експортувати
            </button>
          </div>
          <ErrorBox error={start.error} />
        </>
      )}
      {data.files.length > 0 && (
        <div className="row" style={{ flexWrap: "wrap", gap: 8, marginTop: 10 }}>
          {data.files.map((f) => (
            <a key={f.kind} className="btn sm" href={f.url} title={`${f.crs?.name ?? ""} · ${dateTime(f.created ?? "")}`}>
              <Icon name="download" size={14} /> {f.file} <span className="muted">{bytes(f.bytes)}</span>
            </a>
          ))}
        </div>
      )}
      {geo && data.files.length > 0 && (
        <div className="faint" style={{ marginTop: 8 }}>
          CRS — {data.files[0]?.crs?.name ?? "?"}; висоти еліпсоїдні WGS84 ({geo.altitude_reference}). Похибка кадру:{" "}
          {geo.frame_error_m == null ? "невідома" : `${(geo.frame_error_m * 1000).toFixed(2)} мм`} ({geo.earth_model}); абсолютна
          точність джерела в бюджет не входить.
          {geo.warnings.map((w) => (
            <div key={w}>⚠️ {w}</div>
          ))}
        </div>
      )}
    </div>
  );
}

const METRIC_LABEL: Record<string, string> = {
  mean_psnr: "PSNR (test)",
  mean_ssim: "SSIM (test)",
  mean_lpips: "LPIPS (test)",
};

function formatMetric(key: string, v: number): string {
  if (key.includes("psnr")) return `${v.toFixed(2)} dB`;
  if (key.includes("ssim") || key.includes("lpips")) return v.toFixed(4);
  return Number.isInteger(v) ? v.toLocaleString("uk-UA") : v.toFixed(2);
}

function FragmentKV({ k, v }: { k: string; v: string }) {
  return (
    <>
      <dt>{k}</dt>
      <dd className="mono">{v}</dd>
    </>
  );
}
