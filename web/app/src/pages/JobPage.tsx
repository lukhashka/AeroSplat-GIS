import { useMemo } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { useCancelJob, usePipeline, useRunDetail } from "../api/hooks";
import { useJobStream } from "../api/stream";
import type { Job, PipelineInfo } from "../api/types";
import { Icon } from "../components/Icon";
import { LogView } from "../components/LogView";
import { ErrorBox } from "../components/Modal";
import { StageTimeline, type PlannedStage } from "../components/StageTimeline";
import { JobBadge } from "../components/StatusBadge";
import { between, dateTime, duration } from "../lib/format";
import { useNow } from "../lib/useNow";

const ALL_STAGES = ["geo", "exif", "sfm", "dense", "parse", "train", "eval", "georef"];
// aerosplat.export.run.KINDS: the order an export job reports its kinds in.
const EXPORT_ORDER = ["ortho", "dsm", "dtm", "ndsm", "las", "ply"];

/** Stages the job will go through, from its payload (pipeline) or spec (diagnostic). */
function plannedStages(job: Job | null, pipeline: PipelineInfo | undefined): PlannedStage[] {
  if (!job) return [];
  const label = (name: string) => pipeline?.stages.find((s) => s.name === name)?.label ?? name;
  if (job.kind === "diagnostic") {
    const n = Number(job.payload["stages"] ?? 0);
    return Array.from({ length: n }, (_, i) => ({ name: `step${i + 1}`, label: `Diagnostic step ${i + 1}/${n}` }));
  }
  if (job.kind === "export") {
    const kinds = (job.payload["kinds"] as string[] | undefined) ?? [];
    return EXPORT_ORDER.filter((k) => kinds.includes(k)).map((k) => ({ name: k, label: `Експорт: ${k}` }));
  }
  const p = job.payload as { stages?: string[] | null; from_stage?: string | null; to_stage?: string | null };
  let names = ALL_STAGES;
  if (p.stages?.length) names = p.stages;
  else {
    const a = p.from_stage ? ALL_STAGES.indexOf(p.from_stage) : 0;
    const b = p.to_stage ? ALL_STAGES.indexOf(p.to_stage) + 1 : ALL_STAGES.length;
    names = ALL_STAGES.slice(a, b);
  }
  return names.map((name) => ({ name, label: label(name) }));
}

export function JobPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const stream = useJobStream(id);
  const { data: pipeline } = usePipeline();
  const cancel = useCancelJob();
  const job = stream.job;
  const active = !!job && (job.status === "queued" || job.status === "running");
  const now = useNow(active);

  const planned = useMemo(() => plannedStages(job, pipeline), [job, pipeline]);
  // The run this job produced; only a run with splat/cloud products has a 3D scene.
  const res = (job?.result ?? {}) as { site?: string; epoch?: string; run_id?: string };
  const [resProject = "", resSite = ""] = (res.site ?? "").split("/");
  const { data: runDetail } = useRunDetail(
    resProject,
    resSite,
    res.epoch ?? "baseline",
    job?.status === "succeeded" && res.run_id ? res.run_id : null,
  );
  const finished = stream.stages.filter((s) => s.state === "done" || s.state === "skipped").length;
  const total = planned.length || stream.stages[0]?.total || 0;
  const fraction =
    job?.status === "succeeded" ? 1 : total ? (finished + (stream.stages.some((s) => s.state === "started") ? 0.5 : 0)) / total : 0;

  if (stream.error) {
    return (
      <div className="page">
        <ErrorBox error={stream.error} />
        <Link to="/jobs">← До задач</Link>
      </div>
    );
  }
  if (!job) return <div className="page faint">Підключення до задачі…</div>;

  const result = job.result as { site?: string; epoch?: string; run_id?: string };
  const hasScene = job.status === "succeeded" && !!runDetail?.scene;
  const elapsed = between(job.started_at, job.finished_at ?? new Date(now).toISOString());
  const overrides = (job.payload["overrides"] ?? {}) as Record<string, unknown>;

  return (
    <div className="page">
      <div className="page-head">
        <div style={{ minWidth: 0 }}>
          <div className="crumbs">
            <Link to="/jobs">Задачі</Link> / <span className="mono">{job.id.slice(0, 12)}</span>
          </div>
          <h1>{job.title || job.id}</h1>
          <div className="sub row" style={{ gap: 12 }}>
            <JobBadge status={job.status} cancelling={job.cancel_requested} />
            <span>
              {job.kind} · {job.resource.toUpperCase()}
            </span>
            {job.started_at && <span className="num">⏱ {duration(elapsed)}</span>}
            {!stream.connected && active && <span className="badge mute">перепідключення…</span>}
          </div>
        </div>
        <div className="row">
          {result.site && (
            <Link className="btn" to={`/sites/${result.site}`}>
              <Icon name="map" /> Ділянка
            </Link>
          )}
          {hasScene && result.site && (
            <button className="btn primary" onClick={() => navigate(`/viewer/${result.site}`)}>
              <Icon name="cube" /> Відкрити 3D
            </button>
          )}
          {active && (
            <button className="btn danger" disabled={job.cancel_requested || cancel.isPending} onClick={() => cancel.mutate(job.id)}>
              <Icon name="stop" /> {job.cancel_requested ? "Скасовується…" : "Скасувати"}
            </button>
          )}
        </div>
      </div>

      <div className={`progress${job.status === "succeeded" ? " done" : job.status === "failed" ? " bad" : ""}`}>
        <i style={{ width: `${Math.round(fraction * 100)}%` }} />
      </div>

      {job.status === "failed" && job.error && <div className="error-box mono">{job.error}</div>}
      {job.status === "queued" && (
        <div className="note">
          Задача чекає на {job.resource.toUpperCase()}-воркер. Якщо він не запущений: <code>aerosplat up</code> або{" "}
          <code>aerosplat worker --resource {job.resource}</code>.
        </div>
      )}

      <div className="grid-2">
        <div className="panel">
          <div className="panel-head">
            <h2>Журнал</h2>
            <span className="faint num" style={{ fontSize: 12 }}>
              {(stream.lines.length + stream.droppedLines).toLocaleString("uk-UA")} рядків
            </span>
          </div>
          <LogView lines={stream.lines} dropped={stream.droppedLines} />
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <div className="panel">
            <div className="panel-head">
              <h2>Стадії</h2>
              <span className="faint num" style={{ fontSize: 12 }}>
                {finished}/{total}
              </span>
            </div>
            <StageTimeline planned={planned} live={stream.stages} now={now} jobActive={active} />
          </div>
          <div className="panel">
            <div className="panel-head">
              <h2>Деталі</h2>
            </div>
            <div className="panel-body">
              <dl className="kv">
                <dt>Створено</dt>
                <dd>{dateTime(job.created_at)}</dd>
                <dt>Почато</dt>
                <dd>{dateTime(job.started_at)}</dd>
                <dt>Завершено</dt>
                <dd>{dateTime(job.finished_at)}</dd>
                <dt>Воркер</dt>
                <dd className="mono">{job.worker_id ?? "—"}</dd>
                {result.run_id && (
                  <>
                    <dt>Run</dt>
                    <dd className="mono">
                      {result.epoch}/{result.run_id}
                    </dd>
                  </>
                )}
                {typeof job.payload["capture"] === "string" && (
                  <>
                    <dt>Захоплення</dt>
                    <dd className="mono">{String(job.payload["capture"])}</dd>
                  </>
                )}
                {Object.keys(overrides).length > 0 && (
                  <>
                    <dt>Параметри</dt>
                    <dd className="mono">
                      {Object.entries(overrides)
                        .map(([k, v]) => `${k}=${v}`)
                        .join("  ")}
                    </dd>
                  </>
                )}
              </dl>
            </div>
          </div>
        </div>
      </div>
      <ErrorBox error={cancel.error} />
    </div>
  );
}
