import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { useJobs, useSites, useSystem } from "../api/hooks";
import { CreateSiteDialog } from "../components/CreateSiteDialog";
import { Icon } from "../components/Icon";
import { JobsTable } from "../components/JobsTable";
import { SiteMap } from "../components/SiteMap";
import { KIND_LABEL, RunBadge } from "../components/StatusBadge";
import { timeAgo } from "../lib/format";

export function Dashboard() {
  const navigate = useNavigate();
  const { data: sites, isLoading } = useSites();
  const { data: sys } = useSystem();
  const { data: jobs = [] } = useJobs({ limit: 6 });
  const [creating, setCreating] = useState(false);

  const markers = useMemo(
    () =>
      (sites ?? []).map((s) => ({
        key: s.site.project_id + "/" + s.site.id,
        name: s.site.name || s.site.id,
        lat: s.site.origin.lat,
        lon: s.site.origin.lon,
        busy: s.active_jobs > 0,
      })),
    [sites],
  );

  const totals = useMemo(() => {
    const list = sites ?? [];
    return {
      sites: list.length,
      projects: new Set(list.map((s) => s.site.project_id)).size,
      images: list.reduce((a, s) => a + s.images, 0),
      captures: list.reduce((a, s) => a + s.captures, 0),
      runs: list.reduce((a, s) => a + s.runs, 0),
      scenes: list.filter((s) => s.scene).length,
    };
  }, [sites]);

  const gpuOnline = sys?.workers.some((w) => w.resource === "gpu" && w.online);

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Ділянки</h1>
          <div className="sub">Території з власною метричною системою координат, їхні зйомки та реконструкції.</div>
        </div>
        <button className="btn primary" onClick={() => setCreating(true)}>
          <Icon name="plus" /> Нова ділянка
        </button>
      </div>

      <div className="stats">
        <div className="stat">
          <div className="label">Ділянки</div>
          <div className="value">{totals.sites}</div>
          <div className="hint">у {totals.projects} проєктах · {totals.scenes} з 3D-сценою</div>
        </div>
        <div className="stat">
          <div className="label">Кадри</div>
          <div className="value">{totals.images.toLocaleString("uk-UA")}</div>
          <div className="hint">у {totals.captures} захопленнях</div>
        </div>
        <div className="stat">
          <div className="label">Реконструкції</div>
          <div className="value">{totals.runs}</div>
          <div className="hint">run-ів пайплайна</div>
        </div>
        <div className="stat">
          <div className="label">Черга</div>
          <div className="value" style={{ color: sys?.queue.running ? "var(--amber)" : undefined }}>
            {sys ? sys.queue.running + sys.queue.queued : "—"}
          </div>
          <div className="hint">
            {sys ? `${sys.queue.running} виконується · ${sys.queue.queued} чекає` : "…"}
            {sys && !gpuOnline && <span style={{ color: "var(--red)" }}> · GPU-воркер офлайн</span>}
          </div>
        </div>
      </div>

      <SiteMap tall sites={markers} onSiteClick={(key) => navigate(`/sites/${key}`)} />

      {isLoading ? null : !sites?.length ? (
        <div className="panel">
          <div className="empty">
            <div className="big">Ще немає жодної ділянки</div>
            Створи ділянку кліком на карті, або перенеси наявний датасет:
            <div style={{ marginTop: 10 }}>
              <code>aerosplat catalog migrate-dataset dataset_v6</code>
            </div>
            <div style={{ marginTop: 16 }}>
              <button className="btn primary" onClick={() => setCreating(true)}>
                <Icon name="plus" /> Нова ділянка
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div className="site-cards">
          {sites.map((s) => (
            <Link key={s.site.project_id + s.site.id} to={`/sites/${s.site.project_id}/${s.site.id}`} className="site-card">
              <div className="row" style={{ justifyContent: "space-between" }}>
                <div style={{ minWidth: 0 }}>
                  <div className="title">{s.site.name || s.site.id}</div>
                  <div className="key">
                    {s.site.project_id}/{s.site.id}
                  </div>
                </div>
                {s.active_jobs > 0 ? (
                  <span className="badge run">
                    <span className="dot" />
                    {s.active_jobs} задач
                  </span>
                ) : s.latest_run ? (
                  <RunBadge status={s.latest_run.status} />
                ) : (
                  <span className="badge mute">без run-ів</span>
                )}
              </div>
              <div className="facts">
                <span>
                  <b>{s.captures}</b> захоплень
                </span>
                <span>
                  <b>{s.images}</b> кадрів
                </span>
                <span>
                  <b>{s.runs}</b> run-ів
                </span>
              </div>
              <div className="row faint" style={{ marginTop: 10, fontSize: 12 }}>
                <span>{KIND_LABEL[s.site.kind]}</span>
                <span className="grow" />
                {s.scene && (
                  <span className="badge accent">
                    <Icon name="cube" size={12} /> 3D
                  </span>
                )}
                <span>{s.latest_run ? timeAgo(s.latest_run.finished_at ?? s.latest_run.created_at) : ""}</span>
              </div>
            </Link>
          ))}
        </div>
      )}

      <div className="panel">
        <div className="panel-head">
          <h2>Останні задачі</h2>
          <Link to="/jobs" className="btn ghost sm">
            Усі задачі →
          </Link>
        </div>
        <div className="panel-body flush">
          <JobsTable jobs={jobs} compact />
        </div>
      </div>

      {creating && <CreateSiteDialog onClose={() => setCreating(false)} />}
    </div>
  );
}
