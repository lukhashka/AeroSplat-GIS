import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { useDiagnostic, useJobs } from "../api/hooks";
import type { JobStatus } from "../api/types";
import { Icon } from "../components/Icon";
import { JobsTable } from "../components/JobsTable";
import { ErrorBox } from "../components/Modal";

const FILTERS: { id: string; label: string; status?: JobStatus[] }[] = [
  { id: "all", label: "Усі" },
  { id: "active", label: "Активні", status: ["queued", "running"] },
  { id: "done", label: "Успішні", status: ["succeeded"] },
  { id: "failed", label: "Невдалі", status: ["failed", "cancelled"] },
];

export function JobsPage() {
  const navigate = useNavigate();
  const [filter, setFilter] = useState("all");
  const current = FILTERS.find((f) => f.id === filter)!;
  const { data: jobs = [], isLoading } = useJobs({ status: current.status, limit: 200 });
  const diag = useDiagnostic();

  const runDiagnostic = (resource: "gpu" | "cpu") =>
    diag.mutate(resource, { onSuccess: (job) => navigate(`/jobs/${job.id}`) });

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Задачі</h1>
          <div className="sub">Черга воркерів: реконструкції на GPU (по одній) і службові задачі на CPU.</div>
        </div>
        <div className="row">
          <button className="btn" onClick={() => runDiagnostic("gpu")} disabled={diag.isPending} title="Фіктивні стадії: перевіряє, що GPU-воркер живий і стрім працює">
            <Icon name="pulse" /> Перевірити GPU-воркер
          </button>
          <button className="btn" onClick={() => runDiagnostic("cpu")} disabled={diag.isPending}>
            <Icon name="pulse" /> CPU-воркер
          </button>
        </div>
      </div>
      <ErrorBox error={diag.error} />
      <div className="panel">
        <div className="panel-head">
          <div className="segmented">
            {FILTERS.map((f) => (
              <button key={f.id} className={filter === f.id ? "on" : ""} onClick={() => setFilter(f.id)}>
                {f.label}
              </button>
            ))}
          </div>
          <span className="faint" style={{ fontSize: 12 }}>
            {jobs.length} задач
          </span>
        </div>
        <div className="panel-body flush">{isLoading ? <div className="empty">Завантаження…</div> : <JobsTable jobs={jobs} />}</div>
      </div>
    </div>
  );
}
