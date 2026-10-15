import { useNavigate } from "react-router-dom";

import type { Job } from "../api/types";
import { between, duration, timeAgo } from "../lib/format";
import { useNow } from "../lib/useNow";
import { JobBadge } from "./StatusBadge";

export function JobsTable({ jobs, compact }: { jobs: Job[]; compact?: boolean }) {
  const navigate = useNavigate();
  const now = useNow(jobs.some((j) => j.status === "running"));
  if (!jobs.length) return <div className="empty">Задач ще не було.</div>;
  return (
    <div className="table-wrap">
      <table className="list">
        <thead>
          <tr>
            <th>Статус</th>
            <th>Задача</th>
            {!compact && <th>Тип</th>}
            <th>Створено</th>
            <th>Тривалість</th>
          </tr>
        </thead>
        <tbody>
          {jobs.map((j) => {
            const secs = j.started_at ? between(j.started_at, j.finished_at ?? new Date(now).toISOString()) : null;
            return (
              <tr key={j.id} className="click" onClick={() => navigate(`/jobs/${j.id}`)}>
                <td className="nowrap">
                  <JobBadge status={j.status} cancelling={j.cancel_requested} />
                </td>
                <td style={{ width: "100%", maxWidth: 0 }}>
                  <div style={{ fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {j.title || j.id}
                  </div>
                  {j.error && !compact && (
                    <div className="faint mono" style={{ fontSize: 11.5, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {j.error.split("\n").slice(-1)[0]}
                    </div>
                  )}
                </td>
                {!compact && (
                  <td className="muted nowrap">
                    {j.kind} · {j.resource.toUpperCase()}
                  </td>
                )}
                <td className="muted nowrap" title={j.created_at}>
                  {timeAgo(j.created_at, now)}
                </td>
                <td className="num muted nowrap">{duration(secs)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
