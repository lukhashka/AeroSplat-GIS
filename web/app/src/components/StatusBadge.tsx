import type { JobStatus, RunStatus } from "../api/types";

const JOB: Record<JobStatus, [string, string]> = {
  queued: ["wait", "у черзі"],
  running: ["run", "виконується"],
  succeeded: ["ok", "готово"],
  failed: ["bad", "помилка"],
  cancelled: ["mute", "скасовано"],
};

const RUN: Record<RunStatus, [string, string]> = {
  running: ["run", "виконується"],
  ok: ["ok", "успішно"],
  failed: ["bad", "помилка"],
};

export function JobBadge({ status, cancelling }: { status: JobStatus; cancelling?: boolean }) {
  const [cls, label] = JOB[status];
  return (
    <span className={`badge ${cls}`}>
      <span className="dot" />
      {cancelling && status === "running" ? "скасовується…" : label}
    </span>
  );
}

export function RunBadge({ status }: { status: RunStatus }) {
  const [cls, label] = RUN[status];
  return (
    <span className={`badge ${cls}`}>
      <span className="dot" />
      {label}
    </span>
  );
}

export const SOURCE_LABEL: Record<string, string> = {
  earthstudio: "Earth Studio",
  drone: "Дрон",
  handheld: "Ручна зйомка",
  internet: "Інтернет",
};

export const KIND_LABEL: Record<string, string> = {
  urban: "Міська ділянка",
  depot: "Склад / хаб",
  critical: "Критичний об'єкт",
};
