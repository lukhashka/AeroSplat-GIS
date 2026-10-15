import type { StageView } from "../api/stream";
import { duration } from "../lib/format";

export interface PlannedStage {
  name: string;
  label: string;
}

const ICON: Record<string, string> = { done: "✓", failed: "✕", skipped: "–", started: "", pending: "" };

/** Stage list: planned stages (from the job payload) merged with live stage events. */
export function StageTimeline({
  planned,
  live,
  now,
  jobActive,
}: {
  planned: PlannedStage[];
  live: StageView[];
  now: number;
  jobActive: boolean;
}) {
  const byName = new Map(live.map((s) => [s.stage, s]));
  const names = planned.length ? planned.map((p) => p.name) : live.map((s) => s.stage);
  const labels = new Map(planned.map((p) => [p.name, p.label]));
  if (!names.length) return <div className="empty">Стадії з'являться, щойно задача стартує.</div>;

  return (
    <div className="stages">
      {names.map((name) => {
        const s = byName.get(name);
        const state = s?.state ?? (jobActive ? "pending" : "pending");
        const label = s?.label ?? labels.get(name) ?? name;
        let detail = "";
        if (s?.state === "done") detail = duration(s.seconds);
        else if (s?.state === "started" && s.startedAt) detail = duration((now - new Date(s.startedAt).getTime()) / 1000);
        else if (s?.state === "skipped") detail = s.reason ?? "пропущено";
        else if (s?.state === "failed") detail = s.error ?? "помилка";
        return (
          <div key={name} className={`stage ${state}`}>
            <div className="ico">{ICON[state]}</div>
            <div>
              <div className="name">{label.replace(/^\d+\/\d+\s+/, "")}</div>
              <div className="detail mono">{name}</div>
            </div>
            <div className={`detail num${s?.state === "failed" ? " mono" : ""}`} style={s?.state === "failed" ? { color: "var(--red)" } : undefined}>
              {detail}
            </div>
          </div>
        );
      })}
    </div>
  );
}
