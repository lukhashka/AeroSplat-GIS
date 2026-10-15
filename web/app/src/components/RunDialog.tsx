import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { usePipeline, useStartRun } from "../api/hooks";
import type { CaptureOut, Epoch, RunRequest } from "../api/types";
import { ErrorBox, Modal } from "./Modal";
import { SOURCE_LABEL } from "./StatusBadge";

// Knobs worth a field in the dialog; everything else stays at RunConfig defaults
// (or goes in "Інші параметри" as KEY=VALUE lines).
const QUICK: { key: string; label: string; hint?: string }[] = [
  { key: "iterations", label: "Ітерації 3DGS", hint: "5500 ≈ 4 хв на RTX 4060" },
  { key: "max_gaussians", label: "Макс. гаусіанів" },
  { key: "max_res", label: "Роздільність тренування, px" },
  { key: "cloud_source", label: "Хмара для ініціалізації", hint: "dense — після MVS, sparse — швидше" },
];

function parseExtra(text: string): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const i = line.indexOf("=");
    if (i < 1) throw new Error(`рядок «${line}»: очікується KEY=VALUE`);
    const k = line.slice(0, i).trim();
    const v = line.slice(i + 1).trim();
    out[k] =
      v === "true" ? true : v === "false" ? false : v === "none" || v === "null" ? null : v !== "" && !Number.isNaN(Number(v)) ? Number(v) : v;
  }
  return out;
}

export function RunDialog({
  project,
  site,
  captures,
  epochs,
  onClose,
}: {
  project: string;
  site: string;
  captures: CaptureOut[];
  epochs: Epoch[];
  onClose: () => void;
}) {
  const navigate = useNavigate();
  const { data: pipeline } = usePipeline();
  const start = useStartRun(project, site);
  const [capture, setCapture] = useState(captures[0]?.capture.id ?? "");
  const [epoch, setEpoch] = useState(epochs[0]?.id ?? "baseline");
  const [selected, setSelected] = useState<string[] | null>(null);
  const [quick, setQuick] = useState<Record<string, string>>({});
  const [extra, setExtra] = useState("");
  const [parseError, setParseError] = useState<string | null>(null);

  const stages = pipeline?.stages ?? [];
  const chosen = selected ?? stages.map((s) => s.name);
  const toggle = (name: string) =>
    setSelected((prev) => {
      const cur = prev ?? stages.map((s) => s.name);
      return cur.includes(name) ? cur.filter((n) => n !== name) : stages.map((s) => s.name).filter((n) => n === name || cur.includes(n));
    });

  const submit = () => {
    let overrides: Record<string, unknown>;
    try {
      overrides = parseExtra(extra);
    } catch (e) {
      setParseError((e as Error).message);
      return;
    }
    setParseError(null);
    for (const [k, v] of Object.entries(quick)) {
      if (v.trim() === "") continue;
      overrides[k] = Number.isNaN(Number(v)) ? v : Number(v);
    }
    const all = stages.map((s) => s.name);
    const body: RunRequest = { epoch, capture, overrides };
    if (chosen.length !== all.length) body.stages = all.filter((n) => chosen.includes(n));
    start.mutate(body, { onSuccess: (job) => navigate(`/jobs/${job.id}`) });
  };

  return (
    <Modal
      wide
      title="Запустити реконструкцію"
      subtitle="Задача стане в чергу GPU-воркера. Одночасно виконується лише одна GPU-задача."
      onClose={onClose}
      footer={
        <>
          <button className="btn ghost" onClick={onClose}>
            Скасувати
          </button>
          <button className="btn primary" onClick={submit} disabled={!capture || !chosen.length || start.isPending}>
            {start.isPending ? "Постановка в чергу…" : "Запустити"}
          </button>
        </>
      }
    >
      <div className="form-grid">
        <label className="field">
          <span>Захоплення</span>
          <select value={capture} onChange={(e) => setCapture(e.target.value)}>
            {captures.map((c) => (
              <option key={c.capture.id} value={c.capture.id}>
                {c.capture.id} · {SOURCE_LABEL[c.capture.source]} · {c.images} кадрів
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>Епоха</span>
          <select value={epoch} onChange={(e) => setEpoch(e.target.value)}>
            {epochs.map((e) => (
              <option key={e.id} value={e.id}>
                {e.label || e.id}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="field">
        <span>Стадії</span>
        <div className="chips">
          {stages.map((s) => (
            <span key={s.name} className={`chip${chosen.includes(s.name) ? " on" : ""}`} onClick={() => toggle(s.name)} title={s.label}>
              {s.name}
            </span>
          ))}
        </div>
        <small>Стадії, яким нема чого робити (напр. georef без відомих поз), пропускаються автоматично.</small>
      </div>

      <div className="form-grid">
        {QUICK.map((q) => (
          <label className="field" key={q.key}>
            <span>{q.label}</span>
            <input
              value={quick[q.key] ?? ""}
              placeholder={String(pipeline?.defaults[q.key] ?? "")}
              onChange={(e) => setQuick({ ...quick, [q.key]: e.target.value })}
            />
            {q.hint && <small>{q.hint}</small>}
          </label>
        ))}
        <label className="field full">
          <span>Інші параметри (KEY=VALUE, по рядку)</span>
          <textarea
            rows={3}
            className="mono"
            value={extra}
            onChange={(e) => setExtra(e.target.value)}
            placeholder={"matcher=exhaustive\nscale_reg=0.02"}
          />
          <small>Ключі RunConfig (src/aerosplat/config.py); перевіряються сервером до постановки в чергу.</small>
        </label>
      </div>
      <ErrorBox error={parseError ?? start.error} />
    </Modal>
  );
}
