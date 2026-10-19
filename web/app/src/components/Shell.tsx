import { useQuery, useQueryClient } from "@tanstack/react-query";
import { NavLink, Outlet } from "react-router-dom";

import { auth } from "../api/client";
import { useRenderStatus, useSystem } from "../api/hooks";
import type { SystemInfo, WorkerInfo } from "../api/types";
import { Icon } from "./Icon";

function WorkerLine({ label, workers }: { label: string; workers: WorkerInfo[] }) {
  const online = workers.filter((w) => w.online);
  const busy = online.some((w) => w.current_job);
  const state = !online.length ? "" : busy ? "busy" : "on";
  const text = !online.length ? "офлайн" : busy ? "працює" : "очікує";
  return (
    <div className="line">
      <span className={`led ${state}`} />
      <span className="grow">{label}</span>
      <span style={{ color: online.length ? "var(--text)" : "var(--text-faint)" }}>{text}</span>
    </div>
  );
}

function RenderLine() {
  const { data } = useRenderStatus();
  const loaded = data?.loaded?.length ?? 0;
  const state = !data?.online ? "" : loaded ? "busy" : "on";
  const text = !data?.online ? "офлайн" : loaded ? "сцена в VRAM" : "очікує";
  return (
    <div className="line" title="Сервіс рендерингу 3DGS (вид із камери, ортофото)">
      <span className={`led ${state}`} />
      <span className="grow">Рендер</span>
      <span style={{ color: data?.online ? "var(--text)" : "var(--text-faint)" }}>{text}</span>
    </div>
  );
}

/** "Signed in as <token>" when the page is used from another machine; nothing on this one. */
function SessionLine() {
  const qc = useQueryClient();
  const { data } = useQuery({ queryKey: ["whoami"], queryFn: auth.whoami, staleTime: 60_000 });
  if (data?.via !== "token") return null;
  const logout = async () => {
    await auth.logout().catch(() => undefined);
    await qc.invalidateQueries();
  };
  return (
    <div className="line" title="Запити підписуються іменем токена (журнал аудиту)">
      <span className="grow faint">
        токен <b className="mono">{data.name}</b>
      </span>
      <button className="btn ghost sm" onClick={logout}>
        вийти
      </button>
    </div>
  );
}

function SystemWidget({ sys }: { sys: SystemInfo | undefined }) {
  if (!sys) return <div className="sys faint">З'єднання з сервером…</div>;
  const gpu = sys.gpu?.[0];
  const workers = sys.workers;
  return (
    <div className="sys">
      <WorkerLine label="GPU-воркер" workers={workers.filter((w) => w.resource === "gpu")} />
      <WorkerLine label="CPU-воркер" workers={workers.filter((w) => w.resource === "cpu")} />
      <RenderLine />
      {gpu && (
        <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
          <div className="line">
            <Icon name="chip" size={14} />
            <span className="grow" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {gpu.name.replace("NVIDIA GeForce ", "")}
            </span>
            <span className="num">{gpu.utilization.toFixed(0)}%</span>
          </div>
          <div className="meter" title={`VRAM ${gpu.memory_used_mb.toFixed(0)} / ${gpu.memory_total_mb.toFixed(0)} MB`}>
            <i style={{ width: `${(100 * gpu.memory_used_mb) / gpu.memory_total_mb}%` }} />
          </div>
          <div className="faint num" style={{ fontSize: 11 }}>
            VRAM {(gpu.memory_used_mb / 1024).toFixed(1)} / {(gpu.memory_total_mb / 1024).toFixed(1)} GB · {gpu.temperature_c.toFixed(0)}°C
          </div>
        </div>
      )}
      <SessionLine />
      <div className="faint" style={{ fontSize: 11 }}>
        v{sys.version}
        {sys.git_commit ? ` · ${sys.git_commit.slice(0, 7)}` : ""}
      </div>
    </div>
  );
}

export function Shell() {
  const { data: sys } = useSystem();
  const active = sys ? sys.queue.running + sys.queue.queued : 0;
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <img src={`${import.meta.env.BASE_URL}favicon.svg`} width={30} height={30} alt="" />
          <div>
            AeroSplat
            <small>GIS · 3DGS · БПЛА</small>
          </div>
        </div>
        <NavLink to="/" end className="nav-link">
          <Icon name="map" /> Ділянки
        </NavLink>
        <NavLink to="/jobs" className="nav-link">
          <Icon name="jobs" /> Задачі
          {active > 0 && <span className="count">{active}</span>}
        </NavLink>
        <a href="/web/" className="nav-link" target="_blank" rel="noreferrer">
          <Icon name="cube" /> Класичний в'ювер
          <Icon name="external" size={13} style={{ marginLeft: "auto", opacity: 0.6 }} />
        </a>
        <a href="/docs" className="nav-link" target="_blank" rel="noreferrer">
          <Icon name="report" /> API (OpenAPI)
          <Icon name="external" size={13} style={{ marginLeft: "auto", opacity: 0.6 }} />
        </a>
        <div className="spacer" />
        <SystemWidget sys={sys} />
      </aside>
      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
