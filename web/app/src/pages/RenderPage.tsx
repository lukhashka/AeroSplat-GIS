import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { ApiError, renderSite } from "../api/client";
import { useRenderStatus } from "../api/hooks";
import type { RenderMeta, RenderMode, RenderOutput, RenderRequest } from "../api/types";
import { Icon } from "../components/Icon";
import { ErrorBox } from "../components/Modal";
import { coord } from "../lib/format";

const SIZES: { label: string; w: number; h: number }[] = [
  { label: "800 × 600", w: 800, h: 600 },
  { label: "1280 × 720", w: 1280, h: 720 },
  { label: "1600 × 1200", w: 1600, h: 1200 },
  { label: "1024 × 1024", w: 1024, h: 1024 },
  { label: "2048 × 2048", w: 2048, h: 2048 },
];

const round = (v: number, d = 2) => Math.round(v * 10 ** d) / 10 ** d;

function NumField({
  label,
  value,
  onChange,
  step = 1,
  hint,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
  hint?: string;
}) {
  const [text, setText] = useState(String(round(value)));
  useEffect(() => {
    if (Number(text) !== value) setText(String(round(value)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);
  const commit = (s: string) => {
    setText(s);
    const n = Number(s);
    if (s.trim() !== "" && Number.isFinite(n)) onChange(n);
  };
  return (
    <label className="field">
      <span>{label}</span>
      <input
        inputMode="decimal"
        className="num"
        value={text}
        onChange={(e) => commit(e.target.value)}
        onKeyDown={(e) => {
          if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
          e.preventDefault();
          const k = e.shiftKey ? 10 : 1;
          commit(String(round(value + (e.key === "ArrowUp" ? 1 : -1) * step * k)));
        }}
      />
      {hint && <small>{hint}</small>}
    </label>
  );
}

/** Start poses that frame the scene, from its bounds in Site ENU. */
function framing(meta: RenderMeta): Record<RenderMode, RenderRequest> {
  const { min, max } = meta.scene_bounds_enu;
  const ce = (min[0] + max[0]) / 2;
  const cn = (min[1] + max[1]) / 2;
  const se = max[0] - min[0];
  const sn = max[1] - min[1];
  const size = Math.max(se, sn, 10);
  const base = { roll_deg: 0, width: 1280, height: 720, output: "rgb" as RenderOutput, fov_deg: 60 };
  return {
    persp: {
      ...base,
      mode: "persp",
      position: { e: round(ce), n: round(cn - 0.7 * sn), u: round(max[2] + 0.3 * size) },
      yaw_deg: 0,
      pitch_deg: -35,
      extent_m: round(size * 1.05, 0),
    },
    ortho: {
      ...base,
      mode: "ortho",
      width: 1024,
      height: 1024,
      position: { e: round(ce), n: round(cn) },
      yaw_deg: 0,
      pitch_deg: -90,
      extent_m: round(size * 1.05, 0),
    },
  };
}

/** Pixel of the shown image under the mouse -> ENU metres, for a straight-down orthographic render. */
function groundAt(meta: RenderMeta, img: HTMLImageElement, ev: React.MouseEvent): [number, number] | null {
  const g = meta.ground_grid;
  if (!g) return null;
  const r = img.getBoundingClientRect();
  const i = ((ev.clientX - r.left) / r.width) * img.naturalWidth - 0.5;
  const j = ((ev.clientY - r.top) / r.height) * img.naturalHeight - 0.5;
  return [g.pixel00_enu[0] + i * g.du_enu[0] + j * g.dv_enu[0], g.pixel00_enu[1] + i * g.du_enu[1] + j * g.dv_enu[1]];
}

export function RenderPage() {
  const { project = "", site = "" } = useParams();
  const scene = `${project}/${site}`;
  const { data: status } = useRenderStatus();

  // One tiny orthographic render tells us where the scene is.
  const probe = useQuery({
    queryKey: ["render-probe", project, site],
    queryFn: async () => {
      const probeReq: RenderRequest = {
        mode: "ortho",
        position: { e: 0, n: 0 },
        yaw_deg: 0,
        pitch_deg: -90,
        roll_deg: 0,
        width: 16,
        height: 16,
        fov_deg: 60,
        extent_m: 5000,
        output: "alpha",
      };
      const r = await renderSite(project, site, probeReq);
      URL.revokeObjectURL(r.url);
      return r.meta;
    },
    staleTime: 5 * 60_000,
    // 503 = the service is still starting (torch + gsplat take a few seconds) or just restarted.
    retry: (n, e) => e instanceof ApiError && e.status === 503 && n < 30,
    retryDelay: 1500,
  });
  const frames = useMemo(() => (probe.data ? framing(probe.data) : null), [probe.data]);

  const [req, setReq] = useState<RenderRequest | null>(null);
  useEffect(() => {
    if (frames && !req) setReq(frames.persp);
  }, [frames, req]);

  const [shot, setShot] = useState<{ url: string; meta: RenderMeta } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [hover, setHover] = useState<[number, number] | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    if (!req) return;
    const ctl = new AbortController();
    const t = setTimeout(() => {
      setBusy(true);
      renderSite(project, site, req, ctl.signal)
        .then((r) => {
          setShot((old) => {
            if (old) URL.revokeObjectURL(old.url);
            return r;
          });
          setError(null);
        })
        .catch((e) => {
          if ((e as Error).name !== "AbortError") setError(e);
        })
        .finally(() => {
          if (!ctl.signal.aborted) setBusy(false);
        });
    }, 250);
    return () => {
      clearTimeout(t);
      ctl.abort();
    };
  }, [req, project, site]);

  const set = (patch: Partial<RenderRequest>) => setReq((r) => (r ? { ...r, ...patch } : r));
  const setPos = (k: "e" | "n" | "u", v: number) => setReq((r) => (r ? { ...r, position: { ...r.position, [k]: v } } : r));
  const switchMode = (mode: RenderMode) => {
    if (!frames || !req || mode === req.mode) return;
    setReq({ ...frames[mode], output: req.output });
  };

  const ortho = req?.mode === "ortho";
  const offline = status && !status.online;
  const meta = shot?.meta;

  return (
    <div className="page" style={{ maxWidth: "none", gap: 14 }}>
      <div className="page-head">
        <div>
          <div className="crumbs">
            <Link to="/">Ділянки</Link> / <Link to={`/sites/${scene}`}>{scene}</Link> / Камера
          </div>
          <h1>Вид із камери</h1>
        </div>
        {req && (
          <div className="row">
            <div className="segmented">
              <button className={!ortho ? "on" : ""} onClick={() => switchMode("persp")}>
                Перспектива
              </button>
              <button className={ortho ? "on" : ""} onClick={() => switchMode("ortho")}>
                Ортографічна
              </button>
            </div>
            {shot && (
              <a className="btn" href={shot.url} download={`${project}-${site}-${req.mode}-${req.output}.png`}>
                <Icon name="upload" style={{ transform: "rotate(180deg)" }} /> PNG
              </a>
            )}
          </div>
        )}
      </div>

      {offline && (
        <div className="error-box">
          Сервіс рендерингу не працює. Запусти його разом з усім: <code>aerosplat up</code> (або окремо:{" "}
          <code>aerosplat render-service</code>).
        </div>
      )}
      {probe.error && <ErrorBox error={probe.error} />}
      {!probe.error && !req && <div className="faint">Завантаження сцени на GPU…</div>}

      {req && (
        <div className="render-layout">
          <div className="panel render-controls">
            <div className="panel-body" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <div className="form-grid">
                <NumField label="Схід, e (м)" value={req.position.e} step={5} onChange={(v) => setPos("e", v)} />
                <NumField label="Північ, n (м)" value={req.position.n} step={5} onChange={(v) => setPos("n", v)} />
                {!ortho ? (
                  <NumField label="Висота, u (м)" value={req.position.u ?? 0} step={5} onChange={(v) => setPos("u", v)} />
                ) : (
                  <label className="field">
                    <span>Висота, u (м)</span>
                    <input value={req.position.u === undefined ? "над сценою" : req.position.u} readOnly />
                  </label>
                )}
                <NumField label="Курс, yaw (°)" value={req.yaw_deg} step={5} hint="0 = північ, 90 = схід" onChange={(v) => set({ yaw_deg: v })} />
                <NumField
                  label="Нахил, pitch (°)"
                  value={req.pitch_deg}
                  step={5}
                  hint="−90 = униз"
                  onChange={(v) => set({ pitch_deg: Math.max(-90, Math.min(90, v)) })}
                />
                <NumField label="Крен, roll (°)" value={req.roll_deg} step={5} onChange={(v) => set({ roll_deg: v })} />
                {!ortho ? (
                  <NumField label="Поле зору, вертикальне (°)" value={req.fov_deg} step={5} onChange={(v) => set({ fov_deg: Math.max(2, Math.min(160, v)) })} />
                ) : (
                  <NumField label="Ширина кадру (м)" value={req.extent_m} step={50} onChange={(v) => set({ extent_m: Math.max(1, v) })} />
                )}
              </div>
              <div className="form-grid">
                <label className="field">
                  <span>Що малювати</span>
                  <select value={req.output} onChange={(e) => set({ output: e.target.value as RenderOutput })}>
                    <option value="rgb">Колір</option>
                    <option value="depth">Глибина (16 біт)</option>
                    <option value="alpha">Прозорість</option>
                  </select>
                </label>
                <label className="field">
                  <span>Розмір</span>
                  <select
                    value={`${req.width}x${req.height}`}
                    onChange={(e) => {
                      const [w, h] = e.target.value.split("x").map(Number);
                      set({ width: w!, height: h! });
                    }}
                  >
                    {SIZES.map((s) => (
                      <option key={s.label} value={`${s.w}x${s.h}`}>
                        {s.label}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="row">
                <button className="btn sm" onClick={() => frames && setReq(frames[req.mode])}>
                  <Icon name="refresh" size={14} /> Вписати в сцену
                </button>
                <span className="faint" style={{ fontSize: 11.5 }}>
                  ↑/↓ у полі — крок, Shift — ×10
                </span>
              </div>
              {meta && (
                <dl className="kv" style={{ fontSize: 12.5 }}>
                  <dt>Камера</dt>
                  <dd className="num">
                    {meta.camera_enu.map((v) => v.toFixed(1)).join(", ")} м
                    {meta.camera_geo && (
                      <>
                        <br />
                        {coord(meta.camera_geo.lat, meta.camera_geo.lon)} · h {meta.camera_geo.h.toFixed(1)} м
                      </>
                    )}
                  </dd>
                  <dt>Система</dt>
                  <dd>{meta.frame}</dd>
                  <dt>Рендер</dt>
                  <dd className="num">
                    {meta.render_ms} мс · {(meta.gaussians / 1e6).toFixed(2)} млн гаусіан
                  </dd>
                  {meta.ground_grid && (
                    <>
                      <dt>Піксель</dt>
                      <dd className="num">{(meta.ground_grid.m_per_px * 100).toFixed(1)} см на піксель</dd>
                    </>
                  )}
                  {meta.depth && (
                    <>
                      <dt>Глибина</dt>
                      <dd className="num">0…{meta.depth.max_m.toFixed(0)} м (значення / 65535 × макс.)</dd>
                    </>
                  )}
                </dl>
              )}
            </div>
          </div>

          <div className="render-stage">
            <ErrorBox error={error} />
            {shot ? (
              <div className="render-img">
                <img
                  ref={imgRef}
                  src={shot.url}
                  alt={`Рендер ${scene}`}
                  onMouseMove={(e) => setHover(imgRef.current ? groundAt(shot.meta, imgRef.current, e) : null)}
                  onMouseLeave={() => setHover(null)}
                />
                {busy && <div className="render-busy">рендер…</div>}
                {hover && (
                  <div className="render-readout num">
                    E {hover[0].toFixed(1)} · N {hover[1].toFixed(1)} м
                  </div>
                )}
              </div>
            ) : (
              <div className="faint">{busy ? "Рендер…" : ""}</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
