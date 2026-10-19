import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";

import type { FlightTrajectory, SimEvent } from "../api/types";
import { atCap, indexAt, PHASE_LABEL, phaseColor } from "../lib/flight";

const LEFT = 52;
const RIGHT = 14;
const BAND = 12;
const TRACK = 64;
const GAP = 14;
const AXIS = 20;
const CLEARANCE_TOP_M = 12; // the oracle's field saturates well before; cut like report.html

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [w, setW] = useState(800);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => e && setW(Math.max(320, e.contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

const fmtT = (s: number) => {
  const sign = s < 0 ? "−" : "";
  const a = Math.abs(s);
  return `${sign}${Math.floor(a / 60)}:${(a % 60).toFixed(1).padStart(4, "0")}`;
};

function path(xs: number[], ys: (number | null)[]): string {
  let d = "";
  let pen = false;
  for (let i = 0; i < xs.length; i++) {
    const y = ys[i];
    if (y === null || y === undefined || !Number.isFinite(y)) {
      pen = false;
      continue;
    }
    d += `${pen ? "L" : "M"}${xs[i]!.toFixed(1)},${y.toFixed(1)}`;
    pen = true;
  }
  return d;
}

/**
 * Phase band, clearance and height above home on one clock (simulation seconds from the
 * mission start). Two value tracks, each with its own y: never one chart with two scales.
 * Hover shows the values under the cursor; a click or drag seeks.
 */
export function FlightTimeline({
  trajectory,
  index,
  events,
  onSeek,
}: {
  trajectory: FlightTrajectory;
  index: number;
  events: SimEvent[];
  onSeek: (index: number) => void;
}) {
  const [box, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const dragging = useRef(false);
  const tr = trajectory;
  const t0 = tr.t[0] ?? 0;
  const t1 = tr.t[tr.t.length - 1] ?? 1;
  const start = tr.mission_start_t ?? t0;
  const plotW = width - LEFT - RIGHT;
  const x = (t: number) => LEFT + ((t - t0) / Math.max(t1 - t0, 1e-6)) * plotW;
  const hasClearance = !!tr.clearance;
  const yClr = BAND + GAP;
  const yAlt = yClr + (hasClearance ? TRACK + GAP : 0);
  const height = yAlt + TRACK + AXIS;

  const altMax = useMemo(() => Math.max(5, ...tr.alt_rel_m) * 1.1, [tr]);
  const altMin = useMemo(() => Math.min(0, ...tr.alt_rel_m), [tr]);
  // x() changes only with the width and the recording
  const xs = useMemo(() => tr.t.map(x), [tr, width]);
  const altPath = useMemo(
    () => path(xs, tr.alt_rel_m.map((a) => yAlt + TRACK - ((a - altMin) / (altMax - altMin)) * TRACK)),
    [xs, tr, yAlt, altMax, altMin],
  );
  const clr = tr.clearance;
  const clrPath = useMemo(
    () =>
      clr
        ? path(
            xs,
            clr.clearance_m.map((c) => (c === null ? null : yClr + TRACK - (Math.min(c, CLEARANCE_TOP_M) / CLEARANCE_TOP_M) * TRACK)),
          )
        : "",
    [xs, clr, yClr],
  );

  const kAt = (clientX: number) => {
    const r = box.current!.getBoundingClientRect();
    const t = t0 + ((clientX - r.left - LEFT) / plotW) * (t1 - t0);
    return indexAt(tr.t, t);
  };
  const down = (e: ReactPointerEvent) => {
    dragging.current = true;
    (e.target as Element).setPointerCapture?.(e.pointerId);
    onSeek(kAt(e.clientX));
  };
  const move = (e: ReactPointerEvent) => {
    const k = kAt(e.clientX);
    setHover(k);
    if (dragging.current) onSeek(k);
  };

  const cur = Math.max(0, Math.min(index, tr.t.length - 1));
  const hv = hover ?? null;
  const ticks = useMemo(() => {
    const span = t1 - start;
    const step = [5, 10, 15, 30, 60, 120].find((s) => span / s <= 10) ?? 300;
    const out: number[] = [];
    for (let s = Math.ceil((t0 - start) / step) * step; start + s <= t1; s += step) out.push(s);
    return out;
  }, [t0, t1, start]);

  return (
    <div ref={box} className="timeline" onPointerLeave={() => setHover(null)}>
      <svg
        width={width}
        height={height}
        role="img"
        aria-label="Фази, кліренс і висота за часом"
        onPointerDown={down}
        onPointerMove={move}
        onPointerUp={() => (dragging.current = false)}
      >
        {/* phase band */}
        <text x={LEFT - 8} y={BAND - 2} className="tl-label" textAnchor="end">
          фаза
        </text>
        {tr.segments.map((s, i) => {
          const a = x(tr.t[s.i0]!);
          const b = x(tr.t[s.i1]!);
          return (
            <rect
              key={i}
              x={a}
              y={0}
              width={Math.max(b - a - 1, 1)}
              height={BAND}
              rx={2}
              fill={phaseColor(s.phase, s.failsafe)}
              opacity={s.phase === "ground" ? 0.35 : 1}
            >
              <title>{`${PHASE_LABEL[s.phase] ?? s.phase}${s.failsafe ? ` · failsafe ${s.mode}` : ""}`}</title>
            </rect>
          );
        })}

        {/* clearance */}
        {clr && (
          <g>
            <text x={LEFT - 8} y={yClr + 10} className="tl-label" textAnchor="end">
              кліренс
            </text>
            <text x={LEFT - 8} y={yClr + 24} className="tl-tick" textAnchor="end">
              {CLEARANCE_TOP_M} м
            </text>
            <text x={LEFT - 8} y={yClr + TRACK} className="tl-tick" textAnchor="end">
              0
            </text>
            <line x1={LEFT} x2={LEFT + plotW} y1={yClr + TRACK} y2={yClr + TRACK} className="tl-axis" />
            {Number.isFinite(clr.required_m) && (
              <g>
                <line
                  x1={LEFT}
                  x2={LEFT + plotW}
                  y1={yClr + TRACK - (clr.required_m / CLEARANCE_TOP_M) * TRACK}
                  y2={yClr + TRACK - (clr.required_m / CLEARANCE_TOP_M) * TRACK}
                  className="tl-threshold"
                />
                <text x={LEFT + plotW} y={yClr + TRACK - (clr.required_m / CLEARANCE_TOP_M) * TRACK - 4} className="tl-tick" textAnchor="end">
                  потрібно {clr.required_m.toFixed(1)} м
                </text>
              </g>
            )}
            <path d={clrPath} className="tl-line" />
          </g>
        )}

        {/* height above home */}
        <text x={LEFT - 8} y={yAlt + 10} className="tl-label" textAnchor="end">
          висота
        </text>
        <text x={LEFT - 8} y={yAlt + 24} className="tl-tick" textAnchor="end">
          {altMax.toFixed(0)} м
        </text>
        <text x={LEFT - 8} y={yAlt + TRACK} className="tl-tick" textAnchor="end">
          {altMin.toFixed(0)}
        </text>
        <line x1={LEFT} x2={LEFT + plotW} y1={yAlt + TRACK} y2={yAlt + TRACK} className="tl-axis" />
        <path d={altPath} className="tl-line" />

        {/* time axis, from the mission start */}
        {ticks.map((s) => (
          <text key={s} x={x(start + s)} y={height - 4} className="tl-tick" textAnchor="middle">
            {fmtT(s)}
          </text>
        ))}

        {/* events */}
        {events.map((e, i) => (
          <line key={i} x1={x(e.t)} x2={x(e.t)} y1={BAND + 2} y2={BAND + 8} className={`tl-event ${e.kind}`}>
            <title>{e.detail}</title>
          </line>
        ))}

        {/* hover crosshair, then the current instant on top */}
        {hv !== null && <line x1={xs[hv]} x2={xs[hv]} y1={0} y2={height - AXIS} className="tl-hover" />}
        <line x1={xs[cur]} x2={xs[cur]} y1={0} y2={height - AXIS} className="tl-cursor" />
      </svg>
      {hv !== null && (
        <div className="tl-tip" style={{ left: Math.min(xs[hv]! + 12, width - 190) }}>
          <b>{fmtT(tr.t[hv]! - start)}</b> · {PHASE_LABEL[tr.phase[hv]!] ?? tr.phase[hv]}
          {tr.failsafe[hv] ? ` · ${tr.mode[hv]}` : ""}
          <br />
          висота {tr.alt_rel_m[hv]!.toFixed(2)} м
          {clr && clr.clearance_m[hv] !== null && clr.clearance_m[hv] !== undefined && (
            <>
              {" · "}кліренс {atCap(clr.clearance_m[hv]!, clr.cap_m) ? "≥ " : ""}
              {clr.clearance_m[hv]!.toFixed(2)} м
            </>
          )}
        </div>
      )}
    </div>
  );
}
