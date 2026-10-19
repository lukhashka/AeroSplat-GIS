import { useEffect, useMemo, useRef, useState } from "react";

import type { FlightTrajectory, MissionGeometry } from "../api/types";
import { overlayMessage } from "../lib/flight";
import { missionToServed, originOffsetM, type SceneFrame, type Vec3 } from "../lib/frames";

const SOURCE = "aerosplat-mission";

export type SceneLayer = "corridor" | "fence" | "target" | "waypoints" | "trajectory" | "drone";
export type SceneView = "mission" | "top" | "target" | "side";

export type SceneStatus =
  | { state: "loading" }
  | { state: "ready"; scene: string | null; count: number | null; originOffsetM: number | null }
  | { state: "error"; message: string };

interface Meta extends SceneFrame {
  dataset?: string;
  count?: number;
  extent: number;
  near_cull_fraction?: number; // appGLSL.js frameCamera
}

/** A scene-meta for a mission with nothing under it: Site ENU about the mission's middle. */
function bareMeta(g: MissionGeometry): Meta {
  const [lo, hi] = g.bbox_enu;
  const center = [0, 1, 2].map((i) => (lo[i]! + hi[i]!) / 2);
  const extent = Math.max(Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2, 10);
  return {
    center,
    extent,
    scene_scale: 100 / extent,
    up_axis_rotation: { axis: "x", degrees: -90 },
    metric: { origin: g.origin },
    near_cull_fraction: 0.001,
  };
}

/**
 * The classic 3DGS viewer (web/mission.html) in a frame, with the mission drawn into its scene
 * by web/mission-overlay.js. Points go over as served coordinates of that scene (frames.ts).
 * `scene` null draws the mission over a plain grid.
 */
export function MissionScene({
  scene,
  geometry,
  trajectory,
  index,
  layers,
  xray,
  follow,
  view,
  onStatus,
}: {
  scene: string | null;
  geometry: MissionGeometry;
  trajectory: FlightTrajectory | null;
  index: number;
  layers: Record<SceneLayer, boolean>;
  xray: boolean;
  follow: boolean;
  view: { name: SceneView; n: number };
  onStatus: (s: SceneStatus) => void;
}) {
  const frame = useRef<HTMLIFrameElement>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const onStatusRef = useRef(onStatus);
  onStatusRef.current = onStatus;
  const src = scene ? `/web/mission.html?dataset=${encodeURIComponent(scene)}&mode=splat` : "/web/mission.html?noscene=1";

  const send = (msg: Record<string, unknown>) =>
    frame.current?.contentWindow?.postMessage({ source: SOURCE, ...msg }, location.origin);

  // A new scene (or none) is a new frame: start over.
  useEffect(() => {
    setMeta(null);
    onStatusRef.current({ state: "loading" });
  }, [src]);

  useEffect(() => {
    const onMessage = (e: MessageEvent) => {
      if (e.origin !== location.origin || e.source !== frame.current?.contentWindow) return;
      const msg = e.data as { source?: string; type?: string; meta?: Meta; message?: string };
      if (msg?.source !== SOURCE) return;
      if (msg.type === "ready" && msg.meta) {
        if (!msg.meta.metric) {
          onStatusRef.current({
            state: "error",
            message: `Сцена ${scene} не має метричного кадру (вільний кадр COLMAP): місію в неї не покласти.`,
          });
          return;
        }
        setMeta(msg.meta);
        onStatusRef.current({
          state: "ready",
          scene,
          count: scene ? (msg.meta.count ?? null) : null,
          originOffsetM: originOffsetM(geometry.origin, msg.meta.metric.origin),
        });
      } else if (msg.type === "error") {
        onStatusRef.current({ state: "error", message: msg.message ?? "помилка переглядача" });
      }
    };
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, [scene, geometry.origin]);

  const toServed = useMemo(() => (meta ? missionToServed(geometry.origin, meta) : null), [meta, geometry.origin]);
  const trackToServed = useMemo(
    () => (meta && trajectory ? missionToServed(trajectory.origin, meta) : null),
    [meta, trajectory],
  );

  // the overlay itself: every time the mission or the run changes
  useEffect(() => {
    if (toServed) send({ type: "overlay", data: overlayMessage(geometry, trajectory, toServed, trackToServed ?? toServed) });
  }, [toServed, trackToServed, geometry, trajectory]);

  useEffect(() => {
    if (toServed) send({ type: "layers", visible: layers, xray, follow });
  }, [toServed, layers, xray, follow, geometry, trajectory]);

  useEffect(() => {
    if (!toServed || !trackToServed || !trajectory || !trajectory.enu.length) return;
    const k = Math.max(0, Math.min(index, trajectory.enu.length - 1));
    const p: Vec3 = trackToServed(trajectory.enu[k]!);
    send({ type: "drone", index: k, p, yawDeg: trajectory.yaw_deg[k], failsafe: trajectory.failsafe[k] });
  }, [toServed, trackToServed, trajectory, index, geometry]);

  useEffect(() => {
    if (toServed && view.n > 0) send({ type: "view", name: view.name });
  }, [view]);

  return (
    <iframe
      key={src}
      ref={frame}
      className="mission-frame"
      title="Місія над 3D-сценою"
      src={src}
      onLoad={() => {
        // No scene to wait for: place the mission's own frame and let the overlay say ready.
        if (!scene) send({ type: "place", meta: bareMeta(geometry) });
      }}
    />
  );
}
