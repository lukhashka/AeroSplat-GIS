// What the flight view draws, in one place: phase groups and their colours, the event feed, and
// the overlay message for the 3D frame (web/mission-overlay.js) in the scene's served frame.

import type { FlightTrajectory, MissionGeometry, SimEvent } from "../api/types";
import { headingToEnu, type Vec3 } from "./frames";

// Nine phases (aerosplat.sim.replay) are too many hues to tell apart; five groups, in flight
// order, take the first five categorical slots (validated: dataviz palette, dark surface).
// A failsafe is a state, not a phase: it wears the status "critical" red, always with a label.
export const PHASE_GROUPS = [
  { id: "vertical", label: "Зліт і посадка", phases: ["takeoff", "land"], color: "#3987e5" },
  { id: "transit", label: "Транзит", phases: ["transit", "climb", "return"], color: "#d95926" },
  { id: "approach", label: "Спуск і підхід до H", phases: ["descent", "approach"], color: "#199e70" },
  { id: "hold", label: "Утримання в H", phases: ["hold"], color: "#c98500" },
  { id: "exit", label: "Відхід до E", phases: ["exit"], color: "#d55181" },
] as const;
export const FAILSAFE_COLOR = "#d03b3b";
export const GROUND_COLOR = "#5d6f83";

export const PHASE_LABEL: Record<string, string> = {
  ground: "на землі",
  takeoff: "зліт",
  transit: "транзит",
  descent: "спуск до E",
  approach: "підхід до H",
  hold: "утримання в H",
  exit: "відхід до E",
  climb: "набір висоти",
  return: "повернення",
  land: "посадка",
  other: "інше",
};

export function phaseColor(phase: string, failsafe = false): string {
  if (failsafe) return FAILSAFE_COLOR;
  const g = PHASE_GROUPS.find((x) => (x.phases as readonly string[]).includes(phase));
  return g ? g.color : GROUND_COLOR;
}

// Colours of the static layers. Neutral where they are reference geometry (plan, fence), the
// target in the accent, H and E in the colours of the phases that end on them.
export const LAYER_COLORS = {
  fence: "#ff6b6b",
  target: "#3ddbc4",
  hover: "#c98500",
  entry: "#199e70",
  waypoint: "#dbe5ef",
  plan: "#8a9cb0",
  home: "#dbe5ef",
  drone: "#ffffff",
  droneFailsafe: FAILSAFE_COLOR,
  corridor: "#7cb8ff",
  corridorRisk: "#ffb84d",
};

export const INVARIANT_LABEL: Record<string, string> = {
  clearance: "Кліренс",
  fence: "Геофенс",
  hover: "Зависання в H",
  heading: "Курс у H",
  altitude_frame: "Кадр висот",
  failsafe: "Failsafe",
  completion: "Завершення",
};

export const ROLE_LABEL: Record<string, string> = {
  takeoff: "зліт",
  over_entry: "над E",
  entry: "E",
  hover: "H",
  exit: "E (назад)",
  over_exit: "над E",
  over_home: "над домом",
  land: "посадка",
};

/** At the oracle's cap a clearance means "at least that" (aerosplat.sim.oracles). */
export const atCap = (c: number, cap: number | null) => cap !== null && Math.abs(c - cap) < 1e-3;

/** Index of the last sample at or before t (binary search; t sorted). */
export function indexAt(t: number[], time: number): number {
  let lo = 0;
  let hi = t.length - 1;
  if (hi < 0 || time <= t[0]!) return 0;
  if (time >= t[hi]!) return hi;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (t[mid]! <= time) lo = mid;
    else hi = mid;
  }
  return lo;
}

/** The events worth a line in the feed (no battery ticks, no duplicate STATUSTEXT). */
export function feedEvents(events: SimEvent[]): SimEvent[] {
  const keep = new Set(["mode", "armed", "disarmed", "mission_start", "reached", "fault_on", "fault_off", "fence", "end", "statustext", "landed_state"]);
  const out: SimEvent[] = [];
  for (const e of events) {
    if (!keep.has(e.kind)) continue;
    if (e.kind === "statustext" && /^(Mission: |Reached command)/.test(e.detail)) continue; // `reached` says it
    if (e.kind === "fence" && e.detail === "inside") continue;
    const prev = out[out.length - 1];
    if (prev && prev.kind === e.kind && prev.detail === e.detail && Math.abs(prev.t - e.t) < 0.5) continue;
    out.push(e);
  }
  return out;
}

export function eventText(e: SimEvent): string {
  switch (e.kind) {
    case "mode":
      return `режим ${e.detail}${e.data.previous ? ` (з ${String(e.data.previous)})` : ""}`;
    case "armed":
      return "arm: мотори ввімкнено";
    case "disarmed":
      return "disarm: мотори вимкнено";
    case "mission_start":
      return "старт місії";
    case "reached":
      return `досягнуто: ${ROLE_LABEL[e.detail] ?? e.detail}`;
    case "fault_on":
      return `відмова: ${e.detail}`;
    case "fault_off":
      return `відмова знята: ${e.detail}`;
    case "fence":
      return `геофенс: ${e.detail}`;
    case "end":
      return `кінець: ${e.detail}`;
    case "landed_state":
      return `стан: ${e.detail}`;
    default:
      return e.detail;
  }
}

/**
 * The 3D frame's overlay message: everything in the scene's served frame (see frames.ts).
 * The mission is in Site ENU; the recording in the ENU of the origin it was converted with
 * (the same one, unless `origin_offset_m` says otherwise), hence two converters.
 */
export function overlayMessage(
  g: MissionGeometry,
  tr: FlightTrajectory | null,
  toServed: (p: Vec3) => Vec3,
  trackToServed: (p: Vec3) => Vec3,
) {
  const S = (p: Vec3) => toServed(p);
  // Directions are differences of converted points, so they follow any frame change too.
  const dir = (at: Vec3, d: Vec3): Vec3 => {
    const a = S(at);
    const b = S([at[0] + d[0], at[1] + d[1], at[2] + d[2]]);
    return [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  };
  const fence = g.fence;
  const points = tr ? tr.enu.map(trackToServed) : [];
  const all: Vec3[] = [S(g.bbox_enu[0]), S(g.bbox_enu[1]), ...points];
  const lo: Vec3 = [Infinity, Infinity, Infinity];
  const hi: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const p of all)
    for (let i = 0; i < 3; i++) {
      lo[i] = Math.min(lo[i]!, p[i]!);
      hi[i] = Math.max(hi[i]!, p[i]!);
    }
  const t = g.target;
  return {
    palette: LAYER_COLORS,
    bbox: [lo, hi],
    home: S(g.home.enu),
    floorZ: S(g.home.enu)[2],
    droneRadius: g.r_drone_m,
    waypoints: g.waypoints.map((w) => ({ p: S(w.enu), label: String(w.seq), role: w.role })),
    plannedPath: [S(g.home.enu), ...g.waypoints.map((w) => S(w.enu))],
    legs: g.legs.map((l) => ({
      name: l.name,
      points: l.points_enu.map(S),
      radius: l.radius_m,
      color: l.risk_only ? LAYER_COLORS.corridorRisk : LAYER_COLORS.corridor,
    })),
    fence: {
      floor: fence.polygon_enu.map(([x, y]) => S([x, y, fence.floor_z])),
      ceiling: fence.polygon_enu.map(([x, y]) => S([x, y, fence.ceiling_z])),
    },
    target: {
      label: t.facade_id ? `${t.ap_class} · ${t.facade_id}` : t.ap_class,
      outline: t.polygon_enu.map(S),
      center: S(t.center.enu),
      hover: S(t.hover.enu),
      entry: S(t.entry.enu),
      direction: dir(t.hover.enu, t.direction),
      headingDir: t.heading_deg === null ? null : dir(t.hover.enu, headingToEnu(t.heading_deg)),
    },
    trajectory: {
      points,
      segments: (tr?.segments ?? [])
        .filter((s) => s.phase !== "ground")
        .map((s) => ({ i0: s.i0, i1: s.i1, color: phaseColor(s.phase, s.failsafe), failsafe: s.failsafe })),
    },
  };
}

export type OverlayData = ReturnType<typeof overlayMessage>;
