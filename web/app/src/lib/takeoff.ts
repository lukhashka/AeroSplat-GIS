// The takeoff-spot search and mission planning from the page (api_mission): what to send, how
// to read the answers. No React and no runtime imports, so `npm test` runs it under plain Node.
import type { MissionCreate, MissionRefusal, PlanningProfile, TakeoffQuery, Vec3 } from "../api/types";

// Server defaults of TakeoffQuery; `home_search` always searches with these.
export const SEARCH_DEFAULTS = { max_distance_m: 150, n: 5 } as const;

/** Why columns were rejected (aerosplat.mission.takeoff.REASONS), each under its first failed test. */
export const REJECT_LABEL: Record<string, string> = {
  near_grid_edge: "біля краю сітки",
  not_measured: "поверхню не виміряно",
  building: "у контурі будівлі",
  not_ground: "не земля (далі від DTM, ніж допуск)",
  obstacle_within_radius: "перешкода ближче за r",
  home_check: "не пройшли перевірку дому",
  transit_blocked: "немає висоти транзиту",
  transit_too_high: "транзит вище стелі профілю",
  takeoff_leg: "перешкода на зльоті",
  transit_leg: "перешкода на транзиті",
  not_checked: "не перевірено (ліміт точних перевірок)",
};

/** Non-zero rejections in the server's order (the order of REASONS). */
export function rejectedRows(rejected: Record<string, number>): { code: string; label: string; count: number }[] {
  return Object.entries(rejected)
    .filter(([, count]) => count > 0)
    .map(([code, count]) => ({ code, label: REJECT_LABEL[code] ?? code, count }));
}

/** A number typed by the operator: blank is "the server's default", anything else must parse. */
export function parseOptional(text: string): { value: number | undefined } | { error: string } {
  const t = text.trim().replace(",", ".");
  if (!t) return { value: undefined };
  const v = Number(t);
  return Number.isFinite(v) ? { value: v } : { error: `«${text.trim()}» — не число` };
}

/** The profile from the form: only what was typed; an unparsable field is an error, never a default. */
export function profileFromForm(form: { sigma_gnss_m: string; r_drone_m: string }):
  | { profile: PlanningProfile | undefined }
  | { error: string } {
  const profile: PlanningProfile = {};
  for (const key of ["sigma_gnss_m", "r_drone_m"] as const) {
    const p = parseOptional(form[key]);
    if ("error" in p) return { error: `${key}: ${p.error}` };
    if (p.value !== undefined) profile[key] = p.value;
  }
  return { profile: Object.keys(profile).length ? profile : undefined };
}

/** Identity of a search: a result found with other parameters must not be planned with these. */
export function searchKey(targetId: string, q: TakeoffQuery): string {
  return JSON.stringify([
    targetId,
    q.run_id ?? "",
    q.profile ?? {},
    q.max_distance_m ?? SEARCH_DEFAULTS.max_distance_m,
    q.n ?? SEARCH_DEFAULTS.n,
  ]);
}

export type HomeChoice = { kind: "candidate"; enu: Vec3 } | { kind: "search" };

/** POST /missions body: home from a chosen candidate, or from the server's own search — never both. */
export function planBody(targetId: string, home: HomeChoice, opts: { run_id?: string; profile?: PlanningProfile } = {}): MissionCreate {
  const body: MissionCreate = { target_id: targetId };
  if (opts.run_id) body.run_id = opts.run_id;
  if (opts.profile && Object.keys(opts.profile).length) body.profile = opts.profile;
  if (home.kind === "candidate") body.home_enu = [home.enu[0], home.enu[1], home.enu[2]];
  else body.home_search = true;
  return body;
}

/** The planner's refusal (409 `{message, reasons}`) inside an ApiError, else null. */
export function refusalOf(err: unknown): MissionRefusal | null {
  if (!err || typeof err !== "object" || !("detail" in err)) return null;
  const d = (err as { detail: unknown }).detail;
  if (!d || typeof d !== "object") return null;
  const { message, reasons } = d as Partial<MissionRefusal>;
  if (typeof message !== "string" || !Array.isArray(reasons)) return null;
  return { message, reasons };
}

/** A closed ring of `n` segments around (cx, cy), radius r, in Site ENU metres. */
export function circleEnu(cx: number, cy: number, r: number, n = 72): [number, number][] {
  const ring: [number, number][] = [];
  for (let k = 0; k < n; k++) {
    const a = (2 * Math.PI * k) / n;
    ring.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
  }
  ring.push(ring[0]!);
  return ring;
}

export const fmtEnu = (p: readonly number[]) => `${p[0]!.toFixed(1)}, ${p[1]!.toFixed(1)}`;
