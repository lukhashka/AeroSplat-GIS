// The takeoff-spot search and mission planning from the page (api_mission): what to send, how
// to read the answers. No React and no runtime imports, so `npm test` runs it under plain Node.
import type { MissionCreate, MissionRefusal, PlanningProfile, TakeoffParams, TakeoffQuery, Vec3 } from "../api/types";

// Server defaults of TakeoffParams; `home_search` always searches with these.
export const SEARCH_DEFAULTS = { max_distance_m: 150, n: 5, ground_tol_m: 1, max_checks: 40 } as const;
// The server's bounds of max_checks (api_mission.TakeoffParams).
export const MAX_CHECKS_LIMIT = 400;

/** Why columns were rejected (aerosplat.mission.takeoff.REASONS), each under its first failed test. */
export const REJECT_LABEL: Record<string, string> = {
  target_near_grid_edge: "ціль (кінець транзиту) ближче до краю сітки, ніж r + 2 вокселі",
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

/** A count typed by the operator (max_checks): blank is the server's default, else a whole number in 1..max. */
export function parseCount(text: string, max: number): { value: number | undefined } | { error: string } {
  const p = parseOptional(text);
  if ("error" in p || p.value === undefined) return p;
  return Number.isInteger(p.value) && p.value >= 1 && p.value <= max
    ? p
    : { error: `«${text.trim()}» — потрібне ціле число від 1 до ${max}` };
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
    q.ground_tol_m ?? SEARCH_DEFAULTS.ground_tol_m,
    q.max_checks ?? SEARCH_DEFAULTS.max_checks,
  ]);
}

/** Only the search's own parameters of a query (the server refuses anything else in home_from_search). */
export function searchParams(q: TakeoffParams): TakeoffParams {
  const out: TakeoffParams = {};
  for (const key of ["max_distance_m", "n", "ground_tol_m", "max_checks"] as const) {
    if (q[key] !== undefined) out[key] = q[key];
  }
  return out;
}

/** Candidate #rank of the search made with `search`, or the server's own search (`home_search`). */
export type HomeChoice = { kind: "candidate"; rank: number; enu: Vec3; search: TakeoffParams } | { kind: "search" };

/** POST /missions body: home from a chosen candidate, or from the server's own search — never both.
 *  A candidate goes as home_enu plus where it came from; the server runs that search again and
 *  refuses (409 takeoff_spot_changed) when #rank is no longer that spot. */
export function planBody(targetId: string, home: HomeChoice, opts: { run_id?: string; profile?: PlanningProfile } = {}): MissionCreate {
  const body: MissionCreate = { target_id: targetId };
  if (opts.run_id) body.run_id = opts.run_id;
  if (opts.profile && Object.keys(opts.profile).length) body.profile = opts.profile;
  if (home.kind === "candidate") {
    body.home_enu = [home.enu[0], home.enu[1], home.enu[2]];
    body.home_from_search = { rank: home.rank, ...searchParams(home.search) };
  } else body.home_search = true;
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
