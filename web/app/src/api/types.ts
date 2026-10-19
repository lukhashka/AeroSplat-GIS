// Mirrors of the pydantic models served by /api/v2 (aerosplat.store.models,
// aerosplat.jobs.models, aerosplat.server.api_v2). Keep in step with them.

export type SiteKind = "urban" | "depot" | "critical";
export type CaptureSource = "earthstudio" | "drone" | "handheld" | "internet";
export type RunStatus = "running" | "ok" | "failed";
export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";
export type JobKind = "pipeline" | "diagnostic";
export type Resource = "gpu" | "cpu";
export type StageState = "started" | "skipped" | "done" | "failed";

export interface GeoPoint {
  lat: number;
  lon: number;
  h: number;
}

export interface BBox {
  west: number;
  south: number;
  east: number;
  north: number;
}

export interface Project {
  id: string;
  name: string;
  description: string;
  created_at: string;
}

export interface Site {
  id: string;
  project_id: string;
  name: string;
  kind: SiteKind;
  origin: GeoPoint;
  bbox: BBox | null;
  taxonomy_packs: string[];
  created_at: string;
}

export interface Epoch {
  id: string;
  label: string;
  valid_from: string | null;
  valid_to: string | null;
  created_at: string;
}

export interface Capture {
  id: string;
  source: CaptureSource;
  license: string;
  author: string;
  attribution: string;
  captured_at: string | null;
  pose_prior_sigma_m: number | null;
  metadata_file: string | null;
  frame_prefix: string | null;
  notes: string;
  created_at: string;
}

export interface CaptureOut {
  capture: Capture;
  images: number;
  bytes: number;
  with_prior: number;
}

export interface ImageRec {
  name: string;
  sha256: string;
  bytes: number;
  width: number | null;
  height: number | null;
  prior: GeoPoint | null;
  prior_sigma_m: number | null;
  registration_status: string;
}

export interface Run {
  id: string;
  site_key: string;
  epoch_id: string;
  capture_ids: string[];
  status: RunStatus;
  earth_model: string;
  georeferenced: boolean;
  config: Record<string, unknown>;
  git_commit: string | null;
  products: Record<string, string>;
  created_at: string;
  finished_at: string | null;
  notes: string;
}

export interface RunBrief {
  id: string;
  epoch_id: string;
  status: RunStatus;
  earth_model: string;
  georeferenced: boolean;
  created_at: string;
  finished_at: string | null;
  products: string[];
}

export interface SiteSummary {
  site: Site;
  captures: number;
  images: number;
  runs: number;
  latest_run: RunBrief | null;
  scene: string | null;
  active_jobs: number;
}

export interface Job {
  id: string;
  kind: JobKind;
  resource: Resource;
  status: JobStatus;
  priority: number;
  site_key: string | null;
  title: string;
  payload: Record<string, unknown>;
  result: Record<string, unknown>;
  error: string | null;
  cancel_requested: boolean;
  worker_id: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface SiteDetail {
  site: Site;
  epochs: Epoch[];
  captures: CaptureOut[];
  runs: Run[];
  jobs: Job[];
  scene: string | null;
}

export interface StageEventData {
  stage: string;
  label: string;
  state: StageState;
  index: number;
  total: number;
  seconds?: number;
  error?: string;
  reason?: string;
}

export type JobEvent =
  | { seq: number; job_id: string; ts: string; type: "status"; data: { status: JobStatus; error?: string; cancel_requested?: boolean } }
  | { seq: number; job_id: string; ts: string; type: "stage"; data: StageEventData }
  | { seq: number; job_id: string; ts: string; type: "log"; data: { lines: string[] } }
  | { seq: number; job_id: string; ts: string; type: "result"; data: Record<string, unknown> };

export interface WorkerInfo {
  id: string;
  resource: Resource;
  host: string;
  pid: number;
  started_at: string;
  heartbeat_at: string;
  current_job: string | null;
  stopped_at: string | null;
  online: boolean;
}

export interface GpuInfo {
  name: string;
  memory_used_mb: number;
  memory_total_mb: number;
  utilization: number;
  temperature_c: number;
}

export interface SystemInfo {
  version: string;
  git_commit: string | null;
  projects_root: string;
  workers: WorkerInfo[];
  queue: Record<JobStatus, number>;
  gpu: GpuInfo[] | null;
}

export interface StageInfo {
  stage: string;
  status: string;
  seconds: number | null;
  error: string | null;
}

export interface RunDetail {
  run: Run;
  stages: StageInfo[];
  metrics: Record<string, unknown> | null;
  has_report: boolean;
  scene: string | null;
}

export interface PipelineInfo {
  stages: { name: string; label: string }[];
  config_schema: { properties: Record<string, { type?: string; default?: unknown; description?: string }> };
  defaults: Record<string, unknown>;
}

export interface RunRequest {
  epoch?: string;
  capture?: string | null;
  stages?: string[] | null;
  from_stage?: string | null;
  to_stage?: string | null;
  resume?: boolean;
  overrides?: Record<string, unknown>;
  priority?: number;
}

// ── render service (aerosplat/render/models.py) ─────────────────────────
export type RenderMode = "persp" | "ortho";
export type RenderOutput = "rgb" | "depth" | "alpha";

export interface RenderPosition {
  e: number;
  n: number;
  u?: number;
}

export interface RenderRequest {
  mode: RenderMode;
  position: RenderPosition;
  yaw_deg: number;
  pitch_deg: number;
  roll_deg: number;
  width: number;
  height: number;
  fov_deg: number;
  extent_m: number;
  output: RenderOutput;
}

export interface RenderMeta {
  scene: string;
  run: string | null;
  mode: RenderMode;
  frame: string;
  width: number;
  height: number;
  render_ms: number;
  gaussians: number;
  sh_degree: number | null;
  camera_enu: [number, number, number];
  camera_geo?: { lat: number; lon: number; h: number };
  scene_bounds_enu: { min: [number, number, number]; max: [number, number, number] };
  ground_grid?: { m_per_px: number; pixel00_enu: [number, number]; du_enu: [number, number]; dv_enu: [number, number] };
  depth?: { max_m: number };
}

export interface RenderStatus {
  online: boolean;
  device?: string;
  vram_allocated_mb?: number;
  idle_unload_s?: number;
  loaded?: { scene: string; gaussians: number; idle_s: number }[];
}

// ── semantics: facades, taxonomy, authored access points ───────────────
// (aerosplat/semantics/models.py, aerosplat/taxonomy, aerosplat/authored,
//  aerosplat/server/api_semantics.py). ACCESS_POINTS §2-§5.
export type SurfaceKind = "wall" | "roof_flat" | "roof_pitched" | "ground";
export type ApKind = "opening" | "pad" | "marker" | "constraint";
export type Lifecycle = "candidate" | "verified" | "rejected" | "stale";
export type HandoverMode = "hover_handoff" | "tether_lower" | "drop" | "land" | "pickup" | "enter";

export type Vec3 = [number, number, number];

/** How well the captures see a facade. Without coverage a detection is not evidence (§5.3). */
export interface Coverage {
  cell_m: number;
  cells: number;
  visible: number;
  covered: number;
  mean_views: number;
  max_gsd_cm: number;
  best_gsd_cm: number | null;
  median_angle_deg: number | null;
  occlusion_tested: boolean;
}

export interface Facade {
  id: string;
  building_id: string;
  kind: SurfaceKind;
  origin: Vec3;
  normal: Vec3;
  u_axis: Vec3;
  azimuth_deg: number;
  u_min: number;
  u_max: number;
  v_ground: number;
  v_max: number;
  plane_rms_m: number;
  n_points: number;
  coverage: Coverage | null;
}

export interface BuildingRec {
  id: string;
  footprint: [number, number][];
  area_m2: number;
  ground_z: number;
  roof_z: number;
  height_m: number;
  facade_ids: string[];
}

export interface TerrainModel {
  res_m: number;
  x0: number;
  y0: number;
  rows: number;
  cols: number;
  n_points: number;
  filled_cells: number;
}

/** `views[j][i]`: column i, row j, **row 0 at the bottom** of the wall. 0 = the splat invented it. */
export interface CoverageGrid {
  cell_m: number;
  nu: number;
  nv: number;
  u_min: number;
  v_min: number;
  max_gsd_cm: number;
  max_views: number;
  views: number[][];
}

export interface FacadeDetail {
  facade: Facade;
  building: BuildingRec | null;
  coverage_grid: CoverageGrid | null;
  run_id: string;
  epoch_id: string;
  features: number;
}

export interface SemanticsOut {
  run_id: string;
  epoch_id: string;
  finished_at: string | null;
  frame: string;
  terrain: TerrainModel;
  buildings: BuildingRec[];
  facades: Facade[];
  warnings: string[];
  params: Record<string, unknown>;
  features_by_facade: Record<string, number>;
  scene: string | null;
}

export interface TaxonomyAttribute {
  id: string;
  labels: { uk: string; en: string };
  values: string[];
  default: string | null;
  free_text: boolean;
  dynamic: boolean;
}

export interface TaxonomyClass {
  id: string;
  parent: string | null;
  kind: ApKind;
  pack: string;
  labels: { uk: string; en: string };
  orientation: "vertical" | "horizontal" | "any";
  size_prior_m: { width: [number, number]; height: [number, number] } | null;
  attributes: TaxonomyAttribute[];
  handover: {
    allowed: HandoverMode[];
    default: HandoverMode;
    standoff_m: number;
    min_clearance_m: number;
    approach_cone_deg: number;
  } | null;
  citygml: string | null;
}

export interface TaxonomyOut {
  version: string;
  packs: { pack: string; version: string }[];
  classes: TaxonomyClass[];
}

/** Pixel ↔ (u, v) of a facade orthophoto: the facade's answer to `ground_grid`. */
export interface FacadeGrid {
  facade_id: string;
  m_per_px: number;
  pixel00_uv: [number, number];
  du_uv: [number, number];
  dv_uv: [number, number];
  u_min: number;
  u_max: number;
  v_min: number;
  v_max: number;
  origin_enu: Vec3;
  u_axis: Vec3;
  normal: Vec3;
  v_ground: number;
  azimuth_deg: number;
}

export interface FacadeOrthoMeta extends RenderMeta {
  facade_grid: FacadeGrid;
  facade: {
    id: string;
    building_id: string;
    azimuth_deg: number;
    length_m: number;
    height_m: number;
    v_ground: number;
    u_min: number;
    u_max: number;
    v_max: number;
    plane_rms_m: number;
  };
  gsd_cm: number;
}

export interface OrthoRequest {
  gsd_cm?: number;
  margin_m?: number;
  standoff_m?: number;
  output?: RenderOutput;
  run_id?: string | null;
}

export interface ApGeometry {
  frame: string;
  polygon_uv: [number, number][];
  center: Vec3;
  normal: Vec3;
  u_axis: Vec3;
  width: number;
  height: number;
  u_center: number;
  v_center: number;
}

export interface ApAddress {
  floor_index: number | null;
  floor_label: string | null;
  column: number | null;
  ordinal_on_floor: number | null;
  floor_ambiguous: boolean;
  canonical: string | null;
}

export interface ApHost {
  facade_id: string;
  building_id: string;
  origin: Vec3;
  normal: Vec3;
  u_axis: Vec3;
  v_ground: number;
  azimuth_deg: number;
}

export interface ApProvenance {
  origin: "pipeline" | "manual";
  run_id: string | null;
  epoch_id: string | null;
  model: string | null;
  support_views: number | null;
  evidence: string[];
  created_by: string;
  created_at: string;
  edits: { by: string; at: string; what: string }[];
}

export interface AccessPoint {
  id: string;
  site_key: string;
  class: string;
  kind: ApKind;
  taxonomy: string;
  host: ApHost;
  geometry: ApGeometry;
  attributes: Record<string, string>;
  address: ApAddress | null;
  confidence: number | null;
  lifecycle: Lifecycle;
  provenance: ApProvenance;
  notes: string;
}

export interface AccessPointCreate {
  facade_id: string;
  class: string;
  polygon_uv: [number, number][];
  attributes?: Record<string, string>;
  lifecycle?: Lifecycle;
  notes?: string;
  by?: string;
}

export interface AccessPointPatch {
  class?: string;
  polygon_uv?: [number, number][];
  attributes?: Record<string, string>;
  lifecycle?: Lifecycle;
  notes?: string;
  by?: string;
}

export interface AddressRow {
  floor_index: number;
  floor_label: string;
  v_center: number;
  v_bottom: number;
  members: string[];
}

export interface AddressOut {
  facade_id: string;
  rows: AddressRow[];
  column_centres: number[];
  floor_height_m: number | null;
  ambiguous: boolean;
  reasons: string[];
  features: AccessPoint[];
}

export interface AuditEntry {
  at: string;
  by: string;
  op: string;
  feature_id: string;
  site_key: string;
  what: string;
  before: Record<string, unknown> | null;
  after: Record<string, unknown> | null;
}

// ── missions and their simulated flights (aerosplat.server.api_mission, api_sim) ──
export type MissionVerdict = "feasible" | "feasible_with_risks" | "infeasible";

export interface MissionFinding {
  code: string;
  severity: "blocking" | "risk";
  leg: string | null;
  detail: string;
  at: Vec3 | null;
}

export interface MissionSummary {
  id: string;
  target_id: string;
  ap_id: string;
  verdict: MissionVerdict;
  findings: MissionFinding[];
  created_at: string;
  created_by: string;
}

export interface GeoVertex {
  enu: Vec3;
  lon: number;
  lat: number;
  h_ellipsoid: number;
}

export interface GeoWaypoint extends GeoVertex {
  seq: number;
  role: string;
  command: number;
  alt_rel_m: number;
}

export interface GeoLeg {
  name: string;
  points_enu: Vec3[];
  lonlat: [number, number][];
  radius_m: number;
  sigma_m: number;
  min_clearance_m: number | null;
  excluded_surface: string | null;
  risk_only: boolean;
}

export interface MissionGeometry {
  id: string;
  site_key: string;
  verdict: MissionVerdict;
  findings: MissionFinding[];
  frame: "site_enu";
  origin: GeoPoint;
  home: GeoVertex;
  transit_z: number;
  waypoints: GeoWaypoint[];
  legs: GeoLeg[];
  fence: {
    polygon_enu: [number, number][];
    lonlat: [number, number][];
    floor_z: number;
    ceiling_z: number;
    alt_max_rel_m: number;
    radius_m: number;
  };
  target: {
    kind: string;
    ap_id: string;
    ap_class: string;
    facade_id: string | null;
    polygon_enu: Vec3[];
    lonlat: [number, number][];
    center: GeoVertex;
    normal: Vec3;
    direction: Vec3;
    hover: GeoVertex;
    entry: GeoVertex;
    heading_deg: number | null;
  };
  rally: GeoVertex[];
  r_drone_m: number;
  bbox_enu: [Vec3, Vec3];
  twin: { run_id: string; epoch_id: string; format: string; res_m: number; scene: string; legacy: boolean };
}

export interface SimVerdict {
  invariant: string;
  status: "pass" | "fail" | "n/a" | string;
  expected: string | null;
  detail: string;
}

export interface SimRunSummary {
  mission_id: string;
  scenario: string;
  seed: number | null;
  description: string;
  passed: boolean;
  outcome: string;
  reason: string | null;
  summary: string;
  verdicts: SimVerdict[];
  samples: number;
  mission_sim_s: number | null;
  firmware: string | null;
  has_trajectory: boolean;
  has_clearance: boolean;
  written_at: string;
}

export interface SimEvent {
  t: number;
  kind: string;
  detail: string;
  data: Record<string, unknown>;
}

export interface FlightSegment {
  phase: string;
  failsafe: boolean;
  mode: string;
  i0: number;
  i1: number;
}

export interface FlightTrajectory {
  mission_id: string;
  scenario: string;
  seed: number | null;
  origin: GeoPoint;
  origin_offset_m: number;
  home_enu: Vec3;
  mission_start_t: number | null;
  t: number[];
  enu: Vec3[];
  lonlat: [number, number][];
  alt_rel_m: number[];
  speed_mps: number[];
  yaw_deg: number[];
  mode: string[];
  armed: boolean[];
  seq: number[];
  phase: string[];
  failsafe: boolean[];
  segments: FlightSegment[];
  phases: string[];
  clearance: {
    clearance_m: (number | null)[];
    target_distance_m: (number | null)[];
    required_m: number;
    cap_m: number | null;
  } | null;
  events: SimEvent[];
}

/** report.json of a run (aerosplat.sim.report.report_dict); only what the page reads. */
export interface SimReport {
  passed: boolean;
  summary: string;
  outcome: string;
  reason: string | null;
  detail: string;
  scenario: { scenario: string; description: string; vehicle: string; faults: unknown[] };
  verdicts: {
    invariant: string;
    status: string;
    detail: string;
    expected: string | null;
    metrics: Record<string, unknown>;
  }[];
  warnings: string[];
  firmware: { tag?: string; version?: string; git?: string };
  timing: Record<string, number>;
  faults: { fault: string; reason: string; on: number | null; off: number | null; note: string }[];
}

// ── targets, takeoff spots and planning (aerosplat.mission.models, api_mission) ──
export type TargetState = "draft" | "confirmed" | "revoked";

export interface SigmaTerm {
  value_m: number;
  source: string;
}

export interface MissionTarget {
  id: string;
  site_key: string;
  ap_id: string;
  ap_class: string;
  ap_kind: string;
  taxonomy: string;
  handover: {
    mode: string;
    standoff_m: number;
    min_clearance_m: number;
    approach_cone_deg: number;
    entry_distance_m: number | null;
    hover_dh_m: number;
    max_sigma_m: number;
    source: string;
    overridden: string[];
  };
  sigma: { ap: SigmaTerm; georef: SigmaTerm };
  state: TargetState;
  created_by: string;
  created_at: string;
  confirmed_by: string | null;
  confirmed_at: string | null;
  revoked_by: string | null;
  revoked_at: string | null;
  notes: string;
}

/** The vehicle and the margins; only the fields sent, the server's defaults fill the rest. */
export interface PlanningProfile {
  r_drone_m?: number;
  sigma_gnss_m?: number;
  k_sigma?: number;
  transit_margin_m?: number;
  max_alt_rel_m?: number;
  wire_height_m?: number;
  surface_relief_m?: number;
}

/** How far and how hard the takeoff search looks; unset = the server's default. */
export interface TakeoffParams {
  max_distance_m?: number; // default 150
  n?: number; // default 5
  ground_tol_m?: number; // default 1
  max_checks?: number; // default 40, 1..400: exact checks before "not confirmed, raise max_checks"
}

export interface TakeoffQuery extends TakeoffParams {
  run_id?: string;
  profile?: PlanningProfile;
}

/** `home_enu` is candidate #rank of the search with these parameters (on the plan's run and profile). */
export interface HomeFromSearch extends TakeoffParams {
  rank: number;
}

/** A place to arm and take off from, checked as a plan's home and takeoff would be. */
export interface TakeoffCandidate {
  rank: number;
  enu: Vec3; // on the measured ground
  distance_m: number; // horizontally, to the target
  open_radius_m: number;
  transit_z: number | null; // null for `best_open`, which was never checked
  climb_m: number | null;
  above_dtm_m: number;
  notes: string[];
}

/** Every candidate found, or an honest account of why there is none (`message`). */
export interface TakeoffSearch {
  target: Vec3; // where the transit ends (E, or the pad)
  radius_m: number; // takeoff corridor r = r_drone + m + k·σ_transit
  relief_m: number;
  sigma_transit_m: number;
  max_distance_m: number;
  spacing_m: number;
  ground_tol_m: number;
  grid_limited: boolean;
  searched_columns: number;
  rejected: Record<string, number>; // first failed test -> columns
  checked: number;
  candidates: TakeoffCandidate[];
  best_open: TakeoffCandidate | null;
  sigma_gnss_would_fit_m: number | null;
  message: string;
}

export interface MissionCreate {
  target_id: string;
  run_id?: string;
  home_enu?: Vec3; // default: the Site origin, on the surface
  home_search?: boolean; // instead of home_enu: the nearest spot of the takeoff search
  // with home_enu: the server runs the search again; 409 takeoff_spot_changed when #rank is no longer there
  home_from_search?: HomeFromSearch;
  profile?: PlanningProfile;
  altitude_frame?: "relative_home";
  by?: string;
}

/** POST /missions answers with the whole plan; this is the part the page reads. */
export interface MissionPlan {
  id: string;
  site_key: string;
  verdict: MissionVerdict;
  findings: MissionFinding[];
  home: { enu: Vec3; lat: number; lon: number; h_ellipsoid: number; source: string };
  transit_z: number;
  occupancy: { run_id: string; epoch_id: string; format: string; res_m: number };
}

/** The `detail` of a 409 from the planner: why there is no plan, every reason named. */
export interface MissionRefusal {
  message: string;
  reasons: MissionFinding[];
}
