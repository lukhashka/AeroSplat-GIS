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
