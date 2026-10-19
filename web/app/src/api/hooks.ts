import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, api2, upload } from "./client";
import type {
  AccessPoint,
  AccessPointCreate,
  AccessPointPatch,
  AddressOut,
  AuditEntry,
  CarriedOut,
  PadsOut,
  ReanchorOut,
  RegularizeOut,
  GridfitOut,
  GridfitRequest,
  CaptureOut,
  CaptureSource,
  ImageRec,
  Job,
  JobStatus,
  PipelineInfo,
  RenderStatus,
  Project,
  RunDetail,
  RunRequest,
  Site,
  SiteDetail,
  SiteKind,
  SiteSummary,
  SystemInfo,
  GeoPoint,
  FacadeDetail,
  FlightTrajectory,
  MissionCreate,
  MissionGeometry,
  MissionPlan,
  MissionSummary,
  MissionTarget,
  SemanticsOut,
  SimReport,
  SimRunSummary,
  TakeoffQuery,
  TakeoffSearch,
  TaxonomyOut,
} from "./types";

export const keys = {
  system: ["system"] as const,
  projects: ["projects"] as const,
  sites: ["sites"] as const,
  site: (p: string, s: string) => ["site", p, s] as const,
  images: (p: string, s: string, c: string) => ["images", p, s, c] as const,
  run: (p: string, s: string, e: string, r: string) => ["run", p, s, e, r] as const,
  jobs: (params: object) => ["jobs", params] as const,
  job: (id: string) => ["job", id] as const,
  pipeline: ["pipeline"] as const,
};

export const useSystem = () =>
  useQuery({ queryKey: keys.system, queryFn: () => api.get<SystemInfo>("/system"), refetchInterval: 3000 });

export const useRenderStatus = () =>
  useQuery({ queryKey: ["render-status"], queryFn: () => api.get<RenderStatus>("/render/status"), refetchInterval: 4000 });

export const useProjects = () => useQuery({ queryKey: keys.projects, queryFn: () => api.get<Project[]>("/projects") });

export const useSites = () =>
  useQuery({ queryKey: keys.sites, queryFn: () => api.get<SiteSummary[]>("/sites"), refetchInterval: 5000 });

export const useSite = (project: string, site: string) =>
  useQuery({
    queryKey: keys.site(project, site),
    queryFn: () => api.get<SiteDetail>(`/sites/${project}/${site}`),
    refetchInterval: (q) =>
      q.state.data?.jobs.some((j) => j.status === "queued" || j.status === "running") ? 2000 : 10000,
  });

export const useCaptureImages = (project: string, site: string, capture: string | null) =>
  useQuery({
    queryKey: keys.images(project, site, capture ?? ""),
    queryFn: () => api.get<ImageRec[]>(`/sites/${project}/${site}/captures/${capture}/images`),
    enabled: !!capture,
    staleTime: 60_000,
  });

export const useRunDetail = (project: string, site: string, epoch: string, run: string | null) =>
  useQuery({
    queryKey: keys.run(project, site, epoch, run ?? ""),
    queryFn: () => api.get<RunDetail>(`/sites/${project}/${site}/runs/${epoch}/${run}`),
    enabled: !!run,
  });

export const useJobs = (params: { status?: JobStatus[]; site?: string; limit?: number } = {}) =>
  useQuery({
    queryKey: keys.jobs(params),
    queryFn: () => {
      const q = new URLSearchParams();
      params.status?.forEach((s) => q.append("status", s));
      if (params.site) q.set("site", params.site);
      q.set("limit", String(params.limit ?? 50));
      return api.get<Job[]>(`/jobs?${q}`);
    },
    refetchInterval: 3000,
  });

export const usePipeline = () =>
  useQuery({ queryKey: keys.pipeline, queryFn: () => api.get<PipelineInfo>("/pipeline"), staleTime: Infinity });

// ── mutations ────────────────────────────────────────────────────────────
export function useCreateSite() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (v: {
      project: string;
      newProject: boolean;
      id: string;
      name: string;
      kind: SiteKind;
      origin: GeoPoint;
    }) => {
      if (v.newProject) await api.post<Project>("/projects", { id: v.project, name: v.project });
      return api.post<Site>(`/projects/${v.project}/sites`, {
        id: v.id,
        name: v.name,
        kind: v.kind,
        origin: v.origin,
      });
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: keys.sites });
      qc.invalidateQueries({ queryKey: keys.projects });
    },
  });
}

export function useUploadCapture(project: string, site: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (v: {
      captureId: string;
      source: CaptureSource;
      license: string;
      author: string;
      files: File[];
      onProgress: (f: number) => void;
    }) => {
      const form = new FormData();
      form.set("capture_id", v.captureId);
      form.set("source", v.source);
      form.set("license", v.license);
      form.set("author", v.author);
      for (const f of v.files) form.append("files", f, f.name);
      return upload<CaptureOut>(`/sites/${project}/${site}/captures`, form, v.onProgress);
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: keys.site(project, site) });
      qc.invalidateQueries({ queryKey: keys.sites });
    },
  });
}

export function useStartRun(project: string, site: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: RunRequest) => api.post<Job>(`/sites/${project}/${site}/runs`, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: keys.site(project, site) });
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}

export function useCancelJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.post<Job>(`/jobs/${id}/cancel`),
    onSuccess: (job) => {
      qc.setQueryData(keys.job(job.id), job);
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}

export function useDiagnostic() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (resource: "gpu" | "cpu") =>
      api.post<Job>("/jobs/diagnostic", { resource, stages: 4, seconds_per_stage: 1.5, lines_per_stage: 6 }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs"] }),
  });
}

export const refreshScenes = () => api.post<{ scenes: string[]; changed: string[] }>("/scenes/refresh");

// ── semantics, taxonomy and authored access points (ACCESS_POINTS §3-§5) ──
export const semanticKeys = {
  semantics: (p: string, s: string, run: string | null = null) => ["semantics", p, s, run ?? ""] as const,
  facade: (p: string, s: string, f: string, run: string | null = null) =>
    ["facade", p, s, f, run ?? ""] as const,
  taxonomy: (p: string, s: string) => ["taxonomy", p, s] as const,
  features: (p: string, s: string, f: string | null) => ["access-points", p, s, f ?? ""] as const,
  audit: (p: string, s: string) => ["audit", p, s] as const,
};

// `runId` null: the newest run with semantics.
export const useSemantics = (project: string, site: string, runId: string | null = null) =>
  useQuery({
    queryKey: semanticKeys.semantics(project, site, runId),
    queryFn: () =>
      api.get<SemanticsOut>(
        `/sites/${project}/${site}/semantics` + (runId ? `?run_id=${encodeURIComponent(runId)}` : ""),
      ),
    retry: false,
    staleTime: 30_000,
  });

// `runId` null: the newest run with semantics. An older run keeps its own facade ids, which
// is where features drawn before a newer run was computed still live.
export const useFacade = (project: string, site: string, facadeId: string | null,
                          runId: string | null = null) =>
  useQuery({
    queryKey: semanticKeys.facade(project, site, facadeId ?? "", runId),
    queryFn: () =>
      api.get<FacadeDetail>(
        `/sites/${project}/${site}/facades/${facadeId}` +
          (runId ? `?run_id=${encodeURIComponent(runId)}` : ""),
      ),
    enabled: !!facadeId,
    staleTime: 30_000,
  });

export const useTaxonomy = (project: string, site: string) =>
  useQuery({
    queryKey: semanticKeys.taxonomy(project, site),
    queryFn: () => api.get<TaxonomyOut>(`/sites/${project}/${site}/taxonomy`),
    staleTime: 5 * 60_000,
  });

// With a facade and its run: only the features native to that run's facade (same frame).
// Facade ids are per run, so the id alone can name another wall in another run (§5.9).
export const useAccessPoints = (project: string, site: string, facadeId: string | null,
                                runId: string | null = null, enabled = true) =>
  useQuery({
    queryKey: [...semanticKeys.features(project, site, facadeId), runId ?? ""],
    queryFn: () => {
      const q = new URLSearchParams();
      if (facadeId) q.set("facade", facadeId);
      if (facadeId && runId) q.set("run_id", runId);
      const qs = q.toString();
      return api.get<AccessPoint[]>(`/sites/${project}/${site}/access-points${qs ? `?${qs}` : ""}`);
    },
    enabled,
    staleTime: 2000,
  });

/** Features drawn in other runs that land on this facade of `runId` (shown, carried on request). */
export const useCarried = (project: string, site: string, facadeId: string | null, runId: string | null) =>
  useQuery({
    queryKey: ["carried", project, site, facadeId ?? "", runId ?? ""],
    queryFn: () =>
      api.get<CarriedOut[]>(
        `/sites/${project}/${site}/facades/${facadeId}/carried` +
          (runId ? `?run_id=${encodeURIComponent(runId)}` : ""),
      ),
    enabled: !!facadeId && !!runId,
    staleTime: 5000,
  });

export const useAudit = (project: string, site: string, limit = 50) =>
  useQuery({
    queryKey: semanticKeys.audit(project, site),
    queryFn: () => api.get<AuditEntry[]>(`/sites/${project}/${site}/audit?limit=${limit}`),
    staleTime: 2000,
  });

/** Pad zones on the roofs and in the yards of buildings (§5.7): candidates only, never verified. */
export function usePadCandidates(project: string, site: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { building_ids: string[]; run_id: string | null; dry_run?: boolean; by?: string }) =>
      api.post<PadsOut>(`/sites/${project}/${site}/pads/candidates`, body),
    onSuccess: (r) => {
      if (r.dry_run) return;
      qc.invalidateQueries({ queryKey: ["access-points", project, site] });
      qc.invalidateQueries({ queryKey: semanticKeys.audit(project, site) });
    },
  });
}

/** Create, change and delete authored features. Nothing here is ever automatic (ADR-0009). */
export function useAccessPointEdits(project: string, site: string, facadeId: string | null) {
  const qc = useQueryClient();
  const base = `/sites/${project}/${site}/access-points`;
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["access-points", project, site] });
    qc.invalidateQueries({ queryKey: ["facade", project, site] });
    qc.invalidateQueries({ queryKey: ["semantics", project, site] });
    qc.invalidateQueries({ queryKey: ["carried", project, site] });
    qc.invalidateQueries({ queryKey: semanticKeys.audit(project, site) });
  };
  return {
    create: useMutation({
      mutationFn: (body: AccessPointCreate) => api.post<AccessPoint>(base, body),
      onSuccess: refresh,
    }),
    update: useMutation({
      mutationFn: ({ id, ...body }: AccessPointPatch & { id: string }) =>
        api2.patch<AccessPoint>(`${base}/${id}`, body),
      onSuccess: refresh,
    }),
    remove: useMutation({
      mutationFn: (id: string) => api2.del<AccessPoint>(`${base}/${id}`),
      onSuccess: refresh,
    }),
    address: useMutation({
      mutationFn: (body: { floor_offset?: number; by?: string; run_id?: string | null }) =>
        api.post<AddressOut>(
          `/sites/${project}/${site}/facades/${facadeId}/address`,
          body,
        ),
      onSuccess: refresh,
    }),
    // Snap the openings' edges to the facade's rows and columns within a tolerance (§5.6
    // item 3). A dry run first shows what would move; nothing moves further than tol_m.
    regularize: useMutation({
      mutationFn: (body: { run_id: string | null; tol_m?: number; dry_run?: boolean; by?: string }) =>
        api.post<RegularizeOut>(`/sites/${project}/${site}/facades/${facadeId}/regularize`, body),
      onSuccess: (r) => {
        if (!r.dry_run) refresh();
      },
    }),
    // Carry a feature of another run onto this facade -- only when the operator asks, and
    // only when it lands on exactly this facade (§5.9). The address is recomputed after.
    reanchor: useMutation({
      mutationFn: ({ id, ...body }: { id: string; facade_id: string; run_id: string | null; by?: string }) =>
        api.post<ReanchorOut>(`${base}/${id}/reanchor`, body),
      onSuccess: refresh,
    }),
    // Window candidates from the facade's periodic grid: stored as `candidate`,
    // never verified (ADR-0009); the operator reviews them with V / X.
    gridfit: useMutation({
      mutationFn: (body: GridfitRequest) =>
        api.post<GridfitOut>(
          `/sites/${project}/${site}/facades/${facadeId}/candidates/gridfit`,
          body,
        ),
      onSuccess: refresh,
    }),
  };
}

// ── missions and simulated flights (api_mission, api_sim) ────────────────
export const flightKeys = {
  missions: (p: string, s: string) => ["missions", p, s] as const,
  targets: (p: string, s: string) => ["targets", p, s] as const,
  sims: (p: string, s: string) => ["sims", p, s] as const,
  geometry: (p: string, s: string, m: string) => ["mission-geometry", p, s, m] as const,
  report: (p: string, s: string, m: string, run: string, seed: number | null) =>
    ["sim-report", p, s, m, run, seed ?? ""] as const,
  trajectory: (p: string, s: string, m: string, run: string, seed: number | null) =>
    ["sim-trajectory", p, s, m, run, seed ?? ""] as const,
};

const seedQuery = (seed: number | null) => (seed === null ? "" : `?seed=${seed}`);

export const useMissions = (project: string, site: string) =>
  useQuery({
    queryKey: flightKeys.missions(project, site),
    queryFn: () => api.get<MissionSummary[]>(`/sites/${project}/${site}/missions`),
    staleTime: 30_000,
  });

export const useSiteSims = (project: string, site: string) =>
  useQuery({
    queryKey: flightKeys.sims(project, site),
    queryFn: () => api.get<SimRunSummary[]>(`/sites/${project}/${site}/sims`),
    staleTime: 30_000,
  });

export const useMissionGeometry = (project: string, site: string, mission: string | null) =>
  useQuery({
    queryKey: flightKeys.geometry(project, site, mission ?? ""),
    queryFn: () => api.get<MissionGeometry>(`/sites/${project}/${site}/missions/${mission}/geometry`),
    enabled: !!mission,
    staleTime: Infinity,
  });

export const useSimReport = (project: string, site: string, mission: string | null, run: string | null, seed: number | null) =>
  useQuery({
    queryKey: flightKeys.report(project, site, mission ?? "", run ?? "", seed),
    queryFn: () =>
      api.get<SimReport>(`/sites/${project}/${site}/missions/${mission}/sims/${run}/report${seedQuery(seed)}`),
    enabled: !!mission && !!run,
    staleTime: Infinity,
  });

export const useSimTrajectory = (
  project: string,
  site: string,
  mission: string | null,
  run: string | null,
  seed: number | null,
) =>
  useQuery({
    queryKey: flightKeys.trajectory(project, site, mission ?? "", run ?? "", seed),
    queryFn: () =>
      api.get<FlightTrajectory>(`/sites/${project}/${site}/missions/${mission}/sims/${run}/trajectory${seedQuery(seed)}`),
    enabled: !!mission && !!run,
    staleTime: Infinity,
  });

export const useTargets = (project: string, site: string) =>
  useQuery({
    queryKey: flightKeys.targets(project, site),
    queryFn: () => api.get<MissionTarget[]>(`/sites/${project}/${site}/targets`),
    staleTime: 10_000,
  });

/** Ranked takeoff spots near a target. Read-only on the server, but an explicit operator action. */
export const useTakeoffSearch = (project: string, site: string) =>
  useMutation({
    mutationFn: ({ targetId, query }: { targetId: string; query: TakeoffQuery }) =>
      api.post<TakeoffSearch>(`/sites/${project}/${site}/targets/${targetId}/takeoff`, query),
  });

/** Plan a mission (201 whatever the verdict; 409 with named reasons when there is no plan). */
export function usePlanMission(project: string, site: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: MissionCreate) => api.post<MissionPlan>(`/sites/${project}/${site}/missions`, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: flightKeys.missions(project, site) }),
  });
}
