import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, api2, upload } from "./client";
import type {
  AccessPoint,
  AccessPointCreate,
  AccessPointPatch,
  AddressOut,
  AuditEntry,
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
  SemanticsOut,
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
  semantics: (p: string, s: string) => ["semantics", p, s] as const,
  facade: (p: string, s: string, f: string) => ["facade", p, s, f] as const,
  taxonomy: (p: string, s: string) => ["taxonomy", p, s] as const,
  features: (p: string, s: string, f: string | null) => ["access-points", p, s, f ?? ""] as const,
  audit: (p: string, s: string) => ["audit", p, s] as const,
};

export const useSemantics = (project: string, site: string) =>
  useQuery({
    queryKey: semanticKeys.semantics(project, site),
    queryFn: () => api.get<SemanticsOut>(`/sites/${project}/${site}/semantics`),
    retry: false,
    staleTime: 30_000,
  });

export const useFacade = (project: string, site: string, facadeId: string | null) =>
  useQuery({
    queryKey: semanticKeys.facade(project, site, facadeId ?? ""),
    queryFn: () => api.get<FacadeDetail>(`/sites/${project}/${site}/facades/${facadeId}`),
    enabled: !!facadeId,
    staleTime: 30_000,
  });

export const useTaxonomy = (project: string, site: string) =>
  useQuery({
    queryKey: semanticKeys.taxonomy(project, site),
    queryFn: () => api.get<TaxonomyOut>(`/sites/${project}/${site}/taxonomy`),
    staleTime: 5 * 60_000,
  });

export const useAccessPoints = (project: string, site: string, facadeId: string | null) =>
  useQuery({
    queryKey: semanticKeys.features(project, site, facadeId),
    queryFn: () => {
      const q = facadeId ? `?facade=${encodeURIComponent(facadeId)}` : "";
      return api.get<AccessPoint[]>(`/sites/${project}/${site}/access-points${q}`);
    },
    staleTime: 2000,
  });

export const useAudit = (project: string, site: string, limit = 50) =>
  useQuery({
    queryKey: semanticKeys.audit(project, site),
    queryFn: () => api.get<AuditEntry[]>(`/sites/${project}/${site}/audit?limit=${limit}`),
    staleTime: 2000,
  });

/** Create, change and delete authored features. Nothing here is ever automatic (ADR-0009). */
export function useAccessPointEdits(project: string, site: string, facadeId: string | null) {
  const qc = useQueryClient();
  const base = `/sites/${project}/${site}/access-points`;
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["access-points", project, site] });
    qc.invalidateQueries({ queryKey: ["facade", project, site] });
    qc.invalidateQueries({ queryKey: semanticKeys.semantics(project, site) });
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
      mutationFn: (body: { floor_offset?: number; by?: string }) =>
        api.post<AddressOut>(
          `/sites/${project}/${site}/facades/${facadeId}/address`,
          body,
        ),
      onSuccess: refresh,
    }),
  };
}
