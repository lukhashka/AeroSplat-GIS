import type { RenderMeta, RenderRequest } from "./types";

export const API = "/api/v2";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: unknown,
  ) {
    super(formatDetail(detail) || `HTTP ${status}`);
  }
}

function formatDetail(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    // FastAPI / pydantic validation errors
    return detail
      .map((d) => {
        if (d && typeof d === "object" && "msg" in d) {
          const loc = "loc" in d && Array.isArray(d.loc) ? d.loc.filter((x: unknown) => x !== "body").join(".") : "";
          return loc ? `${loc}: ${String(d.msg)}` : String(d.msg);
        }
        return JSON.stringify(d);
      })
      .join("; ");
  }
  return detail ? JSON.stringify(detail) : "";
}

async function parse<T>(res: Response): Promise<T> {
  const text = await res.text();
  const body = text ? JSON.parse(text) : null;
  if (!res.ok) throw new ApiError(res.status, body?.detail ?? body?.error ?? body);
  return body as T;
}

export const api = {
  get: <T>(path: string) => fetch(API + path).then((r) => parse<T>(r)),
  post: <T>(path: string, body?: unknown) =>
    fetch(API + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    }).then((r) => parse<T>(r)),
};

/** Multipart upload with progress (fetch has no upload progress events). */
export function upload<T>(path: string, form: FormData, onProgress?: (fraction: number) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", API + path);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = xhr.responseText ? JSON.parse(xhr.responseText) : null;
      } catch {
        body = xhr.responseText;
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as T);
      else {
        const detail = body && typeof body === "object" && "detail" in body ? (body as { detail: unknown }).detail : body;
        reject(new ApiError(xhr.status, detail));
      }
    };
    xhr.onerror = () => reject(new ApiError(0, "network error"));
    xhr.send(form);
  });
}

export const thumbUrl = (project: string, site: string, capture: string, name: string, size = 256) =>
  `${API}/sites/${project}/${site}/captures/${capture}/images/${encodeURIComponent(name)}/thumb?size=${size}`;

export const viewerUrl = (scene: string, mode: "splat" | "dense" = "splat") =>
  `/web/?dataset=${encodeURIComponent(scene)}&mode=${mode}`;

export function wsUrl(path: string): string {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}${API}${path}`;
}

/** POST a render request; resolves to the PNG as an object URL plus the camera/georeference the server used. */
export async function renderSite(
  project: string,
  site: string,
  body: RenderRequest,
  signal?: AbortSignal,
): Promise<{ url: string; meta: RenderMeta }> {
  const res = await fetch(`${API}/sites/${project}/${site}/render`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    const text = await res.text();
    let detail: unknown = text;
    try {
      detail = JSON.parse(text).detail ?? text;
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, detail);
  }
  const meta = JSON.parse(res.headers.get("X-Render-Meta") ?? "{}") as RenderMeta;
  return { url: URL.createObjectURL(await res.blob()), meta };
}
