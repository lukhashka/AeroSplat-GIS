const rtf = new Intl.RelativeTimeFormat("uk", { numeric: "auto" });

export function timeAgo(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return "—";
  const diff = (new Date(iso).getTime() - now) / 1000;
  const abs = Math.abs(diff);
  if (abs < 45) return rtf.format(Math.round(diff), "second");
  if (abs < 2700) return rtf.format(Math.round(diff / 60), "minute");
  if (abs < 64800) return rtf.format(Math.round(diff / 3600), "hour");
  return rtf.format(Math.round(diff / 86400), "day");
}

export function dateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleString("uk-UA", { dateStyle: "short", timeStyle: "medium" });
}

export function duration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  if (seconds < 1) return `${Math.round(seconds * 1000)} мс`;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} с`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  if (m < 60) return `${m} хв ${s.toString().padStart(2, "0")} с`;
  return `${Math.floor(m / 60)} год ${(m % 60).toString().padStart(2, "0")} хв`;
}

export function between(a: string | null | undefined, b: string | null | undefined): number | null {
  if (!a) return null;
  const end = b ? new Date(b).getTime() : Date.now();
  return (end - new Date(a).getTime()) / 1000;
}

export function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(v < 10 ? 1 : 0)} ${units[i]}`;
}

export function coord(lat: number, lon: number): string {
  const ns = lat >= 0 ? "N" : "S";
  const ew = lon >= 0 ? "E" : "W";
  return `${Math.abs(lat).toFixed(5)}° ${ns}, ${Math.abs(lon).toFixed(5)}° ${ew}`;
}

/** Directory-safe id like the catalog's Slug: lowercase, digits, . _ - */
export function slugify(text: string): string {
  return text
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^\w\s.-]/g, "")
    .trim()
    .replace(/[\s_]+/g, "-")
    .replace(/^[^a-z0-9]+/, "")
    .slice(0, 64);
}

export const SLUG_RE = /^[a-z0-9][a-z0-9_.-]{0,63}$/;
