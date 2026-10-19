// Frames of the flight view (docs/ARCHITECTURE.md, "Місія / проліт"):
//
//   WGS84 (lat, lon, h ellipsoidal)  ⇄  Site ENU (x east, y north, z up, metres from the Site
//   origin)  →  served (Site ENU of the *scene* minus scene-meta.center)  →  viewer world
//   (sceneRoot: up-axis rotation, then uniform scale; WebGL is Y-up).
//
// The server's EnuFrame (pyproj, aerosplat.geo.crs) is the reference for the first arrow; the
// API already gives every point both ways. This module is needed only where the scene under a
// mission has another origin than the mission's Site, and to state the viewer transform once,
// in one place, with a test (web/app/tests/frames.test.ts).
//
// No imports: the test runs this file directly under Node's type stripping.

export type Vec3 = [number, number, number];

export interface Geodetic {
  lat: number;
  lon: number;
  h: number;
}

export interface UpAxisRotation {
  axis: "x" | "y" | "z";
  degrees: number;
}

/** The part of /api/v1/scene-meta that places a scene. */
export interface SceneFrame {
  center: number[];
  scene_scale: number;
  up_axis_rotation: UpAxisRotation | null;
  metric: { origin: Geodetic } | null;
}

const A = 6378137.0; // WGS84
const F = 1 / 298.257223563;
const E2 = F * (2 - F);
const RAD = Math.PI / 180;

export function geodeticToEcef(g: Geodetic): Vec3 {
  const lat = g.lat * RAD;
  const lon = g.lon * RAD;
  const s = Math.sin(lat);
  const n = A / Math.sqrt(1 - E2 * s * s);
  return [
    (n + g.h) * Math.cos(lat) * Math.cos(lon),
    (n + g.h) * Math.cos(lat) * Math.sin(lon),
    (n * (1 - E2) + g.h) * s,
  ];
}

export function ecefToGeodetic(p: Vec3): Geodetic {
  const [x, y, z] = p;
  const lon = Math.atan2(y, x);
  const r = Math.hypot(x, y);
  // Fixed-point on the latitude; converges to < 1e-12 rad in a handful of steps at any height
  // a drone flies.
  let lat = Math.atan2(z, r * (1 - E2));
  let h = 0;
  for (let i = 0; i < 10; i++) {
    const s = Math.sin(lat);
    const n = A / Math.sqrt(1 - E2 * s * s);
    h = r / Math.cos(lat) - n;
    const next = Math.atan2(z, r * (1 - (E2 * n) / (n + h)));
    if (Math.abs(next - lat) < 1e-14) {
      lat = next;
      break;
    }
    lat = next;
  }
  return { lat: lat / RAD, lon: lon / RAD, h };
}

/** Rows of the ECEF → ENU rotation at an origin: east, north, up. */
function enuAxes(o: Geodetic): [Vec3, Vec3, Vec3] {
  const lat = o.lat * RAD;
  const lon = o.lon * RAD;
  const sl = Math.sin(lat);
  const cl = Math.cos(lat);
  const so = Math.sin(lon);
  const co = Math.cos(lon);
  return [
    [-so, co, 0],
    [-sl * co, -sl * so, cl],
    [cl * co, cl * so, sl],
  ];
}

export function enuToEcef(origin: Geodetic, enu: Vec3): Vec3 {
  const o = geodeticToEcef(origin);
  const [e, n, u] = enuAxes(origin);
  return [
    o[0] + e[0] * enu[0] + n[0] * enu[1] + u[0] * enu[2],
    o[1] + e[1] * enu[0] + n[1] * enu[1] + u[1] * enu[2],
    o[2] + e[2] * enu[0] + n[2] * enu[1] + u[2] * enu[2],
  ];
}

export function ecefToEnu(origin: Geodetic, p: Vec3): Vec3 {
  const o = geodeticToEcef(origin);
  const d: Vec3 = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
  const [e, n, u] = enuAxes(origin);
  const dot = (a: Vec3) => a[0] * d[0] + a[1] * d[1] + a[2] * d[2];
  return [dot(e), dot(n), dot(u)];
}

export const enuToGeodetic = (origin: Geodetic, enu: Vec3): Geodetic => ecefToGeodetic(enuToEcef(origin, enu));
export const geodeticToEnu = (origin: Geodetic, g: Geodetic): Vec3 => ecefToEnu(origin, geodeticToEcef(g));

/** Distance in metres between two origins (0 for the same Site). */
export function originOffsetM(a: Geodetic, b: Geodetic): number {
  const p = geodeticToEnu(a, b);
  return Math.hypot(p[0], p[1], p[2]);
}

/** Below this the two frames are one: no conversion, and nothing to warn about. */
export const SAME_ORIGIN_M = 0.001;

/**
 * Site ENU of a mission → served coordinates of a scene. Identity plus the centre when the
 * scene is georeferenced with the same origin (the usual case: one Site, one frame); through
 * ECEF when its origin differs. `null` when the scene has no metric frame at all (a free COLMAP
 * frame): a mission cannot be placed in it.
 */
export function missionToServed(missionOrigin: Geodetic, scene: SceneFrame): ((enu: Vec3) => Vec3) | null {
  if (!scene.metric) return null;
  const c = scene.center;
  const cx = c[0] ?? 0;
  const cy = c[1] ?? 0;
  const cz = c[2] ?? 0;
  const so = scene.metric.origin;
  if (originOffsetM(missionOrigin, so) < SAME_ORIGIN_M) return (p) => [p[0] - cx, p[1] - cy, p[2] - cz];
  return (p) => {
    const q = ecefToEnu(so, enuToEcef(missionOrigin, p));
    return [q[0] - cx, q[1] - cy, q[2] - cz];
  };
}

/** What sceneRoot does to a served point (appGLSL.js applySceneTransform): rotate, then scale. */
export function servedToViewer(p: Vec3, scene: Pick<SceneFrame, "scene_scale" | "up_axis_rotation">): Vec3 {
  const r = scene.up_axis_rotation;
  let [x, y, z] = p;
  if (r) {
    const t = r.degrees * RAD;
    const c = Math.cos(t);
    const s = Math.sin(t);
    if (r.axis === "x") [y, z] = [c * y - s * z, s * y + c * z];
    else if (r.axis === "y") [x, z] = [c * x + s * z, -s * x + c * z];
    else [x, y] = [c * x - s * y, s * x + c * y];
  }
  const k = scene.scene_scale;
  return [x * k, y * k, z * k];
}

/** Compass heading (degrees clockwise from north) → unit vector in ENU. */
export function headingToEnu(deg: number): Vec3 {
  return [Math.sin(deg * RAD), Math.cos(deg * RAD), 0];
}
