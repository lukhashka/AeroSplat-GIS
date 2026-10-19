// Frames of the flight view: a known waypoint must land where the server, the splats and the
// map put it. Run with `npm test` (Node's built-in runner; Node ≥ 22.18 strips the types).
//
// Reference numbers are from the server's EnuFrame (pyproj, aerosplat.geo.crs), for the hover
// point H of the dataset_v6 window mission (msn_01M58J881BDXS3T3MNFXP8X6QX): its row 7 of
// mission.waypoints is 40.890899659, -74.216862431.
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  enuToGeodetic,
  geodeticToEnu,
  headingToEnu,
  missionToServed,
  originOffsetM,
  servedToViewer,
  type Geodetic,
  type SceneFrame,
  type Vec3,
} from "../src/lib/frames.ts";

const SITE: Geodetic = { lat: 40.89104597075257, lon: -74.21634205070951, h: 233.0 };
const H: Vec3 = [-43.854783166902834, -16.248223495839767, -184.2320898996386];
const H_WGS84: Geodetic = { lat: 40.890899658588545, lon: -74.21686243149915, h: 48.76808139681816 };
// The same point in the ENU of another origin (EnuFrame(OTHER).from_geodetic(H_WGS84)).
const OTHER: Geodetic = { lat: 40.8925, lon: -74.2141, h: 180.0 };
const H_IN_OTHER: Vec3 = [-232.80228013460663, -177.7185067088861, -131.2386430147941];

const near = (a: number, b: number, tol: number, what: string) =>
  assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b} (tol ${tol})`);
const nearVec = (a: Vec3, b: Vec3, tol: number, what: string) =>
  a.forEach((v, i) => near(v, b[i]!, tol, `${what}[${i}]`));

test("Site ENU -> WGS84 matches the server (and the exported waypoint row)", () => {
  const g = enuToGeodetic(SITE, H);
  near(g.lat, H_WGS84.lat, 1e-10, "lat");
  near(g.lon, H_WGS84.lon, 1e-10, "lon");
  near(g.h, H_WGS84.h, 1e-4, "h");
  near(g.lat, 40.890899659, 5e-10, "lat as in mission.waypoints");
  near(g.lon, -74.216862431, 5e-10, "lon as in mission.waypoints");
});

test("WGS84 -> Site ENU round trip is sub-millimetre", () => {
  nearVec(geodeticToEnu(SITE, enuToGeodetic(SITE, H)), H, 1e-4, "H");
});

test("a scene with the mission's own origin is the mission frame minus the centre", () => {
  const scene: SceneFrame = {
    center: [-40, -10, -180],
    scene_scale: 0.25,
    up_axis_rotation: { axis: "x", degrees: -90 },
    metric: { origin: SITE },
  };
  const toServed = missionToServed(SITE, scene)!;
  nearVec(toServed(H), [H[0] + 40, H[1] + 10, H[2] + 180], 1e-12, "served H");
});

test("a scene with another origin goes through ECEF", () => {
  const scene: SceneFrame = { center: [0, 0, 0], scene_scale: 1, up_axis_rotation: null, metric: { origin: OTHER } };
  nearVec(missionToServed(SITE, scene)!(H), H_IN_OTHER, 1e-4, "H in the other frame");
  assert.ok(originOffsetM(SITE, OTHER) > 100);
  near(originOffsetM(SITE, SITE), 0, 1e-9, "same origin");
});

test("a scene without a metric frame cannot hold a mission", () => {
  const scene: SceneFrame = { center: [0, 0, 0], scene_scale: 1, up_axis_rotation: { axis: "x", degrees: 90 }, metric: null };
  assert.equal(missionToServed(SITE, scene), null);
});

test("viewer frame: Rx(-90°) keeps up up and sends north into the screen (ADR-0001)", () => {
  const scene = { scene_scale: 0.5, up_axis_rotation: { axis: "x" as const, degrees: -90 } };
  nearVec(servedToViewer([0, 0, 10], scene), [0, 5, 0], 1e-12, "up -> +Y");
  nearVec(servedToViewer([0, 10, 0], scene), [0, 0, -5], 1e-12, "north -> -Z");
  nearVec(servedToViewer([10, 0, 0], scene), [5, 0, 0], 1e-12, "east -> +X");
});

test("H of the window mission lands in the viewer where the splats around it are", () => {
  // scene-meta of the legacy dataset_v6 scene as the viewer gets it, rounded: the splat at the
  // window is served at H - center, then sceneRoot turns and scales it.
  const scene: SceneFrame = {
    center: [-3.1, 2.4, -195.7],
    scene_scale: 0.3,
    up_axis_rotation: { axis: "x", degrees: -90 },
    metric: { origin: SITE },
  };
  const served = missionToServed(SITE, scene)!(H);
  const w = servedToViewer(served, scene);
  nearVec(w, [(H[0] + 3.1) * 0.3, (H[2] + 195.7) * 0.3, -(H[1] - 2.4) * 0.3], 1e-9, "viewer H");
});

test("compass heading -> ENU direction", () => {
  nearVec(headingToEnu(0), [0, 1, 0], 1e-12, "north");
  nearVec(headingToEnu(90), [1, 0, 0], 1e-12, "east");
  // The window mission faces 221.6°: south-west, into the facade whose normal is (0.66, 0.75).
  const d = headingToEnu(221.56624028455786);
  near(d[0] * 0.663485481190806 + d[1] * 0.7481891580669988, -1, 1e-9, "heading = -normal");
});
