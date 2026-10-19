// Takeoff-spot search and planning from the page (src/lib/takeoff.ts): what goes to the server,
// and that the planner's refusals reach the operator word for word. Run with `npm test`.
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  circleEnu,
  parseCount,
  planBody,
  profileFromForm,
  refusalOf,
  rejectedRows,
  searchKey,
} from "../src/lib/takeoff.ts";

// The detail of a 409 as aerosplat.server.api_mission._refused builds it.
const OUTSIDE_GRID =
  "the mission leaves the occupancy grid (x -40.0…40.0, y -30.0…30.0 m): home (100.0, 100.0) needs 6.1 m to the " +
  "edge, has -60.0. Outside the grid nothing is known, so the corridor cannot be checked. Rebuild occupancy with " +
  "an extent that holds it, e.g. --set occupancy_bbox=-40,-30,120,120 (or a larger occupancy_radius, or a Site bbox)";

test("a chosen candidate goes as home_enu with the search it came from, never with home_search", () => {
  // The query the result was found with: run and profile go with the plan, the rest says which search.
  const query = { run_id: "r1", profile: { sigma_gnss_m: 0.1 }, max_distance_m: 60, max_checks: 120 };
  const body = planBody(
    "tgt_1",
    { kind: "candidate", rank: 2, enu: [-5.2, -37.8, -166.7], search: query },
    { run_id: query.run_id, profile: query.profile },
  );
  assert.deepEqual(body, {
    target_id: "tgt_1",
    run_id: "r1",
    profile: { sigma_gnss_m: 0.1 },
    home_enu: [-5.2, -37.8, -166.7],
    home_from_search: { rank: 2, max_distance_m: 60, max_checks: 120 }, // the server forbids other fields here
  });
  assert.equal("home_search" in body, false);

  const plain = planBody("tgt_1", { kind: "candidate", rank: 1, enu: [1, 2, 3], search: {} });
  assert.deepEqual(plain.home_from_search, { rank: 1 }); // a search with the server's defaults
});

test("the server's own search goes as home_search, without home_enu", () => {
  const body = planBody("tgt_1", { kind: "search" }, { profile: {} });
  assert.deepEqual(body, { target_id: "tgt_1", home_search: true });
  assert.equal("home_enu" in body, false);
  assert.equal("profile" in body, false); // empty profile: the server's defaults, not {}
});

test("refusals come out with every reason verbatim", () => {
  const err = {
    status: 409,
    detail: {
      message: `outside_grid: ${OUTSIDE_GRID}`,
      reasons: [{ code: "outside_grid", severity: "blocking", leg: null, detail: OUTSIDE_GRID, at: [100, 100, 0] }],
    },
  };
  const r = refusalOf(err);
  assert.ok(r);
  assert.equal(r.reasons.length, 1);
  assert.equal(r.reasons[0]!.code, "outside_grid");
  assert.equal(r.reasons[0]!.detail, OUTSIDE_GRID);
  assert.match(r.reasons[0]!.detail, /occupancy_bbox=-40,-30,120,120/);

  const none = refusalOf({ status: 409, detail: { message: "no_takeoff_spot: no takeoff spot within 150 m", reasons: [] } });
  assert.deepEqual(none, { message: "no_takeoff_spot: no takeoff spot within 150 m", reasons: [] });
});

test("anything else is not a refusal", () => {
  assert.equal(refusalOf(null), null);
  assert.equal(refusalOf(new Error("x")), null);
  assert.equal(refusalOf({ status: 404, detail: "run r1 has occupancy but no semantics" }), null);
  assert.equal(refusalOf({ status: 422, detail: [{ loc: ["body", "n"], msg: "too big" }] }), null);
});

test("the profile holds only what was typed, and a typo is an error, not a default", () => {
  assert.deepEqual(profileFromForm({ sigma_gnss_m: "", r_drone_m: " " }), { profile: undefined });
  assert.deepEqual(profileFromForm({ sigma_gnss_m: "0,1", r_drone_m: "" }), { profile: { sigma_gnss_m: 0.1 } });
  assert.deepEqual(profileFromForm({ sigma_gnss_m: "1.5", r_drone_m: "0.6" }), { profile: { sigma_gnss_m: 1.5, r_drone_m: 0.6 } });
  const bad = profileFromForm({ sigma_gnss_m: "1.5 м", r_drone_m: "" });
  assert.ok("error" in bad && bad.error.startsWith("sigma_gnss_m"));
});

test("rejections keep the server's order and drop zeros", () => {
  const rows = rejectedRows({ near_grid_edge: 0, not_measured: 12, building: 300, obstacle_within_radius: 40, something_new: 2 });
  assert.deepEqual(
    rows.map((r) => [r.code, r.count]),
    [
      ["not_measured", 12],
      ["building", 300],
      ["obstacle_within_radius", 40],
      ["something_new", 2],
    ],
  );
  assert.equal(rows.at(-1)!.label, "something_new"); // an unknown reason is shown by its code
});

test("a target too near the grid's edge is one row, in Ukrainian", () => {
  // The search answers at once: every column under target_near_grid_edge, nothing else.
  const rows = rejectedRows({ target_near_grid_edge: 25600, near_grid_edge: 0, not_checked: 0 });
  assert.equal(rows.length, 1);
  assert.equal(rows[0]!.count, 25600);
  assert.notEqual(rows[0]!.label, "target_near_grid_edge");
  assert.match(rows[0]!.label, /краю сітки/);
});

test("a search is identified by its parameters", () => {
  const a = searchKey("tgt_1", { profile: { sigma_gnss_m: 1.5 } });
  assert.equal(a, searchKey("tgt_1", { profile: { sigma_gnss_m: 1.5 }, max_distance_m: 150, n: 5 })); // the defaults
  assert.notEqual(a, searchKey("tgt_1", { profile: { sigma_gnss_m: 0.1 } }));
  assert.notEqual(a, searchKey("tgt_2", { profile: { sigma_gnss_m: 1.5 } }));
  assert.notEqual(a, searchKey("tgt_1", { profile: { sigma_gnss_m: 1.5 }, run_id: "r2" }));
  assert.equal(a, searchKey("tgt_1", { profile: { sigma_gnss_m: 1.5 }, max_checks: 40 }));
  assert.notEqual(a, searchKey("tgt_1", { profile: { sigma_gnss_m: 1.5 }, max_checks: 200 }));
});

test("max_checks is a whole number within the server's bounds, or blank for its default", () => {
  assert.deepEqual(parseCount(" ", 400), { value: undefined });
  assert.deepEqual(parseCount("200", 400), { value: 200 });
  for (const bad of ["0", "401", "12.5", "багато"]) assert.ok("error" in parseCount(bad, 400), bad);
});

test("circles are closed rings at the radius", () => {
  const ring = circleEnu(10, -5, 3, 36);
  assert.equal(ring.length, 37);
  assert.deepEqual(ring[0], ring[ring.length - 1]);
  for (const [x, y] of ring) assert.ok(Math.abs(Math.hypot(x - 10, y + 5) - 3) < 1e-9);
});
