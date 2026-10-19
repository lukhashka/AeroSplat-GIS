// `npm run e2e`: the Playwright smoke, skipped (exit 0, with a hint) when Playwright or its
// Chromium is not installed, so a machine without browsers is not "red" for no reason.
// Extra arguments go to `playwright test` (e.g. `npm run e2e -- --headed`).
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";

let chromium;
try {
  ({ chromium } = await import("@playwright/test"));
} catch {
  console.log("e2e skipped: @playwright/test is not installed (npm install in web/app).");
  process.exit(0);
}
let exe = "";
try {
  exe = chromium.executablePath();
} catch {
  exe = "";
}
if (!exe || !existsSync(exe)) {
  console.log("e2e skipped: Chromium for Playwright is not installed. Install it with:\n  npx playwright install chromium");
  process.exit(0);
}
const npx = process.platform === "win32" ? "npx.cmd" : "npx";
const res = spawnSync(npx, ["playwright", "test", ...process.argv.slice(2)], { stdio: "inherit", shell: process.platform === "win32" });
process.exit(res.status ?? 1);
