import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "@playwright/test";

// E2E smoke of the built UI against throwaway servers (tools/e2e_server.py): never the owner's
// projects/ and never ports 8000/8001. Run with `npm run e2e` (builds first); README "E2E-тест UI",
// docs/ARCHITECTURE.md §3.11.
const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "../..");

function python(): string {
  if (process.env.AEROSPLAT_PYTHON) return process.env.AEROSPLAT_PYTHON;
  for (const p of [join(repo, "venv", "Scripts", "python.exe"), join(repo, "venv", "bin", "python")]) {
    if (existsSync(p)) return p;
  }
  return "python";
}

export const PORT = Number(process.env.E2E_PORT ?? 18765);
export const AUTH_PORT = PORT + 1;
export const TOKEN_FILE = join(tmpdir(), "aerosplat-e2e", `token-${AUTH_PORT}.txt`);

const env = {
  ...process.env,
  PYTHONPATH: join(repo, "src"), // this checkout's code, whatever is pip-installed
  PYTHONIOENCODING: "utf-8",
  PYTHONUTF8: "1",
} as Record<string, string>;

const server = (port: number, extra: string) => ({
  command: `"${python()}" tools/e2e_server.py --port ${port} ${extra}`.trim(),
  cwd: repo,
  url: `http://127.0.0.1:${port}/api/v2/auth/whoami`,
  reuseExistingServer: false,
  timeout: 180_000,
  env,
  stdout: "pipe" as const,
  stderr: "pipe" as const,
});

export default defineConfig({
  testDir: "e2e",
  timeout: 90_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
  webServer: [server(PORT, ""), server(AUTH_PORT, `--require-auth --token-file "${TOKEN_FILE}"`)],
});
