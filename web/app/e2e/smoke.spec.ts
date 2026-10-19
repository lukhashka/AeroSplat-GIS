import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { expect, test, type Page } from "@playwright/test";

import { AUTH_PORT, PORT, TOKEN_FILE } from "../playwright.config";

// ROADMAP 6.3 criterion, as far as it goes without a GPU: create a site, upload frames, watch
// the ingest job, see the capture; open the pages that matter (sites, jobs, facade editor,
// flight). Fixtures come from tools/e2e_server.py; nothing here touches the owner's projects/.

const samples = [0, 1, 2].map((i) => join(tmpdir(), "aerosplat-e2e", String(PORT), "samples", `e2e_${i}.jpg`));

async function createSiteViaUi(page: Page, name: string): Promise<void> {
  await page.getByRole("button", { name: "Нова ділянка" }).first().click();
  const dialog = page.getByRole("dialog", { name: "Нова ділянка" });
  await expect(dialog).toBeVisible();
  // The origin is a click on the map (MapLibre, WebGL). Wait until it can take clicks.
  const canvas = dialog.locator("canvas.maplibregl-canvas");
  await expect(canvas).toBeVisible();
  await page.waitForTimeout(500);
  await canvas.click({ position: { x: 200, y: 120 } });
  await expect(dialog.getByText(/^Початок:/)).toBeVisible();
  await dialog.getByPlaceholder("напр. Склад на Подолі").fill(name);
  await dialog.getByRole("button", { name: "Створити ділянку" }).click();
}

test("sites page shows the fixture sites", async ({ page }) => {
  await page.goto("/app/");
  await expect(page.getByRole("heading", { name: "Ділянки", level: 1 })).toBeVisible();
  await expect(page.getByText("Pier (facades)")).toBeVisible();
  await expect(page.getByText("Street (mission)")).toBeVisible();
  // the sidebar sees the CPU worker that `aerosplat up` started
  await expect(page.locator(".sys").getByText("CPU-воркер")).toBeVisible();
});

test("create site -> upload capture -> ingest job on the CPU worker -> capture on the site", async ({ page }) => {
  await page.goto("/app/");
  await createSiteViaUi(page, "E2E Site");
  await expect(page).toHaveURL(/\/app\/sites\/demo\/e2e-site$/);
  await expect(page.getByRole("heading", { name: "E2E Site", level: 1 })).toBeVisible();

  await page.getByRole("button", { name: "Завантажити кадри" }).click();
  const dialog = page.getByRole("dialog", { name: "Нове захоплення" });
  await dialog.locator('input[type="file"]').setInputFiles(samples);
  await expect(dialog.getByText(/3\s*зображень/)).toBeVisible();
  await dialog.getByPlaceholder(/^handheld-/).fill("phone-e2e");
  await dialog.getByRole("button", { name: /^Завантажити/ }).click();

  // the upload returns the ingest job; its page follows it over the WebSocket
  await expect(page).toHaveURL(/\/app\/jobs\/[0-9a-f]{32}$/);
  await expect(page.getByRole("heading", { level: 1 })).toContainText("захоплення phone-e2e");
  await expect(page.getByText("ingest · CPU")).toBeVisible();
  await expect(page.getByText("Реєстрація в каталозі")).toBeVisible();
  await expect(page.locator(".page-head").getByText("готово")).toBeVisible({ timeout: 60_000 });
  await expect(page.locator(".log, .logview, pre").first()).toContainText("3/3");

  await page.getByRole("link", { name: "Ділянка" }).click();
  await expect(page).toHaveURL(/\/app\/sites\/demo\/e2e-site$/);
  await expect(page.getByText("phone-e2e").first()).toBeVisible();
  await expect(page.getByRole("button", { name: "Запустити run" })).toBeEnabled();

  await page.goto("/app/jobs");
  await expect(page.getByText("захоплення phone-e2e").first()).toBeVisible();
});

test("facade editor opens on the fixture facade", async ({ page }) => {
  await page.goto("/app/facades/demo/pier");
  await expect(page.getByRole("heading", { name: "Фасади й точки доступу", level: 1 })).toBeVisible();
  const row = page.getByRole("row").filter({ hasText: "північ" });
  await expect(row).toContainText("N1");
  await row.getByRole("button", { name: "Редактор" }).click();
  await expect(page).toHaveURL(/\/app\/facade\/demo\/pier\/B01\/N1/);
  await expect(page.getByRole("heading", { name: "Редактор фасаду B01/N1", level: 1 })).toBeVisible();
});

test("flight page shows the fixture mission and its simulated flight", async ({ page }) => {
  await page.goto("/app/flight/demo/street");
  await expect(page.getByRole("heading", { name: "Місія / проліт", level: 1 })).toBeVisible();
  await expect(page.getByRole("combobox").nth(1)).toContainText("street-window-delivered");
  await expect(page.getByRole("heading", { name: "Вердикт оракулів" })).toBeVisible();
});

test("not local: the UI asks for a token, the session cookie is HttpOnly", async ({ page, context }) => {
  const base = `http://127.0.0.1:${AUTH_PORT}`;
  await page.goto(`${base}/app/`);
  const dialog = page.getByRole("dialog", { name: "Потрібен токен доступу" });
  await expect(dialog).toBeVisible();
  await expect(page.getByText("Pier (facades)")).toHaveCount(0);

  await dialog.getByLabel("Токен доступу").fill("asg_wrong");
  await dialog.getByRole("button", { name: "Увійти" }).click();
  await expect(dialog.locator(".error-box")).toContainText("invalid or revoked token");

  await dialog.getByLabel("Токен доступу").fill(readFileSync(TOKEN_FILE, "utf-8").trim());
  await dialog.getByRole("button", { name: "Увійти" }).click();
  await expect(dialog).toBeHidden();
  await expect(page.getByText("Pier (facades)")).toBeVisible();
  await expect(page.locator(".sys")).toContainText("токен e2e");

  const cookie = (await context.cookies(base)).find((c) => c.name === "aerosplat_token");
  expect(cookie?.httpOnly).toBe(true);
  expect(cookie?.sameSite).toBe("Strict");
  expect(await page.evaluate(() => document.cookie)).not.toContain("aerosplat_token");

  await page.locator(".sys").getByRole("button", { name: "вийти" }).click();
  await expect(dialog).toBeVisible();
});
