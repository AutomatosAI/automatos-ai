import { expect, test, type Page } from '@playwright/test'
import { chooseCoreStyle, CORE_STYLES, STUDIO_STYLE, type CoreStyle } from '../support/hello-theme'

// PRD-261 A-00: the lane's sample test. A fresh browser on the local edition gets Studio, the
// app reaches the local backend for its workspace, and the Studio desktop shell draws.
// PRD-261 A-19f4 (W-A8): the same under the Classic and under the Studio style cookie, each style
// drawing its own desktop chrome (support/hello-theme.ts CORE_STYLES).

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const SMOKE_PATH = '/agents'
const WORKSPACE_PATH = '/api/workspaces/current'
const STUDIO_CLASS = /\bstudio\b/
const OK = 200

/** Opens /agents and checks the workspace answered, `core`'s chrome drew and nothing threw. */
async function expectShellDraws(page: Page, core: CoreStyle): Promise<void> {
  const pageErrors: Error[] = []
  page.on('pageerror', (error) => pageErrors.push(error))

  // Registered before the navigation so the response cannot be missed. The method check skips
  // the CORS preflight (OPTIONS), which the browser sends first because :3000 calls :8000.
  const workspaceResponse = page.waitForResponse(
    (response) => response.url().endsWith(WORKSPACE_PATH) && response.request().method() === 'GET',
  )
  await page.goto(SMOKE_PATH)

  expect((await workspaceResponse).status()).toBe(OK)
  if (core.studio) await expect(page.locator('html')).toHaveClass(STUDIO_CLASS)
  else await expect(page.locator('html')).not.toHaveClass(STUDIO_CLASS)
  await expect(page.locator(core.shell)).toBeVisible()
  expect(pageErrors.map((error) => error.message)).toEqual([])
}

test('the local edition draws the Studio shell on /agents', { tag: '@smoke' }, async ({ page }) => {
  await expectShellDraws(page, STUDIO_STYLE)
})

for (const core of CORE_STYLES) {
  test(`under the '${core.style}' cookie the local edition draws the ${core.style} chrome on /agents`, { tag: '@smoke' }, async ({ page }) => {
    await chooseCoreStyle(page, core, APP_URL)
    await expectShellDraws(page, core)
  })
}
