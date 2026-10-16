import { expect, test } from '@playwright/test'

// PRD-261 A-00: the lane's sample test. A fresh browser on the local edition gets Studio, the
// app reaches the local backend for its workspace, and the Studio desktop shell draws.

const SMOKE_PATH = '/agents'
const WORKSPACE_PATH = '/api/workspaces/current'
const STUDIO_CLASS = /\bstudio\b/

test('the local edition draws the Studio shell on /agents', { tag: '@smoke' }, async ({ page }) => {
  const pageErrors: Error[] = []
  page.on('pageerror', (error) => pageErrors.push(error))

  // Registered before the navigation so the response cannot be missed. The method check skips
  // the CORS preflight (OPTIONS), which the browser sends first because :3000 calls :8000.
  const workspaceResponse = page.waitForResponse(
    (response) => response.url().endsWith(WORKSPACE_PATH) && response.request().method() === 'GET',
  )
  await page.goto(SMOKE_PATH)

  expect((await workspaceResponse).status()).toBe(200)
  await expect(page.locator('html')).toHaveClass(STUDIO_CLASS)
  await expect(page.locator('.sh-shell')).toBeVisible()
  expect(pageErrors.map((error) => error.message)).toEqual([])
})
