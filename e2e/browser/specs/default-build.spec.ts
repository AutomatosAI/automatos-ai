import { expect, test } from '@playwright/test'
import { THEME_E2E_PATH } from '../support/hello-theme'
import '../support/page-globals'
import { themeRequestPaths, watchPage } from '../support/watch'

// PRD-261 A-04a (@off): the theme host ships in every build, and a fresh browser on Studio meets
// none of it: no request for a theme (the shims, the active install, a theme's files), no host
// global, no status attribute on <html>, and the Studio shell drawn. The import map itself is on
// every page from A-03 on (PA-13), so this no longer asserts its absence.

const WORKSPACE_PATH = '/api/workspaces/current'
const STUDIO_CLASS = /\bstudio\b/

test('Studio loads none of the theme host', { tag: '@off' }, async ({ page }) => {
  const watch = watchPage(page)
  // The app's first GET of its workspace: the client has mounted, so the host's effect has run.
  const appRan = page.waitForResponse((response) => response.url().endsWith(WORKSPACE_PATH) && response.request().method() === 'GET')
  await page.goto(THEME_E2E_PATH)
  await appRan
  await expect(page.locator('html')).toHaveClass(STUDIO_CLASS)
  await expect(page.locator('.sh-shell')).toBeVisible()

  expect(await page.evaluate(() => typeof window.__AUTOMATOS_THEME_HOST__)).toBe('undefined')
  await expect(page.locator('html')).not.toHaveAttribute('data-theme-host')
  expect(themeRequestPaths(watch.requests)).toEqual([])
})
