import { expect, test, type Page } from '@playwright/test'
import { fulfillBackendJson } from '../support/backend-route'
import { THEME_E2E_PATH } from '../support/hello-theme'
import '../support/page-globals'
import { themeRequestPaths } from '../support/watch'

// PRD-261 A-04a (@off): the theme host ships in every build, and a fresh browser on Studio meets
// none of it: no request for a theme (the shims, the active install, a theme's files), no host
// global, no status attribute on <html>, and the Studio shell drawn. The import map itself is on
// every page from A-03 on (PA-13), so this no longer asserts its absence. A-02c (PA-12): the server
// holds the theme choice, so the tab asks GET /api/themes/active once; that is its only theme
// request. The answer is Studio's here (a server naming an install moves the tab onto it, which
// specs/theme-choice.spec.ts proves), so the spec does not depend on the operator's stored choice.

const WORKSPACE_PATH = '/api/workspaces/current'
const STUDIO_CLASS = /\bstudio\b/
const ACTIVE_PATH = '/api/themes/active'
const ACTIVE_ROUTE = '**/api/themes/active'
const OK = 200
/** GET /api/themes/active for a member with no choice and no workspace default (A-07b). */
const NO_CHOICE = { choice: 'studio', theme: null }

/** The URLs of the page's GET requests (a preflight is an OPTIONS, not a request for a theme). */
function watchGets(page: Page): string[] {
  const gets: string[] = []
  page.on('request', (request) => {
    if (request.method() === 'GET') gets.push(request.url())
  })
  return gets
}

test('Studio loads none of the theme host', { tag: '@off' }, async ({ page }) => {
  // The app's first GET of its workspace: the client has mounted, so the host's effect has run.
  const appRan = page.waitForResponse((response) => response.url().endsWith(WORKSPACE_PATH) && response.request().method() === 'GET')
  await page.route(ACTIVE_ROUTE, (route) => fulfillBackendJson(route, OK, NO_CHOICE))
  const gets = watchGets(page)
  const choiceChecked = page.waitForResponse((response) => new URL(response.url()).pathname === ACTIVE_PATH && response.request().method() === 'GET')
  await page.goto(THEME_E2E_PATH)
  await appRan
  await choiceChecked
  await expect(page.locator('html')).toHaveClass(STUDIO_CLASS)
  await expect(page.locator('.sh-shell')).toBeVisible()

  expect(await page.evaluate(() => typeof window.__AUTOMATOS_THEME_HOST__)).toBe('undefined')
  await expect(page.locator('html')).not.toHaveAttribute('data-theme-host')
  // A cross-origin preflight (OPTIONS) is not a request for a theme: the GETs are counted.
  expect(themeRequestPaths(gets)).toEqual([ACTIVE_PATH])
})
