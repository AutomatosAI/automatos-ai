import { expect, test, type Page } from '@playwright/test'
import { fulfillBackendJson } from '../support/backend-route'
import { chooseCoreStyle, CORE_STYLES, STUDIO_STYLE, THEME_E2E_PATH, type CoreStyle } from '../support/hello-theme'
import '../support/page-globals'
import { themeRequestPaths } from '../support/watch'

// PRD-261 A-04a (@off): the theme host ships in every build, and a fresh browser on Studio meets
// none of it: no request for a theme (the shims, the active install, a theme's files), no host
// global, no status attribute on <html>, and the Studio shell drawn. The import map itself is on
// every page from A-03 on (PA-13), so this no longer asserts its absence. A-02c (PA-12): the server
// holds the theme choice, so the tab asks GET /api/themes/active once; that is its only theme
// request. The answer is Studio's here (a server naming an install moves the tab onto it, which
// specs/theme-choice.spec.ts proves), so the spec does not depend on the operator's stored choice.
// The server's HTML of a page a ThemeSlot page entry wraps carries the Studio shell once, in no
// boundary React outlined: with the core page as a Suspense fallback the page came down twice (the
// fallback, then a hidden copy swapped in later), and WebKit drew two shells (theme-slot.tsx). The
// traces of two CI runs hold that HTML: /agents in run 39280303014 and /chat in run 39257435820,
// each with `<template id="B:0">`, `<div hidden id="S:0">` and two shells, about 17 KB apiece, past
// the 12,800 bytes React outlines from. No run captured /deliverables before the change, so there
// the check holds the page to the same shape without having seen it fail; its layout's own test
// (app/deliverables/__tests__/deliverables-layout.test.tsx) proves the boundary is gone.
//
// PRD-261 A-19f4 (W-A8): both checks also run under the Classic and under the Studio style cookie,
// each with its own chrome (support/hello-theme.ts CORE_STYLES): a style the cookie names loads none
// of the theme host either, and the server sends each page with that style's chrome once, in no
// outlined boundary. The fresh browser's checks stay as they were.

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const WORKSPACE_PATH = '/api/workspaces/current'
const STUDIO_CLASS = /\bstudio\b/
const ACTIVE_PATH = '/api/themes/active'
const STYLE_COOKIE = 'automatos-style'
const ACTIVE_ROUTE = '**/api/themes/active'
const OK = 200
/** GET /api/themes/active for a member with no choice and no workspace default (A-07b). */
const NO_CHOICE = { choice: 'studio', theme: null }
const CHAT_PATH = '/chat'
const DELIVERABLES_PATH = '/deliverables'
/** Pages a ThemeSlot page entry wraps (each route's app/<route>/layout.tsx), opened on Studio. */
const THEME_SLOT_PATHS = [THEME_E2E_PATH, CHAT_PATH, DELIVERABLES_PATH]
/** What React's streaming renderer writes for a boundary it outlined: the pending marker, its template, the hidden copy. */
const OUTLINED_BOUNDARY_MARKS = ['<!--$?-->', '<template id="B:', 'id="S:']

function occurrences(text: string, needle: string): number {
  return text.split(needle).length - 1
}

/** The URLs of the page's GET requests (a preflight is an OPTIONS, not a request for a theme). */
function watchGets(page: Page): string[] {
  const gets: string[] = []
  page.on('request', (request) => {
    if (request.method() === 'GET') gets.push(request.url())
  })
  return gets
}

/** Opens the page once the app has run, and checks it met none of the theme host under `core`'s style. */
async function expectNoThemeHost(page: Page, core: CoreStyle): Promise<void> {
  // The app's first GET of its workspace: the client has mounted, so the host's effect has run.
  const appRan = page.waitForResponse((response) => response.url().endsWith(WORKSPACE_PATH) && response.request().method() === 'GET')
  const gets = watchGets(page)
  const choiceChecked = page.waitForResponse((response) => new URL(response.url()).pathname === ACTIVE_PATH && response.request().method() === 'GET')
  await page.goto(THEME_E2E_PATH)
  await appRan
  await choiceChecked
  if (core.studio) await expect(page.locator('html')).toHaveClass(STUDIO_CLASS)
  else await expect(page.locator('html')).not.toHaveClass(STUDIO_CLASS)
  await expect(page.locator(core.shell)).toBeVisible()

  expect(await page.evaluate(() => typeof window.__AUTOMATOS_THEME_HOST__)).toBe('undefined')
  await expect(page.locator('html')).not.toHaveAttribute('data-theme-host')
  // A cross-origin preflight (OPTIONS) is not a request for a theme: the GETs are counted.
  expect(themeRequestPaths(gets)).toEqual([ACTIVE_PATH])
}

test('Studio loads none of the theme host', { tag: '@off' }, async ({ page }) => {
  await page.route(ACTIVE_ROUTE, (route) => fulfillBackendJson(route, OK, NO_CHOICE))
  await expectNoThemeHost(page, STUDIO_STYLE)
})

for (const core of CORE_STYLES) {
  test(`under the '${core.style}' cookie the page loads none of the theme host`, { tag: '@off' }, async ({ page }) => {
    await chooseCoreStyle(page, core, APP_URL)
    await expectNoThemeHost(page, core)
  })
}

/** Who asks for the page's HTML: a fresh browser (no cookie, so Studio) or one on each core style. */
interface HtmlCase {
  readonly who: string
  readonly cookie: string | null
  readonly core: CoreStyle
}

const HTML_CASES: readonly HtmlCase[] = [
  { who: 'a fresh browser', cookie: null, core: STUDIO_STYLE },
  ...CORE_STYLES.map((core) => ({ who: `the '${core.style}' cookie`, cookie: core.style, core })),
]

for (const { who, cookie, core } of HTML_CASES) {
  for (const route of THEME_SLOT_PATHS) {
    test(`for ${who} the server sends ${route} with one ${core.style} chrome, in no outlined boundary`, { tag: '@off' }, async ({ request }) => {
      const headers = cookie === null ? undefined : { cookie: `${STYLE_COOKIE}=${cookie}` }
      const response = await request.get(route, { headers })
      expect(response.ok()).toBe(true)
      const html = await response.text()
      expect(occurrences(html, core.shellInHtml)).toBe(1)
      expect(OUTLINED_BOUNDARY_MARKS.filter((mark) => html.includes(mark))).toEqual([])
    })
  }
}
