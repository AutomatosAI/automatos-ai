import { expect, test, type Page } from '@playwright/test'
import { chooseStyle, THEME_E2E_PATH } from '../support/hello-theme'
import { CLI_TIMEOUT_MS, installWithKitCli, type ThemeManifest } from '../support/kit-cli'
import { resetKitPackage } from '../support/kit-package'
import { watchPage, type PageWatch } from '../support/watch'

// PRD-261 B-03h (@installed): any Design Kit theme, installed by the kit's own CLI and held to what
// every theme owes the host, with nothing theme-specific. The run's theme (Play on a pull request or a
// push; a dispatch's kit_theme otherwise) is installed with `automatos-theme install <file> --to
// http://localhost:3000 --choose` (support/kit-cli.ts), then opened on the page the host specs use and
// on every page and route its theme.json declares (a page on its route, a `[param]` segment as an id
// no workspace has; a route at /x/<route>). On each:
//
// - the host runs the theme: `data-theme-host="active"` and the theme's htmlClass on <html>;
// - there is no fallback notice: the host never went back to Studio;
// - the page drew with no error boundary: not the page's own inside the frame
//   (components/theme-host/page-boundary.tsx), not Next's not-found one (app/not-found.tsx, where a
//   route the theme declares but never registered ends up), and no uncaught error.
//
// It stores the member's choice on the lane's one backend, so the workflow runs @installed in its own
// step, one test at a time; the test removes the install and puts the choice back on Studio.

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const INSTALLED_TAG = '@installed'
const STATUS_ATTR = 'data-theme-host'
/** frontend/lib/theme-host/constants.ts HTML_CLASS_PATTERN. */
const HTML_CLASS_PATTERN = /^[a-z][a-z0-9-]{2,31}$/
/** components/theme-host/theme-host-provider.tsx THEME_FALLBACK_NOTICE. */
const FALLBACK_NOTICE = 'The theme could not start, so Studio is showing instead.'
/** components/theme-host/page-boundary.tsx PAGE_ERROR_TEST_ID. */
const PAGE_ERROR_TEST_ID = 'theme-page-error'
/** What Next's not-found boundary renders with the 404 (next's client/components/http-access-fallback/error-boundary.js). */
const NOT_FOUND_MARK = 'meta[name="robots"][content="noindex"]'
const ROUTE_PREFIX = '/x/'
const SEGMENT_SEPARATOR = '/'
/** A `[param]` segment of a page's route (frontend/lib/theme-host/constants.ts THEME_PAGE_ROUTES: '/missions/[id]'). */
const DYNAMIC_SEGMENT = /^\[[^\]/]+\]$/
/** An id no workspace has, for a page's `[param]` segment. */
const UNKNOWN_ID = '00000000-0000-4000-8000-0000000b0308'
/** The package's entry, stylesheet and fonts come through the files proxy on a cold load. */
const THEME_LOAD_TIMEOUT_MS = 30_000
/** How long a page has, once the host is active, to draw: its page, an error boundary or the 404. */
const SETTLE_MS = 1_500
/** One address: a cold load, the theme's load and the settle. */
const ADDRESS_TIMEOUT_MS = 45_000

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** A page's route as an address: each `[param]` segment becomes UNKNOWN_ID. */
function pageAddress(route: string): string {
  const segments = route.split(SEGMENT_SEPARATOR)
  return segments.map((segment) => (DYNAMIC_SEGMENT.test(segment) ? UNKNOWN_ID : segment)).join(SEGMENT_SEPARATOR)
}

/** The host specs' page, then every page and route theme.json declares, each once. */
function addressesOf(manifest: ThemeManifest): string[] {
  const pages = isRecord(manifest.pages) ? Object.keys(manifest.pages) : []
  const routes = Array.isArray(manifest.routes) ? manifest.routes.filter((route): route is string => typeof route === 'string') : []
  const declared = [...pages.map(pageAddress), ...routes.map((route) => `${ROUTE_PREFIX}${route}`)]
  return [...new Set([THEME_E2E_PATH, ...declared])]
}

/** `<html>`'s class list holding the theme's htmlClass. */
function htmlClassOf(manifest: ThemeManifest): RegExp {
  const htmlClass = String(manifest.htmlClass ?? '')
  expect(htmlClass, "the active answer carries the theme's htmlClass").toMatch(HTML_CLASS_PATTERN)
  return new RegExp(`(^|\\s)${htmlClass}(\\s|$)`)
}

/** Opens `address` cold and holds it to the host's contract for an installed theme. */
async function expectThemeRuns(page: Page, watch: PageWatch, address: string, htmlClass: RegExp): Promise<void> {
  const errorsBefore = watch.pageErrors.length
  await page.goto(address)
  const html = page.locator('html')
  await expect(html, `${address}: the host runs the theme`).toHaveAttribute(STATUS_ATTR, 'active', { timeout: THEME_LOAD_TIMEOUT_MS })
  await expect(html, `${address}: <html> carries the theme's class`).toHaveClass(htmlClass)
  await page.waitForTimeout(SETTLE_MS)
  await expect(html, `${address}: still active once the page drew`).toHaveAttribute(STATUS_ATTR, 'active')
  await expect(page.getByText(FALLBACK_NOTICE), `${address}: no fallback notice`).toHaveCount(0)
  await expect(page.getByTestId(PAGE_ERROR_TEST_ID), `${address}: no page error boundary`).toHaveCount(0)
  await expect(page.locator(NOT_FOUND_MARK), `${address}: not the 404`).toHaveCount(0)
  expect(watch.pageErrors.slice(errorsBefore), `${address}: no uncaught error`).toEqual([])
}

test.describe('PRD-261 B-03h — a Design Kit theme installed by the kit CLI', () => {
  test('installed with automatos-theme install --to the web app, it runs on every page and route it declares', { tag: INSTALLED_TAG }, async ({ page, request }) => {
    test.setTimeout(CLI_TIMEOUT_MS + ADDRESS_TIMEOUT_MS)
    const watch = watchPage(page)
    const install = await installWithKitCli(request, APP_URL)
    try {
      const addresses = addressesOf(install.manifest)
      test.setTimeout(CLI_TIMEOUT_MS + addresses.length * ADDRESS_TIMEOUT_MS)
      const htmlClass = htmlClassOf(install.manifest)
      await chooseStyle(page, install.style, APP_URL)
      for (const address of addresses) {
        await test.step(address, () => expectThemeRuns(page, watch, address, htmlClass))
      }
    } finally {
      await resetKitPackage(request, install.themeId)
    }
  })
})
