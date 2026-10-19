import fs from 'fs'
import path from 'path'
import { expect, test, type APIRequestContext, type Locator, type Page } from '@playwright/test'
import { chooseStyle } from '../support/hello-theme'

// PRD-261 A-07d (@host): Settings › Appearance › Themes, end to end on the local edition, against
// the real backend and the real files route (no route interception):
//
//   upload hello.autotheme (built by the lane's "Build the test theme packages" step from
//   orchestrator/tests/fixtures/themes/build.py) → the scan report and "Private · not reviewed" →
//   Use this theme → the opt-in dialog with the server's wording → opt in → the choice is stored
//   and the page reloads into the theme → reload again → still active, and the server says so.
//
// Chromium only: the lane's backend is one database for the whole run, and an opt-in is recorded
// per member and package for good (there is no opt-out route), so a second browser would find the
// opt-in already made and could not prove the dialog. The spec removes the install and puts the
// choice back on Studio before and after, so no other spec starts on this theme.

const API_URL = process.env.E2E_API_URL ?? 'http://localhost:8000'
const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const PACKAGES_DIR = process.env.E2E_THEME_PACKAGES ?? ''
const HELLO_PACKAGE = 'hello.autotheme'
const HELLO_THEME_ID = 'automatos-e2e/hello'
const HELLO_NAME = 'Hello theme'
const SETTINGS_PATH = '/settings'
const STATUS_ATTR = 'data-theme-host'
const STYLE_COOKIE = 'automatos-style'
const INSTALLED_STYLE_PREFIX = 'theme:install:'
const STUDIO = 'studio'
const OPT_IN_WORDING = /runs with your access to all your workspaces/
const PRIVATE_NOT_REVIEWED = 'Private · not reviewed'
const THEME_LOAD_TIMEOUT_MS = 30_000

interface InstalledRow {
  readonly installId: string
  readonly themeId: string
}

async function installedRows(request: APIRequestContext): Promise<InstalledRow[]> {
  const response = await request.get(`${API_URL}/api/themes/installed`)
  expect(response.status()).toBe(200)
  return ((await response.json()) as { installs: InstalledRow[] }).installs
}

/** Uninstall every hello install and choose Studio, so the run starts and ends without the theme. */
async function resetHello(request: APIRequestContext): Promise<void> {
  for (const row of await installedRows(request)) {
    if (row.themeId === HELLO_THEME_ID) {
      expect((await request.delete(`${API_URL}/api/themes/installs/${row.installId}`)).status()).toBe(200)
    }
  }
  expect((await request.put(`${API_URL}/api/themes/choice`, { data: { choice: STUDIO } })).status()).toBe(200)
}

async function openThemes(page: Page): Promise<void> {
  await page.goto(SETTINGS_PATH)
  await page.getByRole('navigation', { name: 'Sections' }).getByRole('button', { name: 'Appearance', exact: true }).click()
  await expect(page.getByRole('region', { name: 'Installed themes' })).toBeVisible()
}

function helloRow(page: Page): Locator {
  return page.locator('[data-testid^="installed-theme-"]').filter({ hasText: HELLO_NAME })
}

async function expectHelloActive(page: Page): Promise<void> {
  await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, 'active', { timeout: THEME_LOAD_TIMEOUT_MS })
  await expect(page.getByTestId('hello-theme-frame')).toBeVisible()
}

test.describe('Settings › Appearance › Themes', () => {
  test.beforeEach(async ({ page, browserName }) => {
    test.skip(browserName !== 'chromium', 'one database per run: the opt-in is proved once (see the header)')
    expect(PACKAGES_DIR, 'E2E_THEME_PACKAGES is set by the lane step that builds the packages').not.toBe('')
    await resetHello(page.request)
    await chooseStyle(page, STUDIO, APP_URL)
  })

  test.afterEach(async ({ page, browserName }) => {
    if (browserName === 'chromium') await resetHello(page.request)
  })

  test('upload hello.autotheme, opt in, choose, reload: the theme is active', { tag: '@host' }, async ({ page }) => {
    await openThemes(page)
    await page.getByTestId('install-theme-input').setInputFiles(path.join(PACKAGES_DIR, HELLO_PACKAGE))

    const result = page.getByTestId('theme-upload-result')
    await expect(result).toContainText(PRIVATE_NOT_REVIEWED)
    await expect(result.getByTestId('theme-scan-report')).toContainText('Sizes')

    const row = helloRow(page)
    await expect(row).toContainText(PRIVATE_NOT_REVIEWED)
    await row.getByRole('button', { name: 'Use this theme' }).click()

    const dialog = page.getByTestId('theme-opt-in-dialog')
    await expect(dialog.getByTestId('theme-opt-in-wording')).toHaveText(OPT_IN_WORDING)
    await dialog.getByRole('button', { name: 'Opt in and use' }).click()

    // The choice is stored, the style cookie is written and the page reloads into the theme.
    await expectHelloActive(page)
    const [install] = (await installedRows(page.request)).filter((item) => item.themeId === HELLO_THEME_ID)
    const active = await page.request.get(`${API_URL}/api/themes/active`)
    expect(((await active.json()) as { choice: string }).choice).toBe(install.installId)
    const cookies = await page.context().cookies(APP_URL)
    expect(cookies.find((cookie) => cookie.name === STYLE_COOKIE)?.value).toBe(`${INSTALLED_STYLE_PREFIX}${install.installId}`)

    await page.reload()
    await expectHelloActive(page)
  })
})

// The package must exist before the run: a missing file is a lane failure, not a skip.
test('the lane built hello.autotheme', { tag: '@host' }, async ({ browserName }) => {
  test.skip(browserName !== 'chromium', 'checked once')
  expect(fs.existsSync(path.join(PACKAGES_DIR, HELLO_PACKAGE))).toBe(true)
})
