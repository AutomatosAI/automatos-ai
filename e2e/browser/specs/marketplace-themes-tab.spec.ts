import { execFileSync } from 'child_process'
import fs from 'fs'
import os from 'os'
import path from 'path'
import { expect, test, type APIRequestContext, type Page } from '@playwright/test'
import { chooseStyle } from '../support/hello-theme'

// PRD-261 C-04b (@host): the Marketplace's Themes tab, end to end on the local edition, against
// the real backend and the real files route (no route interception):
//
//   /marketplace › Themes → Publish theme: submit hello-c04b.autotheme (C-01b's form, C-01a's
//   route) → approve it (C-03a; the local operator is super_admin and the local edition has no
//   isolated review stage) → reload › Themes → the listing card → its detail → Add to my
//   workspace (C-04a, install by reference) → Use this theme → the choice is stored and the page
//   reloads into the theme → reload again → still active, and the server says the install
//   references the approved Marketplace package.
//
// Approve goes through the API: its page is C-03b's, proved by admin-theme-review.spec.ts. The
// package has its own theme id (fixtures/hello-c04b/build.py), so no other spec meets its listing
// or install. Chromium only: the lane's backend is one database for the whole run, and a listing
// install is per workspace, so a second browser would race this one for it. The spec removes the
// install and puts the choice back on Studio before and after.

const API_URL = process.env.E2E_API_URL ?? 'http://localhost:8000'
const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const BUILDER = path.resolve(__dirname, '..', 'fixtures', 'hello-c04b', 'build.py')
const PACKAGE_NAME = 'hello-c04b.autotheme'
const PACKAGE_TYPE = 'application/zip'
const THEME_ID = 'automatos-e2e/hello-c04b'
const LISTING_ID = 'automatos-e2e.hello-c04b'
const THEME_NAME = 'Hello Marketplace (C-04b)'
const MARKETPLACE_PATH = '/marketplace'
const SUBMIT_PATH = '/api/marketplace/themes/submit'
const THEMES_TAB = 'Themes'
const PUBLISH_THEME = 'Publish theme'
const SUBMIT_FOR_REVIEW = 'Submit for review'
const STATUS_ATTR = 'data-theme-host'
const STYLE_COOKIE = 'automatos-style'
const INSTALLED_STYLE_PREFIX = 'theme:install:'
const STUDIO = 'studio'
const PENDING = 'pending'
const MARKETPLACE_OWNER = 'marketplace'
const THEME_LOAD_TIMEOUT_MS = 30_000
// Q9: one open submission per author, and every spec's author is the local operator. Another
// W-C2 spec's submission waiting for review is a 409 until that spec reviews it, so the submit
// is retried for this long before the spec fails.
const OPEN_SUBMISSION_WAIT_MS = 45_000

interface Submission {
  readonly package: { readonly id: string; readonly status: string; readonly themeId: string }
}

interface InstalledRow {
  readonly installId: string
  readonly themeId: string
  readonly packageId: string
  readonly ownerType: string
  readonly reviewed: boolean
}

function buildPackage(): Buffer {
  const folder = fs.mkdtempSync(path.join(os.tmpdir(), 'hello-c04b-'))
  const out = path.join(folder, PACKAGE_NAME)
  execFileSync('python3', [BUILDER, '--out', out], { stdio: 'pipe' })
  return fs.readFileSync(out)
}

async function openThemesTab(page: Page): Promise<void> {
  await page.goto(MARKETPLACE_PATH)
  await page.getByRole('tab', { name: THEMES_TAB }).click()
}

/**
 * Submits the package through the tab's Publish theme form (201, or 200 with the version held
 * already), waiting out another spec's open submission (its 409 shows as the form's refusal).
 * The answer is read off the form's own request, so the spec approves the package it sent.
 */
async function submitThroughTab(page: Page): Promise<Submission> {
  const buffer = buildPackage()
  const publish = page.getByRole('region', { name: PUBLISH_THEME })
  const held: { answer?: Submission } = {}
  await expect(async () => {
    const answered = page.waitForResponse((response) =>
      response.url().includes(SUBMIT_PATH) && response.request().method() === 'POST')
    await publish.getByTestId('publish-theme-input').setInputFiles({ name: PACKAGE_NAME, mimeType: PACKAGE_TYPE, buffer })
    await publish.getByRole('button', { name: SUBMIT_FOR_REVIEW }).click()
    const response = await answered
    expect(response.ok(), `POST ${SUBMIT_PATH} answered ${response.status()}: ${await response.text()}`).toBe(true)
    held.answer = (await response.json()) as Submission
    await expect(publish.getByTestId('publish-result')).toContainText(THEME_ID)
  }).toPass({ timeout: OPEN_SUBMISSION_WAIT_MS })
  expect(held.answer, 'the submission answered').toBeDefined()
  return held.answer as Submission
}

/** Approves the version unless an earlier run already did (a person reviews every version, D3). */
async function approve(request: APIRequestContext, submission: Submission): Promise<void> {
  if (submission.package.status !== PENDING) return
  const response = await request.post(`${API_URL}/api/admin/themes/${submission.package.id}/approve`)
  expect(response.ok(), `POST …/approve answered ${response.status()}: ${await response.text()}`).toBe(true)
}

async function installedRows(request: APIRequestContext): Promise<InstalledRow[]> {
  const response = await request.get(`${API_URL}/api/themes/installed`)
  expect(response.status()).toBe(200)
  return ((await response.json()) as { installs: InstalledRow[] }).installs
}

/** Uninstall this theme and choose Studio, so the run starts and ends without it. */
async function reset(request: APIRequestContext): Promise<void> {
  for (const row of await installedRows(request)) {
    if (row.themeId === THEME_ID) {
      expect((await request.delete(`${API_URL}/api/themes/installs/${row.installId}`)).status()).toBe(200)
    }
  }
  expect((await request.put(`${API_URL}/api/themes/choice`, { data: { choice: STUDIO } })).status()).toBe(200)
}

async function expectThemeActive(page: Page): Promise<void> {
  await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, 'active', { timeout: THEME_LOAD_TIMEOUT_MS })
  await expect(page.getByTestId('hello-theme-frame')).toBeVisible()
}

test.describe('Marketplace › Themes', () => {
  test.beforeEach(async ({ page, browserName }) => {
    test.skip(browserName !== 'chromium', 'one database per run: one browser installs the listing (see the header)')
    await reset(page.request)
    await chooseStyle(page, STUDIO, APP_URL)
  })

  test.afterEach(async ({ page, browserName }) => {
    if (browserName === 'chromium') await reset(page.request)
  })

  test('submit, approve, add from the Themes tab, use it: the theme is active', { tag: '@host' }, async ({ page }) => {
    await openThemesTab(page)
    const submission = await submitThroughTab(page)
    expect(submission.package.themeId).toBe(THEME_ID)
    await approve(page.request, submission)

    await openThemesTab(page) // a fresh load, so the grid lists the version just approved
    const card = page.getByTestId(`theme-listing-${LISTING_ID}`)
    await expect(card).toContainText(THEME_NAME)
    await card.click()

    const detail = page.getByTestId('theme-listing-detail')
    await detail.getByRole('button', { name: 'Add to my workspace' }).click()
    await detail.getByRole('button', { name: 'Use this theme' }).click()

    // A reviewed version needs no opt-in: the choice is stored and the page reloads into the theme.
    await expectThemeActive(page)
    const [install] = (await installedRows(page.request)).filter((row) => row.themeId === THEME_ID)
    expect(install.packageId, 'the install references the approved Marketplace package').toBe(submission.package.id)
    expect(install.ownerType).toBe(MARKETPLACE_OWNER)
    expect(install.reviewed).toBe(true)
    const active = await page.request.get(`${API_URL}/api/themes/active`)
    expect(((await active.json()) as { choice: string }).choice).toBe(install.installId)
    const cookies = await page.context().cookies(APP_URL)
    expect(cookies.find((cookie) => cookie.name === STYLE_COOKIE)?.value).toBe(`${INSTALLED_STYLE_PREFIX}${install.installId}`)

    await page.reload()
    await expectThemeActive(page)
  })
})
