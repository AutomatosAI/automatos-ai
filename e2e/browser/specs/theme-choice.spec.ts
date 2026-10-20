import { execFileSync } from 'child_process'
import fs from 'fs'
import os from 'os'
import path from 'path'
import { expect, test, type APIRequestContext, type Browser, type Page } from '@playwright/test'
import { fulfillBackendJson } from '../support/backend-route'
import { THEME_E2E_PATH } from '../support/hello-theme'

// PRD-261 A-02c (@host): the server holds the theme choice (PA-12), the style cookie caches it.
//
// 1. A member picks an installed theme in the theme menu of one browser; that browser stores the
//    choice (PUT /api/themes/choice with the install's id) and reloads into it. A fresh browser
//    (no cookie, no storage) asks GET /api/themes/active once, is told that choice, rewrites its
//    cookie and reloads exactly once: its second document is painted by the server as the theme.
// 2. A fresh browser in a workspace whose default is an installed theme lands on it the same way:
//    GET /api/themes/active answers a member with no choice of their own with the default's
//    install id (services/themes/defaults.py resolve_active, proved by the orchestrator tests).
//
// fixtures/hello-a02c/build.py builds the package; it is uploaded and opted in to for real, so the
// root layout's server-side boot (GET /api/themes/boot/{installId}) paints it. The choice routes
// are answered here, from what the first browser stored: the local operator's stored choice is
// read once by every other spec's tab (A-02c), so this spec never changes it. A reviewed package
// (one that may be a default) can't be made in the lane before the Marketplace review (C-03a), so
// test 2's answer is the one resolve_active gives for that case. Chromium only: one database.

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const BACKEND_URL = process.env.E2E_BACKEND_URL ?? 'http://localhost:8000'
const BUILDER = path.resolve(__dirname, '..', 'fixtures', 'hello-a02c', 'build.py')
const PACKAGE_NAME = 'hello-a02c.autotheme'
const PACKAGE_TYPE = 'application/zip'
const THEME_ID = 'automatos-e2e/hello-a02c'
const THEME_NAME = 'Hello choice A-02c'
const THEME_CLASS = 'hello'
const PRIVATE_NOT_REVIEWED = 'Private · not reviewed'
const STYLE_COOKIE = 'automatos-style'
const INSTALLED_STYLE_PREFIX = 'theme:install:'
const ACTIVE_ROUTE = '**/api/themes/active'
const CHOICE_ROUTE = '**/api/themes/choice'
const ENTRY_FILE = 'dist/entry.js'
// Without this marker a browser is moved to Studio once (contexts/ui-style-context.tsx).
const APPEARANCE_DEFAULTS = { key: 'automatos-appearance-defaults', value: '2026-09-17-studio-dark' }
// The project's desktop width: the Studio header, with the theme menu, renders from 1024 px.
const DESKTOP_VIEWPORT = { width: 1440, height: 900 }
const OK = 200
const BOOT_ELEMENT = /<script\b[^>]*\bid="automatos-theme-boot"[^>]*>([\s\S]*?)<\/script>/

interface UploadAnswer {
  readonly package: { readonly id: string; readonly digest: string; readonly version: string }
  readonly install: { readonly id: string } | null
}

interface Installed {
  readonly installId: string
  readonly packageId: string
  readonly digest: string
  readonly version: string
}

function buildPackage(): Buffer {
  const folder = fs.mkdtempSync(path.join(os.tmpdir(), 'hello-a02c-'))
  const out = path.join(folder, PACKAGE_NAME)
  execFileSync('python3', [BUILDER, '--out', out], { stdio: 'pipe' })
  return fs.readFileSync(out)
}

/** Uploads the package (201, or 200 when it was uploaded before) and opts in to it. */
async function install(request: APIRequestContext): Promise<Installed> {
  const upload = await request.post(`${BACKEND_URL}/api/themes/import`, {
    multipart: { file: { name: PACKAGE_NAME, mimeType: PACKAGE_TYPE, buffer: buildPackage() } },
  })
  expect(upload.ok(), `POST /api/themes/import answered ${upload.status()}: ${await upload.text()}`).toBe(true)
  const answer = (await upload.json()) as UploadAnswer
  expect(answer.install, 'the upload installs the theme in the workspace').not.toBeNull()
  const installId = answer.install?.id ?? ''
  const optIn = await request.post(`${BACKEND_URL}/api/themes/installs/${installId}/opt-in`, {
    data: { packageId: answer.package.id },
  })
  expect(optIn.ok(), `POST …/opt-in answered ${optIn.status()}: ${await optIn.text()}`).toBe(true)
  return { installId, packageId: answer.package.id, digest: answer.package.digest, version: answer.package.version }
}

/**
 * GET /api/themes/active naming the install as the member's choice (or the workspace default),
 * with a manifest in the shape the client loader reads (lib/theme-host/manifest.ts).
 */
function activeAnswer(theme: Installed, reviewed: boolean): Record<string, unknown> {
  const manifest = {
    schema: 'automatos-theme-spike/v1',
    id: THEME_ID,
    name: THEME_NAME,
    version: theme.version,
    engines: { sdk: '^1.0.0' },
    base: 'studio',
    htmlClass: THEME_CLASS,
    entry: ENTRY_FILE,
    styleSheet: 'dist/theme.css',
    preset: 'looks/presets/default.look.json',
    slots: ['frame@1', 'lookEditor@1', 'styleSheet@1'],
  }
  const active = {
    installId: theme.installId,
    themeId: THEME_ID,
    packageId: theme.packageId,
    version: theme.version,
    digest: theme.digest,
    manifest,
    reviewed,
    revoked: null,
    needsUpdate: false,
  }
  return { choice: theme.installId, theme: active }
}

/** The first browser: on Studio, its server choice Studio, and every PUT of a choice recorded. */
async function pickingBrowser(browser: Browser, stored: string[]): Promise<Page> {
  const context = await browser.newContext({ baseURL: APP_URL, viewport: DESKTOP_VIEWPORT })
  await context.addInitScript(({ key, value }) => window.localStorage.setItem(key, value), APPEARANCE_DEFAULTS)
  await context.addCookies([{ name: STYLE_COOKIE, value: 'studio', url: APP_URL }])
  const page = await context.newPage()
  await page.route(ACTIVE_ROUTE, (route) => fulfillBackendJson(route, OK, { choice: 'studio', theme: null }))
  await page.route(CHOICE_ROUTE, async (route) => {
    const body = route.request().method() === 'PUT' ? (route.request().postDataJSON() as { choice?: unknown }) : null
    if (typeof body?.choice === 'string') stored.push(body.choice)
    await fulfillBackendJson(route, OK, { choice: body?.choice ?? 'studio', theme: null })
  })
  return page
}

/** The HTML of every document the page loads: the first navigation, then every reload. */
function watchDocuments(page: Page): Promise<string>[] {
  const documents: Promise<string>[] = []
  page.on('response', (response) => {
    const request = response.request()
    if (request.resourceType() === 'document' && request.frame() === page.mainFrame()) documents.push(response.text())
  })
  return documents
}

/** The classes the server's theme boot puts on <html> in a document's HTML. */
function bootClasses(html: string): unknown {
  const body = BOOT_ELEMENT.exec(html)?.[1]
  expect(body, "the server's HTML has the theme boot").toBeDefined()
  return (JSON.parse(body ?? 'null') as { addClasses?: unknown } | null)?.addClasses
}

async function styleCookie(page: Page): Promise<string | undefined> {
  return (await page.context().cookies(APP_URL)).find((cookie) => cookie.name === STYLE_COOKIE)?.value
}

/** A fresh browser told `answer` lands on the install: its cookie, one reload, the server's paint. */
async function expectFreshBrowserLands(page: Page, theme: Installed, answer: Record<string, unknown>): Promise<void> {
  const documents = watchDocuments(page)
  await page.route(ACTIVE_ROUTE, (route) => fulfillBackendJson(route, OK, answer))
  expect(await styleCookie(page), 'a fresh browser has no style cookie').toBeUndefined()
  await page.goto(THEME_E2E_PATH)
  await expect.poll(() => styleCookie(page)).toBe(`${INSTALLED_STYLE_PREFIX}${theme.installId}`)
  await expect.poll(() => documents.length).toBe(2)
  // The first document is Studio's; the second is painted by the server as the theme.
  expect(bootClasses(await documents[0])).not.toEqual(expect.arrayContaining([THEME_CLASS]))
  expect(bootClasses(await documents[1])).toEqual(expect.arrayContaining(['studio', THEME_CLASS]))
  await page.waitForLoadState('networkidle')
  expect(documents.length, 'the tab reloads once, never again').toBe(2)
}

test.describe('PRD-261 A-02c — the server holds the theme choice', () => {
  test.beforeEach(({ browserName }) => {
    test.skip(browserName !== 'chromium', 'one database per run: the install is proved once')
  })

  test('a choice made in one browser applies in a fresh browser on its first load', { tag: '@host' }, async ({ browser, page, request }) => {
    const theme = await install(request)
    const stored: string[] = []
    const picking = await pickingBrowser(browser, stored)
    await picking.goto(THEME_E2E_PATH)
    await picking.getByRole('button', { name: 'Appearance' }).first().click()
    const item = picking.getByRole('menuitemradio', { name: new RegExp(THEME_NAME) })
    await expect(item).toContainText(PRIVATE_NOT_REVIEWED)
    await item.click()
    await expect.poll(() => stored).toEqual([theme.installId])
    await picking.context().close()

    await expectFreshBrowserLands(page, theme, { ...activeAnswer(theme, false), choice: stored[0] })
  })

  test("a fresh browser lands on the workspace's default installed theme", { tag: '@host' }, async ({ page, request }) => {
    const theme = await install(request)
    await expectFreshBrowserLands(page, theme, activeAnswer(theme, true))
  })
})
