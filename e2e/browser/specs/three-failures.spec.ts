import { execFileSync } from 'child_process'
import fs from 'fs'
import os from 'os'
import path from 'path'
import { expect, test, type APIRequestContext, type Page, type Request } from '@playwright/test'
import { fulfillBackendJson } from '../support/backend-route'
import { THEME_E2E_PATH, THEME_SDK_ROOT } from '../support/hello-theme'
import '../support/page-globals'
import { watchPage } from '../support/watch'

// PRD-261 A-04c (@host): three failures in ten minutes, and the safe-mode page. The package is
// A-06a's hello-throws variant (fixtures/hello-a04c/build.py), uploaded to the local backend and
// opted in to; its entry throws while it is evaluated, so every load fails with reason `load`.
//
// 1. Loaded three times (a reload between each), the third failure puts the member back on
//    Studio: the page sends PUT /api/themes/choice { choice: 'studio' } to the real backend, whose
//    answer (the active theme after the write) has the choice 'studio'; the style cookie is
//    'studio', the revert's notice shows, and the next load paints Studio with no theme code.
// 2. /safe-mode loads no theme code under the installed theme's cookie, and its link back keeps
//    the tab in safe mode.
//
// The browser's GET /api/themes/active is answered here, naming this install (A-08c's pattern),
// so the local operator's stored choice is never set to a broken theme other specs would read.

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const BACKEND_URL = process.env.E2E_BACKEND_URL ?? 'http://localhost:8000'
const BUILDER = path.resolve(__dirname, '..', 'fixtures', 'hello-a04c', 'build.py')
const PACKAGE_NAME = 'hello-a04c.autotheme'
const PACKAGE_TYPE = 'application/zip'
const THEME_ID = 'automatos-e2e/hello-a04c'
const STYLE_COOKIE = 'automatos-style'
const INSTALLED_STYLE_PREFIX = 'theme:install:'
const STUDIO = 'studio'
const THEME_CLASS = /\bhello\b/
const ENTRY_FILE = 'dist/entry.js'
const STATUS_ATTR = 'data-theme-host'
const REASON_ATTR = 'data-theme-host-reason'
const ACTIVE_ROUTE = '**/api/themes/active'
const CHOICE_PATH = '/api/themes/choice'
const FILES_PATH = '/api/themes/files/'
const WORKSPACE_PATH = '/api/workspaces/current'
const SAFE_MODE_PAGE = '/safe-mode'
const SAFE_MODE_BACK = /\/chat\?safe_mode=themes$/
const REVERTED_NOTICE = /failed three times in ten minutes/
const SAFE_MODE_NOTICE = /Safe mode: themes are off in this tab/
const SAFE_MODE_BACK_TEST_ID = 'safe-mode-back'
// Without this marker a fresh browser is moved to Studio once (contexts/ui-style-context.tsx).
const APPEARANCE_DEFAULTS = { key: 'automatos-appearance-defaults', value: '2026-09-17-studio-dark' }

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
  const folder = fs.mkdtempSync(path.join(os.tmpdir(), 'hello-a04c-'))
  const out = path.join(folder, PACKAGE_NAME)
  execFileSync('python3', [BUILDER, '--out', out], { stdio: 'pipe' })
  return fs.readFileSync(out)
}

/** Uploads the package (201, or 200 when a parallel project already did) and opts in. */
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
 * The active-theme answer naming this install, with the manifest in the shape the client loader
 * reads (lib/theme-host/manifest.ts parseSpikeManifest), pointing at the package's own files.
 */
function activeAnswer(theme: Installed): Record<string, unknown> {
  const manifest = {
    schema: 'automatos-theme-spike/v1',
    id: THEME_ID,
    name: 'Hello throws (A-04c)',
    version: theme.version,
    engines: { sdk: '^1.0.0' },
    base: 'studio',
    htmlClass: 'hello',
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
    reviewed: false,
    revoked: null,
    needsUpdate: false,
  }
  return { choice: theme.installId, theme: active }
}

/** Puts this browser on the install's style, and answers its active-theme read with the install. */
async function chooseInstalled(page: Page, theme: Installed): Promise<void> {
  await page.addInitScript(({ key, value }) => window.localStorage.setItem(key, value), APPEARANCE_DEFAULTS)
  await page.context().addCookies([{ name: STYLE_COOKIE, value: `${INSTALLED_STYLE_PREFIX}${theme.installId}`, url: APP_URL }])
  await page.route(ACTIVE_ROUTE, (route) => fulfillBackendJson(route, 200, activeAnswer(theme)))
}

function isChoicePut(sent: Request): boolean {
  return sent.method() === 'PUT' && new URL(sent.url()).pathname === CHOICE_PATH
}

/** The paths of the requests that reached for theme code: the shims or a theme's files. */
function themeCodePaths(urls: readonly string[]): string[] {
  return urls
    .map((url) => new URL(url).pathname)
    .filter((pathname) => pathname.startsWith(THEME_SDK_ROOT) || pathname.startsWith(FILES_PATH))
}

async function styleCookie(page: Page): Promise<string | undefined> {
  return (await page.context().cookies(APP_URL)).find((cookie) => cookie.name === STYLE_COOKIE)?.value
}

async function expectFailedLoad(page: Page): Promise<void> {
  const html = page.locator('html')
  await expect(html).toHaveAttribute(STATUS_ATTR, 'failed')
  await expect(html).toHaveAttribute(REASON_ATTR, 'load')
}

/** The app's first GET of its workspace: the client has mounted, so the host's effect has run. */
function waitForWorkspace(page: Page): Promise<unknown> {
  return page.waitForResponse((response) => response.url().endsWith(WORKSPACE_PATH) && response.request().method() === 'GET')
}

test.describe('PRD-261 A-04c — three failures in ten minutes, and safe mode', () => {
  test('the third failure in ten minutes puts the server choice back on Studio', { tag: '@host' }, async ({ page, request }) => {
    const theme = await install(request)
    await chooseInstalled(page, theme)
    const choicePuts: Request[] = []
    page.on('request', (sent) => {
      if (isChoicePut(sent)) choicePuts.push(sent)
    })

    await page.goto(THEME_E2E_PATH)
    await expectFailedLoad(page)
    await page.reload()
    await expectFailedLoad(page)
    expect(choicePuts, 'two failures leave the choice alone').toHaveLength(0)

    const reverted = page.waitForRequest(isChoicePut)
    await page.reload()
    const sent = await reverted
    expect(sent.postDataJSON()).toEqual({ choice: STUDIO })
    const answered = await sent.response()
    expect(answered?.ok(), `PUT ${CHOICE_PATH} answered ${answered?.status()}`).toBe(true)
    expect(((await answered?.json()) as { choice?: unknown } | undefined)?.choice, "the server's choice after the write").toBe(STUDIO)
    await expect(page.getByText(REVERTED_NOTICE)).toBeVisible()
    expect(await styleCookie(page)).toBe(STUDIO)

    // The next load is Studio from the server's first paint: no theme status, class or code.
    await page.unroute(ACTIVE_ROUTE)
    const watch = watchPage(page)
    const appRan = waitForWorkspace(page)
    await page.reload()
    await appRan
    const html = page.locator('html')
    await expect(html).not.toHaveAttribute(STATUS_ATTR)
    await expect(html).not.toHaveClass(THEME_CLASS)
    expect(themeCodePaths(watch.requests)).toEqual([])
    expect(choicePuts, 'one revert').toHaveLength(1)
  })

  test('/safe-mode loads no theme code, and its link back stays in safe mode', { tag: '@host' }, async ({ page, request }) => {
    const theme = await install(request)
    await chooseInstalled(page, theme)
    const watch = watchPage(page)

    await page.goto(SAFE_MODE_PAGE)
    const html = page.locator('html')
    await expect(html).toHaveAttribute(STATUS_ATTR, 'safe-mode')
    await expect(page.getByText(SAFE_MODE_NOTICE)).toBeVisible()
    expect(themeCodePaths(watch.requests)).toEqual([])
    expect(await page.evaluate(() => typeof window.__AUTOMATOS_THEME_HOST__)).toBe('undefined')

    await page.getByTestId(SAFE_MODE_BACK_TEST_ID).click()
    await page.waitForURL(SAFE_MODE_BACK)
    await expect(html).toHaveAttribute(STATUS_ATTR, 'safe-mode')
    await page.reload()
    await expect(html).toHaveAttribute(STATUS_ATTR, 'safe-mode')
    expect(themeCodePaths(watch.requests)).toEqual([])
    expect(await page.evaluate(() => typeof window.__AUTOMATOS_THEME_HOST__)).toBe('undefined')
  })
})
