import { execFileSync } from 'child_process'
import { createHash } from 'crypto'
import fs from 'fs'
import os from 'os'
import path from 'path'
import { expect, test, type APIRequestContext, type Page, type Route } from '@playwright/test'
import { fulfillBackendJson } from '../support/backend-route'
import { THEME_E2E_PATH } from '../support/hello-theme'

// PRD-261 A-08c (@host): first paint for an installed theme, with import-map integrity. Nothing
// the server does is intercepted: fixtures/hello-a08c/build.py builds A-06a's hello package under
// its own id, it is uploaded to the local backend and opted in to, and the root layout fetches its
// boot on the server (frontend/lib/theme-host/boot-server.ts → GET /api/themes/boot/{installId}).
//
// 1. Under the `theme:install:<id>` cookie the server's HTML carries the theme's boot ('studio
//    hello') and an import map whose integrity pins dist/entry.js to the sha256 of the bytes the
//    files route serves; <html> has the theme's class before the first animation frame.
// 2. With the real entry the host gets past the import (it does not fail with `load`); with one
//    changed byte (route interception on the files proxy) the browser refuses the module and the
//    host falls back with reason `load`. Chromium and WebKit; Firefox is covered by the record
//    spec (R2, import-map integrity support), as in A-01.
//
// The browser's GET /api/themes/active is answered here for the tampered-chunk tests only, naming
// this install, so the local operator's stored choice (which other specs read) is never changed.

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const BACKEND_URL = process.env.E2E_BACKEND_URL ?? 'http://localhost:8000'
const BUILDER = path.resolve(__dirname, '..', 'fixtures', 'hello-a08c', 'build.py')
const PACKAGE_NAME = 'hello-a08c.autotheme'
const PACKAGE_TYPE = 'application/zip'
const STYLE_COOKIE = 'automatos-style'
const INSTALLED_STYLE_PREFIX = 'theme:install:'
const THEME_CLASS = 'hello'
const STUDIO_AND_THEME = ['studio', THEME_CLASS]
const ENTRY_FILE = 'dist/entry.js'
const STATUS_ATTR = 'data-theme-host'
const REASON_ATTR = 'data-theme-host-reason'
const HOST_SETTLED = /^(active|failed)$/
// The entry opens with a line comment ('// PRD-261 …'): the byte at this offset is inside it, so the
// changed module is still valid JavaScript and only its integrity pin can refuse it.
const TAMPERED_OFFSET = 3
/** 'P' becomes 'Q' (and back), so the comment stays a comment. */
const BYTE_P = 0x50
const BYTE_Q = 0x51
const LINE_COMMENT = '//'
const DROPPED_HEADERS = new Set(['content-length', 'content-encoding'])
// Without this marker a fresh browser is moved to Studio once (contexts/ui-style-context.tsx).
const APPEARANCE_DEFAULTS = { key: 'automatos-appearance-defaults', value: '2026-09-17-studio-dark' }
const IMPORT_MAP = /<script type="importmap">([\s\S]*?)<\/script>/
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

type FirstFrameWindow = Window & { __firstFrameClasses?: string[] }

function buildPackage(): Buffer {
  const folder = fs.mkdtempSync(path.join(os.tmpdir(), 'hello-a08c-'))
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

function entryPath(theme: Installed): string {
  return `/api/themes/files/${theme.digest}/${ENTRY_FILE}`
}

async function chooseInstalled(page: Page, theme: Installed): Promise<void> {
  await page.addInitScript(({ key, value }) => window.localStorage.setItem(key, value), APPEARANCE_DEFAULTS)
  await page.context().addCookies([{ name: STYLE_COOKIE, value: `${INSTALLED_STYLE_PREFIX}${theme.installId}`, url: APP_URL }])
}

/** The JSON a script element of the server's HTML holds. */
function scriptJson(html: string, pattern: RegExp): Record<string, unknown> {
  const body = pattern.exec(html)?.[1]
  expect(body, `the server's HTML has ${pattern}`).toBeDefined()
  return JSON.parse(body ?? 'null') as Record<string, unknown>
}

/**
 * The active-theme answer naming this install. The manifest is in the shape the client loader
 * reads (lib/theme-host/manifest.ts parseSpikeManifest), pointing at the package's own files.
 */
function activeAnswer(theme: Installed): Record<string, unknown> {
  const manifest = {
    schema: 'automatos-theme-spike/v1',
    id: 'automatos-e2e/hello-a08c',
    name: 'Hello first paint (A-08c)',
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
    themeId: manifest.id,
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

async function answerActive(page: Page, theme: Installed): Promise<void> {
  await page.route('**/api/themes/active', (route) => fulfillBackendJson(route, 200, activeAnswer(theme)))
}

/** The same bytes with one changed inside the opening line comment. */
function withOneChangedByte(original: Buffer): Buffer {
  expect(original.subarray(0, LINE_COMMENT.length).toString('utf8'), 'the entry opens with a line comment').toBe(LINE_COMMENT)
  const changed = Buffer.from(original)
  changed[TAMPERED_OFFSET] = changed[TAMPERED_OFFSET] === BYTE_Q ? BYTE_P : BYTE_Q
  return changed
}

/** Serves the entry with one changed byte; a ranged status probe goes through as it is. */
async function tamperEntry(route: Route): Promise<void> {
  if ((await route.request().headerValue('range')) !== null) {
    await route.continue()
    return
  }
  const real = await route.fetch()
  const headers = Object.fromEntries(Object.entries(real.headers()).filter(([name]) => !DROPPED_HEADERS.has(name)))
  await route.fulfill({ status: real.status(), headers, body: withOneChangedByte(await real.body()) })
}

async function settledHost(page: Page): Promise<{ status: string | null; reason: string | null }> {
  const html = page.locator('html')
  await expect(html).toHaveAttribute(STATUS_ATTR, HOST_SETTLED)
  return { status: await html.getAttribute(STATUS_ATTR), reason: await html.getAttribute(REASON_ATTR) }
}

test.describe('PRD-261 A-08c — first paint for an installed theme, with integrity', () => {
  test("the server's HTML carries the theme's boot and pins its entry's sha256", { tag: '@host' }, async ({ request }) => {
    const theme = await install(request)
    const cookie = `${STYLE_COOKIE}=${INSTALLED_STYLE_PREFIX}${theme.installId}`
    const page = await request.get(THEME_E2E_PATH, { headers: { cookie } })
    expect(page.ok()).toBe(true)
    const html = await page.text()
    expect(scriptJson(html, BOOT_ELEMENT).addClasses).toEqual(STUDIO_AND_THEME)

    const entry = await request.get(entryPath(theme))
    expect(entry.status()).toBe(200)
    const expected = `sha256-${createHash('sha256').update(await entry.body()).digest('base64')}`
    const integrity = scriptJson(html, IMPORT_MAP).integrity as Record<string, string> | undefined
    expect(integrity?.[entryPath(theme)]).toBe(expected)
  })

  test("puts the theme's class on <html> before the first frame", { tag: '@host' }, async ({ page, request }) => {
    const theme = await install(request)
    await chooseInstalled(page, theme)
    // The server holds no choice of this install, so ThemeChoiceSync would reload the tab mid-read.
    await answerActive(page, theme)
    await page.addInitScript(() => {
      requestAnimationFrame(() => {
        ;(window as FirstFrameWindow).__firstFrameClasses = Array.from(document.documentElement.classList)
      })
    })
    await page.goto(THEME_E2E_PATH)
    await page.waitForFunction(() => (window as FirstFrameWindow).__firstFrameClasses !== undefined)
    const classes = await page.evaluate(() => (window as FirstFrameWindow).__firstFrameClasses)
    expect(classes).toEqual(expect.arrayContaining(STUDIO_AND_THEME))
  })

  test('loads the real entry past the import: no `load` failure', { tag: '@host' }, async ({ page, request }) => {
    const theme = await install(request)
    await chooseInstalled(page, theme)
    await answerActive(page, theme)
    const entryRequests: string[] = []
    page.on('request', (sent) => {
      if (new URL(sent.url()).pathname === entryPath(theme)) entryRequests.push(sent.url())
    })
    await page.goto(THEME_E2E_PATH)
    const host = await settledHost(page)
    expect(entryRequests.length, 'the host asked for the entry').toBeGreaterThan(0)
    expect(host.reason, `the untampered entry passes its integrity pin (host ${host.status})`).not.toBe('load')
  })

  test('refuses a tampered entry chunk: the host falls back with reason `load`', { tag: '@host' }, async ({ page, request }) => {
    const theme = await install(request)
    await chooseInstalled(page, theme)
    await answerActive(page, theme)
    let tampered = 0
    await page.route(`**${entryPath(theme)}`, async (route) => {
      tampered += 1
      await tamperEntry(route)
    })
    await page.goto(THEME_E2E_PATH)
    expect(await settledHost(page)).toEqual({ status: 'failed', reason: 'load' })
    expect(tampered, 'the browser asked for the entry, and got the changed bytes').toBeGreaterThan(0)
  })
})
