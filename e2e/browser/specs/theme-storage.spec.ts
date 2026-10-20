import { execFileSync } from 'child_process'
import fs from 'fs'
import os from 'os'
import path from 'path'
import { expect, test, type APIRequestContext, type Page } from '@playwright/test'
import { fulfillBackendJson } from '../support/backend-route'
import { THEME_E2E_PATH } from '../support/hello-theme'
import { watchPage } from '../support/watch'

// PRD-261 E3c (@host): a theme's state, kept through the SDK's storage area, lasts and is the theme's
// own. fixtures/hello-e3/build.py builds two packages that differ only in their theme id; both are
// uploaded to the local backend and opted in to, so the root layout's boot gives each its registry
// entry and the host publishes the storage scope (components/theme-host/theme-state-scope.tsx).
//
// 1. Under hello-e3 the frame's greeting starts empty; its button keeps one, and this browser holds it
//    under the documented key (plan §2.2.1): this theme, the local edition's one user `local`.
// 2. After a reload the frame reads the greeting back.
// 3. Under the other theme id the frame reads nothing, and hello-e3's state is still kept.
//
// The browser's GET /api/themes/active is answered here, naming the install the tab is on (A-08c's
// pattern), so the local operator's stored choice, which other specs read, is never changed.

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const BACKEND_URL = process.env.E2E_BACKEND_URL ?? 'http://localhost:8000'
const BUILDER = path.resolve(__dirname, '..', 'fixtures', 'hello-e3', 'build.py')
const PACKAGE_TYPE = 'application/zip'
const PACKAGE_SUFFIX = '.autotheme'
const STYLE_COOKIE = 'automatos-style'
const INSTALLED_STYLE_PREFIX = 'theme:install:'
const ACTIVE_ROUTE = '**/api/themes/active'
const STATUS_ATTR = 'data-theme-host'
const RUNNING = 'active'
const ENTRY_FILE = 'dist/entry.js'
/** The SDK range the packages ask for (build.py ENGINES): the first SDK with the storage area. */
const SDK_RANGE = '^1.1.0'
// Without this marker a fresh browser is moved to Studio once (contexts/ui-style-context.tsx).
const APPEARANCE_DEFAULTS = { key: 'automatos-appearance-defaults', value: '2026-09-17-studio-dark' }
/** lib/theme-sdk/internal/theme-state-store.ts: `automatos-theme-state:v1:<b|s>:<themeId>:<scope>`. */
const STATE_KEY_PREFIX = 'automatos-theme-state:v1:'
const BROWSER_LIFETIME = 'b'
/** The local edition's one user (theme-state-scope.tsx LOCAL_EDITION_USER_KEY); the lane runs it. */
const LOCAL_USER = 'local'
/** fixtures/hello-e3/entry.js: the key it keeps and the value its button keeps. */
const GREETING_KEY = 'greeting'
const GREETING = 'kept by hello-e3'
const NOTHING_KEPT = JSON.stringify(null)
const STATE_TEST_ID = 'hello-e3-state'
const SAVE_TEST_ID = 'hello-e3-save'

interface StoragePackage {
  /** build.py's --variant. */
  readonly variant: string
  readonly themeId: string
  readonly name: string
}

const HELLO_E3: StoragePackage = { variant: 'hello-e3', themeId: 'automatos-e2e/hello-e3', name: 'Hello storage (E3c)' }
const HELLO_E3_OTHER: StoragePackage = {
  variant: 'hello-e3-other',
  themeId: 'automatos-e2e/hello-e3-other',
  name: 'Hello storage, another theme (E3c)',
}

interface UploadAnswer {
  readonly package: { readonly id: string; readonly digest: string; readonly version: string }
  readonly install: { readonly id: string } | null
}

interface Installed {
  readonly pkg: StoragePackage
  readonly installId: string
  readonly packageId: string
  readonly digest: string
  readonly version: string
}

function buildPackage(pkg: StoragePackage): Buffer {
  const folder = fs.mkdtempSync(path.join(os.tmpdir(), `${pkg.variant}-`))
  const out = path.join(folder, `${pkg.variant}${PACKAGE_SUFFIX}`)
  execFileSync('python3', [BUILDER, '--variant', pkg.variant, '--out', out], { stdio: 'pipe' })
  return fs.readFileSync(out)
}

/** Uploads the package (201, or 200 when a parallel project already did) and opts in. */
async function install(request: APIRequestContext, pkg: StoragePackage): Promise<Installed> {
  const upload = await request.post(`${BACKEND_URL}/api/themes/import`, {
    multipart: { file: { name: `${pkg.variant}${PACKAGE_SUFFIX}`, mimeType: PACKAGE_TYPE, buffer: buildPackage(pkg) } },
  })
  expect(upload.ok(), `POST /api/themes/import answered ${upload.status()}: ${await upload.text()}`).toBe(true)
  const answer = (await upload.json()) as UploadAnswer
  expect(answer.install, 'the upload installs the theme in the workspace').not.toBeNull()
  const installId = answer.install?.id ?? ''
  const optIn = await request.post(`${BACKEND_URL}/api/themes/installs/${installId}/opt-in`, {
    data: { packageId: answer.package.id },
  })
  expect(optIn.ok(), `POST …/opt-in answered ${optIn.status()}: ${await optIn.text()}`).toBe(true)
  return { pkg, installId, packageId: answer.package.id, digest: answer.package.digest, version: answer.package.version }
}

/**
 * The active-theme answer naming this install, with the manifest in the shape the client loader
 * reads (lib/theme-host/manifest.ts parseThemeManifest), pointing at the package's own files.
 */
function activeAnswer(theme: Installed): Record<string, unknown> {
  const manifest = {
    schema: 'automatos-theme-spike/v1',
    id: theme.pkg.themeId,
    name: theme.pkg.name,
    version: theme.version,
    engines: { sdk: SDK_RANGE },
    base: 'studio',
    htmlClass: 'hello',
    entry: ENTRY_FILE,
    styleSheet: 'dist/theme.css',
    preset: 'looks/presets/default.look.json',
    slots: ['frame@1', 'lookEditor@1', 'styleSheet@1'],
  }
  const active = {
    installId: theme.installId,
    themeId: theme.pkg.themeId,
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

/** Answers every active-theme read with the install the tab is on now. */
async function answerActive(page: Page, current: () => Installed): Promise<void> {
  await page.route(ACTIVE_ROUTE, (route) => fulfillBackendJson(route, 200, activeAnswer(current())))
}

/** Puts this browser on the install's style for the next navigation. */
async function chooseInstall(page: Page, theme: Installed): Promise<void> {
  await page.context().addCookies([{ name: STYLE_COOKIE, value: `${INSTALLED_STYLE_PREFIX}${theme.installId}`, url: APP_URL }])
}

/** The key the host keeps a theme's browser-lifetime, account-wide state under, in the local edition. */
function blobKey(pkg: StoragePackage): string {
  return `${STATE_KEY_PREFIX}${BROWSER_LIFETIME}:${pkg.themeId}:u:${LOCAL_USER}`
}

async function storedBlob(page: Page, pkg: StoragePackage): Promise<unknown> {
  const raw = await page.evaluate((key) => window.localStorage.getItem(key), blobKey(pkg))
  return raw === null ? null : (JSON.parse(raw) as unknown)
}

/** The host runs the theme and the theme's state is ready: the scope is published. */
async function expectReady(page: Page): Promise<void> {
  await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, RUNNING)
  await expect(page.getByTestId(STATE_TEST_ID)).toHaveAttribute('data-ready', 'true')
}

test.describe('PRD-261 E3c — a theme keeps its state through the SDK', () => {
  test('hello-e3 keeps a value across a reload; another theme id reads none of it', { tag: '@host' }, async ({ page, request }) => {
    const watch = watchPage(page)
    const first = await install(request, HELLO_E3)
    const other = await install(request, HELLO_E3_OTHER)
    let current = first
    await page.addInitScript(({ key, value }) => window.localStorage.setItem(key, value), APPEARANCE_DEFAULTS)
    await answerActive(page, () => current)
    const state = page.getByTestId(STATE_TEST_ID)

    await chooseInstall(page, first)
    await page.goto(THEME_E2E_PATH)
    await expectReady(page)
    await expect(state).toHaveAttribute('data-value', NOTHING_KEPT)
    await page.getByTestId(SAVE_TEST_ID).click()
    await expect(state).toHaveAttribute('data-last-write', 'ok')
    await expect(state).toHaveAttribute('data-value', JSON.stringify(GREETING))
    await expect(state).toHaveAttribute('data-persisted', 'true')
    expect(await storedBlob(page, HELLO_E3)).toMatchObject({ v: 1, data: { [GREETING_KEY]: GREETING } })

    await page.reload()
    await expectReady(page)
    await expect(state).toHaveAttribute('data-value', JSON.stringify(GREETING))

    current = other
    await chooseInstall(page, other)
    await page.goto(THEME_E2E_PATH)
    await expectReady(page)
    await expect(state).toHaveAttribute('data-value', NOTHING_KEPT)
    expect(await storedBlob(page, HELLO_E3_OTHER)).toBeNull()
    expect(await storedBlob(page, HELLO_E3)).toMatchObject({ data: { [GREETING_KEY]: GREETING } })

    expect(watch.pageErrors).toEqual([])
  })
})
