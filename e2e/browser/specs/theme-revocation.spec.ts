import { execFileSync } from 'child_process'
import fs from 'fs'
import os from 'os'
import path from 'path'
import { expect, test, type APIRequestContext, type Page } from '@playwright/test'
import { chooseStyle, THEME_E2E_PATH } from '../support/hello-theme'

// PRD-261 C-05l: a revoked theme falls back on the next load, and lifting it brings it back. End to
// end on the local edition, against the real backend, the real files route and the real admin
// routes (no route interception), in Chromium and WebKit:
//
//   upload fixtures/hello-c05's package privately → opt in → choose it → the hello frame draws →
//   a digest `disable` through POST /api/admin/themes/revocations → reload: Studio draws, the page
//   asks for nothing under /api/themes/files/<digest>/, and the host's notice says the theme was
//   withdrawn, with the reason → lift it (POST …/revocations/{id}/lift) → reload: the frame is back.
//   A digest `warn` keeps the theme running, with the host's warning and its reason.
//
// The feed's rows reaching theme_revocations is C-05h's pytest; the rows reaching the browser is this
// spec; together they are the PRD's C-05 proof.
//
// It stores the member's theme choice on the lane's one backend, which every other spec's tab reads,
// so it carries `@package` and runs in the workflow's step that runs such specs one test at a time,
// after the others (the README's "Play from the Design Kit"), with no workflow change. Chromium and
// WebKit run it in turn there: the upload of the same bytes answers 200 the second time, an opt-in
// already made is accepted again, and each test lifts its own revocations (their reasons start with
// REASON_PREFIX), removes the install and puts the choice back on Studio, before and after.

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const BACKEND_URL = process.env.E2E_BACKEND_URL ?? 'http://localhost:8000'
const BUILDER = path.resolve(__dirname, '..', 'fixtures', 'hello-c05', 'build.py')
const PACKAGE_NAME = 'hello-c05.autotheme'
const PACKAGE_TYPE = 'application/zip'
const THEME_ID = 'automatos-e2e/hello-c05'
const INSTALLED_STYLE_PREFIX = 'theme:install:'
const STUDIO = 'studio'
const STATUS_ATTR = 'data-theme-host'
const STUDIO_SHELL = '.sh-shell'
const HELLO_FRAME = 'hello-theme-frame'
const FILES_PATH = '/api/themes/files/'
// lib/theme-host/look.ts: the <link> the host adds for a running theme's style sheet.
const THEME_STYLE_LINK = 'link[data-theme-host-style]'
// lib/theme-host/choice-sync.ts CHOICE_CHECK_KEY_PREFIX: the tab's record of its one server check.
const CHOICE_CHECK_PREFIX = 'automatos-theme-choice:'
// components/theme-host/theme-host-provider.tsx THEME_WITHDRAWN_NOTICE and THEME_REVOKED_WARNING_NOTICE.
const WITHDRAWN_NOTICE = 'This theme was withdrawn, so it is turned off for now.'
const WARNING_NOTICE = 'This theme is still running, but its maintainer withdrew this version.'
const REASON_PREFIX = 'C-05l browser proof:'
const DISABLE_REASON = `${REASON_PREFIX} taken down while it is checked.`
const WARN_REASON = `${REASON_PREFIX} still runs, update soon.`
const THEME_LOAD_TIMEOUT_MS = 30_000
const SPEC_TIMEOUT_MS = 120_000
const OK = 200
const CREATED = 201

interface UploadAnswer {
  readonly package: { readonly id: string; readonly themeId: string; readonly digest: string }
  readonly install: { readonly id: string } | null
}

interface Installed {
  readonly installId: string
  readonly digest: string
}

interface RevocationRow {
  readonly id: string
  readonly reason: string
  readonly origin: string
  readonly liftedAt: string | null
}

type Level = 'warn' | 'disable'

function buildPackage(): Buffer {
  const folder = fs.mkdtempSync(path.join(os.tmpdir(), 'hello-c05-'))
  const out = path.join(folder, PACKAGE_NAME)
  execFileSync('python3', [BUILDER, '--out', out], { stdio: 'pipe' })
  return fs.readFileSync(out)
}

/** Uploads the package (201, or 200 for bytes the other browser stored) and opts in, as Install theme does. */
async function installHello(request: APIRequestContext): Promise<Installed> {
  const upload = await request.post(`${BACKEND_URL}/api/themes/import`, {
    multipart: { file: { name: PACKAGE_NAME, mimeType: PACKAGE_TYPE, buffer: buildPackage() } },
  })
  expect(upload.ok(), `POST /api/themes/import answered ${upload.status()}: ${await upload.text()}`).toBe(true)
  const answer = (await upload.json()) as UploadAnswer
  expect(answer.package.themeId).toBe(THEME_ID)
  expect(answer.install, 'the upload installs the theme in the workspace').not.toBeNull()
  const installId = answer.install?.id ?? ''
  const optIn = await request.post(`${BACKEND_URL}/api/themes/installs/${installId}/opt-in`, {
    data: { packageId: answer.package.id },
  })
  expect(optIn.ok(), `POST …/opt-in answered ${optIn.status()}: ${await optIn.text()}`).toBe(true)
  return { installId, digest: answer.package.digest }
}

/** Stores the install as the member's choice, and puts this browser on its style before the first navigation. */
async function chooseHello(page: Page, theme: Installed): Promise<void> {
  const choice = await page.request.put(`${BACKEND_URL}/api/themes/choice`, { data: { choice: theme.installId } })
  expect(choice.status(), `PUT /api/themes/choice answered ${choice.status()}: ${await choice.text()}`).toBe(OK)
  await chooseStyle(page, `${INSTALLED_STYLE_PREFIX}${theme.installId}`, APP_URL)
}

/** Revokes the package's digest at `level` through the admin route; the new row's id. */
async function revoke(request: APIRequestContext, digest: string, level: Level, reason: string): Promise<string> {
  const response = await request.post(`${BACKEND_URL}/api/admin/themes/revocations`, { data: { digest, level, reason } })
  const answered = `POST /api/admin/themes/revocations answered ${response.status()}: ${await response.text()}`
  expect(response.status(), answered).toBe(CREATED)
  return ((await response.json()) as { revocation: RevocationRow }).revocation.id
}

async function lift(request: APIRequestContext, revocationId: string): Promise<void> {
  const response = await request.post(`${BACKEND_URL}/api/admin/themes/revocations/${revocationId}/lift`)
  expect(response.status(), `POST …/lift answered ${response.status()}: ${await response.text()}`).toBe(OK)
}

/** Lifts every live revocation this spec wrote, removes every install of the package and chooses Studio. */
async function reset(request: APIRequestContext): Promise<void> {
  const listed = await request.get(`${BACKEND_URL}/api/admin/themes/revocations`)
  expect(listed.status()).toBe(OK)
  const rows = ((await listed.json()) as { revocations: RevocationRow[] }).revocations
  for (const row of rows.filter((item) => item.liftedAt === null && item.reason.startsWith(REASON_PREFIX))) {
    await lift(request, row.id)
  }
  const installed = await request.get(`${BACKEND_URL}/api/themes/installed`)
  expect(installed.status()).toBe(OK)
  for (const row of ((await installed.json()) as { installs: { installId: string; themeId: string }[] }).installs) {
    if (row.themeId === THEME_ID) {
      expect((await request.delete(`${BACKEND_URL}/api/themes/installs/${row.installId}`)).status()).toBe(OK)
    }
  }
  expect((await request.put(`${BACKEND_URL}/api/themes/choice`, { data: { choice: STUDIO } })).status()).toBe(OK)
}

/** Every request the page makes for one of the package's files, filled in as they arrive. */
function watchFiles(page: Page, digest: string): string[] {
  const asked: string[] = []
  page.on('request', (request) => {
    const url = new URL(request.url())
    if (url.pathname.startsWith(`${FILES_PATH}${digest}/`)) asked.push(url.pathname)
  })
  return asked
}

async function expectHelloActive(page: Page): Promise<void> {
  await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, 'active', { timeout: THEME_LOAD_TIMEOUT_MS })
  await expect(page.getByTestId(HELLO_FRAME)).toBeVisible()
}

/**
 * Waits for the tab's one check of the server's choice (ThemeChoiceSync): it agreed with the tab, so
 * no later load of this tab moves the style; the reloads below follow only what the server answers.
 */
async function waitForChoiceCheck(page: Page): Promise<void> {
  const checked = (prefix: string): boolean => Object.keys(window.sessionStorage).some((key) => key.startsWith(prefix))
  await expect.poll(() => page.evaluate(checked, CHOICE_CHECK_PREFIX)).toBe(true)
}

/** Installs, chooses and opens the theme: the hello frame draws. */
async function openHello(page: Page): Promise<Installed> {
  const theme = await installHello(page.request)
  await chooseHello(page, theme)
  await page.goto(THEME_E2E_PATH)
  await expectHelloActive(page)
  await waitForChoiceCheck(page)
  return theme
}

test.describe('a revoked theme on the next load (C-05l)', () => {
  test.beforeEach(async ({ page }) => {
    await reset(page.request)
  })

  test.afterEach(async ({ page }) => {
    await reset(page.request)
  })

  const DISABLE_TITLE = 'a digest disable shows Studio with the notice and asks for none of its files; lifted, the theme returns'
  test(DISABLE_TITLE, { tag: '@package' }, async ({ page }) => {
    test.setTimeout(SPEC_TIMEOUT_MS)
    const theme = await openHello(page)
    const revocationId = await revoke(page.request, theme.digest, 'disable', DISABLE_REASON)

    const asked = watchFiles(page, theme.digest)
    await page.reload()
    const notice = page.locator('[data-sonner-toast]').filter({ hasText: WITHDRAWN_NOTICE })
    await expect(notice).toBeVisible({ timeout: THEME_LOAD_TIMEOUT_MS })
    await expect(notice).toContainText(DISABLE_REASON)
    await expect(page.locator(STUDIO_SHELL)).toBeVisible()
    await expect(page.getByTestId(HELLO_FRAME)).toHaveCount(0)
    await expect(page.locator('html')).not.toHaveAttribute(STATUS_ATTR)
    // A file the browser had cached would make no request: nothing of the theme's is on the page either.
    await expect(page.locator(THEME_STYLE_LINK)).toHaveCount(0)
    expect(asked, `no request under ${FILES_PATH}${theme.digest}/`).toEqual([])

    await lift(page.request, revocationId)
    await page.reload()
    await expectHelloActive(page)
    await expect(page.locator('[data-sonner-toast]').filter({ hasText: WITHDRAWN_NOTICE })).toHaveCount(0)
  })

  test('a digest warn keeps the theme running, with the warning and its reason', { tag: '@package' }, async ({ page }) => {
    test.setTimeout(SPEC_TIMEOUT_MS)
    const theme = await openHello(page)
    const revocationId = await revoke(page.request, theme.digest, 'warn', WARN_REASON)

    await page.reload()
    await expectHelloActive(page)
    const warning = page.locator('[data-sonner-toast]').filter({ hasText: WARNING_NOTICE })
    await expect(warning).toBeVisible()
    await expect(warning).toContainText(WARN_REASON)

    await lift(page.request, revocationId)
    await page.reload()
    await expectHelloActive(page)
    await expect(page.locator('[data-sonner-toast]').filter({ hasText: WARNING_NOTICE })).toHaveCount(0)
  })
})

// The fixture builds: a builder that fails is a lane failure, not a skip.
test('the lane builds hello-c05.autotheme', { tag: '@package' }, async ({ browserName }) => {
  test.skip(browserName !== 'chromium', 'checked once')
  expect(buildPackage().length).toBeGreaterThan(0)
})
