import fs from 'fs'
import path from 'path'
import { expect, type APIRequestContext, type Page } from '@playwright/test'
import { chooseStyle } from './hello-theme'

// PRD-261 A-17a: Play as the Design Kit packs it (AutomatosAI/automatos-play at the commit
// .github/workflows/browser-e2e.yml pins in KIT_REF), installed in the local edition through the
// routes and bodies the Install theme UI sends (A-07d, components/settings/themes and
// lib/api-client.ts):
//
//   POST /api/themes/import                    multipart, the file under `file`   (importTheme)
//   POST /api/themes/installs/{id}/opt-in      { packageId }                       (optInTheme)
//   PUT  /api/themes/choice                    { choice: <installId> }             (putThemeChoice)
//
// then the style cookie the choice writes (`theme:install:<installId>`, useChooseTheme's setStyle).
// Nothing is intercepted. The lane's `package` job builds the file, and the browser job downloads
// it and names it in E2E_KIT_PACKAGE.

const BACKEND_URL = process.env.E2E_BACKEND_URL ?? 'http://localhost:8000'
/** The downloaded .autotheme; empty when the lane did not name one. */
export const KIT_PACKAGE_FILE = process.env.E2E_KIT_PACKAGE ?? ''
/** Play's theme id in the package's theme.json (A-20: its htmlClass `play` is any theme's, kept for no id). */
export const PLAY_THEME_ID = 'automatos/play'
const PACKAGE_TYPE = 'application/zip'
const INSTALLED_STYLE_PREFIX = 'theme:install:'
const STUDIO = 'studio'
const OK = 200

interface UploadAnswer {
  readonly package: { readonly id: string; readonly themeId: string; readonly digest: string; readonly version: string }
  readonly install: { readonly id: string } | null
}

/** A row of GET /api/themes/installed. */
export interface InstalledRow {
  readonly installId: string
  readonly themeId: string
}

interface ChoiceAnswer {
  readonly choice?: unknown
  readonly theme?: { readonly themeId?: unknown } | null
}

/** The kit's Play package as installed and opted in to in the lane's workspace. */
export interface KitInstall {
  readonly installId: string
  readonly packageId: string
  readonly digest: string
  readonly version: string
  /** `theme:install:<installId>` */
  readonly style: string
}

/** The package's bytes; a missing file is a lane failure, never a skip. */
export function readKitPackage(): Buffer {
  expect(KIT_PACKAGE_FILE, 'E2E_KIT_PACKAGE is set by the lane step that runs the @package specs').not.toBe('')
  expect(fs.existsSync(KIT_PACKAGE_FILE), `the package job's ${KIT_PACKAGE_FILE} was downloaded`).toBe(true)
  return fs.readFileSync(KIT_PACKAGE_FILE)
}

/** Uploads the package (201, or 200 for bytes already here, installed again if they were removed) and opts in. */
export async function installKitPackage(request: APIRequestContext): Promise<KitInstall> {
  const upload = await request.post(`${BACKEND_URL}/api/themes/import`, {
    multipart: { file: { name: path.basename(KIT_PACKAGE_FILE), mimeType: PACKAGE_TYPE, buffer: readKitPackage() } },
  })
  expect(upload.ok(), `POST /api/themes/import answered ${upload.status()}: ${await upload.text()}`).toBe(true)
  const answer = (await upload.json()) as UploadAnswer
  expect(answer.package.themeId, "the package is the kit's Play").toBe(PLAY_THEME_ID)
  expect(answer.install, 'the upload installs the theme in the workspace').not.toBeNull()
  const installId = answer.install?.id ?? ''
  const optIn = await request.post(`${BACKEND_URL}/api/themes/installs/${installId}/opt-in`, {
    data: { packageId: answer.package.id },
  })
  expect(optIn.ok(), `POST …/opt-in answered ${optIn.status()}: ${await optIn.text()}`).toBe(true)
  return {
    installId,
    packageId: answer.package.id,
    digest: answer.package.digest,
    version: answer.package.version,
    style: `${INSTALLED_STYLE_PREFIX}${installId}`,
  }
}

/**
 * Chooses the install as the member's theme (the server answers the active theme it gives: Play)
 * and puts this browser on its style before the first navigation, as the choice does.
 */
export async function chooseKitPackage(page: Page, install: KitInstall, appUrl: string): Promise<void> {
  const choice = await page.request.put(`${BACKEND_URL}/api/themes/choice`, { data: { choice: install.installId } })
  expect(choice.status(), `PUT /api/themes/choice answered ${choice.status()}: ${await choice.text()}`).toBe(OK)
  const active = (await choice.json()) as ChoiceAnswer
  expect(active.choice).toBe(install.installId)
  expect(active.theme?.themeId).toBe(PLAY_THEME_ID)
  await chooseStyle(page, install.style, appUrl)
}

/** The workspace's theme installs. */
export async function installedRows(request: APIRequestContext): Promise<InstalledRow[]> {
  const response = await request.get(`${BACKEND_URL}/api/themes/installed`)
  expect(response.status()).toBe(OK)
  return ((await response.json()) as { installs: InstalledRow[] }).installs
}

/** Puts the member's choice back on Studio and removes every install `remove` picks, so no other run starts on one. */
export async function resetInstalls(request: APIRequestContext, remove: (row: InstalledRow) => boolean): Promise<void> {
  expect((await request.put(`${BACKEND_URL}/api/themes/choice`, { data: { choice: STUDIO } })).status()).toBe(OK)
  for (const row of await installedRows(request)) {
    if (remove(row)) {
      expect((await request.delete(`${BACKEND_URL}/api/themes/installs/${row.installId}`)).status()).toBe(OK)
    }
  }
}

/** Puts the member's choice back on Studio and removes every install of Play's package, so no other run starts on it. */
export async function resetKitPackage(request: APIRequestContext): Promise<void> {
  await resetInstalls(request, (row) => row.themeId === PLAY_THEME_ID)
}
