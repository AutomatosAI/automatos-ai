import { expect, test, type APIRequestContext, type Page } from '@playwright/test'
import { chooseKitPackage, installKitPackage, PLAY_THEME_ID, resetKitPackage } from './kit-package'

// PRD-261 A-17b (@parity), A-19f4: the Play specs, run against Play as the Design Kit packs it,
// installed in the local edition (support/kit-package.ts): the kit's .autotheme uploaded, opted in
// to and chosen, then the install's style cookie. Until W-A8 each spec also ran under builtin Play,
// compiled into core, and held the two to the same assertions; W-A8 deleted builtin Play, so
// PLAY_VARIANTS is the package alone, each spec runs once, and these specs are core's proof that
// Play still runs. describeParity and the variant shape stay, so each spec reads as it did.
//
// The specs store the member's choice and Play's looks on the lane's one backend, so the workflow
// runs @parity with @package, in their own step, one test at a time. Every test starts and ends on
// Studio, with no install of Play's package and no looks kept for Play on the server. The server
// holds a member's looks for Play only while its package is installed (404 otherwise) and keeps them
// across an uninstall, so they are emptied right after each install and again before the reset
// removes the package.

export const PARITY_TAG = '@parity'
export const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const BACKEND_URL = process.env.E2E_BACKEND_URL ?? 'http://localhost:8000'

/** The package's mascot files, under its files folder (the kit's themes/play/src/lib/mascot/media.ts MASCOT_DIR). */
const PACKAGE_MASCOT_DIR = 'assets/mascot/'
const THEME_FILES_PATH = '/api/themes/files/'
/** Play's looks on the server, kept under its theme id. */
const PLAY_LOOKS_PATH = `/api/themes/looks/${PLAY_THEME_ID}`
const NO_LOOKS = { looks: [], active: null }
const STATUS_ATTR = 'data-theme-host'
const PLAY_CLASS = /\bplay\b/
const OK = 200
/** Play's looks when no install of its package is in the workspace: the server holds none it would serve. */
const NOT_INSTALLED = 404
/** The package's entry, stylesheet and fonts come through the files proxy on a cold load. */
export const THEME_LOAD_TIMEOUT_MS = 30_000

export type PlayVariantId = 'package'

/** Play in the lane's workspace, ready to be chosen. */
export interface AvailablePlay {
  readonly variant: PlayVariantId
  /** The style cookie Play runs under. */
  readonly style: string
  /** The path a mascot file ('idle.png') is served at under this Play. */
  mascotPath(file: string): string
  /** Chooses Play as the member's theme on the server and puts this browser on its style, before the first navigation. */
  choose(page: Page): Promise<void>
  /** Waits until the page runs Play: the theme host has loaded the package. */
  ready(page: Page): Promise<void>
}

export interface PlayVariant {
  readonly id: PlayVariantId
  readonly title: string
  /** Makes Play available without choosing it: the package uploaded and opted in to, with no looks kept. */
  makeAvailable(request: APIRequestContext): Promise<AvailablePlay>
}

/** Empties the member's looks for Play. The package must be installed: the server refuses it otherwise. */
async function emptyPlayLooks(request: APIRequestContext): Promise<void> {
  const looks = await request.put(`${BACKEND_URL}${PLAY_LOOKS_PATH}`, { data: NO_LOOKS })
  expect(looks.status(), `PUT ${PLAY_LOOKS_PATH} answered ${looks.status()}: ${await looks.text()}`).toBe(OK)
}

/** Empties the member's looks for Play when its package is installed; with none, there are none to serve. */
async function emptyInstalledPlayLooks(request: APIRequestContext): Promise<void> {
  const held = await request.get(`${BACKEND_URL}${PLAY_LOOKS_PATH}`)
  if (held.status() === NOT_INSTALLED) return
  expect(held.status(), `GET ${PLAY_LOOKS_PATH} answered ${held.status()}: ${await held.text()}`).toBe(OK)
  await emptyPlayLooks(request)
}

const PACKAGE_PLAY: PlayVariant = {
  id: 'package',
  title: "the Design Kit's Play package",
  async makeAvailable(request) {
    const install = await installKitPackage(request)
    // An install brings back the looks kept under Play's theme id before it: start with none.
    await emptyPlayLooks(request)
    return {
      variant: 'package',
      style: install.style,
      mascotPath: (file) => `${THEME_FILES_PATH}${install.digest}/${PACKAGE_MASCOT_DIR}${file}`,
      choose: (page) => chooseKitPackage(page, install, APP_URL),
      ready: async (page) => {
        const html = page.locator('html')
        await expect(html).toHaveAttribute(STATUS_ATTR, 'active', { timeout: THEME_LOAD_TIMEOUT_MS })
        await expect(html).toHaveClass(PLAY_CLASS)
      },
    }
  },
}

/** The Plays each spec runs under: since W-A8, the installed package only. */
export const PLAY_VARIANTS: readonly PlayVariant[] = [PACKAGE_PLAY]

/** Studio on the server, no looks kept for Play, and no install of Play's package. */
export async function resetPlayParity(request: APIRequestContext): Promise<void> {
  await emptyInstalledPlayLooks(request)
  await resetKitPackage(request)
}

/** Makes Play available, chooses it, opens `path` and waits until Play runs there. */
export async function openPlay(page: Page, request: APIRequestContext, variant: PlayVariant, path: string): Promise<AvailablePlay> {
  const play = await variant.makeAvailable(request)
  await play.choose(page)
  await page.goto(path)
  await play.ready(page)
  return play
}

/**
 * The tests of one spec, written once and run under each Play (a describe per variant), each test
 * starting and ending on a clean slate (resetPlayParity).
 */
export function describeParity(title: string, body: (variant: PlayVariant) => void): void {
  for (const variant of PLAY_VARIANTS) {
    test.describe(`${title} (${variant.title})`, () => {
      test.beforeEach(async ({ request }) => {
        await resetPlayParity(request)
      })
      test.afterEach(async ({ request }) => {
        await resetPlayParity(request)
      })
      body(variant)
    })
  }
}
