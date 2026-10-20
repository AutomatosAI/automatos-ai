import { expect, test, type APIRequestContext, type Page } from '@playwright/test'
import { chooseStyle } from './hello-theme'
import { chooseKitPackage, installKitPackage, PLAY_THEME_ID, resetKitPackage } from './kit-package'

// PRD-261 A-17b (@parity): the same Play specs, run once against builtin Play (compiled into core,
// the style `theme:builtin:play`) and once against Play as the Design Kit packs it, installed in the
// local edition (support/kit-package.ts). describeParity writes each spec's tests once and runs them
// under both; only the setup differs:
//
//   builtin  the member's choice `builtin:play` on the server, then the style cookie;
//   package  the kit's .autotheme uploaded, opted in to and chosen, then the install's style cookie.
//
// Both runs store the member's choice (and builtin Play's looks) on the lane's one backend, so the
// workflow runs @parity with @package, in their own step, one test at a time. Every test starts and
// ends on Studio, with no install of Play's package and no looks kept for Play on the server.
//
// Where the package cannot pass an assertion builtin Play passes because core treats an installed
// theme differently, the test keeps its assertions and is marked test.fixme for the package only,
// naming the core gap (knownPackageGap, the GAP_* texts below; e2e/browser/parity-checklist.md lists
// them for the owner).

export const PARITY_TAG = '@parity'
export const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const BACKEND_URL = process.env.E2E_BACKEND_URL ?? 'http://localhost:8000'

const BUILTIN_PLAY_STYLE = 'theme:builtin:play'
const BUILTIN_PLAY_CHOICE = 'builtin:play'
/** Builtin Play's mascot files: frontend/public/auto/ (lib/mascot/media.ts MASCOT_DIR). */
const BUILTIN_MASCOT_ROOT = '/auto/'
/** The package's mascot files, under its files folder (the kit's lib/mascot/media.ts MASCOT_DIR). */
const PACKAGE_MASCOT_DIR = 'assets/mascot/'
const THEME_FILES_PATH = '/api/themes/files/'
/** Play's looks on the server; builtin Play and the package keep them under one theme id. */
const PLAY_LOOKS_PATH = `/api/themes/looks/${PLAY_THEME_ID}`
const NO_LOOKS = { looks: [], active: null }
const CHOICE_PATH = '/api/themes/choice'
const STATUS_ATTR = 'data-theme-host'
const PLAY_CLASS = /\bplay\b/
const OK = 200
/** The package's entry, stylesheet and fonts come through the files proxy on a cold load. */
export const THEME_LOAD_TIMEOUT_MS = 30_000

export type PlayVariantId = 'builtin' | 'package'

/** Play in the lane's workspace, ready to be chosen. */
export interface AvailablePlay {
  readonly variant: PlayVariantId
  /** The style cookie Play runs under. */
  readonly style: string
  /** The path a mascot file ('idle.png') is served at under this Play. */
  mascotPath(file: string): string
  /** Chooses Play as the member's theme on the server and puts this browser on its style, before the first navigation. */
  choose(page: Page): Promise<void>
  /** Waits until the page runs Play (the package: the theme host has loaded it). */
  ready(page: Page): Promise<void>
}

export interface PlayVariant {
  readonly id: PlayVariantId
  readonly title: string
  /** Makes Play available without choosing it: nothing for builtin Play; the package uploaded and opted in to. */
  makeAvailable(request: APIRequestContext): Promise<AvailablePlay>
}

async function chooseBuiltinPlay(page: Page): Promise<void> {
  const answer = await page.request.put(`${BACKEND_URL}${CHOICE_PATH}`, { data: { choice: BUILTIN_PLAY_CHOICE } })
  expect(answer.status(), `PUT ${CHOICE_PATH} answered ${answer.status()}: ${await answer.text()}`).toBe(OK)
  expect(((await answer.json()) as { choice?: unknown }).choice).toBe(BUILTIN_PLAY_CHOICE)
  await chooseStyle(page, BUILTIN_PLAY_STYLE, APP_URL)
}

const BUILTIN_PLAY: PlayVariant = {
  id: 'builtin',
  title: 'builtin Play',
  async makeAvailable() {
    return {
      variant: 'builtin',
      style: BUILTIN_PLAY_STYLE,
      mascotPath: (file) => `${BUILTIN_MASCOT_ROOT}${file}`,
      choose: chooseBuiltinPlay,
      ready: async (page) => {
        await expect(page.locator('html')).toHaveClass(PLAY_CLASS)
      },
    }
  },
}

const PACKAGE_PLAY: PlayVariant = {
  id: 'package',
  title: "the Design Kit's Play package",
  async makeAvailable(request) {
    const install = await installKitPackage(request)
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

export const PLAY_VARIANTS: readonly PlayVariant[] = [BUILTIN_PLAY, PACKAGE_PLAY]

/** Studio on the server, no install of Play's package, and no looks kept for Play. */
export async function resetPlayParity(request: APIRequestContext): Promise<void> {
  await resetKitPackage(request)
  const looks = await request.put(`${BACKEND_URL}${PLAY_LOOKS_PATH}`, { data: NO_LOOKS })
  expect(looks.status(), `PUT ${PLAY_LOOKS_PATH} answered ${looks.status()}: ${await looks.text()}`).toBe(OK)
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
 * The tests of one spec, written once and run under each Play: a describe per variant, each test
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

/** Marks the running test fixme for the package only, naming the core gap; builtin Play runs it. */
export function knownPackageGap(variant: PlayVariant, gap: string): void {
  test.fixme(variant.id === 'package', gap)
}
