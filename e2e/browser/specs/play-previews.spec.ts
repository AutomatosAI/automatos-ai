import fs from 'fs'
import path from 'path'
import { expect, test, type APIRequestContext, type Locator, type Page, type PageScreenshotOptions, type Response } from '@playwright/test'
import { fulfillBackendJson } from '../support/backend-route'
import { noStoredConversations } from '../support/chat-stream'
import { chooseKitPackage, installKitPackage, PLAY_THEME_ID } from '../support/kit-package'
import { messageBox } from '../support/play-chat'
import { resetPlayParity, THEME_LOAD_TIMEOUT_MS } from '../support/play-parity'
import { watchPage } from '../support/watch'

// PRD-261 C-06q1 (@package; PB-24, PB-26): Play 1.1.0's listing previews, shot in the lane. Play is
// the Design Kit's package, uploaded, opted in to and chosen as installed-play.spec.ts does
// (support/kit-package.ts). For each preset the page's boot carries, picked on Make it yours as a
// member picks it, the spec saves the frame on /chat as <preset>.png, and under the default preset
// /deliverables and /looks as <preset>-deliverables.png and <preset>-looks.png, in
// e2e/browser/previews/. The workflow uploads that folder as `play-previews`; the owner picks from it
// and the kit commits the picks (C-06q2). Nothing here is committed or changes Play's bytes.
//
// Each shot is 1280x800 at device scale 1, one image pixel per CSS pixel (scale: 'css'), and is held
// to what core's package reader and listing take: a PNG of at most 512 KB (package.py
// IMAGE_MAX_BYTES), a name that fits the manifest's previews pattern (manifest.py PREVIEW_PATTERN)
// and belongs to its preset by the listing's longest-match rule (listings.py _owner_preset).
//
// Deterministic: reduced motion (Auto is his still, never a clip frame), his still loaded and him at
// rest (the first open's hello is over and the message box Play focuses is left), the fonts loaded,
// the look's variables on <html>, animations off and two shots in a row equal. The conversations list and the shelf answer with none, so only
// the lane's own member and workspace show. The lane has no LLM key, so the host's local-edition key
// nudge is dismissed before the first load, as a member dismisses it, and every chat shot is held to
// showing none.
//
// Chromium only. The specs store the member's choice and Play's looks on the lane's one backend, so
// they run in the @package step, one test at a time; each starts and ends on Studio, with no install
// of Play's package and no looks kept for Play (resetPlayParity).

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const ONLY_BROWSER = 'chromium'
const PREVIEW_DIR = path.resolve(__dirname, '..', 'previews')
const VIEWPORT = { width: 1280, height: 800 }
const DEVICE_SCALE = 1
/** Play 1.1.0's presets (the kit's themes/play/src/lib/looks/presets.ts); the boot must carry exactly these. */
const PRESET_IDS = ['ember', 'obsidian', 'aurora', 'workshop', 'daylight'] as const
const DEFAULT_PRESET = 'ember'

type ScreenId = 'chat' | 'deliverables' | 'looks'
/** The screen a bare <preset>.png shows; every other screen is named <preset>-<screen>.png. */
const FRAME_SCREEN: ScreenId = 'chat'
/** The default preset's shots, in the order they are taken; every other preset has the frame only. */
const DEFAULT_PRESET_SCREENS: readonly ScreenId[] = ['looks', 'chat', 'deliverables']

// What core's reader and listing take (orchestrator/services/themes/).
const PACKAGE_PREVIEW_DIR = 'assets/previews/'
const PNG_SUFFIX = '.png'
/** listings.py SCREEN_SEPARATOR. */
const SCREEN_SEPARATOR = '-'
/** manifest.py PREVIEW_PATTERN (ASSET_NAME), held with fullmatch. */
const PREVIEW_PATTERN = /^assets\/previews\/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.png$/
/** package.py IMAGE_MAX_BYTES: 512 KB. */
const PREVIEW_MAX_BYTES = 512 * 1024
const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
/** The IHDR chunk's width and height, right after the signature, the chunk length and its type. */
const PNG_WIDTH_AT = 16
const PNG_HEIGHT_AT = 20

const SHOT_OPTIONS: PageScreenshotOptions = { type: 'png', scale: 'css', animations: 'disabled', caret: 'hide', fullPage: false }
const STABLE_SHOT_GAP_MS = 500
const STABLE_SHOT_TRIES = 10
/** Install, Make it yours and each screen's cold load, plus the hello one-shot (5 s) on the first /chat. */
const SHOT_TEST_TIMEOUT_MS = 180_000

/** The JSON element THEME_BOOT_SCRIPT reads (lib/theme-host/boot.ts THEME_BOOT_ELEMENT_ID). */
const BOOT_ELEMENT = '#automatos-theme-boot'
/** lib/theme-host/boot-sanitize.ts CSS_VAR_NAME_PATTERN: the only names the boot and the runtime paint. */
const CSS_VAR_NAME = /^--[a-z0-9-]{1,64}$/
const STATUS_ATTR = 'data-theme-host'
const PLAY_CLASS = /\bplay\b/
/** Play's frame: its sidebar (the kit's themes/play/src/components/play/shell/play-sidebar.tsx). */
const PLAY_FRAME = 'aside[aria-label="Sidebar"]'
/** The sidebar's connection pill once GET /health answered (the kit's shell/connection-pill.tsx). */
const CONNECTED = /^Connected\b/
/** Auto's stage (the kit's components/mascot/auto-mascot.tsx): his still or his clip, and his ring. */
const AUTO_STAGE = '[data-testid="auto-stage"]'
const FONTS_LOADED = 'loaded'

// Make it yours (the kit's components/looks/): the Looks row's cards and the editor.
const LOOKS_ROW = 'Looks'
const EDITOR = 'Edit the look'
const PRESSED_ATTR = 'aria-pressed'
const LOOKS_API_PATH = `/api/themes/looks/${PLAY_THEME_ID}`
/** Room for the store's pause before it sends (LOOK_SEND_DELAY_MS) and the round trip. */
const SAVE_TIMEOUT_MS = 20_000
// The empty chat (the kit's chat/first-open.tsx) and Auto at rest (lib/mascot/media.ts CAPTIONS.idle).
const HERO = 'Say hello'
const AT_REST = 'Ready when you are'
/** Auto while the message box has focus (lib/mascot/media.ts CAPTIONS.listening; reducer.ts input-focus). */
const LISTENING = 'Listening'
// The shelf (the kit's deliverables/play-shelf.tsx, shelf-body.tsx and lib/play/shelf.ts emptyMessage).
const SHELF_TITLE = 'Your shelf'
const EMPTY_SHELF = 'Nothing on your shelf yet. Ask Auto for a report or a document and it lands here.'
const START_A_CHAT = 'Start a chat'
/** GET /api/deliverables (hooks/use-deliverables-api.ts fetchListPage), answered with an empty page of PAGE_SIZE. */
const DELIVERABLES_LIST_PATH = '/api/deliverables'
const SHELF_PAGE_SIZE = 24
const EMPTY_SHELF_PAGE = { success: true, deliverables: [], total: 0, limit: SHELF_PAGE_SIZE, offset: 0 }
const OK = 200
// The host's local-edition key nudge over the chat (core's components/local/first-run-nudge.tsx): it
// stays hidden once dismissed, which it remembers in localStorage under FIRST_RUN_NUDGE_STORAGE_KEY
// as '1' and reads after mount. The e2e package does not import the frontend, so the key is copied.
const NUDGE_DISMISSED = { key: 'automatos:local-first-run-nudge:dismissed', value: '1' }
/** first-run-nudge.tsx FIRST_RUN_NUDGE_MESSAGE: the nudge's words, never in a chat shot. */
const NUDGE_MESSAGE = 'Add an LLM key to bring Auto to life'

type Vars = Readonly<Record<string, string>>
type VarTable = Readonly<Record<string, Readonly<Record<string, Vars>>>>

/** A preset as the boot carries it (A-19c): a look document of Play's format. */
interface BootLook {
  readonly id: string
  readonly name: string
  readonly [knob: string]: string | number
}

/** The parts of the page's boot the spec reads (lib/theme-host/boot-types.ts ThemeBoot). */
interface PageBoot {
  readonly prepaint: {
    readonly knobs: VarTable
    readonly pairs: ReadonlyArray<{ readonly table: VarTable }>
    readonly numeric: ReadonlyArray<{ readonly var: string }>
  }
  readonly looks: { readonly defaultLook: string; readonly presets: readonly BootLook[] }
}

interface Shot {
  readonly preset: string
  readonly screen: ScreenId
}

interface Owner {
  readonly preset: string | null
  readonly screen: string | null
}

interface Screen {
  readonly path: string
  /** Waits until the screen shows what its shot is of, under `look`. */
  ready(page: Page, look: BootLook): Promise<void>
  /** Holds the settled frame to what a listing may show, right before it is shot. */
  framed?(page: Page): Promise<void>
}

const SCREENS: Readonly<Record<ScreenId, Screen>> = {
  chat: { path: '/chat', ready: chatAtRest, framed: chatFramed },
  deliverables: { path: '/deliverables', ready: emptyShelf },
  looks: { path: '/looks', ready: looksChosen },
}

function shotsOf(preset: string): Shot[] {
  const screens = preset === DEFAULT_PRESET ? DEFAULT_PRESET_SCREENS : [FRAME_SCREEN]
  return screens.map((screen) => ({ preset, screen }))
}

const ALL_SHOTS: readonly Shot[] = PRESET_IDS.flatMap(shotsOf)

function previewName({ preset, screen }: Shot): string {
  return screen === FRAME_SCREEN ? `${preset}${PNG_SUFFIX}` : `${preset}${SCREEN_SEPARATOR}${screen}${PNG_SUFFIX}`
}

/** listings.py _owner_preset: the preset (and screen) a file stem belongs to, the longest preset id first. */
function ownerPreset(stem: string, presets: readonly string[]): Owner {
  for (const preset of [...presets].sort((a, b) => b.length - a.length)) {
    if (stem === preset) return { preset, screen: null }
    if (stem.startsWith(`${preset}${SCREEN_SEPARATOR}`)) return { preset, screen: stem.slice(preset.length + SCREEN_SEPARATOR.length) }
  }
  return { preset: null, screen: null }
}

function ownerOf({ preset, screen }: Shot): Owner {
  return { preset, screen: screen === FRAME_SCREEN ? null : screen }
}

/** The file is there, a PNG of the viewport's size, at most 512 KB, and named for its preset as the listing reads it. */
function checkPreview(shot: Shot, presets: readonly string[]): void {
  const name = previewName(shot)
  const file = path.join(PREVIEW_DIR, name)
  expect(fs.existsSync(file), `${name} was saved in ${PREVIEW_DIR}`).toBe(true)
  const bytes = fs.readFileSync(file)
  expect(bytes.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE), `${name} starts with the PNG signature`).toBe(true)
  expect(bytes.length, `${name} is at most ${PREVIEW_MAX_BYTES} bytes (package.py IMAGE_MAX_BYTES)`).toBeLessThanOrEqual(PREVIEW_MAX_BYTES)
  const size = { width: bytes.readUInt32BE(PNG_WIDTH_AT), height: bytes.readUInt32BE(PNG_HEIGHT_AT) }
  expect(size, `${name} is the viewport, one image pixel per CSS pixel`).toEqual(VIEWPORT)
  expect(`${PACKAGE_PREVIEW_DIR}${name}`, "the name fits the manifest's previews pattern").toMatch(PREVIEW_PATTERN)
  const stem = name.slice(0, -PNG_SUFFIX.length)
  expect(ownerPreset(stem, presets), `${name} belongs to its preset by the listing's rule`).toEqual(ownerOf(shot))
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/** The page's boot, as THEME_BOOT_SCRIPT read it: the prepaint table and the presets. */
async function readBoot(page: Page): Promise<PageBoot> {
  const boot = JSON.parse((await page.locator(BOOT_ELEMENT).textContent()) || 'null') as Partial<PageBoot> | null
  expect(boot?.prepaint?.knobs, "the page's boot carries installed Play's prepaint table").toBeTruthy()
  expect(boot?.looks?.presets?.length, "the page's boot carries installed Play's presets").toBeGreaterThan(0)
  return boot as PageBoot
}

function presetOf(boot: PageBoot, id: string): BootLook {
  const look = boot.looks.presets.find((preset) => preset.id === id)
  expect(look, `the boot carries the preset ${id}`).toBeDefined()
  return look as BootLook
}

/** The names the pair tables and the numeric rules paint: those depend on more than one knob. */
function comboNames(table: PageBoot['prepaint']): Set<string> {
  const pairs = table.pairs.flatMap((pair) => Object.values(pair.table).flatMap((row) => Object.values(row).flatMap((vars) => Object.keys(vars))))
  return new Set([...pairs, ...table.numeric.map((rule) => rule.var)])
}

/** The variables `look`'s choices paint from the boot's table (later tables win), less the combined ones. */
function choiceVars(table: PageBoot['prepaint'], look: BootLook): Vars {
  const combos = comboNames(table)
  const entries = Object.keys(table.knobs).flatMap((knob) => Object.entries(table.knobs[knob][String(look[knob])] ?? {}))
  const painted = entries.filter(([name, value]) => CSS_VAR_NAME.test(name) && typeof value === 'string' && !combos.has(name))
  return Object.fromEntries(painted.map(([name, value]) => [name, value.trim()]))
}

/** The value of each of `names` on <html>'s inline style now. */
function inlineVars(page: Page, names: readonly string[]): Promise<Vars> {
  return page.evaluate((wanted) => Object.fromEntries(wanted.map((name) => [name, document.documentElement.style.getPropertyValue(name).trim()])), names)
}

async function expectPainted(page: Page, boot: PageBoot, look: BootLook): Promise<void> {
  const vars = choiceVars(boot.prepaint, look)
  expect(Object.keys(vars).length, `the boot's table paints variables for ${look.id}`).toBeGreaterThan(0)
  await expect.poll(() => inlineVars(page, Object.keys(vars)), { message: `<html> carries ${look.id}'s variables` }).toEqual(vars)
}

/** The host runs installed Play here, in Play's frame, and the backend answered its health check. */
async function expectInstalledPlay(page: Page): Promise<void> {
  const html = page.locator('html')
  await expect(html).toHaveAttribute(STATUS_ATTR, 'active', { timeout: THEME_LOAD_TIMEOUT_MS })
  await expect(html).toHaveClass(PLAY_CLASS)
  const frame = page.locator(PLAY_FRAME)
  await expect(frame).toBeVisible()
  await expect(frame.getByRole('status').filter({ hasText: CONNECTED })).toBeVisible({ timeout: THEME_LOAD_TIMEOUT_MS })
}

function lookCard(page: Page, look: BootLook): Locator {
  return page.getByRole('region', { name: LOOKS_ROW, exact: true }).getByRole('button', { name: new RegExp(`^${escapeRegExp(look.name)}\\b`) })
}

async function looksChosen(page: Page, look: BootLook): Promise<void> {
  await expect(page.getByRole('complementary', { name: EDITOR, exact: true })).toBeVisible({ timeout: THEME_LOAD_TIMEOUT_MS })
  await expect(lookCard(page, look)).toHaveAttribute(PRESSED_ATTR, 'true')
}

/**
 * The empty chat with Auto big and at rest. Play focuses an empty chat's message box (the kit's
 * chat/play-chat-view.tsx autoFocus) and Auto listens while it has focus (reducer.ts input-focus,
 * queued behind the first open's hello; only input-blur ends it), so once hello has ended he shows
 * listening. The box is then left, as a member who clicks away leaves it, and Auto rests.
 */
async function chatAtRest(page: Page): Promise<void> {
  const box = messageBox(page)
  await expect(box).toBeVisible({ timeout: THEME_LOAD_TIMEOUT_MS })
  await expect(box, "Play focuses the empty chat's message box").toBeFocused()
  const hero = page.getByRole('region', { name: HERO, exact: true })
  const listening = hero.getByRole('img', { name: LISTENING, exact: true })
  await expect(listening, 'hello is over and Auto listens to the focused box').toBeVisible({ timeout: THEME_LOAD_TIMEOUT_MS })
  await box.blur()
  await expect(hero.getByRole('img', { name: AT_REST, exact: true }), 'the box left, Auto rests').toBeVisible({ timeout: THEME_LOAD_TIMEOUT_MS })
}

/** The chat as a listing shows it: Play alone, with no host nudge over it (openMakeItYours dismissed it). */
async function chatFramed(page: Page): Promise<void> {
  await expect(page.getByText(NUDGE_MESSAGE), "the host's key nudge stays dismissed").toHaveCount(0)
}

async function emptyShelf(page: Page): Promise<void> {
  await expect(page.getByRole('heading', { name: SHELF_TITLE, level: 1, exact: true })).toBeVisible({ timeout: THEME_LOAD_TIMEOUT_MS })
  await expect(page.getByText(EMPTY_SHELF, { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: START_A_CHAT, exact: true })).toBeEnabled()
  await expect(page.locator(AUTO_STAGE).first()).toBeVisible()
}

/** Runs in the page: every stage of Auto's in view shows his still, loaded. */
function stillsLoaded(stage: string): boolean {
  const inView = (element: Element): boolean => {
    const box = element.getBoundingClientRect()
    return box.bottom > 0 && box.right > 0 && box.top < window.innerHeight && box.left < window.innerWidth
  }
  const stages = Array.from(document.querySelectorAll(stage)).filter(inView)
  return stages.every((element) => {
    const still = element.querySelector('img')
    return still !== null && still.complete && still.naturalWidth > 0
  })
}

/** Runs in the page: the fonts' status once every pending load has settled. */
function fontsSettled(): Promise<string> {
  return document.fonts.ready.then(() => document.fonts.status)
}

async function expectSettled(page: Page): Promise<void> {
  await expect(page.locator(`${AUTO_STAGE} video`), 'under reduced motion Auto is his still: no clip plays').toHaveCount(0)
  await page.waitForFunction(stillsLoaded, AUTO_STAGE, { timeout: THEME_LOAD_TIMEOUT_MS })
  expect(await page.evaluate(fontsSettled), 'every font the page uses has loaded').toBe(FONTS_LOADED)
}

/** Shoots until two shots in a row are equal, so no frame caught mid-change is kept. */
async function stableShot(page: Page): Promise<Buffer> {
  let previous = await page.screenshot(SHOT_OPTIONS)
  for (let tries = 1; tries < STABLE_SHOT_TRIES; tries += 1) {
    await page.waitForTimeout(STABLE_SHOT_GAP_MS)
    const next = await page.screenshot(SHOT_OPTIONS)
    if (next.equals(previous)) return next
    previous = next
  }
  throw new Error(`The page never held still: no two shots in a row were equal in ${STABLE_SHOT_TRIES} tries.`)
}

async function shoot(page: Page, boot: PageBoot, look: BootLook, shot: Shot): Promise<void> {
  const file = path.join(PREVIEW_DIR, previewName(shot))
  fs.mkdirSync(PREVIEW_DIR, { recursive: true })
  fs.rmSync(file, { force: true })
  const screen = SCREENS[shot.screen]
  await page.goto(screen.path)
  await expectInstalledPlay(page)
  await screen.ready(page, look)
  await expectPainted(page, boot, look)
  await expectSettled(page)
  await screen.framed?.(page)
  fs.writeFileSync(file, await stableShot(page))
  checkPreview(shot, boot.looks.presets.map((preset) => preset.id))
}

/**
 * Installs and chooses Play's package and opens Make it yours; the conversations and the shelf answer
 * with none, and the host's key nudge is dismissed before the first load.
 */
async function openMakeItYours(page: Page, request: APIRequestContext): Promise<PageBoot> {
  const install = await installKitPackage(request)
  await chooseKitPackage(page, install, APP_URL)
  await page.addInitScript(({ key, value }) => window.localStorage.setItem(key, value), NUDGE_DISMISSED)
  await noStoredConversations(page)
  await page.route((url) => url.pathname === DELIVERABLES_LIST_PATH, (route) => fulfillBackendJson(route, OK, EMPTY_SHELF_PAGE))
  await page.goto(SCREENS.looks.path)
  await expectInstalledPlay(page)
  const boot = await readBoot(page)
  expect(boot.looks.presets.map((preset) => preset.id).sort(), "the boot carries Play 1.1.0's presets").toEqual([...PRESET_IDS].sort())
  expect(boot.looks.defaultLook).toBe(DEFAULT_PRESET)
  await expect(page.getByRole('complementary', { name: EDITOR, exact: true })).toBeVisible({ timeout: THEME_LOAD_TIMEOUT_MS })
  return boot
}

function isSaveOf(response: Response, look: BootLook): boolean {
  if (response.request().method() !== 'PUT' || new URL(response.url()).pathname !== LOOKS_API_PATH) return false
  const body = response.request().postDataJSON() as { active?: { id?: unknown } | null } | null
  return body?.active?.id === look.id
}

/** Picks `look` in the Looks row, as a member does, and waits until the server kept it; the default look is already chosen. */
async function pickLook(page: Page, look: BootLook): Promise<void> {
  const card = lookCard(page, look)
  await expect(card).toBeVisible()
  if ((await card.getAttribute(PRESSED_ATTR)) !== 'true') {
    const saved = page.waitForResponse((response) => isSaveOf(response, look), { timeout: SAVE_TIMEOUT_MS })
    await card.press('Enter')
    expect((await saved).status(), `PUT ${LOOKS_API_PATH} kept ${look.id}`).toBe(OK)
  }
  await expect(card).toHaveAttribute(PRESSED_ATTR, 'true')
}

test.use({ viewport: VIEWPORT, deviceScaleFactor: DEVICE_SCALE, contextOptions: { reducedMotion: 'reduce' } })
test.skip(({ browserName }) => browserName !== ONLY_BROWSER, 'Chromium only: one set of previews, and the specs share one backend user')

test.describe("PRD-261 C-06q1 — installed Play 1.1.0's preview shots", () => {
  test.describe.configure({ timeout: SHOT_TEST_TIMEOUT_MS })

  test.beforeEach(async ({ request }) => {
    await resetPlayParity(request)
  })

  test.afterEach(async ({ request }) => {
    await resetPlayParity(request)
  })

  for (const preset of PRESET_IDS) {
    const names = shotsOf(preset).map(previewName).join(', ')
    test(`under ${preset}: ${names}`, { tag: '@package' }, async ({ page, request }) => {
      const watch = watchPage(page)
      const boot = await openMakeItYours(page, request)
      const look = presetOf(boot, preset)
      await pickLook(page, look)
      for (const shot of shotsOf(preset)) await shoot(page, boot, look, shot)
      expect(watch.pageErrors).toEqual([])
    })
  }
})

test(`the previews folder holds exactly ${ALL_SHOTS.length} previews, each a PNG of at most 512 KB named for its preset`, { tag: '@package' }, () => {
  for (const shot of ALL_SHOTS) checkPreview(shot, PRESET_IDS)
  const saved = fs.readdirSync(PREVIEW_DIR).filter((name) => name.endsWith(PNG_SUFFIX)).sort()
  expect(saved).toEqual(ALL_SHOTS.map(previewName).sort())
})
