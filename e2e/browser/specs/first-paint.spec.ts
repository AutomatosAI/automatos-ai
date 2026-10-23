import { expect, test, type APIRequestContext, type Page } from '@playwright/test'
import {
  answerCoreChoice,
  chooseCoreStyle,
  chooseStyle,
  CORE_STYLES,
  STUDIO_STYLE,
  THEME_E2E_PATH,
  type CoreStyle,
} from '../support/hello-theme'

// PRD-261 A-03 (@smoke, so it runs in every build of the lane): first paint from data, under the
// Classic and under the Studio style cookie (A-19f4). Since W-A8 Play is not in core: it runs only
// as the package a workspace installs (specs/play-parity/). So a core page asks for none of Play:
// no stylesheet that declares Play's tokens, none of Play's face variables, nothing from builtin
// Play's old mascot folder or from a theme's files. The cookies builtin Play ran under (the legacy
// `play`, and its builtin theme style) load Studio, with no `play` class on <html> before the first
// animation frame or after, and no Play stylesheet in the server's own response. Builtin Play's old
// page, /conversations, answers the 404. The theme import map is on every page, ahead of every
// module script. Each test opens its own page.
//
// The old cookie and folder are built from their parts (BUILTIN_THEME_PREFIX, MASCOT_SEGMENT), so a
// search of e2e/ for builtin Play's names finds nothing (A-19f4's seam check): what is gone is only
// ever asserted absent here.

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const STYLE_COOKIE = 'automatos-style'
const PLAY = 'play'
/** A builtin theme's style was `theme:builtin:<id>`; builtin Play's id was `play`. */
const BUILTIN_THEME_PREFIX = 'theme:builtin:'
/** The cookies builtin Play ran under, the legacy one and its theme style: both now parse to Studio. */
const RETIRED_PLAY_COOKIES = [PLAY, `${BUILTIN_THEME_PREFIX}${PLAY}`]
/** Builtin Play's mascot files were in public/ under this folder: the first segment of their path. */
const MASCOT_SEGMENT = 'auto'
/** A theme's own files (the files proxy): a core style asks for none. */
const THEME_FILES_ROOT = '/api/themes/files/'
/** A declaration only Play's stylesheet makes (its page colour token on html.play), in core or in the package. */
const PLAY_THEME_MARKER = '--pl-color-bg:'
/** The prefix of Play's custom properties: the boot paints none of them on <html> for a core style. */
const PLAY_TOKEN_PREFIX = '--pl-'
/** The custom properties builtin Play's face classes defined on <html> (core's app/play-fonts.ts until W-A8). */
const PLAY_FONT_VARS = ['--pl-font-instrument-serif', '--pl-font-bricolage', '--pl-font-fredoka']
const NO_PLAY_FACES = Object.fromEntries(PLAY_FONT_VARS.map((name) => [name, '']))
const STUDIO_CLASS = /\bstudio\b/
const PLAY_CLASS = /\bplay\b/
const CONVERSATIONS_PATH = '/conversations'
const NOT_FOUND = 404
/** app/not-found.tsx: Next's own 404 title, and its plain page's heading and words. */
const NOT_FOUND_TITLE = '404: This page could not be found.'
const NOT_FOUND_HEADING = '404'
const NOT_FOUND_WORDS = 'This page could not be found.'
const IMPORT_MAP_TAG = '<script type="importmap">'
const LINK_TAG = /<link\b[^>]*>/gi
const STYLESHEET_REL = /\brel="stylesheet"/i
const HREF_ATTR = /\bhref="([^"]+)"/i
const ESCAPED_AMPERSAND = /&amp;/g
const MODULE_SCRIPT_TAG = /<script\b[^>]*\btype="module"[^>]*>/gi
const FIRST_PAINT_TAGS = ['@smoke', '@first-paint']

interface FirstFrame {
  readonly classes: string[]
  /** The Play custom properties set inline on <html>. */
  readonly playVars: string[]
}

type FirstFrameWindow = Window & { __firstFrame?: FirstFrame }

/** Every stylesheet the page fetched, as text. Register before the first navigation. */
function watchStylesheets(page: Page): () => Promise<string[]> {
  const bodies: Promise<string>[] = []
  page.on('response', (response) => {
    if (response.request().resourceType() === 'stylesheet') bodies.push(response.text())
  })
  return () => Promise.all(bodies)
}

/** The URL of every request the page makes. Register before the first navigation. */
function watchRequests(page: Page): string[] {
  const urls: string[] = []
  page.on('request', (request) => urls.push(request.url()))
  return urls
}

/** The paths of the requests for builtin Play's old mascot folder or for a theme's files. */
function playFileRequests(urls: readonly string[]): string[] {
  return urls
    .map((url) => new URL(url).pathname)
    .filter((pathname) => pathname.split('/')[1] === MASCOT_SEGMENT || pathname.startsWith(THEME_FILES_ROOT))
}

/** The computed value of each of Play's face variables on <html> ('' when nothing defines it). */
function playFaceVars(page: Page): Promise<Record<string, string>> {
  return page.evaluate((names) => {
    const root = getComputedStyle(document.documentElement)
    return Object.fromEntries(names.map((name) => [name, root.getPropertyValue(name).trim()]))
  }, PLAY_FONT_VARS)
}

/** The stylesheets a server response links in its HTML, as the hrefs it wrote. */
function linkedStylesheets(html: string): string[] {
  return Array.from(html.matchAll(LINK_TAG), (match) => match[0])
    .filter((tag) => STYLESHEET_REL.test(tag))
    .flatMap((tag) => {
      const href = HREF_ATTR.exec(tag)?.[1]
      return href === undefined ? [] : [href.replace(ESCAPED_AMPERSAND, '&')]
    })
}

/** The HTML the server sends for the page under the style cookie. */
async function serverHtml(request: APIRequestContext, cookie: string): Promise<string> {
  const response = await request.get(THEME_E2E_PATH, { headers: { cookie: `${STYLE_COOKIE}=${cookie}` } })
  expect(response.ok()).toBe(true)
  return response.text()
}

/** The text of every stylesheet the server's HTML for the page links, under the style cookie. */
async function serverLinkedCss(request: APIRequestContext, cookie: string): Promise<string[]> {
  const hrefs = linkedStylesheets(await serverHtml(request, cookie))
  return Promise.all(hrefs.map(async (href) => (await request.get(href)).text()))
}

/** Records <html>'s classes and its inline Play custom properties at the first animation frame. */
async function recordFirstFrame(page: Page): Promise<void> {
  await page.addInitScript((prefix) => {
    requestAnimationFrame(() => {
      const root = document.documentElement
      const playVars = Array.from(root.style).filter((name) => name.startsWith(prefix))
      ;(window as FirstFrameWindow).__firstFrame = { classes: Array.from(root.classList), playVars }
    })
  }, PLAY_TOKEN_PREFIX)
}

async function firstFrame(page: Page): Promise<FirstFrame | undefined> {
  await page.waitForFunction(() => (window as FirstFrameWindow).__firstFrame !== undefined)
  return page.evaluate(() => (window as FirstFrameWindow).__firstFrame)
}

/** <html> carries the style's own class (Studio's, or none for Classic) and never Play's. */
async function expectStyleClasses(page: Page, core: CoreStyle): Promise<void> {
  const html = page.locator('html')
  if (core.studio) await expect(html).toHaveClass(STUDIO_CLASS)
  else await expect(html).not.toHaveClass(STUDIO_CLASS)
  await expect(html).not.toHaveClass(PLAY_CLASS)
}

/** Where the import map sits among the page's scripts once it has run. */
function importMapOrder(page: Page): Promise<{ maps: number; allAfter: boolean }> {
  return page.evaluate(() => {
    const maps = Array.from(document.scripts).filter((script) => script.type === 'importmap')
    const modules = Array.from(document.scripts).filter((script) => script.type === 'module')
    const first = maps[0]
    const allAfter = first !== undefined && modules.every((script) => (first.compareDocumentPosition(script) & Node.DOCUMENT_POSITION_FOLLOWING) !== 0)
    return { maps: maps.length, allAfter }
  })
}

for (const core of CORE_STYLES) {
  test(`under the '${core.style}' cookie a page asks for none of Play: no stylesheet, no face, none of its files`, { tag: FIRST_PAINT_TAGS }, async ({ page }) => {
    const requests = watchRequests(page)
    const stylesheets = watchStylesheets(page)
    await chooseCoreStyle(page, core, APP_URL)
    await page.goto(THEME_E2E_PATH)
    await expect(page.locator(core.shell)).toBeVisible()
    await page.waitForLoadState('load')

    const sheets = await stylesheets()
    expect(sheets.length, 'the page fetched its stylesheets, so the check below is not empty').toBeGreaterThan(0)
    expect(sheets.filter((css) => css.includes(PLAY_THEME_MARKER))).toEqual([])
    expect(await playFaceVars(page)).toEqual(NO_PLAY_FACES)
    expect(playFileRequests(requests)).toEqual([])
    await expectStyleClasses(page, core)
  })
}

for (const cookie of RETIRED_PLAY_COOKIES) {
  test(`the retired Play cookie '${cookie}' loads Studio: no play class from the first frame, no Play CSS, none of its files`, { tag: FIRST_PAINT_TAGS }, async ({ page, request }) => {
    const requests = watchRequests(page)
    const stylesheets = watchStylesheets(page)
    // The server agrees with the style the cookie now stands for, whatever another spec stored.
    await answerCoreChoice(page, STUDIO_STYLE.style)
    await chooseStyle(page, cookie, APP_URL)
    await recordFirstFrame(page)
    await page.goto(THEME_E2E_PATH)

    const frame = await firstFrame(page)
    expect(frame?.classes).toEqual(expect.arrayContaining([STUDIO_STYLE.style]))
    expect(frame?.classes).not.toContain(PLAY)
    expect(frame?.playVars).toEqual([])

    await expect(page.locator(STUDIO_STYLE.shell)).toBeVisible()
    await page.waitForLoadState('load')
    await expectStyleClasses(page, STUDIO_STYLE)
    expect((await stylesheets()).filter((css) => css.includes(PLAY_THEME_MARKER))).toEqual([])
    expect(playFileRequests(requests)).toEqual([])

    // Nor does the server's own response link a Play stylesheet under the cookie.
    const linked = await serverLinkedCss(request, cookie)
    expect(linked.length).toBeGreaterThan(0)
    expect(linked.filter((css) => css.includes(PLAY_THEME_MARKER))).toEqual([])
  })
}

for (const core of CORE_STYLES) {
  test(`under the '${core.style}' cookie /conversations, builtin Play's old page, answers the 404`, { tag: FIRST_PAINT_TAGS }, async ({ page }) => {
    await chooseCoreStyle(page, core, APP_URL)
    const response = await page.goto(CONVERSATIONS_PATH)

    expect(response?.status()).toBe(NOT_FOUND)
    expect(new URL(page.url()).pathname, 'no redirect: the old page is gone').toBe(CONVERSATIONS_PATH)
    await expect(page).toHaveTitle(NOT_FOUND_TITLE)
    await expect(page.getByRole('heading', { level: 1, name: NOT_FOUND_HEADING })).toBeVisible()
    await expect(page.getByText(NOT_FOUND_WORDS, { exact: true })).toBeVisible()
    await expect(page.locator('html')).not.toHaveClass(PLAY_CLASS)
  })
}

for (const core of CORE_STYLES) {
  test(`under the '${core.style}' cookie the import map is on the page, ahead of every module script`, { tag: FIRST_PAINT_TAGS }, async ({ page, request }) => {
    const html = await serverHtml(request, core.style)
    const mapAt = html.indexOf(IMPORT_MAP_TAG)
    expect(mapAt).toBeGreaterThanOrEqual(0)
    expect(html.indexOf(IMPORT_MAP_TAG, mapAt + 1)).toBe(-1)
    const moduleTags = Array.from(html.matchAll(MODULE_SCRIPT_TAG), (match) => match.index ?? 0)
    expect(moduleTags.filter((at) => at < mapAt)).toEqual([])

    await chooseCoreStyle(page, core, APP_URL)
    await page.goto(THEME_E2E_PATH)
    await expect(page.locator(core.shell)).toBeVisible()
    expect(await importMapOrder(page)).toEqual({ maps: 1, allAfter: true })
  })
}
