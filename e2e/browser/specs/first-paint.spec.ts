import { expect, test, type Page } from '@playwright/test'
import { THEME_E2E_PATH } from '../support/hello-theme'

// PRD-261 A-03 (@smoke, so it runs in every build of the lane): first paint from data. A Studio
// page requests none of builtin Play's stylesheet and none of its face files; under the Play
// cookie (legacy and current) <html> carries 'studio play' and the look's variables before the
// first animation frame; and the theme import map is on a Studio page, ahead of every module
// script. Each test opens its own page.

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const STYLE_COOKIE = 'automatos-style'
const PLAY_COOKIES = ['play', 'theme:builtin:play']
const APPEARANCE_DEFAULTS = { key: 'automatos-appearance-defaults', value: '2026-09-17-studio-dark' }
/** A declaration only app/play-theme.css makes: Play's page colour token on html.play. */
const PLAY_THEME_MARKER = '--pl-color-bg:'
/** The custom properties Play's face classes define on <html> (app/play-fonts.ts). */
const PLAY_FONT_VARS = ['--pl-font-instrument-serif', '--pl-font-bricolage', '--pl-font-fredoka']
/** A ground's variable, a Studio token and the glow: what the boot paints from the look. */
const LOOK_VARS = ['--pl-ground-bg', '--background', '--pl-glow']
const STUDIO_AND_PLAY = ['studio', 'play']
const IMPORT_MAP_TAG = '<script type="importmap">'
const MODULE_SCRIPT_TAG = /<script\b[^>]*\btype="module"[^>]*>/gi

interface FirstFrame {
  readonly classes: string[]
  readonly vars: Record<string, string>
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

/** The url() sources of the @font-face rules behind Play's face variables, as absolute URLs. */
async function playFontUrls(page: Page): Promise<string[]> {
  return page.evaluate((vars) => {
    const unquote = (value: string) => value.trim().replace(/^['"]|['"]$/g, '')
    const root = getComputedStyle(document.documentElement)
    const families = vars.flatMap((name) => root.getPropertyValue(name).split(',')).map(unquote).filter(Boolean)
    const urls: string[] = []
    for (const sheet of Array.from(document.styleSheets)) {
      const rules = Array.from(sheet.cssRules).filter((rule): rule is CSSFontFaceRule => rule instanceof CSSFontFaceRule)
      for (const rule of rules.filter((face) => families.includes(unquote(face.style.getPropertyValue('font-family'))))) {
        const sources = Array.from(rule.style.getPropertyValue('src').matchAll(/url\(\s*['"]?([^'")]+)['"]?\s*\)/g))
        urls.push(...sources.map((match) => new URL(match[1], sheet.href ?? document.baseURI).href))
      }
    }
    return urls
  }, PLAY_FONT_VARS)
}

async function keepTheChosenStyle(page: Page): Promise<void> {
  // Without the defaults marker, a fresh browser is moved to Studio once (ui-style-context.tsx).
  await page.addInitScript(({ key, value }) => {
    window.localStorage.setItem(key, value)
  }, APPEARANCE_DEFAULTS)
}

test('a Studio page requests none of Play: no stylesheet chunk, no face file', { tag: ['@smoke', '@first-paint'] }, async ({ page }) => {
  const requests: string[] = []
  page.on('request', (request) => requests.push(request.url()))
  const stylesheets = watchStylesheets(page)
  await page.goto(THEME_E2E_PATH)
  await expect(page.locator('.sh-shell')).toBeVisible()
  await page.waitForLoadState('load')

  const sheets = await stylesheets()
  expect(sheets.length).toBeGreaterThan(0)
  expect(sheets.filter((css) => css.includes(PLAY_THEME_MARKER))).toEqual([])

  const fontUrls = await playFontUrls(page)
  expect(fontUrls.length, "Play's @font-face rules are on the page, so the check below is not empty").toBeGreaterThan(0)
  expect(fontUrls.filter((url) => requests.includes(url))).toEqual([])
})

for (const cookie of PLAY_COOKIES) {
  test(`under the '${cookie}' cookie, <html> has 'studio play' and the look before the first frame`, { tag: ['@smoke', '@first-paint'] }, async ({ page }) => {
    await keepTheChosenStyle(page)
    await page.addInitScript((vars) => {
      requestAnimationFrame(() => {
        const root = document.documentElement
        const painted = Object.fromEntries(vars.map((name) => [name, root.style.getPropertyValue(name)]))
        ;(window as FirstFrameWindow).__firstFrame = { classes: Array.from(root.classList), vars: painted }
      })
    }, LOOK_VARS)
    await page.context().addCookies([{ name: STYLE_COOKIE, value: cookie, url: APP_URL }])
    const stylesheets = watchStylesheets(page)
    await page.goto(THEME_E2E_PATH)
    await page.waitForFunction(() => (window as FirstFrameWindow).__firstFrame !== undefined)

    const frame = await page.evaluate(() => (window as FirstFrameWindow).__firstFrame)
    expect(frame?.classes).toEqual(expect.arrayContaining(STUDIO_AND_PLAY))
    for (const name of LOOK_VARS) expect({ name, set: (frame?.vars[name] ?? '') !== '' }).toEqual({ name, set: true })

    await page.waitForLoadState('load')
    expect((await stylesheets()).some((css) => css.includes(PLAY_THEME_MARKER))).toBe(true)
  })
}

test('the import map is on a Studio page, ahead of every module script', { tag: ['@smoke', '@first-paint'] }, async ({ page, request }) => {
  const html = await (await request.get(THEME_E2E_PATH)).text()
  const mapAt = html.indexOf(IMPORT_MAP_TAG)
  expect(mapAt).toBeGreaterThanOrEqual(0)
  expect(html.indexOf(IMPORT_MAP_TAG, mapAt + 1)).toBe(-1)
  const moduleTags = Array.from(html.matchAll(MODULE_SCRIPT_TAG), (match) => match.index ?? 0)
  expect(moduleTags.filter((at) => at < mapAt)).toEqual([])

  await page.goto(THEME_E2E_PATH)
  await expect(page.locator('.sh-shell')).toBeVisible()
  const order = await page.evaluate(() => {
    const maps = Array.from(document.scripts).filter((script) => script.type === 'importmap')
    const modules = Array.from(document.scripts).filter((script) => script.type === 'module')
    const first = maps[0]
    const allAfter = first !== undefined && modules.every((script) => (first.compareDocumentPosition(script) & Node.DOCUMENT_POSITION_FOLLOWING) !== 0)
    return { maps: maps.length, allAfter }
  })
  expect(order).toEqual({ maps: 1, allAfter: true })
})
