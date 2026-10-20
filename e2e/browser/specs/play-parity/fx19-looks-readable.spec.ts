import { expect, test, type APIRequestContext, type Locator, type Page } from '@playwright/test'
import { computedStyleOf, resolvedColour } from '../../support/computed-style'
import {
  describeParity,
  GAP_INSTALLED_THEME_HAS_NO_ENTRY,
  knownPackageGap,
  openPlay,
  PARITY_TAG,
  type PlayVariant,
} from '../../support/play-parity'

// PRD-261 A-17b (@parity), P260-FX-19 (M18, M19, M73): Make it yours stays readable, under builtin
// Play and the installed package, read from what the page draws (computed styles).
//
// 1. M18, M19 on the default look (Ember): every chosen option and the chosen look card are edged in
//    the accent as text (--pl-color-accent-text), every accent swatch carries its text2 edge, the
//    colour samples keep their own colours in forced colours (forced-color-adjust: none), and in
//    forced colours (Chromium emulates them) a chosen option gets its 2px outline.
// 2. M18 on Paper (Daylight): there the accent as text is not the raw accent, and the chosen option,
//    the chosen card and the swatch edges still use it.
// 3. M73: a 40-character look name with no space keeps "Make it yours." readable at 1024 px and
//    stays whole in its card.

const LOOKS_PATH = '/looks'
const EDITOR = 'Edit the look'
const LOOKS_ROW = 'Looks'
const MANAGE = 'This look'
const DAYLIGHT = /^Daylight\b/
/** The option groups whose chosen button is edged (components/looks/look-options.tsx). */
const EDGED_GROUPS = ['Background', 'Type', "Auto's shell", "Auto's light · ring"]
const ACCENT_GROUP = /^Accent · /
const ACCENT_TEXT = 'var(--pl-color-accent-text)'
const RAW_ACCENT = 'var(--pl-color-accent)'
const TEXT_2 = 'var(--pl-color-text-2)'
const PRESSED = '[aria-pressed="true"]'
const INSET = 'inset'
const NO_ADJUST = 'none'
const FORCED_OUTLINE = { style: 'solid', width: '2px' }
/** 40 characters with no space (the M73 case, docs/testing/PRD-260-PLAY-RUN-CHECKLIST.md step 43). */
const LONG_NAME = 'abcdefghijklmnopqrstuvwxyz0123456789abcd'
const NARROW_DESKTOP = { width: 1024, height: 900 }
/** .headerSide's max-width (components/looks/looks.module.css): the status never takes more of the row. */
const STATUS_SIDE_MAX = 0.4
const HEADING_SIDE_MIN = 0.5
const ROUNDING_PX = 1

async function openLooks(page: Page, request: APIRequestContext, variant: PlayVariant): Promise<Locator> {
  await openPlay(page, request, variant, LOOKS_PATH)
  const editor = page.getByRole('complementary', { name: EDITOR, exact: true })
  await expect(editor).toBeVisible()
  return editor
}

/** The chosen option of each edged group, and the chosen look card. */
function chosenEdges(page: Page, editor: Locator): Locator[] {
  const options = EDGED_GROUPS.map((name) => editor.getByRole('group', { name, exact: true }).locator(PRESSED))
  const card = page.getByRole('region', { name: LOOKS_ROW, exact: true }).locator(PRESSED)
  return [...options, card]
}

async function expectEdgedInAccentText(page: Page, editor: Locator): Promise<void> {
  const accentText = await resolvedColour(editor, ACCENT_TEXT)
  for (const chosen of chosenEdges(page, editor)) {
    await expect(chosen).toHaveCount(1)
    expect(await computedStyleOf(chosen, 'border-top-color')).toBe(accentText)
  }
}

async function expectSwatchEdges(editor: Locator): Promise<void> {
  const text2 = await resolvedColour(editor, TEXT_2)
  const swatches = editor.getByRole('group', { name: ACCENT_GROUP }).getByRole('button')
  await expect(swatches.first()).toBeVisible()
  for (const swatch of await swatches.all()) {
    const shadow = await computedStyleOf(swatch, 'box-shadow')
    expect(shadow).toContain(INSET)
    expect(shadow).toContain(text2)
  }
}

async function expectSamplesKeepColours(page: Page, editor: Locator): Promise<void> {
  const supported = await page.evaluate(() => CSS.supports('forced-color-adjust', 'none'))
  if (!supported) return
  const samples = await editor.getByRole('group', { name: ACCENT_GROUP }).getByRole('button').all()
  expect(samples.length).toBeGreaterThan(0)
  for (const sample of samples) expect(await computedStyleOf(sample, 'forced-color-adjust')).toBe(NO_ADJUST)
}

async function expectForcedColourOutlines(page: Page, editor: Locator, browserName: string): Promise<void> {
  await page.emulateMedia({ forcedColors: 'active' })
  const forced = await page.evaluate(() => window.matchMedia('(forced-colors: active)').matches)
  if (browserName === 'chromium') expect(forced, 'Chromium emulates forced colours').toBe(true)
  if (!forced) return
  for (const chosen of chosenEdges(page, editor)) {
    expect(await computedStyleOf(chosen, 'outline-style')).toBe(FORCED_OUTLINE.style)
    expect(await computedStyleOf(chosen, 'outline-width')).toBe(FORCED_OUTLINE.width)
  }
}

/** Saves the active look as the user's own and renames it, through Make it yours. */
async function saveAndRename(page: Page, editor: Locator, name: string): Promise<void> {
  await editor.getByRole('button', { name: 'Save as new look', exact: true }).click()
  const manage = page.getByRole('region', { name: MANAGE, exact: true })
  await manage.getByRole('button', { name: 'Rename', exact: true }).click()
  await manage.getByLabel('New name', { exact: true }).fill(name)
  await manage.getByRole('button', { name: 'Save name', exact: true }).click()
  await expect(page.getByRole('region', { name: LOOKS_ROW, exact: true }).getByRole('button', { name: new RegExp(`^${name}\\b`) })).toBeVisible()
}

interface HeaderGeometry {
  readonly row: number
  readonly side: number
  readonly heading: number
}

/** The widths of the header row, the status's side and the heading's side, from the status line. */
function headerGeometry(status: Locator): Promise<HeaderGeometry> {
  return status.evaluate((line) => {
    const side = line.parentElement
    const row = side?.parentElement
    const heading = row?.firstElementChild
    const width = (element: Element | null | undefined) => element?.getBoundingClientRect().width ?? 0
    return { row: width(row), side: width(side), heading: width(heading) }
  })
}

describeParity('Make it yours stays readable (FX-19)', (variant) => {
  test('chosen options and card are edged in the accent as text, swatches are edged, samples keep their colours in forced colours', { tag: PARITY_TAG }, async ({ page, request, browserName }) => {
    const editor = await openLooks(page, request, variant)
    await expectEdgedInAccentText(page, editor)
    await expectSwatchEdges(editor)
    await expectSamplesKeepColours(page, editor)
    await expectForcedColourOutlines(page, editor, browserName)
  })

  test('on Paper (Daylight) the edges use the accent as text, not the raw accent', { tag: PARITY_TAG }, async ({ page, request }) => {
    knownPackageGap(variant, GAP_INSTALLED_THEME_HAS_NO_ENTRY)
    const editor = await openLooks(page, request, variant)
    const daylight = page.getByRole('region', { name: LOOKS_ROW, exact: true }).getByRole('button', { name: DAYLIGHT })
    await daylight.click()
    await expect(daylight).toHaveAttribute('aria-pressed', 'true')
    await expect(page.locator('html')).toHaveClass(/\blight\b/)
    expect(await resolvedColour(editor, ACCENT_TEXT)).not.toBe(await resolvedColour(editor, RAW_ACCENT))
    await expectEdgedInAccentText(page, editor)
    await expectSwatchEdges(editor)
  })

  test('a 40-character name keeps the heading readable at 1024 px and stays whole in its card', { tag: PARITY_TAG }, async ({ page, request }) => {
    knownPackageGap(variant, GAP_INSTALLED_THEME_HAS_NO_ENTRY)
    const editor = await openLooks(page, request, variant)
    await saveAndRename(page, editor, LONG_NAME)
    await page.setViewportSize(NARROW_DESKTOP)
    await editor.getByRole('group', { name: 'Background', exact: true }).getByRole('button', { name: 'Obsidian' }).click()

    const status = page.getByRole('status').filter({ hasText: `${LONG_NAME} · edited` })
    await expect(status).toBeVisible()
    const geometry = await headerGeometry(status)
    expect(geometry.side).toBeLessThanOrEqual(geometry.row * STATUS_SIDE_MAX + ROUNDING_PX)
    expect(geometry.heading).toBeGreaterThanOrEqual(geometry.row * HEADING_SIDE_MIN)

    const card = page.getByRole('region', { name: LOOKS_ROW, exact: true }).getByRole('button', { name: new RegExp(`^${LONG_NAME}\\b`) })
    const clipped = await card.evaluate((button) => button.scrollWidth > button.clientWidth)
    expect(clipped).toBe(false)
  })
})
