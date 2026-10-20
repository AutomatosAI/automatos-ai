import { expect, test, type Page } from '@playwright/test'
import {
  describeParity,
  GAP_INSTALLED_BOOT_PAINTS_NO_STORED_LOOK,
  GAP_INSTALLED_THEME_HAS_NO_ENTRY,
  knownPackageGap,
  openPlay,
  PARITY_TAG,
} from '../../support/play-parity'

// PRD-261 A-17b (@parity): Play below 1024 px (PRD-260 D7, K4), under builtin Play and the
// installed package. Play's frame and pages step aside and the core's own chrome draws, recoloured
// by the look; the look's tone is forced there too, over the user's own; and a look the user chose
// before paints from the first frame. The look's values are read from the page's own boot data
// (the JSON element THEME_BOOT_SCRIPT reads, lib/theme-host/boot.ts): its prepaint table holds every
// variable each ground paints, built by the same builders the look runtime uses.

const NARROW = { width: 900, height: 900 }
const DESKTOP = { width: 1440, height: 900 }
const FRAME_PATH = '/agents'
const LOOKS_PATH = '/looks'
/** Play's frame: its sidebar (components/play/shell/play-sidebar.tsx). */
const PLAY_FRAME = 'aside[aria-label="Sidebar"]'
/** The core chrome's header below 1024 px (components/layout/header.tsx). */
const CORE_BRAND = 'Automatos ship mark'
const BOOT_ELEMENT = '#automatos-theme-boot'
/** Ember's ground (the default look) and Daylight's (lib/looks/presets.ts). */
const DEFAULT_GROUND_KNOB = 'ground'
const DAYLIGHT = /^Daylight\b/
const DAYLIGHT_GROUND = 'paper'
const LOOKS_ROW = 'Looks'
/** next-themes' stored tone (components/providers.tsx storageKey). */
const STORED_TONE = { key: 'automatos-theme', value: 'light' }
const DARK = /\bdark\b/
const LIGHT = /\blight\b/
const TONE_SET_BY_LOOK = 'Set by your look'
const FIRST_FRAME_KEY = '__parityFirstFrame'

type Vars = Readonly<Record<string, string>>

interface BootGround {
  readonly defaultGround: string
  readonly grounds: Readonly<Record<string, Vars>>
}

/** The boot's prepaint table, as the page carries it: each ground's variables and the default look's ground. */
function bootGrounds(page: Page): Promise<BootGround> {
  return page.locator(BOOT_ELEMENT).evaluate((element, knob) => {
    const boot = JSON.parse(element.textContent || 'null') as {
      prepaint?: { knobs?: Record<string, Record<string, Record<string, string>>>; defaults?: Record<string, unknown> }
    } | null
    return {
      defaultGround: String(boot?.prepaint?.defaults?.[knob] ?? ''),
      grounds: boot?.prepaint?.knobs?.[knob] ?? {},
    }
  }, DEFAULT_GROUND_KNOB)
}

/** The value of each of `names` on <html>'s inline style now. */
function inlineVars(page: Page, names: readonly string[]): Promise<Vars> {
  return page.evaluate((wanted) => Object.fromEntries(wanted.map((name) => [name, document.documentElement.style.getPropertyValue(name).trim()])), names)
}

/** Records <html>'s inline custom properties at the first animation frame of every load. */
async function recordFirstFrame(page: Page): Promise<void> {
  await page.addInitScript((key) => {
    requestAnimationFrame(() => {
      const style = document.documentElement.style
      const vars: Record<string, string> = {}
      for (let index = 0; index < style.length; index += 1) {
        const name = style[index]
        if (name.startsWith('--')) vars[name] = style.getPropertyValue(name).trim()
      }
      Object.defineProperty(window, key, { value: vars, configurable: true })
    })
  }, FIRST_FRAME_KEY)
}

function firstFrame(page: Page): Promise<Vars | undefined> {
  return page.evaluate((key) => (window as unknown as Record<string, Vars | undefined>)[key], FIRST_FRAME_KEY)
}

/** Each ground variable trimmed, as the page reads them back. */
function trimmed(vars: Vars): Vars {
  return Object.fromEntries(Object.entries(vars).map(([name, value]) => [name, value.trim()]))
}

async function expectCoreChrome(page: Page): Promise<void> {
  await expect(page.getByRole('img', { name: CORE_BRAND })).toBeVisible()
  await expect(page.locator(PLAY_FRAME)).toHaveCount(0)
}

describeParity('Play below 1024 px', (variant) => {
  test.use({ viewport: NARROW })

  test("the core chrome draws, in the look's colours", { tag: PARITY_TAG }, async ({ page, request }) => {
    await openPlay(page, request, variant, FRAME_PATH)
    await expectCoreChrome(page)
    const { defaultGround, grounds } = await bootGrounds(page)
    const ground = trimmed(grounds[defaultGround] ?? {})
    expect(Object.keys(ground).length, "the boot's prepaint table names the default look's ground").toBeGreaterThan(0)
    await expect.poll(() => inlineVars(page, Object.keys(ground))).toEqual(ground)
  })

  test("the look's tone is forced over the user's own, and the theme menu says so", { tag: PARITY_TAG }, async ({ page, request }) => {
    knownPackageGap(variant, GAP_INSTALLED_THEME_HAS_NO_ENTRY)
    await page.addInitScript(({ key, value }) => window.localStorage.setItem(key, value), STORED_TONE)
    await openPlay(page, request, variant, FRAME_PATH)
    await expectCoreChrome(page)
    const html = page.locator('html')
    await expect(html).toHaveClass(DARK)
    await expect(html).not.toHaveClass(LIGHT)

    await page.getByRole('button', { name: 'Appearance', exact: true }).click()
    await expect(page.getByText(TONE_SET_BY_LOOK, { exact: true })).toBeVisible()
    await expect(page.getByRole('menuitemradio', { name: 'Light', exact: true })).toHaveAttribute('aria-disabled', 'true')
  })

  test('a look chosen before paints from the first frame, and sets the tone', { tag: PARITY_TAG }, async ({ page, request }) => {
    knownPackageGap(variant, GAP_INSTALLED_BOOT_PAINTS_NO_STORED_LOOK)
    await page.setViewportSize(DESKTOP)
    await openPlay(page, request, variant, LOOKS_PATH)
    const daylight = page.getByRole('region', { name: LOOKS_ROW, exact: true }).getByRole('button', { name: DAYLIGHT })
    await daylight.click()
    await expect(daylight).toHaveAttribute('aria-pressed', 'true')
    await expect(page.locator('html')).toHaveClass(LIGHT)

    await page.setViewportSize(NARROW)
    await recordFirstFrame(page)
    await page.goto(FRAME_PATH)
    await expectCoreChrome(page)
    const paper = trimmed((await bootGrounds(page)).grounds[DAYLIGHT_GROUND] ?? {})
    expect(Object.keys(paper).length, "the boot's prepaint table names Daylight's ground").toBeGreaterThan(0)
    await expect.poll(() => firstFrame(page)).toBeDefined()
    expect(await firstFrame(page)).toMatchObject(paper)
    await expect(page.locator('html')).toHaveClass(LIGHT)
  })
})
