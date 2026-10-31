import { expect, test, type Locator, type Page } from '@playwright/test'
import { describeParity, openPlay, PARITY_TAG } from '../../support/play-parity'

// PRD-261 A-17b (@parity): Play's narrow widths (PRD-260 D7, K4), under the installed package
// (A-19f4: since W-A8 the only Play). Play 1.2.0 draws from a phone's width up (its theme.json's
// minWidth, 360, which core keeps: frontend/lib/theme-host/installed-entry.ts):
//
// 1. Below 360 px Play's frame and pages step aside and the core's own chrome draws, recoloured by
//    the look; the look's tone is forced there too, over the user's own, and the core header's theme
//    menu says so; and a look the user chose before paints from the first frame.
// 2. From 360 px to 1023 px Play draws its own phone frame (the kit's themes/play/src/components/play/
//    shell/phone-bar.tsx and phone-menu.tsx, lib/play/phone-layout.ts): a top bar with Menu and the
//    places nav (Chats and Shelf), no sidebar and no core chrome, in the look's colours, with the
//    look's tone forced; the theme menu, one of the host's header utilities, is in Play's Menu, and it
//    says so too. Checked at 900 x 900 (a tablet) and 390 x 844 (a phone).
//
// The look's values are read from the page's own boot data (the JSON element THEME_BOOT_SCRIPT reads,
// lib/theme-host/boot.ts): its prepaint table holds every variable each ground paints, built by the
// same builders the look runtime uses.

/** Play's minWidth (the kit's themes/play/theme.json): below it, the core's chrome. */
const PLAY_MIN_WIDTH = 360
const BELOW_MIN_WIDTH = { width: 340, height: 800 }
/** Play's phone layout: narrower than 1024 px (the kit's lib/play/phone-layout.ts DESKTOP_MIN_WIDTH). */
const PHONE_FRAME_VIEWPORTS = [
  { width: 900, height: 900 },
  { width: 390, height: 844 },
] as const
const DESKTOP = { width: 1440, height: 900 }
const FRAME_PATH = '/agents'
const LOOKS_PATH = '/looks'
/** Play's desktop frame: its sidebar (the kit's themes/play/src/components/play/shell/play-sidebar.tsx). */
const PLAY_SIDEBAR = 'aside[aria-label="Sidebar"]'
/** Play's phone frame: its top bar, Menu and the places nav (the kit's shell/phone-bar.tsx TOP_BAR_LABEL, MENU_LABEL, PLACES_LABEL, CHATS_LABEL, SHELF_LABEL). */
const PHONE_TOP_BAR = 'header[aria-label="Top bar"]'
const MENU_LABEL = 'Menu'
const PLACES_LABEL = 'Chats and shelf'
const CHATS_LABEL = 'Chats'
const SHELF_LABEL = 'Shelf'
/** The menu Menu opens (the kit's shell/phone-menu.tsx MENU_TITLE): a modal dialog holding the host's header utilities. */
const MENU_TITLE = 'Menu'
/** The host's theme menu (components/ui/theme-toggle.tsx), in the core header and in the host's header utilities. */
const THEME_MENU = 'Appearance'
/** The core chrome's header below the desktop width (components/layout/header.tsx). */
const CORE_BRAND = 'Automatos ship mark'
const BOOT_ELEMENT = '#automatos-theme-boot'
/** Ember's ground (the default look) and Daylight's (the kit's themes/play/src/lib/looks/presets.ts). */
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
  await expect(page.locator(PLAY_SIDEBAR)).toHaveCount(0)
  await expect(page.locator(PHONE_TOP_BAR)).toHaveCount(0)
}

/** Play's phone frame: its top bar with Menu, Chats and Shelf; no sidebar and no core chrome. */
async function expectPhoneFrame(page: Page): Promise<void> {
  const bar = page.locator(PHONE_TOP_BAR)
  await expect(bar).toBeVisible()
  await expect(bar.getByRole('button', { name: MENU_LABEL, exact: true })).toBeVisible()
  const places = bar.getByRole('navigation', { name: PLACES_LABEL, exact: true })
  await expect(places.getByRole('link', { name: CHATS_LABEL, exact: true })).toBeVisible()
  await expect(places.getByRole('link', { name: SHELF_LABEL, exact: true })).toBeVisible()
  await expect(page.locator(PLAY_SIDEBAR)).toHaveCount(0)
  await expect(page.getByRole('img', { name: CORE_BRAND })).toHaveCount(0)
}

/** Opens Play's menu from its top bar; the menu, open. */
async function openPhoneMenu(page: Page): Promise<Locator> {
  await page.locator(PHONE_TOP_BAR).getByRole('button', { name: MENU_LABEL, exact: true }).click()
  const menu = page.getByRole('dialog', { name: MENU_TITLE, exact: true })
  await expect(menu).toBeVisible()
  return menu
}

/** <html> carries the default look's ground, as the boot's prepaint table names it. */
async function expectDefaultLookPainted(page: Page): Promise<void> {
  const { defaultGround, grounds } = await bootGrounds(page)
  const ground = trimmed(grounds[defaultGround] ?? {})
  expect(Object.keys(ground).length, "the boot's prepaint table names the default look's ground").toBeGreaterThan(0)
  await expect.poll(() => inlineVars(page, Object.keys(ground))).toEqual(ground)
}

/** A light tone stored as the user's own, before the first load. */
async function storeLightTone(page: Page): Promise<void> {
  await page.addInitScript(({ key, value }) => window.localStorage.setItem(key, value), STORED_TONE)
}

/** The look's tone (Ember's dark) is on <html>, not the user's stored light one. */
async function expectToneForced(page: Page): Promise<void> {
  const html = page.locator('html')
  await expect(html).toHaveClass(DARK)
  await expect(html).not.toHaveClass(LIGHT)
}

/** The theme menu, opened from its button in `holder`, says the look sets the tone and rests its Tone items. */
async function expectThemeMenuSaysSo(page: Page, holder: Page | Locator): Promise<void> {
  await holder.getByRole('button', { name: THEME_MENU, exact: true }).click()
  await expect(page.getByText(TONE_SET_BY_LOOK, { exact: true })).toBeVisible()
  await expect(page.getByRole('menuitemradio', { name: 'Light', exact: true })).toHaveAttribute('aria-disabled', 'true')
}

describeParity(`Play below its minWidth (${PLAY_MIN_WIDTH} px)`, (variant) => {
  test.use({ viewport: BELOW_MIN_WIDTH })

  test("the core chrome draws, in the look's colours", { tag: PARITY_TAG }, async ({ page, request }) => {
    await openPlay(page, request, variant, FRAME_PATH)
    await expectCoreChrome(page)
    await expectDefaultLookPainted(page)
  })

  test("the look's tone is forced over the user's own, and the theme menu says so", { tag: PARITY_TAG }, async ({ page, request }) => {
    await storeLightTone(page)
    await openPlay(page, request, variant, FRAME_PATH)
    await expectCoreChrome(page)
    await expectToneForced(page)
    await expectThemeMenuSaysSo(page, page)
  })

  test('a look chosen before paints from the first frame, and sets the tone', { tag: PARITY_TAG }, async ({ page, request }) => {
    await page.setViewportSize(DESKTOP)
    await openPlay(page, request, variant, LOOKS_PATH)
    const daylight = page.getByRole('region', { name: LOOKS_ROW, exact: true }).getByRole('button', { name: DAYLIGHT })
    await daylight.click()
    await expect(daylight).toHaveAttribute('aria-pressed', 'true')
    await expect(page.locator('html')).toHaveClass(LIGHT)

    await page.setViewportSize(BELOW_MIN_WIDTH)
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

describeParity(`Play's phone frame, from ${PLAY_MIN_WIDTH} px to 1023 px`, (variant) => {
  for (const viewport of PHONE_FRAME_VIEWPORTS) {
    const size = `${viewport.width} x ${viewport.height}`

    test(`at ${size} Play draws its phone frame, in the look's colours`, { tag: PARITY_TAG }, async ({ page, request }) => {
      await page.setViewportSize(viewport)
      await openPlay(page, request, variant, FRAME_PATH)
      await expectPhoneFrame(page)
      await expectDefaultLookPainted(page)
    })

    test(`at ${size} the look's tone is forced over the user's own, and the theme menu in Play's Menu says so`, { tag: PARITY_TAG }, async ({ page, request }) => {
      await page.setViewportSize(viewport)
      await storeLightTone(page)
      await openPlay(page, request, variant, FRAME_PATH)
      await expectPhoneFrame(page)
      await expectToneForced(page)
      await expectThemeMenuSaysSo(page, await openPhoneMenu(page))
    })
  }
})
