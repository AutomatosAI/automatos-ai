/**
 * PRD-244 D1, PRD-261 A-02a, A-19f1 — the Style axis is Classic, Studio or an installed theme
 * (`theme:install:<uuid>`). No theme is compiled into the app, so the axis lists Classic and Studio
 * only; Studio renders as <html class="studio">, Classic with no style class, and an installed theme
 * on its base, Studio, until its own boot answers (its own class comes with that boot). Anything else
 * the cookie may say falls back to the default, and the appearance defaults are untouched.
 *
 * PRD-260 M07, PRD-261 A-03: the classes reach <html> through the one constant <head> script,
 * THEME_BOOT_SCRIPT, applying the style's boot; it leaves every other class (the tone, the font
 * variables) alone. The same assertions run the boot of every style, an installed theme's included.
 *
 * Replaces lib/__tests__/ui-style-play.test.ts (the core ledger, A-19f2); the Classic and Studio
 * first paint of the root layout is app/__tests__/root-layout.test.tsx's.
 *
 * The last block holds the values builtin Play left in browsers and on the server (its style cookie,
 * its server choice, its tone cookie): each now names nothing. It is the one place in the frontend
 * allowed to spell them (lib/__tests__/prd261-no-theme-in-core.test.ts excepts it by its title).
 */
import { afterEach, describe, expect, it } from 'vitest'

import { createThemeBootResolver, baseThemeBoot, THEME_BOOT_SCRIPT, type ThemeBoot } from '@/lib/theme-host/boot'
import { reconcileTarget, styleOfChoice } from '@/lib/theme-host/choice-sync'
import { DEFAULT_THEME_BASE } from '@/lib/theme-host/registry'
import { THEME_TONE_COOKIE, themeToneCookieValue } from '@/lib/theme-host/looks/tone-cookie'
import {
  APPEARANCE_DEFAULTS_KEY,
  APPEARANCE_DEFAULTS_VERSION,
  DEFAULT_TONE,
  DEFAULT_UI_STYLE,
  STUDIO_HTML_CLASS,
  STYLE_HTML_CLASS_NAMES,
  UI_STYLES,
  isUiStyle,
  parseUiStyle,
  rendersStudio,
  styleHtmlClasses,
  type UiStyle,
} from '@/lib/ui-style'

import { cookieReader, runThemeBoot } from '@/lib/theme-host/__tests__/boot-harness'
import { PLAY_PACKAGE_ENTRY } from '@/lib/theme-host/__tests__/installed-entry-fixture'
import { HELLO_ENTRY, HELLO_THEME_ID, helloThemeBoot } from '@/lib/theme-host/looks/__tests__/hello-looks'

/** An installed theme's style: its classes are its base's (Studio) until its boot answers. */
const INSTALL_STYLE: UiStyle = 'theme:install:0b7e4c1a-9f3d-4c2e-8a51-3d2f6b7a9c10'
const BOOTED_STYLES: readonly UiStyle[] = [...UI_STYLES, INSTALL_STYLE]

describe('PRD-261 A-19f1 — the Style axis: Classic, Studio or an installed theme', () => {
  it('lists Classic and Studio, in that order, and no compiled-in theme', () => {
    expect(UI_STYLES).toEqual(['classic', 'studio'])
  })

  it("accepts Classic, Studio and an installed theme's style from the cookie", () => {
    for (const style of BOOTED_STYLES) {
      expect(isUiStyle(style)).toBe(true)
      expect(parseUiStyle(style)).toBe(style)
    }
  })

  it('falls back to the default for a tampered, unknown or missing cookie', () => {
    const tampered = [
      'STUDIO', ' studio', 'studio ', 'studio;classic', 'classic studio', '<script>', 'dark', '',
      'theme:install:x', 'theme:install:0B7E4C1A-9F3D-4C2E-8A51-3D2F6B7A9C10', `${INSTALL_STYLE} `, 'theme:other', 'install:x',
    ]
    for (const raw of tampered) expect(parseUiStyle(raw), JSON.stringify(raw)).toBe(DEFAULT_UI_STYLE)
    expect(parseUiStyle(null)).toBe(DEFAULT_UI_STYLE)
    expect(parseUiStyle(undefined)).toBe(DEFAULT_UI_STYLE)
    for (const raw of [1, {}, ['studio'], null, undefined, true]) expect(isUiStyle(raw)).toBe(false)
  })

  it('maps each style to its <html> classes: none for Classic, Studio for Studio and a theme drawn on it', () => {
    expect(STUDIO_HTML_CLASS).toBe('studio')
    expect(DEFAULT_THEME_BASE).toBe('studio')
    expect(styleHtmlClasses('classic')).toEqual([])
    expect(styleHtmlClasses('studio')).toEqual([STUDIO_HTML_CLASS])
    expect(styleHtmlClasses(INSTALL_STYLE)).toEqual([STUDIO_HTML_CLASS])
  })

  it('returns a fresh array each call, so a caller cannot change the table', () => {
    const first = styleHtmlClasses('studio')
    first.push('tampered')
    expect(styleHtmlClasses('studio')).toEqual([STUDIO_HTML_CLASS])
  })

  it('toggles every class any style can emit, and only those', () => {
    const emitted = new Set(BOOTED_STYLES.flatMap((style) => styleHtmlClasses(style)))
    expect([...emitted].sort()).toEqual([...STYLE_HTML_CLASS_NAMES].sort())
    expect(STYLE_HTML_CLASS_NAMES).toEqual([STUDIO_HTML_CLASS])
  })

  it('renders Studio for Studio and an installed theme, never for Classic (PRD-260 D2)', () => {
    expect(rendersStudio('studio')).toBe(true)
    expect(rendersStudio(INSTALL_STYLE)).toBe(true)
    expect(rendersStudio('classic')).toBe(false)
  })

  it('leaves the appearance defaults as they were (Studio in the Dark tone)', () => {
    expect(DEFAULT_UI_STYLE).toBe('studio')
    expect(DEFAULT_TONE).toBe('dark')
    expect(APPEARANCE_DEFAULTS_KEY).toBe('automatos-appearance-defaults')
    expect(APPEARANCE_DEFAULTS_VERSION).toBe('2026-09-17-studio-dark')
  })
})

const htmlClasses = (): string[] => [...document.documentElement.classList]

describe('PRD-260 M07 — the style classes come from a <head> script, not from React', () => {
  afterEach(() => {
    document.documentElement.className = ''
    document.documentElement.removeAttribute('style')
  })

  it("adds Studio's class after the tone next-themes put there", () => {
    document.documentElement.className = 'dark'
    runThemeBoot(baseThemeBoot('studio'))
    expect(htmlClasses()).toEqual(['dark', STUDIO_HTML_CLASS])
  })

  it("puts on exactly each style's classes, removing the others and keeping everything else", () => {
    for (const style of BOOTED_STYLES) {
      document.documentElement.className = 'font-a light studio'
      runThemeBoot(baseThemeBoot(style))
      const expected = ['font-a', 'light', ...styleHtmlClasses(style)].sort()
      expect({ style, classes: htmlClasses().sort() }).toEqual({ style, classes: expected })
    }
  })

  it('has one constant script for every style, safe inside a <script> element', () => {
    expect(typeof THEME_BOOT_SCRIPT).toBe('string')
    expect(THEME_BOOT_SCRIPT).not.toContain('</')
    expect(THEME_BOOT_SCRIPT).not.toContain('<!--')
    expect(THEME_BOOT_SCRIPT).not.toMatch(/[\u2028\u2029]/)
    for (const style of BOOTED_STYLES) {
      expect({ style, classes: baseThemeBoot(style).addClasses }).toEqual({ style, classes: styleHtmlClasses(style) })
    }
  })
})

// lib/__tests__/prd261-no-theme-in-core.test.ts excepts this block by its title, written here as a literal.
describe('the values builtin Play left behind name nothing', () => {
  /** Builtin Play's style cookie: its bare name (before the axis took theme refs) and its theme ref. */
  const OLD_STYLE_COOKIES: readonly string[] = ['play', 'theme:builtin:play']
  /** Builtin Play's choice as the server stored it before A-19f3. */
  const OLD_SERVER_CHOICE = 'builtin:play'
  /** The cookie builtin Play kept its look's tone in. */
  const OLD_TONE_COOKIE = 'automatos-play-tone'

  it('reads its style cookie, either value, as the default style, which boots as Studio', () => {
    for (const raw of OLD_STYLE_COOKIES) {
      expect(isUiStyle(raw), raw).toBe(false)
      expect(parseUiStyle(raw), raw).toBe(DEFAULT_UI_STYLE)
      expect(baseThemeBoot(parseUiStyle(raw)).addClasses, raw).toEqual([STUDIO_HTML_CLASS])
    }
  })

  it('reads its server choice as one this build does not know: it moves no tab', () => {
    expect(styleOfChoice(OLD_SERVER_CHOICE)).toBeNull()
    for (const current of BOOTED_STYLES) expect(reconcileTarget(current, { choice: OLD_SERVER_CHOICE })).toBeNull()
  })

  it('never reads its tone cookie under Classic or Studio: the tone stays the user\'s', async () => {
    for (const value of ['light', 'dark']) {
      const resolve = createThemeBootResolver(cookieReader({ [OLD_TONE_COOKIE]: value }), async () => helloThemeBoot())
      for (const style of UI_STYLES) expect((await resolve(style)).tone, `${style} ${value}`).toEqual({ policy: 'user', value: null })
    }
  })

  it("never reads its tone cookie for an installed theme whose look sets the tone, Play's package included", async () => {
    const playPackageBoot = helloThemeBoot({ entry: { ...HELLO_ENTRY, themeId: PLAY_PACKAGE_ENTRY.themeId } })
    const cases: ReadonlyArray<readonly [string, ThemeBoot]> = [
      [themeToneCookieValue(HELLO_THEME_ID, 'dark'), helloThemeBoot()],
      ['dark', helloThemeBoot()],
      [themeToneCookieValue(PLAY_PACKAGE_ENTRY.themeId, 'dark'), playPackageBoot],
      ['dark', playPackageBoot],
    ]
    for (const [value, boot] of cases) {
      const old = createThemeBootResolver(cookieReader({ [OLD_TONE_COOKIE]: value }), async () => boot)
      expect((await old(INSTALL_STYLE)).tone, value).toEqual({ policy: 'from-look', value: null })
    }
    // Not vacuous: the same theme's own value under the runtime's cookie is its tone.
    const own = createThemeBootResolver(cookieReader({ [THEME_TONE_COOKIE]: themeToneCookieValue(PLAY_PACKAGE_ENTRY.themeId, 'dark') }), async () => playPackageBoot)
    expect((await own(INSTALL_STYLE)).tone).toEqual({ policy: 'from-look', value: 'dark' })
  })
})
