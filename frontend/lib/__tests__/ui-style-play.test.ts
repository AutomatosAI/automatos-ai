/**
 * PRD-260 D1/D2/D11 — the Style axis gains Play: the cookie may say 'play'
 * (anything else still falls back to the default), Play renders as
 * <html class="studio play">, and the appearance defaults are untouched
 * (Play is opt-in). M07: the classes reach <html> through a <head> script,
 * which leaves every other class (the tone, the font variables) alone.
 *
 * PRD-261 A-02a: Play is now builtin Play on the axis, `theme:builtin:play`;
 * the legacy 'play' cookie parses to it, and its <html> classes are unchanged.
 * PRD-261 A-03: that script is the one constant THEME_BOOT_SCRIPT, applying
 * each style's boot; the per-style STYLE_CLASS_SCRIPT is gone, so the same
 * assertions now run the boot of every style, an installed theme's included.
 */
import { afterEach, describe, it, expect } from 'vitest'
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
import {
  BUILTIN_PLAY_HTML_CLASS as PLAY_HTML_CLASS,
  BUILTIN_PLAY_STYLE as PLAY,
} from '@/lib/theme-host/builtin/play-entry'
import { THEME_BOOT_SCRIPT, builtinThemeBoot } from '@/lib/theme-host/boot'
import { cookieReader, runThemeBoot } from '@/lib/theme-host/__tests__/boot-harness'

describe('PRD-260 — the Style axis with Play', () => {
  it('lists Classic, Studio and Play in that order', () => {
    expect(UI_STYLES).toEqual(['classic', 'studio', 'theme:builtin:play'])
  })

  it('accepts play from the cookie, legacy and current, as builtin Play', () => {
    expect(isUiStyle(PLAY)).toBe(true)
    expect(parseUiStyle('play')).toBe(PLAY)
    expect(parseUiStyle(PLAY)).toBe(PLAY)
    expect(parseUiStyle('studio')).toBe('studio')
    expect(parseUiStyle('classic')).toBe('classic')
  })

  it('falls back to the default for a tampered or missing cookie', () => {
    for (const raw of ['PLAY', ' play', 'play ', 'play;studio', 'studio play', '<script>', 'dark', '', 'theme:builtin:play ', 'theme:install:x']) {
      expect(parseUiStyle(raw), JSON.stringify(raw)).toBe(DEFAULT_UI_STYLE)
    }
    expect(parseUiStyle(null)).toBe(DEFAULT_UI_STYLE)
    expect(parseUiStyle(undefined)).toBe(DEFAULT_UI_STYLE)
    for (const raw of [1, {}, ['play'], null, undefined, true]) expect(isUiStyle(raw)).toBe(false)
  })

  it('maps each style to its <html> classes', () => {
    expect(PLAY_HTML_CLASS).toBe('play')
    expect(STUDIO_HTML_CLASS).toBe('studio')
    expect(styleHtmlClasses('classic')).toEqual([])
    expect(styleHtmlClasses('studio')).toEqual([STUDIO_HTML_CLASS])
    expect(styleHtmlClasses(PLAY)).toEqual([STUDIO_HTML_CLASS, PLAY_HTML_CLASS])
  })

  it('returns a fresh array each call, so a caller cannot change the table', () => {
    const first = styleHtmlClasses(PLAY)
    first.push('tampered')
    expect(styleHtmlClasses(PLAY)).toEqual(['studio', 'play'])
  })

  it('toggles every class any style can emit', () => {
    const emitted = new Set(UI_STYLES.flatMap((s) => styleHtmlClasses(s)))
    expect([...emitted].sort()).toEqual([...STYLE_HTML_CLASS_NAMES].sort())
  })

  it('renders Studio for Studio and Play, never for Classic (D2)', () => {
    expect(rendersStudio('studio')).toBe(true)
    expect(rendersStudio(PLAY)).toBe(true)
    expect(rendersStudio('classic')).toBe(false)
  })

  it('leaves the appearance defaults as they were: Play is opt-in (D11)', () => {
    expect(DEFAULT_UI_STYLE).toBe('studio')
    expect(DEFAULT_TONE).toBe('dark')
    expect(APPEARANCE_DEFAULTS_KEY).toBe('automatos-appearance-defaults')
    expect(APPEARANCE_DEFAULTS_VERSION).toBe('2026-09-17-studio-dark')
  })
})

/** An installed theme's style: its classes are its base's (Studio) until A-08c's boot answers. */
const INSTALL_STYLE: UiStyle = 'theme:install:0b7e4c1a-9f3d-4c2e-8a51-3d2f6b7a9c10'
const BOOTED_STYLES: readonly UiStyle[] = [...UI_STYLES, INSTALL_STYLE]

// Runs a style's <head> boot the way the browser does: its JSON element, then the constant
// script as a classic script in the page. Only in tests: the app itself never evaluates strings.
function runStyleScript(style: UiStyle): void {
  runThemeBoot(builtinThemeBoot(style, cookieReader()))
}

const htmlClasses = () => [...document.documentElement.classList]

describe('PRD-260 M07 — the style classes come from a <head> script, not from React', () => {
  afterEach(() => {
    document.documentElement.className = ''
    document.documentElement.removeAttribute('style')
  })

  it("adds Play's classes after the tone next-themes put there", () => {
    document.documentElement.className = 'dark'
    runStyleScript(PLAY)
    expect(htmlClasses()).toEqual(['dark', STUDIO_HTML_CLASS, PLAY_HTML_CLASS])
  })

  it("puts on exactly each style's classes, removing the others and keeping everything else", () => {
    for (const style of BOOTED_STYLES) {
      document.documentElement.className = 'font-a light studio play'
      runStyleScript(style)
      const kept = ['font-a', 'light']
      const expected = [...kept, ...styleHtmlClasses(style)].sort()
      expect({ style, classes: htmlClasses().sort() }).toEqual({ style, classes: expected })
    }
  })

  it('has one constant script for every style, safe inside a <script> element', () => {
    expect(typeof THEME_BOOT_SCRIPT).toBe('string')
    expect(THEME_BOOT_SCRIPT).not.toContain('</')
    expect(THEME_BOOT_SCRIPT).not.toContain('<!--')
    expect(THEME_BOOT_SCRIPT).not.toMatch(/[\u2028\u2029]/)
    for (const style of BOOTED_STYLES) {
      expect({ style, classes: builtinThemeBoot(style, cookieReader()).addClasses }).toEqual({ style, classes: styleHtmlClasses(style) })
    }
  })
})
