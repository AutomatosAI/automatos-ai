/**
 * PRD-260 D1/D2/D11 — the Style axis gains Play: the cookie may say 'play'
 * (anything else still falls back to the default), Play renders as
 * <html class="studio play">, and the appearance defaults are untouched
 * (Play is opt-in). M07: the classes reach <html> through STYLE_CLASS_SCRIPT,
 * which leaves every other class (the tone, the font variables) alone.
 */
import { afterEach, describe, it, expect } from 'vitest'
import {
  APPEARANCE_DEFAULTS_KEY,
  APPEARANCE_DEFAULTS_VERSION,
  DEFAULT_TONE,
  DEFAULT_UI_STYLE,
  PLAY_HTML_CLASS,
  STUDIO_HTML_CLASS,
  STYLE_CLASS_SCRIPT,
  STYLE_HTML_CLASS_NAMES,
  UI_STYLES,
  isUiStyle,
  parseUiStyle,
  rendersStudio,
  styleHtmlClasses,
  type UiStyle,
} from '@/lib/ui-style'

describe('PRD-260 — the Style axis with Play', () => {
  it('lists Classic, Studio and Play in that order', () => {
    expect(UI_STYLES).toEqual(['classic', 'studio', 'play'])
  })

  it('accepts play from the cookie', () => {
    expect(isUiStyle('play')).toBe(true)
    expect(parseUiStyle('play')).toBe('play')
    expect(parseUiStyle('studio')).toBe('studio')
    expect(parseUiStyle('classic')).toBe('classic')
  })

  it('falls back to the default for a tampered or missing cookie', () => {
    for (const raw of ['PLAY', ' play', 'play ', 'play;studio', 'studio play', '<script>', 'dark', '']) {
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
    expect(styleHtmlClasses('play')).toEqual([STUDIO_HTML_CLASS, PLAY_HTML_CLASS])
  })

  it('returns a fresh array each call, so a caller cannot change the table', () => {
    const first = styleHtmlClasses('play')
    first.push('tampered')
    expect(styleHtmlClasses('play')).toEqual(['studio', 'play'])
  })

  it('toggles every class any style can emit', () => {
    const emitted = new Set(UI_STYLES.flatMap((s) => styleHtmlClasses(s)))
    expect([...emitted].sort()).toEqual([...STYLE_HTML_CLASS_NAMES].sort())
  })

  it('renders Studio for Studio and Play, never for Classic (D2)', () => {
    expect(rendersStudio('studio')).toBe(true)
    expect(rendersStudio('play')).toBe(true)
    expect(rendersStudio('classic')).toBe(false)
  })

  it('leaves the appearance defaults as they were: Play is opt-in (D11)', () => {
    expect(DEFAULT_UI_STYLE).toBe('studio')
    expect(DEFAULT_TONE).toBe('dark')
    expect(APPEARANCE_DEFAULTS_KEY).toBe('automatos-appearance-defaults')
    expect(APPEARANCE_DEFAULTS_VERSION).toBe('2026-09-17-studio-dark')
  })
})

// Runs a style's <head> script the way the browser does, as a classic script in the page. Only
// in tests: the app itself never evaluates strings.
function runStyleScript(style: UiStyle): void {
  new Function(STYLE_CLASS_SCRIPT[style])()
}

const htmlClasses = () => [...document.documentElement.classList]

describe('PRD-260 M07 — the style classes come from a <head> script, not from React', () => {
  afterEach(() => {
    document.documentElement.className = ''
  })

  it("adds Play's classes after the tone next-themes put there", () => {
    document.documentElement.className = 'dark'
    runStyleScript('play')
    expect(htmlClasses()).toEqual(['dark', STUDIO_HTML_CLASS, PLAY_HTML_CLASS])
  })

  it("puts on exactly each style's classes, removing the others and keeping everything else", () => {
    for (const style of UI_STYLES) {
      document.documentElement.className = 'font-a light studio play'
      runStyleScript(style)
      const kept = ['font-a', 'light']
      const expected = [...kept, ...styleHtmlClasses(style)].sort()
      expect({ style, classes: htmlClasses().sort() }).toEqual({ style, classes: expected })
    }
  })

  it('has one constant per style, safe inside a <script> element', () => {
    expect(Object.keys(STYLE_CLASS_SCRIPT).sort()).toEqual([...UI_STYLES].sort())
    for (const style of UI_STYLES) {
      expect(STYLE_CLASS_SCRIPT[style], style).not.toMatch(/[<\u2028\u2029]/)
    }
  })
})
