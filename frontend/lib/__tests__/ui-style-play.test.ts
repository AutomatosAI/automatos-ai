/**
 * PRD-260 D1/D2/D11 — the Style axis gains Play: the cookie may say 'play'
 * (anything else still falls back to the default), Play renders as
 * <html class="studio play">, and the appearance defaults are untouched
 * (Play is opt-in).
 */
import { describe, it, expect } from 'vitest'
import {
  APPEARANCE_DEFAULTS_KEY,
  APPEARANCE_DEFAULTS_VERSION,
  DEFAULT_TONE,
  DEFAULT_UI_STYLE,
  PLAY_HTML_CLASS,
  STUDIO_HTML_CLASS,
  STYLE_HTML_CLASS_NAMES,
  UI_STYLES,
  isUiStyle,
  parseUiStyle,
  rendersStudio,
  styleHtmlClasses,
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
