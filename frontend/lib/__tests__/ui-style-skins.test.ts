/**
 * PRD-259 S-004b — the style cookie also takes `skin:<id>`, and the skin's base rides in its
 * own cookie (decision K1). Both are parsed strictly: a tampered value is the default, never
 * an attribute or a class the server writes.
 */
import { describe, expect, it } from 'vitest'

import {
  DEFAULT_SKIN_BASE,
  DEFAULT_UI_STYLE,
  UI_SKIN_BASE_COOKIE,
  UI_SKIN_TONE_COOKIE,
  baseOf,
  htmlAttrsFor,
  isUiStyle,
  lockedToneFor,
  parseSkinBase,
  parseSkinTone,
  parseUiStyle,
  skinIdOf,
  skinStyle,
} from '@/lib/ui-style'

const ID = '0f8c2b6e-4d1a-4c3e-9b7a-2e5d6f8a1c90'
const TAMPERED = [
  'skin:"><script>alert(1)</script>',
  'skin:../x',
  'skin:',
  'skin:0f8c2b6e-4d1a-4c3e-9b7a-2e5d6f8a1c90"',
  'skin:0f8c2b6e-4d1a-4c3e-9b7a-2e5d6f8a1c90 x',
  'SKIN:0f8c2b6e-4d1a-4c3e-9b7a-2e5d6f8a1c90',
  'studio ',
  'matte',
]

describe('the style cookie with skins', () => {
  it('names the base cookie automatos-skin-base, defaulting to Studio', () => {
    expect(UI_SKIN_BASE_COOKIE).toBe('automatos-skin-base')
    expect(DEFAULT_SKIN_BASE).toBe('studio')
  })

  it('accepts classic, studio and skin:<uuid>', () => {
    expect(parseUiStyle('classic')).toBe('classic')
    expect(parseUiStyle('studio')).toBe('studio')
    expect(parseUiStyle(`skin:${ID}`)).toBe(`skin:${ID}`)
    expect(skinStyle(ID)).toBe(`skin:${ID}`)
  })

  it('turns a tampered or missing cookie into the default', () => {
    for (const value of [...TAMPERED, undefined, null, '']) {
      expect(parseUiStyle(value), String(value)).toBe(DEFAULT_UI_STYLE)
      expect(isUiStyle(value), String(value)).toBe(false)
    }
  })

  it('parses the base cookie strictly', () => {
    expect(parseSkinBase('classic')).toBe('classic')
    expect(parseSkinBase('studio')).toBe('studio')
    for (const value of ['Classic', 'skin', '"><b>', undefined, null, '']) expect(parseSkinBase(value)).toBe('studio')
  })

  it('reads the skin id and the base of a style', () => {
    expect(skinIdOf(`skin:${ID}`)).toBe(ID)
    expect(skinIdOf('studio')).toBeNull()
    expect(baseOf(`skin:${ID}`, 'classic')).toBe('classic')
    expect(baseOf(`skin:${ID}`, undefined)).toBe('studio')
    // The base cookie means nothing unless a skin is chosen.
    expect(baseOf('classic', 'studio')).toBe('classic')
    expect(baseOf('studio', 'classic')).toBe('studio')
  })
})

describe('htmlAttrsFor — what the server puts on <html>', () => {
  it('a Studio-based skin: the studio class and data-skin', () => {
    expect(htmlAttrsFor(`skin:${ID}`, 'studio')).toEqual({ style: `skin:${ID}`, base: 'studio', studioClass: true, skinId: ID })
  })

  it('a Classic-based skin: no studio class, but data-skin', () => {
    expect(htmlAttrsFor(`skin:${ID}`, 'classic')).toEqual({ style: `skin:${ID}`, base: 'classic', studioClass: false, skinId: ID })
  })

  it('a skin with no base cookie sits on Studio', () => {
    expect(htmlAttrsFor(`skin:${ID}`, undefined)).toMatchObject({ studioClass: true, skinId: ID })
  })

  it('Classic and Studio as before, with no data-skin', () => {
    expect(htmlAttrsFor('classic', 'studio')).toEqual({ style: 'classic', base: 'classic', studioClass: false, skinId: null })
    expect(htmlAttrsFor('studio', undefined)).toEqual({ style: 'studio', base: 'studio', studioClass: true, skinId: null })
  })

  it('a tampered cookie: the default style, no data-skin', () => {
    for (const value of TAMPERED) {
      expect(htmlAttrsFor(value, 'classic'), value).toEqual({ style: DEFAULT_UI_STYLE, base: DEFAULT_UI_STYLE, studioClass: true, skinId: null })
    }
  })
})

describe('the tone cookie — a single-tone skin\'s lock, read by the server', () => {
  const TAMPERED_TONES = ['dark;x', '"><script>alert(1)</script>', 'Dark', 'system', ' dark', 'light ', '']

  it('names the tone cookie automatos-skin-tone', () => {
    expect(UI_SKIN_TONE_COOKIE).toBe('automatos-skin-tone')
  })

  it('parses light and dark, and nothing else', () => {
    expect(parseSkinTone('light')).toBe('light')
    expect(parseSkinTone('dark')).toBe('dark')
    for (const value of [...TAMPERED_TONES, undefined, null]) expect(parseSkinTone(value), String(value)).toBeUndefined()
  })

  it('a valid skin style with a valid tone cookie locks that tone', () => {
    expect(lockedToneFor(`skin:${ID}`, 'dark')).toBe('dark')
    expect(lockedToneFor(`skin:${ID}`, 'light')).toBe('light')
  })

  it('a tampered or missing tone cookie locks nothing', () => {
    for (const value of [...TAMPERED_TONES, undefined, null]) expect(lockedToneFor(`skin:${ID}`, value), String(value)).toBeUndefined()
  })

  it('a tone cookie means nothing unless the style is a valid skin', () => {
    for (const style of ['classic', 'studio', undefined, null, ...TAMPERED]) {
      expect(lockedToneFor(style, 'dark'), String(style)).toBeUndefined()
    }
  })
})
