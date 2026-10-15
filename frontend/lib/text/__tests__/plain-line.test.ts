/**
 * PRD-260 P260-FX-10 (M56) — one line of plain text (lib/text/plain-line.ts):
 * the gate a Look's name passes (lib/looks/validate.ts) and the fold a link's
 * ?draft= goes through (lib/play/draft.ts). Every special character is written
 * as an escape, so none of them hides in this file.
 */
import { describe, expect, it } from 'vitest'

import { oneLine, UNSAFE_TEXT } from '@/lib/text/plain-line'

/** A woman technologist: two emoji joined by a ZWJ. */
const JOINED_EMOJI = '\u{1F469}\u200D\u{1F4BB}'

describe('UNSAFE_TEXT', () => {
  it('finds controls, line and paragraph separators, bidi overrides and isolates, zero-width spaces and the BOM', () => {
    for (const char of ['\n', '\r', '\u0085', '\u2028', '\u2029', '\u202E', '\u2066', '\u061C', '\u200B', '\u200E', '\uFEFF']) {
      expect(UNSAFE_TEXT.test(`a${char}b`)).toBe(true)
    }
  })

  it('leaves plain words, ZWJ emoji sequences and ZWNJ alone', () => {
    for (const text of ['Ember copy', `${JOINED_EMOJI} at work`, 'a\u200Cb', 'Caf\u00E9 \u2014 50%']) {
      expect(UNSAFE_TEXT.test(text)).toBe(false)
    }
  })
})

describe('oneLine', () => {
  it('folds each run of white space and unsafe characters into one space', () => {
    expect(oneLine('Do X\n\n\nAbout "Q3": ')).toBe('Do X About "Q3":')
    expect(oneLine('a \u202E\t b')).toBe('a b')
    expect(oneLine('a\u00A0\u00A0b')).toBe('a b')
  })

  it('trims the ends, so text of nothing but spaces and unsafe characters is empty', () => {
    expect(oneLine('\u202Eabc')).toBe('abc')
    expect(oneLine('  \n\u200B\uFEFF ')).toBe('')
  })

  it('keeps an emoji sequence whole', () => {
    expect(oneLine(`${JOINED_EMOJI}\nat work`)).toBe(`${JOINED_EMOJI} at work`)
  })
})
