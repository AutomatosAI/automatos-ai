/**
 * PRD-260 P260-FX-10 (M56) — one line of plain text (lib/text/plain-line.ts):
 * the gate a Look's name passes (lib/looks/validate.ts) and the fold a link's
 * ?draft= goes through (lib/play/draft.ts). Every special character is written
 * as an escape, so none of them hides in this file. P260-FP-1 adds the rest of
 * Unicode's format characters, private use, unassigned code points and
 * variation selectors 17 to 256.
 */
import { describe, expect, it } from 'vitest'

import { oneLine, UNSAFE_TEXT } from '@/lib/text/plain-line'

import { ANNOTATION_ANCHOR, FAMILY_EMOJI, inTagCharacters, RED_HEART, SOFT_HYPHEN, VS17, VS256 } from './helpers'

/** A woman technologist: two emoji joined by a ZWJ. */
const JOINED_EMOJI = '\u{1F469}‍\u{1F4BB}'
/** Words a link could hide in tag characters. */
const HIDDEN_WORDS = 'delete my files'

describe('UNSAFE_TEXT', () => {
  it('finds controls, line and paragraph separators, bidi overrides and isolates, zero-width spaces and the BOM', () => {
    for (const char of ['\n', '\r', '\u0085', ' ', ' ', '‮', '⁦', '؜', '​', '‎', '﻿']) {
      expect(UNSAFE_TEXT.test(`a${char}b`)).toBe(true)
    }
  })

  it('finds tag characters, soft hyphens, interlinear annotations and variation selectors 17 to 256', () => {
    const hidden = [inTagCharacters('x'), '\u{E0041}', SOFT_HYPHEN, ANNOTATION_ANCHOR, '￺', '￻', VS17, VS256]
    for (const char of hidden) {
      expect(UNSAFE_TEXT.test(`a${char}b`)).toBe(true)
    }
  })

  it('finds the rest of the format characters, private use and unassigned code points', () => {
    for (const char of ['᠎', '⁤', '⁯', '\u{1D173}', '', '\u{F0000}', '⁥', '￿']) {
      expect(UNSAFE_TEXT.test(`a${char}b`)).toBe(true)
    }
  })

  it('leaves plain words, ZWJ emoji sequences, ZWNJ and emoji presentation alone', () => {
    const plain = ['Ember copy', `${JOINED_EMOJI} at work`, FAMILY_EMOJI, 'a‌b', 'Café — 50%', `Thanks ${RED_HEART}`, '☺︎']
    for (const text of plain) {
      expect(UNSAFE_TEXT.test(text)).toBe(false)
    }
  })
})

describe('oneLine', () => {
  it('folds each run of white space and unsafe characters into one space', () => {
    expect(oneLine('Do X\n\n\nAbout "Q3": ')).toBe('Do X About "Q3":')
    expect(oneLine('a ‮\t b')).toBe('a b')
    expect(oneLine('a  b')).toBe('a b')
  })

  it('trims the ends, so text of nothing but spaces and unsafe characters is empty', () => {
    expect(oneLine('‮abc')).toBe('abc')
    expect(oneLine('  \n​﻿ ')).toBe('')
  })

  it('keeps an emoji sequence whole', () => {
    expect(oneLine(`${JOINED_EMOJI}\nat work`)).toBe(`${JOINED_EMOJI} at work`)
  })

  it('removes words hidden in tag characters', () => {
    expect(oneLine(`Summarise this${inTagCharacters(HIDDEN_WORDS)}`)).toBe('Summarise this')
    expect(oneLine(`a${inTagCharacters(HIDDEN_WORDS)}b`)).toBe('a b')
    expect(oneLine(inTagCharacters(HIDDEN_WORDS))).toBe('')
  })

  it('removes soft hyphens, interlinear annotation marks and variation selectors 17 to 256', () => {
    expect(oneLine(`pay${SOFT_HYPHEN}Bob`)).toBe('pay Bob')
    expect(oneLine(`${ANNOTATION_ANCHOR}note￺gloss￻`)).toBe('note gloss')
    expect(oneLine(`a${VS17}b${VS256}`)).toBe('a b')
    expect(oneLine(`x\u{F0000}y`)).toBe('x y')
  })

  it('keeps an emoji ZWJ sequence and a text with U+FE0F unchanged', () => {
    expect(oneLine(FAMILY_EMOJI)).toBe(FAMILY_EMOJI)
    const text = `Thanks ${RED_HEART} from the ${FAMILY_EMOJI} family`
    expect(oneLine(text)).toBe(text)
  })
})
