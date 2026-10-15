// @vitest-environment node
/** PRD-260 — validateLook, the one gate for a look from storage, a file or pasted JSON. */
import { describe, expect, it } from 'vitest'

import { PRESETS } from '@/lib/looks/presets'
import { validateLook } from '@/lib/looks/validate'
import {
  ANNOTATION_ANCHOR,
  FAMILY_EMOJI,
  inTagCharacters,
  RED_HEART,
  SOFT_HYPHEN,
  VS17,
  VS256,
} from '@/lib/text/__tests__/helpers'

const VALID = {
  schema: 'automatos-interface-look/v1',
  id: 'c1',
  name: 'My look 1',
  sub: 'Saved just now',
  interface: 'play',
  ground: 'paper',
  accent: 'rose',
  font: 'bric',
  shell: 'cobalt',
  light: 'amber',
  glow: 0,
}

function errorsFor(input: unknown): readonly string[] {
  const result = validateLook(input)
  return result.ok ? [] : result.errors
}

function rejects(input: unknown, fragment: string): void {
  const result = validateLook(input)
  expect(result.ok).toBe(false)
  expect(errorsFor(input).join('\n')).toContain(fragment)
}

/** A look as JSON.parse makes it: '__proto__' becomes an own key, not the prototype. */
function parsedWith(extra: string): unknown {
  return JSON.parse(`${JSON.stringify(VALID).slice(0, -1)},${extra}}`)
}

describe('validateLook', () => {
  it('accepts every preset as it is', () => {
    for (const preset of PRESETS) {
      expect(validateLook(preset)).toEqual({ ok: true, look: preset })
    }
  })

  it('accepts a JSON round trip and returns a fresh object', () => {
    const parsed: unknown = JSON.parse(JSON.stringify(VALID))
    const result = validateLook(parsed)
    expect(result).toEqual({ ok: true, look: VALID })
    expect(result.ok && result.look).not.toBe(parsed)
  })

  it('accepts glow at both ends and trims name and sub', () => {
    expect(validateLook({ ...VALID, glow: 40 }).ok).toBe(true)
    const result = validateLook({ ...VALID, name: '  Dusk  ', sub: ' ' })
    expect(result.ok && result.look).toMatchObject({ name: 'Dusk', sub: '' })
  })

  it("rejects anything that isn't an object", () => {
    for (const input of [null, undefined, 'ember', 3, [], [VALID]]) {
      expect(errorsFor(input)).toEqual(['A Look must be a JSON object.'])
    }
  })

  it('rejects unknown fields, including raw colours and CSS', () => {
    rejects({ ...VALID, bg: '#000000' }, `"bg" isn't part of a Look`)
    rejects({ ...VALID, css: 'body{}' }, `"css" isn't part of a Look`)
    rejects({ ...VALID, thumbnail: 'https://example.com/x.png' }, `"thumbnail" isn't part of a Look`)
  })

  it("rejects a '__proto__' field and never lets it reach the look", () => {
    const input = parsedWith('"__proto__":{"ground":"paper","polluted":true}')
    rejects(input, `"__proto__" isn't part of a Look`)
    expect(({} as Record<string, unknown>).polluted).toBeUndefined()
  })

  it('rejects a wrong or missing schema, and another interface', () => {
    rejects({ ...VALID, schema: 'automatos-interface-look/v0' }, '"schema" must be')
    const noSchema = Object.fromEntries(Object.entries(VALID).filter(([key]) => key !== 'schema'))
    rejects(noSchema, '"schema" must be')
    rejects({ ...VALID, interface: 'command' }, '"interface" must be "play"')
  })

  it('rejects values outside the catalogue', () => {
    rejects({ ...VALID, ground: 'midnight' }, `"ground" must be one of the catalogue's backgrounds: graphite, obsidian, indigo, paper.`)
    rejects({ ...VALID, accent: '#FF6A1F' }, '"accent" must be one of')
    rejects({ ...VALID, font: 'comic' }, '"font" must be one of')
    rejects({ ...VALID, light: { hex: '#FFFFFF' } }, '"light" must be one of')
  })

  it('counts own catalogue keys only', () => {
    rejects({ ...VALID, shell: 'toString' }, '"shell" must be one of')
    rejects({ ...VALID, ground: '__proto__' }, '"ground" must be one of')
    rejects({ ...VALID, accent: 'hasOwnProperty' }, '"accent" must be one of')
    rejects({ ...VALID, font: 'constructor' }, '"font" must be one of')
  })

  it('rejects a glow outside 0 to 40 or not a whole number', () => {
    for (const glow of [-1, 41, 12.5, '20', Number.NaN, Number.POSITIVE_INFINITY, null]) {
      rejects({ ...VALID, glow }, '"glow" must be a whole number from 0 to 40.')
    }
  })

  it('rejects a name or sub over 40 characters, but counts characters, not UTF-16 units', () => {
    rejects({ ...VALID, name: 'x'.repeat(41) }, '"name" must be at most 40 characters.')
    rejects({ ...VALID, sub: 'x'.repeat(41) }, '"sub" must be at most 40 characters.')
    expect(validateLook({ ...VALID, name: '🔥'.repeat(40) }).ok).toBe(true)
    rejects({ ...VALID, name: '🔥'.repeat(41) }, '"name" must be at most 40 characters.')
  })

  it("rejects a name that isn't one line of text, or is empty", () => {
    rejects({ ...VALID, name: 42 }, '"name" must be text.')
    rejects({ ...VALID, name: '   ' }, `"name" can't be empty.`)
    rejects({ ...VALID, sub: 'two\nlines' }, '"sub" must be one line of plain text.')
    rejects({ ...VALID, name: 'a\u0085b' }, '"name" must be one line of plain text.')
  })

  it('rejects bidi overrides, zero-width spaces and separators, but keeps emoji sequences', () => {
    for (const name of ['Ember\u202Ecopy', 'a\u2066b\u2069', 'a\u200Bb', 'a\u2028b', 'a\u200Eb']) {
      rejects({ ...VALID, name }, '"name" must be one line of plain text.')
    }
    rejects({ ...VALID, sub: 'x\u202Dy' }, '"sub" must be one line of plain text.')
    expect(validateLook({ ...VALID, name: '🧑\u200D🎨 Studio' }).ok).toBe(true)
    expect(validateLook({ ...VALID, name: 'می\u200Cخواهم' }).ok).toBe(true)
  })

  // P260-FP-1: the rest of the format characters, private use and variation selectors 17 to 256 hide words too.
  it('rejects tag characters, soft hyphens, interlinear annotations, private use and variation selectors 17 to 256', () => {
    const hidden = [`Ember${inTagCharacters('send it')}`, `Em${SOFT_HYPHEN}ber`, `a${ANNOTATION_ANCHOR}b`, `a${VS17}b`, 'a\uE000b']
    for (const name of hidden) {
      rejects({ ...VALID, name }, '"name" must be one line of plain text.')
    }
    rejects({ ...VALID, sub: `Dusk${VS256}` }, '"sub" must be one line of plain text.')
    rejects({ ...VALID, name: inTagCharacters('send it') }, `"name" can't be empty.`)
  })

  it('keeps an emoji ZWJ sequence and U+FE0F in a name unchanged', () => {
    for (const name of [`${FAMILY_EMOJI} Home`, `Dusk ${RED_HEART}`]) {
      const result = validateLook({ ...VALID, name })
      expect(result.ok && result.look.name).toBe(name)
    }
  })

  it('rejects a name that is blank to the eye', () => {
    for (const name of ['\u3164\u3164', '\u200B', '\u202E', '\u00A0\u2003']) {
      rejects({ ...VALID, name }, `"name" can't be empty.`)
    }
  })

  it('rejects a bad id', () => {
    for (const id of ['', 'has space', '-dash-first', 'x'.repeat(65), 7, '../etc']) {
      rejects({ ...VALID, id }, '"id" must be')
    }
  })

  it('reports every problem at once and never echoes a long key back', () => {
    const longKey = 'k'.repeat(500)
    const errors = errorsFor({ ...VALID, accent: 'neon', glow: 99, [longKey]: true })
    expect(errors).toHaveLength(3)
    expect(errors.every((error) => error.length < 200)).toBe(true)
  })

  it('names at most five unknown fields and counts the rest, so a huge file gives a short list', () => {
    const junk = Object.fromEntries(Array.from({ length: 1000 }, (_, i) => [`junk${i}`, i]))
    const errors = errorsFor({ ...VALID, ...junk })
    expect(errors).toHaveLength(6)
    expect(errors[0]).toBe(`"junk0" isn't part of a Look. Remove it.`)
    expect(errors[5]).toBe("…and 995 more fields that aren't part of a Look.")
    const six = Object.fromEntries(Array.from({ length: 6 }, (_, i) => [`x${i}`, i]))
    expect(errorsFor({ ...VALID, ...six })[5]).toBe("…and 1 more field that isn't part of a Look.")
  })

  it('never quotes an unsafe character back', () => {
    expect(errorsFor({ ...VALID, 'a\u202Eb\nc': 1 })).toEqual([`"a?b?c" isn't part of a Look. Remove it.`])
    expect(errorsFor({ ...VALID, [`a${SOFT_HYPHEN}b${VS17}`]: 1 })).toEqual([`"a?b?" isn't part of a Look. Remove it.`])
  })
})
