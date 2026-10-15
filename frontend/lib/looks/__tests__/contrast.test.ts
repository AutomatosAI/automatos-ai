// @vitest-environment node
/**
 * PRD-260 — every look the catalogue can make reads at WCAG AA: 4 grounds × 6
 * accents = 24 pairs, for Play's own variables and for Studio's tokens
 * (measured on the triplets converted back to sRGB, as the browser draws them).
 */
import { describe, expect, it } from 'vitest'

import { ACCENTS, ACCENT_KEYS, ERROR_COLOUR, GROUNDS, GROUND_KEYS } from '@/lib/looks/catalogue'
import {
  AA_NON_TEXT,
  AA_TEXT,
  READABLE_TARGET,
  contrastRatio,
  mixHex,
  readableOn,
  readableOnAll,
  readableOnTinted,
  readsOnAll,
  relativeLuminance,
} from '@/lib/looks/contrast'
import { lookToCssVars } from '@/lib/looks/css'
import { tripletToHex } from '@/lib/looks/hsl'
import { EMBER } from '@/lib/looks/presets'
import { PRIMARY_TINT_MAX, lookToStudioTokens, textFills } from '@/lib/looks/studio-tokens'

const PAIRS = GROUND_KEYS.flatMap((ground) => ACCENT_KEYS.map((accent) => ({ ground, accent })))
const CHANNEL_TOLERANCE = 4
const WRAPPED = /^hsl\((.+)\)$/

function channels(hex: string): readonly number[] {
  return [1, 3, 5].map((at) => parseInt(hex.slice(at, at + 2), 16))
}

function near(actual: string, expected: string, tolerance: number): boolean {
  const a = channels(actual)
  return channels(expected).every((value, at) => Math.abs(value - (a[at] ?? -999)) <= tolerance)
}

function expectRatio(label: string, foreground: string, background: string, minimum: number): void {
  const ratio = contrastRatio(foreground, background)
  const pair = `${label} (${ratio.toFixed(2)}:1)`
  expect({ pair, passes: ratio >= minimum }).toEqual({ pair, passes: true })
}

function expectAa(label: string, foreground: string, background: string): void {
  expectRatio(label, foreground, background, AA_TEXT)
}

/** A Studio token as the browser draws it: the triplet (or the wrapped alias) back to #RRGGBB. */
function drawn(tokens: Readonly<Record<string, string>>, name: string): string {
  const value = tokens[name] ?? ''
  return tripletToHex(WRAPPED.exec(value)?.[1] ?? value)
}

describe('WCAG helpers', () => {
  it('compute luminance and contrast', () => {
    expect(relativeLuminance('#000000')).toBe(0)
    expect(relativeLuminance('#FFFFFF')).toBeCloseTo(1, 10)
    expect(contrastRatio('#000000', '#ffffff')).toBeCloseTo(21, 5)
    expect(contrastRatio('#777777', '#FFFFFF')).toBeCloseTo(4.48, 2)
    expect(contrastRatio('#FFFFFF', '#000000')).toBe(contrastRatio('#000000', '#FFFFFF'))
  })

  it('refuse anything but #rrggbb', () => {
    expect(() => relativeLuminance('rgba(0,0,0,.5)')).toThrow("isn't a #rrggbb colour")
    expect(() => relativeLuminance('#FFF')).toThrow()
  })

  it('mix in sRGB and round', () => {
    expect(mixHex('#000000', '#FFFFFF', 0)).toBe('#000000')
    expect(mixHex('#000000', '#FFFFFF', 1)).toBe('#FFFFFF')
    expect(mixHex('#000000', '#FFFFFF', 0.5)).toBe('#808080')
    expect(mixHex('#ff6a1f', '#ff6a1f', 0)).toBe('#FF6A1F')
  })

  it('match the ratios measured on 14 Oct', () => {
    const { graphite, paper } = GROUNDS
    expect(contrastRatio(graphite.text, graphite.bg)).toBeCloseTo(16.36, 1)
    expect(contrastRatio(graphite.text2, graphite.bg)).toBeCloseTo(8.54, 1)
    expect(contrastRatio(ACCENTS.molten.ink, ACCENTS.molten.hex)).toBeCloseTo(6.74, 1)
    expect(contrastRatio(paper.text2, paper.bg)).toBeCloseTo(5.12, 1)
    // Why readableOn exists: raw accent and error as text fail on Paper.
    expect(contrastRatio(ACCENTS.molten.hex, paper.bg)).toBeCloseTo(2.49, 1)
    expect(contrastRatio(ERROR_COLOUR, paper.bg)).toBeCloseTo(2.41, 1)
  })
})

describe('readableOn', () => {
  it('keeps a colour that already reads on both grounds', () => {
    const { graphite } = GROUNDS
    expect(readableOn(ACCENTS.molten.hex, graphite.bg, graphite.surface, graphite.text)).toBe('#FF6A1F')
    expect(readableOn(ERROR_COLOUR, graphite.bg, graphite.surface, graphite.text)).toBe(ERROR_COLOUR)
  })

  it("darkens molten and error coral for Paper, toward the PRD's values", () => {
    const { paper } = GROUNDS
    const molten = readableOn(ACCENTS.molten.hex, paper.bg, paper.surface, paper.text)
    const error = readableOn(ERROR_COLOUR, paper.bg, paper.surface, paper.text)
    expect({ molten, near: near(molten, '#A84A1B', CHANNEL_TOLERANCE) }).toEqual({ molten, near: true })
    expect({ error, near: near(error, '#A84A49', CHANNEL_TOLERANCE) }).toEqual({ error, near: true })
    for (const colour of [molten, error]) {
      expect(contrastRatio(colour, paper.bg)).toBeGreaterThanOrEqual(READABLE_TARGET)
      expect(contrastRatio(colour, paper.surface)).toBeGreaterThanOrEqual(READABLE_TARGET)
    }
  })

  it("falls back to the ground's text when no mix reads", () => {
    expect(readableOn('#777777', '#777777', '#777777', '#787878')).toBe('#787878')
  })
})

describe('readableOnAll and readableOnTinted', () => {
  it('readableOn is readableOnAll on the bg and the surface', () => {
    for (const { ground, accent } of PAIRS) {
      const { bg, surface, text } = GROUNDS[ground]
      const { hex } = ACCENTS[accent]
      expect(readableOn(hex, bg, surface, text)).toBe(readableOnAll(hex, [bg, surface], text))
    }
  })

  it("reads on every fill it is given, Studio's chips included", () => {
    for (const { ground, accent } of PAIRS) {
      const fills = textFills(ground)
      const colour = readableOnAll(ACCENTS[accent].hex, fills, GROUNDS[ground].text)
      expect({ ground, accent, reads: readsOnAll(colour, fills) }).toEqual({ ground, accent, reads: true })
    }
  })

  it('also reads on each fill tinted by the result itself, where the flat answer does not', () => {
    const { text } = GROUNDS.graphite
    const fills = textFills('graphite')
    const flat = readableOnAll(ACCENTS.molten.hex, fills, text)
    const tinted = readableOnTinted(ACCENTS.molten.hex, fills, text, PRIMARY_TINT_MAX)
    const tintedBy = (colour: string) => fills.map((fill) => mixHex(fill, colour, PRIMARY_TINT_MAX))
    expect(flat).toBe('#FF6A1F')
    expect(readsOnAll(flat, tintedBy(flat))).toBe(false)
    expect(tinted).toBe('#FB9F6F')
    expect(readsOnAll(tinted, [...fills, ...tintedBy(tinted)])).toBe(true)
  })

  it('falls back to the text when no mix reads', () => {
    expect(readableOnAll('#777777', ['#777777'], '#787878')).toBe('#787878')
    expect(readableOnTinted('#777777', ['#777777'], '#787878', PRIMARY_TINT_MAX)).toBe('#787878')
  })
})

describe("every ground × accent pair reads at AA — Play's variables", () => {
  it('covers 24 pairs', () => {
    expect(PAIRS).toHaveLength(24)
  })

  it.each(PAIRS)('$ground with $accent', ({ ground, accent }) => {
    const g = GROUNDS[ground]
    const a = ACCENTS[accent]
    const vars = lookToCssVars({ ...EMBER, ground, accent })
    const accentText = vars['--pl-accent-text'] ?? ''
    const errorText = vars['--pl-error-text'] ?? ''
    const sheet = vars['--pl-sheet-bg'] ?? ''
    expectAa('text on bg', g.text, g.bg)
    expectAa('text on surface', g.text, g.surface)
    expectAa('text2 on bg', g.text2, g.bg)
    expectAa('text2 on surface', g.text2, g.surface)
    expectAa('ink on accent', vars['--pl-accent-ink'] ?? '', a.hex)
    expectAa(`accentText ${accentText} on bg`, accentText, g.bg)
    expectAa(`accentText ${accentText} on surface`, accentText, g.surface)
    expectAa(`errorText ${errorText} on bg`, errorText, g.bg)
    expectAa(`errorText ${errorText} on surface`, errorText, g.surface)
    expectAa('sheet text on sheet', vars['--pl-sheet-text'] ?? '', sheet)
    expectAa('sheet accent on sheet', vars['--pl-sheet-accent'] ?? '', sheet)
  })
})

describe("every ground × accent pair reads at AA — Studio's tokens", () => {
  const TEXT_PAIRS: readonly (readonly [string, string])[] = [
    ['--foreground', '--background'],
    ['--card-foreground', '--card'],
    ['--popover-foreground', '--popover'],
    ['--primary-foreground', '--primary'],
    ['--secondary-foreground', '--secondary'],
    ['--muted-foreground', '--background'],
    ['--muted-foreground', '--card'],
    ['--muted-foreground', '--muted'],
    ['--muted-foreground', '--secondary'],
    ['--accent-foreground', '--accent'],
    ['--destructive-foreground', '--destructive'],
    ['--primary', '--background'],
    ['--primary', '--card'],
    ['--accent', '--background'],
    ['--accent', '--card'],
    ['--destructive', '--background'],
    ['--destructive', '--card'],
    ['--fg-3', '--bg-1'],
    ['--fg-3', '--paper'],
    ['--fg-3', '--bg-2'],
    ['--fg-3', '--muted'],
  ]
  /** Studio's text-primary chips: bg-primary/10, /15 and /20 over each fill Studio sets them on. */
  const PRIMARY_TINTS = [0.1, 0.15, 0.2]
  const TINTED_FILLS = ['--background', '--card', '--secondary', '--muted']

  function expectPrimaryReadsOnItsTints(tokens: Readonly<Record<string, string>>): void {
    const primary = drawn(tokens, '--primary')
    for (const alpha of PRIMARY_TINTS) {
      for (const fill of TINTED_FILLS) {
        expectAa(`--primary on --primary/${alpha} over ${fill}`, primary, mixHex(drawn(tokens, fill), primary, alpha))
      }
    }
  }

  it.each(PAIRS)('$ground with $accent', ({ ground, accent }) => {
    const tokens = lookToStudioTokens({ ...EMBER, ground, accent })
    for (const [foreground, background] of TEXT_PAIRS) {
      expectAa(`${foreground} on ${background}`, drawn(tokens, foreground), drawn(tokens, background))
    }
    expectPrimaryReadsOnItsTints(tokens)
    expectRatio('--ring on --background', drawn(tokens, '--ring'), drawn(tokens, '--background'), AA_NON_TEXT)
  })
})
