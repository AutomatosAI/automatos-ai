// @vitest-environment node
/**
 * PRD-260 — every look the catalogue can make reads at WCAG AA: 4 grounds × 6
 * accents = 24 pairs, for Play's own variables and for Studio's tokens
 * (measured on the triplets converted back to sRGB, as the browser draws them),
 * and (M50) over the accent tints and the page glow Play draws text on.
 */
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

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
} from '@/lib/look-kit/contrast'
import { ACCENT_FAINT_ALPHA, ACCENT_HALO_ALPHA, ACCENT_TEXT_TINTS, GLOW_SCALE, SNAG_GLOW, rgba } from '@/lib/look-kit/css'
import { tripletToHex } from '@/lib/look-kit/hsl'
import { PRIMARY_TINT_MAX } from '@/lib/look-kit/studio-tokens'
import { ACCENTS, ACCENT_KEYS, ERROR_COLOUR, GROUNDS, GROUND_KEYS } from '@/lib/looks/catalogue'
import { GLOW_SNAG_VAR, GLOW_VAR, lookToCssVars, lookToStudioTokens, safeGlow, textFills } from '@/lib/looks/engine'
import { EMBER } from '@/lib/looks/presets'
import { GLOW_MAX, GLOW_MIN } from '@/lib/looks/types'

const PAIRS = GROUND_KEYS.flatMap((ground) => ACCENT_KEYS.map((accent) => ({ ground, accent })))
const CHANNEL_TOLERANCE = 4
const WRAPPED = /^hsl\((.+)\)$/
const PLAY_THEME_CSS = path.resolve(__dirname, '../../../app/play-theme.css')
const RGBA_ALPHA = /^rgba\([\d,]+,([\d.]+)\)$/
const PERCENT = 100
/** What Play sets straight on the page, so over the glow (lib/look-kit/css.ts pageTexts). */
const PAGE_TEXTS = ['--pl-ground-text', '--pl-ground-text-2', '--pl-accent-text', '--pl-error-text'] as const

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

/** The alpha of an rgba() look variable, as the page paints it. */
function alphaOf(vars: Readonly<Record<string, string>>, name: string): number {
  return Number(RGBA_ALPHA.exec(vars[name] ?? '')?.[1] ?? Number.NaN)
}

/** app/play-theme.css, comments stripped. */
function playTheme(): string {
  return readFileSync(PLAY_THEME_CSS, 'utf8').replace(/\/\*[\s\S]*?\*\//g, ' ')
}

/** The percent app/play-theme.css mixes the accent at for one of its --pl-tint-accent-* tints. */
function tintPercent(css: string, name: string): number {
  const mix = new RegExp(`${name}:\\s*color-mix\\(in srgb, var\\(--pl-accent\\) ([\\d.]+)%, transparent\\)`)
  return Number(mix.exec(css)?.[1] ?? Number.NaN)
}

/** Every text in PAGE_TEXTS reads at AA on `fill`. */
function expectPageTextsAa(vars: Readonly<Record<string, string>>, fill: string, where: string): void {
  for (const name of PAGE_TEXTS) expectAa(`${name} ${vars[name]} ${where}`, vars[name] ?? '', fill)
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

describe("every ground × accent pair reads at AA over Play's accent tints and page glow (M50)", () => {
  it('measures the tints app/play-theme.css draws: the halo 18%, the fill 12%, the faint 4%; the snag glow per ground', () => {
    const css = playTheme()
    expect(ACCENT_TEXT_TINTS).toEqual([0.18, 0.12, 0.04])
    expect(tintPercent(css, '--pl-tint-accent-halo') / PERCENT).toBeCloseTo(ACCENT_HALO_ALPHA, 10)
    expect(tintPercent(css, '--pl-tint-accent-faint') / PERCENT).toBeCloseTo(ACCENT_FAINT_ALPHA, 10)
    expect(css).toMatch(/--pl-tint-accent:\s*var\(--pl-accent-fill\);/)
    expect(css).toMatch(/--pl-page-glow:[^;]*var\(--pl-glow\)/)
    expect(css).toMatch(/--pl-page-glow-snag:[^;]*var\(--pl-glow-snag\)/)
    expect(css).not.toMatch(/--pl-page-glow-snag:[^;]*var\(--pl-error\)/)
  })

  it.each(PAIRS)('$ground with $accent: the accent text reads on the halo, the fill and the faint tint', ({ ground, accent }) => {
    const { bg } = GROUNDS[ground]
    const { hex } = ACCENTS[accent]
    const accentText = lookToCssVars({ ...EMBER, ground, accent })['--pl-accent-text'] ?? ''
    for (const alpha of ACCENT_TEXT_TINTS) {
      expectAa(`accentText ${accentText} on the accent at ${alpha}`, accentText, mixHex(bg, hex, alpha))
    }
  })

  it.each(PAIRS)('$ground with $accent: the page text reads over the glow at every setting, at its brightest', ({ ground, accent }) => {
    const { bg } = GROUNDS[ground]
    const { hex } = ACCENTS[accent]
    for (let glow = GLOW_MIN; glow <= GLOW_MAX; glow += 1) {
      const vars = lookToCssVars({ ...EMBER, ground, accent, glow })
      const alpha = alphaOf(vars, GLOW_VAR)
      expect({ glow, alpha, within: alpha <= glow / GLOW_SCALE }).toEqual({ glow, alpha, within: true })
      expectPageTextsAa(vars, mixHex(bg, hex, alpha), `over the glow at ${glow}`)
    }
  })

  it.each(PAIRS)('$ground with $accent: the glow is the one asked for up to the safe glow, the largest that reads', ({ ground, accent }) => {
    const cap = safeGlow(ground, accent)
    const { bg } = GROUNDS[ground]
    const { hex } = ACCENTS[accent]
    const painted = (glow: number) => alphaOf(lookToCssVars({ ...EMBER, ground, accent, glow }), GLOW_VAR)
    expect(painted(cap)).toBe(cap / GLOW_SCALE)
    expect(painted(GLOW_MAX)).toBe(cap / GLOW_SCALE)
    const vars = lookToCssVars({ ...EMBER, ground, accent })
    const past = mixHex(bg, hex, Math.min(cap + 1, GLOW_MAX) / GLOW_SCALE)
    const pastReads = PAGE_TEXTS.every((name) => contrastRatio(vars[name] ?? '', past) >= AA_TEXT)
    expect({ cap, largest: cap === GLOW_MAX || !pastReads }).toEqual({ cap, largest: true })
  })

  it.each(GROUND_KEYS)('%s: the page text of every accent reads over the snag glow, one coral per ground', (ground) => {
    const { bg } = GROUNDS[ground]
    const looks = ACCENT_KEYS.map((accent) => lookToCssVars({ ...EMBER, ground, accent }))
    const alpha = alphaOf(looks[0] ?? {}, GLOW_SNAG_VAR)
    expect(alpha).toBeGreaterThan(0)
    expect(alpha).toBeLessThanOrEqual(SNAG_GLOW / GLOW_SCALE)
    for (const vars of looks) {
      expect(vars[GLOW_SNAG_VAR]).toBe(rgba(ERROR_COLOUR, alpha))
      expectPageTextsAa(vars, mixHex(bg, ERROR_COLOUR, alpha), 'over the snag glow')
    }
  })
})
