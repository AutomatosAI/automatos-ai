// @vitest-environment node
/**
 * PRD-260 D3 — a look as Studio's tokens: HSL triplets in Studio's own format
 * (tailwind.config.ts reads hsl(var(--x))), the pre-wrapped aliases wrapped as
 * globals.css writes them, the same names for every look, Studio's consequence
 * tokens in Play's coral (D-REVIEW F9) and no other semantic token, and no name
 * shared with Play's --pl-* set (K2).
 */
import { describe, expect, it } from 'vitest'

import { readableOn } from '@/lib/look-kit/contrast'
import { formatNumber, hexToTriplet, tripletToHex } from '@/lib/look-kit/hsl'
import { ACCENT_TINT_ALPHA_DARK, ACCENT_TINT_ALPHA_LIGHT, overGround } from '@/lib/look-kit/studio-tokens'
import { ACCENTS, ACCENT_KEYS, ERROR_COLOUR, GROUNDS, GROUND_KEYS, LIGHTS, SHELLS } from '@/lib/looks/catalogue'
import {
  LOOK_VAR_NAMES,
  STUDIO_TOKEN_NAMES,
  isDarkGround,
  lookToStudioTokens,
  studioAccent,
} from '@/lib/looks/engine'
import { DAYLIGHT, EMBER, PRESETS } from '@/lib/looks/presets'

const TRIPLET = /^\d{1,3}(\.\d{1,2})? \d{1,3}(\.\d{1,2})?% \d{1,3}(\.\d{1,2})?%$/
const WRAPPED = /^hsl\((.+?)( \/ 0\.\d{2})?\)$/
const ALIASES = ['--paper', '--bg-1', '--bg-2', '--fg', '--fg-2', '--fg-3', '--border-2', '--accent-2']
const TRIPLET_TOKENS = [
  '--background', '--foreground', '--card', '--card-foreground', '--popover', '--popover-foreground',
  '--primary', '--primary-foreground', '--secondary', '--secondary-foreground', '--muted', '--muted-foreground',
  '--accent', '--accent-foreground', '--destructive', '--destructive-foreground', '--border', '--input', '--ring',
  '--background-gradient-from', '--background-gradient-to', '--glass-border-hover', '--cc-hover', '--cc-hover-border',
]
const CONSEQUENCE = ['--accent', '--destructive']
const SEMANTIC = ['--success', '--warning', '--info', '--agent', '--chart-1', '--stage-1', '--navy', '--olive']
/** Play's coral on Paper, darkened to read (readableOn of #FF6B6B on #F4EEE8 and #FFFFFF). */
const PAPER_CORAL = '1.3 38.66% 46.67%' // #A54B49

const EVERY_PAIR = GROUND_KEYS.flatMap((ground) => ACCENT_KEYS.map((accent) => ({ ...EMBER, ground, accent })))

describe('lookToStudioTokens', () => {
  it("names exactly Studio's tokens and aliases", () => {
    expect([...STUDIO_TOKEN_NAMES].sort()).toEqual([...TRIPLET_TOKENS, ...ALIASES].sort())
  })

  it('writes every token as a triplet and every alias wrapped, for all 24 pairs', () => {
    for (const look of EVERY_PAIR) {
      const tokens = lookToStudioTokens(look)
      for (const name of TRIPLET_TOKENS) expect({ name, value: tokens[name] }).toEqual({ name, value: expect.stringMatching(TRIPLET) })
      for (const name of ALIASES) {
        const inner = WRAPPED.exec(tokens[name] ?? '')?.[1] ?? ''
        expect({ name, inner }).toEqual({ name, inner: expect.stringMatching(TRIPLET) })
      }
    }
  })

  it('sets the same names for every look', () => {
    for (const look of [...PRESETS, ...EVERY_PAIR]) {
      expect(Object.keys(lookToStudioTokens(look)).sort()).toEqual([...STUDIO_TOKEN_NAMES].sort())
    }
  })

  it("gives Ember's background, foreground, primary and consequence colour as hand-computed triplets", () => {
    const tokens = lookToStudioTokens(EMBER)
    expect(tokens['--background']).toBe('20 8.57% 6.86%') // #131110
    expect(tokens['--foreground']).toBe('30 35.29% 93.33%') // #F4EEE8
    // Molten mixed 40% toward the text: raw #FF6A1F is 3.86:1 under a bg-primary/20 chip on --muted.
    expect(tokens['--primary']).toBe('20.57 94.59% 70.98%') // #FB9F6F
    expect(tokens['--primary-foreground']).toBe('20 8.57% 6.86%') // #131110, the ground's bg
    expect(tokens['--ring']).toBe(tokens['--primary'])
    expect(tokens['--bg-1']).toBe('hsl(20 8.57% 6.86%)')
    expect(tokens['--accent']).toBe('0 100% 70.98%') // #FF6B6B, Play's coral, readable on Graphite as it is
    expect(tokens['--accent-foreground']).toBe('20 8.57% 6.86%')
    expect(tokens['--accent-2']).toBe('hsl(0 100% 70.98% / 0.14)')
  })

  it("uses the readable accent and coral on Paper, with the ground's bg as their ink and Studio Light's tint", () => {
    const tokens = lookToStudioTokens(DAYLIGHT)
    expect(tokens['--background']).toBe('30 35.29% 93.33%')
    expect(tokens['--primary']).toBe('20.21 64.63% 28.82%') // #793A1A
    expect(tokens['--primary-foreground']).toBe(hexToTriplet(GROUNDS.paper.bg))
    expect(tokens['--accent']).toBe(PAPER_CORAL)
    expect(tokens['--destructive']).toBe(PAPER_CORAL)
    expect(tokens['--destructive-foreground']).toBe(hexToTriplet(GROUNDS.paper.bg))
    expect(tokens['--accent-2']).toBe(`hsl(${PAPER_CORAL} / 0.10)`)
  })

  it('composites the rgba lines and chips over the ground before converting', () => {
    const { graphite } = GROUNDS
    const tokens = lookToStudioTokens(EMBER)
    expect(tripletToHex(tokens['--border'] ?? '')).toBe(overGround(graphite.line, graphite.bg))
    expect(tripletToHex(tokens['--secondary'] ?? '')).toBe(overGround(graphite.chip, graphite.bg))
    expect(tripletToHex(tokens['--muted'] ?? '')).toBe(overGround(graphite.chip, graphite.surface))
    expect(tokens['--input']).toBe(tokens['--border'])
  })

  it("draws --accent and --destructive in the ground's readable coral for all 24 pairs, never the look's accent", () => {
    for (const look of EVERY_PAIR) {
      const { bg, surface, text } = GROUNDS[look.ground]
      const coral = hexToTriplet(readableOn(ERROR_COLOUR, bg, surface, text))
      const tint = (isDarkGround(look.ground) ? ACCENT_TINT_ALPHA_DARK : ACCENT_TINT_ALPHA_LIGHT).toFixed(2)
      const accentFill = hexToTriplet(studioAccent(look.ground, look.accent).fill)
      const tokens = lookToStudioTokens(look)
      const pair = `${look.ground} × ${look.accent}`
      for (const name of CONSEQUENCE) expect({ pair, name, value: tokens[name] }).toEqual({ pair, name, value: coral })
      expect({ pair, accent: tokens['--accent'] === accentFill }).toEqual({ pair, accent: false })
      expect({ pair, primary: tokens['--primary'] }).toEqual({ pair, primary: accentFill })
      expect({ pair, ink: tokens['--destructive-foreground'] }).toEqual({ pair, ink: tokens['--accent-foreground'] })
      expect({ pair, tint: tokens['--accent-2'] }).toEqual({ pair, tint: `hsl(${coral} / ${tint})` })
    }
  })

  it('leaves the other semantic tokens to Studio (the tone follows the ground)', () => {
    for (const look of PRESETS) {
      const names = Object.keys(lookToStudioTokens(look))
      for (const name of SEMANTIC) expect(names).not.toContain(name)
    }
  })

  it("shares no name with Play's --pl-* variables", () => {
    expect(LOOK_VAR_NAMES.every((name) => name.startsWith('--pl-'))).toBe(true)
    expect(STUDIO_TOKEN_NAMES.filter((name) => LOOK_VAR_NAMES.includes(name))).toEqual([])
  })
})

describe('HSL helpers', () => {
  it('format at most two decimals with no trailing zeros', () => {
    expect(formatNumber(20.0893)).toBe('20.09')
    expect(formatNumber(99.999999)).toBe('100')
    expect(formatNumber(6.8627)).toBe('6.86')
    expect(formatNumber(0)).toBe('0')
  })

  it('round-trip every catalogue colour to the same #RRGGBB', () => {
    const solids = [
      ...Object.values(GROUNDS).flatMap((ground) => [ground.bg, ground.surface, ground.text, ground.text2]),
      ...Object.values(ACCENTS).flatMap((accent) => [accent.hex, accent.ink]),
      ...Object.values(SHELLS).map((shell) => shell.hex),
      ...Object.values(LIGHTS).map((light) => light.hex),
      '#000000',
      '#FFFFFF',
    ]
    for (const hex of solids) expect({ hex, back: tripletToHex(hexToTriplet(hex)) }).toEqual({ hex, back: hex })
  })

  it('round-trip every token of every pair', () => {
    for (const look of EVERY_PAIR) {
      for (const value of Object.values(lookToStudioTokens(look))) {
        const triplet = WRAPPED.exec(value)?.[1] ?? value
        expect(hexToTriplet(tripletToHex(triplet))).toBe(triplet)
      }
    }
  })

  it('refuse what is not a triplet or an rgba() colour', () => {
    expect(() => tripletToHex('hsl(0 0% 0%)')).toThrow("isn't an 'H S% L%' triplet")
    expect(() => tripletToHex('20 8.571% 6.86%')).toThrow()
    expect(() => tripletToHex('360 0% 0%')).toThrow('is out of range')
    expect(() => tripletToHex('0 101% 0%')).toThrow('is out of range')
    expect(() => tripletToHex('0 0% 100.01%')).toThrow('is out of range')
    expect(() => overGround('#FFFFFF', '#000000')).toThrow("isn't an rgba() colour")
    expect(overGround('rgba(255,255,255,.1)', '#000000')).toBe('#1A1A1A')
  })
})
