// @vitest-environment node
/**
 * PRD-261 M-01 — the phone's token list is Studio's (PB-34). MOBILE_TOKEN_NAMES (mobile-manifest.ts; the
 * parity test holds contract.py's equal) and PRD-268's STUDIO_STATUS_TOKEN_NAMES (K13; the phone draws
 * none of them until US-T01) are exactly the names look-kit's lookToStudioTokens writes as 'H S% L%'
 * triplets, for any theme's catalogue (a made-up one here, no theme's own), and none of the eight
 * aliases it writes pre-wrapped. The manifest's colour check accepts exactly what look-kit's
 * parsers read, so a checked block parses on the phone, which vendors look-kit.
 */
import { describe, expect, it } from 'vitest'

import { parseHex, toHex } from '@/lib/look-kit/contrast'
import { tripletToHex } from '@/lib/look-kit/hsl'
import { STUDIO_STATUS_TOKEN_NAMES, lookToStudioTokens } from '@/lib/look-kit/studio-tokens'
import type { LookCatalogue } from '@/lib/look-kit/types'
import { isMobileColour } from '../mobile-colour'
import { MOBILE_CONTRAST_PAIRS, MOBILE_REQUIRED_TOKENS, MOBILE_TOKEN_NAMES } from '../mobile-manifest'

const ZINC: LookCatalogue = {
  prefix: 'zinc',
  errorColour: '#E5484D',
  glow: { min: 0, max: 20 },
  grounds: {
    slate: {
      bg: '#0F1115', surface: '#181B21', text: '#ECEEF2', text2: '#9AA3AF',
      line: 'rgba(236,238,242,.14)', chip: 'rgba(236,238,242,.07)', panel: 'rgba(236,238,242,.03)',
    },
    chalk: {
      bg: '#F7F7F4', surface: '#FFFFFF', text: '#1B1E23', text2: '#5B6470',
      line: 'rgba(27,30,35,.12)', chip: 'rgba(27,30,35,.05)', panel: 'rgba(255,255,255,.6)',
    },
  },
  accents: { amber: { hex: '#F5A524', ink: '#2A1A00' }, iris: { hex: '#6E56CF', ink: '#FFFFFF' } },
  typeStyles: {
    plain: {
      family: 'var(--font-geist-sans),sans-serif', style: 'normal', weight: 500, letterSpacing: '0',
      sizes: { hero: 48, brand: 16, status: 28, card: 16 },
    },
  },
  shells: { matte: { hex: '#7A8290', filter: 'none' } },
  lights: { cool: { hex: '#9AD0FF' } },
}
const LOOKS = Object.keys(ZINC.grounds).flatMap((ground) => Object.keys(ZINC.accents).map((accent) => ({ ground, accent })))
/** The aliases globals.css writes pre-wrapped, as studio-tokens.ts writes them: hsl(…), --accent-2 with an alpha. */
const ALIASES = ['--paper', '--bg-1', '--bg-2', '--fg', '--fg-2', '--fg-3', '--border-2', '--accent-2']

/** True when look-kit's `parse` reads `value` (it throws on anything else). */
function reads(parse: (value: string) => string, value: string): boolean {
  try {
    parse(value)
    return true
  } catch {
    return false
  }
}

const isTriplet = (value: string): boolean => reads(tripletToHex, value)
const isHex = (value: string): boolean => reads((hex) => toHex(parseHex(hex)), value)
const sorted = (names: readonly string[]): string[] => [...names].sort()

describe("PRD-261 M-01 — the mobile token list and the status tokens are the tokens lookToStudioTokens writes as triplets", () => {
  it.each(LOOKS)('names every token written as a triplet for the $ground ground and the $accent accent', (look) => {
    const tokens = lookToStudioTokens(ZINC, look)
    const triplets = Object.keys(tokens).filter((name) => isTriplet(tokens[name]))
    expect(sorted(triplets)).toEqual(sorted([...MOBILE_TOKEN_NAMES, ...STUDIO_STATUS_TOKEN_NAMES]))
  })

  it.each(LOOKS)('leaves out exactly the eight pre-wrapped aliases ($ground, $accent)', (look) => {
    const tokens = lookToStudioTokens(ZINC, look)
    const wrapped = Object.keys(tokens).filter((name) => !isTriplet(tokens[name]))
    expect(sorted(wrapped)).toEqual(sorted(ALIASES))
    ALIASES.forEach((alias) => expect(MOBILE_TOKEN_NAMES).not.toContain(alias))
  })

  it('lists each name once, requires the ground and its text, and measures only listed tokens', () => {
    expect(new Set(MOBILE_TOKEN_NAMES).size).toBe(MOBILE_TOKEN_NAMES.length)
    expect(MOBILE_TOKEN_NAMES).toHaveLength(24)
    MOBILE_REQUIRED_TOKENS.forEach((name) => expect(MOBILE_TOKEN_NAMES).toContain(name))
    MOBILE_CONTRAST_PAIRS.forEach(([foreground, background]) => {
      expect(MOBILE_TOKEN_NAMES).toContain(foreground)
      expect(MOBILE_TOKEN_NAMES).toContain(background)
    })
    const kinds = MOBILE_CONTRAST_PAIRS.map(([, , kind]) => kind)
    expect([kinds.filter((kind) => kind === 'text').length, kinds.filter((kind) => kind === 'ui').length]).toEqual([8, 2])
  })
})

describe("PRD-261 M-01 — the block's colour check reads what look-kit reads", () => {
  it.each(LOOKS)('accepts every triplet lookToStudioTokens writes ($ground, $accent)', (look) => {
    const tokens = lookToStudioTokens(ZINC, look)
    MOBILE_TOKEN_NAMES.forEach((name) => expect(isMobileColour(tokens[name])).toBe(true))
  })

  it.each([
    '0 0% 100%', '0 0% 0%', '359.99 100% 100%', '360 0% 0%', '361 0% 0%', '0 101% 0%', '0 0% 100.01%', '0 0% 99.999%',
    '007 0% 0%', '1.5 2.25% 3%', '-1 0% 0%', '.5 0% 0%', '0 0 0', '0  0% 0%', ' 0 0% 0%', '0 0% 0%\n', 'hsl(0 0% 100%)',
    '#fff', '#FFFFFF', '#2f6fed', '#2f6fed80', '#gggggg', '', 'red', '\u0661 0% 0%',
  ])('%j passes exactly when tripletToHex or parseHex reads it', (value) => {
    expect(isMobileColour(value)).toBe(isTriplet(value) || isHex(value))
  })

  it('refuses anything but text', () => {
    const odd: unknown[] = [undefined, null, 0, true, ['0 0% 0%'], { h: 0 }]
    odd.forEach((value) => expect(isMobileColour(value)).toBe(false))
  })
})
