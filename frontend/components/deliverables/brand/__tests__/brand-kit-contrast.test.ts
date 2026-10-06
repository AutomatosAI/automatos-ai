/**
 * PRD-255 Wave 1, US-007 — the Brand kit page's contrast badges: the WCAG ratio computed from
 * the hexes, shown rounded down (a near miss never reads as the target, as the server says
 * it), each role measured on its ground against the least it needs.
 */
import { describe, it, expect } from 'vitest'

import {
  contrastRatio, formatRatio, parseHex, roleContrast, LARGE_TEXT_MIN_CONTRAST, TEXT_MIN_CONTRAST,
} from '../contrast'

describe('contrastRatio', () => {
  it('is 21 for black on white and 1 for a colour on itself', () => {
    expect(contrastRatio('#000000', '#ffffff')).toBeCloseTo(21, 5)
    expect(contrastRatio('#ffffff', '#000')).toBeCloseTo(21, 5)
    expect(contrastRatio('#c2410c', '#c2410c')).toBeCloseTo(1, 5)
  })

  it('matches the WCAG reference values', () => {
    // #767676 is the lightest grey that passes 4.5:1 on white; #777777 just misses.
    expect(contrastRatio('#767676', '#ffffff')!).toBeGreaterThanOrEqual(TEXT_MIN_CONTRAST)
    expect(contrastRatio('#777777', '#ffffff')!).toBeLessThan(TEXT_MIN_CONTRAST)
  })

  it('reads #rgb and refuses what is not a hex colour', () => {
    expect(parseHex('#abc')).toEqual([0xaa, 0xbb, 0xcc])
    expect(parseHex('orange')).toBeNull()
    expect(contrastRatio('#12', '#ffffff')).toBeNull()
    expect(contrastRatio(undefined, '#ffffff')).toBeNull()
  })
})

describe('formatRatio', () => {
  it('rounds down, so 4.48 never shows as 4.5', () => {
    expect(formatRatio(4.48)).toBe('4.4:1')
    expect(formatRatio(4.5)).toBe('4.5:1')
    expect(formatRatio(21)).toBe('21.0:1')
  })
})

describe('roleContrast', () => {
  const palette = { ink: '#1a1a2e', paper: '#ffffff', surface_2: '#ebe8e1', accent: '#ea580c', muted: '#999999', rule: '#dddddd' }

  it('measures a text role on the paper against 4.5:1, an accent against 3:1', () => {
    expect(roleContrast(palette, 'ink')).toMatchObject({ passes: true })
    expect(roleContrast(palette, 'ink')!.label).toMatch(/:1 on paper$/)
    expect(roleContrast(palette, 'muted')).toMatchObject({ passes: false })
    const accent = roleContrast(palette, 'accent')!
    expect(accent.ratio).toBeGreaterThanOrEqual(LARGE_TEXT_MIN_CONTRAST)
    expect(accent.ratio).toBeLessThan(TEXT_MIN_CONTRAST)
    expect(accent.passes).toBe(true)
  })

  it('measures a ground with the body text on it, and shows a hairline with no target', () => {
    expect(roleContrast(palette, 'surface_2')!.label).toMatch(/:1 ink on it$/)
    expect(roleContrast(palette, 'rule')!.passes).toBeNull()
  })

  it('has no badge when a colour it reads is missing', () => {
    expect(roleContrast(palette, 'accent_2')).toBeNull()
  })
})
