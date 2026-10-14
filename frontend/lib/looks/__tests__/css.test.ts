// @vitest-environment node
/**
 * PRD-260 — a look as Play's variables: every name --pl-* (K2), the same set
 * for every look, the display families through K6's next/font variables.
 */
import { describe, expect, it } from 'vitest'

import { ACCENT_KEYS, GROUNDS, GROUND_KEYS, TYPE_STYLES, TYPE_STYLE_KEYS } from '@/lib/looks/catalogue'
import { AA_TEXT, READABLE_TARGET, contrastRatio } from '@/lib/looks/contrast'
import { LOOK_VAR_NAMES, fontVars, lookToCssVars, rgbList, rgba, sheetOf } from '@/lib/looks/css'
import { DAYLIGHT, EMBER, PRESETS } from '@/lib/looks/presets'

describe('lookToCssVars', () => {
  it("gives Ember's variables", () => {
    expect(lookToCssVars(EMBER)).toEqual({
      '--pl-ground-bg': '#131110',
      '--pl-ground-surface': '#1D1917',
      '--pl-ground-text': '#F4EEE8',
      '--pl-ground-text-2': '#B7ADA4',
      '--pl-ground-line': 'rgba(244,238,232,.12)',
      '--pl-ground-chip': 'rgba(244,238,232,.06)',
      '--pl-ground-panel': 'rgba(244,238,232,.025)',
      '--pl-sheet-bg': '#F4EEE8',
      '--pl-sheet-text': '#131110',
      '--pl-auto-dusk': '#5D5753',
      '--pl-error': '#FF6B6B',
      '--pl-error-text': '#FF6B6B',
      '--pl-accent': '#FF6A1F',
      '--pl-accent-ink': '#1A0A02',
      '--pl-accent-line': 'rgba(255,106,31,0.45)',
      '--pl-accent-fill': 'rgba(255,106,31,0.12)',
      '--pl-accent-text': '#FF6A1F',
      '--pl-sheet-accent': '#AA4A1A',
      '--pl-display-family': 'var(--pl-font-instrument-serif),serif',
      '--pl-display-style': 'italic',
      '--pl-display-weight': '400',
      '--pl-display-tracking': '0',
      '--pl-display-hero': '72px',
      '--pl-display-brand': '28px',
      '--pl-display-status': '40px',
      '--pl-display-card': '22px',
      '--pl-auto-shell': '#C8692A',
      '--pl-auto-shell-filter': 'none',
      '--pl-auto-light': '#4BF0B7',
      '--pl-glow': 'rgba(255,106,31,0.24)',
    })
  })

  it('names only --pl-* variables (K2)', () => {
    for (const look of PRESETS) {
      for (const name of Object.keys(lookToCssVars(look))) expect(name).toMatch(/^--pl-[a-z0-9-]+$/)
    }
  })

  it('darkens accent and error text on Paper (Daylight) and keeps the fills raw', () => {
    const vars = lookToCssVars(DAYLIGHT)
    expect(vars['--pl-accent']).toBe('#FF6A1F')
    expect(vars['--pl-accent-text']).toBe('#AA4C1C')
    expect(vars['--pl-error']).toBe('#FF6B6B')
    expect(vars['--pl-error-text']).toBe('#A54B49')
    for (const name of ['--pl-accent-text', '--pl-error-text'] as const) {
      const colour = vars[name] ?? ''
      expect(contrastRatio(colour, GROUNDS.paper.bg)).toBeGreaterThanOrEqual(READABLE_TARGET)
      expect(contrastRatio(colour, GROUNDS.paper.surface)).toBeGreaterThanOrEqual(READABLE_TARGET)
    }
  })

  it('prints Deliverables on a cream sheet on dark grounds and on the surface on Paper', () => {
    expect(sheetOf('graphite')).toEqual({ bg: GROUNDS.graphite.text, text: GROUNDS.graphite.bg })
    expect(sheetOf('indigo')).toEqual({ bg: GROUNDS.indigo.text, text: GROUNDS.indigo.bg })
    expect(sheetOf('paper')).toEqual({ bg: GROUNDS.paper.surface, text: GROUNDS.paper.text })
  })

  it("keeps the sheet's text and accent text readable for all 24 ground × accent pairs", () => {
    for (const ground of GROUND_KEYS) {
      for (const accent of ACCENT_KEYS) {
        const vars = lookToCssVars({ ...EMBER, ground, accent })
        const sheet = vars['--pl-sheet-bg'] ?? ''
        for (const name of ['--pl-sheet-text', '--pl-sheet-accent'] as const) {
          const ratio = contrastRatio(vars[name] ?? '', sheet)
          const pair = `${ground} x ${accent}: ${name} on --pl-sheet-bg (${ratio.toFixed(2)}:1)`
          expect({ pair, passes: ratio >= AA_TEXT }).toEqual({ pair, passes: true })
        }
      }
    }
  })

  it('sets the same variables for every look', () => {
    for (const look of PRESETS) expect(Object.keys(lookToCssVars(look)).sort()).toEqual([...LOOK_VAR_NAMES].sort())
  })

  it('turns the glow off at 0 and scales it to 40%', () => {
    expect(lookToCssVars({ ...EMBER, glow: 0 })['--pl-glow']).toBe('rgba(255,106,31,0)')
    expect(lookToCssVars({ ...EMBER, glow: 40 })['--pl-glow']).toBe('rgba(255,106,31,0.4)')
  })

  it("takes the type style and Auto's finish from the catalogue", () => {
    const vars = lookToCssVars({ ...EMBER, font: 'clean', shell: 'cobalt', light: 'ice' })
    expect(vars['--pl-display-family']).toBe('var(--font-geist-sans),sans-serif')
    expect(vars['--pl-display-style']).toBe('normal')
    expect(vars['--pl-display-weight']).toBe('600')
    expect(vars['--pl-display-tracking']).toBe('-.03em')
    expect(vars['--pl-auto-shell-filter']).toBe('hue-rotate(195deg) saturate(1.1)')
    expect(vars['--pl-auto-light']).toBe('#9FD8FF')
  })

  it("reaches every display face through K6's next/font variables, never a literal family", () => {
    expect(fontVars('serif')['--pl-display-family']).toBe('var(--pl-font-instrument-serif),serif')
    expect(fontVars('bric')['--pl-display-family']).toBe('var(--pl-font-bricolage),sans-serif')
    expect(fontVars('round')['--pl-display-family']).toBe('var(--pl-font-fredoka),sans-serif')
    for (const key of TYPE_STYLE_KEYS) {
      const family = fontVars(key)['--pl-display-family'] ?? ''
      expect(family).toMatch(/^var\(--(pl-font-[a-z-]+|font-geist-sans)\),(serif|sans-serif)$/)
      for (const other of TYPE_STYLE_KEYS) expect(family).not.toContain(TYPE_STYLES[other].face)
    }
  })
})

describe('rgba helpers', () => {
  it('split #rrggbb into channels', () => {
    expect(rgbList('#FF6A1F')).toBe('255,106,31')
    expect(rgba('#4bf0b7', 0.5)).toBe('rgba(75,240,183,0.5)')
  })

  it('refuse anything but #rrggbb', () => {
    expect(() => rgbList('red')).toThrow("isn't a #rrggbb colour")
    expect(() => rgbList('#FFF')).toThrow()
  })
})
