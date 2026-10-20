// @vitest-environment node
/**
 * PRD-261 A-13a — the look runtime is theme-neutral (PA-9): it paints and checks any theme's looks
 * from that theme's own catalogue and spec, under that theme's own prefix, and keeps PRD-260's
 * rules for every theme: Studio's consequence tokens are the error colour, never the accent (F9,
 * P260-FX-17), and the page glow never takes text under AA (M50, P260-FX-18). A made-up theme,
 * "acme", proves it: nothing here is Play's.
 */
import { describe, expect, it } from 'vitest'

import { readableOn } from '@/lib/look-kit/contrast'
import { GLOW_SCALE, cssVarName, glowVar, lookToCssVars, rgba, safeGlow } from '@/lib/look-kit/css'
import { hexToTriplet } from '@/lib/look-kit/hsl'
import { lookToStudioTokens } from '@/lib/look-kit/studio-tokens'
import type { LookCatalogue, LookPicks } from '@/lib/look-kit/types'
import { validateLook, type LookKnobSpec, type LookSpec } from '@/lib/look-kit/validate'
import { STUDIO_TOKEN_NAMES } from '@/lib/looks/engine'
import { validateLookSchema } from '@/lib/theme-host/look-schema'

const ACME: LookCatalogue = {
  prefix: 'acme',
  errorColour: '#D03030',
  glow: { min: 0, max: 30 },
  grounds: {
    night: {
      bg: '#101418', surface: '#1A2028', text: '#F0F4F8', text2: '#A8B4C0',
      line: 'rgba(240,244,248,.12)', chip: 'rgba(240,244,248,.06)', panel: 'rgba(240,244,248,.03)',
    },
    day: {
      bg: '#FAFAF7', surface: '#FFFFFF', text: '#14181C', text2: '#5A6470',
      line: 'rgba(20,24,28,.14)', chip: 'rgba(20,24,28,.06)', panel: 'rgba(255,255,255,.5)',
    },
  },
  accents: { teal: { hex: '#2EC4B6', ink: '#04201D' }, plum: { hex: '#9B5DE5', ink: '#0E0418' } },
  typeStyles: {
    plain: {
      family: 'var(--font-geist-sans),sans-serif', style: 'normal', weight: 600, letterSpacing: '0',
      sizes: { hero: 56, brand: 18, status: 30, card: 17 },
    },
  },
  shells: { steel: { hex: '#8A96A3', filter: 'none' } },
  lights: { warm: { hex: '#FFB347' } },
}

const PICKS: LookPicks = { ground: 'night', accent: 'teal', font: 'plain', shell: 'steel', light: 'warm', glow: 30 }
const ACME_VAR = /^--acme-[a-z0-9-]+$/

const SPEC: LookSpec = {
  lookSchema: 'acme-look/v2',
  interface: null,
  knobs: new Map<string, LookKnobSpec>([
    ['ground', { kind: 'choice', choices: ['night', 'day'] }],
    ['glow', { kind: 'number', min: 0, max: 1 }],
  ]),
}
const LOOK = { schema: 'acme-look/v2', id: 'mine-1', name: 'Mine', sub: '', ground: 'day', glow: 0.5 }

function errorsFor(spec: LookSpec, input: unknown): readonly string[] {
  const result = validateLook(spec, input)
  return result.ok ? [] : result.errors
}

describe("the look runtime paints any theme's catalogue", () => {
  it("names every variable under the catalogue's own prefix", () => {
    const vars = lookToCssVars(ACME, PICKS)
    expect(Object.keys(vars).filter((name) => !ACME_VAR.test(name))).toEqual([])
    expect(vars[cssVarName(ACME, 'ground-bg')]).toBe('#101418')
    expect(vars[cssVarName(ACME, 'accent')]).toBe('#2EC4B6')
    expect(vars[cssVarName(ACME, 'auto-light')]).toBe('#FFB347')
  })

  it("never paints the glow past the pair's safe glow, within the catalogue's own range", () => {
    for (const ground of ['night', 'day']) {
      for (const accent of ['teal', 'plum']) {
        const safe = safeGlow(ACME, ground, accent)
        expect(safe).toBeLessThanOrEqual(ACME.glow.max)
        const vars = lookToCssVars(ACME, { ...PICKS, ground, accent, glow: ACME.glow.max })
        expect(vars[glowVar(ACME)]).toBe(rgba(ACME.accents[accent].hex, safe / GLOW_SCALE))
      }
    }
  })

  it("gives Studio's consequence tokens the catalogue's error colour, never the accent (F9, P260-FX-17)", () => {
    for (const ground of ['night', 'day']) {
      const { bg, surface, text } = ACME.grounds[ground]
      const consequence = hexToTriplet(readableOn(ACME.errorColour, bg, surface, text))
      const tokens = lookToStudioTokens(ACME, { ground, accent: 'plum' })
      expect(tokens['--accent']).toBe(consequence)
      expect(tokens['--destructive']).toBe(consequence)
      expect(tokens['--primary']).not.toBe(consequence)
    }
  })

  it("sets the same Studio tokens for any theme as for Play's looks", () => {
    const names = Object.keys(lookToStudioTokens(ACME, { ground: 'day', accent: 'teal' }))
    expect([...names].sort()).toEqual([...STUDIO_TOKEN_NAMES].sort())
  })

  it('throws, naming the section, for a key the catalogue lacks or only inherits', () => {
    for (const ground of ['paper', 'toString', '__proto__']) {
      expect(() => lookToCssVars(ACME, { ...PICKS, ground })).toThrow(`"${ground}" isn't a ground of this catalogue`)
    }
    expect(() => lookToCssVars(ACME, { ...PICKS, font: 'serif' })).toThrow("isn't a type style")
  })
})

describe("the look runtime checks a look against the theme's own spec", () => {
  it('accepts a look with no interface when the spec names none, as a fresh object in the spec order', () => {
    const input = { glow: 0.5, ground: 'day', sub: '', name: '  Mine ', id: 'mine-1', schema: 'acme-look/v2' }
    const result = validateLook(SPEC, input)
    expect(result).toEqual({ ok: true, look: LOOK })
    expect(result.ok && Object.keys(result.look)).toEqual(['schema', 'id', 'name', 'sub', 'ground', 'glow'])
  })

  it("refuses an interface the spec doesn't name, and another theme's schema", () => {
    expect(errorsFor(SPEC, { ...LOOK, interface: 'play' })).toEqual([`"interface" isn't part of a Look. Remove it.`])
    expect(errorsFor(SPEC, { ...LOOK, schema: 'automatos-interface-look/v1' })).toEqual([
      `This isn't a v2 Look: "schema" must be "acme-look/v2".`,
    ])
  })

  it('needs the interface when the spec names one, and keeps it after the text fields', () => {
    const spec: LookSpec = { ...SPEC, interface: 'acme' }
    expect(errorsFor(spec, LOOK)).toEqual([`This Look is for another interface: "interface" must be "acme".`])
    const result = validateLook(spec, { ...LOOK, interface: 'acme' })
    expect(result.ok && Object.keys(result.look)).toEqual(['schema', 'id', 'name', 'sub', 'interface', 'ground', 'glow'])
  })

  it("checks each knob by the spec: its choices, and a number range that isn't whole unless it says so", () => {
    expect(errorsFor(SPEC, { ...LOOK, ground: 'paper' })).toEqual(['"ground" must be one of its choices: night, day.'])
    expect(errorsFor(SPEC, { ...LOOK, glow: 2 })).toEqual(['"glow" must be a number from 0 to 1.'])
    expect(errorsFor(SPEC, { ...LOOK, glow: Number.NaN })).toEqual(['"glow" must be a number from 0 to 1.'])
    expect(errorsFor(SPEC, { ...LOOK, ground: 'toString' })).toHaveLength(1)
  })

  it("takes a theme's checked looks/schema.json as its spec, as it is", () => {
    const { schema, errors } = validateLookSchema({
      schema: 'automatos-look-schema/v1',
      look: { schema: 'acme-look/v1' },
      knobs: {
        ground: { kind: 'choice', label: 'Background', choices: ['night', 'day'], default: 'night' },
        glow: { kind: 'number', min: 0, max: 40, integer: true, default: 10 },
      },
    })
    expect(errors).toEqual([])
    if (schema === null) throw new Error('the acme look schema should check')
    const look = { schema: 'acme-look/v1', id: 'a', name: 'A', sub: 'B', ground: 'night', glow: 10 }
    expect(validateLook(schema, look)).toEqual({ ok: true, look })
    expect(errorsFor(schema, { ...look, glow: 10.5 })).toEqual(['"glow" must be a whole number from 0 to 40.'])
  })
})
