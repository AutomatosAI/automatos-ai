// @vitest-environment node
/**
 * PRD-260 §4 — the style sheet's data, computed from the look it is given,
 * ported from Automatos Play's tests/looks/style.test.ts: every colour a look
 * sets (named --pl-*, K2) with its readable variants explained, the live
 * contrast checks, Copy tokens, the type specimens at the look's sizes, the
 * spacing and radii as app/play-theme.css holds them, and Auto's 12 states
 * with the ring each takes.
 */
import fs from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

import {
  ACCENTS,
  ACCENT_KEYS,
  ERROR_COLOUR,
  GROUNDS,
  GROUND_KEYS,
  LIGHTS,
  TYPE_STYLES,
  TYPE_STYLE_KEYS,
} from '@/lib/looks/catalogue'
import { duskOf, lookToCssVars } from '@/lib/looks/css'
import { AURORA, DAYLIGHT, EMBER, PRESETS } from '@/lib/looks/presets'
import {
  COLOUR_TOKENS,
  MOTION_NOTES,
  RADII,
  SPACING_STEPS,
  TEXT_SPECIMENS,
  colourSwatches,
  contrastCheck,
  contrastChecks,
  displayLine,
  displaySpecimens,
  formatRatio,
  mascotSlots,
  radiusToken,
  rootBlock,
  variantNote,
} from '@/lib/looks/style'
import { MASCOT_STATES } from '@/lib/mascot/types'

const FRONTEND = path.resolve(__dirname, '../../..')
const PLAY_THEME_CSS = 'app/play-theme.css'
const COLOUR_VALUE = /^(#[0-9A-F]{6}|rgba\()/i

/** The --pl-* tokens app/play-theme.css declares in its html.play block, comments stripped. */
function themeTokens(): ReadonlyMap<string, string> {
  const css = fs.readFileSync(path.join(FRONTEND, PLAY_THEME_CSS), 'utf8').replace(/\/\*[\s\S]*?\*\//g, ' ')
  const block = /html\.play\s*\{([^}]*)\}/.exec(css)?.[1] ?? ''
  const entries = Array.from(block.matchAll(/(--pl-[\w-]+)\s*:\s*([^;]+);/g), (match): [string, string] => [
    match[1] ?? '',
    (match[2] ?? '').replace(/\s+/g, ' ').trim(),
  ])
  return new Map(entries)
}

function swatch(name: string, look = EMBER) {
  const found = colourSwatches(look).find((entry) => entry.name === name)
  if (!found) throw new Error(`no swatch ${name}`)
  return found
}

function ratio(label: string, look = EMBER): number {
  const found = contrastChecks(look).find((entry) => entry.label === label)
  if (!found) throw new Error(`no contrast check ${label}`)
  return found.ratio
}

describe('colour swatches', () => {
  it('cover every colour a look sets, and nothing else', () => {
    const colours = Object.entries(lookToCssVars(EMBER))
      .filter(([, value]) => COLOUR_VALUE.test(value))
      .map(([name]) => name)
      .sort()
    expect(COLOUR_TOKENS.map((token) => token.name).sort()).toEqual(colours)
    expect(COLOUR_TOKENS.every((token) => token.name.startsWith('--pl-') && token.role.length > 0)).toBe(true)
  })

  it("show the active look's values", () => {
    expect(swatch('--pl-ground-bg').value).toBe(GROUNDS.graphite.bg)
    expect(swatch('--pl-accent').value).toBe(ACCENTS.molten.hex)
    expect(swatch('--pl-ground-bg', DAYLIGHT).value).toBe(GROUNDS.paper.bg)
    expect(colourSwatches(DAYLIGHT).map((entry) => entry.value)).toEqual(
      COLOUR_TOKENS.map((token) => lookToCssVars(DAYLIGHT)[token.name]),
    )
  })

  it('note a readable variant only where it differs from the raw colour', () => {
    expect(swatch('--pl-accent-text').note).toBeNull()
    expect(swatch('--pl-error-text').note).toBeNull()
    expect(swatch('--pl-accent-text', DAYLIGHT).value).not.toBe(ACCENTS.molten.hex)
    expect(swatch('--pl-accent-text', DAYLIGHT).note).toBe('darkened for reading on Paper')
    expect(swatch('--pl-error-text', DAYLIGHT).note).toBe('darkened for reading on Paper')
    expect(swatch('--pl-accent', DAYLIGHT).note).toBeNull()
  })

  it('say which way a variant moved', () => {
    expect(variantNote('#FF6A1F', '#ff6a1f', 'Paper')).toBeNull()
    expect(variantNote('#FF6A1F', '#AA4C1C', 'Paper')).toBe('darkened for reading on Paper')
    expect(variantNote('#333333', '#777777', 'Night')).toBe('lightened for reading on Night')
  })
})

describe('contrast checks', () => {
  it('match the Ember style sheet', () => {
    expect(ratio('text / bg')).toBeCloseTo(16.36, 1)
    expect(ratio('text-2 / bg')).toBeCloseTo(8.54, 1)
    expect(ratio('ink / molten')).toBeCloseTo(6.74, 1)
    expect(formatRatio(ratio('text / bg'))).toBe('16.36:1')
  })

  it('check text, secondary text, button ink and both readable variants', () => {
    expect(contrastChecks(EMBER).map((entry) => entry.label)).toEqual([
      'text / bg',
      'text-2 / bg',
      'text / surface',
      'text-2 / surface',
      'ink / molten',
      'accent text / bg',
      'error text / bg',
    ])
    const daylight = contrastChecks(DAYLIGHT).find((entry) => entry.label === 'accent text / bg')
    expect(daylight?.foreground).toBe(lookToCssVars(DAYLIGHT)['--pl-accent-text'])
  })

  it('pass AA in every ground and accent pairing', () => {
    for (const ground of GROUND_KEYS) {
      for (const accent of ACCENT_KEYS) {
        for (const entry of contrastChecks({ ...EMBER, ground, accent })) {
          expect({ ground, accent, label: entry.label, passes: entry.passes }).toEqual({
            ground,
            accent,
            label: entry.label,
            passes: true,
          })
        }
      }
    }
  })

  it('fail below 4.5:1 and pass at it', () => {
    expect(contrastCheck('grey / grey', '#777777', '#888888').passes).toBe(false)
    const black = contrastCheck('text / bg', '#000000', '#FFFFFF')
    expect(black).toMatchObject({ label: 'text / bg', foreground: '#000000', background: '#FFFFFF', passes: true })
    expect(black.ratio).toBeCloseTo(21, 5)
  })
})

describe('Copy tokens', () => {
  it('is every look variable as a :root block', () => {
    const vars = lookToCssVars(EMBER)
    const lines = rootBlock(EMBER).split('\n')
    expect(lines[0]).toBe(':root {')
    expect(lines[lines.length - 1]).toBe('}')
    expect(lines).toHaveLength(Object.keys(vars).length + 2)
    expect(lines).toContain(`  --pl-ground-bg: ${GROUNDS.graphite.bg};`)
    expect(lines).toContain(`  --pl-accent-text: ${vars['--pl-accent-text']};`)
    expect(rootBlock(DAYLIGHT)).toContain(`  --pl-ground-bg: ${GROUNDS.paper.bg};`)
  })
})

describe('type specimens', () => {
  it("use the look's own display sizes", () => {
    for (const key of TYPE_STYLE_KEYS) {
      const { sizes } = TYPE_STYLES[key]
      expect(displaySpecimens({ ...EMBER, font: key }).map((specimen) => specimen.label)).toEqual([
        `Hero ${sizes.hero}px`,
        `Status ${sizes.status}px`,
        `Title ${sizes.card}px`,
      ])
    }
    expect(displaySpecimens(EMBER).map((specimen) => specimen.sample)).toEqual([
      'The fun way.',
      'Searching.',
      'Laptop shortlist',
    ])
  })

  it("name the display face by the design's name, not the next/font variable", () => {
    expect(displayLine(EMBER)).toBe('Editorial · Instrument Serif italic 400')
    expect(displayLine({ ...EMBER, font: 'bric' })).toBe('Characterful · Bricolage Grotesque normal 700')
    expect(displayLine(AURORA)).toBe('Characterful · Bricolage Grotesque normal 700')
    for (const key of TYPE_STYLE_KEYS) expect(displayLine({ ...EMBER, font: key })).not.toContain('var(')
  })

  it('label reading type with the sizes in app/play-theme.css', () => {
    const tokens = themeTokens()
    for (const specimen of TEXT_SPECIMENS) {
      const value = tokens.get(specimen.token) ?? ''
      const numbers = specimen.label.match(/\d+(\.\d+)?/g) ?? []
      expect(numbers.length).toBeGreaterThan(0)
      for (const number of numbers) {
        expect({ token: specimen.token, number, found: value.includes(number) }).toEqual({
          token: specimen.token,
          number,
          found: true,
        })
      }
    }
  })
})

describe('spacing and radii', () => {
  it('match the 4px scale in app/play-theme.css', () => {
    const tokens = themeTokens()
    expect(SPACING_STEPS).toHaveLength(7)
    for (const step of SPACING_STEPS) expect(tokens.get(step.token)).toBe(`${step.px}px`)
    expect(SPACING_STEPS.every((step) => step.px % 4 === 0)).toBe(true)
  })

  it('are the five Ember radii, each a token in app/play-theme.css', () => {
    const tokens = themeTokens()
    expect(RADII.map((radius) => radius.kind)).toEqual(['paper', 'tool', 'button', 'card', 'pill'])
    for (const radius of RADII) {
      const px = /^\d+/.exec(radius.label)?.[0] ?? '999'
      expect(radiusToken(radius.kind)).toBe(`--pl-radius-${radius.kind}`)
      expect(tokens.get(radiusToken(radius.kind))).toBe(`${px}px`)
    }
  })
})

describe('motion notes', () => {
  it("quote the theme's own motion numbers", () => {
    const tokens = themeTokens()
    expect(MOTION_NOTES).toHaveLength(5)
    expect(MOTION_NOTES.join(' ')).toContain(tokens.get('--pl-rise-distance') ?? 'missing')
    expect(MOTION_NOTES.join(' ')).toContain(tokens.get('--pl-duration-message') ?? 'missing')
    expect(MOTION_NOTES.join(' ')).toContain(tokens.get('--pl-duration-card') ?? 'missing')
    expect(MOTION_NOTES.join(' ')).toContain(tokens.get('--pl-duration-ring') ?? 'missing')
    expect(MOTION_NOTES.join(' ')).toContain(tokens.get('--pl-duration-glow') ?? 'missing')
  })
})

describe('mascot slots', () => {
  it("are Auto's 12 states with the ring each takes under the look", () => {
    const slots = mascotSlots(EMBER)
    expect(slots.map((slot) => slot.state)).toEqual([...MASCOT_STATES])
    expect(slots.every((slot) => slot.when.length > 0)).toBe(true)
    const ringOf = (state: string, look = EMBER) => mascotSlots(look).find((slot) => slot.state === state)?.ring
    expect(ringOf('idle')).toBe(LIGHTS.mint.hex)
    expect(ringOf('thinking')).toBe(ACCENTS.molten.hex)
    expect(ringOf('oops')).toBe(ERROR_COLOUR)
    expect(ringOf('sleeping')).toBe(duskOf('graphite'))
    expect(ringOf('idle', { ...EMBER, light: 'ice' })).toBe(LIGHTS.ice.hex)
  })

  it('are computed for every preset', () => {
    for (const look of PRESETS) expect(mascotSlots(look)).toHaveLength(MASCOT_STATES.length)
  })
})
