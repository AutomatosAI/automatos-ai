// @vitest-environment node
/** PRD-260 — the Ember catalogue and the five presets, as Play's Looks prototype has them. */
import { describe, expect, it } from 'vitest'

import {
  ACCENTS,
  ACCENT_KEYS,
  ERROR_COLOUR,
  GROUNDS,
  GROUND_KEYS,
  LIGHTS,
  LIGHT_KEYS,
  SHELLS,
  SHELL_KEYS,
  TYPE_STYLES,
  TYPE_STYLE_KEYS,
  isKeyOf,
} from '@/lib/looks/catalogue'
import { DEFAULT_LOOK, PRESETS, findPreset, isPresetId } from '@/lib/looks/presets'
import { LOOK_SCHEMA } from '@/lib/looks/types'

const HEX = /^#[0-9A-F]{6}$/

describe('the Ember catalogue', () => {
  it('has the sections and keys of the Looks prototype, in order', () => {
    expect(GROUND_KEYS).toEqual(['graphite', 'obsidian', 'indigo', 'paper'])
    expect(ACCENT_KEYS).toEqual(['molten', 'cream', 'mint', 'gold', 'periwinkle', 'rose'])
    expect(TYPE_STYLE_KEYS).toEqual(['serif', 'clean', 'bric', 'round'])
    expect(SHELL_KEYS).toEqual(['bronze', 'gold', 'silver', 'cobalt'])
    expect(LIGHT_KEYS).toEqual(['mint', 'amber', 'ice', 'rose'])
  })

  it("keeps the prototype's values", () => {
    expect(GROUNDS.graphite).toMatchObject({ bg: '#131110', surface: '#1D1917', text: '#F4EEE8', text2: '#B7ADA4' })
    expect(GROUNDS.indigo.name).toBe('Night')
    expect(GROUNDS.paper.panel).toBe('rgba(255,255,255,.5)')
    expect(ACCENTS.molten).toEqual({ name: 'Molten', hex: '#FF6A1F', ink: '#1A0A02' })
    expect(TYPE_STYLES.serif).toEqual({
      name: 'Editorial',
      face: 'Instrument Serif',
      family: 'var(--pl-font-instrument-serif),serif',
      generic: 'serif',
      style: 'italic',
      weight: 400,
      letterSpacing: '0',
      sizes: { hero: 72, brand: 28, status: 40, card: 22 },
    })
    expect(TYPE_STYLES.clean).toMatchObject({ face: 'Geist', family: 'var(--font-geist-sans),sans-serif', generic: 'sans-serif', letterSpacing: '-.03em' })
    expect(TYPE_STYLES.round.sizes.brand).toBe(20)
    expect(SHELLS.bronze.filter).toBe('none')
    expect(SHELLS.cobalt.filter).toBe('hue-rotate(195deg) saturate(1.1)')
    expect(LIGHTS.ice.hex).toBe('#9FD8FF')
  })

  it("names each type style's generic family as the one its family falls back to (M99)", () => {
    for (const style of Object.values(TYPE_STYLES)) {
      expect({ face: style.face, ends: style.family.endsWith(`,${style.generic}`) }).toEqual({ face: style.face, ends: true })
    }
  })

  it('fixes the error colour', () => {
    expect(ERROR_COLOUR).toBe('#FF6B6B')
  })

  it('writes every solid colour as #RRGGBB', () => {
    const solids = [
      ...Object.values(GROUNDS).flatMap((ground) => [ground.bg, ground.surface, ground.text, ground.text2]),
      ...Object.values(ACCENTS).flatMap((accent) => [accent.hex, accent.ink]),
      ...Object.values(SHELLS).map((shell) => shell.hex),
      ...Object.values(LIGHTS).map((light) => light.hex),
      ERROR_COLOUR,
    ]
    for (const colour of solids) expect(colour).toMatch(HEX)
  })

  it('only counts own keys as catalogue keys', () => {
    expect(isKeyOf(ACCENTS, 'molten')).toBe(true)
    expect(isKeyOf(ACCENTS, 'toString')).toBe(false)
    expect(isKeyOf(ACCENTS, '__proto__')).toBe(false)
    expect(isKeyOf(ACCENTS, 3)).toBe(false)
  })
})

describe('the presets', () => {
  it("are the prototype's five, with Ember as the default", () => {
    expect(PRESETS.map((look) => look.name)).toEqual(['Ember', 'Obsidian', 'Aurora', 'Workshop', 'Daylight'])
    expect(DEFAULT_LOOK.id).toBe('ember')
    expect(DEFAULT_LOOK).toMatchObject({ ground: 'graphite', accent: 'molten', font: 'serif', shell: 'bronze', light: 'mint', glow: 24 })
    expect(findPreset('daylight')).toMatchObject({ ground: 'paper', accent: 'molten', glow: 10 })
    expect(findPreset('aurora')).toMatchObject({ ground: 'obsidian', accent: 'mint', font: 'bric', light: 'ice', glow: 20 })
    expect(LOOK_SCHEMA).toBe('automatos-interface-look/v1')
  })

  it('have unique ids and know which ids are presets', () => {
    expect(new Set(PRESETS.map((look) => look.id)).size).toBe(PRESETS.length)
    expect(isPresetId('workshop')).toBe(true)
    expect(isPresetId('c1')).toBe(false)
    expect(findPreset('c1')).toBeUndefined()
  })
})
