// @vitest-environment node
/**
 * PRD-260 D6 — the light ring under Auto, ported from Automatos Play's
 * tests/mascot/ring.test.ts: each state's tone, the look colour each tone takes,
 * and the module CSS drawing the same mapping through app/play-theme.css's
 * --pl-color-* tokens.
 */
import fs from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

import { ACCENTS, ERROR_COLOUR, GROUNDS, LIGHTS } from '@/lib/looks/catalogue'
import { duskOf } from '@/lib/looks/css'
import { AURORA, EMBER } from '@/lib/looks/presets'
import type { Look } from '@/lib/looks/types'
import { RING_VARS, ringColour, ringTone, type RingTone } from '@/lib/mascot/ring'
import { MASCOT_STATES, type MascotName } from '@/lib/mascot/types'

const FRONTEND = path.resolve(__dirname, '../../..')
const MODULE_CSS = 'components/mascot/auto-mascot.module.css'
const PLAY_THEME_CSS = 'app/play-theme.css'
/** The custom property the stage sets for its ring (the module's own, --pl-* by K2). */
const RING_PROPERTY = '--pl-auto-ring'

const CALM: readonly MascotName[] = ['hello', 'idle', 'listening', 'celebrate']
const BUSY: readonly MascotName[] = ['thinking', 'working', 'searching', 'reading']
const DUSK: readonly MascotName[] = ['sleeping', 'lost']
const TONES = Object.keys(RING_VARS) as RingTone[]

function coloursOf(states: readonly MascotName[], look: Look): readonly string[] {
  return Array.from(new Set(states.map((state) => ringColour(state, look))))
}

function readCss(file: string): string {
  return fs.readFileSync(path.join(FRONTEND, file), 'utf8').replace(/\/\*[\s\S]*?\*\//g, ' ')
}

/** Innermost rule blocks: selector and body (no nested braces in either file's rules). */
function rules(css: string): { selector: string; body: string }[] {
  return Array.from(css.matchAll(/([^{}]+)\{([^{}]*)\}/g), (match) => ({ selector: (match[1] ?? '').trim(), body: match[2] ?? '' }))
}

function declared(body: string, property: string): string | undefined {
  const escaped = property.replace(/[.*+?^${}()|[\]\\-]/g, '\\$&')
  return new RegExp(`(?<![\\w-])${escaped}\\s*:\\s*([^;]+);`).exec(body)?.[1]?.trim()
}

describe('ringTone', () => {
  it('lights the ring by state group: calm, busy, oops, asleep or lost, loading', () => {
    const tones = Object.fromEntries(MASCOT_STATES.map((state) => [state, ringTone(state)]))
    expect(tones).toEqual({
      hello: 'light',
      idle: 'light',
      listening: 'light',
      celebrate: 'light',
      thinking: 'accent',
      working: 'accent',
      searching: 'accent',
      reading: 'accent',
      oops: 'error',
      sleeping: 'dusk',
      lost: 'dusk',
      loading: 'quiet',
    })
  })
})

describe('ringColour', () => {
  it("under Ember: his light calm, the accent busy, the coral oops, the ground's dusk asleep and lost", () => {
    expect(coloursOf(CALM, EMBER)).toEqual([LIGHTS[EMBER.light].hex])
    expect(coloursOf(BUSY, EMBER)).toEqual([ACCENTS[EMBER.accent].hex])
    expect(ringColour('oops', EMBER)).toBe(ERROR_COLOUR)
    expect(coloursOf(DUSK, EMBER)).toEqual([duskOf(EMBER.ground)])
    expect(ringColour('loading', EMBER)).toBe(GROUNDS[EMBER.ground].text2)
  })

  it('follows the look: Aurora takes its own light, accent and dusk, and the same coral', () => {
    expect(coloursOf(CALM, AURORA)).toEqual([LIGHTS[AURORA.light].hex])
    expect(coloursOf(BUSY, AURORA)).toEqual([ACCENTS[AURORA.accent].hex])
    expect(ringColour('oops', AURORA)).toBe(ERROR_COLOUR)
    expect(coloursOf(DUSK, AURORA)).toEqual([duskOf(AURORA.ground)])
  })

  it('makes dusk from the ground only, never the light or the accent', () => {
    const changed: Look = { ...EMBER, accent: 'rose', light: 'amber', shell: 'cobalt' }
    expect(ringColour('sleeping', changed)).toBe(duskOf(EMBER.ground))
    expect(ringColour('lost', { ...EMBER, ground: 'paper' })).toBe(duskOf('paper'))
  })
})

describe('the ring in auto-mascot.module.css', () => {
  const moduleRules = rules(readCss(MODULE_CSS))
  const themeBody = rules(readCss(PLAY_THEME_CSS)).find((rule) => rule.selector === 'html.play')?.body ?? ''

  /** The look variable a tone's rule draws the ring in, through the theme's --pl-color-* token. */
  function lookVarFor(tone: RingTone): string | undefined {
    const selector = tone === 'light' ? '.stage' : `.stage[data-ring="${tone}"]`
    const rule = moduleRules.find((candidate) => candidate.selector === selector)
    const token = /^var\((--pl-color-[\w-]+)\)$/.exec(declared(rule?.body ?? '', RING_PROPERTY) ?? '')?.[1]
    return token === undefined ? undefined : declared(themeBody, token)
  }

  it('draws each tone in the same look variable as ringColour', () => {
    for (const tone of TONES) expect({ tone, ring: lookVarFor(tone) }).toEqual({ tone, ring: `var(${RING_VARS[tone]})` })
  })

  it('draws the ring itself in that colour', () => {
    const ring = moduleRules.find((rule) => rule.selector === '.ring')
    expect(declared(ring?.body ?? '', 'border')).toBe(`2px solid var(${RING_PROPERTY})`)
  })

  it("tints the clip and the still with the look's shell filter, and nothing else", () => {
    const filtered = moduleRules.filter((rule) => declared(rule.body, 'filter') !== undefined)
    expect(filtered.map((rule) => rule.selector)).toEqual(['.figure'])
    expect(declared(filtered[0]?.body ?? '', 'filter')).toBe('var(--pl-filter-shell)')
    expect(declared(themeBody, '--pl-filter-shell')).toBe('var(--pl-auto-shell-filter)')
  })
})
