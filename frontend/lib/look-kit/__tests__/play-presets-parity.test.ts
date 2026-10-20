// @vitest-environment node
/**
 * PRD-261 A-13a — the look runtime moves to lib/look-kit without changing one variable.
 * fixtures/play-presets-pre-move.json holds what Play's 5 presets painted BEFORE the move (Play's
 * --pl-* variables and Studio's tokens, main a290c428f); the code that paints them now (the
 * runtime in lib/look-kit with Play's catalogue bound in by lib/looks/engine.ts) must give
 * exactly those names, in the same order, with exactly those values.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'

import { lookToCssVars, lookToStudioTokens } from '@/lib/looks/engine'
import { PRESETS } from '@/lib/looks/presets'

type Vars = Readonly<Record<string, string>>

interface PresetVars {
  readonly id: string
  readonly cssVars: Vars
  readonly studioTokens: Vars
}

const FIXTURE = path.join(__dirname, 'fixtures', 'play-presets-pre-move.json')
const { presets } = JSON.parse(readFileSync(FIXTURE, 'utf8')) as { readonly presets: readonly PresetVars[] }
const PLAY_PRESET_COUNT = 5

function fixtureOf(id: string): PresetVars {
  const found = presets.find((entry) => entry.id === id)
  if (!found) throw new Error(`the parity fixture has no preset "${id}"`)
  return found
}

describe("Play's presets paint as they did before the look runtime moved", () => {
  it('the fixture holds each of the 5 presets once, in order', () => {
    expect(presets.map((entry) => entry.id)).toEqual(PRESETS.map((look) => look.id))
    expect(presets).toHaveLength(PLAY_PRESET_COUNT)
  })

  it("gives every preset Play's variables exactly as before, in the same order", () => {
    for (const look of PRESETS) {
      const vars = lookToCssVars(look)
      const before = fixtureOf(look.id).cssVars
      expect({ id: look.id, vars }).toEqual({ id: look.id, vars: before })
      expect(Object.keys(vars)).toEqual(Object.keys(before))
    }
  })

  it("gives every preset Studio's tokens exactly as before, in the same order", () => {
    for (const look of PRESETS) {
      const tokens = lookToStudioTokens(look)
      const before = fixtureOf(look.id).studioTokens
      expect({ id: look.id, tokens }).toEqual({ id: look.id, tokens: before })
      expect(Object.keys(tokens)).toEqual(Object.keys(before))
    }
  })
})
