// @vitest-environment node
/**
 * PRD-261 A-19d2 — one-time, while builtin Play's look runtime is still in core (A-19f2 deletes this
 * file with lib/looks in W-A8; the core ledger names Playwright `@parity fx19-looks-readable` as
 * what supersedes it): the generic look runtime paints installed Play exactly as builtin Play's own
 * runtime paints it. For the five presets and a seeded sample of valid looks, paint() over Play's
 * prepaint table equals lookDocumentVars(look) (Play's --pl-* variables and Studio's tokens, FX-17/18)
 * minus motion, which the runtimes mark apart (builtin data-pl-motion/--pl-motion, the generic one
 * data-theme-motion/--theme-motion). And the generic checks built from that table take every one
 * of those looks, so a Play look kept today is a look the generic runtime runs.
 */
import { describe, expect, it } from 'vitest'

import type { CssVars } from '@/lib/look-kit/css'
import { ACCENT_KEYS, GROUND_KEYS, LIGHT_KEYS, SHELL_KEYS, TYPE_STYLE_KEYS } from '@/lib/looks/catalogue'
import { MOTION_VAR, lookDocumentVars } from '@/lib/looks/document'
import { validateLook } from '@/lib/looks/engine'
import { DEFAULT_LOOK, PRESETS } from '@/lib/looks/presets'
import { GLOW_MAX, GLOW_MIN, LOOK_INTERFACE, LOOK_SCHEMA, type Look } from '@/lib/looks/types'
import { sanitizeBootLooks, sanitizePrepaint } from '@/lib/theme-host/boot-sanitize'
import { lookPrepaintTable } from '@/lib/theme-host/builtin/play-prepaint'
import { paint } from '@/lib/theme-host/looks/paint'
import { checkLook, themeLookRulesOf } from '@/lib/theme-host/looks/spec'

/** The sample is the same on every run: a seeded generator, not Math.random. */
const SEED = 0x19d2
const SAMPLE_SIZE = 96
const UINT32 = 2 ** 32

/** mulberry32: a small, well-mixed 32-bit generator; values in [0, 1). */
function seeded(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / UINT32
  }
}

function pick<T>(next: () => number, list: readonly T[]): T {
  return list[Math.floor(next() * list.length)]
}

function sampleLooks(): Look[] {
  const next = seeded(SEED)
  return Array.from({ length: SAMPLE_SIZE }, (_, index): Look => ({
    schema: LOOK_SCHEMA,
    interface: LOOK_INTERFACE,
    id: `sample-${index + 1}`,
    name: `Sample ${index + 1}`,
    sub: 'Seeded',
    ground: pick(next, GROUND_KEYS),
    accent: pick(next, ACCENT_KEYS),
    font: pick(next, TYPE_STYLE_KEYS),
    shell: pick(next, SHELL_KEYS),
    light: pick(next, LIGHT_KEYS),
    glow: GLOW_MIN + Math.floor(next() * (GLOW_MAX - GLOW_MIN + 1)),
  }))
}

function withoutMotion(vars: CssVars): CssVars {
  return Object.fromEntries(Object.entries(vars).filter(([name]) => name !== MOTION_VAR))
}

/** Play's table as an installed Play's boot carries it, through the browser's gate. */
const TABLE = sanitizePrepaint(lookPrepaintTable())
const LOOKS = [...PRESETS, ...sampleLooks()]

describe("PRD-261 A-19d2 — the generic runtime paints Play's table as builtin Play's runtime paints a look", () => {
  it('reads the table through the gate whole, and samples valid looks of every kind', () => {
    expect(TABLE).toEqual(lookPrepaintTable())
    expect(LOOKS.filter((look) => !validateLook(look).ok).map((look) => look.id)).toEqual([])
    expect(new Set(LOOKS.map((look) => look.ground)).size).toBe(GROUND_KEYS.length)
    expect(new Set(LOOKS.map((look) => look.accent)).size).toBe(ACCENT_KEYS.length)
  })

  it.each(LOOKS.map((look) => [look.id, look] as const))('%s: paint(table) equals lookDocumentVars minus motion', (_id, look) => {
    if (TABLE === null) throw new Error("Play's table didn't pass the gate")
    // A spread: Look is an interface, and paint() takes any record of a look's fields.
    expect(paint(TABLE, { ...look })).toEqual(withoutMotion(lookDocumentVars(look)))
  })

  it("the generic checks built from Play's table take every one of those looks as it is", () => {
    const rules = themeLookRulesOf({
      prepaint: TABLE,
      lookGuard: { schema: LOOK_SCHEMA, iface: LOOK_INTERFACE },
      looks: sanitizeBootLooks({ defaultLook: DEFAULT_LOOK.id, presets: PRESETS }),
    })
    if (rules === null) throw new Error("Play's boot has no look rules")
    expect(rules.presets.map((look) => look.id)).toEqual(PRESETS.map((look) => look.id))
    expect(LOOKS.map((look) => checkLook(rules, look))).toEqual(LOOKS.map((look) => ({ ok: true, look })))
  })
})
