// @vitest-environment node
/**
 * PRD-261 A-19d1 — a theme's look rules from its boot alone (plan §2.2.4): the LookSpec
 * (lib/look-kit/validate.ts) from the boot's prepaint knobs, numeric rules and the guard's schema and
 * interface; a look outside the knobs, or one the table can't paint, is refused; the presets ride the
 * boot, the default first, and one that fails is left out. Run on hello-a09b's data.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { PrepaintNumericRule, PrepaintTable } from '@/lib/theme-host/boot-types'
import {
  checkLook,
  freshLookId,
  isPresetId,
  lookHeaderOf,
  lookSpecOf,
  NOT_PAINTABLE,
  themeLookRulesOf,
  type ThemeLookBoot,
} from '@/lib/theme-host/looks/spec'

import { DUSK, HELLO_BOOT_FILE, HELLO_RULES, helloBoot, INK, myLook, PAPER } from './hello-looks'

const SCHEMA = 'automatos-e2e-a09b-look/v1'
const INTERFACE = 'hello-a09b'

function errorsOf(input: unknown): readonly string[] {
  const result = checkLook(HELLO_RULES, input)
  return result.ok ? [] : result.errors
}

/** hello's boot with `fields` over it. */
function bootWith(fields: Partial<ThemeLookBoot>): ThemeLookBoot {
  return { ...helloBoot(), ...fields }
}

function tableOf(boot: ThemeLookBoot): PrepaintTable {
  if (boot.prepaint === null) throw new Error("hello-a09b's prepaint table didn't pass the gate")
  return boot.prepaint
}

beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe("the spec, from the boot's prepaint table and guard", () => {
  it("has the guard's header, a choice knob per knob table and a number knob per numeric rule", () => {
    const { spec } = HELLO_RULES
    expect(spec.lookSchema).toBe(SCHEMA)
    expect(spec.interface).toBe(INTERFACE)
    expect(Array.from(spec.knobs)).toEqual([
      ['ground', { kind: 'choice', choices: ['paper', 'ink', 'dusk'] }],
      ['accent', { kind: 'choice', choices: ['blue', 'green'] }],
      ['glow', { kind: 'number', min: 0, max: 40, integer: true }],
    ])
  })

  it("takes the range every rule on one knob accepts, whole when any rule is", () => {
    const table = tableOf(helloBoot())
    const [glow] = table.numeric
    const second: PrepaintNumericRule = { ...glow, var: '--a09b-glow-soft', min: 5, max: 60, integer: false }
    const spec = lookSpecOf({ ...table, numeric: [{ ...glow, integer: false }, second] }, { lookSchema: SCHEMA, interface: null })
    expect(spec.knobs.get('glow')).toEqual({ kind: 'number', min: 5, max: 40, integer: false })
    const whole = lookSpecOf({ ...table, numeric: [glow, second] }, { lookSchema: SCHEMA, interface: null })
    expect(whole.knobs.get('glow')).toEqual({ kind: 'number', min: 5, max: 40, integer: true })
  })

  it("without a guard (no interface in the theme's schema), the default preset's schema and no interface", () => {
    const presets = HELLO_RULES.presets.map(({ interface: _iface, ...look }) => look)
    const boot = bootWith({ lookGuard: null, looks: { defaultLook: 'paper', presets } })
    expect(lookHeaderOf(boot)).toEqual({ lookSchema: SCHEMA, interface: null })
    const rules = themeLookRulesOf(boot)
    if (rules === null) throw new Error('a boot without a guard still has look rules')
    expect(rules.spec.interface).toBeNull()
    expect(rules.presets.map((look) => look.id)).toEqual(['paper', 'dusk', 'ink'])
    expect(checkLook(rules, presets[1]).ok).toBe(true)
    expect(checkLook(rules, myLook())).toEqual({ ok: false, errors: [`"interface" isn't part of a Look. Remove it.`] })
  })

  it('is null without a table, a header, presets, or one preset that passes', () => {
    expect(themeLookRulesOf(bootWith({ prepaint: null }))).toBeNull()
    expect(themeLookRulesOf(bootWith({ looks: null }))).toBeNull()
    expect(themeLookRulesOf(bootWith({ looks: undefined }))).toBeNull()
    expect(themeLookRulesOf(bootWith({ lookGuard: null, looks: { defaultLook: 'x', presets: [{ id: 'x' }] } }))).toBeNull()
    const foreign = HELLO_RULES.presets.map((look) => ({ ...look, schema: 'another-theme-look/v1' }))
    expect(themeLookRulesOf(bootWith({ looks: { defaultLook: 'paper', presets: foreign } }))).toBeNull()
  })
})

describe("the presets, from the boot's looks", () => {
  it('are the default first, the rest in the boot order, each as checked', () => {
    expect(HELLO_BOOT_FILE.themeJson.look).toBe('paper')
    expect(HELLO_RULES.defaultLook).toBe(PAPER)
    expect(HELLO_RULES.presets).toEqual([PAPER, DUSK, INK])
    expect(PAPER).toEqual({
      schema: SCHEMA,
      id: 'paper',
      name: 'Paper',
      sub: 'Light, with a blue accent',
      interface: INTERFACE,
      ground: 'paper',
      accent: 'blue',
      glow: 12,
    })
    expect(Object.isFrozen(PAPER)).toBe(true)
  })

  it('leave out a preset that fails, with a warning; a failing default gives way to the first that passes', () => {
    const [paper, dusk, ink] = HELLO_RULES.presets
    const boot = bootWith({ looks: { defaultLook: 'paper', presets: [{ ...paper, glow: 99 }, dusk, ink] } })
    const rules = themeLookRulesOf(boot)
    expect(rules?.defaultLook).toEqual(DUSK)
    expect(rules?.presets).toEqual([DUSK, INK])
    expect(console.warn).toHaveBeenCalledTimes(1)
  })

  it('know their ids, and a fresh id is one no preset or saved look uses', () => {
    expect(['paper', 'dusk', 'ink'].every((id) => isPresetId(HELLO_RULES, id))).toBe(true)
    expect(isPresetId(HELLO_RULES, 'my-look-1')).toBe(false)
    expect(freshLookId(HELLO_RULES, [])).toBe('look-1')
    expect(freshLookId(HELLO_RULES, [myLook({ id: 'look-1' }), myLook({ id: 'look-2' })])).toBe('look-3')
    expect(freshLookId({ presets: [{ ...PAPER, id: 'look-1' }] }, [])).toBe('look-2')
  })
})

describe('checkLook (the one gate for a look from outside the code)', () => {
  it('takes a look of the theme, trimmed, with only the spec fields, frozen', () => {
    const result = checkLook(HELLO_RULES, { ...myLook(), name: '  Night  ' })
    expect(result).toEqual({ ok: true, look: myLook({ name: 'Night' }) })
    expect(result.ok && Object.isFrozen(result.look)).toBe(true)
  })

  it('refuses a look outside the knobs, saying why in plain words', () => {
    expect(errorsOf({ ...myLook(), ground: 'graphite' })).toEqual(['"ground" must be one of its choices: paper, ink, dusk.'])
    expect(errorsOf({ ...myLook(), accent: '#ff0000', glow: 99 })).toEqual([
      '"accent" must be one of its choices: blue, green.',
      '"glow" must be a whole number from 0 to 40.',
    ])
    expect(errorsOf({ ...myLook(), glow: 12.5 })).toEqual(['"glow" must be a whole number from 0 to 40.'])
    expect(errorsOf({ ...myLook(), css: 'body{}' })).toEqual([`"css" isn't part of a Look. Remove it.`])
    expect(errorsOf({ ...myLook(), interface: 'another' })).toEqual([`This Look is for another interface: "interface" must be "${INTERFACE}".`])
    expect(errorsOf({ ...myLook(), schema: 'another-theme-look/v1' })).toEqual([`This isn't a v1 Look: "schema" must be "${SCHEMA}".`])
    expect(errorsOf('paper')).toEqual(['A Look must be a JSON object.'])
  })

  it("refuses a look the spec passes but the table can't paint (a pair without its entry)", () => {
    const table = tableOf(helloBoot())
    const pair = { a: 'ground', b: 'accent', table: { paper: { blue: { '--a09b-on-accent': '#ffffff' } } } }
    const rules = { spec: HELLO_RULES.spec, table: { ...table, pairs: [pair] } }
    expect(checkLook(rules, PAPER).ok).toBe(true)
    expect(checkLook(rules, myLook())).toEqual({ ok: false, errors: [NOT_PAINTABLE] })
  })
})
