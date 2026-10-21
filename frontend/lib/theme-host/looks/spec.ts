/**
 * PRD-261 A-19d1 (plan §2.2.4) — what one theme's looks are, from its boot alone: the look spec
 * (lib/look-kit/validate.ts LookSpec) built from the boot's prepaint table and look guard, the table
 * every look must paint from, and the theme's presets with its default look first. No theme's
 * catalogue, variable prefix or look format is written here: the generic look runtime
 * (lib/theme-host/looks) runs any theme's looks from its data (the THEME_MARKERS scan holds it).
 *
 * The spec's knobs: one choice knob per table of `prepaint.knobs` (its options' keys, in order), then
 * one number knob per knob the numeric rules paint (several rules on one knob: the range all of them
 * accept). Its header is the guard's schema and interface. A theme whose looks/schema.json names no
 * interface has no guard (orchestrator serve.py theme_looks), so its default preset's `schema` is the
 * header and its looks carry no interface.
 *
 * A look passes when it fits the spec and the table (boot-sanitize.ts lookFitsPrepaint: every pair
 * and cap the boot script reads), so every look the runtime keeps is one the boot script paints on
 * the next load. The presets ride the boot (A-19c) and are checked the same way: one that fails is
 * left out with a warning, and when the default look's preset fails the first preset that passes is
 * the default (the server's rule, serve.py boot_looks). A boot without a table, a header or one
 * preset that passes has no look rules: null.
 */
import { validateLook, type LookFields, type LookKnobSpec, type LookNumberSpec, type LookSpec } from '@/lib/look-kit/validate'

import { lookFitsPrepaint } from '../boot-sanitize'
import type { PrepaintNumericRule, PrepaintTable, ThemeBoot, ThemeBootLooks } from '../boot-types'

/** A look that passed: its header fields are text, its knobs what the spec says. */
export type ThemeLook = LookFields & {
  readonly schema: string
  readonly id: string
  readonly name: string
  readonly sub: string
}

/** A look the theme's checks took (as kept: trimmed, only the spec's fields), or why they refused it. */
export type ThemeLookCheck =
  | { readonly ok: true; readonly look: ThemeLook }
  | { readonly ok: false; readonly errors: readonly string[] }

/** One theme's looks as the runtime runs them. */
export interface ThemeLookRules {
  readonly spec: LookSpec
  /** The boot's prepaint table: a look that passes paints from it. */
  readonly table: PrepaintTable
  /** The presets that passed, the default look first. They are never saved over, renamed or deleted. */
  readonly presets: readonly ThemeLook[]
  /** What a scope without an active look starts on, and what deleting the active look goes back to. */
  readonly defaultLook: ThemeLook
}

/** The parts of a (sanitized) boot the rules are built from. */
export type ThemeLookBoot = Pick<ThemeBoot, 'prepaint' | 'lookGuard' | 'looks'>

/** A look's header: its schema id and, when the theme names one, its interface. */
export type LookHeader = Pick<LookSpec, 'lookSchema' | 'interface'>

/** The words for a look the spec passes but the table can't paint (a pair or cap without its entry). */
export const NOT_PAINTABLE = "This Look doesn't fit this theme: one of its picks has nothing to paint with."

/** A fresh id is this prefix and the lowest whole number no preset or saved look uses. */
const FRESH_ID_PREFIX = 'look-'
const FIRST_FRESH_NUMBER = 1

type KnobEntry = readonly [string, LookKnobSpec]

function choiceKnobs(table: PrepaintTable): KnobEntry[] {
  return Object.keys(table.knobs).map((name): KnobEntry => [name, { kind: 'choice', choices: Object.keys(table.knobs[name]) }])
}

/** The range every rule on one knob accepts: the highest min, the lowest max, whole when any rule is. */
function numberKnob(rules: readonly PrepaintNumericRule[]): LookNumberSpec {
  return {
    kind: 'number',
    min: Math.max(...rules.map((rule) => rule.min)),
    max: Math.min(...rules.map((rule) => rule.max)),
    integer: rules.some((rule) => rule.integer),
  }
}

function numberKnobs(table: PrepaintTable): KnobEntry[] {
  const names = table.numeric.map((rule) => rule.knob).filter((name, index, all) => all.indexOf(name) === index)
  return names.map((name): KnobEntry => [name, numberKnob(table.numeric.filter((rule) => rule.knob === name))])
}

/** The spec of a theme's looks: `header`, then the table's choice knobs and its numeric rules' knobs. */
export function lookSpecOf(table: PrepaintTable, header: LookHeader): LookSpec {
  return Object.freeze({ ...header, knobs: new Map<string, LookKnobSpec>([...choiceKnobs(table), ...numberKnobs(table)]) })
}

/** The guard's header; without a guard, the default preset's schema and no interface; else null. */
export function lookHeaderOf(boot: ThemeLookBoot): LookHeader | null {
  if (boot.lookGuard !== null) return { lookSchema: boot.lookGuard.schema, interface: boot.lookGuard.iface }
  const schema = boot.looks?.presets[0]?.schema
  return typeof schema === 'string' ? { lookSchema: schema, interface: null } : null
}

function checkWith(spec: LookSpec, table: PrepaintTable, input: unknown): ThemeLookCheck {
  const result = validateLook(spec, input)
  if (!result.ok) return result
  if (!lookFitsPrepaint(table, result.look)) return { ok: false, errors: [NOT_PAINTABLE] }
  return { ok: true, look: Object.freeze(result.look) as ThemeLook }
}

/** `input` checked as one of the theme's looks: the look as kept, or every reason it isn't one. */
export function checkLook(rules: Pick<ThemeLookRules, 'spec' | 'table'>, input: unknown): ThemeLookCheck {
  return checkWith(rules.spec, rules.table, input)
}

function presetsOf(spec: LookSpec, table: PrepaintTable, looks: ThemeBootLooks): ThemeLook[] {
  return looks.presets.flatMap((preset) => {
    const checked = checkWith(spec, table, preset)
    if (checked.ok) return [checked.look]
    console.warn(`[looks] the theme's preset "${String(preset.id)}" doesn't fit its own looks, so it's left out:`, checked.errors)
    return []
  })
}

/** The theme's look rules from its boot, or null when the boot has none to run (see the header). */
export function themeLookRulesOf(boot: ThemeLookBoot): ThemeLookRules | null {
  const table = boot.prepaint
  const looks = boot.looks ?? null
  const header = lookHeaderOf(boot)
  if (table === null || looks === null || header === null) return null
  const spec = lookSpecOf(table, header)
  const passed = presetsOf(spec, table, looks)
  const first: ThemeLook | undefined = passed[0]
  const defaultLook = passed.find((look) => look.id === looks.defaultLook) ?? first
  if (defaultLook === undefined) return null
  const presets = Object.freeze([defaultLook, ...passed.filter((look) => look !== defaultLook)])
  return Object.freeze({ spec, table, presets, defaultLook })
}

/** True for one of the theme's presets' ids. */
export function isPresetId(rules: Pick<ThemeLookRules, 'presets'>, id: string): boolean {
  return rules.presets.some((look) => look.id === id)
}

/** 'look-N' with the lowest N no preset and none of `saved` uses (a valid look id). */
export function freshLookId(rules: Pick<ThemeLookRules, 'presets'>, saved: readonly Pick<ThemeLook, 'id'>[]): string {
  const ids = new Set([...rules.presets, ...saved].map((look) => look.id))
  let n = FIRST_FRESH_NUMBER
  while (ids.has(`${FRESH_ID_PREFIX}${n}`)) n += 1
  return `${FRESH_ID_PREFIX}${n}`
}
