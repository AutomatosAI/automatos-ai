import { applyCssVars, type CssVars } from '@/lib/look-kit/css'

import { CSS_VAR_NAME_PATTERN, lookFitsPrepaint } from '../boot-sanitize'
import type { PrepaintNumericRule, PrepaintTable, PrepaintVars } from '../boot-types'

/**
 * PRD-261 A-19d2 (plan §2.2.4) — a look on <html>, from the theme's prepaint table alone: the walk
 * THEME_BOOT_SCRIPT (lib/theme-host/boot.ts) makes before first paint, as a pure function, so the
 * runtime paints after hydration exactly what the boot script painted (paint.test.ts runs both).
 * Each choice knob's table in the table's order, then the pair tables, then each numeric rule as
 * `rgba(<rgb>,<alpha>)` with alpha = value / scale, capped by its cap table (a theme's safe glow).
 * A later table's value for a name wins, as a later setProperty does. Only names that pass the
 * boot's custom-property pattern and string values are painted, as the script checks them again.
 *
 * Nothing here knows a theme: the variables, Studio's tokens among them, are the table's own,
 * precomputed by the theme (FX-17/18 for Play: paint-play-parity.test.ts holds the two equal).
 */

/** The only numeric template the boot script paints. */
const RGBA_ALPHA = 'rgba-alpha'

type LookValues = Readonly<Record<string, unknown>>

function own(record: Readonly<Record<string, unknown>> | null | undefined, key: unknown): boolean {
  return record !== null && record !== undefined && typeof key === 'string' && Object.prototype.hasOwnProperty.call(record, key)
}

/** The variables of one table entry the boot script would set: pattern-checked names, string values. */
function paintable(vars: PrepaintVars | undefined): [string, string][] {
  if (vars === undefined) return []
  return Object.keys(vars)
    .filter((name) => CSS_VAR_NAME_PATTERN.test(name) && typeof vars[name] === 'string')
    .map((name): [string, string] => [name, vars[name]])
}

/** One numeric rule's variable for `look`, or null when the boot script skips the rule. */
function numericVar(rule: PrepaintNumericRule, look: LookValues): [string, string] | null {
  if (rule.template !== RGBA_ALPHA || !CSS_VAR_NAME_PATTERN.test(rule.var)) return null
  const value = look[rule.knob]
  const colourKey = look[rule.color.knob]
  if (typeof value !== 'number' || !own(rule.color.rgb, colourKey)) return null
  let alpha = value / rule.scale
  if (rule.capTable !== null && rule.caps !== null) {
    const row = rule.caps[String(look[rule.capTable[0]])]
    alpha = Math.min(alpha, row[String(look[rule.capTable[1]])])
  }
  return [rule.var, `rgba(${rule.color.rgb[String(colourKey)]},${alpha})`]
}

/** The entries the table paints for a look that fits it, in the boot script's order. */
function walk(table: PrepaintTable, look: LookValues): [string, string][] {
  const knobs = Object.keys(table.knobs).flatMap((knob) => paintable(table.knobs[knob][String(look[knob])]))
  const pairs = table.pairs.flatMap((pair) => paintable(pair.table[String(look[pair.a])][String(look[pair.b])]))
  const numeric = table.numeric.flatMap((rule) => {
    const entry = numericVar(rule, look)
    return entry === null ? [] : [entry]
  })
  return [...knobs, ...pairs, ...numeric]
}

/**
 * The variables THEME_BOOT_SCRIPT puts on <html> for `look`: the table's default look's when `look`
 * doesn't fit the table, as the script falls back; none when the defaults don't fit either.
 */
export function paint(table: PrepaintTable, look: LookValues): CssVars {
  const painted = lookFitsPrepaint(table, look) ? look : table.defaults
  if (!lookFitsPrepaint(table, painted)) return {}
  return Object.fromEntries(walk(table, painted))
}

/** Every variable the table can paint, for any look: what leaving the theme takes off <html>. */
export function paintNames(table: PrepaintTable): readonly string[] {
  const names = new Set<string>()
  const add = (vars: PrepaintVars) => paintable(vars).forEach(([name]) => names.add(name))
  Object.values(table.knobs).forEach((options) => Object.values(options).forEach(add))
  table.pairs.forEach((pair) => Object.values(pair.table).forEach((row) => Object.values(row).forEach(add)))
  table.numeric.filter((rule) => CSS_VAR_NAME_PATTERN.test(rule.var)).forEach((rule) => names.add(rule.var))
  return Array.from(names)
}

/**
 * Paints the look on `root` (normally <html>) with style.setProperty: a variable the table paints
 * for another look but not for this one comes off, so nothing of the previous look stays.
 */
export function applyPaint(root: HTMLElement, table: PrepaintTable, look: LookValues): void {
  const vars = paint(table, look)
  paintNames(table)
    .filter((name) => !Object.prototype.hasOwnProperty.call(vars, name))
    .forEach((name) => root.style.removeProperty(name))
  applyCssVars(root, vars)
}

/** Takes every variable the table can paint off `root`; other inline styles stay. */
export function clearPaint(root: HTMLElement, table: PrepaintTable): void {
  paintNames(table).forEach((name) => root.style.removeProperty(name))
}
