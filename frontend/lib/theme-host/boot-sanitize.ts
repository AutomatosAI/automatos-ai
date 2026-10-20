/**
 * PRD-261 A-03 — the gate every theme boot passes before it is written into the page. A class
 * that fails HTML_CLASS_PATTERN, a custom property name that fails CSS_VAR_NAME_PATTERN and a
 * value that fails the CSS-value grammar below are dropped, never written; a prepaint table
 * whose default look no longer fits what is left is dropped whole. The boot script
 * (THEME_BOOT_SCRIPT) applies what passes with classList and style.setProperty only.
 *
 * The grammar is deliberately small: the characters colours, lengths, numbers, font stacks and
 * filter lists need, and only the functions in CSS_VALUE_FUNCTIONS, so no url(), image() or
 * expression() and no ';', '{', '}', quote, '<', '!' or '@' can reach a declaration.
 *
 * An installed theme's registry entry data (W-A6b) passes installed-entry.ts's own check.
 */
import { HTML_CLASS_PATTERN } from './constants'
import { sanitizeIntegrity } from './import-map'
import { sanitizeInstalledEntry } from './installed-entry'
import type { ThemeTonePolicy } from './registry-types'
import type {
  PrepaintNumericRule,
  PrepaintPair,
  PrepaintPairTable,
  PrepaintTable,
  PrepaintVars,
  ThemeBoot,
  ThemeLookGuard,
  ThemeLookStorage,
  ThemeTone,
} from './boot-types'

/** A custom property the boot may set: lower-case letters, digits and hyphens after '--'. */
export const CSS_VAR_NAME_PATTERN = /^--[a-z0-9-]{1,64}$/
export const CSS_VALUE_MAX_LENGTH = 200
const CSS_VALUE_CHARS = /^[A-Za-z0-9#%.,()/ +-]+$/
const CSS_FUNCTION_NAME = /([A-Za-z-]+)\(/g
/** The only functions a prepaint value may call. */
export const CSS_VALUE_FUNCTIONS: ReadonlySet<string> = new Set([
  'rgb', 'rgba', 'hsl', 'hsla', 'var', 'calc', 'min', 'max', 'clamp',
  'hue-rotate', 'saturate', 'brightness', 'grayscale', 'contrast', 'sepia', 'invert',
])
/** A look's field name, and a key of one of its tables. */
const KNOB_NAME = /^[A-Za-z][A-Za-z0-9_]{0,31}$/
const OPTION_KEY_MAX_LENGTH = 64
const RGB_LIST = /^\d{1,3},\d{1,3},\d{1,3}$/
/**
 * A look storage key: builtin Play's `automatos-play.look.current`, and an installed theme's
 * `automatos-theme.<publisher>/<name>.look.current` (play-look-storage.ts installedLookKeys, W-A6b
 * gap G2), whose theme id holds a '/' and may be 102 characters long.
 */
const STORAGE_KEY = /^[A-Za-z0-9._:/-]{1,160}$/
const GUARD_MAX_LENGTH = 128
/** Caps are alphas. */
const CAP_MIN = 0
const CAP_MAX = 1

type UnknownRecord = Readonly<Record<string, unknown>>

function isRecord(value: unknown): value is UnknownRecord {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function isOptionKey(key: string): boolean {
  return key.length > 0 && key.length <= OPTION_KEY_MAX_LENGTH
}

export function isThemeClass(name: unknown): name is string {
  return typeof name === 'string' && HTML_CLASS_PATTERN.test(name)
}

/** The CSS-value grammar: allowed characters, balanced parentheses, allowed functions only. */
export function isCssValue(value: unknown): value is string {
  if (typeof value !== 'string' || value.length === 0 || value.length > CSS_VALUE_MAX_LENGTH) return false
  if (!CSS_VALUE_CHARS.test(value)) return false
  if (value.split('(').length !== value.split(')').length) return false
  const names = Array.from(value.matchAll(CSS_FUNCTION_NAME), (match) => match[1].toLowerCase())
  return names.every((name) => CSS_VALUE_FUNCTIONS.has(name))
}

export function sanitizeClasses(classes: unknown): readonly string[] {
  if (!Array.isArray(classes)) return []
  return classes.filter(isThemeClass).filter((name, index, names) => names.indexOf(name) === index)
}

export function sanitizeVars(vars: unknown): PrepaintVars {
  if (!isRecord(vars)) return {}
  return Object.fromEntries(
    Object.entries(vars).filter(
      (entry): entry is [string, string] => CSS_VAR_NAME_PATTERN.test(entry[0]) && isCssValue(entry[1]),
    ),
  )
}

/** The own keys that name an option, each with its value as `clean` returns it; null drops the key. */
function sanitizeTable<V>(table: unknown, clean: (value: unknown) => V | null): Readonly<Record<string, V>> {
  if (!isRecord(table)) return {}
  const kept: [string, V][] = []
  for (const [key, value] of Object.entries(table)) {
    const cleaned = isOptionKey(key) ? clean(value) : null
    if (cleaned !== null) kept.push([key, cleaned])
  }
  return Object.fromEntries(kept)
}

function sanitizeCap(value: unknown): number | null {
  return isFiniteNumber(value) && value >= CAP_MIN && value <= CAP_MAX ? value : null
}

function sanitizePairTable<V>(table: unknown, clean: (value: unknown) => V | null): PrepaintPairTable<V> {
  return sanitizeTable(table, (row) => sanitizeTable(row, clean))
}

function sanitizePair(pair: unknown): PrepaintPair | null {
  if (!isRecord(pair) || typeof pair.a !== 'string' || typeof pair.b !== 'string') return null
  if (!KNOB_NAME.test(pair.a) || !KNOB_NAME.test(pair.b)) return null
  return { a: pair.a, b: pair.b, table: sanitizePairTable(pair.table, sanitizeVars) }
}

function sanitizeCapTable(rule: UnknownRecord): Pick<PrepaintNumericRule, 'capTable' | 'caps'> | null {
  if (rule.capTable === null) return { capTable: null, caps: null }
  const pair = rule.capTable
  if (!Array.isArray(pair) || pair.length !== 2) return null
  const [a, b] = pair
  if (typeof a !== 'string' || typeof b !== 'string' || !KNOB_NAME.test(a) || !KNOB_NAME.test(b)) return null
  return { capTable: [a, b], caps: sanitizePairTable(rule.caps, sanitizeCap) }
}

function sanitizeColor(color: unknown): PrepaintNumericRule['color'] | null {
  if (!isRecord(color) || typeof color.knob !== 'string' || !KNOB_NAME.test(color.knob)) return null
  const rgb = sanitizeTable(color.rgb, (value) => (typeof value === 'string' && RGB_LIST.test(value) ? value : null))
  return { knob: color.knob, rgb }
}

function sanitizeNumericRule(rule: unknown): PrepaintNumericRule | null {
  if (!isRecord(rule) || rule.template !== 'rgba-alpha' || typeof rule.integer !== 'boolean') return null
  if (typeof rule.var !== 'string' || !CSS_VAR_NAME_PATTERN.test(rule.var)) return null
  if (typeof rule.knob !== 'string' || !KNOB_NAME.test(rule.knob)) return null
  const { min, max, scale } = rule
  if (!isFiniteNumber(min) || !isFiniteNumber(max) || !isFiniteNumber(scale) || min > max || scale <= 0) return null
  const caps = sanitizeCapTable(rule)
  const color = sanitizeColor(rule.color)
  if (caps === null || color === null) return null
  return { var: rule.var, knob: rule.knob, min, max, integer: rule.integer, scale, ...caps, color, template: 'rgba-alpha' }
}

function hasKey(table: unknown, key: unknown): boolean {
  return isRecord(table) && typeof key === 'string' && Object.prototype.hasOwnProperty.call(table, key)
}

function numericFits(rule: PrepaintNumericRule, look: UnknownRecord): boolean {
  const value = look[rule.knob]
  if (!isFiniteNumber(value) || value < rule.min || value > rule.max) return false
  if (rule.integer && value % 1 !== 0) return false
  if (!hasKey(rule.color.rgb, look[rule.color.knob])) return false
  if (rule.capTable === null) return true
  const [rowKnob, columnKnob] = rule.capTable
  if (rule.caps === null || !hasKey(rule.caps, look[rowKnob])) return false
  return hasKey(rule.caps[String(look[rowKnob])], look[columnKnob])
}

/** Whether a look names an option of every table and fits every numeric rule. */
export function lookFitsPrepaint(table: PrepaintTable, look: UnknownRecord): boolean {
  const knobs = Object.entries(table.knobs).every(([knob, options]) => hasKey(options, look[knob]))
  const pairs = table.pairs.every((pair) => hasKey(pair.table, look[pair.a]) && hasKey(pair.table[String(look[pair.a])], look[pair.b]))
  return knobs && pairs && table.numeric.every((rule) => numericFits(rule, look))
}

function sanitizeDefaults(defaults: unknown): Readonly<Record<string, string | number>> {
  if (!isRecord(defaults)) return {}
  return Object.fromEntries(
    Object.entries(defaults).filter(
      (entry): entry is [string, string | number] => KNOB_NAME.test(entry[0]) && (typeof entry[1] === 'string' || isFiniteNumber(entry[1])),
    ),
  )
}

export function sanitizePrepaint(prepaint: unknown): PrepaintTable | null {
  if (!isRecord(prepaint)) return null
  const knobs = Object.fromEntries(
    Object.entries(isRecord(prepaint.knobs) ? prepaint.knobs : {})
      .filter(([knob]) => KNOB_NAME.test(knob))
      .map(([knob, options]) => [knob, sanitizeTable(options, sanitizeVars)] as const),
  )
  const list = (value: unknown): unknown[] => (Array.isArray(value) ? value : [])
  const table: PrepaintTable = {
    knobs,
    pairs: list(prepaint.pairs).map(sanitizePair).filter((pair): pair is PrepaintPair => pair !== null),
    numeric: list(prepaint.numeric).map(sanitizeNumericRule).filter((rule): rule is PrepaintNumericRule => rule !== null),
    defaults: sanitizeDefaults(prepaint.defaults),
  }
  return lookFitsPrepaint(table, table.defaults) ? table : null
}

export function sanitizeLookStorage(storage: unknown): ThemeLookStorage | null {
  if (!isRecord(storage)) return null
  const { pointerKey, scopedPrefix, fallbackKey } = storage
  const isKey = (key: unknown): key is string => typeof key === 'string' && STORAGE_KEY.test(key)
  return isKey(pointerKey) && isKey(scopedPrefix) && isKey(fallbackKey) ? { pointerKey, scopedPrefix, fallbackKey } : null
}

export function sanitizeLookGuard(guard: unknown): ThemeLookGuard | null {
  if (!isRecord(guard)) return null
  const { schema, iface } = guard
  const fits = (value: unknown): value is string => typeof value === 'string' && value.length > 0 && value.length <= GUARD_MAX_LENGTH
  return fits(schema) && fits(iface) ? { schema, iface } : null
}

const TONE_POLICIES: readonly ThemeTonePolicy[] = ['from-look', 'user', 'light', 'dark']
const TONES: readonly ThemeTone[] = ['light', 'dark']
/** A boot whose tone is not one of the above leaves the tone to the user. */
const FALLBACK_TONE: ThemeBoot['tone'] = { policy: 'user', value: null }

/** A tone, strictly: 'light' or 'dark', anything else → none. */
export function parseThemeTone(value: unknown): ThemeTone | null {
  return TONES.find((tone) => tone === value) ?? null
}

/** The tone as a strict pair: a known policy, and 'light', 'dark' or null. */
export function sanitizeTone(tone: unknown): ThemeBoot['tone'] {
  if (!isRecord(tone)) return FALLBACK_TONE
  const policy = TONE_POLICIES.find((known) => known === tone.policy)
  if (policy === undefined) return FALLBACK_TONE
  return { policy, value: parseThemeTone(tone.value) }
}

/** The boot as it may be written into the page: every part checked, anything failing dropped. */
export function sanitizeThemeBoot(boot: ThemeBoot): ThemeBoot {
  const addClasses = sanitizeClasses(boot.addClasses)
  return {
    addClasses,
    removeClasses: sanitizeClasses(boot.removeClasses).filter((name) => !addClasses.includes(name)),
    tone: sanitizeTone(boot.tone),
    prepaint: sanitizePrepaint(boot.prepaint),
    lookStorage: sanitizeLookStorage(boot.lookStorage),
    lookGuard: sanitizeLookGuard(boot.lookGuard),
    integrity: sanitizeIntegrity(boot.integrity),
    entry: sanitizeInstalledEntry(boot.entry),
  }
}
