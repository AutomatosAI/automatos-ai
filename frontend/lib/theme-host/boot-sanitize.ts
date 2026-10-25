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
 *
 * PRD-261 A-19c (host 1.1.0): a prepaint table's tone rule is kept only when it names a choice
 * knob of what is left of the table and gives every one of its choices 'light' or 'dark' (else
 * the rule alone is dropped); the boot's presets (`looks`) are kept at most BOOT_PRESETS_MAX, each
 * a flat look document of at most BOOT_PRESET_MAX_BYTES with a look id, the default first. The
 * server checked both already (css_value.py, serve.py boot_looks); this gate never trusts it.
 */
import { LOOK_ID_PATTERN } from '@/lib/look-kit/validate'

import { HTML_CLASS_PATTERN } from './constants'
import { sanitizeIntegrity } from './import-map'
import { sanitizeInstalledEntry } from './installed-entry'
import type { ThemeTonePolicy } from './registry-types'
import type {
  PrepaintNumericRule,
  PrepaintPair,
  PrepaintPairTable,
  PrepaintTable,
  PrepaintToneRule,
  PrepaintVars,
  ThemeBoot,
  ThemeBootLooks,
  ThemeBootPreset,
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
 * A look storage key: an installed theme's `automatos-theme.<publisher>/<name>.look.current`
 * (lib/theme-host/looks/storage.ts themeLookKeys, W-A6b gap G2), whose theme id holds a '/' and
 * may be 102 characters long.
 */
const STORAGE_KEY = /^[A-Za-z0-9._:/-]{1,160}$/
const GUARD_MAX_LENGTH = 128
/** Caps are alphas. */
const CAP_MIN = 0
const CAP_MAX = 1
/**
 * contract.py BOOT_PRESETS_MAX and BOOT_PRESET_MAX_BYTES (validator-parity.test.ts holds them
 * equal). A preset is measured here as its JSON text's length, which is never more than the
 * file's bytes, so the browser is never stricter than the server.
 */
export const BOOT_PRESETS_MAX = 12
export const BOOT_PRESET_MAX_BYTES = 4096

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

/**
 * A tone rule over what is left of the table: its knob a choice knob (never a numeric rule's) and
 * every one of that knob's choices given 'light' or 'dark'; null otherwise. Only the knob's own
 * choices are kept.
 */
export function sanitizeToneRule(rule: unknown, table: PrepaintTable): PrepaintToneRule | null {
  if (!isRecord(rule)) return null
  const knob = rule.knob
  const values = rule.values
  if (typeof knob !== 'string' || !isRecord(values) || !hasKey(table.knobs, knob)) return null
  if (table.numeric.some((numeric) => numeric.knob === knob)) return null
  const choices = Object.keys(table.knobs[knob])
  const mapped = choices.flatMap((choice): [string, ThemeTone][] => {
    const tone = hasKey(values, choice) ? parseThemeTone(values[choice]) : null
    return tone === null ? [] : [[choice, tone]]
  })
  if (choices.length === 0 || mapped.length !== choices.length) return null
  return { knob, values: Object.fromEntries(mapped) }
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
  if (!lookFitsPrepaint(table, table.defaults)) return null
  const tone = sanitizeToneRule(prepaint.tone, table)
  return tone === null ? table : { ...table, tone }
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

function isPresetEntry(entry: [string, unknown]): entry is [string, string | number] {
  return typeof entry[1] === 'string' || isFiniteNumber(entry[1])
}

/** One preset: a flat look document (text and numbers only) with a look id, within the size cap; a copy. */
function sanitizePreset(preset: unknown): ThemeBootPreset | null {
  if (!isRecord(preset)) return null
  const id = preset.id
  if (typeof id !== 'string' || !LOOK_ID_PATTERN.test(id)) return null
  const entries = Object.entries(preset)
  const kept = entries.filter(isPresetEntry)
  if (kept.length !== entries.length || JSON.stringify(preset).length > BOOT_PRESET_MAX_BYTES) return null
  return Object.freeze(Object.fromEntries(kept))
}

/**
 * The boot's presets (A-19c): of the first BOOT_PRESETS_MAX, the ones that pass, each id once;
 * null when none passes or the default look is not the first of them (the server puts it first).
 */
export function sanitizeBootLooks(looks: unknown): ThemeBootLooks | null {
  if (!isRecord(looks)) return null
  const list = looks.presets
  const defaultLook = looks.defaultLook
  if (!Array.isArray(list) || typeof defaultLook !== 'string') return null
  const passed = list.slice(0, BOOT_PRESETS_MAX).map(sanitizePreset).filter((preset): preset is ThemeBootPreset => preset !== null)
  const presets = passed.filter((preset, index) => passed.findIndex((other) => other.id === preset.id) === index)
  if (presets.length === 0 || presets[0].id !== defaultLook) return null
  return { defaultLook, presets }
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
    looks: sanitizeBootLooks(boot.looks),
  }
}

/**
 * The boot route's answer (GET /api/themes/boot/{installId}) as a boot the page may use, whoever
 * asked: the root layout on the server (boot-server.ts) or the browser when the page's boot carries
 * no entry (boot-client.ts). Throws when it is not an object.
 */
export function parseInstalledBoot(value: unknown): ThemeBoot {
  if (!isRecord(value)) throw new Error('the boot route did not answer with an object')
  // Every field is checked by sanitizeThemeBoot, which reads each one as unknown.
  return sanitizeThemeBoot(value as unknown as ThemeBoot)
}
