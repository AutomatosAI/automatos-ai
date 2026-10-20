import { UNSAFE_TEXT } from '@/lib/text/plain-line'

/**
 * PRD-260 — the one gate for a look from anywhere outside the code: storage, an
 * imported file, pasted JSON. Pure. Errors are plain sentences a designer can
 * act on; they never echo long input back.
 *
 * PRD-261 A-13a — moved from lib/looks/validate.ts into the core look runtime
 * (PA-9) and made schema-driven: the look's schema id, its interface (when it
 * names one) and its knobs come from the theme's LookSpec, the same shape as a
 * checked looks/schema.json (lib/theme-host/look-schema.ts LookSchema), so no
 * theme's interface or catalogue is written here. A look is
 * `{schema, id, name, sub, interface (when the spec names one), <every knob>}`.
 */

/** For `name` and `sub`, counted in characters (code points). */
export const LOOK_TEXT_MAX = 40
/** Letters, digits and dashes, so an id is safe as a storage key or a React key. */
export const LOOK_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/

export interface LookChoiceSpec {
  readonly kind: 'choice'
  readonly choices: readonly string[]
  /** How an error names the choices ("the catalogue's backgrounds"); "its choices" when absent. */
  readonly listName?: string
}

export interface LookNumberSpec {
  readonly kind: 'number'
  readonly min: number
  readonly max: number
  readonly integer?: boolean
}

export type LookKnobSpec = LookChoiceSpec | LookNumberSpec

/** What a theme's looks are: a checked looks/schema.json (lib/theme-host/look-schema.ts) fits it as it is. */
export interface LookSpec {
  readonly lookSchema: string
  readonly interface: string | null
  readonly knobs: ReadonlyMap<string, LookKnobSpec>
}

/** A look that passed: the header fields, then every knob, in the spec's order. */
export type LookFields = Readonly<Record<string, string | number>>

export type LookCheck =
  | { readonly ok: true; readonly look: LookFields }
  | { readonly ok: false; readonly errors: readonly string[] }

const ECHO_MAX = 40
/** Unknown fields are named one by one up to this many, then one line counts the rest. */
const UNKNOWN_FIELDS_NAMED = 5
const ECHO_REPLACEMENT = '?'
const BASE_FIELDS: readonly string[] = ['schema', 'id', 'name', 'sub']
const INTERFACE_FIELD = 'interface'
const DEFAULT_LIST_NAME = 'its choices'
/** A schema id ends in its version: 'automatos-interface-look/v1' → 'v1'. */
const SCHEMA_VERSION = /\/(v\d+)$/

/** Every character that doesn't belong in a one-line name (lib/text/plain-line.ts). */
const UNSAFE_TEXT_ALL = new RegExp(UNSAFE_TEXT.source, 'gu')
/** Blank to the eye: spaces, format characters and the Hangul fillers. */
const INVISIBLE = /[\p{Z}\p{Cf}\u115F\u1160\u3164\uFFA0]/gu

type Fields = Readonly<Record<string, unknown>>
type Checked<T> = { readonly value: T; readonly error: null } | { readonly value: null; readonly error: string }

function pass<T>(value: T): Checked<T> {
  return { value, error: null }
}

function fail(error: string): { readonly value: null; readonly error: string } {
  return { value: null, error }
}

function isRecord(value: unknown): value is Fields {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** Input as an error message may quote it: unsafe characters shown as '?', at most ECHO_MAX characters. */
function echo(text: string): string {
  const chars = Array.from(text.replace(UNSAFE_TEXT_ALL, ECHO_REPLACEMENT))
  return chars.length > ECHO_MAX ? `${chars.slice(0, ECHO_MAX).join('')}…` : chars.join('')
}

/** The fields a look of this spec has, in order. */
function lookFieldsOf(spec: LookSpec): readonly string[] {
  const header = spec.interface === null ? BASE_FIELDS : [...BASE_FIELDS, INTERFACE_FIELD]
  return [...header, ...Array.from(spec.knobs.keys())]
}

function unknownFieldErrors(spec: LookSpec, input: Fields): string[] {
  const known = lookFieldsOf(spec)
  const unknown = Object.keys(input).filter((key) => !known.includes(key))
  const named = unknown.slice(0, UNKNOWN_FIELDS_NAMED).map((key) => `"${echo(key)}" isn't part of a Look. Remove it.`)
  const rest = unknown.length - named.length
  if (rest === 0) return named
  const more = rest === 1 ? "1 more field that isn't" : `${rest} more fields that aren't`
  return [...named, `…and ${more} part of a Look.`]
}

/** 'a v1 Look' when the schema id carries its version, else 'a Look for this theme'. */
function lookKind(spec: LookSpec): string {
  const version = SCHEMA_VERSION.exec(spec.lookSchema)?.[1]
  return version === undefined ? 'a Look for this theme' : `a ${version} Look`
}

function headerErrors(spec: LookSpec, input: Fields): string[] {
  const errors: string[] = []
  if (input.schema !== spec.lookSchema) {
    errors.push(`This isn't ${lookKind(spec)}: "schema" must be "${spec.lookSchema}".`)
  }
  if (spec.interface !== null && input.interface !== spec.interface) {
    errors.push(`This Look is for another interface: "interface" must be "${spec.interface}".`)
  }
  return errors
}

function checkId(value: unknown): Checked<string> {
  if (typeof value === 'string' && LOOK_ID_PATTERN.test(value)) return pass(value)
  return fail('"id" must be 1 to 64 letters, digits, dashes or underscores, starting with a letter or digit.')
}

function checkText(field: 'name' | 'sub', value: unknown): Checked<string> {
  if (typeof value !== 'string') return fail(`"${field}" must be text.`)
  const trimmed = value.trim()
  if (field === 'name' && trimmed.replace(INVISIBLE, '').length === 0) return fail('"name" can\'t be empty.')
  if (Array.from(trimmed).length > LOOK_TEXT_MAX) return fail(`"${field}" must be at most ${LOOK_TEXT_MAX} characters.`)
  if (UNSAFE_TEXT.test(trimmed)) return fail(`"${field}" must be one line of plain text.`)
  return pass(trimmed)
}

function checkChoice(name: string, knob: LookChoiceSpec, value: unknown): Checked<string> {
  if (typeof value === 'string' && knob.choices.includes(value)) return pass(value)
  return fail(`"${name}" must be one of ${knob.listName ?? DEFAULT_LIST_NAME}: ${knob.choices.join(', ')}.`)
}

function isInRange(knob: LookNumberSpec, value: number): boolean {
  const inRange = Number.isFinite(value) && value >= knob.min && value <= knob.max
  return inRange && (knob.integer !== true || Number.isInteger(value))
}

function checkNumber(name: string, knob: LookNumberSpec, value: unknown): Checked<number> {
  if (typeof value === 'number' && isInRange(knob, value)) return pass(value)
  const kind = knob.integer === true ? 'a whole number' : 'a number'
  return fail(`"${name}" must be ${kind} from ${knob.min} to ${knob.max}.`)
}

function checkKnob(name: string, knob: LookKnobSpec, value: unknown): Checked<string | number> {
  return knob.kind === 'choice' ? checkChoice(name, knob, value) : checkNumber(name, knob, value)
}

type CheckedField = readonly [string, Checked<string | number>]
type Entry = readonly [string, string | number]

interface CheckedLook {
  /** id, name and sub. */
  readonly head: readonly CheckedField[]
  /** Every knob, in the spec's order. */
  readonly knobs: readonly CheckedField[]
}

function checkFields(spec: LookSpec, input: Fields): CheckedLook {
  return {
    head: [['id', checkId(input.id)], ['name', checkText('name', input.name)], ['sub', checkText('sub', input.sub)]],
    knobs: Array.from(spec.knobs, ([name, knob]): CheckedField => [name, checkKnob(name, knob, input[name])]),
  }
}

function errorsOf(fields: readonly CheckedField[]): string[] {
  return fields.flatMap(([, result]) => (result.error === null ? [] : [result.error]))
}

function valuesOf(fields: readonly CheckedField[]): Entry[] {
  return fields.flatMap(([name, result]): Entry[] => (result.error === null ? [[name, result.value]] : []))
}

/** A fresh look with only the spec's fields, in its order, `name` and `sub` trimmed. */
function toLook(spec: LookSpec, checked: CheckedLook): LookFields {
  const header: readonly Entry[] = spec.interface === null ? [] : [[INTERFACE_FIELD, spec.interface]]
  const schema: Entry = ['schema', spec.lookSchema]
  return Object.fromEntries([schema, ...valuesOf(checked.head), ...header, ...valuesOf(checked.knobs)])
}

/** `input` checked against `spec`: the look, or every problem at once as plain sentences. */
export function validateLook(spec: LookSpec, input: unknown): LookCheck {
  if (!isRecord(input)) return { ok: false, errors: ['A Look must be a JSON object.'] }
  const checked = checkFields(spec, input)
  const fieldErrors = [...errorsOf(checked.head), ...errorsOf(checked.knobs)]
  const errors = [...unknownFieldErrors(spec, input), ...headerErrors(spec, input), ...fieldErrors]
  return errors.length === 0 ? { ok: true, look: toLook(spec, checked) } : { ok: false, errors }
}
