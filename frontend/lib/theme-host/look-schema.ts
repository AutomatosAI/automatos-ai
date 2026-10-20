/**
 * PRD-261 A-06b — a theme's looks, checked against the theme's own `looks/schema.json`.
 *
 * The TypeScript twin of `orchestrator/services/themes/look_schema.py`; `__tests__/validator-
 * parity.test.ts` holds both to the same corpus (`orchestrator/tests/fixtures/themes/cases.json`,
 * `lookSchemas` and `looks`). There is no fixed field list in core (PRD §4.2 "Looks"): each package
 * declares its knobs, and every look (a preset in the package, or one a member saves) is checked
 * against them. A look is `{schema, interface (when the schema names one), id, name, sub,
 * <every knob>}`. Nothing here throws on bad input.
 */
import {
  echo, errorIf, hasOwn, isFiniteNumber, isOneOf, isRecord, isWholeNumber, own, plainTextProblem, themeError,
  unknownFieldErrors, type JsonRecord, type ThemeError,
} from './css-value'
import { LOOK_ID_PATTERN } from '@/lib/look-kit/validate'

export const LOOK_SCHEMA_SCHEMA = 'automatos-look-schema/v1'
export const LOOK_SCHEMA_PATH = 'looks/schema.json'

export const LOOK_ERROR_CODES: readonly string[] = [
  'BAD_LOOK_SCHEMA', 'BAD_KNOB', 'LOOK_NOT_AN_OBJECT', 'LOOK_UNKNOWN_FIELD', 'LOOK_WRONG_SCHEMA', 'LOOK_BAD_ID',
  'LOOK_BAD_TEXT', 'LOOK_BAD_VALUE',
]
const SCHEMA_FIELDS: readonly string[] = ['schema', 'look', 'knobs']
const LOOK_HEADER_FIELDS: readonly string[] = ['schema', 'interface']
const LOOK_TEXT_FIELDS: readonly string[] = ['name', 'sub']
const LOOK_BASE_FIELDS: readonly string[] = ['schema', 'id', 'name', 'sub']
const LOOK_RESERVED_NAMES: readonly string[] = ['schema', 'interface', 'id', 'name', 'sub']
const CHOICE_FIELDS: readonly string[] = ['kind', 'label', 'choices', 'default']
const NUMBER_FIELDS: readonly string[] = ['kind', 'label', 'min', 'max', 'integer', 'default']
const KNOB_KINDS: readonly string[] = ['choice', 'number']
const OPTIONAL_TEXT_FIELD = 'sub'

const SCHEMA_ID_PATTERN = /^[a-z][a-z0-9.-]{0,47}\/v[0-9]{1,3}$/
const INTERFACE_PATTERN = /^[a-z][a-z0-9-]{0,31}$/
const KNOB_NAME_PATTERN = /^[a-z][A-Za-z0-9]{0,31}$/
const CHOICE_PATTERN = /^[a-z0-9][a-z0-9-]{0,31}$/
const MAX_KNOBS = 32
const MAX_CHOICES = 64
const KNOB_NUMBER_LIMIT = 1_000_000
const LABEL_MAX = 40
const LOOK_TEXT_MAX = 40

export interface ChoiceKnob {
  readonly kind: 'choice'
  readonly label?: string
  readonly choices: readonly string[]
  readonly default: string
}

export interface NumberKnob {
  readonly kind: 'number'
  readonly label?: string
  readonly min: number
  readonly max: number
  readonly integer?: boolean
  readonly default: number
}

export type LookKnob = ChoiceKnob | NumberKnob

/** A checked looks/schema.json: the look documents' header and the knobs by name. */
export interface LookSchema {
  readonly lookSchema: string
  readonly interface: string | null
  readonly knobs: ReadonlyMap<string, LookKnob>
}

export interface LookSchemaResult {
  readonly schema: LookSchema | null
  readonly errors: readonly ThemeError[]
}

function single(code: string, path: string, message: string): ThemeError[] {
  return [themeError(code, path, message)]
}

function labelErrors(knob: JsonRecord, where: string): ThemeError[] {
  if (!hasOwn(knob, 'label')) return []
  const problem = plainTextProblem(knob.label, `${where}.label`, LABEL_MAX)
  return problem === null ? [] : single('BAD_KNOB', `${where}.label`, problem)
}

/** 1 to MAX_CHOICES different choice keys. */
function isChoiceList(value: unknown): value is string[] {
  if (!Array.isArray(value) || value.length === 0 || value.length > MAX_CHOICES) return false
  const keys = value.every((choice) => typeof choice === 'string' && CHOICE_PATTERN.test(choice))
  return keys && new Set(value).size === value.length
}

function choiceErrors(knob: JsonRecord, where: string): ThemeError[] {
  const choices = own(knob, 'choices')
  if (!isChoiceList(choices)) {
    return single('BAD_KNOB', `${where}.choices`, (
      `"${where}.choices" must list 1 to ${MAX_CHOICES} different keys: lower-case letters, digits and dashes.`
    ))
  }
  return isOneOf(choices, own(knob, 'default'))
    ? []
    : single('BAD_KNOB', `${where}.default`, `"${where}.default" must be one of its choices.`)
}

/** Why `value` is not a number from `low` to `high` (whole when `integer`), or null. */
function numberProblem(low: number, high: number, integer: boolean, value: unknown): string | null {
  if (!isFiniteNumber(value) || !(low <= value && value <= high)) return `a number from ${low} to ${high}`
  if (integer && !isWholeNumber(value)) return `a whole number from ${low} to ${high}`
  return null
}

/** A finite number within ±KNOB_NUMBER_LIMIT. */
function isKnobBound(value: unknown): value is number {
  return isFiniteNumber(value) && Math.abs(value) <= KNOB_NUMBER_LIMIT
}

function numberErrors(knob: JsonRecord, where: string): ThemeError[] {
  const low = own(knob, 'min')
  const high = own(knob, 'max')
  const integer = hasOwn(knob, 'integer') ? knob.integer : false
  if (!isKnobBound(low) || !isKnobBound(high) || !(low < high)) {
    return single('BAD_KNOB', where, `"${where}" needs "min" below "max", both within ±${KNOB_NUMBER_LIMIT}.`)
  }
  if (typeof integer !== 'boolean') {
    return single('BAD_KNOB', `${where}.integer`, `"${where}.integer" must be true or false.`)
  }
  if (integer && !(isWholeNumber(low) && isWholeNumber(high))) {
    return single('BAD_KNOB', where, `"${where}" is whole numbers only, so "min" and "max" must be too.`)
  }
  if (numberProblem(low, high, integer, own(knob, 'default')) !== null) {
    return single('BAD_KNOB', `${where}.default`, `"${where}.default" must be a number the knob accepts.`)
  }
  return []
}

function knobErrors(name: string, knob: unknown): ThemeError[] {
  const where = `knobs.${echo(name)}`
  if (!KNOB_NAME_PATTERN.test(name) || LOOK_RESERVED_NAMES.includes(name)) {
    return single('BAD_KNOB', where, `"${where}" needs a name of 1 to 32 letters and digits, not a look field.`)
  }
  if (!isRecord(knob) || !isOneOf(KNOB_KINDS, own(knob, 'kind'))) {
    return single('BAD_KNOB', where, `"${where}" must be an object whose "kind" is "choice" or "number".`)
  }
  const isChoice = knob.kind === 'choice'
  const unknown = unknownFieldErrors(knob, isChoice ? CHOICE_FIELDS : NUMBER_FIELDS, 'BAD_KNOB', where, 'a look')
  const shape = isChoice ? choiceErrors(knob, where) : numberErrors(knob, where)
  return [...unknown, ...labelErrors(knob, where), ...shape]
}

function lookHeaderErrors(look: unknown): ThemeError[] {
  if (!isRecord(look)) return single('BAD_LOOK_SCHEMA', 'look', '"look" must be an object naming the looks\' "schema".')
  const errors = unknownFieldErrors(look, LOOK_HEADER_FIELDS, 'BAD_LOOK_SCHEMA', 'look', 'a look')
  const schema = own(look, 'schema')
  const badSchema = typeof schema !== 'string' || !SCHEMA_ID_PATTERN.test(schema)
  const iface = own(look, 'interface')
  const badInterface = hasOwn(look, 'interface') && (typeof iface !== 'string' || !INTERFACE_PATTERN.test(iface))
  return [
    ...errors,
    ...errorIf(badSchema, 'BAD_LOOK_SCHEMA', 'look.schema', '"look.schema" must be an id such as "automatos-interface-look/v1".'),
    ...errorIf(badInterface, 'BAD_LOOK_SCHEMA', 'look.interface', (
      '"look.interface" must be 1 to 32 lower-case letters, digits and dashes.'
    )),
  ]
}

function headerErrors(document: JsonRecord): ThemeError[] {
  const errors = unknownFieldErrors(document, SCHEMA_FIELDS, 'BAD_LOOK_SCHEMA', LOOK_SCHEMA_PATH, 'a look')
  const schema = own(document, 'schema') === LOOK_SCHEMA_SCHEMA
    ? []
    : single('BAD_LOOK_SCHEMA', 'schema', `"schema" must be "${LOOK_SCHEMA_SCHEMA}".`)
  return [...errors, ...schema, ...lookHeaderErrors(own(document, 'look'))]
}

function knobsErrors(knobs: unknown): ThemeError[] {
  if (!isRecord(knobs) || Object.keys(knobs).length === 0 || Object.keys(knobs).length > MAX_KNOBS) {
    return single('BAD_LOOK_SCHEMA', 'knobs', `"knobs" must be an object of 1 to ${MAX_KNOBS} knobs.`)
  }
  return Object.keys(knobs).flatMap((name) => knobErrors(name, knobs[name]))
}

/** A checked looks/schema.json, or the errors saying why it isn't one. */
export function validateLookSchema(document: unknown): LookSchemaResult {
  if (!isRecord(document)) {
    const errors = single('BAD_LOOK_SCHEMA', LOOK_SCHEMA_PATH, `${LOOK_SCHEMA_PATH} must be a JSON object.`)
    return { schema: null, errors }
  }
  const errors = [...headerErrors(document), ...knobsErrors(own(document, 'knobs'))]
  if (errors.length > 0) return { schema: null, errors }
  const look = document.look as JsonRecord
  const knobs = document.knobs as Readonly<Record<string, LookKnob>>
  const schema: LookSchema = Object.freeze({
    lookSchema: look.schema as string,
    interface: hasOwn(look, 'interface') ? (look.interface as string) : null,
    knobs: new Map(Object.keys(knobs).map((name): [string, LookKnob] => [name, knobs[name]])),
  })
  return { schema, errors: [] }
}

/** What a knob's value must be, or null when `value` is that. */
function knobValueProblem(knob: LookKnob, value: unknown): string | null {
  if (knob.kind === 'choice') return isOneOf(knob.choices, value) ? null : `one of ${knob.choices.join(', ')}`
  return numberProblem(knob.min, knob.max, knob.integer === true, value)
}

function lookIdentityErrors(schema: LookSchema, look: JsonRecord): ThemeError[] {
  const wrongSchema = own(look, 'schema') !== schema.lookSchema
  const wrongInterface = schema.interface !== null && own(look, 'interface') !== schema.interface
  const lookId = own(look, 'id')
  const badId = typeof lookId !== 'string' || !LOOK_ID_PATTERN.test(lookId)
  return [
    ...errorIf(wrongSchema, 'LOOK_WRONG_SCHEMA', 'schema', (
      `This look is for another theme: "schema" must be "${schema.lookSchema}".`
    )),
    ...errorIf(wrongInterface, 'LOOK_WRONG_SCHEMA', 'interface', `"interface" must be "${schema.interface}".`),
    ...errorIf(badId, 'LOOK_BAD_ID', 'id', (
      '"id" must be 1 to 64 letters, digits, dashes or underscores, starting with a letter or digit.'
    )),
  ]
}

function lookTextErrors(look: JsonRecord): ThemeError[] {
  return LOOK_TEXT_FIELDS.flatMap((field) => {
    const value = own(look, field)
    const problem = plainTextProblem(value, field, LOOK_TEXT_MAX)
    if (problem === null || (field === OPTIONAL_TEXT_FIELD && value === '')) return []
    return single('LOOK_BAD_TEXT', field, problem)
  })
}

/** Every reason `look` doesn't fit `schema`; empty when it does. `sub` may be empty. */
export function validateLook(schema: LookSchema, look: unknown): ThemeError[] {
  if (!isRecord(look)) return single('LOOK_NOT_AN_OBJECT', '', 'A look must be a JSON object.')
  const header = schema.interface !== null ? [...LOOK_BASE_FIELDS, 'interface'] : LOOK_BASE_FIELDS
  const known = [...header, ...Array.from(schema.knobs.keys())]
  const values = Array.from(schema.knobs).flatMap(([name, knob]) => {
    const problem = knobValueProblem(knob, own(look, name))
    return problem === null ? [] : single('LOOK_BAD_VALUE', name, `"${name}" must be ${problem}.`)
  })
  return [
    ...unknownFieldErrors(look, known, 'LOOK_UNKNOWN_FIELD', '', 'a look'),
    ...lookIdentityErrors(schema, look),
    ...lookTextErrors(look),
    ...values,
  ]
}
