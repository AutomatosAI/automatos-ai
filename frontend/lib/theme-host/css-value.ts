/**
 * PRD-261 A-06b — the value checks every theme validator shares, and the CSS-value grammar for a
 * theme's first-paint values (`boot/prepaint.json`).
 *
 * The TypeScript twin of `orchestrator/services/themes/css_value.py` and of the error shape and
 * plain-text rules in `contract.py`. `__tests__/validator-parity.test.ts` holds both to the same
 * corpus (`orchestrator/tests/fixtures/themes/cases.json`), so a value the server refuses is
 * refused here too, with the same code.
 *
 * The root layout's one constant script applies prepaint values with `style.setProperty` before
 * React runs (A-03), so a value must never carry a URL, a quote, markup or anything that could
 * end a declaration. A value passes when it is, end to end, a run of numbers with a unit from
 * `UNITS`, `#` colours of 3, 4, 6 or 8 hex digits, identifiers, and calls to `FUNCTIONS` nested at
 * most `MAX_FUNCTION_DEPTH` deep (a `var()` naming a custom property first), separated by spaces,
 * commas or `/`. Nothing here throws on bad input.
 *
 * PRD-261 A-19c (host 1.1.0): a prepaint table may carry a tone rule, `tone: {knob, values}`. The
 * knob must be a choice knob of the table (a key of `knobs`, not a `numeric` rule's knob) and
 * `values` must give every one of its choices 'light' or 'dark', nothing else; otherwise
 * BAD_TONE_RULE, with the sentence the server gives (the corpus pins a few).
 *
 * The browser's first paint has its own, smaller gate, kept here as the one TypeScript copy:
 * boot-sanitize.ts imports `isBootValue`, and css_value.py is its Python twin. Only the functions in
 * BOOT_VALUE_FUNCTIONS (no gradient, no color-mix()), the characters in BOOT_VALUE_CHARS and at most
 * PREPAINT_VALUE_MAX UTF-16 units pass it. A value it drops vanishes with no error, and so does every
 * value built on it with var(), so the server's reader refuses one (`prepaintDropErrors`, BAD_CSS_VALUE,
 * the sentence the server gives; the corpus's prepaintDrops cases hold both languages to it).
 */
import { UNSAFE_TEXT } from '@/lib/text/plain-line'

/** One refusal: a code, where it is (a file or a dotted field) and a sentence. */
export interface ThemeError {
  readonly code: string
  readonly path: string
  readonly message: string
}

export type JsonRecord = Readonly<Record<string, unknown>>

/** Messages never echo more than this many characters of the input back. */
const ECHO_MAX = 40
const ECHO_REPLACEMENT = '?'
const ELLIPSIS = '…'

export function themeError(code: string, path: string, message: string): ThemeError {
  return Object.freeze({ code, path, message })
}

/** A JSON object (not an array, not null). */
export function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** True when `record` has `key` itself: never a name it inherits, such as `constructor`. */
export function hasOwn(record: JsonRecord, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(record, key)
}

/** `record[key]` when `record` has `key` itself, else undefined (Python's `dict.get`). */
export function own(record: JsonRecord, key: string): unknown {
  return hasOwn(record, key) ? record[key] : undefined
}

/** `value` is one of `list`'s strings (Python's `value in tuple`). */
export function isOneOf(list: readonly string[], value: unknown): boolean {
  return typeof value === 'string' && list.includes(value)
}

/** The text's characters as code points, as Python counts them. */
function codePoints(text: string): string[] {
  return Array.from(text)
}

/** `text` as a message may quote it: unsafe characters as '?', at most ECHO_MAX characters. */
export function echo(text: string): string {
  const chars = codePoints(text)
  const shown = chars.slice(0, ECHO_MAX).map((char) => (UNSAFE_TEXT.test(char) ? ECHO_REPLACEMENT : char)).join('')
  return chars.length > ECHO_MAX ? `${shown}${ELLIPSIS}` : shown
}

/**
 * Why `value` is not one line of plain text of 1 to `maxChars` characters, or null. Nothing is
 * trimmed: the value is stored as given, so the two validators can't disagree about trimming.
 */
export function plainTextProblem(value: unknown, field: string, maxChars: number): string | null {
  if (typeof value !== 'string') return `"${field}" must be text.`
  if (/^\s*$/.test(value)) return `"${field}" can't be empty.`
  if (codePoints(value).length > maxChars) return `"${field}" must be at most ${maxChars} characters.`
  return UNSAFE_TEXT.test(value) ? `"${field}" must be one line of plain text.` : null
}

/** A JSON number with no fraction (8 and 8.0 alike); never a boolean. */
export function isWholeNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value)
}

/** A JSON number that is finite and not a boolean. */
export function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

/** `code` for every key of `record` outside `known`; `where` is the record's dotted path. */
export function unknownFieldErrors(
  record: JsonRecord,
  known: readonly string[],
  code: string,
  where: string,
  owner: string,
): ThemeError[] {
  return Object.keys(record)
    .filter((key) => !known.includes(key))
    .map((key) => {
      const field = where ? `${where}.${echo(key)}` : echo(key)
      return themeError(code, field, `"${echo(key)}" isn't part of ${where || owner}. Remove it.`)
    })
}

/** One refusal when `refused`, else none: the shape every check below returns. */
export function errorIf(refused: boolean, code: string, path: string, message: string): ThemeError[] {
  return refused ? [themeError(code, path, message)] : []
}

// ---------------------------------------------------------------------------
// The CSS-value grammar
// ---------------------------------------------------------------------------

export const PREPAINT_ERROR_CODES: readonly string[] = ['BAD_PREPAINT', 'BAD_CSS_VALUE', 'BAD_TONE_RULE']
export const PREPAINT_PATH = 'boot/prepaint.json'
/**
 * contract.py PREPAINT_VALUE_MAX (validator-parity.test.ts holds them equal): the longest prepaint value, the
 * grammar's cap and the browser's first-paint gate's (boot-sanitize.ts, through bootValueProblem below).
 */
export const PREPAINT_VALUE_MAX = 200
const CSS_VALUE_MAX = PREPAINT_VALUE_MAX
const MAX_FUNCTION_DEPTH = 4
const PREPAINT_MAX_ERRORS = 20
const CUSTOM_PROPERTY_PREFIX = '--'
export const CUSTOM_PROPERTY_PATTERN = /^--[a-z][a-z0-9-]{1,63}$/
/** What a prepaint table may name besides custom properties: knob keys, table names, templates. */
const PREPAINT_KEY_PATTERN = /^[A-Za-z0-9_.-]{1,64}$/
const VAR_NAME = /^[ ]*(--[A-Za-z0-9_-]*)/
const VAR_FUNCTION = 'var'

const FUNCTIONS: ReadonlySet<string> = new Set([
  'rgb', 'rgba', 'hsl', 'hsla', 'var', 'color-mix', 'linear-gradient', 'radial-gradient',
  'hue-rotate', 'saturate', 'brightness', 'grayscale', 'contrast', 'invert', 'sepia', 'opacity',
  'blur', 'drop-shadow',
])
const UNITS: ReadonlySet<string> = new Set([
  '%', 'px', 'rem', 'em', 'ch', 'ex', 'vh', 'vw', 'vmin', 'vmax', 'dvh', 'svh', 'lvh',
  'deg', 'rad', 'turn', 's', 'ms', 'fr',
])
const HEX_LENGTHS: ReadonlySet<number> = new Set([3, 4, 6, 8])

/** One token of a value (css_value.py TOKEN, alternative for alternative); anything else is refused. */
const TOKEN_SOURCE = [
  '(?<space>[ ]+)',
  '(?<fn>-{0,2}[A-Za-z_][A-Za-z0-9_-]*)\\(',
  '(?<number>[+-]?(?:[0-9]+(?:\\.[0-9]+)?|\\.[0-9]+))(?<unit>%|[a-z]+)?',
  '(?<hash>#[0-9A-Fa-f]+)',
  '(?<ident>-{0,2}[A-Za-z_][A-Za-z0-9_-]*)',
  '(?<close>\\))',
  '(?<separator>[,/])',
].join('|')

type TokenGroups = Readonly<Record<string, string | undefined>>

/** Why one matched token is refused, or null. */
function tokenProblem(groups: TokenGroups): string | null {
  const { unit, hash, fn } = groups
  if (unit !== undefined && !UNITS.has(unit)) return `the unit "${unit}" is not one a theme may use`
  if (hash !== undefined && !HEX_LENGTHS.has(hash.length - 1)) return 'a # colour has 3, 4, 6 or 8 hex digits'
  if (fn !== undefined && !FUNCTIONS.has(fn.toLowerCase())) return `"${echo(fn)}()" is not a function a theme may use`
  return null
}

/** var() must start with a custom property name. */
function varProblem(value: string, at: number): string | null {
  const name = VAR_NAME.exec(value.slice(at))
  if (name === null || !CUSTOM_PROPERTY_PATTERN.test(name[1])) return 'var() must name a custom property such as --theme-accent'
  return null
}

/** The nesting depth after this token, or why it is refused. */
function depthAfter(groups: TokenGroups, value: string, end: number, depth: number): [number, string | null] {
  const fn = groups.fn
  if (fn !== undefined) {
    if (depth >= MAX_FUNCTION_DEPTH) return [depth, `functions nest at most ${MAX_FUNCTION_DEPTH} deep`]
    return [depth + 1, fn.toLowerCase() === VAR_FUNCTION ? varProblem(value, end) : null]
  }
  if (groups.close !== undefined) return [depth - 1, depth > 0 ? null : 'a ) closes nothing']
  return [depth, null]
}

/** Why `value` is not a CSS value a theme may set before first paint, or null. */
export function cssValueProblem(value: unknown): string | null {
  if (typeof value !== 'string' || value.length === 0) return 'a CSS value must be non-empty text'
  if (codePoints(value).length > CSS_VALUE_MAX) return `a CSS value must be at most ${CSS_VALUE_MAX} characters`
  const token = new RegExp(TOKEN_SOURCE, 'y')
  let depth = 0
  while (token.lastIndex < value.length) {
    const at = token.lastIndex
    const match = token.exec(value)
    if (match === null) return `"${echo(String.fromCodePoint(value.codePointAt(at) ?? 0))}" can't appear in a CSS value here`
    const groups: TokenGroups = match.groups ?? {}
    const refusal = tokenProblem(groups)
    if (refusal !== null) return refusal
    const [next, problem] = depthAfter(groups, value, token.lastIndex, depth)
    if (problem !== null) return problem
    depth = next
  }
  return depth === 0 ? null : 'a ( is never closed'
}

/** True when `value` passes the grammar above. */
export function isCssValue(value: unknown): value is string {
  return cssValueProblem(value) === null
}

// ---------------------------------------------------------------------------
// The prepaint table
// ---------------------------------------------------------------------------

/** A node to check: its dotted path, its key in an object (null in a list) and its value. */
type PrepaintNode = readonly [path: string, key: string | null, value: unknown]

/** A non-variable leaf: a finite number, a boolean, null, or text that passes the grammar. */
function scalarError(where: string, value: unknown): ThemeError | null {
  if (typeof value === 'object' || typeof value === 'boolean' || isFiniteNumber(value)) return null
  const problem = typeof value === 'string' ? cssValueProblem(value) : 'only text, numbers, true, false and null'
  return problem === null ? null : themeError('BAD_CSS_VALUE', where, `${where}: ${problem}.`)
}

/** The error for one key and its value, or null. */
function leafError(path: string, key: string, value: unknown): ThemeError | null {
  const where = `${PREPAINT_PATH}#${echo(path)}`
  if (key.startsWith(CUSTOM_PROPERTY_PREFIX)) {
    const problem = CUSTOM_PROPERTY_PATTERN.test(key)
      ? cssValueProblem(value)
      : `"${echo(key)}" is not a custom property name`
    return problem === null ? null : themeError('BAD_CSS_VALUE', where, `${where}: ${problem}.`)
  }
  if (!PREPAINT_KEY_PATTERN.test(key)) {
    return themeError('BAD_PREPAINT', where, `${where}: "${echo(key)}" is not a name a prepaint table may use.`)
  }
  return scalarError(where, value)
}

/** Each child of an object (with its key) or of a list (key null); nothing for a leaf. */
function childrenOf(node: unknown, path: string): PrepaintNode[] {
  if (Array.isArray(node)) return node.map((child, index): PrepaintNode => [`${path}[${index}]`, null, child])
  if (!isRecord(node)) return []
  return Object.keys(node).map((key): PrepaintNode => [path ? `${path}.${key}` : key, key, node[key]])
}

/**
 * Every refused name or value in a prepaint table, at most PREPAINT_MAX_ERRORS. Each `--name` key
 * is a custom property and its value passes the grammar, and every other text passes it too.
 */
export function prepaintErrors(document: unknown): ThemeError[] {
  if (!isRecord(document)) return [themeError('BAD_PREPAINT', PREPAINT_PATH, `${PREPAINT_PATH} must be a JSON object.`)]
  return [...walkErrors(document), ...toneRuleErrors(document)].slice(0, PREPAINT_MAX_ERRORS)
}

/** Every refused name or value the walk of the table finds, at most PREPAINT_MAX_ERRORS. */
function walkErrors(document: JsonRecord): ThemeError[] {
  const errors: ThemeError[] = []
  const stack: PrepaintNode[] = childrenOf(document, '').reverse()
  while (stack.length > 0 && errors.length < PREPAINT_MAX_ERRORS) {
    const [path, key, value] = stack.pop() as PrepaintNode
    const error = key === null ? scalarError(`${PREPAINT_PATH}#${echo(path)}`, value) : leafError(path, key, value)
    if (error !== null) errors.push(error)
    stack.push(...childrenOf(value, path).reverse())
  }
  return errors
}

// ---------------------------------------------------------------------------
// The tone rule (A-19c, host 1.1.0): css_value.py tone_rule_errors, sentence for sentence
// ---------------------------------------------------------------------------

/** contract.py TONE_RULE_KEY, TONE_RULE_FIELDS and TONE_RULE_TONES (validator-parity.test.ts holds them equal). */
export const TONE_RULE_KEY = 'tone'
export const TONE_RULE_FIELDS: readonly string[] = ['knob', 'values']
export const TONE_RULE_TONES: readonly string[] = ['light', 'dark']
const TONE_RULE_CODE = 'BAD_TONE_RULE'
const TONE_RULE_PATH = `${PREPAINT_PATH}#${TONE_RULE_KEY}`
const TONES_TEXT = '"light" or "dark"'

/** One refusal of the tone rule, at `tone<field>`. */
function toneError(field: string, problem: string): ThemeError {
  const where = `${TONE_RULE_PATH}${field}`
  return themeError(TONE_RULE_CODE, where, `${where}: ${problem}.`)
}

/** `{knob: its choices}` for every knob of the table whose options are an object. */
function choiceKnobs(document: JsonRecord): Readonly<Record<string, readonly string[]>> {
  const knobs = own(document, 'knobs')
  if (!isRecord(knobs)) return {}
  const entries = Object.keys(knobs).flatMap((name): [string, string[]][] => {
    const options = knobs[name]
    return isRecord(options) ? [[name, Object.keys(options)]] : []
  })
  return Object.fromEntries(entries)
}

/** The knob of every numeric rule of the table. */
function numberKnobs(document: JsonRecord): ReadonlySet<string> {
  const rules = own(document, 'numeric')
  if (!Array.isArray(rules)) return new Set()
  const knobs = rules.map((rule) => (isRecord(rule) ? own(rule, 'knob') : undefined))
  return new Set(knobs.filter((knob): knob is string => typeof knob === 'string'))
}

/** Why `knob` can't carry the tone rule, or null. */
function toneKnobProblem(document: JsonRecord, knob: string): string | null {
  if (numberKnobs(document).has(knob)) return `"${echo(knob)}" is a number knob; the tone rule needs a choice knob`
  if (!hasOwn(choiceKnobs(document), knob)) return `"${echo(knob)}" is not a knob of this table`
  return null
}

/** Every choice of `knob` mapped to light or dark, and nothing else mapped. */
function toneValueErrors(knob: string, choices: readonly string[], values: JsonRecord): ThemeError[] {
  const unmapped = choices
    .filter((choice) => !hasOwn(values, choice))
    .map((choice) => toneError(`.values.${echo(choice)}`, `the choice "${echo(choice)}" of "${echo(knob)}" has no tone; map it to ${TONES_TEXT}`))
  const wrong = Object.keys(values).flatMap((choice): ThemeError[] => {
    if (!choices.includes(choice)) return [toneError(`.values.${echo(choice)}`, `"${echo(choice)}" is not a choice of "${echo(knob)}"`)]
    return isOneOf(TONE_RULE_TONES, values[choice]) ? [] : [toneError(`.values.${echo(choice)}`, `a tone is ${TONES_TEXT}`)]
  })
  return [...unmapped, ...wrong]
}

/**
 * Why the table's tone rule (`tone`) is refused; empty when it has none or it is valid. The rule
 * names one choice knob of the table and gives each of its choices a tone, so a look runtime can
 * set the tone from the look alone (A-19d2) and every look has one.
 */
export function toneRuleErrors(document: unknown): ThemeError[] {
  if (!isRecord(document) || !hasOwn(document, TONE_RULE_KEY)) return []
  const rule = document[TONE_RULE_KEY]
  if (!isRecord(rule)) return [toneError('', 'the tone rule must be an object with "knob" and "values"')]
  const errors = Object.keys(rule)
    .filter((key) => !TONE_RULE_FIELDS.includes(key))
    .map((key) => toneError(`.${echo(key)}`, `"${echo(key)}" isn't part of the tone rule. Remove it`))
  const knob = own(rule, 'knob')
  const values = own(rule, 'values')
  if (typeof knob !== 'string') return [...errors, toneError('.knob', '"knob" must name a choice knob of this table')]
  const problem = toneKnobProblem(document, knob)
  if (problem !== null) return [...errors, toneError('.knob', problem)]
  if (!isRecord(values)) return [...errors, toneError('.values', `"values" must map every choice of "${echo(knob)}" to ${TONES_TEXT}`)]
  return [...errors, ...toneValueErrors(knob, choiceKnobs(document)[knob], values)]
}

// ---------------------------------------------------------------------------
// The browser's first-paint gate (boot-sanitize.ts) and what it drops: css_value.py, sentence for sentence
// ---------------------------------------------------------------------------

/** css_value.py BOOT_VALUE_CHARS, BOOT_FUNCTION_NAME and BOOT_VALUE_FUNCTIONS (the functions in its order). */
export const BOOT_VALUE_CHARS = /^[A-Za-z0-9#%.,()/ +-]+$/
export const BOOT_FUNCTION_NAME = /([A-Za-z-]+)\(/g
export const BOOT_VALUE_FUNCTIONS: ReadonlySet<string> = new Set([
  'rgb', 'rgba', 'hsl', 'hsla', 'var', 'calc', 'min', 'max', 'clamp',
  'hue-rotate', 'saturate', 'brightness', 'grayscale', 'contrast', 'sepia', 'invert',
])
const GRADIENT_SUFFIX = '-gradient'
const DROPPED = "the browser's first paint drops it, and every value built on it"

/** The first function the gate doesn't run, said as what it is; null when it runs every one. */
function functionProblem(value: string): string | null {
  for (const match of value.matchAll(BOOT_FUNCTION_NAME)) {
    const name = match[1]
    if (BOOT_VALUE_FUNCTIONS.has(name.toLowerCase())) continue
    if (name.toLowerCase().endsWith(GRADIENT_SUFFIX)) return `a gradient, ${echo(name)}()`
    return `a call to ${echo(name)}(), a function the first paint never runs`
  }
  return null
}

/** What `value` is that the browser's first paint drops (boot-sanitize.ts's checks, in its order), or null. */
export function bootValueProblem(value: unknown): string | null {
  if (typeof value !== 'string' || value.length === 0) return 'not text'
  if (value.length > PREPAINT_VALUE_MAX) return `over ${PREPAINT_VALUE_MAX} characters`
  const refused = codePoints(value).find((char) => !BOOT_VALUE_CHARS.test(char))
  if (refused !== undefined) return `text with "${echo(refused)}", a character the first paint refuses`
  if (value.split('(').length !== value.split(')').length) return "text whose brackets don't pair"
  return functionProblem(value)
}

/** True when the browser's first paint keeps `value`: boot-sanitize.ts's gate for every prepaint value. */
export function isBootValue(value: unknown): value is string {
  return bootValueProblem(value) === null
}

/** One cell of the table: its dotted path, the look it paints for and its variables. */
type Cell = readonly [path: string, look: string, variables: unknown]

/** Every option of every table under `knobs`. */
function knobCells(document: JsonRecord): Cell[] {
  const knobs = own(document, 'knobs')
  if (!isRecord(knobs)) return []
  return Object.keys(knobs).flatMap((knob) => {
    const options = knobs[knob]
    if (!isRecord(options)) return []
    return Object.keys(options).map((option): Cell => [`knobs.${knob}.${option}`, `${echo(knob)} "${echo(option)}"`, options[option]])
  })
}

/** Every cell of one pair table whose two knobs are named. */
function pairCells(pair: unknown, index: number): Cell[] {
  if (!isRecord(pair)) return []
  const [a, b, table] = [own(pair, 'a'), own(pair, 'b'), own(pair, 'table')]
  if (typeof a !== 'string' || typeof b !== 'string' || !isRecord(table)) return []
  return Object.keys(table).flatMap((row) => {
    const columns = table[row]
    if (!isRecord(columns)) return []
    const look = (column: string): string => `${echo(a)} "${echo(row)}" and ${echo(b)} "${echo(column)}"`
    return Object.keys(columns).map((column): Cell => [`pairs[${index}].table.${row}.${column}`, look(column), columns[column]])
  })
}

/** A refusal for each variable of one cell whose value the gate drops. */
function cellDrops([path, look, variables]: Cell): ThemeError[] {
  if (!isRecord(variables)) return []
  return Object.keys(variables).flatMap((name) => {
    const problem = name.startsWith(CUSTOM_PROPERTY_PREFIX) ? bootValueProblem(variables[name]) : null
    if (problem === null) return []
    const where = `${PREPAINT_PATH}#${echo(`${path}.${name}`)}`
    return [themeError('BAD_CSS_VALUE', where, `${where}: ${echo(name)} for ${look} is ${problem}: ${DROPPED}.`)]
  })
}

/**
 * Every value of the table's cells the browser's first paint drops, at most PREPAINT_MAX_ERRORS: the cells
 * boot-sanitize.ts sanitizeVars reads, each option of each table under `knobs` and each cell of each pair
 * table. What prepaintErrors refuses is its alone; the server's reader runs this on a table that passed it.
 */
export function prepaintDropErrors(document: unknown): ThemeError[] {
  if (!isRecord(document)) return []
  const pairs = own(document, 'pairs')
  const pairList = Array.isArray(pairs) ? pairs.flatMap((pair: unknown, index) => pairCells(pair, index)) : []
  return [...knobCells(document), ...pairList].flatMap(cellDrops).slice(0, PREPAINT_MAX_ERRORS)
}
