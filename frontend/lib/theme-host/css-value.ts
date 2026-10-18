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

export const PREPAINT_ERROR_CODES: readonly string[] = ['BAD_PREPAINT', 'BAD_CSS_VALUE']
export const PREPAINT_PATH = 'boot/prepaint.json'
const CSS_VALUE_MAX = 200
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
  if (name === null || !CUSTOM_PROPERTY_PATTERN.test(name[1])) return 'var() must name a custom property such as --pl-accent'
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
