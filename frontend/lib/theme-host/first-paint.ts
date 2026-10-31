/**
 * PRD-261 — what the browser would silently leave out of a theme's look files: the TypeScript twin of
 * orchestrator/services/themes/first_paint.py, which the server's package reader runs on upload.
 *
 * It asks the browser itself: the table as boot-sanitize.ts `sanitizePrepaint` keeps it, and the look spec
 * looks/spec.ts `lookSpecOf` builds from the kept table, the knobs a look must carry. Every refusal is the
 * server's, code, path and sentence (the corpus's `firstPaint` cases, which validator-parity.test.ts and
 * test_prd261_theme_package.py both run):
 *
 * - what the gate drops of the table (css-value.ts `prepaintDropErrors`: BAD_PREPAINT, BAD_CSS_VALUE);
 * - defaults that don't fit what it keeps (PREPAINT_DEFAULTS_UNFIT): the whole table is dropped, so nothing
 *   is painted from it and the theme has no looks;
 * - a looks/schema.json knob the look spec lacks (KNOB_NOT_PAINTED), and a look-spec knob the schema lacks
 *   (KNOB_NOT_IN_SCHEMA): either way every look fails, the presets included.
 *
 * Only a table that passes `prepaintErrors` is looked at. The host never runs this: it is held to the
 * server's verdicts so the two can't drift.
 */
import { sanitizePrepaint } from './boot-sanitize'
import type { PrepaintTable } from './boot-types'
import { echo, isRecord, own, prepaintDropErrors, prepaintErrors, PREPAINT_PATH, themeError, type JsonRecord, type ThemeError } from './css-value'
import { LOOK_SCHEMA_PATH, type LookSchema } from './look-schema'
import { lookSpecOf } from './looks/spec'

export const KNOB_NOT_PAINTED = 'KNOB_NOT_PAINTED'
export const KNOB_NOT_IN_SCHEMA = 'KNOB_NOT_IN_SCHEMA'
export const PREPAINT_DEFAULTS_UNFIT = 'PREPAINT_DEFAULTS_UNFIT'
/** first_paint.py FIRST_PAINT_ERROR_CODES. */
export const FIRST_PAINT_ERROR_CODES: readonly string[] = [
  'BAD_PREPAINT', 'BAD_CSS_VALUE', PREPAINT_DEFAULTS_UNFIT, KNOB_NOT_PAINTED, KNOB_NOT_IN_SCHEMA,
]
const SPEC = `the browser builds a theme's looks from ${PREPAINT_PATH} alone (a choice knob for each table under "knobs", `
  + 'a number knob for each "numeric" rule)'

/** The look spec's knobs, in lookSpecOf's order. */
export function specKnobs(table: PrepaintTable, schema: LookSchema): string[] {
  return Array.from(lookSpecOf(table, { lookSchema: schema.lookSchema, interface: schema.interface }).knobs.keys())
}

function notPainted(name: string): ThemeError {
  const knob = echo(name)
  const where = `${LOOK_SCHEMA_PATH}#knobs.${knob}`
  return themeError(KNOB_NOT_PAINTED, where, (
    `${where}: "${knob}" is a knob the look table never paints: ${SPEC}, so its look spec leaves "${knob}" out `
    + 'and every look fails, the presets included. Paint it there, or take it out of the schema.'
  ))
}

function notInSchema(name: string, table: PrepaintTable, document: JsonRecord): ThemeError {
  const knob = echo(name)
  const rules = own(document, 'numeric')
  const at = Math.max(0, (Array.isArray(rules) ? rules : []).findIndex((rule: unknown) => isRecord(rule) && own(rule, 'knob') === name))
  const where = Object.prototype.hasOwnProperty.call(table.knobs, name) ? `${PREPAINT_PATH}#knobs.${knob}` : `${PREPAINT_PATH}#numeric[${at}].knob`
  return themeError(KNOB_NOT_IN_SCHEMA, where, (
    `${where}: "${knob}" is a knob the table paints that ${LOOK_SCHEMA_PATH} doesn't declare: ${SPEC}, so its look spec `
    + `needs "${knob}" in every look and no look may carry it, and every look fails, the presets included. Declare `
    + 'it in the schema, or take it out of the table.'
  ))
}

function unfit(): ThemeError {
  const where = `${PREPAINT_PATH}#defaults`
  return themeError(PREPAINT_DEFAULTS_UNFIT, where, (
    `${where}: "defaults" don't fit what the browser's first paint keeps of the table (an option of every knob `
    + "table, a cell of every pair table, and for every numeric rule a value in range and a colour it keeps), so it "
    + 'drops the whole table: nothing is painted from it and the theme has no looks.'
  ))
}

/** first_paint.py `first_paint_table_errors`: every refusal above for one table and the theme's checked look schema. */
export function firstPaintErrors(schema: LookSchema | null, table: unknown): ThemeError[] {
  if (!isRecord(table) || prepaintErrors(table).length > 0) return []
  const dropped = prepaintDropErrors(table)
  const kept = sanitizePrepaint(table)
  if (kept === null) return [...dropped, unfit()]
  if (schema === null) return dropped
  const spec = specKnobs(kept, schema)
  const unpainted = Array.from(schema.knobs.keys()).filter((name) => !spec.includes(name)).map(notPainted)
  const unknown = spec.filter((name) => !schema.knobs.has(name)).map((name) => notInSchema(name, kept, table))
  return [...dropped, ...unpainted, ...unknown]
}
