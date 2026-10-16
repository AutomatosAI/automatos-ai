// PRD-259 §3 — the one gate for a skin.json from anywhere outside the code: an imported
// package, the Skin Kit, a cached copy. Pure. The backend's services/skins/validate.py applies
// the same rules with the same codes; orchestrator/tests/fixtures/skins/cases.json proves it.

import {
  FONT_FILE_PATTERN,
  FONT_SLOTS,
  GLOW_MAX,
  GLOW_MIN,
  MASCOT_LIGHTS,
  MASCOT_MODES,
  MASCOT_SHELLS,
  PREVIEW_PATH,
  RADIUS_MAX,
  RADIUS_MIN,
  REQUIRED_FIELDS,
  SHIPPED_FONTS,
  SKIN_BASES,
  SKIN_FIELDS,
  SKIN_SCHEMA_ID,
  SKIN_TONES,
  SLUG_MAX,
  SLUG_PATTERN,
  TEXT_FORBIDDEN,
  TEXT_LIMITS,
  VERSION_MAX,
  VERSION_PATTERN,
  type Skin,
  type SkinError,
  type SkinTone,
  type SkinValidation,
  type TextField,
} from './schema'
import { hasOwn, isRecord, oneOf, skinError, unknownFields, type Fields } from './fields'
import { checkTokens } from './tokens'

const MASCOT_FIELDS = ['mode', 'shell', 'light'] as const
const SHAPE_LIMITS = { radius: [RADIUS_MIN, RADIUS_MAX], glow: [GLOW_MIN, GLOW_MAX] } as const

const matches = (pattern: RegExp, max: number) => (value: unknown) =>
  typeof value === 'string' && [...value].length <= max && pattern.test(value)

/** Structural fields: present means it must pass, else SCHEMA. */
const HEADER_CHECKS: readonly { field: string; ok: (value: unknown) => boolean; message: string }[] = [
  { field: 'schema', ok: (v) => v === SKIN_SCHEMA_ID, message: `"schema" must be "${SKIN_SCHEMA_ID}".` },
  { field: 'slug', ok: matches(SLUG_PATTERN, SLUG_MAX), message: `"slug" must be kebab-case, 1 to ${SLUG_MAX} characters.` },
  { field: 'version', ok: matches(VERSION_PATTERN, VERSION_MAX), message: '"version" must be a semantic version like "1.0.0".' },
  { field: 'base', ok: (v) => oneOf(SKIN_BASES, v), message: `"base" must be one of: ${SKIN_BASES.join(', ')}.` },
  { field: 'preview', ok: (v) => v === PREVIEW_PATH, message: `"preview" must be "${PREVIEW_PATH}".` },
]

function headerErrors(input: Fields): SkinError[] {
  const missing = REQUIRED_FIELDS.filter((field) => !hasOwn(input, field))
    .map((field) => skinError('SCHEMA', field, `"${field}" is required.`))
  const bad = HEADER_CHECKS.filter(({ field, ok }) => hasOwn(input, field) && !ok(input[field]))
    .map(({ field, message }) => skinError('SCHEMA', field, message))
  return [...missing, ...bad]
}

function hasControl(text: string): boolean {
  return [...text].some((char) => {
    const code = char.codePointAt(0) ?? 0
    return code < 0x20 || (code >= 0x7f && code <= 0x9f)
  })
}

function textError(field: TextField, value: unknown): SkinError | null {
  if (typeof value !== 'string') return skinError('SCHEMA', field, `"${field}" must be text.`)
  const { min, max } = TEXT_LIMITS[field]
  const length = [...value].length
  if (length < min || length > max) return skinError('BAD_TEXT', field, `"${field}" must be ${min} to ${max} characters.`)
  const lower = value.toLowerCase()
  if (hasControl(value) || TEXT_FORBIDDEN.some((bad) => lower.includes(bad))) {
    return skinError('BAD_TEXT', field, `"${field}" must be one line of plain text: no markup, links or control characters.`)
  }
  return null
}

function textErrors(input: Fields): SkinError[] {
  return (Object.keys(TEXT_LIMITS) as TextField[]).flatMap((field) => {
    const error = hasOwn(input, field) ? textError(field, input[field]) : null
    return error === null ? [] : [error]
  })
}

/** The skin's tones, or null when the field is missing or invalid. */
function parseTones(value: unknown): readonly SkinTone[] | null {
  if (!Array.isArray(value) || value.length < 1 || value.length > SKIN_TONES.length) return null
  if (!value.every((tone) => oneOf(SKIN_TONES, tone))) return null
  return new Set(value).size === value.length ? (value as SkinTone[]) : null
}

function fontErrors(raw: unknown): SkinError[] {
  if (!isRecord(raw)) return [skinError('BAD_FONT', 'fonts', '"fonts" must be an object with "sans", "serif" and/or "mono".')]
  const bad = FONT_SLOTS.filter((slot) => hasOwn(raw, slot)).filter((slot) => {
    const value = raw[slot]
    return !(oneOf(SHIPPED_FONTS, value) || (typeof value === 'string' && FONT_FILE_PATTERN.test(value)))
  })
  const errors = bad.map((slot) => skinError('BAD_FONT', `fonts.${slot}`, `"fonts.${slot}" must be one of ${SHIPPED_FONTS.join(', ')} or a woff2 file in assets/fonts/.`))
  return [...unknownFields(raw, FONT_SLOTS, 'fonts'), ...errors]
}

function shapeErrors(raw: unknown): SkinError[] {
  if (!isRecord(raw)) return [skinError('BAD_SHAPE', 'shape', '"shape" must be an object with "radius" and/or "glow".')]
  const keys = Object.keys(SHAPE_LIMITS) as (keyof typeof SHAPE_LIMITS)[]
  const errors = keys.filter((key) => hasOwn(raw, key)).flatMap((key) => {
    const [min, max] = SHAPE_LIMITS[key]
    const value = raw[key]
    const ok = typeof value === 'number' && Number.isInteger(value) && value >= min && value <= max
    return ok ? [] : [skinError('BAD_SHAPE', `shape.${key}`, `"shape.${key}" must be a whole number from ${min} to ${max}.`)]
  })
  return [...unknownFields(raw, keys, 'shape'), ...errors]
}

function mascotErrors(raw: unknown): SkinError[] {
  if (!isRecord(raw)) return [skinError('BAD_MASCOT', 'mascot', '"mascot" must be an object with a "mode".')]
  const checks: readonly { key: string; list: readonly string[]; required: boolean }[] = [
    { key: 'mode', list: MASCOT_MODES, required: true },
    { key: 'shell', list: MASCOT_SHELLS, required: false },
    { key: 'light', list: MASCOT_LIGHTS, required: false },
  ]
  const errors = checks
    .filter(({ key, list, required }) => (hasOwn(raw, key) ? !oneOf(list, raw[key]) : required))
    .map(({ key, list }) => skinError('BAD_MASCOT', `mascot.${key}`, `"mascot.${key}" must be one of: ${list.join(', ')}.`))
  return [...unknownFields(raw, MASCOT_FIELDS, 'mascot'), ...errors]
}

function sectionErrors(input: Fields): SkinError[] {
  return [
    ...(hasOwn(input, 'fonts') ? fontErrors(input.fonts) : []),
    ...(hasOwn(input, 'shape') ? shapeErrors(input.shape) : []),
    ...(hasOwn(input, 'mascot') ? mascotErrors(input.mascot) : []),
  ]
}

const OPTIONAL_COPIED = ['author', 'description', 'licence', 'preview'] as const
const SECTIONS_COPIED = ['fonts', 'shape', 'mascot'] as const

/** A fresh Skin with only the known fields; called once every check has passed. */
function toSkin(input: Fields, tones: readonly SkinTone[], tokens: Skin['tokens']): Skin {
  const optional = OPTIONAL_COPIED.filter((key) => hasOwn(input, key)).map((key) => [key, input[key]] as const)
  const sections = SECTIONS_COPIED.filter((key) => hasOwn(input, key)).map((key) => [key, { ...(input[key] as Fields) }] as const)
  return {
    schema: SKIN_SCHEMA_ID,
    slug: input.slug as string,
    name: input.name as string,
    version: input.version as string,
    base: input.base as Skin['base'],
    tones: [...tones],
    tokens,
    ...(Object.fromEntries([...optional, ...sections]) as Partial<Skin>),
  }
}

export function validateSkin(input: unknown): SkinValidation {
  if (!isRecord(input)) return { ok: false, errors: [skinError('SCHEMA', '', 'A skin must be a JSON object.')] }
  const tones = hasOwn(input, 'tones') ? parseTones(input.tones) : null
  const tonesError = hasOwn(input, 'tones') && tones === null
    ? [skinError('SCHEMA', 'tones', '"tones" must be ["light", "dark"], ["light"] or ["dark"].')]
    : []
  const tokens = hasOwn(input, 'tokens') ? checkTokens(input.tokens, tones) : { errors: [], tokens: {} }
  const errors = [
    ...unknownFields(input, SKIN_FIELDS, ''),
    ...headerErrors(input),
    ...textErrors(input),
    ...tonesError,
    ...tokens.errors,
    ...sectionErrors(input),
  ]
  if (errors.length > 0 || tones === null) return { ok: false, errors }
  return { ok: true, skin: toSkin(input, tones, tokens.tokens) }
}
