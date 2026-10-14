// PRD-259 §3 — a skin's tokens: allowed names only, colour values only, the tones they cover,
// and AA contrast on the skin's own values (decision K5). Mirrored by services/skins/tokens.py.

import { normaliseColor } from './color'
import { contrastRatio } from './contrast'
import {
  ALLOWED_TOKENS,
  CONTRAST_PAIRS,
  DERIVED_ALIASES,
  SKIN_TONES,
  type ContrastPair,
  type SkinError,
  type SkinToken,
  type SkinTone,
} from './schema'
import { echo, hasOwn, isRecord, oneOf, skinError, unknownFields, type Fields } from './fields'

export type ToneValues = Readonly<Partial<Record<SkinToken, string>>>

interface ToneResult {
  readonly errors: readonly SkinError[]
  readonly values: ToneValues | null
}

function checkTokenEntry(tone: SkinTone, key: string, value: unknown): SkinError | null {
  const path = `tokens.${tone}.${key}`
  if ((DERIVED_ALIASES as readonly string[]).includes(key)) {
    return skinError('DERIVED_TOKEN', path, `"${key}" is derived by Studio from the core tokens; a skin can't set it.`)
  }
  if (!oneOf(ALLOWED_TOKENS, key)) return skinError('UNKNOWN_TOKEN', path, `"${echo(key)}" isn't a token a skin can set.`)
  if (normaliseColor(value) === null) {
    return skinError('BAD_VALUE', path, `"${key}" must be an HSL triplet like "16 100% 50%" or a #rrggbb colour.`)
  }
  return null
}

function normalisedValues(map: Fields): ToneValues {
  const entries = Object.keys(map).flatMap((key) => {
    const value = oneOf(ALLOWED_TOKENS, key) ? normaliseColor(map[key]) : null
    return value === null ? [] : [[key, value] as const]
  })
  return Object.fromEntries(entries)
}

function checkPair(tone: SkinTone, map: Fields, values: ToneValues, pair: ContrastPair): SkinError | null {
  const { foreground, background, min } = pair
  const fgSet = hasOwn(map, foreground)
  const bgSet = hasOwn(map, background)
  if (!fgSet && !bgSet) return null
  if (fgSet !== bgSet) {
    const missing = fgSet ? background : foreground
    const given = fgSet ? foreground : background
    return skinError('PAIR_INCOMPLETE', `tokens.${tone}.${missing}`, `Setting "${given}" in ${tone} needs "${missing}" too: they are checked as a pair.`)
  }
  const fg = values[foreground]
  const bg = values[background]
  if (fg === undefined || bg === undefined) return null
  const ratio = contrastRatio(fg, bg)
  if (ratio >= min) return null
  return skinError('CONTRAST', `tokens.${tone}.${foreground}`, `"${foreground}" on "${background}" in ${tone} is ${ratio.toFixed(2)}:1; it must be at least ${min}:1.`)
}

function checkTone(tone: SkinTone, raw: unknown): ToneResult {
  if (!isRecord(raw)) return { errors: [skinError('SCHEMA', `tokens.${tone}`, `"tokens.${tone}" must be an object of token values.`)], values: null }
  const entryErrors = Object.keys(raw).flatMap((key) => {
    const error = checkTokenEntry(tone, key, raw[key])
    return error === null ? [] : [error]
  })
  const values = normalisedValues(raw)
  const pairErrors = CONTRAST_PAIRS.flatMap((pair) => {
    const error = checkPair(tone, raw, values, pair)
    return error === null ? [] : [error]
  })
  return { errors: [...entryErrors, ...pairErrors], values }
}

function toneMismatch(tone: SkinTone, present: boolean, tones: readonly SkinTone[] | null): SkinError | null {
  if (tones === null || present === tones.includes(tone)) return null
  const message = present
    ? `"tokens.${tone}" is set but "tones" doesn't include "${tone}".`
    : `"tones" includes "${tone}", so "tokens.${tone}" is required.`
  return skinError('TONES_MISMATCH', `tokens.${tone}`, message)
}

export interface TokensResult {
  readonly errors: readonly SkinError[]
  readonly tokens: Partial<Record<SkinTone, ToneValues>>
}

/** `tones` is null when the skin's own `tones` field was invalid (reported elsewhere). */
export function checkTokens(raw: unknown, tones: readonly SkinTone[] | null): TokensResult {
  if (!isRecord(raw)) return { errors: [skinError('SCHEMA', 'tokens', '"tokens" must be an object with "light" and/or "dark".')], tokens: {} }
  const results = SKIN_TONES.map((tone) => {
    const present = hasOwn(raw, tone)
    const mismatch = toneMismatch(tone, present, tones)
    const checked = present ? checkTone(tone, raw[tone]) : { errors: [], values: null }
    return { tone, errors: [...(mismatch ? [mismatch] : []), ...checked.errors], values: checked.values }
  })
  const tokens = Object.fromEntries(results.flatMap((r) => (r.values === null ? [] : [[r.tone, r.values] as const])))
  return { errors: [...unknownFields(raw, SKIN_TONES, 'tokens'), ...results.flatMap((r) => r.errors)], tokens }
}
