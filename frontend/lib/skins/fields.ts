// PRD-259 §3 — small shared helpers for the skin validator (validate.ts, tokens.ts).

import type { SkinError, SkinErrorCode } from './schema'

export type Fields = Readonly<Record<string, unknown>>

/** Messages never echo more than this many characters of the input back. */
const ECHO_MAX = 40

export function isRecord(value: unknown): value is Fields {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function hasOwn(record: Fields, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(record, key)
}

export function oneOf<T extends string>(list: readonly T[], value: unknown): value is T {
  return typeof value === 'string' && (list as readonly string[]).includes(value)
}

export function echo(text: string): string {
  const chars = [...text]
  return chars.length > ECHO_MAX ? `${chars.slice(0, ECHO_MAX).join('')}…` : text
}

export function joinPath(prefix: string, key: string): string {
  return prefix ? `${prefix}.${key}` : key
}

export function skinError(code: SkinErrorCode, path: string, message: string): SkinError {
  return { code, path, message }
}

/** UNKNOWN_FIELD for every key of `record` that isn't in `known`. */
export function unknownFields(record: Fields, known: readonly string[], prefix: string): SkinError[] {
  return Object.keys(record)
    .filter((key) => !known.includes(key))
    .map((key) => skinError('UNKNOWN_FIELD', joinPath(prefix, key), `"${echo(key)}" isn't part of a skin. Remove it.`))
}
