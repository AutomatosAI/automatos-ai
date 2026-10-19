/**
 * PRD-261 A-12f — how the SDK's pages/documents area (pages/documents.ts) hands a theme what a
 * core hook gave it, and checks what a theme hands back. Not an area: nothing here is exported
 * from '@automatos/theme-sdk'.
 *
 * - A query becomes a frozen `KnowledgeQuery` whose data is a copy of the fields its type names
 *   (one level deep), so a theme never holds, or changes, the objects react-query caches for the
 *   core page.
 * - A mutation becomes a frozen `KnowledgeAction` whose variables go through `prepare` first: it
 *   converts them to what the core hook takes, or throws a TypeError naming why not.
 */
import { useMemo } from 'react'

import type { KnowledgeAction, KnowledgeQuery } from '../pages/documents'

/** The parts of react-query's query result the view reads. */
export interface CoreQuery {
  readonly data: unknown
  readonly error: unknown
  readonly isInitialLoading: boolean
  readonly isFetching: boolean
  readonly refetch: () => unknown
}

/** The parts of react-query's mutation result the view reads. */
export interface CoreMutation<W, R> {
  readonly mutate: (variables: W) => void
  readonly mutateAsync: (variables: W) => Promise<R>
  readonly isLoading: boolean
  readonly error: unknown
  readonly data: R | undefined
  readonly reset: () => void
}

/** Turns a theme's variables into the core hook's, or throws a TypeError naming why not. */
export type Prepare<V, W> = (variables: V) => W
/** Copies a backend answer into what the theme sees, or null when it is not that shape. */
export type Shape<T> = (raw: unknown) => T | null

const SDK_ERROR_PREFIX = 'Automatos theme SDK: '
const EMPTY_LIST: readonly never[] = Object.freeze([])

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function messageOf(error: unknown): string | null {
  if (error === null || error === undefined) return null
  return error instanceof Error ? error.message : String(error)
}

/** Refuses a theme's input: throws the TypeError every SDK area throws. */
export function refuse(reason: string): never {
  throw new TypeError(`${SDK_ERROR_PREFIX}${reason}`)
}

/** A field's value, one level deep: an array or an object is a frozen copy, never the cache's own. */
function copyValue(value: unknown): unknown {
  if (Array.isArray(value)) return Object.freeze(value.slice())
  return isRecord(value) ? Object.freeze({ ...value }) : value
}

/** A frozen copy of the named fields `raw` has, or null when it is not an object. */
function pickFields(raw: unknown, fields: readonly string[]): Readonly<Record<string, unknown>> | null {
  if (!isRecord(raw)) return null
  const present = fields.filter((field) => Object.prototype.hasOwnProperty.call(raw, field))
  return Object.freeze(Object.fromEntries(present.map((field) => [field, copyValue(raw[field])])))
}

/** The shape of one record with `fields`. */
export function recordOf<T>(fields: readonly string[]): Shape<T> {
  return (raw) => pickFields(raw, fields) as T | null
}

/** A frozen list of row copies with `fields`; an answer that is not a list is an empty one. */
export function rowsOf<T>(raw: unknown, fields: readonly string[]): readonly T[] {
  if (!Array.isArray(raw)) return EMPTY_LIST
  const rows = raw.map((row) => pickFields(row, fields)).filter((row) => row !== null)
  return Object.freeze(rows) as unknown as readonly T[]
}

/** The shape of a list of rows with `fields`. */
export function listOf<T>(fields: readonly string[]): Shape<readonly T[]> {
  return (raw) => rowsOf<T>(raw, fields)
}

function isPositiveId(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0
}

/** A positive whole number that may go into a URL path, or a refusal naming `what` it should have been. */
export function checkedId(value: unknown, what: string): number {
  return isPositiveId(value) ? value : refuse(`${String(value)} is not a ${what} id.`)
}

/** Text with something in it, or a refusal. */
export function checkedText(value: unknown, what: string): string {
  return typeof value === 'string' && value.trim().length > 0 ? value : refuse(`${what} must be text.`)
}

/** A copy of a list of text (none is an empty list), or a refusal. */
export function textList(value: unknown, what: string): string[] {
  if (value === undefined) return []
  const ok = Array.isArray(value) && value.every((entry) => typeof entry === 'string')
  return ok ? (value as string[]).slice() : refuse(`${what} must be a list of text.`)
}

/** The theme's view of a core query, with its data shaped by `shape` (a module-level function). */
export function useQueryView<T>(query: CoreQuery, shape: Shape<T>): KnowledgeQuery<T> {
  const { data, error, isInitialLoading, isFetching, refetch } = query
  return useMemo(
    () =>
      Object.freeze({
        data: data === undefined ? null : shape(data),
        isLoading: isInitialLoading,
        isFetching,
        error: messageOf(error),
        refetch: () => {
          void refetch()
        },
      }),
    [data, error, isInitialLoading, isFetching, refetch, shape],
  )
}

/** The theme's view of a core mutation; `prepare` checks and converts the variables before they are sent. */
export function useActionView<V, W, R>(mutation: CoreMutation<W, R>, prepare: Prepare<V, W>): KnowledgeAction<V, R> {
  const { mutate, mutateAsync, isLoading, error, data, reset } = mutation
  return useMemo(
    () =>
      Object.freeze({
        run: (variables: V) => mutate(prepare(variables)),
        runAsync: (variables: V) =>
          new Promise<R>((resolve, reject) => {
            mutateAsync(prepare(variables)).then(resolve, reject)
          }),
        isRunning: isLoading,
        error: messageOf(error),
        result: data ?? null,
        reset,
      }),
    [mutate, mutateAsync, isLoading, error, data, reset, prepare],
  )
}
