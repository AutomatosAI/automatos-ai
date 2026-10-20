/**
 * PRD-261 W-A4P — the one way the SDK's page areas (pages/*.ts) hand a theme what a core
 * react-query hook gave them. Not an area: nothing here is exported from '@automatos/theme-sdk'.
 *
 * - A query becomes a frozen `{ data, isLoading, isFetching, error, refetch }` and a mutation a
 *   frozen `{ run, runAsync, isRunning, error, result, reset }`: an error is its message, `refetch`
 *   returns nothing, so a theme never holds react-query's own result or its observer.
 * - The data, the result and what `runAsync` resolves with are deep, frozen copies (frozenCopy),
 *   so a theme never holds, or changes, the objects react-query caches for the core page: an
 *   in-place `sort` or `splice` throws in the theme instead of reordering the core page's list.
 */
import { useMemo } from 'react'

/** A read: null data until it answers; `isLoading` is the first load only. */
export interface ThemeQuery<T> {
  readonly data: T | null
  readonly isLoading: boolean
  readonly isFetching: boolean
  readonly error: string | null
  readonly refetch: () => void
}

/** A write: `run` reports failure in `error`; `runAsync` rejects with it. */
export interface ThemeAction<V, R> {
  readonly run: (variables: V) => void
  readonly runAsync: (variables: V) => Promise<R>
  readonly isRunning: boolean
  readonly error: string | null
  readonly result: R | null
  readonly reset: () => void
}

/** The parts of react-query's query result the view reads. */
export interface CoreQuery {
  readonly data: unknown
  readonly error: unknown
  readonly isInitialLoading: boolean
  readonly isFetching: boolean
  readonly refetch: () => unknown
}

/** The parts of react-query's mutation result the view reads. */
export interface CoreMutation<C> {
  readonly mutate: (variables: C) => void
  readonly mutateAsync: (variables: C) => Promise<unknown>
  readonly isLoading: boolean
  readonly error: unknown
  readonly data: unknown
  readonly reset: () => void
}

/** Turns a theme's variables into the core hook's, or throws a TypeError naming why not. */
export type Prepare<V, C> = (variables: V) => C
/** Copies a backend answer into what the theme sees, or null when it is not that shape. */
export type Shape<T> = (raw: unknown) => T | null

const SDK_ERROR_PREFIX = 'Automatos theme SDK: '

/** Refuses a theme's input: throws the TypeError every SDK area throws. */
export function refuse(reason: string): never {
  throw new TypeError(`${SDK_ERROR_PREFIX}${reason}`)
}

/** An error as the theme sees it: its message, or null for none. */
export function messageOf(error: unknown): string | null {
  if (error === null || error === undefined) return null
  return error instanceof Error ? error.message : String(error)
}

function isPlainRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}

/** A deep, frozen copy of a backend answer's lists and plain objects; any other value as it is. */
export function frozenCopy<T>(value: T): T {
  if (Array.isArray(value)) return Object.freeze(value.map((entry: unknown) => frozenCopy(entry))) as T
  if (!isPlainRecord(value)) return value
  const entries = Object.entries(value).map(([key, entry]) => [key, frozenCopy(entry)] as const)
  return Object.freeze(Object.fromEntries(entries)) as T
}

/** The default shape: the whole answer, copied. */
function copied(raw: unknown): unknown {
  return frozenCopy(raw)
}

/**
 * The theme's view of a core query. `refusal` stands in for the error when the SDK asked for
 * nothing; `shape` (a module-level function) copies the answer, the whole of it by default.
 */
export function useQueryView<T>(
  query: CoreQuery,
  refusal: string | null = null,
  shape: Shape<T> = copied as Shape<T>,
): ThemeQuery<T> {
  const { data, error, isInitialLoading, isFetching, refetch } = query
  return useMemo(
    () =>
      Object.freeze({
        data: data === undefined ? null : shape(data),
        isLoading: isInitialLoading,
        isFetching,
        error: refusal ?? messageOf(error),
        refetch: () => {
          void refetch()
        },
      }),
    [data, error, isInitialLoading, isFetching, refetch, refusal, shape],
  )
}

/** The theme's view of a core mutation; `prepare` checks and converts the variables before they are sent. */
export function useActionView<V, C, R>(mutation: CoreMutation<C>, prepare: Prepare<V, C>): ThemeAction<V, R> {
  const { mutate, mutateAsync, isLoading, error, data, reset } = mutation
  return useMemo(
    () =>
      Object.freeze({
        run: (variables: V) => mutate(prepare(variables)),
        runAsync: (variables: V) =>
          new Promise<R>((resolve, reject) => {
            mutateAsync(prepare(variables)).then((answer) => resolve(frozenCopy(answer) as R), reject)
          }),
        isRunning: isLoading,
        error: messageOf(error),
        result: (data === undefined ? null : frozenCopy(data)) as R | null,
        reset,
      }),
    [mutate, mutateAsync, isLoading, error, data, reset, prepare],
  )
}
