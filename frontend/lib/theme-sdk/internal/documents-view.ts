/**
 * PRD-261 A-12f — how the SDK's pages/documents area (pages/documents.ts) hands a theme what a
 * core hook gave it, and checks what a theme hands back. Not an area: nothing here is exported
 * from '@automatos/theme-sdk'.
 *
 * - A query becomes a frozen `KnowledgeQuery` whose data is a deep, frozen copy of the fields its
 *   type names, so a theme never holds, or changes, the objects react-query caches for the core
 *   page. The views themselves are the ones every page area uses (internal/query-view.ts).
 * - A mutation becomes a frozen `KnowledgeAction` whose variables go through `prepare` first: it
 *   converts them to what the core hook takes, or throws a TypeError naming why not.
 */
import type { KnowledgeAction, KnowledgeQuery } from '../pages/documents'

import {
  frozenCopy,
  refuse,
  useActionView as useSharedActionView,
  useQueryView as useSharedQueryView,
  type CoreMutation as SharedCoreMutation,
  type CoreQuery,
  type Prepare,
  type Shape,
} from './query-view'

export { refuse, type CoreQuery, type Prepare, type Shape }

/** A core mutation typed by its answer, so an action's result type is the core hook's own. */
export interface CoreMutation<W, R> extends SharedCoreMutation<W> {
  readonly mutateAsync: (variables: W) => Promise<R>
  readonly data: R | undefined
}

const EMPTY_LIST: readonly never[] = Object.freeze([])

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** A frozen copy of the named fields `raw` has (each a deep, frozen copy), or null when it is not an object. */
function pickFields(raw: unknown, fields: readonly string[]): Readonly<Record<string, unknown>> | null {
  if (!isRecord(raw)) return null
  const present = fields.filter((field) => Object.prototype.hasOwnProperty.call(raw, field))
  return Object.freeze(Object.fromEntries(present.map((field) => [field, frozenCopy(raw[field])])))
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
  return useSharedQueryView<T>(query, null, shape)
}

/** The theme's view of a core mutation; `prepare` checks and converts the variables before they are sent. */
export function useActionView<V, W, R>(mutation: CoreMutation<W, R>, prepare: Prepare<V, W>): KnowledgeAction<V, R> {
  return useSharedActionView<V, W, R>(mutation, prepare)
}
