import type { Deliverable } from '@/hooks/use-deliverables-api'

/**
 * PRD-260 §4 — which Deliverables this browser has already shown on Play's
 * shelf, so a new one makes Auto celebrate once; ported from Automatos Play's
 * lib/deliverables/seen.ts onto the core's Deliverable (created_at). A
 * Deliverable is new when its id hasn't been seen and it is newer than the
 * newest one seen: old rows met for the first time (a chip, Load more) don't
 * count. The first visit only remembers, it doesn't celebrate.
 *
 * A per-browser convenience in localStorage, which may be missing or blocked:
 * then the shelf remembers for as long as it is open. Nothing else reads it.
 */

export type SeenRecord = {
  readonly ids: readonly string[]
  /** created_at of the newest Deliverable seen (the backend's own clock and format). */
  readonly newest: string | null
}

type Dated = Pick<Deliverable, 'id' | 'created_at'>

export const SEEN_KEY = 'automatos-play:seen-deliverables'
/** The most ids kept; the newest ones win. */
export const SEEN_CAP = 500

function time(iso: string | null | undefined): number | null {
  if (!iso) return null
  const parsed = Date.parse(iso)
  return Number.isNaN(parsed) ? null : parsed
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function parseSeen(raw: string | null): SeenRecord | null {
  if (!raw) return null
  try {
    const value: unknown = JSON.parse(raw)
    if (!isRecord(value) || !Array.isArray(value.ids)) return null
    const ids = value.ids.filter((id): id is string => typeof id === 'string').slice(-SEEN_CAP)
    const newest = typeof value.newest === 'string' ? value.newest : null
    return { ids, newest }
  } catch (error) {
    console.warn('[shelf] the seen Deliverables record was unreadable, starting again', error)
    return null
  }
}

export function readSeen(): SeenRecord | null {
  try {
    return parseSeen(window.localStorage.getItem(SEEN_KEY))
  } catch (error) {
    console.warn("[shelf] can't read localStorage; new Deliverables are remembered only while the shelf is open", error)
    return null
  }
}

export function writeSeen(record: SeenRecord): void {
  try {
    window.localStorage.setItem(SEEN_KEY, JSON.stringify(record))
  } catch (error) {
    console.warn("[shelf] can't save the seen Deliverables to localStorage", error)
  }
}

/** The loaded Deliverables that count as new (none on the first visit). */
export function freshDeliverables<T extends Dated>(record: SeenRecord | null, items: readonly T[]): readonly T[] {
  const newest = time(record?.newest)
  if (!record || newest === null) return []
  const seen = new Set(record.ids)
  return items.filter((item) => {
    const created = time(item.created_at)
    return !seen.has(item.id) && created !== null && created > newest
  })
}

function newestOf(items: readonly Dated[], start: string | null): string | null {
  return items.reduce<string | null>((best, item) => {
    const created = time(item.created_at)
    const top = time(best)
    return created !== null && (top === null || created > top) ? item.created_at : best
  }, start)
}

/** The record with these Deliverables remembered. Returns the same record when nothing changes. */
export function markSeen(record: SeenRecord | null, items: readonly Dated[]): SeenRecord {
  const ids = record?.ids ?? []
  const known = new Set(ids)
  const added = items.map((item) => item.id).filter((id) => !known.has(id))
  const newest = newestOf(items, record?.newest ?? null)
  if (record && added.length === 0 && newest === record.newest) return record
  return { ids: [...ids, ...added].slice(-SEEN_CAP), newest }
}
