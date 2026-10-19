/**
 * PRD-261 A-04c — the failure ledger: when each theme last failed in this browser, so a theme
 * that fails three times in ten minutes sends the member back to Studio (the provider does the
 * revert). Times are kept per theme (its style, `theme:install:<uuid>`) in localStorage, so they
 * survive the reloads between failures.
 *
 * Only the theme's own failures count: `render`, `load`, `register`, `look` and `engines`. A
 * refused session (`auth`, A-08b) is the session's fault; `revoked` and `opt-in` are the server's
 * answer, not a broken theme; `manifest` is refused before any of the theme's code runs; and a
 * page that throws inside the frame never reaches the host (A-04a's page boundary), so none of
 * them count.
 *
 * Storage errors (blocked site data, a full store, a value another script wrote) mean "nothing
 * remembered", never a throw: without storage each failure counts alone, so the revert never
 * fires, and the theme still falls back to Studio on every failure.
 */
import { isRecord } from './css-value'
import type { ThemeFailureReason } from './types'

export type LedgerStorage = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>

/** One key for every theme: `{ [theme]: [failedAtMs, …] }`. */
export const FAILURE_LEDGER_KEY = 'automatos-theme-failures'
/** Failures older than this are forgotten. */
export const FAILURE_WINDOW_MS = 10 * 60 * 1000
/** The failure that reaches this count, inside the window, reverts to Studio. */
export const FAILURES_BEFORE_REVERT = 3

/** The reasons that are the theme's own fault. */
const COUNTED_REASONS: readonly ThemeFailureReason[] = ['render', 'load', 'register', 'look', 'engines']
const LOG_PREFIX = '[theme-host]'

type Ledger = Readonly<Record<string, readonly number[]>>

/** Whether a failure with this reason counts toward the revert. */
export function countsTowardRevert(reason: ThemeFailureReason): boolean {
  return COUNTED_REASONS.includes(reason)
}

/** The times inside the window that ends at `now`; a time after `now` (a clock moved back) is dropped. */
function recent(times: readonly number[], now: number): number[] {
  return times.filter((time) => time <= now && now - time < FAILURE_WINDOW_MS)
}

function isTimeList(value: unknown): value is number[] {
  return Array.isArray(value) && value.every((time) => typeof time === 'number' && Number.isFinite(time))
}

/** The stored ledger, keeping only well-formed entries; empty when there is none or it cannot be read. */
function readLedger(storage: LedgerStorage): Ledger {
  try {
    const parsed: unknown = JSON.parse(storage.getItem(FAILURE_LEDGER_KEY) ?? 'null')
    if (!isRecord(parsed)) return {}
    return Object.fromEntries(Object.entries(parsed).filter((entry): entry is [string, number[]] => isTimeList(entry[1])))
  } catch (error) {
    console.warn(LOG_PREFIX, 'the failure ledger could not be read', error)
    return {}
  }
}

/** Writes the ledger, or removes the key when no theme has a recent failure. */
function writeLedger(storage: LedgerStorage, ledger: Ledger): void {
  try {
    if (Object.keys(ledger).length === 0) storage.removeItem(FAILURE_LEDGER_KEY)
    else storage.setItem(FAILURE_LEDGER_KEY, JSON.stringify(ledger))
  } catch (error) {
    console.warn(LOG_PREFIX, 'the failure ledger could not be saved', error)
  }
}

/** The ledger with every theme's old failures dropped, and themes with none left removed. */
function pruned(ledger: Ledger, now: number): Ledger {
  const kept = Object.entries(ledger).map(([theme, times]): [string, number[]] => [theme, recent(times, now)])
  return Object.fromEntries(kept.filter(([, times]) => times.length > 0))
}

/**
 * Records a failure of `theme` at `now` and answers how many it has had in the window, this one
 * included. With no storage, this one alone.
 */
export function recordThemeFailure(theme: string, now: number, storage: LedgerStorage | null): number {
  if (storage === null) return 1
  const ledger = pruned(readLedger(storage), now)
  const times = [...(ledger[theme] ?? []), now]
  writeLedger(storage, { ...ledger, [theme]: times })
  return times.length
}

/** Forgets `theme`'s failures (after a revert the server took), keeping every other theme's. */
export function clearThemeFailures(theme: string, storage: LedgerStorage | null): void {
  if (storage === null) return
  const rest = Object.entries(readLedger(storage)).filter(([key]) => key !== theme)
  writeLedger(storage, Object.fromEntries(rest))
}

/** Whether a theme with this many recent failures goes back to Studio. */
export function shouldRevert(failures: number): boolean {
  return failures >= FAILURES_BEFORE_REVERT
}

/** This browser's localStorage, or null where reading the property itself throws (blocked site data). */
export function safeLocalStorage(): LedgerStorage | null {
  try {
    return window.localStorage
  } catch (error) {
    console.warn(LOG_PREFIX, 'localStorage is not available', error)
    return null
  }
}
