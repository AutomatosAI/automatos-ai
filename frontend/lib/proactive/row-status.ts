/**
 * PRD-268 §6.11.1 (US-W04) — a row's status fields off the wire, and the order rows are listed in.
 *
 * W1 (US-C17a/b) puts `display_state`, `next_step` and `retry` on the needs-you rows, the Missions list and the
 * board's ticket reads. `rowStatusOf` keeps each only when it is what it says: one of §6.11.1's states, a sentence,
 * one of the three retries; anything else reads null, so StatusLabel draws nothing for it. `needsYouFirst` lists the
 * rows that need the person before the others, keeping every other row where it was (§6.12.3: needs_you sorted first).
 */
import { isDisplayState } from './display-state'
import type { Retry, RowStatus } from './types'

const RETRIES: ReadonlySet<string> = new Set<Retry>(['rerun_mission', 'replan_mission', 'retry_ticket'])

/** The three status fields of a row from the API, each null unless it is what it says. */
export function rowStatusOf(row: unknown): Required<RowStatus> {
  const fields = typeof row === 'object' && row !== null ? (row as Record<string, unknown>) : {}
  const { display_state: state, next_step: step, retry } = fields
  return {
    display_state: isDisplayState(state) ? state : null,
    next_step: typeof step === 'string' && step.trim() !== '' ? step : null,
    retry: typeof retry === 'string' && RETRIES.has(retry) ? (retry as Retry) : null,
  }
}

/** The rows that need the person first, then the rest, each group in the order it came. */
export function needsYouFirst<T extends RowStatus>(rows: readonly T[]): T[] {
  return [...rows.filter((row) => row.display_state === 'needs_you'), ...rows.filter((row) => row.display_state !== 'needs_you')]
}
