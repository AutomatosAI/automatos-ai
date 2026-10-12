/**
 * P256-FIX-N13-7 (F414) — one ticket number everywhere: #1394, never 2680.
 *
 * Night 13, Auto named a ticket by its board number (#1394, its `workspace_seq`) while a
 * surface showed its row id (2680), so "did that go through?" got yes, then no, on a number
 * the owner could not find. A ticket's number is the API's `number` (#0042, a mission step's
 * #0051.3), else its `workspace_seq` written the same way. Its row id is never shown as its
 * number: a '#' means a number, and an id is no ticket the owner can look up.
 */

// The board number's digits (orchestrator services/ticket_numbers.NUMBER_DIGITS): #0042.
const NUMBER_DIGITS = 4
// A mission step has no number of its own: the API gives its card's and its place, #0051.3.
const STEP_SOURCE = 'orchestration_task'

export type Numbered = { number?: string | null; workspace_seq?: number | null; source_type?: string | null }

/** The ticket's number as the board shows it, or null when it has none. */
export function ticketNumber(task: Numbered): string | null {
  if (task.number) return task.number
  const seq = task.workspace_seq
  if (task.source_type === STEP_SOURCE || typeof seq !== 'number' || !Number.isInteger(seq)) return null
  return `#${String(seq).padStart(NUMBER_DIGITS, '0')}`
}

/** "ticket #1394", or "this ticket" when it has no number: never "ticket 2680", its row id. */
export function ticketCalled(task: Numbered, capital = false): string {
  const number = ticketNumber(task)
  const called = number ? `ticket ${number}` : 'this ticket'
  return capital ? called[0].toUpperCase() + called.slice(1) : called
}
