/**
 * What an approval click ran, in the server's own words (PRD-193 S4).
 *
 * The /grant route answers 200 whenever the yes is recorded, and the resumed call's
 * result rides on the grant (`details.executed_result`). P256-FIX-N17-2 (F444, night 17):
 * the Governance inbox and the ticket's approvals toasted "granted" for two clicks whose
 * calls the server refused, so the owner learned only by asking. Every surface that
 * approves a grant reads the outcome here.
 */
export interface ExecutedResult {
  success?: boolean
  error?: string | null
  requires_confirmation?: boolean
  resumed_via?: string
}

/** The resumed call's result on a /grant response, or undefined when it carries none. */
export function executedResultOf(response: unknown): ExecutedResult | undefined {
  const grant = (response as { grant?: { details?: { executed_result?: ExecutedResult } } } | null)?.grant
  return grant?.details?.executed_result
}

/**
 * P256-FIX-N17-3 (F439, night 17): a click the server refused says why, in its words. The /grant
 * route answers 422 for a card that can no longer run ("This card expired at 05:12 on 15 Oct;
 * nothing ran…", "…replaced by a newer card…"); the api client puts that detail in the error's
 * message. Any other failure keeps the surface's own line.
 */
export function refusedClickLine(err: unknown, fallback: string): string {
  const status = (err as { status?: unknown } | null)?.status
  const message = err instanceof Error ? err.message.trim() : ''
  return status === 422 && message ? message : fallback
}

/** Post-approve outcome line — honest about what actually happened (S4). */
export function executedOutcomeLine(executed?: ExecutedResult | null): string {
  if (!executed) return 'Approved.'
  if (executed.resumed_via === 'board_task_requeue') {
    return 'Approved — the blocked task was re-queued and will complete under this grant.'
  }
  if (executed.requires_confirmation) {
    return 'Approved, but the action asked for confirmation again — check the pending approvals.'
  }
  if (executed.success) return 'Approved — the action ran.'
  return `Approved, but execution failed: ${executed.error || 'unknown error'}`
}
