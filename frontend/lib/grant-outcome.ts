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
