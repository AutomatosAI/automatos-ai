/**
 * PRD-234 — why a Claude Code session ticket landed in Review.
 *
 * The host reports every tool call its policy refused; the backend keeps the
 * reasons on `runtime_ref.permission_denials` (plus the `denials` count) and
 * forces the ticket into review. This turns them into the lines the ticket shows.
 */

export interface SessionDenial {
  tool?: string
  stage?: string
  reason?: string
  subject?: string
}

export function sessionDenials(ref: Record<string, any> | null | undefined): SessionDenial[] {
  const raw = ref?.permission_denials
  if (!Array.isArray(raw)) return []
  return raw
    .filter((d): d is Record<string, unknown> => !!d && typeof d === 'object')
    .map((d) => ({
      tool: typeof d.tool === 'string' ? d.tool : undefined,
      stage: typeof d.stage === 'string' ? d.stage : undefined,
      reason: typeof d.reason === 'string' ? d.reason : undefined,
      subject: typeof d.subject === 'string' ? d.subject : undefined,
    }))
}

/** "Bash · 'python3 hello.py' is outside this ticket's Bash allowlist" */
export function denialLine(d: SessionDenial): string {
  const head = d.tool ? `${d.tool} · ` : ''
  const body = d.reason || d.subject || 'refused by the session policy'
  return `${head}${body}`
}

/** The one-line explanation for a review'd session ticket, or null when nothing was refused. */
export function reviewReason(ref: Record<string, any> | null | undefined): string | null {
  const count = typeof ref?.denials === 'number' ? ref.denials : sessionDenials(ref).length
  if (!count) return null
  return count === 1
    ? 'Sent to Review: the session policy refused 1 tool call, so the result was not verified end to end.'
    : `Sent to Review: the session policy refused ${count} tool calls, so the result was not verified end to end.`
}
