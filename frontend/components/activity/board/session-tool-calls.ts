/**
 * F167 — what the CLI host decided for each tool call of a session ticket.
 *
 * The host reports every decision (allow, ask or deny), its reason and a hold's
 * answer; the backend keeps them on `runtime_ref.recent_tools` and counts every
 * one in `runtime_ref.tool_decisions`. A call that ran with nobody asked never
 * reads as one the operator approved, and neither does an approval the host
 * reported that the backend has no record of. Same words as the task report
 * (orchestrator/services/session_report.py).
 */

export interface SessionToolCall {
  at?: string
  tool?: string
  subject?: string
  decision?: string
  reason?: string
  answer?: string
}

const HOLD_OUTCOMES: Record<string, string> = {
  approved: 'the operator approved it',
  denied: 'the operator denied it',
  'no answer': 'no answer in time, so it did not run',
  'approval not on record': 'the host reported an approval that is not on record',
}

const text = (value: unknown): string | undefined => (typeof value === 'string' ? value : undefined)

export function sessionToolCalls(ref: Record<string, any> | null | undefined): SessionToolCall[] {
  const raw = ref?.recent_tools
  if (!Array.isArray(raw)) return []
  return raw
    .filter((c): c is Record<string, unknown> => !!c && typeof c === 'object')
    .map((c) => ({
      at: text(c.at), tool: text(c.tool), subject: text(c.subject),
      decision: text(c.decision), reason: text(c.reason), answer: text(c.answer),
    }))
}

/** "ran, nobody was asked" · "held for the operator: …" · "refused by the gate" — '' when the host did not say. */
export function toolCallVerdict(call: SessionToolCall): string {
  if (call.decision === 'allow') return 'ran, nobody was asked'
  if (call.decision === 'ask') return `held for the operator: ${HOLD_OUTCOMES[call.answer ?? ''] ?? 'no answer recorded'}`
  if (call.decision === 'deny') return 'refused by the gate'
  return ''
}

/** The full line for a hover: what the call was about, the verdict and the host's reason. */
export function toolCallTitle(call: SessionToolCall): string {
  const verdict = toolCallVerdict(call)
  const said = verdict && call.reason ? `${verdict} (${call.reason})` : verdict
  return [call.subject || call.tool || '', said].filter(Boolean).join(' — ')
}

/** "3 ran with nobody asked · none held for the operator · 0 refused by the gate", or null before any decision. */
export function toolDecisionsSummary(ref: Record<string, any> | null | undefined): string | null {
  const tally = ref?.tool_decisions
  if (!tally || typeof tally !== 'object') return null
  const count = (key: string): number => (typeof tally[key] === 'number' ? tally[key] : 0)
  const held = count('ask')
  const unrecorded = count('unrecorded')
  const notOnRecord = unrecorded ? `, ${unrecorded} approval${unrecorded !== 1 ? 's' : ''} not on record` : ''
  const heldPart = held ? `${held} held for the operator (${count('approved')} approved${notOnRecord})` : 'none held for the operator'
  return `${count('allow')} ran with nobody asked · ${heldPart} · ${count('deny')} refused by the gate`
}
