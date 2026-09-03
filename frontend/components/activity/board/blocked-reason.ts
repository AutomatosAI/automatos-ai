/**
 * PRD-234 — what a Blocked ticket is waiting for.
 *
 * The backend writes `blocked_reason` as
 * "Awaiting human approval (grant #11): board task requires approval under 'always_ask' policy".
 * This pulls the grant id out so the ticket can offer the Grant button itself.
 */

export interface BlockedInfo {
  grantId: number | null
  text: string
}

export function parseBlockedReason(reason: string | null | undefined): BlockedInfo {
  const text = (reason || '').trim()
  const m = text.match(/grant #(\d+)/i)
  return { grantId: m ? Number(m[1]) : null, text }
}
