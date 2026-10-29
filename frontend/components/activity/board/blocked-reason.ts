/**
 * PRD-234 — what a Blocked ticket is waiting for.
 *
 * The backend writes `blocked_reason` as
 * "Awaiting human approval (grant #11): Ask before any Mission or task runs":
 * the prefix names the grant, and after it is the rule that asked, in words
 * (PRD-268 US-C20b: the server's sentence, never composed here). This pulls the
 * grant id out so the ticket can offer the Grant button itself, and the sentence
 * so the ticket can say which rule asked. A reason written before the sentence
 * ("… board task requires approval under 'always_ask' policy") still names its
 * grant, and has no sentence.
 */

export interface BlockedInfo {
  grantId: number | null
  text: string
  /** The rule that asked, in words: what follows the approval prefix. Absent without one. */
  sentence?: string
}

const GRANT = /grant #(\d+)/i
const APPROVAL_PREFIX = /^Awaiting human approval \(grant #\d+\):\s*/i
// What the gate wrote after the prefix before PRD-268: machine text, never shown as the rule.
const MACHINE_TEXT = /^(board task requires approval|pending approval grant already exists)\b/i

export function parseBlockedReason(reason: string | null | undefined): BlockedInfo {
  const text = (reason || '').trim()
  const m = text.match(GRANT)
  return { grantId: m ? Number(m[1]) : null, text, sentence: sentenceOf(text) }
}

function sentenceOf(text: string): string | undefined {
  const prefix = text.match(APPROVAL_PREFIX)
  if (!prefix) return undefined
  const after = text.slice(prefix[0].length).trim()
  return after && !MACHINE_TEXT.test(after) ? after : undefined
}

/**
 * What the ticket's Blocked panel says under its heading: an approval's rule
 * sentence (nothing for one written before it, whose heading already names the
 * grant), any other reason as written.
 */
export function blockedLine(reason: string | null | undefined): { grantId: number | null; line: string } {
  const { grantId, text, sentence } = parseBlockedReason(reason)
  return { grantId, line: APPROVAL_PREFIX.test(text) ? sentence ?? '' : text }
}
