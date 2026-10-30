/**
 * The member's words for why Auto Live can't start: a refused mint (`POST /api/voice/web-call`) and the reason
 * `GET /api/voice/live-status` gives. Both are read here, so a coded refusal never reaches the screen as JSON.
 *
 * Since #1264 the plan refusal is a coded detail (`{"code": "voice_live_plan_required", "message": …}`, the shape the
 * Automatos Go app keys on); apiClient puts the JSON of a structured detail in the error's message, so reading the
 * message showed the raw object. The parsed body rides the error as `body` (lib/api-client.ts ApiHttpError): a coded
 * detail becomes a sentence by its code, a plain-sentence detail (the other gates' honest reasons) stays as it is.
 */

/** A live-status reason or a refusal code, in the member's words. */
export const LIVE_REASON_SENTENCES: Readonly<Record<string, string>> = Object.freeze({
  voice_live_plan_required: "Auto Live isn't on this workspace's plan.",
  // PRD-265 D-6: a call is paid by the minute from credits; the mint refuses (402) when none are left.
  credits_exhausted: 'Auto Live is paid from your credits, and this workspace has none left. Top up on the Billing tab, then call again.',
  voice_live_platform_off: 'Auto Live is switched off on this Automatos.',
  voice_live_not_armed: "Auto Live isn't set up yet. An admin can set it up in Settings → System Settings → Voice.",
  voice_live_workspace_off: 'Auto Live is off for this workspace. An admin can turn it on in Settings → Voice → Auto Live.',
  voice_live_cap_reached: 'This workspace has used its Auto Live minutes for this month.',
})

export const LIVE_UNAVAILABLE = 'Auto Live is unavailable right now. Try again in a moment.'
export const LIVE_STATUS_UNREAD = "Auto Live's status couldn't be checked. Try again in a moment."
export const LIVE_BROWSER_UNSUPPORTED = 'This browser cannot do live voice — try a current Chrome, Edge or Safari.'
export const LIVE_MIC_DENIED = 'Microphone permission was denied — allow the mic and try again.'
export const LIVE_CONNECT_FAILED = 'Could not connect the call — check your connection and try again.'

interface CodedDetail {
  readonly code: string
  readonly message?: unknown
}

function isCoded(detail: unknown): detail is CodedDetail {
  return typeof detail === 'object' && detail !== null && typeof (detail as { code?: unknown }).code === 'string'
}

function plainText(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

/** A live-status reason (a code) in the member's words; the general sentence for a code this build doesn't know. */
export function liveUnavailableSentence(reason: string | null | undefined): string {
  return (reason && LIVE_REASON_SENTENCES[reason]) || LIVE_UNAVAILABLE
}

/** Why the mint refused, as a sentence for the member: never a code, never an object. */
export function liveRefusalSentence(error: unknown): string {
  const detail = (error as { body?: { detail?: unknown } } | null)?.body?.detail
  const sentence = plainText(detail)
  if (sentence) return sentence
  if (!isCoded(detail)) return LIVE_UNAVAILABLE
  return LIVE_REASON_SENTENCES[detail.code] ?? plainText(detail.message) ?? LIVE_UNAVAILABLE
}
