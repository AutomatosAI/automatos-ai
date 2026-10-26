/**
 * Theme voice — which voice the host can offer: Auto Live when the workspace may use it now (its plan, the platform
 * switch, the Retell credentials, the workspace toggle and the month's minutes: `GET /api/voice/live-status`) and the
 * browser can capture; web quick-ask when the browser has speech recognition and synthesis (lib/voice/support.ts).
 *
 * Availability is what the host offers and what a start in `auto` mode picks; the mint stays the truth, so asking for
 * Live when it isn't available still tries, and its refusal is the member's sentence.
 */
import { apiClient } from '@/lib/api-client'
import type { VoiceMode } from '@/lib/voice/engine'
import { LIVE_BROWSER_UNSUPPORTED, LIVE_STATUS_UNREAD, liveUnavailableSentence } from '@/lib/voice/live/refusal'

export const LIVE_STATUS_PATH = '/api/voice/live-status'

export interface VoiceAvailability {
  /** Web quick-ask can run in this browser. */
  readonly quick: boolean
  /** Auto Live can start for this workspace, in this browser. */
  readonly live: boolean
  /** Why Live can't start, in the member's words; null when it can, or while that is being checked. */
  readonly reason: string | null
}

/** Which voice a start asks for: one engine, or `auto` (Live when available, else quick-ask). */
export type VoiceStartMode = VoiceMode | 'auto'

export type LiveStatusReading =
  | { readonly state: 'checking' }
  | { readonly state: 'failed' }
  | { readonly state: 'read'; readonly available: boolean; readonly reason: string | null }

export const LIVE_STATUS_CHECKING: LiveStatusReading = Object.freeze({ state: 'checking' })
const LIVE_STATUS_FAILED: LiveStatusReading = Object.freeze({ state: 'failed' })

interface LiveStatusAnswer {
  readonly available?: unknown
  readonly reason?: unknown
}

/** live-status's answer, checked: a failed read says so (and why, in the console), never throws. */
export async function readLiveStatus(): Promise<LiveStatusReading> {
  try {
    const answer = await apiClient.request<LiveStatusAnswer>(LIVE_STATUS_PATH)
    return Object.freeze({
      state: 'read',
      available: answer?.available === true,
      reason: typeof answer?.reason === 'string' ? answer.reason : null,
    })
  } catch (error) {
    console.warn("[voice] Auto Live's status could not be read", error)
    return LIVE_STATUS_FAILED
  }
}

function liveReason(reading: LiveStatusReading, liveBrowser: boolean): string | null {
  if (reading.state === 'checking') return null
  if (reading.state === 'failed') return LIVE_STATUS_UNREAD
  if (!reading.available) return liveUnavailableSentence(reading.reason)
  return liveBrowser ? null : LIVE_BROWSER_UNSUPPORTED
}

/** What the host can offer, from live-status and what the browser can do. */
export function voiceAvailability(reading: LiveStatusReading, quick: boolean, liveBrowser: boolean): VoiceAvailability {
  const live = reading.state === 'read' && reading.available && liveBrowser
  return Object.freeze({ quick, live, reason: live ? null : liveReason(reading, liveBrowser) })
}

/**
 * The engine a start runs: the one asked for; for `auto`, Live when it can start, else quick-ask when the browser
 * can, else Live (its mint tells the member why it can't).
 */
export function resolveVoiceMode(requested: VoiceStartMode, available: VoiceAvailability): VoiceMode {
  if (requested !== 'auto') return requested
  if (available.live) return 'live'
  return available.quick ? 'quick' : 'live'
}
