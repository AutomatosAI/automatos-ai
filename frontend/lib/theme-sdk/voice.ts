/**
 * Theme voice (TV-C5) — SDK area `voice` (SDK 1.2.0): talking to Auto by voice, through the host.
 *
 * The host runs the tab's one voice session (components/voice/voice-provider.tsx, mounted around the chat route and
 * every theme route): Auto Live (a call, on the plans that have it) or web quick-ask (hold to talk, on every plan and
 * edition). A theme reaches it only here:
 *
 * - `VoiceButton`: the button core draws (host-cards/voice-button.tsx), the press and every promise around it (the AI
 *   voice caption and the stop control in a protected surface, the microphone prompt, a failure with Try again);
 * - `useVoice()`: the session as a frozen, theme-neutral snapshot, and `stop()`;
 * - `readVoiceLevels()`: the member's and Auto's levels, 0 to 1, for an animation frame (a waveform, a mascot);
 * - `BriefVoice` (SDK 1.3.0, PRD-268 US-C21e): Listen, the host's control that reads the day's brief the host holds
 *   (host-cards/brief-voice.tsx), in the protected surface `brief-voice`; it never plays until pressed and never reads
 *   a theme's text.
 *
 * A theme may place, style and animate the button and call stop(). It never captures audio itself (getUserMedia,
 * SpeechRecognition, speechSynthesis: core's review blocks a theme that does, services/themes/audio_capture.py), never
 * hides the caption or the stop control, and never rewords the AI notice the host speaks once per session.
 *
 * Outside the host's voice provider (a page with no voice) useVoice answers the idle snapshot with nothing available
 * and the reason, and readVoiceLevels answers zeros; neither throws.
 */
import { useMemo } from 'react'

import { VOICE_BUTTON_WORDS } from '@/components/voice/voice-button-view'
import { type HostVoice, readHostVoiceLevels, useOptionalHostVoice } from '@/components/voice/voice-provider'
import { IDLE_VOICE, type VoiceMode, type VoicePhase } from '@/lib/voice/engine'

export { VoiceButton } from './host-cards/voice-button'
export type { VoiceButtonMode, VoiceButtonPress, VoiceButtonProps, VoiceButtonSize } from './host-cards/voice-button'
export { BriefVoice } from './host-cards/brief-voice'

/** connecting → listening (the microphone open) → thinking → speaking; ended (a call Auto Live ended), error. */
export type HostVoicePhase = VoicePhase
/** quick: hold to talk in the browser; live: an Auto Live call. */
export type HostVoiceMode = VoiceMode

/** What the host can offer here. */
export interface HostVoiceAvailability {
  /** Web quick-ask can run in this browser. */
  readonly quick: boolean
  /** Auto Live can start for this workspace, in this browser. */
  readonly live: boolean
  /** Why Live (or, outside the provider, voice) can't start, in the member's words; null when it can. */
  readonly reason: string | null
}

/** The tab's voice session, as a theme may see it. */
export interface HostVoiceState {
  readonly available: HostVoiceAvailability
  /** The session's engine; null while none runs. */
  readonly mode: HostVoiceMode | null
  readonly phase: HostVoicePhase
  readonly micOpen: boolean
  readonly muted: boolean
  /** What the member is saying or said last, as recognised so far. */
  readonly heard: string
  /** What Auto is saying or said last, as far as it has been spoken. */
  readonly reply: string
  readonly durationSec: number
  /** A plain sentence for the member, never a code. */
  readonly error: string | null
  /** The conversation the session talks into. */
  readonly chatId: string | null
  /** Ends the session or Auto's reply now. Always allowed. */
  readonly stop: () => void
}

/** The member's microphone level and Auto's output level, each 0 to 1. */
export interface HostVoiceLevels {
  readonly user: number
  readonly agent: number
}

const NO_OP = (): void => undefined
const LEVEL_MIN = 0
const LEVEL_MAX = 1

const NO_VOICE: HostVoiceState = Object.freeze({
  available: Object.freeze({ quick: false, live: false, reason: VOICE_BUTTON_WORDS.unavailable }),
  mode: IDLE_VOICE.mode,
  phase: IDLE_VOICE.phase,
  micOpen: IDLE_VOICE.micOpen,
  muted: IDLE_VOICE.muted,
  heard: IDLE_VOICE.heard,
  reply: IDLE_VOICE.reply,
  durationSec: IDLE_VOICE.durationSec,
  error: IDLE_VOICE.error,
  chatId: IDLE_VOICE.chatId,
  stop: NO_OP,
})

/** A copy of the host's view: only the snapshot's fields, the availability and stop, never the host itself. */
function stateOf(voice: HostVoice): HostVoiceState {
  const { snapshot, available } = voice
  return Object.freeze({
    available: Object.freeze({ quick: available.quick, live: available.live, reason: available.reason }),
    mode: snapshot.mode,
    phase: snapshot.phase,
    micOpen: snapshot.micOpen,
    muted: snapshot.muted,
    heard: snapshot.heard,
    reply: snapshot.reply,
    durationSec: snapshot.durationSec,
    error: snapshot.error,
    chatId: snapshot.chatId,
    stop: voice.stop,
  })
}

/** The tab's voice session, re-rendering when it changes; the idle snapshot (nothing available) outside the provider. */
export function useVoice(): HostVoiceState {
  const voice = useOptionalHostVoice()
  return useMemo(() => (voice === null ? NO_VOICE : stateOf(voice)), [voice])
}

function level(value: number): number {
  return Number.isFinite(value) ? Math.min(LEVEL_MAX, Math.max(LEVEL_MIN, value)) : LEVEL_MIN
}

/** The levels now, for code inside an animation frame: never triggers a render; zeros while no session runs. */
export function readVoiceLevels(): HostVoiceLevels {
  const { user, agent } = readHostVoiceLevels()
  return Object.freeze({ user: level(user), agent: level(agent) })
}
