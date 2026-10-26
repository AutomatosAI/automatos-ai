/**
 * Theme voice — what the voice button core draws for themes (voice-button.tsx) shows, as data: its words, the CSS
 * custom properties a theme styles it with, and the view of the tab's session from the button's side (whether the
 * session is the button's, its state and engine for the data attributes, the screen reader's line, the title).
 * Pure, so every state is tested without a browser.
 */
import type { HostVoice } from '@/components/voice/voice-provider'
import { IDLE_VOICE, type VoiceMode, type VoicePhase, type VoiceSnapshot } from '@/lib/voice/engine'
import { type VoiceAvailability, resolveVoiceMode } from '@/lib/voice/host/availability'
import { ORB_STATE_LABELS } from '@/lib/voice/orb-state'

import { MIC_LIVE, MIC_MUTED, STOP } from './voice-controls'
import { type VoicePressKind, type VoicePressOptions, ownsSession } from './voice-press'

export type VoiceButtonSize = 'sm' | 'md' | 'lg'

/** Every word the button says. Auto is a name. */
export const VOICE_BUTTON_WORDS = Object.freeze({
  hold: 'Hold to talk',
  toggle: 'Talk to Auto',
  unavailable: "Voice isn't available on this page.",
  allowMic: 'Allow the microphone if your browser asks.',
  stop: STOP,
  close: 'Close voice',
  failed: 'Voice stopped. Try again.',
})

/** The button's label when the theme gives none: what a press does. */
export const DEFAULT_LABELS: Readonly<Record<VoicePressKind, string>> = Object.freeze({
  hold: VOICE_BUTTON_WORDS.hold,
  toggle: VOICE_BUTTON_WORDS.toggle,
})

/**
 * The CSS custom properties a theme styles the button with (set on the button or any ancestor). Each has the host's
 * own default, from the theme tokens, so an unstyled button still reads as Studio's.
 */
export const VOICE_BUTTON_CSS_VARS = Object.freeze([
  '--voice-button-size',
  '--voice-button-radius',
  '--voice-button-bg',
  '--voice-button-fg',
  '--voice-button-active-bg',
  '--voice-button-active-fg',
  '--voice-button-ring',
] as const)

export interface VoiceButtonView {
  /** The tab's session is this button's. */
  readonly owned: boolean
  /** data-voice-state: the session's phase while it is the button's, else idle. */
  readonly state: VoicePhase
  /** data-voice-mode: the session's engine while it is the button's, else the one a press starts. */
  readonly mode: VoiceMode
  readonly micOpen: boolean
  /** What a screen reader hears: the state, and whether the microphone is live. */
  readonly status: string
  /** The button's title: why voice can't start, when a reason is known; else its label. */
  readonly title: string
}

const NO_VOICE: VoiceAvailability = Object.freeze({ quick: false, live: false, reason: VOICE_BUTTON_WORDS.unavailable })

/** The session's state in words, and the microphone's when it is open; nothing at rest. */
export function voiceStatusWords(snapshot: VoiceSnapshot): string {
  if (snapshot.phase === 'idle') return ''
  const state = snapshot.phase === 'error' && snapshot.error ? snapshot.error : ORB_STATE_LABELS[snapshot.phase]
  if (!snapshot.micOpen) return state
  return `${state}. ${snapshot.muted ? MIC_MUTED : MIC_LIVE}`
}

function reasonOf(voice: HostVoice | null, next: VoiceMode, available: VoiceAvailability): string | null {
  if (voice === null) return VOICE_BUTTON_WORDS.unavailable
  return next === 'live' ? available.reason : null
}

/** The tab's session as the button shows it (no host voice: idle, with the reason voice isn't here). */
export function voiceButtonView(voice: HostVoice | null, options: VoicePressOptions, label: string): VoiceButtonView {
  const available = voice?.available ?? NO_VOICE
  const snapshot = voice?.snapshot ?? IDLE_VOICE
  const owned = ownsSession(snapshot, options)
  const next = resolveVoiceMode(options.mode, available)
  return Object.freeze({
    owned,
    state: owned ? snapshot.phase : 'idle',
    mode: owned && snapshot.mode !== null ? snapshot.mode : next,
    micOpen: owned && snapshot.micOpen,
    status: owned ? voiceStatusWords(snapshot) : '',
    title: reasonOf(voice, next, available) ?? label,
  })
}
