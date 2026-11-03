/**
 * Theme voice (SDK 1.2.0) — SDK area `voice` (frontend/lib/theme-sdk/voice.ts): talking to Auto by voice, through the
 * host. The host runs the tab's one voice session: Auto Live (a call, on the plans that have it) or web quick-ask
 * (hold to talk, on every plan and edition).
 *
 * - `VoiceButton`: the button the host draws, with the press, the microphone prompt, connecting, the "AI voice
 *   assistant" caption and the stop control (a protected surface no theme stylesheet can hide), mute and the device
 *   for a live call, a failure with Try again, the silent-microphone notice and the state for screen readers. A theme
 *   places and styles it: `data-voice-state` (idle, connecting, listening, thinking, speaking, ended, error),
 *   `data-voice-mode` (quick, live), `data-voice-press`, `data-voice-size` and `data-voice-mic` (open, closed) on the
 *   root (`[data-voice-button="root"]`), and the custom properties `--voice-button-size`, `--voice-button-radius`,
 *   `--voice-button-bg`, `--voice-button-fg`, `--voice-button-active-bg`, `--voice-button-active-fg` and
 *   `--voice-button-ring`. A prop of the wrong kind throws, naming the prop.
 * - `useVoice()`: the session as a frozen snapshot, and `stop()`.
 * - `readVoiceLevels()`: the member's and Auto's levels, 0 to 1, read inside an animation frame (never a render).
 * - `BriefVoice` (SDK 1.3.0): Listen, the host's control that reads the day's brief the host holds in Auto's voice,
 *   with the "AI voice assistant" caption and Stop while it reads, in a protected surface no theme stylesheet can hide.
 *   It takes no text: a theme never speaks. It never plays until pressed, it is disabled until the host holds the
 *   day's brief, and it waits while the tab's voice session runs. Its root (`[data-brief-voice="root"]`) carries
 *   `data-voice-speaking="true"` while Auto reads, which a theme's speaking line follows. SDK 1.4.0: while Auto reads
 *   one of the brief's rows, the root also carries `data-voice-row="<index>"`, and `onSpeakingRow` hears the index,
 *   then null between rows, during the greeting and once the reading ends or stops. The index is the row's place in
 *   the brief's rows, section by section in the sections' order (needs_you, suggestions, today, since, unchecked):
 *   the number Auto says, less one.
 *
 * A theme never captures audio itself (getUserMedia, SpeechRecognition, speechSynthesis: the host's review blocks a
 * theme that does), never hides the caption or the stop control, and never rewords the AI notice. Outside the host's
 * voice (a page with none) useVoice answers the idle snapshot with nothing available and the reason, readVoiceLevels
 * answers zeros, and VoiceButton is drawn disabled; none throws.
 */
import type { ReactElement } from 'react'

/** connecting → listening (the microphone open) → thinking → speaking; ended (a call Auto Live ended), error. */
export type HostVoicePhase = 'idle' | 'connecting' | 'listening' | 'thinking' | 'speaking' | 'ended' | 'error'

/** quick: hold to talk in the browser; live: an Auto Live call. */
export type HostVoiceMode = 'quick' | 'live'

/** What the host can offer here. */
export interface HostVoiceAvailability {
  /** Web quick-ask can run in this browser. */
  readonly quick: boolean
  /** Auto Live can start for this workspace, in this browser. */
  readonly live: boolean
  /** Why Live (or, outside the host's voice, voice) can't start, in the member's words; null when it can. */
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

/** `auto`: Live when it can start for this workspace, else quick-ask. */
export type VoiceButtonMode = 'auto' | 'quick' | 'live'

/** `hold`: hold to talk; `toggle`: press to start, press again to send or end. */
export type VoiceButtonPress = 'hold' | 'toggle'

export type VoiceButtonSize = 'sm' | 'md' | 'lg'

export interface VoiceButtonProps {
  /** The conversation the button talks into (null: a new chat); left out: the conversation on screen. */
  readonly conversation?: string | null
  /** Default `auto`. */
  readonly mode?: VoiceButtonMode
  /** Default `hold`. */
  readonly press?: VoiceButtonPress
  /** A KeyboardEvent.code that presses the button anywhere but a field the member types in; default `Space`, null for none. */
  readonly hotkey?: string | null
  /** Default `md`. */
  readonly size?: VoiceButtonSize
  /** The button's name for everyone; default: what a press does ("Hold to talk", "Talk to Auto"). */
  readonly label?: string
}

/** The tab's voice session, re-rendering when it changes. */
export declare function useVoice(): HostVoiceState

/** The levels now, for code inside an animation frame; zeros while no session runs. */
export declare function readVoiceLevels(): HostVoiceLevels

/** The host's voice button. */
export declare function VoiceButton(props: VoiceButtonProps): ReactElement

export interface BriefVoiceProps {
  /**
   * Hears the brief row Auto starts reading (its index, in the sections' order), and null once it reads none (taken
   * off the page included), each time that changes (SDK 1.4.0). A value that isn't a function throws, naming the
   * prop; a handler that throws is logged, and Auto reads on.
   */
  readonly onSpeakingRow?: (row: number | null) => void
}

/** The host's Listen for the day's brief the host holds. */
export declare function BriefVoice(props?: BriefVoiceProps): ReactElement
