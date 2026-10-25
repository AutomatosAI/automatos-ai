/**
 * The contract every voice engine the host runs keeps (Live through Retell, quick-ask in the browser), so the host's
 * voice provider, the theme SDK's voice area and Studio's chat read one shape whatever runs underneath.
 */
export type VoicePhase = 'idle' | 'connecting' | 'listening' | 'thinking' | 'speaking' | 'ended' | 'error'
export type VoiceMode = 'quick' | 'live'

export interface VoiceSnapshot {
  readonly mode: VoiceMode | null
  readonly phase: VoicePhase
  readonly micOpen: boolean
  readonly muted: boolean
  /** What the member is saying or said last, as recognised so far. */
  readonly heard: string
  /** What Auto is saying or said last, as far as it has been spoken. */
  readonly reply: string
  readonly durationSec: number
  /** A plain sentence for the member, never a code or an object. */
  readonly error: string | null
  readonly chatId: string | null
}

export interface VoiceLevels {
  /** The member's microphone level, 0 to 1. */
  readonly user: number
  /** Auto's output level, 0 to 1. */
  readonly agent: number
}

export interface VoiceEngine {
  /** Starts a session bound to the conversation (null: a new one). Must be called from a user gesture. */
  start(options: { readonly chatId: string | null }): Promise<void>
  /** Hold-to-talk: the member let go; capture ends and the turn is sent. Live ignores it. */
  release(): void
  /** Ends the session or the reply now. Always allowed. */
  stop(): void
  snapshot(): VoiceSnapshot
  subscribe(listener: (snapshot: VoiceSnapshot) => void): () => void
  /** Read inside an animation frame; never triggers a render. */
  levels(): VoiceLevels
  dispose(): void
}

export const IDLE_VOICE: VoiceSnapshot = Object.freeze({
  mode: null, phase: 'idle', micOpen: false, muted: false, heard: '', reply: '', durationSec: 0, error: null, chatId: null,
})
