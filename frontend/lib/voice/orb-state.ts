/**
 * PRD-207 S5 — the presence orb's pure state machine.
 *
 * Everything time-based takes an explicit `now` (ms) so the reducer is
 * deterministic and unit-testable — no Date.now() inside. The hook feeds it
 * SDK events; the canvas reads the derived state + levels each frame.
 *
 * Dead-air honesty: when the user has finished speaking and Auto's audio
 * hasn't started (tool calls take seconds), the orb must show *thinking* —
 * never silent limbo (§4-S5).
 */

export type OrbState =
  | 'idle'        // mounted, no call
  | 'connecting'  // mint + WebRTC setup
  | 'listening'   // live, Auto silent, floor is the user's
  | 'thinking'    // live, nobody audibly speaking after user input — Auto is working
  | 'speaking'    // Auto's voice is out
  | 'ended'       // call closed normally
  | 'error'

export interface OrbSnapshot {
  state: OrbState
  /** True once agent_start_talking has fired and not yet stopped. */
  agentTalking: boolean
  /** ms timestamp of the last user speech above threshold (0 = never). */
  lastUserVoiceAt: number
  /** ms timestamp agent last stopped talking (0 = never). */
  lastAgentStopAt: number
}

export type OrbEvent =
  | { type: 'call_connecting' }
  | { type: 'call_started' }
  | { type: 'call_ended' }
  | { type: 'error' }
  | { type: 'agent_start_talking' }
  | { type: 'agent_stop_talking'; now: number }
  | { type: 'user_voice'; now: number } // user level crossed the speech threshold
  | { type: 'tick'; now: number }       // periodic re-derivation while live

/** RMS of one raw-PCM frame, clamped to [0, 1]. */
export function rmsLevel(samples: Float32Array | number[]): number {
  const n = samples.length
  if (!n) return 0
  let sum = 0
  for (let i = 0; i < n; i++) {
    const s = (samples as Float32Array)[i]
    sum += s * s
  }
  return Math.min(1, Math.sqrt(sum / n))
}

/** A user level above this counts as speech for thinking-derivation. */
export const USER_SPEECH_THRESHOLD = 0.045
/** Dead-air after the user stops before we admit Auto is "thinking". */
export const THINKING_AFTER_MS = 700

export const initialOrb: OrbSnapshot = {
  state: 'idle',
  agentTalking: false,
  lastUserVoiceAt: 0,
  lastAgentStopAt: 0,
}

function deriveLiveState(snap: OrbSnapshot, now: number): OrbState {
  if (snap.agentTalking) return 'speaking'
  const heardUser = snap.lastUserVoiceAt > 0
  const userQuietFor = now - snap.lastUserVoiceAt
  // The user said something, has gone quiet, and Auto's audio hasn't started:
  // that gap is Auto working (streaming/tools) — show it honestly.
  if (heardUser && userQuietFor >= THINKING_AFTER_MS && snap.lastUserVoiceAt > snap.lastAgentStopAt) {
    return 'thinking'
  }
  return 'listening'
}

export function orbReducer(prev: OrbSnapshot, event: OrbEvent): OrbSnapshot {
  switch (event.type) {
    case 'call_connecting':
      return { ...initialOrb, state: 'connecting' }
    case 'call_started':
      return { ...prev, state: 'listening', agentTalking: false }
    case 'call_ended':
      return { ...prev, state: 'ended', agentTalking: false }
    case 'error':
      return { ...prev, state: 'error', agentTalking: false }
    case 'agent_start_talking': {
      if (prev.state === 'idle' || prev.state === 'ended' || prev.state === 'error') return prev
      return { ...prev, agentTalking: true, state: 'speaking' }
    }
    case 'agent_stop_talking': {
      if (prev.state === 'idle' || prev.state === 'ended' || prev.state === 'error') return prev
      const next = { ...prev, agentTalking: false, lastAgentStopAt: event.now }
      return { ...next, state: deriveLiveState(next, event.now) }
    }
    case 'user_voice': {
      if (prev.state === 'idle' || prev.state === 'ended' || prev.state === 'error' || prev.state === 'connecting') {
        return prev
      }
      const next = { ...prev, lastUserVoiceAt: event.now }
      return { ...next, state: deriveLiveState(next, event.now) }
    }
    case 'tick': {
      if (prev.state !== 'listening' && prev.state !== 'thinking') return prev
      return { ...prev, state: deriveLiveState(prev, event.now) }
    }
    default:
      return prev
  }
}

/** Screen-reader + caption line for each orb state. */
export const ORB_STATE_LABELS: Record<OrbState, string> = {
  idle: 'Voice is off',
  connecting: 'Connecting…',
  listening: 'Auto is listening',
  thinking: 'Auto is thinking…',
  speaking: 'Auto is speaking',
  ended: 'Call ended',
  error: 'Voice connection error',
}
