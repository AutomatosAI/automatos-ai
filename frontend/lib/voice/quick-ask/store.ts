/**
 * Theme voice, web quick-ask: the engine's one mutable slot (automatos-go's quick-ask-store.ts, PR #15). Everything
 * the engine knows is an immutable record replaced whole by set(); the voice snapshot changes only through the
 * state machine (dispatch), and listeners hear each new snapshot once. A listener that throws is logged, never
 * allowed to break the engine or the other listeners.
 */
import type { VoiceSnapshot } from '../engine'
import type { QuickAskDeps } from './deps'
import type { RecognitionPort } from './recognition'
import type { ReplyFeed } from './reply-feed'
import { createSpeaker, type Speaker, type SpeakerEvent, type SynthesisPort } from './speaker'
import { INITIAL_QUICK_ASK, quickAskReducer, type QuickAskEvent, type QuickAskState } from './state'

/** One hold to talk: the browser's recognition for it, the words so far and its timers. */
export interface Hold {
  readonly id: number
  readonly port: RecognitionPort
  readonly words: string
  /** The member let go (or the hold ran out); the last words and the end are awaited. */
  readonly released: boolean
  /** When the browser began capturing; null until it does. */
  readonly capturedAt: number | null
  readonly maxTimer: unknown
  readonly graceTimer: unknown
  readonly tickTimer: unknown
}

/** The turn quick-ask sent, while its reply streams. */
export interface Turn {
  readonly id: number
  readonly controller: AbortController
  readonly feed: ReplyFeed
  readonly agentName: string | null
}

export interface Inner {
  readonly state: QuickAskState
  readonly hold: Hold | null
  readonly turn: Turn | null
  /** This engine queued the AI notice (the tab's sessionStorage remembers it too, when it can). */
  readonly noticed: boolean
  /** When the member's words last changed: the envelope of their level. */
  readonly lastWordsAt: number | null
  readonly serial: number
  readonly disposed: boolean
}

export interface Core {
  readonly deps: QuickAskDeps
  readonly speaker: Speaker
  readonly listeners: Set<(snapshot: VoiceSnapshot) => void>
  inner: Inner
}

const LOG_PREFIX = '[voice]'

const INITIAL_INNER: Inner = Object.freeze({
  state: INITIAL_QUICK_ASK,
  hold: null,
  turn: null,
  noticed: false,
  lastWordsAt: null,
  serial: 0,
  disposed: false,
})

/** Speech for a browser without it: never used (start refuses first), and every sentence ends at once. */
const NO_SYNTHESIS: SynthesisPort = Object.freeze({
  speak: (_text: string, events: { readonly onEnd: () => void }) => events.onEnd(),
  cancel: () => undefined,
  prime: () => undefined,
})

export function createCore(deps: QuickAskDeps, onSpeaker: (core: Core, event: SpeakerEvent) => void): Core {
  const core: Core = {
    deps,
    speaker: createSpeaker(deps.synthesis ?? NO_SYNTHESIS, deps.clock, (event) => onSpeaker(core, event)),
    listeners: new Set(),
    inner: INITIAL_INNER,
  }
  return core
}

export function get(core: Core): Inner {
  return core.inner
}

export function set(core: Core, changes: Partial<Inner>): void {
  core.inner = Object.freeze({ ...core.inner, ...changes })
}

/** A new id for a hold or a turn. */
export function nextId(core: Core): number {
  const serial = core.inner.serial + 1
  set(core, { serial })
  return serial
}

function publish(core: Core, snapshot: VoiceSnapshot): void {
  Array.from(core.listeners).forEach((listener) => {
    try {
      listener(snapshot)
    } catch (error) {
      console.error(LOG_PREFIX, 'a voice listener failed', error)
    }
  })
}

/** The state machine's next state; listeners hear the snapshot when it changed. */
export function dispatch(core: Core, event: QuickAskEvent): void {
  const before = core.inner.state
  const state = quickAskReducer(before, event)
  if (state === before) return
  set(core, { state })
  if (state.view !== before.view) publish(core, state.view)
}
