/**
 * Theme voice, web quick-ask: the one state machine (automatos-go's voice-reducer.ts, PR #15, cut to what a browser
 * hold needs). Pure: the engine feeds it what the recognition, the chat turn and the speaker say, and publishes
 * `view`, the voice contract's snapshot. An event that changes nothing returns the same state object.
 *
 *   start → connecting (the last turn's words and error cleared); capture_started → listening, micOpen;
 *   words → heard (while connecting or listening); sent → thinking; queued: sentences wait for the speaker;
 *   sentence → speaking, reply grows as spoken; drained → back to thinking while the turn still streams, else idle;
 *   finished → idle when nothing is queued; failed → error with its sentence; stopped → idle with the words kept.
 *
 * micOpen opens only when the browser says capture began, and closes when it says capture ended or when the engine
 * ends the hold itself (sent, failed, stopped): never from the phase.
 */
import { IDLE_VOICE, type VoicePhase, type VoiceSnapshot } from '../engine'

export interface QuickAskState {
  readonly view: VoiceSnapshot
  /** The sent turn is still streaming. */
  readonly streaming: boolean
  /** Auto's voice is going or about to: sentences queued and not all ended. */
  readonly speaking: boolean
}

export type QuickAskEvent =
  | { readonly type: 'start'; readonly chatId: string | null }
  | { readonly type: 'capture_started' }
  | { readonly type: 'capture_ended' }
  | { readonly type: 'words'; readonly text: string }
  | { readonly type: 'duration'; readonly seconds: number }
  | { readonly type: 'sent'; readonly text: string }
  | { readonly type: 'chat_id'; readonly chatId: string }
  /** Sentences went to the speaker (said before they are handed over: a refused one drains at once). */
  | { readonly type: 'queued' }
  | { readonly type: 'sentence'; readonly text: string }
  | { readonly type: 'drained' }
  | { readonly type: 'finished' }
  | { readonly type: 'failed'; readonly error: string }
  | { readonly type: 'stopped' }

type EventOf<K extends QuickAskEvent['type']> = Extract<QuickAskEvent, { type: K }>
type Handler<K extends QuickAskEvent['type']> = (state: QuickAskState, event: EventOf<K>) => QuickAskState
type Handlers = { readonly [K in QuickAskEvent['type']]: Handler<K> }

export const INITIAL_QUICK_ASK: QuickAskState = Object.freeze({ view: IDLE_VOICE, streaming: false, speaking: false })

/** The state with these changes, or the same object when none changes anything. */
function patch(state: QuickAskState, view: Partial<VoiceSnapshot>, flags: Partial<QuickAskState> = {}): QuickAskState {
  const viewKeys = Object.keys(view) as (keyof VoiceSnapshot)[]
  const flagKeys = Object.keys(flags) as (keyof QuickAskState)[]
  const changed = viewKeys.some((key) => state.view[key] !== view[key]) || flagKeys.some((key) => state[key] !== flags[key])
  if (!changed) return state
  return Object.freeze({ ...state, ...flags, view: Object.freeze({ ...state.view, ...view }) })
}

const HEARING: ReadonlySet<VoicePhase> = new Set<VoicePhase>(['connecting', 'listening'])
const REPLYING: ReadonlySet<VoicePhase> = new Set<VoicePhase>(['thinking', 'speaking'])

function start(state: QuickAskState, event: EventOf<'start'>): QuickAskState {
  const view = { ...IDLE_VOICE, mode: 'quick' as const, phase: 'connecting' as const, chatId: event.chatId }
  return patch(state, view, { streaming: false, speaking: false })
}

function captureStarted(state: QuickAskState): QuickAskState {
  return patch(state, state.view.phase === 'connecting' ? { phase: 'listening', micOpen: true } : { micOpen: true })
}

function sentence(state: QuickAskState, event: EventOf<'sentence'>): QuickAskState {
  if (!REPLYING.has(state.view.phase)) return state
  const reply = state.view.reply ? `${state.view.reply} ${event.text}` : event.text
  return patch(state, { phase: 'speaking', reply }, { speaking: true })
}

/** The speaker's queue ran out: Auto is done when the turn has finished, else still working on it. */
function drained(state: QuickAskState): QuickAskState {
  if (!REPLYING.has(state.view.phase)) return patch(state, {}, { speaking: false })
  return patch(state, { phase: state.streaming ? 'thinking' : 'idle' }, { speaking: false })
}

function finished(state: QuickAskState): QuickAskState {
  const done = state.view.phase === 'thinking' && !state.speaking
  return patch(state, done ? { phase: 'idle' } : {}, { streaming: false })
}

const HANDLERS: Handlers = {
  start,
  capture_started: captureStarted,
  capture_ended: (state) => patch(state, { micOpen: false }),
  words: (state, event) => (HEARING.has(state.view.phase) ? patch(state, { heard: event.text }) : state),
  duration: (state, event) => patch(state, { durationSec: event.seconds }),
  sent: (state, event) => patch(state, { phase: 'thinking', heard: event.text, reply: '', micOpen: false }, { streaming: true }),
  chat_id: (state, event) => patch(state, { chatId: event.chatId }),
  queued: (state) => (REPLYING.has(state.view.phase) ? patch(state, {}, { speaking: true }) : state),
  sentence,
  drained,
  finished,
  failed: (state, event) =>
    patch(state, { phase: 'error', error: event.error, micOpen: false }, { streaming: false, speaking: false }),
  stopped: (state) => patch(state, { phase: 'idle', error: null, micOpen: false }, { streaming: false, speaking: false }),
}

export function quickAskReducer(state: QuickAskState, event: QuickAskEvent): QuickAskState {
  // TypeScript cannot correlate HANDLERS[type] with the narrowed event; the mapped type checks each entry.
  const handle = HANDLERS[event.type] as Handler<QuickAskEvent['type']>
  return handle(state, event)
}
