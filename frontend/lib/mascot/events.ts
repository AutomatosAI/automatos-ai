import type { MascotEvent } from './types'

/**
 * PRD-260 D5, D6 — the core chat's status and newest reply as Auto's events. Pure.
 *
 * `useChat` (lib/chat/hooks.ts, under useChatWithReceipts) has a status of
 * 'idle' | 'streaming' | 'error' (an awaited detached reply reads as
 * 'streaming'), and its errorCode names a blocked turn ('trial_exhausted'). The
 * words, the model's reasoning and a cap that ended the turn never reach its
 * data callback: useChat writes them onto the newest reply, so they are read
 * from it (mascotEventsFromReply). The data callback's parts are Auto's events
 * through lib/mascot/data-events.ts.
 */

export type ChatStatus = 'idle' | 'streaming' | 'error'

/** The message on Auto's error event when the turn was blocked (errorCode names why). */
export const BLOCKED_TURN = 'The turn was blocked'
/** The message on Auto's error event when the chat's status turned to error. */
export const FAILED_TURN = 'The turn failed'

/**
 * What a change of useChat's status means for Auto: into 'streaming' starts the
 * turn (a send; also a detached reply being awaited), into 'error' fails it, and
 * 'streaming' to 'idle' finishes it — or fails it, when an errorCode says the turn
 * was blocked (a failure and the end of the stream can land in one render).
 */
export function mascotEventsFromStatus(previous: ChatStatus, next: ChatStatus, errorCode: string | null): MascotEvent[] {
  if (previous === next) return []
  if (next === 'streaming') return [{ type: 'send' }]
  if (next === 'error') return [{ type: 'error', message: FAILED_TURN, code: errorCode }]
  if (previous !== 'streaming') return []
  if (errorCode !== null) return [{ type: 'error', message: BLOCKED_TURN, code: errorCode }]
  return [{ type: 'finish', finishReason: null }]
}

/**
 * The newest reply, as far as Auto cares: which one it is, whether its words
 * have started, whether one of its tool calls is still running, how much the
 * model has reasoned on it, and whether a cap ended its turn. useChat writes the
 * words, the reasoning and the cap onto the reply (lib/chat/hooks.ts) instead of
 * handing them to onData, so they are read from the reply itself.
 */
export type ReplyProgress = {
  readonly id: string | null
  readonly hasText: boolean
  readonly toolRunning: boolean
  /** The length of the reply's reasoning: its growth is the model thinking. */
  readonly reasoningChars: number
  /** A cap (limit_reached) ended the reply's turn. */
  readonly limitReached: boolean
}

export const NO_REPLY: ReplyProgress = { id: null, hasText: false, toolRunning: false, reasoningChars: 0, limitReached: false }

/** The reducer only reads that a text event came: its words are on the reply. */
const ANSWER_STARTED: MascotEvent = { type: 'text', text: '' }
/** The reducer only reads that the model reasons: its reasoning is on the reply. */
const MODEL_REASONING: MascotEvent = { type: 'reasoning', delta: '' }
/** The reducer only reads that a cap ended the turn: the reply shows its message. */
const CAP_REACHED: MascotEvent = { type: 'limit_reached', limit: '', value: 0, message: '' }

const sameReply = (previous: ReplyProgress, next: ReplyProgress): boolean => previous.id === next.id

/** The same reply's reasoning growing (between tool rounds too) is the model thinking. */
function reasoningEvents(previous: ReplyProgress, next: ReplyProgress): MascotEvent[] {
  return sameReply(previous, next) && next.reasoningChars > previous.reasoningChars ? [MODEL_REASONING] : []
}

/** The reply's words starting (the narration frame clears them each tool round, so each round's start counts). */
function answerEvents(previous: ReplyProgress, next: ReplyProgress): MascotEvent[] {
  if (!next.hasText) return []
  return sameReply(previous, next) && previous.hasText ? [] : [ANSWER_STARTED]
}

/** A cap ending the reply's turn, told once per reply whatever the status (the frame and the stream's end can land in one render). */
function capEvents(previous: ReplyProgress, next: ReplyProgress): MascotEvent[] {
  const alreadyTold = sameReply(previous, next) && previous.limitReached
  return next.limitReached && !alreadyTold ? [CAP_REACHED] : []
}

/**
 * What a change of the newest reply means for Auto (Play's spec: thinking while
 * the model reasons, the first words → idle, a cap → oops). While the chat
 * streams and no tool runs on the reply, its reasoning growing is one reasoning
 * event and its words starting one text event (both, in that order, when they
 * land in one render). Later words, words or reasoning while a tool runs (the
 * tool's state wins, and its end settles Auto) and any status other than
 * streaming map to nothing, except the cap: its transition is one limit_reached.
 */
export function mascotEventsFromReply(previous: ReplyProgress, next: ReplyProgress, status: ChatStatus): MascotEvent[] {
  const cap = capEvents(previous, next)
  if (status !== 'streaming' || next.toolRunning) return cap
  return [...reasoningEvents(previous, next), ...answerEvents(previous, next), ...cap]
}
