/**
 * Theme voice, web quick-ask: the server voice's queue (server-speech.ts), as pure functions over an immutable list.
 * A piece of Auto's reply goes waiting → loading (its MP3 asked for) → ready → playing, or, when the server can't
 * speak it, → browser → handed (the browser's voice has it). It leaves the queue when it ends. Order is strict: a
 * piece plays only at the head, and goes to the browser's voice only once every piece before it has (the browser
 * keeps its own utterances in order), so two voices never speak at once.
 */
import { SERVER_SPEECH_PREFETCH_AHEAD } from './config'
import type { UtteranceEvents } from './speaker'

export type PieceStage = 'waiting' | 'loading' | 'ready' | 'playing' | 'browser' | 'handed'

export interface Piece {
  readonly id: number
  readonly text: string
  readonly events: UtteranceEvents
  readonly stage: PieceStage
  /** The MP3, once it came. */
  readonly clip: Blob | null
  /** Aborts its request; null when none is in flight. */
  readonly abort: (() => void) | null
  /** The timer of its wait for the MP3, armed once it is due; null when none is armed. */
  readonly waitTimer: unknown
  /** Its audio started playing (onStart told). */
  readonly started: boolean
}

/** What the queue does next: hand a piece to the browser's voice, play the head, or start the head's wait. */
export type QueueStep = { readonly kind: 'hand' | 'play' | 'due'; readonly piece: Piece } | null

export function newPiece(id: number, text: string, events: UtteranceEvents, stage: PieceStage): Piece {
  return Object.freeze({ id, text, events, stage, clip: null, abort: null, waitTimer: null, started: false })
}

export function findPiece(queue: readonly Piece[], id: number): Piece | undefined {
  return queue.find((piece) => piece.id === id)
}

export function updatePiece(queue: readonly Piece[], id: number, changes: Partial<Piece>): readonly Piece[] {
  return Object.freeze(queue.map((piece) => (piece.id === id ? Object.freeze({ ...piece, ...changes }) : piece)))
}

export function withoutPiece(queue: readonly Piece[], id: number): readonly Piece[] {
  return Object.freeze(queue.filter((piece) => piece.id !== id))
}

/** The head and the SERVER_SPEECH_PREFETCH_AHEAD pieces after it whose MP3 is not asked for yet. */
export function toRequest(queue: readonly Piece[]): readonly Piece[] {
  return queue.slice(0, SERVER_SPEECH_PREFETCH_AHEAD + 1).filter((piece) => piece.stage === 'waiting')
}

/**
 * The first piece the browser's voice doesn't have yet decides: one for the browser's voice is handed to it; at the
 * head, a ready one plays and a loading one is due (its wait starts). A server piece behind pieces the browser's
 * voice is still speaking waits for them to end.
 */
export function nextStep(queue: readonly Piece[]): QueueStep {
  const index = queue.findIndex((piece) => piece.stage !== 'handed')
  const piece = queue[index]
  if (piece === undefined) return null
  if (piece.stage === 'browser') return { kind: 'hand', piece }
  if (index > 0) return null
  if (piece.stage === 'ready') return { kind: 'play', piece }
  if (piece.stage === 'loading' && piece.waitTimer === null) return { kind: 'due', piece }
  return null
}
