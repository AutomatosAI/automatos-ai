/**
 * Theme voice, web quick-ask: the server voice port's options and its one mutable slot (server-speech.ts). The queue
 * is an immutable list replaced whole (speech-queue.ts). `moves` are the port's own steps bound to this context, so
 * its request half (server-speech-fetch.ts) and audio half (server-speech-audio.ts) move the queue on without
 * importing the port.
 */
import type { AudioPlayer } from './audio-player'
import type { QuickAskClock } from './clock'
import type { SynthesisPort } from './speaker'
import { findPiece, updatePiece, type Piece } from './speech-queue'
import type { ServerVoiceSession } from './voice-choice'
import type { FetchSpeech } from './voice-api'

export interface ServerSpeechOptions {
  readonly fetchSpeech: FetchSpeech
  readonly player: AudioPlayer
  /** The browser's voice; null when the browser has none (a piece the server can't speak then just ends). */
  readonly fallback: SynthesisPort | null
  readonly session: ServerVoiceSession
  readonly clock: QuickAskClock
}

export interface QueueMoves {
  /** The piece is over: out of the queue, its end told, and the queue moves on. */
  readonly finish: (id: number) => void
  /** The server won't speak the piece: the browser's voice will, in its turn. */
  readonly toBrowser: (id: number) => void
  /** The MP3s due are asked for, and the head plays, goes to the browser's voice or starts its wait. */
  readonly pump: () => void
}

export interface SpeechCtx {
  readonly options: ServerSpeechOptions
  readonly moves: QueueMoves
  state: { readonly ids: number; readonly queue: readonly Piece[] }
}

export const LOG_PREFIX = '[voice]'

export function find(ctx: SpeechCtx, id: number): Piece | undefined {
  return findPiece(ctx.state.queue, id)
}

export function setQueue(ctx: SpeechCtx, queue: readonly Piece[]): void {
  ctx.state = Object.freeze({ ...ctx.state, queue })
}

export function update(ctx: SpeechCtx, id: number, changes: Partial<Piece>): void {
  setQueue(ctx, updatePiece(ctx.state.queue, id, changes))
}

/** The piece needs nothing more from the network or the clock: its request aborted, its wait cancelled. */
export function settle(ctx: SpeechCtx, piece: Piece): void {
  piece.abort?.()
  if (piece.waitTimer !== null) ctx.options.clock.clearTimeout(piece.waitTimer)
}
