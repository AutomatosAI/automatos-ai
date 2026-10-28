/**
 * Theme voice, web quick-ask: the server voice port's audio half (server-speech.ts). The head's MP3 plays on the
 * player: onStart when the audio starts, onBoundary on its timeupdate (Auto's level moves), onEnd when it ends. Audio
 * that fails before it starts goes to the browser's voice; after it started, the piece just ends (it isn't said
 * twice). A piece for the browser's voice is handed to it with the same events and ends when the browser says so.
 * Events of a piece no longer in the queue (cancelled) do nothing.
 */
import { find, LOG_PREFIX, update, type SpeechCtx } from './server-speech-ctx'
import type { Piece } from './speech-queue'

/** The head, while its audio plays. */
function playing(ctx: SpeechCtx, id: number): Piece | null {
  const head = ctx.state.queue[0]
  return head?.id === id && head.stage === 'playing' ? head : null
}

function audioStarted(ctx: SpeechCtx, id: number): void {
  const piece = playing(ctx, id)
  if (piece === null || piece.started) return
  update(ctx, id, { started: true })
  piece.events.onStart()
}

function audioFailed(ctx: SpeechCtx, id: number, error: unknown): void {
  const piece = playing(ctx, id)
  if (piece === null) return
  console.warn(LOG_PREFIX, "the server voice's audio did not play", error)
  if (piece.started) ctx.moves.finish(id)
  else ctx.moves.toBrowser(id)
}

/** The ready head plays. */
export function play(ctx: SpeechCtx, piece: Piece): void {
  const { id, clip, events } = piece
  if (clip === null) return ctx.moves.toBrowser(id)
  update(ctx, id, { stage: 'playing', clip: null })
  try {
    ctx.options.player.play(clip, {
      onPlaying: () => audioStarted(ctx, id),
      onProgress: () => {
        if (playing(ctx, id) !== null) events.onBoundary()
      },
      onEnded: () => {
        // A clip too short to say it was playing still started.
        audioStarted(ctx, id)
        if (playing(ctx, id) !== null) ctx.moves.finish(id)
      },
      onFailed: (error: unknown) => audioFailed(ctx, id, error),
    })
  } catch (error) {
    audioFailed(ctx, id, error)
  }
}

/** The browser's voice speaks the piece (with none, it just ends); the queue moves on when it ends. */
export function hand(ctx: SpeechCtx, piece: Piece): void {
  const { id, text, events } = piece
  update(ctx, id, { stage: 'handed' })
  const { fallback } = ctx.options
  if (fallback === null) return ctx.moves.finish(id)
  const live = (): boolean => find(ctx, id) !== undefined
  try {
    fallback.speak(text, {
      onStart: () => {
        if (live()) events.onStart()
      },
      onBoundary: () => {
        if (live()) events.onBoundary()
      },
      onEnd: () => ctx.moves.finish(id),
    })
  } catch (error) {
    console.warn(LOG_PREFIX, "the browser's voice would not take a sentence", error)
    return ctx.moves.finish(id)
  }
  ctx.moves.pump()
}
