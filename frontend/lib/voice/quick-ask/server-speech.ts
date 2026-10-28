/**
 * Theme voice, web quick-ask: Auto's server voice (Kokoro, POST /api/voice/speech) as the speaker's SynthesisPort, the
 * browser's voice behind it. The shape of automatos-go's server-speaker.ts (27 Oct 2026), except that a piece the
 * server can't speak goes to the browser's voice on its own, and the next piece asks the server again.
 *
 *   speak → the piece joins the queue (speech-queue.ts); the MP3 of the head and of the piece after it is asked for
 *   (server-speech-fetch.ts), so the next sentence is synthesised while this one plays. The head plays once its MP3
 *   is in (server-speech-audio.ts), and waits for it SERVER_VOICE_WAIT_MS at most.
 *   A piece the server refuses, that is too slow, that never reaches it, or whose audio fails before it starts is
 *   spoken by the browser's voice with the same events, so the turn still finishes; with no browser voice it just
 *   ends. Once the page session knows the server can't speak here (tts_off, or no voice for the page's language),
 *   every piece goes straight to the browser's voice and none is asked of the server.
 *   cancel → every request aborted, the audio stopped and released, the browser's voice cancelled; the late answers
 *   and callbacks of the cancelled pieces find nothing and do nothing.
 *   prime → inside the hold's gesture: the audio unlocked, the browser's voice primed, the voice list asked for.
 */
import { hand, play } from './server-speech-audio'
import { find, LOG_PREFIX, setQueue, settle, update, type ServerSpeechOptions, type SpeechCtx } from './server-speech-ctx'
import { armWait, request } from './server-speech-fetch'
import type { SynthesisPort, UtteranceEvents } from './speaker'
import { newPiece, nextStep, toRequest, withoutPiece } from './speech-queue'

export type { ServerSpeechOptions } from './server-speech-ctx'

function finish(ctx: SpeechCtx, id: number): void {
  const piece = find(ctx, id)
  if (piece === undefined) return
  settle(ctx, piece)
  setQueue(ctx, withoutPiece(ctx.state.queue, id))
  piece.events.onEnd()
  pump(ctx)
}

function toBrowser(ctx: SpeechCtx, id: number): void {
  const piece = find(ctx, id)
  if (piece === undefined) return
  settle(ctx, piece)
  update(ctx, id, { stage: 'browser', clip: null, abort: null, waitTimer: null })
  pump(ctx)
}

function pump(ctx: SpeechCtx): void {
  toRequest(ctx.state.queue).forEach((piece) => request(ctx, piece))
  const step = nextStep(ctx.state.queue)
  if (step === null) return
  if (step.kind === 'due') return armWait(ctx, step.piece.id)
  if (step.kind === 'play') return play(ctx, step.piece)
  hand(ctx, step.piece)
}

function speak(ctx: SpeechCtx, text: string, events: UtteranceEvents): void {
  const id = ctx.state.ids + 1
  const piece = newPiece(id, text, events, ctx.options.session.serverless() ? 'browser' : 'waiting')
  ctx.state = Object.freeze({ ids: id, queue: Object.freeze([...ctx.state.queue, piece]) })
  pump(ctx)
}

function cancel(ctx: SpeechCtx): void {
  const { queue } = ctx.state
  setQueue(ctx, Object.freeze([]))
  queue.forEach((piece) => settle(ctx, piece))
  ctx.options.player.stop()
  ctx.options.fallback?.cancel()
}

function prime(ctx: SpeechCtx): void {
  try {
    ctx.options.player.unlock()
  } catch (error) {
    console.warn(LOG_PREFIX, 'audio playback could not be unlocked', error)
  }
  void ctx.options.session.voice()
  ctx.options.fallback?.prime()
}

/** The server voice as a SynthesisPort, with `fallback` (the browser's voice) for what it can't speak. */
export function serverSpeechPort(options: ServerSpeechOptions): SynthesisPort {
  const ctx: SpeechCtx = {
    options,
    moves: Object.freeze({
      finish: (id: number) => finish(ctx, id),
      toBrowser: (id: number) => toBrowser(ctx, id),
      pump: () => pump(ctx),
    }),
    state: Object.freeze({ ids: 0, queue: Object.freeze([]) }),
  }
  return Object.freeze({
    speak: (text: string, events: UtteranceEvents) => speak(ctx, text, events),
    cancel: () => cancel(ctx),
    prime: () => prime(ctx),
  })
}
