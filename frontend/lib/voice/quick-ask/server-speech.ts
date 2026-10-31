/**
 * Theme voice, web quick-ask: Auto's server voice (Kokoro, POST /api/voice/speech) as the speaker's SynthesisPort, the
 * browser's voice behind it. The rule of automatos-go's server-speaker.ts (27 Oct 2026): one answer never goes back
 * and forth between two voices.
 *
 *   speak → the piece joins the queue (speech-queue.ts); the MP3 of the head and of the piece after it is asked for
 *   (server-speech-fetch.ts), so the next sentence is synthesised while this one plays. The head plays once its MP3
 *   is in (server-speech-audio.ts), and waits for it SERVER_VOICE_WAIT_MS at most.
 *   fail → the first piece the server voice fails (a refusal, a network error, too slow once due, audio that won't
 *   play) and every later piece of the same answer are spoken by the browser's voice with the same events, so the
 *   turn still finishes (with no browser voice they just end); the requests in flight are aborted and no more speech
 *   is asked for in this answer. Pieces before it keep the server voice, and the browser's voice speaks only once
 *   they have ended. An answer ends at the next prime() (each hold asks a new question) or cancel(), never when the
 *   queue runs dry: a reply streams sentence by sentence, so the queue can empty mid-answer. The next answer asks the
 *   server again. Once the page session knows the server can't speak here (tts_off, or no voice for the page's
 *   language), every piece of every answer goes straight to the browser's voice.
 *   cancel → every request aborted, the audio stopped and released, the browser's voice cancelled; the late answers
 *   and callbacks of the cancelled pieces find nothing and do nothing.
 *   prime → inside the hold's gesture: the audio unlocked, the browser's voice primed, the voice list asked for.
 */
import { hand, play } from './server-speech-audio'
import { find, LOG_PREFIX, setQueue, settle, type ServerSpeechOptions, type SpeechCtx } from './server-speech-ctx'
import { armWait, request } from './server-speech-fetch'
import type { SynthesisPort, UtteranceEvents } from './speaker'
import { newPiece, nextStep, toBrowserFrom, toRequest, withoutPiece } from './speech-queue'

export type { ServerSpeechOptions } from './server-speech-ctx'

function finish(ctx: SpeechCtx, id: number): void {
  const piece = find(ctx, id)
  if (piece === undefined) return
  settle(ctx, piece)
  setQueue(ctx, withoutPiece(ctx.state.queue, id))
  piece.events.onEnd()
  pump(ctx)
}

/** The server voice failed the piece: it (unless its audio started) and the rest of the answer go to the browser. */
function fail(ctx: SpeechCtx, id: number): void {
  const { queue } = ctx.state
  const index = queue.findIndex((piece) => piece.id === id)
  if (index < 0) return
  queue.slice(index).forEach((piece) => settle(ctx, piece))
  ctx.state = Object.freeze({ ...ctx.state, answerFailed: true, queue: toBrowserFrom(queue, index) })
  console.warn(LOG_PREFIX, "the browser's voice speaks the rest of this answer")
  if (queue[index].started) return finish(ctx, id)
  pump(ctx)
}

/** A new answer: the server voice is asked again (unless the page session knows it can't speak here). */
function newAnswer(ctx: SpeechCtx): void {
  ctx.state = Object.freeze({ ...ctx.state, answerFailed: false })
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
  const browser = ctx.state.answerFailed || ctx.options.session.serverless()
  const piece = newPiece(id, text, events, browser ? 'browser' : 'waiting')
  ctx.state = Object.freeze({ ...ctx.state, ids: id, queue: Object.freeze([...ctx.state.queue, piece]) })
  pump(ctx)
}

function cancel(ctx: SpeechCtx): void {
  const { queue } = ctx.state
  setQueue(ctx, Object.freeze([]))
  newAnswer(ctx)
  queue.forEach((piece) => settle(ctx, piece))
  ctx.options.player.stop()
  ctx.options.fallback?.cancel()
}

function prime(ctx: SpeechCtx): void {
  newAnswer(ctx)
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
      fail: (id: number) => fail(ctx, id),
      pump: () => pump(ctx),
    }),
    state: Object.freeze({ ids: 0, queue: Object.freeze([]), answerFailed: false }),
  }
  return Object.freeze({
    speak: (text: string, events: UtteranceEvents) => speak(ctx, text, events),
    cancel: () => cancel(ctx),
    prime: () => prime(ctx),
  })
}
