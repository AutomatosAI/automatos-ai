/**
 * Theme voice, web quick-ask: the server voice port's request half (server-speech.ts). A piece's MP3 is asked for in
 * the page session's voice. Any answer but audio (a refusal, a network error, a body that isn't audio), or no MP3
 * within SERVER_VOICE_WAIT_MS once the piece is due, fails the answer: the piece and the rest of the answer go to the
 * browser's voice (moves.fail). tts_off also turns the server voice off for the page session. A late answer for a
 * piece no longer loading (cancelled, failed, timed out) does nothing.
 */
import { SERVER_VOICE_WAIT_MS } from './config'
import { find, LOG_PREFIX, settle, update, type SpeechCtx } from './server-speech-ctx'
import type { Piece } from './speech-queue'
import { refusalCode, TTS_OFF } from './voice-api'

/** The piece's MP3 in the session's voice, or null for the browser's voice. Never rejects; never logs the text. */
async function answerFor(ctx: SpeechCtx, text: string, signal: AbortSignal): Promise<Blob | null> {
  const { session, fetchSpeech } = ctx.options
  const voice = await session.voice()
  if (voice === null || session.serverless() || signal.aborted) return null
  try {
    return await fetchSpeech(text, voice, signal)
  } catch (error) {
    const code = refusalCode(error)
    if (code === TTS_OFF) session.turnOff()
    if (!signal.aborted) console.warn(LOG_PREFIX, "the server voice didn't speak a sentence", code ?? error)
    return null
  }
}

function arrived(ctx: SpeechCtx, id: number, clip: Blob | null): void {
  const piece = find(ctx, id)
  if (piece?.stage !== 'loading') return
  if (clip === null) return ctx.moves.fail(id)
  settle(ctx, piece)
  update(ctx, id, { stage: 'ready', clip, abort: null, waitTimer: null })
  ctx.moves.pump()
}

/** Asks for the piece's MP3. */
export function request(ctx: SpeechCtx, piece: Piece): void {
  const controller = new AbortController()
  update(ctx, piece.id, { stage: 'loading', abort: () => controller.abort() })
  void answerFor(ctx, piece.text, controller.signal).then((clip) => arrived(ctx, piece.id, clip))
}

/** The head is due: its MP3 has SERVER_VOICE_WAIT_MS to come, or the browser's voice speaks it and the rest. */
export function armWait(ctx: SpeechCtx, id: number): void {
  const waited = (): void => {
    if (find(ctx, id)?.stage !== 'loading') return
    console.warn(LOG_PREFIX, 'the server voice was too slow for a sentence')
    ctx.moves.fail(id)
  }
  update(ctx, id, { waitTimer: ctx.options.clock.setTimeout(waited, SERVER_VOICE_WAIT_MS) })
}
