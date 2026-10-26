/**
 * Theme voice, web quick-ask: hold to talk (automatos-go's quick-ask-hold.ts, PR #15, for a browser).
 *
 *   start → Auto stops (barge-in: the speech, and a turn still streaming is cancelled on the server, D-STOP) → the
 *   speech is primed inside the gesture → a new recognition opens (connecting; listening once the browser captures).
 *   The words show as they are recognised. release (or HOLD_MAX_MS) → recognition stops and hands over its last
 *   words; its end decides: a stop phrase as the whole utterance ("Auto, stop") is Stop with nothing sent, nothing
 *   sayable is "nothing heard", anything else is sent as a chat turn (turn.ts). Recognition that ends by itself
 *   while held (the browser's own end of speech) decides the same way. A release before capture began stops the
 *   capture the moment it begins; RELEASE_GRACE_MS after a release the hold ends with what was heard, whether or not
 *   the browser ever said it ended. One hold at a time; every event of an old hold is ignored.
 */
import { DURATION_TICK_MS, HOLD_MAX_MS, RELEASE_GRACE_MS } from './config'
import { QUICK_ASK_ERRORS } from './copy'
import type { RecognitionEvents, RecognitionFailure, RecognitionPort } from './recognition'
import { sayable } from './sentences'
import { isStopPhrase } from './stop-phrase'
import { dispatch, get, nextId, set, type Core, type Hold } from './store'
import { interruptReply, sendTurn } from './turn'

const LOG_PREFIX = '[voice]'
const MS_PER_SECOND = 1_000

/** The hold with this id, while it is the current one. */
function current(core: Core, holdId: number): Hold | null {
  const hold = get(core).hold
  return hold !== null && hold.id === holdId ? hold : null
}

function update(core: Core, hold: Hold, changes: Partial<Hold>): void {
  set(core, { hold: Object.freeze({ ...hold, ...changes }) })
}

/** Ends the current hold: its timers cleared, the browser's recognition aborted (a no-op once it ended). */
export function endHold(core: Core): Hold | null {
  const hold = get(core).hold
  if (hold === null) return null
  const timers = [hold.maxTimer, hold.graceTimer, hold.tickTimer]
  timers.forEach((timer) => {
    if (timer !== null) core.deps.clock.clearTimeout(timer)
  })
  set(core, { hold: null })
  try {
    hold.port.abort()
  } catch (error) {
    console.warn(LOG_PREFIX, 'speech recognition did not abort', error)
  }
  return hold
}

function failHold(core: Core, failure: RecognitionFailure): void {
  endHold(core)
  dispatch(core, { type: 'failed', error: QUICK_ASK_ERRORS[failure] })
}

/** Recognition is over: the words are a stop phrase, nothing, or a turn for Auto. */
function finishHold(core: Core, holdId: number): void {
  if (current(core, holdId) === null) return
  const hold = endHold(core)
  const text = hold?.words.trim() ?? ''
  if (isStopPhrase(text)) return dispatch(core, { type: 'stopped' })
  if (!sayable(text)) return dispatch(core, { type: 'failed', error: QUICK_ASK_ERRORS.noSpeech })
  sendTurn(core, text)
}

function stopCapture(core: Core, holdId: number): void {
  const hold = current(core, holdId)
  if (hold === null) return
  try {
    hold.port.stop()
  } catch (error) {
    console.warn(LOG_PREFIX, 'speech recognition did not stop', error)
    finishHold(core, holdId)
  }
}

/** durationSec: the hold in whole seconds, while the microphone is open. */
function tick(core: Core, holdId: number): void {
  const hold = current(core, holdId)
  if (hold === null || hold.capturedAt === null) return
  dispatch(core, { type: 'duration', seconds: Math.floor((core.deps.clock.now() - hold.capturedAt) / MS_PER_SECOND) })
  const after = current(core, holdId)
  if (after === null) return
  update(core, after, { tickTimer: core.deps.clock.setTimeout(() => tick(core, holdId), DURATION_TICK_MS) })
}

function captureStarted(core: Core, holdId: number): void {
  const hold = current(core, holdId)
  if (hold === null) return
  if (hold.capturedAt === null) {
    const tickTimer = core.deps.clock.setTimeout(() => tick(core, holdId), DURATION_TICK_MS)
    update(core, hold, { capturedAt: core.deps.clock.now(), tickTimer })
  }
  dispatch(core, { type: 'capture_started' })
  // Let go before the capture began: it ends now, with whatever was said.
  if (current(core, holdId)?.released) stopCapture(core, holdId)
}

function heard(core: Core, holdId: number, text: string): void {
  const hold = current(core, holdId)
  if (hold === null) return
  update(core, hold, { words: text })
  set(core, { lastWordsAt: core.deps.clock.now() })
  dispatch(core, { type: 'words', text })
}

function holdEvents(core: Core, holdId: number): RecognitionEvents {
  return {
    onCaptureStart: () => captureStarted(core, holdId),
    onCaptureEnd: () => {
      if (current(core, holdId) !== null) dispatch(core, { type: 'capture_ended' })
    },
    onWords: (text: string) => heard(core, holdId, text),
    onError: (failure: RecognitionFailure) => {
      if (current(core, holdId) !== null) failHold(core, failure)
    },
    onEnd: () => finishHold(core, holdId),
  }
}

/** Inside the gesture, so a browser that needs one (iOS Safari) lets Auto speak later. */
function primeSpeech(core: Core): void {
  try {
    core.deps.synthesis?.prime()
  } catch (error) {
    console.warn(LOG_PREFIX, 'speech synthesis could not be primed', error)
  }
}

function newHold(core: Core, port: RecognitionPort): Hold {
  const id = nextId(core)
  const maxTimer = core.deps.clock.setTimeout(() => releaseHold(core), HOLD_MAX_MS)
  return Object.freeze({
    id, port, words: '', released: false, capturedAt: null, maxTimer, graceTimer: null, tickTimer: null,
  })
}

/** A new hold on the conversation (null: a new one); ignored while a hold is open. */
export function startHold(core: Core, chatId: string | null): void {
  const { disposed, hold: open } = get(core)
  if (disposed || open !== null) return
  interruptReply(core, true)
  dispatch(core, { type: 'start', chatId })
  const port = core.deps.synthesis === null ? null : core.deps.recognition()
  if (port === null) return dispatch(core, { type: 'failed', error: QUICK_ASK_ERRORS.unsupported })
  primeSpeech(core)
  const hold = newHold(core, port)
  set(core, { hold, lastWordsAt: null })
  try {
    port.start(holdEvents(core, hold.id))
  } catch (error) {
    console.warn(LOG_PREFIX, 'speech recognition did not start', error)
    if (current(core, hold.id) !== null) failHold(core, 'recognizer')
  }
}

/** The member let go: capture stops and the last words come; the hold ends RELEASE_GRACE_MS later at the latest. */
export function releaseHold(core: Core): void {
  const hold = get(core).hold
  if (hold === null || hold.released) return
  core.deps.clock.clearTimeout(hold.maxTimer)
  const graceTimer = core.deps.clock.setTimeout(() => finishHold(core, hold.id), RELEASE_GRACE_MS)
  update(core, hold, { released: true, graceTimer })
  if (hold.capturedAt !== null) stopCapture(core, hold.id)
}
