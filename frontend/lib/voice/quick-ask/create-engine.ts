/**
 * Theme voice, web quick-ask: the engine. Hold to talk with the browser's speech recognition, the words sent as a
 * normal chat turn with spokenMode, the reply spoken sentence by sentence (Auto's server voice, or the browser's where
 * the server can't), the AI notice once per browser session; Stop and "Auto, stop" end it. It keeps the voice contract (lib/voice/engine.ts) so the host's
 * voice provider, the theme SDK's voice area and Studio's chat read it like Live.
 *
 * Never throws to the caller: an unexpected failure is logged and shown as a plain sentence. A hidden tab or an
 * unloading page stops the microphone and the speech; the turn, if one is streaming, finishes into the chat unheard.
 * Deps are injectable (tests); a missing one is the browser's own (deps.ts).
 */
import type { VoiceEngine, VoiceLevels, VoicePhase, VoiceSnapshot } from '../engine'
import { QUICK_ASK_ERRORS } from './copy'
import { browserDeps, type QuickAskDeps } from './deps'
import { endHold, releaseHold, startHold } from './hold'
import { userLevel } from './levels'
import { createCore, dispatch, get, set, type Core } from './store'
import { interruptReply, onSpeaker } from './turn'

const LOG_PREFIX = '[voice]'
const DONE: Promise<void> = Promise.resolve()
/** A hold or a reply under way: what a hidden tab stops (an error stays on screen for the member to read). */
const BUSY: ReadonlySet<VoicePhase> = new Set<VoicePhase>(['connecting', 'listening', 'thinking', 'speaking'])

/** Runs an entry point; anything unexpected is logged and said as the recognizer stopping, never thrown. */
function guard(core: Core, run: () => void): void {
  try {
    run()
  } catch (error) {
    console.error(LOG_PREFIX, 'quick-ask failed', error)
    endHold(core)
    dispatch(core, { type: 'failed', error: QUICK_ASK_ERRORS.recognizer })
  }
}

/** The member's Stop: nothing sent, the speech stopped, a streaming turn cancelled on the server. */
function stop(core: Core): void {
  endHold(core)
  interruptReply(core, true)
  dispatch(core, { type: 'stopped' })
}

/** The microphone and the speech stop; a streaming turn finishes into the chat unheard. */
function halt(core: Core): void {
  endHold(core)
  interruptReply(core, false)
}

/** The tab went away: what was under way stops (halt), and the voice is idle. */
function leave(core: Core): void {
  if (!BUSY.has(get(core).state.view.phase)) return
  halt(core)
  dispatch(core, { type: 'stopped' })
}

function subscribe(core: Core, listener: (snapshot: VoiceSnapshot) => void): () => void {
  if (get(core).disposed) return () => undefined
  core.listeners.add(listener)
  return () => {
    core.listeners.delete(listener)
  }
}

function levels(core: Core): VoiceLevels {
  const { state, lastWordsAt } = get(core)
  return { user: userLevel(state.view.micOpen, core.deps.clock.now(), lastWordsAt), agent: core.speaker.level() }
}

function dispose(core: Core): void {
  if (get(core).disposed) return
  leave(core)
  halt(core)
  core.listeners.clear()
  set(core, { disposed: true })
}

/** A quick-ask voice engine; `deps` replaces any of the browser's own (tests). */
export function createQuickAskEngine(deps: Partial<QuickAskDeps> = {}): VoiceEngine {
  const core = createCore({ ...browserDeps(), ...deps }, onSpeaker)
  const unwatch = core.deps.onPageHidden(() => guard(core, () => leave(core)))
  return Object.freeze({
    start: (options: { readonly chatId: string | null }) => {
      guard(core, () => startHold(core, options?.chatId || null))
      return DONE
    },
    release: () => guard(core, () => releaseHold(core)),
    stop: () => guard(core, () => stop(core)),
    snapshot: () => get(core).state.view,
    subscribe: (listener: (snapshot: VoiceSnapshot) => void) => subscribe(core, listener),
    levels: () => levels(core),
    dispose: () =>
      guard(core, () => {
        unwatch()
        dispose(core)
      }),
  })
}
