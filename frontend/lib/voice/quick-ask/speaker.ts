/**
 * Theme voice, web quick-ask: Auto's voice through the browser's speech synthesis, behind a port (the tests use a
 * fake). The logic is automatos-go's speaker-core.ts (PR #15) for a browser: each sentence is one utterance on the
 * browser's queue; the speaker says which sentence starts (snapshot.reply grows as spoken) and when the LAST queued
 * one has ended (done, cancelled or failed: the queue ran out either way). stop() starts a new generation, so the
 * late callbacks of a cancelled run are ignored and nothing waits on an utterance the browser dropped.
 */
import type { QuickAskClock } from './clock'
import { agentLevel } from './levels'

export interface UtteranceEvents {
  readonly onStart: () => void
  /** A word boundary, where the browser reports them: lifts Auto's level. */
  readonly onBoundary: () => void
  /** The utterance ended, was cancelled or failed. */
  readonly onEnd: () => void
}

export interface SynthesisPort {
  /** Queues one utterance. */
  speak(text: string, events: UtteranceEvents): void
  /** Empties the queue and stops the current utterance. */
  cancel(): void
  /** Called inside the user gesture that starts a hold: browsers that need one (iOS Safari) can then speak later. */
  prime(): void
}

export type SpeakerEvent = { readonly type: 'sentence'; readonly text: string } | { readonly type: 'drained' }

export interface Speaker {
  speak(sentences: readonly string[]): void
  stop(): void
  speaking(): boolean
  level(): number
}

interface Run {
  /** Bumped by stop(): callbacks of an older generation are ignored. */
  readonly generation: number
  /** Utterances queued and not yet ended. */
  readonly pending: number
  readonly lastBoundaryAt: number | null
}

const LOG_PREFIX = '[voice]'
/** The browser's errors for an utterance cancelled on purpose (stop, barge-in): not worth a warning. */
const CANCELLED: ReadonlySet<string> = new Set(['interrupted', 'canceled'])

export function createSpeaker(port: SynthesisPort, clock: QuickAskClock, onEvent: (event: SpeakerEvent) => void): Speaker {
  let run: Run = { generation: 0, pending: 0, lastBoundaryAt: null }

  function utterance(text: string, generation: number): UtteranceEvents {
    const current = (): boolean => run.generation === generation
    return {
      onStart: () => {
        if (!current()) return
        run = { ...run, lastBoundaryAt: clock.now() }
        onEvent({ type: 'sentence', text })
      },
      onBoundary: () => {
        if (current()) run = { ...run, lastBoundaryAt: clock.now() }
      },
      onEnd: () => {
        if (!current() || run.pending === 0) return
        run = { ...run, pending: run.pending - 1 }
        if (run.pending === 0) onEvent({ type: 'drained' })
      },
    }
  }

  function speak(sentences: readonly string[]): void {
    sentences.forEach((text) => {
      run = { ...run, pending: run.pending + 1 }
      const events = utterance(text, run.generation)
      try {
        port.speak(text, events)
      } catch (error) {
        console.warn(LOG_PREFIX, 'a sentence could not be queued for speech', error)
        events.onEnd()
      }
    })
  }

  function stop(): void {
    run = { generation: run.generation + 1, pending: 0, lastBoundaryAt: null }
    port.cancel()
  }

  return Object.freeze({
    speak,
    stop,
    speaking: () => run.pending > 0,
    level: () => agentLevel(run.pending > 0, clock.now(), run.lastBoundaryAt),
  })
}

/** Ends one utterance's life: out of the keep-alive list, then its end event. */
function ender(
  forget: (utterance: SpeechSynthesisUtterance) => void,
  utterance: SpeechSynthesisUtterance,
  events: UtteranceEvents,
): (event?: SpeechSynthesisErrorEvent) => void {
  return (event) => {
    if (event && !CANCELLED.has(event.error)) console.warn(LOG_PREFIX, 'an utterance failed', event.error)
    forget(utterance)
    events.onEnd()
  }
}

/** The port over the browser's speechSynthesis, speaking in `lang` with the browser's default voice for it. */
export function browserSynthesisPort(synthesis: SpeechSynthesis, lang: string): SynthesisPort {
  // Chrome stops reporting an utterance's events once it is garbage collected: each is kept until it ends.
  let alive: readonly SpeechSynthesisUtterance[] = []
  const forget = (done: SpeechSynthesisUtterance): void => {
    alive = alive.filter((kept) => kept !== done)
  }
  return Object.freeze({
    speak: (text: string, events: UtteranceEvents) => {
      const utterance = new SpeechSynthesisUtterance(text)
      const end = ender(forget, utterance, events)
      utterance.lang = lang
      utterance.onstart = () => events.onStart()
      utterance.onboundary = () => events.onBoundary()
      utterance.onend = () => end()
      utterance.onerror = (event) => end(event)
      alive = [...alive, utterance]
      synthesis.speak(utterance)
    },
    cancel: () => {
      alive = []
      synthesis.cancel()
    },
    prime: () => {
      if (!synthesis.speaking && !synthesis.pending) synthesis.speak(new SpeechSynthesisUtterance(''))
    },
  })
}
