/**
 * Theme voice, web quick-ask — the engine's test harness: a fake recognition per hold, a fake speech synthesis whose
 * utterances the test plays, a fake chat turn the test streams and settles, a clock moved by hand, a memory
 * sessionStorage and a page the test hides. Each test file mocks apiClient: the engine's browser defaults (deps.ts)
 * import the chat's send and Stop, which import it.
 */
import { vi, type Mock } from 'vitest'

import type { ChatTurnHandlers, ChatTurnOutcome, ChatTurnRequest } from '@/lib/chat/turn-stream'
import type { VoiceSnapshot } from '@/lib/voice/engine'
import { createQuickAskEngine, type QuickAskDeps } from '@/lib/voice/quick-ask'
import type { NoticeStorage } from '@/lib/voice/quick-ask/ai-notice'
import type { RecognitionEvents, RecognitionFailure, RecognitionPort } from '@/lib/voice/quick-ask/recognition'
import type { SynthesisPort, UtteranceEvents } from '@/lib/voice/quick-ask/speaker'

export const GREETING = 'Hi, this is Auto, your AI assistant.'

/** Lets the engine's promise chains (the turn's send and settle) run. */
export function flush(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0))
}

/** One hold's recognition, driven by the test. */
export class FakeHold {
  events: RecognitionEvents | null = null
  readonly stop: Mock = vi.fn()
  readonly abort: Mock = vi.fn()
  readonly port: RecognitionPort = {
    start: (events) => {
      this.events = events
    },
    stop: () => this.stop(),
    abort: () => this.abort(),
  }

  capture(): void {
    this.events?.onCaptureStart()
  }

  say(text: string): void {
    this.events?.onWords(text)
  }

  end(): void {
    this.events?.onCaptureEnd()
    this.events?.onEnd()
  }

  fail(failure: RecognitionFailure): void {
    this.events?.onError(failure)
  }
}

export function fakeRecognition() {
  const holds: FakeHold[] = []
  const factory = vi.fn((): RecognitionPort | null => {
    const hold = new FakeHold()
    holds.push(hold)
    return hold.port
  })
  return { factory, holds, last: (): FakeHold => holds[holds.length - 1] }
}

export function fakeSynthesis() {
  const utterances: { readonly text: string; readonly events: UtteranceEvents }[] = []
  const port: SynthesisPort & { readonly cancel: Mock; readonly prime: Mock } = {
    speak: (text, events) => {
      utterances.push({ text, events })
    },
    cancel: vi.fn(),
    prime: vi.fn(),
  }
  return {
    port,
    spoken: (): string[] => utterances.map((utterance) => utterance.text),
    start: (index: number) => utterances[index].events.onStart(),
    end: (index: number) => utterances[index].events.onEnd(),
    /** The utterance starts and ends. */
    play: (index: number) => {
      utterances[index].events.onStart()
      utterances[index].events.onEnd()
    },
  }
}

export interface FakeTurn {
  readonly request: ChatTurnRequest
  readonly handlers: ChatTurnHandlers
  readonly signal: AbortSignal
  readonly settle: (outcome: ChatTurnOutcome) => void
}

export function fakeChat() {
  const turns: FakeTurn[] = []
  const sendTurn = vi.fn(
    (request: ChatTurnRequest, handlers: ChatTurnHandlers, signal: AbortSignal) =>
      new Promise<ChatTurnOutcome>((resolve) => {
        turns.push({ request, handlers, signal, settle: resolve })
      }),
  )
  return { sendTurn, turns, last: (): FakeTurn => turns[turns.length - 1] }
}

export function manualClock() {
  let now = 1_000
  let serial = 0
  let timers: { readonly id: number; readonly at: number; readonly callback: () => void }[] = []
  const clock = {
    now: () => now,
    setTimeout: (callback: () => void, ms: number): unknown => {
      serial += 1
      timers = [...timers, { id: serial, at: now + ms, callback }]
      return serial
    },
    clearTimeout: (handle: unknown) => {
      timers = timers.filter((timer) => timer.id !== handle)
    },
  }
  /** Moves time on, running each timer that falls due, in order. */
  function advance(ms: number): void {
    const until = now + ms
    for (;;) {
      const due = timers.filter((timer) => timer.at <= until).sort((a, b) => a.at - b.at)[0]
      if (due === undefined) break
      timers = timers.filter((timer) => timer !== due)
      now = due.at
      due.callback()
    }
    now = until
  }
  return { clock, advance }
}

export function memoryStorage(): NoticeStorage & { readonly values: Map<string, string> } {
  const values = new Map<string, string>()
  return {
    values,
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => {
      values.set(key, value)
    },
  }
}

export function fakePage() {
  let hidden: (() => void) | null = null
  const unwatch = vi.fn()
  const onPageHidden = vi.fn((callback: () => void) => {
    hidden = callback
    return unwatch
  })
  return { onPageHidden, unwatch, hide: () => hidden?.() }
}

/** An engine over fakes, and the phases its listener heard, each once in a row. */
export function quickAskHarness(overrides: Partial<QuickAskDeps> = {}) {
  const recognition = fakeRecognition()
  const synthesis = fakeSynthesis()
  const chat = fakeChat()
  const time = manualClock()
  const storage = memoryStorage()
  const page = fakePage()
  const cancelTurn = vi.fn(async () => ({ cancelled: true }))
  const announceTurn = vi.fn()
  const engine = createQuickAskEngine({
    recognition: recognition.factory,
    synthesis: synthesis.port,
    sendTurn: chat.sendTurn,
    cancelTurn,
    announceTurn,
    clock: time.clock,
    storage,
    onPageHidden: page.onPageHidden,
    ...overrides,
  })
  const phases: VoiceSnapshot['phase'][] = []
  engine.subscribe((snapshot) => {
    if (phases[phases.length - 1] !== snapshot.phase) phases.push(snapshot.phase)
  })
  return { engine, recognition, synthesis, chat, time, storage, page, cancelTurn, announceTurn, phases }
}

export type Harness = ReturnType<typeof quickAskHarness>

/** A hold that says these words and lets go; the turn it sent. */
export async function ask(harness: Harness, words: string, chatId: string | null = 'c-1'): Promise<FakeTurn> {
  await harness.engine.start({ chatId })
  const hold = harness.recognition.last()
  hold.capture()
  hold.say(words)
  harness.engine.release()
  hold.end()
  await flush()
  return harness.chat.last()
}
