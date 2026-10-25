/**
 * Theme voice, web quick-ask — the browser ports: recognition (continuous, interim results, the browser's error
 * codes as quick-ask's failures, every result's best guess on one line), the speaker (one utterance per sentence,
 * each start told, drained once the last ends, a stopped run's late callbacks ignored, Chrome's utterances kept
 * alive) and the level envelopes.
 */
import { afterEach, describe, expect, it, vi, type Mock } from 'vitest'

import { LEVEL_DECAY_MS } from '@/lib/voice/quick-ask/config'
import { agentLevel, pulse, userLevel } from '@/lib/voice/quick-ask/levels'
import {
  browserRecognitionConstructor,
  browserRecognitionPort,
  recognitionFailure,
  transcriptOf,
  type BrowserRecognition,
  type RecognitionEvents,
} from '@/lib/voice/quick-ask/recognition'
import { browserSynthesisPort, createSpeaker, type SynthesisPort, type UtteranceEvents } from '@/lib/voice/quick-ask/speaker'

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

/** A result list as the browser hands it over: each result's alternatives, best first. */
function results(...guesses: string[]) {
  return guesses.map((transcript) => [{ transcript }])
}

class FakeBrowserRecognition implements BrowserRecognition {
  static last: FakeBrowserRecognition | null = null
  lang = ''
  continuous = false
  interimResults = false
  maxAlternatives = 0
  onstart: (() => void) | null = null
  onaudiostart: (() => void) | null = null
  onaudioend: (() => void) | null = null
  onresult: BrowserRecognition['onresult'] = null
  onerror: BrowserRecognition['onerror'] = null
  onend: (() => void) | null = null
  start = vi.fn()
  stop = vi.fn()
  abort = vi.fn()

  constructor() {
    FakeBrowserRecognition.last = this
  }
}

function recognitionEvents(): Record<keyof RecognitionEvents, Mock> {
  return { onCaptureStart: vi.fn(), onCaptureEnd: vi.fn(), onWords: vi.fn(), onError: vi.fn(), onEnd: vi.fn() }
}

describe('the browser recognition port', () => {
  it('listens continuously with interim results in the given language', () => {
    const port = browserRecognitionPort(FakeBrowserRecognition, 'en-GB')
    port.start(recognitionEvents())
    const browser = FakeBrowserRecognition.last as FakeBrowserRecognition

    expect(browser).toMatchObject({ lang: 'en-GB', continuous: true, interimResults: true, maxAlternatives: 1 })
    expect(browser.start).toHaveBeenCalledTimes(1)
  })

  it('tells capture, words, failures and the end; our own abort is not a failure', () => {
    const events = recognitionEvents()
    browserRecognitionPort(FakeBrowserRecognition, 'en-GB').start(events)
    const browser = FakeBrowserRecognition.last as FakeBrowserRecognition

    browser.onaudiostart?.()
    browser.onresult?.({ results: results('what is', ' on today') })
    browser.onerror?.({ error: 'aborted' })
    browser.onerror?.({ error: 'not-allowed' })
    browser.onend?.()

    expect(events.onCaptureStart).toHaveBeenCalled()
    expect(events.onWords).toHaveBeenCalledWith('what is on today')
    expect(events.onError.mock.calls).toEqual([['permission']])
    expect(events.onCaptureEnd).toHaveBeenCalled()
    expect(events.onEnd).toHaveBeenCalledTimes(1)
  })

  it('stops and aborts the browser object', () => {
    const port = browserRecognitionPort(FakeBrowserRecognition, 'en-GB')
    const browser = FakeBrowserRecognition.last as FakeBrowserRecognition

    port.stop()
    port.abort()

    expect(browser.stop).toHaveBeenCalledTimes(1)
    expect(browser.abort).toHaveBeenCalledTimes(1)
  })

  it('reads every browser error code as a failure the member can be told', () => {
    expect(recognitionFailure('not-allowed')).toBe('permission')
    expect(recognitionFailure('service-not-allowed')).toBe('serviceOff')
    expect(recognitionFailure('no-speech')).toBe('noSpeech')
    expect(recognitionFailure('audio-capture')).toBe('noMicrophone')
    expect(recognitionFailure('network')).toBe('recognitionNetwork')
    expect(recognitionFailure('language-not-supported')).toBe('unsupported')
    expect(recognitionFailure('bad-grammar')).toBe('recognizer')
    expect(recognitionFailure('constructor')).toBe('recognizer')
    expect(recognitionFailure('aborted')).toBeNull()
  })

  it('puts the best guesses on one line, whatever the spacing', () => {
    expect(transcriptOf(results('Hello', ' there  ', 'Auto'))).toBe('Hello there Auto')
    expect(transcriptOf(results())).toBe('')
  })

  it('finds the constructor by either name, or none', () => {
    expect(browserRecognitionConstructor()).toBeNull()
    vi.stubGlobal('webkitSpeechRecognition', FakeBrowserRecognition)
    expect(browserRecognitionConstructor()).toBe(FakeBrowserRecognition)
  })
})

/** A synthesis port that keeps each queued utterance for the test to start and end. */
function fakeSynthesis() {
  const queued: { text: string; events: UtteranceEvents }[] = []
  const port: SynthesisPort & { cancel: Mock } = {
    speak: (text, events) => {
      queued.push({ text, events })
    },
    cancel: vi.fn(),
    prime: vi.fn(),
  }
  return { port, queued }
}

const clock = (at: { now: number }) => ({ now: () => at.now, setTimeout: vi.fn(), clearTimeout: vi.fn() })

describe('createSpeaker', () => {
  it('queues one utterance per sentence, tells each start, and drains once after the last ends', () => {
    const { port, queued } = fakeSynthesis()
    const onEvent = vi.fn()
    const speaker = createSpeaker(port, clock({ now: 0 }), onEvent)

    speaker.speak(['One.', 'Two.'])
    queued[0].events.onStart()
    queued[0].events.onEnd()
    queued[1].events.onStart()

    expect(queued.map((item) => item.text)).toEqual(['One.', 'Two.'])
    expect(speaker.speaking()).toBe(true)
    queued[1].events.onEnd()
    expect(onEvent.mock.calls.map(([event]) => event)).toEqual([
      { type: 'sentence', text: 'One.' },
      { type: 'sentence', text: 'Two.' },
      { type: 'drained' },
    ])
    expect(speaker.speaking()).toBe(false)
  })

  it('cancels on stop and ignores the stopped run’s late callbacks', () => {
    const { port, queued } = fakeSynthesis()
    const onEvent = vi.fn()
    const speaker = createSpeaker(port, clock({ now: 0 }), onEvent)
    speaker.speak(['Cut short.'])

    speaker.stop()
    queued[0].events.onStart()
    queued[0].events.onEnd()

    expect(port.cancel).toHaveBeenCalledTimes(1)
    expect(onEvent).not.toHaveBeenCalled()
    expect(speaker.speaking()).toBe(false)
  })

  it('counts a sentence the browser would not take as ended', () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const onEvent = vi.fn()
    const throwing: SynthesisPort = {
      speak: () => {
        throw new Error('no voice')
      },
      cancel: vi.fn(),
      prime: vi.fn(),
    }
    const speaker = createSpeaker(throwing, clock({ now: 0 }), onEvent)

    speaker.speak(['Nothing.'])

    expect(onEvent).toHaveBeenCalledWith({ type: 'drained' })
    expect(speaker.speaking()).toBe(false)
  })

  it('lifts Auto’s level while speaking and rests at 0 when silent', () => {
    const at = { now: 1_000 }
    const { port, queued } = fakeSynthesis()
    const speaker = createSpeaker(port, clock(at), vi.fn())

    expect(speaker.level()).toBe(0)
    speaker.speak(['Hello.'])
    queued[0].events.onStart()
    expect(speaker.level()).toBeGreaterThan(0.3)
    at.now += LEVEL_DECAY_MS * 2
    expect(speaker.level()).toBeCloseTo(0.3)
    queued[0].events.onEnd()
    expect(speaker.level()).toBe(0)
  })
})

describe('the browser synthesis port', () => {
  class FakeUtterance {
    lang = ''
    onstart: (() => void) | null = null
    onboundary: (() => void) | null = null
    onend: (() => void) | null = null
    onerror: ((event: { error: string }) => void) | null = null
    constructor(readonly text: string) {}
  }

  function fakeBrowserSynthesis() {
    return { speaking: false, pending: false, speak: vi.fn(), cancel: vi.fn() }
  }

  it('speaks each sentence as an utterance in the language and passes on its events', () => {
    vi.stubGlobal('SpeechSynthesisUtterance', FakeUtterance)
    const synthesis = fakeBrowserSynthesis()
    const events = { onStart: vi.fn(), onBoundary: vi.fn(), onEnd: vi.fn() }
    const port = browserSynthesisPort(synthesis as unknown as SpeechSynthesis, 'en-GB')

    port.speak('Hello.', events)
    const utterance = synthesis.speak.mock.calls[0][0] as FakeUtterance
    utterance.onstart?.()
    utterance.onboundary?.()
    utterance.onerror?.({ error: 'interrupted' })

    expect(utterance).toMatchObject({ text: 'Hello.', lang: 'en-GB' })
    expect(events.onStart).toHaveBeenCalled()
    expect(events.onBoundary).toHaveBeenCalled()
    expect(events.onEnd).toHaveBeenCalledTimes(1)
  })

  it('primes with an empty utterance only when nothing is speaking, and cancels the queue', () => {
    vi.stubGlobal('SpeechSynthesisUtterance', FakeUtterance)
    const synthesis = fakeBrowserSynthesis()
    const port = browserSynthesisPort(synthesis as unknown as SpeechSynthesis, 'en-GB')

    port.prime()
    synthesis.speaking = true
    port.prime()
    port.cancel()

    expect(synthesis.speak).toHaveBeenCalledTimes(1)
    expect((synthesis.speak.mock.calls[0][0] as FakeUtterance).text).toBe('')
    expect(synthesis.cancel).toHaveBeenCalledTimes(1)
  })
})

describe('the level envelopes', () => {
  it('pulse falls from the peak to 0 over the decay', () => {
    expect(pulse(100, null, 1)).toBe(0)
    expect(pulse(100, 100, 1)).toBe(1)
    expect(pulse(100 + LEVEL_DECAY_MS / 2, 100, 1)).toBeCloseTo(0.5)
    expect(pulse(100 + LEVEL_DECAY_MS, 100, 1)).toBe(0)
  })

  it('the member’s level needs an open microphone; Auto’s a voice speaking', () => {
    expect(userLevel(false, 100, 100)).toBe(0)
    expect(userLevel(true, 100, 100)).toBeGreaterThan(0)
    expect(agentLevel(false, 100, 100)).toBe(0)
    expect(agentLevel(true, 100, null)).toBe(0.3)
  })
})
