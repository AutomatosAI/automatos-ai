/**
 * Theme voice, web quick-ask — when the server voice fails (server-speech.ts): one answer never goes back and forth
 * between two voices. The first piece the server voice fails (a refusal such as 429 tts_busy or 503 tts_unavailable,
 * a network error, no MP3 within SERVER_VOICE_WAIT_MS once due, audio that won't play) and every later piece of the
 * answer are spoken by the browser's voice, in order after the pieces before it; the requests in flight are aborted
 * and nothing more is asked of the server, even once the queue runs dry. The next answer (after prime() or cancel())
 * asks the server again; tts_off keeps the server off for the page session.
 */
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from 'vitest'

vi.mock('@/lib/api-client', () => ({ apiClient: {} }))

import { SERVER_VOICE_WAIT_MS } from '@/lib/voice/quick-ask/config'
import { serverSpeechPort } from '@/lib/voice/quick-ask/server-speech'
import { createSpeaker } from '@/lib/voice/quick-ask/speaker'

import { flush } from './quick-ask-harness'
import { serverSpeechSetup, utterance } from './server-speech-fakes'

beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  vi.restoreAllMocks()
})

type Setup = ReturnType<typeof serverSpeechSetup>

/** An answer of four pieces, the first played by the server voice: the second is due, the third being fetched. */
async function secondDue(setup: Setup) {
  const onEvent = vi.fn()
  const speaker = createSpeaker(setup.port, setup.time.clock, onEvent)
  speaker.speak(['One.', 'Two.', 'Three.', 'Four.'])
  await flush()
  setup.server.answer(0)
  await flush()
  setup.audio.plays[0].events.onPlaying()
  setup.audio.plays[0].events.onEnded()
  await flush()
  expect(setup.server.texts()).toEqual(['One.', 'Two.', 'Three.'])
  return { onEvent, speaker }
}

/** Pieces 2 to 4 went to the browser's voice: 3's prefetch aborted, 4 never asked for, late answers never played. */
async function expectRestInBrowserVoice(setup: Setup, onEvent: Mock): Promise<void> {
  expect(setup.browser.spoken()).toEqual(['Two.', 'Three.', 'Four.'])
  expect(setup.server.calls[2].signal.aborted).toBe(true)
  setup.server.answer(1)
  setup.server.answer(2)
  await flush()
  for (const index of [0, 1, 2]) setup.browser.play(index)

  expect(setup.server.fetchSpeech).toHaveBeenCalledTimes(3)
  expect(setup.audio.plays).toHaveLength(1)
  expect(onEvent.mock.calls.map(([event]) => event)).toEqual([
    { type: 'sentence', text: 'One.' },
    { type: 'sentence', text: 'Two.' },
    { type: 'sentence', text: 'Three.' },
    { type: 'sentence', text: 'Four.' },
    { type: 'drained' },
  ])
}

describe('the server voice fails: the rest of the answer in the browser’s voice', () => {
  it('a 429 tts_busy on piece 2 of 4: pieces 2 to 4 go to the browser’s voice, and no request is made for 3 or 4', async () => {
    const setup = serverSpeechSetup()
    const { onEvent } = await secondDue(setup)

    setup.server.refuse(1, 'tts_busy', 429)
    await flush()

    await expectRestInBrowserVoice(setup, onEvent)
  })

  it('no MP3 for piece 2 within SERVER_VOICE_WAIT_MS once it is due does the same as the 429', async () => {
    const setup = serverSpeechSetup()
    const { onEvent } = await secondDue(setup)

    setup.time.advance(SERVER_VOICE_WAIT_MS - 1)
    expect(setup.browser.spoken()).toEqual([])
    setup.time.advance(1)

    expect(setup.server.calls[1].signal.aborted).toBe(true)
    await expectRestInBrowserVoice(setup, onEvent)
  })

  it('starts the wait only once a piece is due: the next one is fetched while the head plays, however long', async () => {
    const { port, server, audio, browser, time } = serverSpeechSetup()
    port.speak('One.', utterance())
    port.speak('Two.', utterance())
    await flush()
    server.answer(0)
    await flush()
    audio.plays[0].events.onPlaying()

    time.advance(SERVER_VOICE_WAIT_MS * 2)
    expect(browser.spoken()).toEqual([])
    audio.plays[0].events.onEnded()
    time.advance(SERVER_VOICE_WAIT_MS)

    expect(browser.spoken()).toEqual(['Two.'])
  })

  it('a 503 tts_unavailable on the first piece drops the next one’s MP3, already in; the turn drains', async () => {
    const { port, server, audio, browser, time } = serverSpeechSetup()
    const onEvent = vi.fn()
    const speaker = createSpeaker(port, time.clock, onEvent)
    speaker.speak(['One.', 'Two.'])
    await flush()
    server.answer(1)
    await flush()

    server.refuse(0, 'tts_unavailable', 503)
    await flush()
    browser.play(0)
    browser.play(1)

    expect(audio.plays).toHaveLength(0)
    expect(onEvent.mock.calls.map(([event]) => event)).toEqual([
      { type: 'sentence', text: 'One.' },
      { type: 'sentence', text: 'Two.' },
      { type: 'drained' },
    ])
    expect(speaker.speaking()).toBe(false)
  })

  it('after a network error, the answer’s later sentences go to the browser’s voice even once the queue ran dry', async () => {
    const { port, server, browser } = serverSpeechSetup()
    port.speak('One.', utterance())
    port.speak('Two.', utterance())
    await flush()

    server.calls[0].reject(new TypeError('Failed to fetch'))
    await flush()
    browser.play(0)
    browser.play(1)
    // The reply streams on: its next sentence comes after the queue emptied, and is still this answer's.
    port.speak('Three.', utterance())
    await flush()

    expect(server.calls[1].signal.aborted).toBe(true)
    expect(browser.spoken()).toEqual(['One.', 'Two.', 'Three.'])
    expect(server.fetchSpeech).toHaveBeenCalledTimes(2)
  })

  it('audio that fails before it starts: that piece and the rest go to the browser’s voice, the ready MP3 unplayed', async () => {
    const { port, server, audio, browser } = serverSpeechSetup()
    const one = utterance()
    const two = utterance()
    port.speak('One.', one)
    port.speak('Two.', two)
    await flush()
    server.answer(0)
    server.answer(1)
    await flush()

    audio.plays[0].events.onFailed(new Error('decode'))
    browser.play(0)
    browser.play(1)

    expect(browser.spoken()).toEqual(['One.', 'Two.'])
    expect(audio.plays).toHaveLength(1)
    expect([one.onStart, one.onEnd, two.onStart, two.onEnd].map((mock) => mock.mock.calls.length)).toEqual([1, 1, 1, 1])
  })

  it('audio that fails after it started ends that piece (it isn’t said twice); the rest go to the browser’s voice', async () => {
    const { port, server, audio, browser } = serverSpeechSetup()
    const one = utterance()
    port.speak('One.', one)
    port.speak('Two.', utterance())
    await flush()
    server.answer(0)
    server.answer(1)
    await flush()

    audio.plays[0].events.onPlaying()
    audio.plays[0].events.onFailed(new Error('network'))

    expect([one.onStart.mock.calls.length, one.onEnd.mock.calls.length]).toEqual([1, 1])
    expect(browser.spoken()).toEqual(['Two.'])
    expect(audio.plays).toHaveLength(1)
  })

  it('a piece the server voice plays before the failed one keeps it; the browser’s voice waits for it to end', async () => {
    const { port, server, audio, browser } = serverSpeechSetup()
    port.speak('One.', utterance())
    port.speak('Two.', utterance())
    await flush()
    server.answer(0)
    await flush()
    audio.plays[0].events.onPlaying()

    server.refuse(1, 'tts_busy', 429)
    await flush()
    expect(browser.spoken()).toEqual([])
    audio.plays[0].events.onEnded()

    expect(browser.spoken()).toEqual(['Two.'])
  })

  it('ends the pieces at once where the browser has no voice', async () => {
    const { port, server } = serverSpeechSetup({ fallback: null })
    const one = utterance()
    const two = utterance()
    port.speak('One.', one)
    port.speak('Two.', two)
    await flush()

    server.refuse(0, 'tts_busy', 429)
    await flush()

    expect([one.onEnd.mock.calls.length, two.onEnd.mock.calls.length]).toEqual([1, 1])
    expect(one.onStart).not.toHaveBeenCalled()
  })
})

describe('the next answer asks the server again', () => {
  it('after prime() (the next hold asks a new question)', async () => {
    const { port, server, browser } = serverSpeechSetup()
    port.speak('One.', utterance())
    await flush()
    server.refuse(0, 'tts_busy', 429)
    await flush()
    browser.play(0)

    port.prime()
    port.speak('Next answer.', utterance())
    await flush()

    expect(server.texts()).toEqual(['One.', 'Next answer.'])
    expect(browser.spoken()).toEqual(['One.'])
  })

  it('after cancel()', async () => {
    const { port, server, browser } = serverSpeechSetup()
    port.speak('One.', utterance())
    await flush()
    server.refuse(0, 'tts_unavailable', 503)
    await flush()

    port.cancel()
    port.speak('Again.', utterance())
    await flush()

    expect(server.texts()).toEqual(['One.', 'Again.'])
    expect(browser.spoken()).toEqual(['One.'])
  })

  it('but not after tts_off: the server is asked nothing more this page session, by any engine in the tab', async () => {
    const { port, server, browser, options } = serverSpeechSetup()
    port.speak('One.', utterance())
    await flush()
    server.refuse(0, 'tts_off', 503)
    await flush()

    port.prime()
    port.speak('Two.', utterance())
    port.cancel()
    port.speak('Three.', utterance())
    serverSpeechPort(options).speak('Four.', utterance())
    await flush()

    expect(browser.spoken()).toEqual(['One.', 'Two.', 'Three.', 'Four.'])
    expect(server.fetchSpeech).toHaveBeenCalledTimes(1)
  })
})
