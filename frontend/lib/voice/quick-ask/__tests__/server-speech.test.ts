/**
 * Theme voice, web quick-ask — the server voice port (server-speech.ts): pieces play strictly in order with the next
 * one's MP3 asked for while the head plays (exactly one ahead), each piece's start, level and end told; a piece the
 * server can't speak (a refusal, a network error, too slow, audio that won't play) goes to the browser's voice with
 * the same events and the turn drains; tts_off turns the server off for the page session; cancel aborts, stops and
 * silences late callbacks; prime unlocks the audio and primes the browser's voice.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

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

describe('the server voice: order and prefetch', () => {
  it('asks for the head and the piece after it only, and plays them in order, whichever answer comes first', async () => {
    const { port, server, audio } = serverSpeechSetup()
    port.speak('One.', utterance())
    port.speak('Two.', utterance())
    port.speak('Three.', utterance())
    await flush()

    expect(server.texts()).toEqual(['One.', 'Two.'])
    expect(server.calls.map((call) => call.voice)).toEqual(['af_heart', 'af_heart'])
    server.answer(1)
    await flush()
    expect(audio.plays).toHaveLength(0)
    server.answer(0)
    await flush()
    expect(audio.plays.map((play) => play.clip.size)).toEqual(['One.'.length])
    expect(server.texts()).toEqual(['One.', 'Two.'])

    audio.plays[0].events.onPlaying()
    audio.plays[0].events.onEnded()
    await flush()
    expect(server.texts()).toEqual(['One.', 'Two.', 'Three.'])
    expect(audio.plays).toHaveLength(2)
  })

  it('tells each piece its start when the audio plays, a boundary on each timeupdate, and its end', async () => {
    const { port, server, audio } = serverSpeechSetup()
    const one = utterance()
    const two = utterance()
    port.speak('One.', one)
    port.speak('Two.', two)
    await flush()
    server.answer(0)
    server.answer(1)
    await flush()

    expect(one.onStart).not.toHaveBeenCalled()
    audio.plays[0].events.onPlaying()
    audio.plays[0].events.onProgress()
    audio.plays[0].events.onProgress()
    expect([one.onStart.mock.calls.length, one.onBoundary.mock.calls.length, one.onEnd.mock.calls.length]).toEqual([1, 2, 0])
    audio.plays[0].events.onEnded()
    expect(one.onEnd).toHaveBeenCalledTimes(1)
    expect(two.onStart).not.toHaveBeenCalled()
    audio.plays[1].events.onPlaying()
    audio.plays[1].events.onEnded()
    expect([two.onStart.mock.calls.length, two.onEnd.mock.calls.length]).toEqual([1, 1])
  })

  it('says a clip too short to report playing started before it ended', async () => {
    const { port, server, audio } = serverSpeechSetup()
    const one = utterance()
    port.speak('Ok.', one)
    await flush()
    server.answer(0)
    await flush()

    audio.plays[0].events.onEnded()

    expect(one.onStart).toHaveBeenCalledTimes(1)
    expect(one.onEnd).toHaveBeenCalledTimes(1)
  })
})

describe('the server voice: the browser’s voice for what the server can’t say', () => {
  it('speaks a 503 tts_unavailable piece in the browser’s voice; the next waits for it, plays, and the turn drains', async () => {
    const { port, server, audio, browser, time } = serverSpeechSetup()
    const onEvent = vi.fn()
    const speaker = createSpeaker(port, time.clock, onEvent)
    speaker.speak(['One.', 'Two.'])
    await flush()

    server.refuse(0, 'tts_unavailable', 503)
    server.answer(1)
    await flush()
    expect(browser.spoken()).toEqual(['One.'])
    expect(audio.plays).toHaveLength(0)
    browser.start(0)
    browser.end(0)
    await flush()
    expect(audio.plays).toHaveLength(1)
    audio.plays[0].events.onPlaying()
    audio.plays[0].events.onEnded()

    expect(onEvent.mock.calls.map(([event]) => event)).toEqual([
      { type: 'sentence', text: 'One.' },
      { type: 'sentence', text: 'Two.' },
      { type: 'drained' },
    ])
    expect(speaker.speaking()).toBe(false)
  })

  it('after tts_off asks the server nothing more this page session: every piece goes to the browser’s voice', async () => {
    const { port, server, browser, options } = serverSpeechSetup()
    port.speak('One.', utterance())
    await flush()
    server.refuse(0, 'tts_off', 503)
    await flush()

    port.speak('Two.', utterance())
    port.speak('Three.', utterance())
    // Another engine in the same tab (the same page session) asks nothing either.
    serverSpeechPort(options).speak('Four.', utterance())
    await flush()

    expect(browser.spoken()).toEqual(['One.', 'Two.', 'Three.', 'Four.'])
    expect(server.fetchSpeech).toHaveBeenCalledTimes(1)
  })

  it('speaks a piece whose request never reached the server in the browser’s voice, and asks again for the next', async () => {
    const { port, server, browser } = serverSpeechSetup()
    port.speak('One.', utterance())
    await flush()

    server.calls[0].reject(new TypeError('Failed to fetch'))
    await flush()
    browser.play(0)
    port.speak('Two.', utterance())
    await flush()

    expect(browser.spoken()).toEqual(['One.'])
    expect(server.texts()).toEqual(['One.', 'Two.'])
  })

  it('gives the head SERVER_VOICE_WAIT_MS for its MP3, then aborts it and the browser’s voice speaks it', async () => {
    const { port, server, audio, browser, time } = serverSpeechSetup()
    port.speak('One.', utterance())
    await flush()

    time.advance(SERVER_VOICE_WAIT_MS - 1)
    expect(browser.spoken()).toEqual([])
    time.advance(1)
    expect(browser.spoken()).toEqual(['One.'])
    expect(server.calls[0].signal.aborted).toBe(true)
    server.answer(0)
    await flush()
    expect(audio.plays).toHaveLength(0)
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

  it('speaks a piece whose audio fails before it starts in the browser’s voice; after it started, the piece just ends', async () => {
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
    expect(browser.spoken()).toEqual(['One.'])
    browser.play(0)
    expect([one.onStart.mock.calls.length, one.onEnd.mock.calls.length]).toEqual([1, 1])
    audio.plays[1].events.onPlaying()
    audio.plays[1].events.onFailed(new Error('network'))

    expect([two.onStart.mock.calls.length, two.onEnd.mock.calls.length]).toEqual([1, 1])
    expect(browser.spoken()).toEqual(['One.'])
  })

  it('ends a piece the server can’t say at once where the browser has no voice', async () => {
    const { port, server } = serverSpeechSetup({ fallback: null })
    const one = utterance()
    port.speak('One.', one)
    await flush()

    server.refuse(0, 'tts_busy', 429)
    await flush()

    expect(one.onEnd).toHaveBeenCalledTimes(1)
    expect(one.onStart).not.toHaveBeenCalled()
  })
})

describe('the server voice: cancel and prime', () => {
  it('cancel aborts the requests in flight, stops the audio and the browser’s voice; late answers and callbacks do nothing', async () => {
    const { port, server, audio, browser } = serverSpeechSetup()
    const one = utterance()
    const two = utterance()
    port.speak('One.', one)
    port.speak('Two.', two)
    port.speak('Three.', utterance())
    await flush()
    server.answer(0)
    await flush()
    audio.plays[0].events.onPlaying()

    port.cancel()
    audio.plays[0].events.onProgress()
    audio.plays[0].events.onEnded()
    server.answer(1)
    await flush()

    expect(server.calls[1].signal.aborted).toBe(true)
    expect(audio.player.stop).toHaveBeenCalledTimes(1)
    expect(browser.port.cancel).toHaveBeenCalledTimes(1)
    expect([one.onBoundary, one.onEnd, two.onStart, two.onEnd].map((mock) => mock.mock.calls.length)).toEqual([0, 0, 0, 0])
    expect(audio.plays).toHaveLength(1)
    expect(server.texts()).toEqual(['One.', 'Two.'])
  })

  it('cancel silences the late callbacks of a piece the browser’s voice had', async () => {
    const { port, server, browser } = serverSpeechSetup()
    const one = utterance()
    port.speak('One.', one)
    await flush()
    server.refuse(0, 'tts_unavailable', 503)
    await flush()

    port.cancel()
    browser.play(0)

    expect([one.onStart.mock.calls.length, one.onEnd.mock.calls.length]).toEqual([0, 0])
  })

  it('prime unlocks the audio and primes the browser’s voice in the gesture, and asks for the voice list once', async () => {
    const { port, audio, browser, fetchVoices } = serverSpeechSetup()

    port.prime()
    port.prime()
    await flush()

    expect(audio.player.unlock).toHaveBeenCalledTimes(2)
    expect(browser.port.prime).toHaveBeenCalledTimes(2)
    expect(fetchVoices).toHaveBeenCalledTimes(1)
  })

  it('still primes the browser’s voice when the audio can’t be unlocked', () => {
    const { port, audio, browser } = serverSpeechSetup()
    audio.player.unlock.mockImplementation(() => {
      throw new Error('no audio')
    })

    port.prime()

    expect(browser.port.prime).toHaveBeenCalledTimes(1)
  })
})

describe('the server voice: the voice for the page’s language', () => {
  it('asks for a pt-BR voice for a pt-BR page', async () => {
    const { port, server } = serverSpeechSetup({ language: 'pt-BR' })
    port.speak('Olá.', utterance())
    await flush()

    expect(server.calls.map((call) => call.voice)).toEqual(['pf_dora'])
  })

  it('asks the server nothing for a language it has no voice in (pt-PT, Kokoro has pt-BR only)', async () => {
    const { port, server, browser } = serverSpeechSetup({ language: 'pt-PT' })
    port.speak('Olá.', utterance())
    await flush()
    port.speak('Adeus.', utterance())

    expect(server.fetchSpeech).not.toHaveBeenCalled()
    expect(browser.spoken()).toEqual(['Olá.', 'Adeus.'])
  })

  it('speaks in the browser’s voice when the voice list can’t be had', async () => {
    const { port, server, browser } = serverSpeechSetup({ fetchVoices: async () => Promise.reject(new Error('503')) })
    port.speak('Hello.', utterance())
    await flush()

    expect(server.fetchSpeech).not.toHaveBeenCalled()
    expect(browser.spoken()).toEqual(['Hello.'])
  })
})
