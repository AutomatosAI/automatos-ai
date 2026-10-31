/**
 * Theme voice, web quick-ask — the server voice port (server-speech.ts): pieces play strictly in order with the next
 * one's MP3 asked for while the head plays (exactly one ahead), each piece's start, level and end told; cancel aborts,
 * stops and silences late callbacks; prime unlocks the audio and primes the browser's voice; the voice follows the
 * page's language. When the server voice fails, the rest of the answer goes to the browser's voice:
 * server-speech-fallback.test.ts.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/api-client', () => ({ apiClient: {} }))

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
