/**
 * Theme voice, web quick-ask — the server voice over apiClient (voice-api.ts): POST /api/voice/speech with the piece,
 * the voice and the voice's normal speed, the caller's abort signal passed on, an answer that isn't audio refused;
 * GET /api/voice/voices; the server's refusal code read from apiClient's error body.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ request: vi.fn(), requestBlob: vi.fn() }))
vi.mock('@/lib/api-client', () => ({ apiClient: { request: api.request, requestBlob: api.requestBlob } }))

import { fetchSpeech, fetchVoices, refusalCode, SPEECH_PATH, VOICES_PATH } from '@/lib/voice/quick-ask/voice-api'

import { mp3, refusal, VOICES } from './server-speech-fakes'

beforeEach(() => {
  api.request.mockReset()
  api.requestBlob.mockReset()
})

describe('fetchSpeech', () => {
  it('posts the piece and the voice at the normal speed through apiClient, with the caller’s abort signal', async () => {
    const audio = mp3('Hello.')
    api.requestBlob.mockResolvedValue(audio)
    const controller = new AbortController()

    await expect(fetchSpeech('Hello.', 'af_heart', controller.signal)).resolves.toBe(audio)

    const [path, init] = api.requestBlob.mock.calls[0] as [string, RequestInit]
    expect(path).toBe('/api/voice/speech')
    expect(SPEECH_PATH).toBe('/api/voice/speech')
    expect(init.method).toBe('POST')
    expect(JSON.parse(String(init.body))).toEqual({ text: 'Hello.', voice: 'af_heart', speed: null })
    expect(init.signal).toBe(controller.signal)
  })

  it('rejects an answer that is not audio, or is empty', async () => {
    const signal = new AbortController().signal
    api.requestBlob.mockResolvedValueOnce(new Blob(['<html>'], { type: 'text/html' }))
    api.requestBlob.mockResolvedValueOnce(new Blob([], { type: 'audio/mpeg' }))

    await expect(fetchSpeech('Hello.', 'af_heart', signal)).rejects.toThrow()
    await expect(fetchSpeech('Hello.', 'af_heart', signal)).rejects.toThrow()
  })

  it('passes a refusal on, its code readable', async () => {
    api.requestBlob.mockRejectedValue(refusal('tts_off', 503))

    const error = await fetchSpeech('Hello.', 'af_heart', new AbortController().signal).catch((caught: unknown) => caught)

    expect(refusalCode(error)).toBe('tts_off')
  })
})

describe('fetchVoices and refusalCode', () => {
  it('gets the voice list through apiClient', async () => {
    api.request.mockResolvedValue(VOICES)

    await expect(fetchVoices()).resolves.toBe(VOICES)
    expect(api.request).toHaveBeenCalledWith(VOICES_PATH)
    expect(VOICES_PATH).toBe('/api/voice/voices')
  })

  it('reads the code of a refusal, and none from anything else', () => {
    expect(refusalCode(refusal('tts_busy', 429))).toBe('tts_busy')
    expect(refusalCode(new TypeError('Failed to fetch'))).toBeNull()
    expect(refusalCode(Object.assign(new Error('x'), { body: { detail: 'no code' } }))).toBeNull()
    expect(refusalCode(undefined)).toBeNull()
    expect(refusalCode('tts_off')).toBeNull()
  })
})
