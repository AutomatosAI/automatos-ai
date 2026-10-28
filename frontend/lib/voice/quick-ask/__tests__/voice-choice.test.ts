/**
 * Theme voice, web quick-ask — which server voice speaks (voice-choice.ts): the server's default when it speaks the
 * page's language, else the page's exact accent, else (a page language with no region) the language; none for a
 * regional language the server has no voice in exactly (pt-PT with only pt-BR) or a list that can't be had. The page
 * session asks for the list once and remembers tts_off.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import { chooseVoice, createServerVoiceSession, parseVoiceList } from '@/lib/voice/quick-ask/voice-choice'

import { VOICES } from './server-speech-fakes'

vi.mock('@/lib/api-client', () => ({ apiClient: {} }))

afterEach(() => {
  vi.restoreAllMocks()
})

const LIST = parseVoiceList(VOICES)

describe('parseVoiceList', () => {
  it('reads the default and each voice’s id and accent, dropping a malformed voice', () => {
    const list = parseVoiceList({ default: 'af_heart', voices: [{ id: 'af_heart', accent: 'en-US' }, { id: 7 }, null] })

    expect(list).toEqual({ defaultId: 'af_heart', voices: [{ id: 'af_heart', accent: 'en-US' }] })
  })

  it('is null for a body that is no voice list, and names no default when the server names none', () => {
    expect(parseVoiceList({ cancelled: true })).toBeNull()
    expect(parseVoiceList('voices')).toBeNull()
    expect(parseVoiceList(null)).toBeNull()
    expect(parseVoiceList({ voices: [] })).toEqual({ defaultId: null, voices: [] })
  })
})

describe('chooseVoice', () => {
  it('takes the server’s default for any accent of its language (en-US, en-GB, en-IN)', () => {
    expect(chooseVoice(LIST, 'en-US')).toBe('af_heart')
    expect(chooseVoice(LIST, 'en-GB')).toBe('af_heart')
    expect(chooseVoice(LIST, 'en-IN')).toBe('af_heart')
    expect(chooseVoice(LIST, 'en')).toBe('af_heart')
  })

  it('takes the first voice in exactly the page’s accent otherwise, whatever the case', () => {
    expect(chooseVoice(LIST, 'pt-BR')).toBe('pf_dora')
    expect(chooseVoice(LIST, 'fr-fr')).toBe('ff_siwis')
  })

  it('takes the first voice in the language for a page language that names no region', () => {
    expect(chooseVoice(LIST, 'pt')).toBe('pf_dora')
  })

  it('has none for a region the server has no voice in exactly, or a language it can’t speak', () => {
    expect(chooseVoice(LIST, 'pt-PT')).toBeNull()
    expect(chooseVoice(LIST, 'fr-CA')).toBeNull()
    expect(chooseVoice(LIST, 'de-DE')).toBeNull()
    expect(chooseVoice(null, 'en-US')).toBeNull()
  })

  it('falls back to the exact accent when the default is not among the voices', () => {
    const list = parseVoiceList({ ...VOICES, default: 'xx_gone' })

    expect(chooseVoice(list, 'en-GB')).toBe('bf_emma')
    expect(chooseVoice(list, 'en-US')).toBe('af_heart')
  })
})

describe('createServerVoiceSession', () => {
  it('asks for the list once a page session, and is not serverless while it waits or once it has a voice', async () => {
    const fetchVoices = vi.fn(async (): Promise<unknown> => VOICES)
    const session = createServerVoiceSession(fetchVoices, 'en-GB')

    const first = session.voice()
    expect(session.serverless()).toBe(false)
    await expect(first).resolves.toBe('af_heart')
    await expect(session.voice()).resolves.toBe('af_heart')

    expect(fetchVoices).toHaveBeenCalledTimes(1)
    expect(session.serverless()).toBe(false)
  })

  it('has no voice, and is serverless, when the list can’t be had or holds none for the language', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const failed = createServerVoiceSession(async () => Promise.reject(new Error('503 tts_off')), 'en-US')
    const portuguese = createServerVoiceSession(async () => VOICES, 'pt-PT')

    await expect(failed.voice()).resolves.toBeNull()
    await expect(portuguese.voice()).resolves.toBeNull()

    expect(failed.serverless()).toBe(true)
    expect(portuguese.serverless()).toBe(true)
  })

  it('is serverless for the rest of the page session once the server answered tts_off', () => {
    const session = createServerVoiceSession(async () => VOICES, 'en-US')

    session.turnOff()

    expect(session.serverless()).toBe(true)
  })
})
