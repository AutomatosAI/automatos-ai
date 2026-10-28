/**
 * Theme voice, web quick-ask — the server voice's test fakes: POST /api/voice/speech whose answers the test gives one
 * by one, an audio player whose clips the test plays, the voice list the server sends, and a port over them with the
 * harness's fake browser voice and hand-moved clock. A test file that imports this mocks apiClient (voice-api.ts and,
 * through quick-ask-harness, the engine's browser defaults import it).
 */
import { vi } from 'vitest'

import type { AudioPlayer, PlaybackEvents } from '@/lib/voice/quick-ask/audio-player'
import { serverSpeechPort } from '@/lib/voice/quick-ask/server-speech'
import type { SynthesisPort, UtteranceEvents } from '@/lib/voice/quick-ask/speaker'
import { createServerVoiceSession } from '@/lib/voice/quick-ask/voice-choice'

import { fakeSynthesis, manualClock } from './quick-ask-harness'

/** GET /api/voice/voices as voice-tts lists Kokoro's voices: English first, the default American English. */
export const VOICES = Object.freeze({
  default: 'af_heart',
  voices: [
    { id: 'af_heart', name: 'Heart', accent: 'en-US', gender: 'female' },
    { id: 'af_bella', name: 'Bella', accent: 'en-US', gender: 'female' },
    { id: 'bf_emma', name: 'Emma', accent: 'en-GB', gender: 'female' },
    { id: 'ff_siwis', name: 'Siwis', accent: 'fr-FR', gender: 'female' },
    { id: 'pf_dora', name: 'Dora', accent: 'pt-BR', gender: 'female' },
    { id: 'pm_alex', name: 'Alex', accent: 'pt-BR', gender: 'male' },
  ],
})

/** An MP3 as the server streams it. */
export function mp3(label: string): Blob {
  return new Blob([label], { type: 'audio/mpeg' })
}

/** apiClient's error for a refusal: the status, and the server's {code, detail} as the body. */
export function refusal(code: string, status: number): Error {
  return Object.assign(new Error(code), { status, body: { code, detail: code } })
}

interface SpeechCall {
  readonly text: string
  readonly voice: string
  readonly signal: AbortSignal
  readonly resolve: (clip: Blob) => void
  readonly reject: (error: unknown) => void
}

export function fakeSpeechServer() {
  const calls: SpeechCall[] = []
  const fetchSpeech = vi.fn(
    (text: string, voice: string, signal: AbortSignal) =>
      new Promise<Blob>((resolve, reject) => {
        calls.push({ text, voice, signal, resolve, reject })
      }),
  )
  return {
    fetchSpeech,
    calls,
    texts: (): string[] => calls.map((call) => call.text),
    answer: (index: number) => calls[index].resolve(mp3(calls[index].text)),
    refuse: (index: number, code: string, status: number) => calls[index].reject(refusal(code, status)),
  }
}

export function fakePlayer() {
  const plays: { readonly clip: Blob; readonly events: PlaybackEvents }[] = []
  const player = {
    play: vi.fn((clip: Blob, events: PlaybackEvents) => {
      plays.push({ clip, events })
    }),
    stop: vi.fn(),
    unlock: vi.fn(),
  } satisfies AudioPlayer
  return { player, plays }
}

/** One piece's events, each a mock. */
export function utterance() {
  return { onStart: vi.fn(), onBoundary: vi.fn(), onEnd: vi.fn() } satisfies UtteranceEvents
}

export interface ServerSpeechSetup {
  /** The browser's voice behind the server's; null for a browser with none. Default: the harness's fake. */
  readonly fallback?: SynthesisPort | null
  /** The page's language. Default en-US. */
  readonly language?: string
  /** GET /api/voice/voices. Default: VOICES. */
  readonly fetchVoices?: () => Promise<unknown>
}

/** The server voice port over fakes, with what the test drives. */
export function serverSpeechSetup(setup: ServerSpeechSetup = {}) {
  const server = fakeSpeechServer()
  const audio = fakePlayer()
  const browser = fakeSynthesis()
  const time = manualClock()
  const fetchVoices = vi.fn(setup.fetchVoices ?? (async (): Promise<unknown> => VOICES))
  const session = createServerVoiceSession(fetchVoices, setup.language ?? 'en-US')
  const fallback = setup.fallback === undefined ? browser.port : setup.fallback
  const options = { fetchSpeech: server.fetchSpeech, player: audio.player, fallback, session, clock: time.clock }
  const port = serverSpeechPort(options)
  return { port, options, server, audio, browser, time, session, fetchVoices }
}
