/**
 * Theme voice, web quick-ask: Auto's server voice through apiClient (its base URL, the Clerk bearer where there is one
 * and X-Workspace-ID, as every backend call; core #1348, orchestrator/api/voice_speech.py, permission agents:execute
 * as POST /api/chat). POST /api/voice/speech answers one MP3 for a piece of a reply; GET /api/voice/voices the
 * voices. A refusal is apiClient's error, whose `body` is the server's {code, detail}: refusalCode reads the code
 * (tts_off: this server has no voice; tts_unavailable, tts_bad_request, tts_too_long, tts_busy).
 */
import { apiClient } from '@/lib/api-client'

export const SPEECH_PATH = '/api/voice/speech'
export const VOICES_PATH = '/api/voice/voices'
/** The refusal of a server with no voice set up (VOICE_TTS_URL empty): it asks nothing more this page session. */
export const TTS_OFF = 'tts_off'

const AUDIO_TYPE_PREFIX = 'audio/'

/** One piece of a reply as audio, in the voice; rejects on any answer that is not audio. */
export type FetchSpeech = (text: string, voice: string, signal: AbortSignal) => Promise<Blob>

/** The piece as MP3, at the voice's normal speed; rejects on a refusal, a network error or a body that isn't audio. */
export async function fetchSpeech(text: string, voice: string, signal: AbortSignal): Promise<Blob> {
  const clip = await apiClient.requestBlob(SPEECH_PATH, {
    method: 'POST',
    body: JSON.stringify({ text, voice, speed: null }),
    signal,
  })
  if (clip.size === 0 || !clip.type.startsWith(AUDIO_TYPE_PREFIX)) {
    throw new Error('the server voice answered without audio')
  }
  return clip
}

/** The server's voices, as sent: voice-choice.ts reads them. */
export function fetchVoices(): Promise<unknown> {
  return apiClient.request<unknown>(VOICES_PATH)
}

/** The server's refusal code on a failed request (apiClient's error body), or null when there is none. */
export function refusalCode(error: unknown): string | null {
  if (typeof error !== 'object' || error === null) return null
  const body: unknown = (error as { readonly body?: unknown }).body
  if (typeof body !== 'object' || body === null) return null
  const code: unknown = (body as { readonly code?: unknown }).code
  return typeof code === 'string' ? code : null
}
