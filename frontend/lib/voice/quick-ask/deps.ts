/**
 * Theme voice, web quick-ask: everything the engine reaches outside itself, injectable so its tests use fakes. The
 * browser's defaults: speech recognition in the page's language; Auto's server voice (server-speech.ts, Kokoro through
 * POST /api/voice/speech) in the page's language, with the browser's speech synthesis behind it for what the server
 * can't say; the chat turn sent with spokenMode (lib/chat/turn-stream.ts), the chat's own Stop (lib/chat/api.ts
 * cancelChatTurn), the chat_changed window event the open conversation merges on, the page's clock, this tab's
 * sessionStorage, and the tab being hidden.
 *
 * Built only when an engine is created (in the browser): importing this module touches no browser object, and the
 * server voice's list is asked for only when Auto first speaks or a hold primes it.
 */
import { cancelChatTurn } from '@/lib/chat/api'
import { streamChatTurn, type ChatTurnSender } from '@/lib/chat/turn-stream'
import { CHAT_CHANGED_EVENT } from '@/lib/voice/live-turns'

import type { NoticeStorage } from './ai-notice'
import { browserAudioPlayer } from './audio-player'
import { SYSTEM_CLOCK, type QuickAskClock } from './clock'
import { browserRecognitionConstructor, browserRecognitionPort, type RecognitionFactory } from './recognition'
import { serverSpeechPort } from './server-speech'
import { browserSynthesisPort, type SynthesisPort } from './speaker'
import { createServerVoiceSession, type ServerVoiceSession } from './voice-choice'
import { fetchSpeech, fetchVoices } from './voice-api'

export interface QuickAskDeps {
  /** A new recognition port per hold; null when the browser has no speech recognition. */
  readonly recognition: RecognitionFactory
  /** Auto's voice: the server's, with the browser's speech synthesis behind it; null for none (a test's choice). */
  readonly synthesis: SynthesisPort | null
  /** Sends the heard words as a chat turn with spokenMode and streams the reply back. */
  readonly sendTurn: ChatTurnSender
  /** The chat's Stop: cancels the turn on the server, whichever worker holds it. */
  readonly cancelTurn: (chatId: string) => Promise<unknown>
  /**
   * The turn landed in the conversation. The backend notifies chat_changed only to a client that missed the end of
   * a turn, and this engine read it to the end, so it says so itself: the open chat merges the saved messages.
   */
  readonly announceTurn: (chatId: string) => void
  readonly clock: QuickAskClock
  /** This tab's sessionStorage (the AI notice); null when site data is blocked. */
  readonly storage: NoticeStorage | null
  /** Calls back when the tab is hidden or the page unloads; returns the unsubscribe. */
  readonly onPageHidden: (callback: () => void) => () => void
}

/** Recognition and speech language when the browser names none. */
export const FALLBACK_LANGUAGE = 'en-US'

const LOG_PREFIX = '[voice]'

function pageLanguage(): string {
  return (typeof navigator !== 'undefined' && navigator.language) || FALLBACK_LANGUAGE
}

function recognitionFactory(): RecognitionFactory {
  return () => {
    const Recognition = browserRecognitionConstructor()
    return Recognition === null ? null : browserRecognitionPort(Recognition, pageLanguage())
  }
}

function browserSynthesis(): SynthesisPort | null {
  if (typeof window === 'undefined' || !('speechSynthesis' in window)) return null
  return browserSynthesisPort(window.speechSynthesis, pageLanguage())
}

let pageVoice: ServerVoiceSession | null = null

/** The page session's server voice, whichever engine speaks: its voice list asked for once, a tts_off remembered. */
function pageVoiceSession(): ServerVoiceSession {
  if (pageVoice === null) pageVoice = createServerVoiceSession(fetchVoices, pageLanguage())
  return pageVoice
}

/** Auto's voice: the server's, with the browser's (when it has one) for every piece the server can't say. */
function speechPort(): SynthesisPort {
  return serverSpeechPort({
    fetchSpeech,
    player: browserAudioPlayer(),
    fallback: browserSynthesis(),
    session: pageVoiceSession(),
    clock: SYSTEM_CLOCK,
  })
}

function tabStorage(): NoticeStorage | null {
  try {
    return typeof window === 'undefined' ? null : window.sessionStorage
  } catch (error) {
    console.warn(LOG_PREFIX, 'sessionStorage is not available', error)
    return null
  }
}

function announceTurn(chatId: string): void {
  if (typeof window === 'undefined') return
  window.dispatchEvent(new CustomEvent(CHAT_CHANGED_EVENT, { detail: { chat_id: chatId } }))
}

/** Hidden: the tab went to the background or was switched away from; pagehide: the page is unloading. */
function onPageHidden(callback: () => void): () => void {
  if (typeof window === 'undefined' || typeof document === 'undefined') return () => undefined
  const onVisibility = (): void => {
    if (document.visibilityState === 'hidden') callback()
  }
  document.addEventListener('visibilitychange', onVisibility)
  window.addEventListener('pagehide', callback)
  return () => {
    document.removeEventListener('visibilitychange', onVisibility)
    window.removeEventListener('pagehide', callback)
  }
}

/** The browser's own deps. */
export function browserDeps(): QuickAskDeps {
  return Object.freeze({
    recognition: recognitionFactory(),
    synthesis: speechPort(),
    sendTurn: streamChatTurn,
    cancelTurn: cancelChatTurn,
    announceTurn,
    clock: SYSTEM_CLOCK,
    storage: tabStorage(),
    onPageHidden,
  })
}
