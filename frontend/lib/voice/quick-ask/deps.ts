/**
 * Theme voice, web quick-ask: everything the engine reaches outside itself, injectable so its tests use fakes. The
 * browser's defaults: speech recognition and speech synthesis in the page's language, the chat turn sent with
 * spokenMode (lib/chat/turn-stream.ts), the chat's own Stop (lib/chat/api.ts cancelChatTurn), the chat_changed
 * window event the open conversation merges on, the page's clock, this tab's sessionStorage, and the tab being hidden.
 *
 * Built only when an engine is created (in the browser): importing this module touches no browser object.
 */
import { cancelChatTurn } from '@/lib/chat/api'
import { streamChatTurn, type ChatTurnSender } from '@/lib/chat/turn-stream'

import type { NoticeStorage } from './ai-notice'
import { SYSTEM_CLOCK, type QuickAskClock } from './clock'
import { browserRecognitionConstructor, browserRecognitionPort, type RecognitionFactory } from './recognition'
import { browserSynthesisPort, type SynthesisPort } from './speaker'

export interface QuickAskDeps {
  /** A new recognition port per hold; null when the browser has no speech recognition. */
  readonly recognition: RecognitionFactory
  /** The browser's speech synthesis; null when it has none. */
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

/** The window event the board stream fans chat_changed out as; useChat and the theme chat merge on it. */
export const CHAT_CHANGED_EVENT = 'automatos:chat-changed'
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

function tabStorage(): NoticeStorage | null {
  try {
    return typeof window === 'undefined' ? null : window.sessionStorage
  } catch (error) {
    console.warn(LOG_PREFIX, 'sessionStorage is not available', error)
    return null
  }
}

function announceTurn(chatId: string): void {
  window.dispatchEvent(new CustomEvent(CHAT_CHANGED_EVENT, { detail: { chat_id: chatId } }))
}

function onPageHidden(callback: () => void): () => void {
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
    synthesis: browserSynthesis(),
    sendTurn: streamChatTurn,
    cancelTurn: cancelChatTurn,
    announceTurn,
    clock: SYSTEM_CLOCK,
    storage: tabStorage(),
    onPageHidden,
  })
}
