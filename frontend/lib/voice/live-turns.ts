'use client'

import { create } from 'zustand'

/**
 * Theme voice — the words of the voice session in progress, for the conversation they belong to.
 *
 * The host voice provider publishes the session's exchange (what the member said, what Auto is saying) with the
 * conversation it writes into; every chat on the host's stream (Studio's chat, the chat controller a theme's chat
 * screen runs on) shows them as two live messages at the end of that conversation (lib/chat/live-voice-turns.ts), on
 * the same render path as typed words. Before this, only Studio's chat did, from its own call strip.
 *
 * The words are not saved here: the backend saves each spoken turn and says so with chat_changed, which brings the
 * saved rows into the conversation. At that moment the exchange is saved (`settle`): its live words hide for good,
 * since Retell's transcript of Auto keeps catching up with a reply the backend saved already. The member speaking
 * again is a new exchange (other words of theirs), and it shows. An empty exchange (a new hold-to-talk turn) clears
 * both.
 */

export interface LiveVoiceTurn {
  /** The conversation the session writes into. */
  readonly chatId: string
  readonly userText: string
  readonly agentText: string
}

export interface LiveVoiceWords {
  readonly userText: string
  readonly agentText: string
}

/** The exchange whose saved rows landed: the member's words of it, in its conversation. */
export interface SavedVoiceExchange {
  readonly chatId: string
  readonly userText: string
}

export interface LiveTurnState {
  /** The session's exchange in progress; null when there is none. */
  readonly turn: LiveVoiceTurn | null
  /** The exchange in progress when its conversation's saved rows landed; null once another one starts. */
  readonly saved: SavedVoiceExchange | null
  readonly publish: (turn: LiveVoiceTurn) => void
  /** `chatId`'s saved rows landed (chat_changed): the exchange in progress there is saved. */
  readonly settle: (chatId: string) => void
  readonly clear: () => void
}

/** The window event the SSE lane fans chat_changed out as (hooks/use-board-event-stream.ts). */
export const CHAT_CHANGED_EVENT = 'automatos:chat-changed'

function sameTurn(a: LiveVoiceTurn, b: LiveVoiceTurn): boolean {
  return a.chatId === b.chatId && a.userText === b.userText && a.agentText === b.agentText
}

function isExchangeOf(saved: SavedVoiceExchange | null, turn: LiveVoiceTurn): boolean {
  return saved !== null && saved.chatId === turn.chatId && saved.userText === turn.userText
}

export const useLiveTurnStore = create<LiveTurnState>((set, get) => ({
  turn: null,
  saved: null,
  publish: (turn) => {
    if (!turn.userText && !turn.agentText) {
      set({ turn: null, saved: null })
      return
    }
    const { turn: current, saved } = get()
    if (current && sameTurn(current, turn)) return
    set({
      turn: Object.freeze({ chatId: turn.chatId, userText: turn.userText, agentText: turn.agentText }),
      // Auto's words of a saved exchange keep growing; other words of the member's are a new exchange.
      saved: isExchangeOf(saved, turn) ? saved : null,
    })
  },
  settle: (chatId) => {
    const { turn, saved } = get()
    if (!turn || turn.chatId !== chatId || isExchangeOf(saved, turn)) return
    set({ saved: Object.freeze({ chatId, userText: turn.userText }) })
  },
  clear: () => set({ turn: null, saved: null }),
}))

/** The words still to show in `chatId`'s conversation: null when there are none (saved, or they belong elsewhere). */
export function visibleLiveTurn(
  turn: LiveVoiceTurn | null,
  saved: SavedVoiceExchange | null,
  chatId: string | null | undefined,
): LiveVoiceWords | null {
  if (!turn || !chatId || turn.chatId !== chatId || isExchangeOf(saved, turn)) return null
  if (!turn.userText && !turn.agentText) return null
  return Object.freeze({ userText: turn.userText, agentText: turn.agentText })
}

/** Settles the exchange of every conversation whose saved rows land on `target`; returns what stops listening. */
export function watchSavedVoiceTurns(target: Pick<Window, 'addEventListener' | 'removeEventListener'>): () => void {
  const onChatChanged = (event: Event) => {
    const chatId = (event as CustomEvent<{ chat_id?: unknown } | undefined>).detail?.chat_id
    if (typeof chatId === 'string' && chatId) useLiveTurnStore.getState().settle(chatId)
  }
  target.addEventListener(CHAT_CHANGED_EVENT, onChatChanged)
  return () => target.removeEventListener(CHAT_CHANGED_EVENT, onChatChanged)
}
