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
 * saved rows into the conversation. At that moment the exchange is settled: its words hide, and so do the same words
 * grown on, since Retell's transcript of Auto keeps catching up with a reply the backend saved already. A new
 * exchange (other words) shows again. An empty exchange (a new hold-to-talk turn) clears both.
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

export interface LiveTurnState {
  /** The session's exchange in progress; null when there is none. */
  readonly turn: LiveVoiceTurn | null
  /** The exchange as it was when its conversation's saved rows landed. */
  readonly settled: LiveVoiceTurn | null
  readonly publish: (turn: LiveVoiceTurn) => void
  /** `chatId`'s saved rows landed (chat_changed): the exchange in progress there is saved. */
  readonly settle: (chatId: string) => void
  readonly clear: () => void
}

function sameTurn(a: LiveVoiceTurn, b: LiveVoiceTurn): boolean {
  return a.chatId === b.chatId && a.userText === b.userText && a.agentText === b.agentText
}

export const useLiveTurnStore = create<LiveTurnState>((set, get) => ({
  turn: null,
  settled: null,
  publish: (turn) => {
    if (!turn.userText && !turn.agentText) {
      set({ turn: null, settled: null })
      return
    }
    const current = get().turn
    if (current && sameTurn(current, turn)) return
    set({ turn: Object.freeze({ chatId: turn.chatId, userText: turn.userText, agentText: turn.agentText }) })
  },
  settle: (chatId) => {
    const { turn, settled } = get()
    if (turn && turn.chatId === chatId && turn !== settled) set({ settled: turn })
  },
  clear: () => set({ turn: null, settled: null }),
}))

/** `text` unless it is the saved text or the saved text grown on. */
function unsaved(text: string, savedText: string | undefined): string {
  if (!text) return ''
  return savedText && text.startsWith(savedText) ? '' : text
}

/** The words still to show in `chatId`'s conversation: null when there are none (or they belong elsewhere). */
export function visibleLiveTurn(
  turn: LiveVoiceTurn | null,
  settled: LiveVoiceTurn | null,
  chatId: string | null | undefined,
): LiveVoiceWords | null {
  if (!turn || !chatId || turn.chatId !== chatId) return null
  const saved = settled && settled.chatId === chatId ? settled : null
  const userText = unsaved(turn.userText, saved?.userText)
  const agentText = unsaved(turn.agentText, saved?.agentText)
  return userText || agentText ? Object.freeze({ userText, agentText }) : null
}
