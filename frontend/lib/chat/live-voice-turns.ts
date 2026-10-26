'use client'

/**
 * PRD-207, theme voice — a voice session's words in the conversation they belong to, as the thread's own messages
 * (Gerard: "same as old, she just talks it"): what the member is saying and what Auto is saying, as two entries at
 * the end of the conversation's messages (`voice-live-user`, `voice-live-assistant`), on the same render path as a
 * typed reply. They come from the host voice provider's exchange in progress (lib/voice/live-turns.ts), so Studio's
 * chat and the chat controller every theme's chat screen runs on (lib/chat-controller) show them alike; both stream
 * through useChatWithReceipts, which adds them here.
 *
 * They are never written into useChat's state: when chat_changed brings the saved rows, the exchange is settled and
 * the entries go, and the saved rows stay. Before this, Studio's chat upserted them itself, and only there.
 */
import { useMemo } from 'react'

import { type LiveVoiceWords, useLiveTurnStore, visibleLiveTurn } from '@/lib/voice/live-turns'
import type { ChatMessage } from '@/types'

export const VOICE_LIVE_USER_ID = 'voice-live-user'
export const VOICE_LIVE_ASSISTANT_ID = 'voice-live-assistant'

function liveEntry(id: string, role: 'user' | 'assistant', text: string): ChatMessage {
  return { id, role, content: text, parts: [{ type: 'text', text }] } as unknown as ChatMessage
}

/** `messages` with the session's words at the end; `messages` itself when there are none. */
export function withLiveVoiceTurn(messages: ChatMessage[], words: LiveVoiceWords | null): ChatMessage[] {
  if (!words) return messages
  const live = [
    ...(words.userText ? [liveEntry(VOICE_LIVE_USER_ID, 'user', words.userText)] : []),
    ...(words.agentText ? [liveEntry(VOICE_LIVE_ASSISTANT_ID, 'assistant', words.agentText)] : []),
  ]
  return live.length > 0 ? [...messages, ...live] : messages
}

/** The conversation `chatId`'s messages with the voice session's words in progress there, if any. */
export function useLiveVoiceMessages(messages: ChatMessage[], chatId: string | null | undefined): ChatMessage[] {
  const turn = useLiveTurnStore((state) => state.turn)
  const saved = useLiveTurnStore((state) => state.saved)
  const words = useMemo(() => visibleLiveTurn(turn, saved, chatId), [turn, saved, chatId])
  return useMemo(() => withLiveVoiceTurn(messages, words), [messages, words])
}
