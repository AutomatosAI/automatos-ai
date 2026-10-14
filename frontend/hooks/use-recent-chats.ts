'use client'

import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getChatHistory } from '@/lib/chat/api'
import { useChatSessionStore } from '@/stores/chat-session-store'
import type { Chat } from '@/types'

/**
 * PRD-260 T-004 — the latest conversations for Play's sidebar (its Recent list),
 * from getChatHistory (lib/chat/api.ts). Asked again when the conversation on
 * screen changes, so a new chat joins the list once it has an id.
 */

/** The most the Recent list shows. */
export const RECENT_CHATS_LIMIT = 8
export const RECENT_CHATS_QUERY_KEY = ['chat-history', 'recent'] as const

const NO_CHATS: readonly Chat[] = []

export function useRecentChats(limit: number = RECENT_CHATS_LIMIT): readonly Chat[] {
  const activeChatId = useChatSessionStore((s) => s.session.activeChatId)
  const { data } = useQuery({
    queryKey: [...RECENT_CHATS_QUERY_KEY, limit, activeChatId],
    queryFn: () => getChatHistory(limit),
    // The list stays on screen while it is asked again for a new active chat.
    keepPreviousData: true,
  })
  return useMemo(() => (data ? data.slice(0, limit) : NO_CHATS), [data, limit])
}
