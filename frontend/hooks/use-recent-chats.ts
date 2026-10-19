'use client'

import { useEffect, useMemo, useRef } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useChatScope } from '@/hooks/use-chat-session'
import { CHAT_HISTORY_QUERY_PREFIX } from '@/hooks/use-refresh-history-on-turn-end'
import { getChatHistory } from '@/lib/chat/api'
import { useChatSessionStore } from '@/stores/chat-session-store'
import type { Chat } from '@/types'

/**
 * PRD-260 T-004 — the latest conversations for Play's sidebar (its Recent list),
 * from getChatHistory (lib/chat/api.ts). One list per account (useChatScope);
 * asked again when the conversation on screen changes, so a new chat joins the
 * list once it has an id, and after every turn (useRefreshHistoryOnTurnEnd):
 * P260-FX-10.
 */

/** The most the Recent list shows. */
export const RECENT_CHATS_LIMIT = 8
export const RECENT_CHATS_QUERY_KEY = [...CHAT_HISTORY_QUERY_PREFIX, 'recent'] as const

const NO_CHATS: readonly Chat[] = []

/** Asks Recent again when the active chat changes (not when it first shows: that load is under way). */
function useRefreshOnActiveChat(activeChatId: string | null): void {
  const queryClient = useQueryClient()
  const shown = useRef(activeChatId)
  useEffect(() => {
    if (shown.current === activeChatId) return
    shown.current = activeChatId
    void queryClient.invalidateQueries({ queryKey: RECENT_CHATS_QUERY_KEY })
  }, [activeChatId, queryClient])
}

export function useRecentChats(limit: number = RECENT_CHATS_LIMIT): readonly Chat[] {
  const scope = useChatScope()
  const activeChatId = useChatSessionStore((s) => s.session.activeChatId)
  useRefreshOnActiveChat(activeChatId)
  const { data } = useQuery({
    queryKey: [...RECENT_CHATS_QUERY_KEY, scope, limit],
    queryFn: () => getChatHistory(limit),
    enabled: scope !== null,
  })
  return useMemo(() => (data ? data.slice(0, limit) : NO_CHATS), [data, limit])
}
