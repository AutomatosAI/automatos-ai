'use client'

import { useEffect, useRef } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import type { ChatStatus } from '@/lib/chat/chat-status'

/**
 * PRD-260 P260-FX-10 (M44) — a turn changes the chat history: a new chat joins
 * it, an old one moves to the top with a new title or last line. When the chat's
 * status leaves streaming (the reply finished, was stopped or failed), every
 * history list (Conversations, Recent) is asked again, rather than waiting out
 * the root QueryClient's staleTime.
 */

/** Every chat-history query's key starts with this, so one invalidation refreshes them all. */
export const CHAT_HISTORY_QUERY_PREFIX = ['chat-history'] as const

const STREAMING: ChatStatus = 'streaming'

export function useRefreshHistoryOnTurnEnd(status: ChatStatus): void {
  const queryClient = useQueryClient()
  const previous = useRef(status)
  useEffect(() => {
    const turnEnded = previous.current === STREAMING && status !== STREAMING
    previous.current = status
    if (turnEnded) void queryClient.invalidateQueries({ queryKey: CHAT_HISTORY_QUERY_PREFIX })
  }, [status, queryClient])
}
