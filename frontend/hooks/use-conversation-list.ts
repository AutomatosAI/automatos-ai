'use client'

import { useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useChatScope } from '@/hooks/use-chat-session'
import { CHAT_HISTORY_QUERY_PREFIX } from '@/hooks/use-refresh-history-on-turn-end'
import { getChatHistory } from '@/lib/chat/api'
import type { Chat } from '@/types'

/**
 * PRD-260 T-006 — the Conversations page's list: getChatHistory
 * (lib/chat/api.ts), newest activity first, at most CONVERSATION_LIMIT; the
 * page's search filters what was loaded. Ported from Automatos Play's
 * useConversations onto react-query. A failed load is logged here with its
 * detail; the page says only that it failed, with Try again. The cache is the
 * account's own (useChatScope), asked again each time the page opens and after
 * every turn (useRefreshHistoryOnTurnEnd): P260-FX-10.
 */

/** The history's page size. */
export const CONVERSATION_LIMIT = 50
export const CONVERSATION_LIST_QUERY_KEY = [...CHAT_HISTORY_QUERY_PREFIX, 'conversations'] as const

export type ConversationListState =
  | { readonly status: 'loading' }
  | { readonly status: 'failed'; readonly retry: () => void }
  | { readonly status: 'loaded'; readonly chats: readonly Chat[] }

async function loadConversations(limit: number): Promise<Chat[]> {
  try {
    return await getChatHistory(limit)
  } catch (error) {
    console.error('[play-conversations] the conversation list could not be loaded:', error)
    throw error
  }
}

export function useConversationList(limit: number = CONVERSATION_LIMIT): ConversationListState {
  const scope = useChatScope()
  const query = useQuery({
    queryKey: [...CONVERSATION_LIST_QUERY_KEY, scope, limit],
    queryFn: () => loadConversations(limit),
    enabled: scope !== null,
    // A conversation started since the page was last open is on it when it opens again.
    refetchOnMount: 'always',
  })
  const { refetch } = query
  const retry = useCallback(() => {
    void refetch()
  }, [refetch])
  // A list once loaded stays on screen, even when asking again fails.
  if (query.data) return { status: 'loaded', chats: query.data }
  // Try again shows the loading line until the answer, not the old failure.
  if (query.isError && !query.isFetching) return { status: 'failed', retry }
  return { status: 'loading' }
}
