'use client'

import { useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getChatHistory } from '@/lib/chat/api'
import type { Chat } from '@/types'

/**
 * PRD-260 T-006 — the Conversations page's list: getChatHistory
 * (lib/chat/api.ts), newest activity first, at most CONVERSATION_LIMIT; the
 * page's search filters what was loaded. Ported from Automatos Play's
 * useConversations onto react-query. A failed load is logged here with its
 * detail; the page says only that it failed, with Try again.
 */

/** The history's page size. */
export const CONVERSATION_LIMIT = 50
export const CONVERSATION_LIST_QUERY_KEY = ['chat-history', 'conversations'] as const

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
  const query = useQuery({
    queryKey: [...CONVERSATION_LIST_QUERY_KEY, limit],
    queryFn: () => loadConversations(limit),
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
