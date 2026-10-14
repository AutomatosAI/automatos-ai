'use client'

import { useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { CHAT_HREF } from '@/lib/ticket-links'
import { useChatSessionStore } from '@/stores/chat-session-store'

/**
 * PRD-260 — Play's New chat (the sidebar's and the Conversations page's): a
 * fresh draft in the chat session store, then the chat.
 */
export function useStartChat(): () => void {
  const router = useRouter()
  const newDraft = useChatSessionStore((s) => s.newDraft)
  return useCallback(() => {
    newDraft()
    router.push(CHAT_HREF)
  }, [newDraft, router])
}
