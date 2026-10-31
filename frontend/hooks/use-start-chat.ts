'use client'

import { useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { useChatSessionHydration } from '@/hooks/use-chat-session'
import { CHAT_HREF } from '@/lib/ticket-links'
import { useChatSessionStore } from '@/stores/chat-session-store'
import { useFreshDraftStore } from '@/stores/fresh-draft-store'

/**
 * PRD-260 — Play's New chat (the sidebar's, the Conversations page's and the
 * empty shelf's): a fresh draft in the chat session store, then the chat. New
 * chat lives on every page, so it binds the store itself (the hydration is
 * idempotent) and does nothing until the store is hydrated: a draft written
 * before would be overwritten by the stored session, and in SaaS would save an
 * empty tab list to the server. It also tells Play's chat to show a fresh new
 * chat when one is already on screen (stores/fresh-draft-store.ts).
 */

export interface StartChat {
  /** A fresh draft, then /chat; nothing until `ready`. */
  readonly startChat: () => void
  /** The chat session store is hydrated: New chat can be pressed. */
  readonly ready: boolean
}

export function useStartChat(): StartChat {
  const router = useRouter()
  const { hydrated } = useChatSessionHydration()
  const newDraft = useChatSessionStore((s) => s.newDraft)
  const freshDraft = useFreshDraftStore((s) => s.freshDraft)
  const startChat = useCallback(() => {
    if (!hydrated) return
    newDraft()
    freshDraft()
    router.push(CHAT_HREF)
  }, [hydrated, newDraft, freshDraft, router])
  return { startChat, ready: hydrated }
}
