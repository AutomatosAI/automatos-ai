'use client'

/**
 * PRD-237: bind the conversation session store to the signed-in user and the
 * active workspace, keep it in step with other browser tabs, and flag open
 * conversations that move in the background.
 *
 * Mounted by every surface that shows a conversation (chat page, widget);
 * hydration is idempotent so several mounts cost one load.
 */
import { useEffect } from 'react'
import { useUser } from '@/lib/auth-hooks'
import { useWorkspaceOptional } from '@/components/workspace-provider'
import { chatSessionKey } from '@/lib/chat/chat-session'
import { useChatSessionStore } from '@/stores/chat-session-store'

/**
 * Whose conversations these are: the session key (the workspace and the
 * signed-in user) once both have settled, else null. The session binds under it
 * and Play's history caches are keyed by it (P260-FX-10), so the next account on
 * this browser never sees the last one's conversations.
 */
export function useChatScope(): string | null {
  const workspaceCtx = useWorkspaceOptional()
  const { user, isLoaded } = useUser()
  // Wait for both scopes to settle — hydrating under a provisional key would
  // write the session where later reads never look. A workspace already known
  // keeps its key while it refetches (saving a key from the trial banner
  // refreshes it), so the conversation on screen never blinks out; a different
  // workspace still rehydrates under its own key once it arrives.
  const ready = isLoaded && (workspaceCtx === null || !workspaceCtx.isLoading || workspaceCtx.workspace != null)
  return ready ? chatSessionKey(workspaceCtx?.workspace?.id ?? null, user?.id ?? null) : null
}

export function useChatSessionHydration(): { key: string | null; hydrated: boolean } {
  const key = useChatScope()
  const hydrate = useChatSessionStore((s) => s.hydrate)
  const reload = useChatSessionStore((s) => s.reload)
  const touchChat = useChatSessionStore((s) => s.touchChat)

  useEffect(() => {
    if (key) void hydrate(key)
  }, [key, hydrate])

  // Another browser tab changed the session — follow it.
  useEffect(() => {
    if (!key || typeof window === 'undefined') return
    const onStorage = (event: StorageEvent) => {
      if (event.key === key) reload()
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [key, reload])

  // PRD-205 S7 lane: a message landed in a chat — an open background tab shows a dot.
  useEffect(() => {
    if (typeof window === 'undefined') return
    const onChatChanged = (event: Event) => {
      const detail = (event as CustomEvent).detail as { chat_id?: string } | undefined
      if (detail?.chat_id) touchChat(detail.chat_id)
    }
    window.addEventListener('automatos:chat-changed', onChatChanged)
    return () => window.removeEventListener('automatos:chat-changed', onChatChanged)
  }, [touchChat])

  const hydrated = useChatSessionStore((s) => s.hydrated && s.key === key)
  return { key, hydrated }
}
