/**
 * Conversation session store (PRD-237).
 *
 * The one place that knows which conversation is on screen and which are open
 * as tabs — read by the chat page, the studio shell and the floating widget.
 * State is the pure model in `lib/chat/chat-session`; this store adds
 * persistence:
 *
 *   local edition → the browser only (one operator, one machine);
 *   hosted (saas)  → the browser as cache + the server copy
 *                    (`/api/chat/session`) so tabs follow the user across devices.
 *
 * Owner decision D1 (2026-09-07): the two editions persist differently and the
 * seam is `isSaaS` — nothing else in the store branches on edition.
 */
import { create } from 'zustand'
import { isSaaS } from '@/lib/auth-edition'
import {
  EMPTY_CHAT_SESSION,
  loadChatSession,
  mergeServerSession,
  saveChatSession,
  toServerSession,
  withChatIdAssigned,
  withChatOpened,
  withChatTouched,
  withDraft,
  withDraftClosed,
  withTabClosed,
  withThreadRead,
  withTitles,
  type ChatSession,
} from '@/lib/chat/chat-session'

/** Coalesce rapid tab changes into one server write. */
export const SERVER_SAVE_DEBOUNCE_MS = 400

type Updater = (session: ChatSession, now: number) => ChatSession

export interface ChatSessionStore {
  /** Storage key (workspace + user) this store is bound to; null until hydrated. */
  key: string | null
  /** True once the session for `key` is usable (after the server merge in saas). */
  hydrated: boolean
  session: ChatSession
  hydrate: (key: string) => Promise<void>
  /** Re-read the browser copy (another browser tab changed it). */
  reload: () => void
  openChat: (chatId: string) => void
  newDraft: () => void
  closeTab: (chatId: string) => void
  closeDraft: () => void
  chatIdAssigned: (chatId: string) => void
  markRead: (chatId: string | null) => void
  touchChat: (chatId: string) => void
  setTitles: (titles: Record<string, string>) => void
}

let serverTimer: ReturnType<typeof setTimeout> | null = null

function scheduleServerSave(session: ChatSession): void {
  if (!isSaaS) return
  if (serverTimer) clearTimeout(serverTimer)
  serverTimer = setTimeout(() => {
    serverTimer = null
    void import('@/lib/chat/api')
      .then(({ putChatSession }) => putChatSession(toServerSession(session)))
      .catch(() => {
        // The browser copy is already saved; the server catches up on the next change.
      })
  }, SERVER_SAVE_DEBOUNCE_MS)
}

async function loadServerCopy(key: string, get: () => ChatSessionStore, set: (s: Partial<ChatSessionStore>) => void) {
  try {
    const { getChatSession } = await import('@/lib/chat/api')
    const server = await getChatSession()
    if (get().key !== key) return // workspace or user changed while we waited
    const merged = mergeServerSession(get().session, server, Date.now())
    if (merged !== get().session) {
      set({ session: merged })
      saveChatSession(key, merged)
    }
  } catch {
    // Server copy unavailable — the browser copy still works.
  } finally {
    if (get().key === key) set({ hydrated: true })
  }
}

export const useChatSessionStore = create<ChatSessionStore>((set, get) => {
  const apply = (updater: Updater, { server = true }: { server?: boolean } = {}) => {
    const { key, session } = get()
    const next = updater(session, Date.now())
    if (next === session) return
    set({ session: next })
    if (key) saveChatSession(key, next)
    if (server) scheduleServerSave(next)
  }

  return {
    key: null,
    hydrated: false,
    session: EMPTY_CHAT_SESSION,

    hydrate: async (key) => {
      if (get().key === key && get().hydrated) return
      set({ key, session: loadChatSession(key), hydrated: !isSaaS })
      if (isSaaS) await loadServerCopy(key, get, set)
    },

    reload: () => {
      const { key } = get()
      if (key) set({ session: loadChatSession(key) })
    },

    openChat: (chatId) => apply((s, now) => withChatOpened(s, chatId, now)),
    newDraft: () => apply((s, now) => withDraft(s, now)),
    closeTab: (chatId) => apply((s, now) => withTabClosed(s, chatId, now)),
    closeDraft: () => apply((s, now) => withDraftClosed(s, now)),
    chatIdAssigned: (chatId) => apply((s, now) => withChatIdAssigned(s, chatId, now)),
    markRead: (chatId) => apply((s, now) => withThreadRead(s, chatId, now)),
    // Unread flags and titles are device-local — never a server write.
    touchChat: (chatId) => apply((s) => withChatTouched(s, chatId), { server: false }),
    setTitles: (titles) => apply((s) => withTitles(s, titles), { server: false }),
  }
})
