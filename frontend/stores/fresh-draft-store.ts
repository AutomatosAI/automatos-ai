/**
 * PRD-260 (P260-FX-03, M12) — what New chat tells a theme's chat that the
 * chat session store can't: New chat was pressed. The session model's New chat
 * changes nothing while a new chat is already the active one (withDraft), so a
 * new chat whose first send failed would stay on screen; the SDK's
 * useChatController() (lib/theme-sdk/chat.ts) follows this count and hands the
 * theme a fresh one. useStartChat (hooks/use-start-chat.ts) counts the presses.
 *
 * PRD-261 A-19a — the host's store, not Play's: it was stores/play-chat-store.ts
 * (usePlayChatStore) until the SDK's chat and New chat were the only code that
 * read it, so deleting Play's code can never take it with it.
 */
import { create } from 'zustand'

interface FreshDraftStore {
  /** How many times New chat was pressed in this page session. */
  freshDrafts: number
  freshDraft: () => void
}

export const useFreshDraftStore = create<FreshDraftStore>((set) => ({
  freshDrafts: 0,
  freshDraft: () => set((s) => ({ freshDrafts: s.freshDrafts + 1 })),
}))
