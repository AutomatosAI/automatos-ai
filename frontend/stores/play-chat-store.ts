/**
 * PRD-260 (P260-FX-03, M12) — what Play's New chat tells Play's chat that the
 * chat session store can't: New chat was pressed. The session model's New chat
 * changes nothing while a new chat is already the active one (withDraft), so a
 * new chat whose first send failed would stay on screen; usePlayConversation
 * (the SDK's useChatController(), lib/theme-sdk/chat.ts) follows this count
 * and shows a fresh one. Play's own store: it imports zustand and nothing of
 * the core (PRD-261 A-14a).
 */
import { create } from 'zustand'

interface PlayChatStore {
  /** How many times New chat was pressed in this page session. */
  freshDrafts: number
  freshDraft: () => void
}

export const usePlayChatStore = create<PlayChatStore>((set) => ({
  freshDrafts: 0,
  freshDraft: () => set((s) => ({ freshDrafts: s.freshDrafts + 1 })),
}))
