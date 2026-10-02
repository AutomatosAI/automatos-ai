/**
 * PRD-252 R2 — the ticket the chat page is discussing, if any, and the
 * conversation the discussion is.
 *
 * Set by the discussion bar from `/chat?ticket=<id>`; read by the chat's page
 * context, so every message of the discussion tells Auto which ticket it is
 * about. `chatId` is null while the discussion is the draft it opened, then the
 * conversation that draft became (review of #861: the ticket must not reach
 * another conversation).
 */
import { create } from 'zustand'
import type { Discussion } from '@/lib/discussion'

interface DiscussionStore {
  discussion: Discussion | null
  chatId: string | null
  start: (discussion: Discussion) => void
  /** The discussion's draft became conversation `chatId`. */
  adopt: (chatId: string) => void
  end: () => void
}

export const useDiscussionStore = create<DiscussionStore>((set) => ({
  discussion: null,
  chatId: null,
  start: (discussion) => set({ discussion, chatId: null }),
  adopt: (chatId) => set({ chatId }),
  end: () => set({ discussion: null, chatId: null }),
}))
