/**
 * PRD-252 R2 — the ticket the chat page is discussing, if any.
 *
 * Set by the discussion bar from `/chat?ticket=<id>`; read by the chat's page
 * context, so every message of the discussion tells Auto which ticket it is about.
 */
import { create } from 'zustand'
import type { Discussion } from '@/lib/discussion'

interface DiscussionStore {
  discussion: Discussion | null
  start: (discussion: Discussion) => void
  end: () => void
}

export const useDiscussionStore = create<DiscussionStore>((set) => ({
  discussion: null,
  start: (discussion) => set({ discussion }),
  end: () => set({ discussion: null }),
}))
