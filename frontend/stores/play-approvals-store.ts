/**
 * PRD-260 T-005c (P260-FX-08, M46) — the approval cards Play's chat filed in
 * this page session, held outside the chat screen. The screen is keyed by the
 * conversation, so leaving the chat or switching conversation unmounts it, yet
 * a grant or a Mission still waits on the server: the card has to be there
 * when the conversation is on screen again, as Studio's widgets stay in the
 * workspace store. In memory only, like Studio's widgets (workspace-store.ts
 * doesn't persist them): a reload clears it. Each card knows its conversation
 * ('' for a new chat until the backend names it, then re-keyed to the name),
 * and every change is approvals.ts' pure fileApprovals or withStatus, applied
 * to that conversation's cards.
 */
import { create } from 'zustand'

import {
  type ApprovalCard,
  type ApprovalStatus,
  type FiledApproval,
  fileApprovals,
  withStatus,
} from '@/components/play/chat/approvals'

/** A filed card and the conversation it belongs to. */
export type ChatApproval = FiledApproval & { readonly chatId: string }

interface PlayApprovalsStore {
  readonly filed: readonly ChatApproval[]
  /** File the cards a part raised under a reply of the conversation; true when any of them was new. */
  readonly file: (chatId: string, raised: readonly ApprovalCard[], replyId: string, filedAt: Date) => boolean
  /** A card of the conversation was answered or closed. */
  readonly setStatus: (chatId: string, key: string, status: ApprovalStatus) => void
  /** The backend named the new chat: its cards (filed under `from`, '') go with the name. */
  readonly rename: (from: string, to: string) => void
}

/** One conversation's cards, in the order they were filed. */
export function cardsOfChat(filed: readonly ChatApproval[], chatId: string): readonly ChatApproval[] {
  return filed.filter((card) => card.chatId === chatId)
}

/** `filed` with one conversation's cards changed by a pure change (the same list when nothing changed). Never mutates. */
function inChat(
  filed: readonly ChatApproval[],
  chatId: string,
  change: (cards: readonly FiledApproval[]) => readonly FiledApproval[],
): readonly ChatApproval[] {
  const cards = cardsOfChat(filed, chatId)
  const changed = change(cards)
  if (changed === cards) return filed
  const others = filed.filter((card) => card.chatId !== chatId)
  return [...others, ...changed.map((card): ChatApproval => ({ ...card, chatId }))]
}

/** `filed` with the cards of `from` moved to `to` (the same list when there are none). Never mutates. */
function renamed(filed: readonly ChatApproval[], from: string, to: string): readonly ChatApproval[] {
  if (from === to || !filed.some((card) => card.chatId === from)) return filed
  return filed.map((card) => (card.chatId === from ? { ...card, chatId: to } : card))
}

export const usePlayApprovalsStore = create<PlayApprovalsStore>((set, get) => ({
  filed: [],
  file: (chatId, raised, replyId, filedAt) => {
    const filed = get().filed
    const next = inChat(filed, chatId, (cards) => fileApprovals(cards, raised, replyId, filedAt))
    if (next === filed) return false
    set({ filed: next })
    return true
  },
  setStatus: (chatId, key, status) =>
    set((state) => {
      const next = inChat(state.filed, chatId, (cards) => withStatus(cards, key, status))
      return next === state.filed ? state : { filed: next }
    }),
  rename: (from, to) =>
    set((state) => {
      const next = renamed(state.filed, from, to)
      return next === state.filed ? state : { filed: next }
    }),
}))
