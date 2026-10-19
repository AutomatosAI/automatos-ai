'use client'

/**
 * PRD-260 T-005c (P260-FX-08, M46), PRD-261 A-11a (M28) — the approval cards
 * a theme's chat filed in this page session, held outside the chat screen, and
 * the turn's filing of them. The screen is keyed by the conversation, so
 * leaving the chat or switching conversation unmounts it, yet a grant or a
 * Mission still waits on the server: the card has to be there when the
 * conversation is on screen again, as Studio's widgets stay in the workspace
 * store. In memory only, like Studio's widgets (workspace-store.ts doesn't
 * persist them): a reload clears it. Each card knows its conversation ('' for
 * a new chat until the backend names it, then re-keyed to the name), and every
 * change is lib/chat/approvals.ts' pure fileApprovals or withStatus, applied to
 * that conversation's cards.
 */
import { type MutableRefObject, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { create } from 'zustand'

import {
  type ApprovalCard,
  type ApprovalStatus,
  approvalsInPart,
  type FiledApproval,
  fileApprovals,
  isWaiting,
  looseApprovals,
  withStatus,
} from '@/lib/chat/approvals'
import type { ChatMessage } from '@/types'

import type { ChatApprovals, ChatTurnEvent } from './view-model'

/** A filed card and the conversation it belongs to. */
export type ChatApproval = FiledApproval & { readonly chatId: string }

interface ChatApprovalsStore {
  readonly filed: readonly ChatApproval[]
  /** File the cards a part raised under a reply of the conversation; true when any of them was new. */
  readonly file: (chatId: string, raised: readonly ApprovalCard[], replyId: string, filedAt: Date) => boolean
  /** A card of the conversation was answered or closed. */
  readonly setStatus: (chatId: string, key: string, status: ApprovalStatus) => void
  /** The backend named the new chat: its cards (filed under `from`, '') go with the name. */
  readonly rename: (from: string, to: string) => void
}

const NO_APPROVALS: readonly FiledApproval[] = []

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

export const useChatApprovalsStore = create<ChatApprovalsStore>((set, get) => ({
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

type Emit = (event: ChatTurnEvent) => void

/** The theme hears while a card waits (one still waiting when the chat screen mounts again, too) and once none does. */
function useWaitingEvents(waiting: boolean, emit: Emit): void {
  // null until the first run: a card already waiting then was filed before this screen mounted.
  const was = useRef<boolean | null>(null)
  useEffect(() => {
    if (was.current === null && waiting) emit({ type: 'approval-waiting' })
    if (was.current === true && !waiting) emit({ type: 'approval-settled' })
    was.current = waiting
  }, [waiting, emit])
}

/** The open cards whose reply left the conversation (the same empty list while there are none). */
function useLooseApprovals(filed: readonly FiledApproval[], messages: readonly ChatMessage[]): readonly FiledApproval[] {
  return useMemo(() => {
    const loose = looseApprovals(filed, new Set(messages.map((message) => message.id)))
    return loose.length > 0 ? loose : NO_APPROVALS
  }, [filed, messages])
}

/**
 * The conversation's cards in the store, and the conversation they are filed
 * under. A new chat's are filed under '' until the backend names it; then they
 * go with the name, at once for the parts still to come in the same read of
 * the stream (the ref) and for what the screen shows (the state). The screen
 * is keyed by the conversation, so the name is the only change.
 */
function useFiledApprovals(initialChatId: string) {
  const [chatId, setChatId] = useState(initialChatId)
  const under = useRef(initialChatId)
  const all = useChatApprovalsStore((state) => state.filed)
  const filed = useMemo(() => cardsOfChat(all, chatId), [all, chatId])
  const named = useCallback((name: string) => {
    useChatApprovalsStore.getState().rename(under.current, name)
    under.current = name
    setChatId(name)
  }, [])
  return { filed, under, named }
}

/** A card's answer or close, on the conversation the cards are filed under now. */
function useCardStatus(under: MutableRefObject<string>) {
  const setStatus = useChatApprovalsStore((state) => state.setStatus)
  const answered = useCallback((key: string) => setStatus(under.current, key, 'answered'), [setStatus, under])
  const closed = useCallback((key: string) => setStatus(under.current, key, 'closed'), [setStatus, under])
  return { answered, closed }
}

/**
 * The approvals a part raises, filed under the reply being written (once per
 * grant or Mission), the theme told when one waits; the cards a reply shows,
 * the loose ones, and the owner's answer and close.
 */
export function useLiveApprovals(
  chatId: string,
  messages: readonly ChatMessage[],
  replyId: MutableRefObject<string | undefined>,
  emit: Emit,
) {
  const { filed, under, named } = useFiledApprovals(chatId)
  const onPart = useCallback(
    (part: unknown) => {
      const reply = replyId.current
      const raised = approvalsInPart(part)
      if (reply === undefined || raised.length === 0) return
      const fresh = useChatApprovalsStore.getState().file(under.current, raised, reply, new Date())
      if (fresh) emit({ type: 'approval-waiting' })
    },
    [emit, replyId, under],
  )
  const { answered, closed } = useCardStatus(under)
  const of = useCallback(
    (id: string) => filed.filter((card) => card.replyId === id && card.status !== 'closed'),
    [filed],
  )
  const loose = useLooseApprovals(filed, messages)
  const waiting = isWaiting(filed)
  useWaitingEvents(waiting, emit)
  const approvals = useMemo<ChatApprovals>(
    () => ({ of, loose, answered, closed, waiting }),
    [of, loose, answered, closed, waiting],
  )
  return { onPart, approvals, named }
}
