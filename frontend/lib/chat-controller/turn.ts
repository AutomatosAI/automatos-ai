'use client'

import { type MutableRefObject, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { type DeliverableRef, deliverablesInPart, mergeDeliverables } from '@/lib/chat/deliverables'
import { latestReply } from '@/lib/chat/message-text'
import { type MissionSuggestion, suggestionFromData } from '@/lib/chat/suggestion'
import type { ChatMessage } from '@/types'

import { useLiveApprovals } from './approvals'
import type { ChatTurn, ChatTurnEvent, ChatTurnOptions } from './view-model'

/**
 * PRD-260 §4, PRD-261 A-11a (M28) — what a theme's chat keeps about a turn
 * beyond the core's messages, from useChatWithReceipts' data parts: the
 * Deliverables the turn made (the document tool's generated_document and the
 * receipts that link one; useChat keeps neither on the reply), each announced
 * once (deliverable-added); the Mission Auto suggests, cleared when you send
 * again (as chat.tsx does); a reply that failed in words (an error part),
 * announced (turn-failed) even when the stream ended in the same render; and
 * the approvals a reply raised (T-005c: tool_approval and mission_approval,
 * the core's widgets), each waiting for the owner until answered or closed
 * (approval-waiting, approval-settled). A card outlives its reply: when
 * useChat drops the reply (a request failing mid-turn) or Try again replaces
 * it, the card is loose and stays on screen, so it can still be answered or
 * closed; a reply you stopped stays in the conversation shown
 * (use-stopped-reply.ts), and its cards stay in it. And it outlives the chat
 * screen (M46): the cards live in the approvals store by conversation
 * (approvals.ts), so leaving the chat or switching conversation and coming
 * back shows them again (loose when the server's copy of the reply has its own
 * id), as Studio's widgets stay in the workspace store; a reload clears them,
 * as it clears Studio's.
 */

/** The live Deliverables by the id of the reply that made them. */
type MadeBy = Readonly<Record<string, readonly DeliverableRef[]>>
type Emit = (event: ChatTurnEvent) => void

const NONE: readonly DeliverableRef[] = []

/** One stable emitter for the caller's latest onEvent (none: the events go nowhere). */
export function useEmitter(onEvent: ((event: ChatTurnEvent) => void) | undefined): Emit {
  const latest = useRef(onEvent)
  // Read at the time of the event, so an effect of this same render already sees the new one.
  latest.current = onEvent
  return useCallback((event: ChatTurnEvent) => latest.current?.(event), [])
}

/** The id of the newest reply when it failed in words and is no longer being written; else null. */
export function failedReplyId(messages: readonly ChatMessage[], streaming: boolean): string | null {
  const reply = latestReply(messages)
  return !streaming && reply?.error ? reply.id : null
}

/** turn-failed once per failed reply; a failure already on screen when the chat opened stays quiet. */
function useFailureEvents(messages: readonly ChatMessage[], streaming: boolean, emit: Emit): void {
  const failedId = failedReplyId(messages, streaming)
  const announced = useRef<string | null>(failedId)
  const code = latestReply(messages)?.error?.code ?? null
  useEffect(() => {
    if (failedId === null || failedId === announced.current) return
    announced.current = failedId
    emit({ type: 'turn-failed', code })
  }, [failedId, code, emit])
}

/** The id of the reply being written, for the data parts that arrive while it is. */
function useReplyBeingWritten(messages: readonly ChatMessage[]): MutableRefObject<string | undefined> {
  const replyId = useRef<string | undefined>(undefined)
  useEffect(() => {
    replyId.current = latestReply(messages)?.id
  }, [messages])
  return replyId
}

/** The Deliverables a part announces, filed under the reply being written, each one new once. */
function useLiveDeliverables(replyId: MutableRefObject<string | undefined>, emit: Emit) {
  const [madeBy, setMadeBy] = useState<MadeBy>({})
  const seen = useRef<ReadonlySet<string>>(new Set())

  const onPart = useCallback(
    (part: unknown) => {
      const reply = replyId.current
      const fresh = deliverablesInPart(part).filter((ref) => !seen.current.has(ref.id))
      if (reply === undefined || fresh.length === 0) return
      seen.current = new Set([...Array.from(seen.current), ...fresh.map((ref) => ref.id)])
      setMadeBy((prev) => ({ ...prev, [reply]: mergeDeliverables(prev[reply] ?? NONE, fresh) }))
      emit({ type: 'deliverable-added' })
    },
    [emit, replyId],
  )
  const deliverablesOf = useCallback((id: string) => madeBy[id] ?? NONE, [madeBy])
  return { onPart, deliverablesOf }
}

export function useChatTurn({ chatId, messages, streaming, onEvent }: ChatTurnOptions): ChatTurn {
  const emit = useEmitter(onEvent)
  const [suggestion, setSuggestion] = useState<MissionSuggestion | null>(null)
  const replyId = useReplyBeingWritten(messages)
  const { onPart, deliverablesOf } = useLiveDeliverables(replyId, emit)
  const { onPart: onApprovalPart, approvals, named } = useLiveApprovals(chatId, messages, replyId, emit)
  useFailureEvents(messages, streaming, emit)

  const onData = useCallback(
    (part: unknown) => {
      const suggested = suggestionFromData(part)
      if (suggested) setSuggestion(suggested)
      onPart(part)
      onApprovalPart(part)
    },
    [onPart, onApprovalPart],
  )
  const onSend = useCallback(() => setSuggestion(null), [])

  return useMemo(
    () => ({ onData, onSend, onChatNamed: named, suggestion, deliverablesOf, approvals }),
    [onData, onSend, named, suggestion, deliverablesOf, approvals],
  )
}
