'use client'

import { type Dispatch, type MutableRefObject, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  approvalsInPart,
  type FiledApproval,
  fileApprovals,
  isWaiting,
  looseApprovals,
  withStatus,
} from '@/components/play/chat/approvals'
import { type DeliverableRef, deliverablesInPart, mergeDeliverables } from '@/components/play/chat/deliverables'
import { latestReply } from '@/components/play/chat/message-text'
import { type MissionSuggestion, suggestionFromData } from '@/components/play/chat/suggestion'
import { FAILED_TURN } from '@/lib/mascot/events'
import type { MascotEvent } from '@/lib/mascot/types'
import type { ChatMessage } from '@/types'

/**
 * PRD-260 §4 — what Play's chat keeps about a turn beyond the core's messages,
 * from useChatWithReceipts' data parts: the Deliverables the turn made (the
 * document tool's generated_document and the receipts that link one; useChat
 * keeps neither on the reply), each announced to Auto once
 * (deliverable-added); the Mission Auto suggests, cleared when you send again
 * (as chat.tsx does); a reply that failed in words (an error part), which
 * makes Auto play oops even when the stream ended in the same render; and the
 * approvals a reply raised (T-005c: tool_approval and mission_approval, the
 * core's widgets), each waiting for the owner until answered or closed, with
 * Auto listening meanwhile (approval-waiting, approval-settled). A card
 * outlives its reply: when useChat drops the reply (Stop, a request failing
 * mid-turn) or Try again replaces it, the card is loose and stays on screen,
 * so it can still be answered or closed.
 */

/** The live Deliverables by the id of the reply that made them. */
type MadeBy = Readonly<Record<string, readonly DeliverableRef[]>>

const NONE: readonly DeliverableRef[] = []
const NO_APPROVALS: readonly FiledApproval[] = []

/** The approvals a reply raised, and what the owner does with them on their cards. */
export interface PlayApprovals {
  /** The cards a reply raised and the owner hasn't closed. */
  readonly of: (replyId: string) => readonly FiledApproval[]
  /** The open cards whose reply is no longer in the conversation: shown after it. */
  readonly loose: readonly FiledApproval[]
  /** A card's approve or deny went through. */
  readonly answered: (key: string) => void
  /** A card closed (its X, or the Mission card itself after a decision). */
  readonly closed: (key: string) => void
  /** A card still waits for the owner. */
  readonly waiting: boolean
}

export interface PlayTurnOptions {
  readonly messages: readonly ChatMessage[]
  /** The reply is still being written. */
  readonly streaming: boolean
  /** Auto's events (useAutoMascot's dispatch). */
  readonly dispatch: Dispatch<MascotEvent>
}

export interface PlayTurn {
  /** Pass to useChatWithReceipts' onData beside Auto's own. */
  readonly onData: (part: unknown) => void
  /** A new message or Try again: the turn's suggestion goes, as in chat.tsx. */
  readonly onSend: () => void
  readonly suggestion: MissionSuggestion | null
  /** The Deliverables a reply made live this session (a stored reply's come from its receipts). */
  readonly deliverablesOf: (replyId: string) => readonly DeliverableRef[]
  /** The approvals the turn's replies raised live (chat.tsx's widgets are live-only too). */
  readonly approvals: PlayApprovals
}

/** The id of the newest reply when it failed in words and is no longer being written; else null. */
function failedReplyId(messages: readonly ChatMessage[], streaming: boolean): string | null {
  const reply = latestReply(messages)
  return !streaming && reply?.error ? reply.id : null
}

/** Oops once per failed reply; a failure already on screen when the chat opened stays quiet. */
function useFailureEvents(messages: readonly ChatMessage[], streaming: boolean, dispatch: Dispatch<MascotEvent>): void {
  const failedId = failedReplyId(messages, streaming)
  const announced = useRef<string | null>(failedId)
  const code = latestReply(messages)?.error?.code ?? null
  useEffect(() => {
    if (failedId === null || failedId === announced.current) return
    announced.current = failedId
    dispatch({ type: 'error', message: FAILED_TURN, code })
  }, [failedId, code, dispatch])
}

/** The id of the reply being written, for the data parts that arrive while it is. */
function useReplyBeingWritten(messages: readonly ChatMessage[]): MutableRefObject<string | undefined> {
  const replyId = useRef<string | undefined>(undefined)
  useEffect(() => {
    replyId.current = latestReply(messages)?.id
  }, [messages])
  return replyId
}

/** The Deliverables a part announces, filed under the reply being written, each one new to Auto once. */
function useLiveDeliverables(replyId: MutableRefObject<string | undefined>, dispatch: Dispatch<MascotEvent>) {
  const [madeBy, setMadeBy] = useState<MadeBy>({})
  const seen = useRef<ReadonlySet<string>>(new Set())

  const onPart = useCallback(
    (part: unknown) => {
      const reply = replyId.current
      const fresh = deliverablesInPart(part).filter((ref) => !seen.current.has(ref.id))
      if (reply === undefined || fresh.length === 0) return
      seen.current = new Set([...Array.from(seen.current), ...fresh.map((ref) => ref.id)])
      setMadeBy((prev) => ({ ...prev, [reply]: mergeDeliverables(prev[reply] ?? NONE, fresh) }))
      dispatch({ type: 'deliverable-added' })
    },
    [dispatch, replyId],
  )
  const deliverablesOf = useCallback((id: string) => madeBy[id] ?? NONE, [madeBy])
  return { onPart, deliverablesOf }
}

/** Auto stops listening for the owner once no card waits any more. */
function useSettledEvent(waiting: boolean, dispatch: Dispatch<MascotEvent>): void {
  const was = useRef(false)
  useEffect(() => {
    if (was.current && !waiting) dispatch({ type: 'approval-settled' })
    was.current = waiting
  }, [waiting, dispatch])
}

/** The open cards whose reply left the conversation (the same empty list while there are none). */
function useLooseApprovals(filed: readonly FiledApproval[], messages: readonly ChatMessage[]): readonly FiledApproval[] {
  return useMemo(() => {
    const loose = looseApprovals(filed, new Set(messages.map((message) => message.id)))
    return loose.length > 0 ? loose : NO_APPROVALS
  }, [filed, messages])
}

/** The approvals a part raises, filed under the reply being written (once per grant or Mission); Auto listens. */
function useLiveApprovals(
  messages: readonly ChatMessage[],
  replyId: MutableRefObject<string | undefined>,
  dispatch: Dispatch<MascotEvent>,
) {
  const [filed, setFiled] = useState<readonly FiledApproval[]>(NO_APPROVALS)
  const seen = useRef<ReadonlySet<string>>(new Set())
  const onPart = useCallback(
    (part: unknown) => {
      const reply = replyId.current
      const raised = approvalsInPart(part).filter((card) => !seen.current.has(card.key))
      if (reply === undefined || raised.length === 0) return
      seen.current = new Set([...Array.from(seen.current), ...raised.map((card) => card.key)])
      const filedAt = new Date()
      setFiled((prev) => fileApprovals(prev, raised, reply, filedAt))
      dispatch({ type: 'approval-waiting' })
    },
    [dispatch, replyId],
  )
  const answered = useCallback((key: string) => setFiled((prev) => withStatus(prev, key, 'answered')), [])
  const closed = useCallback((key: string) => setFiled((prev) => withStatus(prev, key, 'closed')), [])
  const of = useCallback(
    (id: string) => filed.filter((card) => card.replyId === id && card.status !== 'closed'),
    [filed],
  )
  const loose = useLooseApprovals(filed, messages)
  const waiting = isWaiting(filed)
  useSettledEvent(waiting, dispatch)
  const approvals = useMemo<PlayApprovals>(
    () => ({ of, loose, answered, closed, waiting }),
    [of, loose, answered, closed, waiting],
  )
  return { onPart, approvals }
}

export function usePlayTurn({ messages, streaming, dispatch }: PlayTurnOptions): PlayTurn {
  const [suggestion, setSuggestion] = useState<MissionSuggestion | null>(null)
  const replyId = useReplyBeingWritten(messages)
  const { onPart, deliverablesOf } = useLiveDeliverables(replyId, dispatch)
  const { onPart: onApprovalPart, approvals } = useLiveApprovals(messages, replyId, dispatch)
  useFailureEvents(messages, streaming, dispatch)

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
    () => ({ onData, onSend, suggestion, deliverablesOf, approvals }),
    [onData, onSend, suggestion, deliverablesOf, approvals],
  )
}
