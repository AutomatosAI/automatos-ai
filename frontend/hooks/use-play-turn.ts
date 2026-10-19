'use client'

import { type Dispatch, useCallback } from 'react'

import { useChatTurn } from '@/lib/chat-controller/turn'
import type { ChatTurn, ChatTurnEvent } from '@/lib/chat-controller/view-model'
import { FAILED_TURN } from '@/lib/mascot/events'
import type { MascotEvent } from '@/lib/mascot/types'
import type { ChatMessage } from '@/types'

/**
 * PRD-260 §4, PRD-261 A-11a (M28) — what Play's chat keeps about a turn: the
 * chat controller's turn (lib/chat-controller/turn.ts: the Deliverables it
 * made, a Mission suggestion, a reply that failed in words, the approval cards
 * it raised, held by conversation in lib/chat-controller/approvals.ts), its
 * events played by Auto: a Deliverable is new (deliverable-added), a card
 * waits and none does any more (approval-waiting, approval-settled), and a
 * reply that failed in words is oops even when the stream ended in the same
 * render (error).
 */

export interface PlayTurnOptions {
  /** The conversation on screen ('' for a new chat), whose approval cards show. */
  readonly chatId: string
  /** The conversation as Play shows it (useChat's, with the replies you stopped kept). */
  readonly messages: readonly ChatMessage[]
  /** The reply is still being written. */
  readonly streaming: boolean
  /** Auto's events (useAutoMascot's dispatch). */
  readonly dispatch: Dispatch<MascotEvent>
}

/** A turn's event as Auto plays it. */
export function mascotEventOf(event: ChatTurnEvent): MascotEvent {
  return event.type === 'turn-failed' ? { type: 'error', message: FAILED_TURN, code: event.code } : event
}

export function usePlayTurn({ chatId, messages, streaming, dispatch }: PlayTurnOptions): ChatTurn {
  const onEvent = useCallback((event: ChatTurnEvent) => dispatch(mascotEventOf(event)), [dispatch])
  return useChatTurn({ chatId, messages, streaming, onEvent })
}
