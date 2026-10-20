'use client'

import { type Dispatch, useCallback, useMemo, useRef } from 'react'

import {
  type HostChatController,
  type HostChatControllerOptions,
  type HostChatConversation,
  type HostChatTurnEvent,
  useChatController,
} from '@automatos/theme-sdk'

import { FAILED_TURN } from '../lib/mascot/events'
import type { MascotEvent } from '../lib/mascot/types'

/**
 * PRD-260 §4, PRD-261 A-11a (M28), A-14a — Play's chat on one conversation:
 * the SDK's (`useChatController(conversation)`, lib/theme-sdk/chat.ts, on the
 * chat controller: the stream, Stop keeping the reply being written, Try
 * again, the Deliverables a turn made, a Mission suggestion, a reply that
 * failed in words, the approval cards it raised, held by conversation in
 * lib/chat-controller/approvals.ts), its parts and events played by Auto: a
 * Deliverable is new (deliverable-added), a card waits and none does any more
 * (approval-waiting, approval-settled), and a reply that failed in words is
 * oops even when the stream ended in the same render (error).
 *
 * Auto is driven by the chat's status, so his driver exists only after the
 * chat: `connect` hands it over. Each part reaches Auto before the turn reads
 * it, as the PRD-260 chat passed them on. Connect it in the render that makes
 * the driver: a card still waiting when the chat screen mounts again is
 * announced in the screen's first effects, and an event before `connect` goes
 * nowhere.
 */

/** What Play's chat drives with the turn: Auto (useAutoMascot's onData and dispatch). */
export interface PlayTurnDriver {
  /** Each data part of the stream, before the turn reads it. */
  readonly onData: (part: unknown) => void
  /** Auto's events. */
  readonly dispatch: Dispatch<MascotEvent>
}

export interface PlayTurn {
  /** The conversation's chat as the SDK's view model. */
  readonly chat: HostChatController
  /** Auto, once he exists; the parts and events from then on go to him. */
  readonly connect: (driver: PlayTurnDriver) => void
}

/** A turn's event as Auto plays it. */
export function mascotEventOf(event: HostChatTurnEvent): MascotEvent {
  return event.type === 'turn-failed' ? { type: 'error', message: FAILED_TURN, code: event.code } : event
}

/** The conversation's chat (one handed out by usePlayConversation); key the caller by `conversation.key`. */
export function usePlayTurn(conversation: HostChatConversation): PlayTurn {
  const driver = useRef<PlayTurnDriver | null>(null)
  const options = useMemo<HostChatControllerOptions>(
    () => ({
      onData: (part: unknown) => driver.current?.onData(part),
      onEvent: (event: HostChatTurnEvent) => driver.current?.dispatch(mascotEventOf(event)),
    }),
    [],
  )
  const chat = useChatController(conversation, options)
  const connect = useCallback((next: PlayTurnDriver) => {
    driver.current = next
  }, [])
  return useMemo(() => ({ chat, connect }), [chat, connect])
}
