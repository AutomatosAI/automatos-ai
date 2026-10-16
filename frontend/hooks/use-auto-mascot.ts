'use client'

import { type Dispatch, useCallback, useEffect, useMemo, useReducer, useRef } from 'react'

import { IDLE_MS, useIdleTimeout } from '@/hooks/use-idle-timeout'
import {
  type ChatStatus,
  mascotEventsFromData,
  mascotEventsFromReply,
  mascotEventsFromStatus,
  NO_REPLY,
  type ReplyProgress,
} from '@/lib/mascot/events'
import { INITIAL_MASCOT, mascotReducer } from '@/lib/mascot/reducer'
import { claimFirstOpen, type SessionSource } from '@/lib/mascot/session'
import type { MascotEvent, MascotName, MascotState } from '@/lib/mascot/types'

/**
 * PRD-260 D5, D6 — Auto driven by the core chat. Play's reducer
 * (lib/mascot/reducer.ts) is fed from useChatWithReceipts' data callback, its
 * status and its newest reply (the model's reasoning, the words starting, a cap)
 * through lib/mascot/events.ts, from the page's own events through dispatch,
 * and from the idle nap. The screen hands `mascot` and `onClipEnded` to
 * <AutoMascot>.
 */

export type BackendStatus = 'up' | 'down'

export interface AutoMascotOptions {
  /** useChat's status: into streaming starts a turn, back to idle finishes it, into error fails it. */
  readonly status?: ChatStatus
  /** useChat's errorCode: set when the last turn was blocked ('trial_exhausted'). */
  readonly errorCode?: string | null
  /**
   * The newest reply: its reasoning thinks, its words starting end thinking, a
   * cap plays oops. Play's chat: replyProgress(messages).
   */
  readonly reply?: ReplyProgress
  /** The connection's state: down puts Auto to sleep until it is up again. Unknown: leave undefined. */
  readonly backend?: BackendStatus
  /** How long without input before Auto naps. */
  readonly idleMs?: number
  /** Where the first open is remembered (tests pass their own). */
  readonly session?: SessionSource
}

export interface AutoMascotDriver {
  readonly state: MascotState
  /** The state to show: AutoMascot's `state`. */
  readonly mascot: MascotName
  /** Pass to useChatWithReceipts' onData, beside the screen's own handling. */
  readonly onData: (part: unknown) => void
  /** The page's events: input-focus, input-blur, send, deliverable-added, not-found, … */
  readonly dispatch: Dispatch<MascotEvent>
  /** hello, once per browser session (lib/mascot/session.ts). True when it plays. */
  readonly greet: () => boolean
  /** AutoMascot's onClipEnded: a one-shot's clip ended. */
  readonly onClipEnded: (state: MascotName) => void
}

/** The chat's status as events: the first render counts as coming from idle, so an awaited reply thinks. */
function useStatusEvents(status: ChatStatus, errorCode: string | null, dispatch: Dispatch<MascotEvent>): void {
  const previous = useRef<ChatStatus>('idle')
  useEffect(() => {
    const events = mascotEventsFromStatus(previous.current, status, errorCode)
    previous.current = status
    events.forEach(dispatch)
  }, [status, errorCode, dispatch])
}

/**
 * The newest reply's reasoning, its words starting and a cap ending its turn,
 * after the status events of the same render (so a send thinks first, and a cap
 * that lands with the stream's end plays oops after the finish).
 */
function useReplyEvents(reply: ReplyProgress, status: ChatStatus, dispatch: Dispatch<MascotEvent>): void {
  const previous = useRef<ReplyProgress>(NO_REPLY)
  const { id, hasText, toolRunning, reasoningChars, limitReached } = reply
  useEffect(() => {
    const next: ReplyProgress = { id, hasText, toolRunning, reasoningChars, limitReached }
    const events = mascotEventsFromReply(previous.current, next, status)
    previous.current = next
    events.forEach(dispatch)
  }, [id, hasText, toolRunning, reasoningChars, limitReached, status, dispatch])
}

function useBackendEvents(backend: BackendStatus | undefined, dispatch: Dispatch<MascotEvent>): void {
  useEffect(() => {
    if (backend === 'down') dispatch({ type: 'backend-down' })
    if (backend === 'up') dispatch({ type: 'backend-up' })
  }, [backend, dispatch])
}

export function useAutoMascot(options: AutoMascotOptions = {}): AutoMascotDriver {
  const { status = 'idle', errorCode = null, reply = NO_REPLY, backend, idleMs = IDLE_MS, session } = options
  const [state, dispatch] = useReducer(mascotReducer, INITIAL_MASCOT)

  useStatusEvents(status, errorCode, dispatch)
  useReplyEvents(reply, status, dispatch)
  useBackendEvents(backend, dispatch)

  const nap = useCallback(() => dispatch({ type: 'idle-timeout' }), [])
  const wake = useCallback(() => dispatch({ type: 'wake' }), [])
  useIdleTimeout(nap, wake, idleMs)

  const onData = useCallback((part: unknown) => mascotEventsFromData(part).forEach(dispatch), [])
  const onClipEnded = useCallback((ended: MascotName) => dispatch({ type: 'clip-ended', state: ended }), [])
  const greet = useCallback((): boolean => {
    const first = claimFirstOpen(session)
    if (first) dispatch({ type: 'app-open' })
    return first
  }, [session])

  return useMemo(
    () => ({ state, mascot: state.mascot, onData, dispatch, greet, onClipEnded }),
    [state, onData, greet, onClipEnded],
  )
}
