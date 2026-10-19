'use client'

import { useCallback, useEffect, useMemo, useRef } from 'react'

import { useRefreshHistoryOnTurnEnd } from '@/hooks/use-refresh-history-on-turn-end'
import { useStoppedReplies } from '@/hooks/use-stopped-reply'
import { type ChatWithReceipts, useChatWithReceipts } from '@/lib/chat/use-chat-with-receipts'
import { usePageContext } from '@/lib/page-context'
import { useChatSessionStore } from '@/stores/chat-session-store'

import { useChatTurn } from './turn'
import type {
  ChatActions,
  ChatControllerOptions,
  ChatControllerView,
  ChatStream,
  ChatStreamHandlers,
  ChatStreamOptions,
} from './view-model'

/**
 * PRD-261 A-11a (M28) — the chat controller: one conversation's stream, Stop,
 * Try again and a turn's details for any theme's chat screen, built from the
 * PRD-260 theme's chat screen wiring. Sending and streaming go through
 * useChatWithReceipts (the hook chat.tsx uses); a new chat's id goes into the
 * session store as chat.tsx does; the history lists are asked again when a
 * turn ends. Stop keeps the reply being written, as
 * stopped (useStoppedReplies, P260-FX-30), and calls the hook's stop(), which
 * also cancels the turn on the server: the conversation shown, and the one the
 * turn reads, is useChat's with the stopped replies kept. Try again is the
 * hook's reload(). Send and Try again clear the turn's suggestion first, as
 * chat.tsx does on a new message.
 *
 * Three layers, so a theme that drives something from the stream's status (a
 * mascot) can sit between them: useChatStream, then useChatTurn (turn.ts) with
 * the theme's events, then useStreamHandlers and useChatActions.
 * useChatController composes all of them for a theme that needs nothing in
 * between.
 */

const UNWIRED: ChatStreamHandlers = { onData: () => undefined, onChatNamed: () => undefined }

/** The stream for the conversation on screen; its parts and the new chat's name go to the handlers connected. */
export function useChatStream({ view, onChatIdAssigned }: ChatStreamOptions): ChatStream {
  const pageContext = usePageContext()
  // useChat takes its callbacks before the turn exists; the ref hands the parts and the new chat's name on.
  const handlers = useRef<ChatStreamHandlers>(UNWIRED)
  const chat = useChatWithReceipts({
    id: view.chatId ?? '',
    initialMessages: view.messages,
    initialAwaitingReply: view.awaitingReply,
    pageContext,
    onData: (part: unknown) => handlers.current.onData(part),
    onChatIdUpdate: (chatId: string) => {
      handlers.current.onChatNamed(chatId)
      onChatIdAssigned?.(chatId)
      // PRD-237: the session store is the channel to the sidebar and the widget, as in chat.tsx.
      useChatSessionStore.getState().chatIdAssigned(chatId)
    },
  })
  useRefreshHistoryOnTurnEnd(chat.status)
  const shown = useStoppedReplies(chat)
  const connect = useCallback((next: ChatStreamHandlers) => {
    handlers.current = next
  }, [])
  return { chat, messages: shown.messages, stop: shown.stop, isStopped: shown.isStopped, connect }
}

/** Connect the stream to the hooks that read its parts and the new chat's name, once they exist. */
export function useStreamHandlers(
  stream: Pick<ChatStream, 'connect'>,
  onData: (part: unknown) => void,
  onChatNamed: (chatId: string) => void,
): void {
  const { connect } = stream
  useEffect(() => {
    connect({ onData, onChatNamed })
  }, [connect, onData, onChatNamed])
}

/** Send and Try again (each clearing the turn's suggestion first, and refused while a turn runs), and Stop. */
export function useChatActions(chat: ChatWithReceipts, onSend: () => void, stop: () => void): ChatActions {
  const { isLoading, sendMessage, reload } = chat
  const send = useCallback(
    (text: string): boolean => {
      if (isLoading) return false
      onSend()
      void sendMessage(text)
      return true
    },
    [isLoading, sendMessage, onSend],
  )
  const retry = useCallback((): boolean => {
    if (isLoading) return false
    onSend()
    reload()
    return true
  }, [isLoading, reload, onSend])
  return useMemo(() => ({ send, retry, stop }), [send, retry, stop])
}

/** The theme's own reading of a part first (a mascot's), then the turn's. */
function usePartsInOrder(theirs: ((part: unknown) => void) | undefined, turn: (part: unknown) => void) {
  const latest = useRef(theirs)
  latest.current = theirs
  return useCallback(
    (part: unknown) => {
      latest.current?.(part)
      turn(part)
    },
    [turn],
  )
}

/** One conversation's chat, whole: the view model a theme's chat screen renders. */
export function useChatController(options: ChatControllerOptions): ChatControllerView {
  const stream = useChatStream(options)
  const { chat } = stream
  const streaming = chat.status === 'streaming'
  const turn = useChatTurn({
    chatId: options.view.chatId ?? '',
    messages: stream.messages,
    streaming,
    onEvent: options.onEvent,
  })
  const onData = usePartsInOrder(options.onData, turn.onData)
  useStreamHandlers(stream, onData, turn.onChatNamed)
  const actions = useChatActions(chat, turn.onSend, stream.stop)
  return {
    ...actions,
    messages: stream.messages,
    liveMessages: chat.messages,
    status: chat.status,
    errorCode: chat.errorCode,
    streaming,
    isStopped: stream.isStopped,
    suggestion: turn.suggestion,
    deliverablesOf: turn.deliverablesOf,
    approvals: turn.approvals,
  }
}
