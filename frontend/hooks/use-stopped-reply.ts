'use client'

import { useCallback, useMemo, useState } from 'react'

import type { ChatMessage } from '@/types'

/**
 * PRD-260 decision F4 (M33, P260-FX-30) — Stop keeps Auto's partial answer, as
 * the approved Play app does. useChat (lib/chat/hooks.ts) drops the reply it
 * was writing when Stop aborts the request (its catch removes the placeholder
 * on an AbortError), and Studio's chat still shows it that way; useChat is far
 * over the function-length limit, so it is not edited. Play keeps a copy
 * instead: Stop takes the reply being written as it is on screen, then calls
 * the hook's stop() (which also cancels the turn on the server), and the
 * conversation Play shows puts the copy back after the message it followed (the
 * owner's message it answered), where it stays through the turns that follow.
 * The reply reads as stopped: AutoReply says "You stopped this answer." under
 * it, and a tool call it left running reads "Stopped: …" (labels.ts
 * toolStatusOf: the reply is no longer being written). A copy whose message
 * left the conversation goes with it. The copies belong to the chat screen,
 * which is keyed by the conversation.
 */

/** A reply Stop ended, and the id of the message it followed. */
export interface StoppedReply {
  readonly reply: ChatMessage
  readonly after: string
}

/** What Stop reads of the chat (useChatWithReceipts). */
export interface StoppableChat {
  readonly messages: readonly ChatMessage[]
  readonly status: string
  /** A reply finishing detached on the server (PRD-237 S7): nothing is being written on this page. */
  readonly awaitingReply?: boolean
  readonly stop: () => void
}

export interface StoppedReplies {
  /** The conversation as Play shows it: useChat's messages with the stopped replies kept. */
  readonly messages: readonly ChatMessage[]
  /** Keep the reply being written, then the hook's stop(). */
  readonly stop: () => void
  /** Stop ended this reply. */
  readonly isStopped: (replyId: string) => boolean
}

const NONE_STOPPED: readonly StoppedReply[] = []

/** The reply being written on this page (the newest message, a reply, while the chat streams), with the message before it. */
export function replyBeingWritten(chat: Omit<StoppableChat, 'stop'>): StoppedReply | null {
  const { messages } = chat
  const reply = messages[messages.length - 1]
  const before = messages[messages.length - 2]
  if (chat.status !== 'streaming' || chat.awaitingReply || reply?.role !== 'assistant' || !before) return null
  return { reply, after: before.id }
}

/** The stopped reply put back after the message it followed; the list as it is when that message left. */
function putBack(list: readonly ChatMessage[], { reply, after }: StoppedReply): readonly ChatMessage[] {
  const index = list.findIndex((message) => message.id === after)
  return index < 0 ? list : [...list.slice(0, index + 1), reply, ...list.slice(index + 1)]
}

/**
 * The conversation with the stopped replies kept: one useChat still holds shows as it was when
 * Stop was pressed; one it dropped goes back after the message it followed. Never mutates.
 */
export function withStoppedReplies(
  messages: readonly ChatMessage[],
  stopped: readonly StoppedReply[],
): readonly ChatMessage[] {
  if (stopped.length === 0) return messages
  const copies = new Map(stopped.map((entry) => [entry.reply.id, entry.reply]))
  const held = new Set(messages.map((message) => message.id))
  const shown = messages.map((message) => copies.get(message.id) ?? message)
  return stopped.filter((entry) => !held.has(entry.reply.id)).reduce<readonly ChatMessage[]>(putBack, shown)
}

export function useStoppedReplies(chat: StoppableChat): StoppedReplies {
  const [stopped, setStopped] = useState<readonly StoppedReply[]>(NONE_STOPPED)
  const { messages, status, awaitingReply, stop: stopChat } = chat
  const stop = useCallback(() => {
    const writing = replyBeingWritten({ messages, status, awaitingReply })
    if (writing) setStopped((prev) => [...prev.filter((entry) => entry.reply.id !== writing.reply.id), writing])
    stopChat()
  }, [messages, status, awaitingReply, stopChat])
  const shown = useMemo(() => withStoppedReplies(messages, stopped), [messages, stopped])
  const isStopped = useCallback((replyId: string) => stopped.some((entry) => entry.reply.id === replyId), [stopped])
  return useMemo(() => ({ messages: shown, stop, isStopped }), [shown, stop, isStopped])
}
