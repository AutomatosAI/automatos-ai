'use client'

/**
 * PRD-256 US-001 (FX-001) — the live receipts reach the reply being streamed.
 *
 * `useChat` (./hooks) reads the stream; it hands a `tool-data` frame to its data callback and
 * passes over a frame type it does not know. The backend sends a `receipts` frame's data on a
 * `tool-data` frame too, under `receipts` (consumers/chatbot/receipts.py `receipts_frames`), and
 * this hook takes it there: the receipts and the lines above the reply go on the last assistant
 * message, then the caller's own data callback runs as before. `useChat` itself, far over the
 * function-length limit, is not touched; the chat page imports this hook in its place.
 *
 * P256-FIX-RVW-22: `useChat` closes a `tool-end` with `success: false` as an error; an ask for the
 * owner's click carries `waiting: true` beside it, and this hook marks that call's line waiting.
 */
import { useRef } from 'react'
import { useChat } from '@/lib/chat/hooks'
import { liveReceiptsFromToolData, withReceiptsOnTheReply } from '@/lib/chat/receipts'
import { waitingToolCallId, withWaitingToolCall } from '@/lib/chat/tool-calls'

type ChatOptions = Parameters<typeof useChat>[0]
type Chat = ReturnType<typeof useChat>

export function useChatWithReceipts(options: ChatOptions): Chat {
  const setMessagesRef = useRef<Chat['setMessages'] | null>(null)
  const { onData } = options
  const chat = useChat({
    ...options,
    onData: (part: unknown) => {
      const live = liveReceiptsFromToolData(part)
      if (live) setMessagesRef.current?.((prev) => withReceiptsOnTheReply(prev, live))
      const waiting = waitingToolCallId(part)
      if (waiting) setMessagesRef.current?.((prev) => withWaitingToolCall(prev, waiting))
      onData?.(part)
    },
  })
  setMessagesRef.current = chat.setMessages
  return chat
}
