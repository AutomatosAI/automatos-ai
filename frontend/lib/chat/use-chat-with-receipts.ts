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
 *
 * PRD-260: `reload` (Studio's regenerate, Play's Try again) is built here on each render's own
 * `sendMessage` (./retry), so a turn that failed can be asked again.
 *
 * PRD-264 EUAI-11a: the agent the owner picked goes to the host's AI label (lib/ai-label), which
 * says "{agent name} is an AI" while that agent answers. The default chat and the chat controller
 * (Play's chat, a theme's) both stream through here.
 *
 * Theme voice: the voice session's words in progress in this conversation end its `messages`
 * (./live-voice-turns), so Studio's chat and every theme's chat show them; useChat's own state and
 * Try again never hold them.
 */
import { useCallback, useRef } from 'react'
import { usePublishAnsweringAgent } from '@/lib/ai-label/store'
import { useChat } from '@/lib/chat/hooks'
import { useLiveVoiceMessages } from '@/lib/chat/live-voice-turns'
import { liveReceiptsFromToolData, withReceiptsOnTheReply } from '@/lib/chat/receipts'
import { lastTurn } from '@/lib/chat/retry'
import { waitingToolCallId, withWaitingToolCall } from '@/lib/chat/tool-calls'

type ChatOptions = Parameters<typeof useChat>[0]
type Chat = ReturnType<typeof useChat>

export type ChatWithReceipts = Chat & {
  /** Ask the owner's last message again; nothing happens while a turn is running. */
  readonly reload: () => void
}

/**
 * The owner's last message sent again, through this render's `sendMessage`. While a turn runs
 * it does nothing: `sendMessage` would refuse, and the message must not leave the list unsent.
 */
function useAskAgain({ messages, setMessages, sendMessage, isLoading }: Chat): () => void {
  return useCallback(() => {
    const turn = lastTurn(messages)
    if (isLoading || !turn) return
    setMessages(turn.kept)
    void sendMessage(turn.content)
  }, [messages, setMessages, sendMessage, isLoading])
}

export function useChatWithReceipts(options: ChatOptions): ChatWithReceipts {
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
  usePublishAnsweringAgent(options.selectedAgentId)
  const reload = useAskAgain(chat)
  const messages = useLiveVoiceMessages(chat.messages, options.id)
  return { ...chat, messages, reload }
}
