'use client'

/**
 * PRD-265 US-008 — Studio's chat shows the paused banner the moment the page opens.
 *
 * The chat page shows its paused callout (`TrialExhaustedBanner`) when `errorCode` is
 * `credits_exhausted`, and `errorCode` came only from a refused send. A workspace whose credits
 * had already run out therefore saw the callout only after writing a message and having it
 * refused; until then only the header's balance pill said "Paused". This hook also reports the
 * billing summary's paused state through that same `errorCode`, so the callout, and the power-up
 * card stepping aside for it, follow either signal.
 *
 * PRD-237: a send in a new chat (no id yet) first makes it the person's draft in the session
 * store, so the page's cold start, still waiting on the newest conversation, never replaces the
 * chat the person just sent in. Every send on the page comes through here: the composer's and the
 * one-tap prompts alike.
 *
 * The chat page imports this hook in place of `useChatWithReceipts`; `Chat` itself, far over the
 * component-length limit, is not touched. The theme chat controller keeps `useChatWithReceipts`.
 * Hosted only: the summary is never fetched in the local edition, so nothing there is paused.
 */
import { useCallback } from 'react'

import { isWorkPaused, useBillingSummary } from '@/hooks/use-billing-summary'
import { type ChatWithReceipts, useChatWithReceipts } from '@/lib/chat/use-chat-with-receipts'
import { CREDITS_EXHAUSTED_CODE } from '@/lib/credits'
import { useChatSessionStore } from '@/stores/chat-session-store'

type ChatOptions = Parameters<typeof useChatWithReceipts>[0]
type SendMessage = ChatWithReceipts['sendMessage']

/** The code the chat page reads: `credits_exhausted` while work is paused, else the last send's own. */
export function errorCodeWithPause(errorCode: string | null, paused: boolean): string | null {
  return paused ? CREDITS_EXHAUSTED_CODE : errorCode
}

/** `sendMessage`, after a send in a new chat (`chatId` empty) has made that chat the session's draft. */
function useSendKeepsNewChat(chatId: string, sendMessage: SendMessage): SendMessage {
  return useCallback<SendMessage>(
    (message) => {
      if (!chatId) useChatSessionStore.getState().draftStarted()
      return sendMessage(message)
    },
    [chatId, sendMessage],
  )
}

export function useStudioChat(options: ChatOptions): ChatWithReceipts {
  const chat = useChatWithReceipts(options)
  const { data: summary } = useBillingSummary()
  const sendMessage = useSendKeepsNewChat(options.id, chat.sendMessage)
  return { ...chat, sendMessage, errorCode: errorCodeWithPause(chat.errorCode, isWorkPaused(summary)) }
}
