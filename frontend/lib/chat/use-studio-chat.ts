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
 * The chat page imports this hook in place of `useChatWithReceipts`; `Chat` itself, far over the
 * component-length limit, is not touched. The theme chat controller keeps `useChatWithReceipts`.
 * Hosted only: the summary is never fetched in the local edition, so nothing there is paused.
 */
import { isWorkPaused, useBillingSummary } from '@/hooks/use-billing-summary'
import { type ChatWithReceipts, useChatWithReceipts } from '@/lib/chat/use-chat-with-receipts'
import { CREDITS_EXHAUSTED_CODE } from '@/lib/credits'

type ChatOptions = Parameters<typeof useChatWithReceipts>[0]

/** The code the chat page reads: `credits_exhausted` while work is paused, else the last send's own. */
export function errorCodeWithPause(errorCode: string | null, paused: boolean): string | null {
  return paused ? CREDITS_EXHAUSTED_CODE : errorCode
}

export function useStudioChat(options: ChatOptions): ChatWithReceipts {
  const chat = useChatWithReceipts(options)
  const { data: summary } = useBillingSummary()
  return { ...chat, errorCode: errorCodeWithPause(chat.errorCode, isWorkPaused(summary)) }
}
