'use client'

/**
 * PRD-268 §6.6, K4(7) (US-W03a) — a tapped reply chip reaches the chat's composer, which sends it as the person's
 * own message.
 *
 * The chips sit under a message and the composer below the list, and chat.tsx, which holds both, is not touched
 * (K4). A tap dispatches the window event `automatos:reply-chip` {chatId, text}. The chat's composer
 * (composer-with-suggestion.tsx) takes it and sends the text through its own sendMessage, in the shape its Send hands
 * the chat (role user, one text part), so it goes the way a typed message goes. While a turn is streaming it does
 * nothing, as Send does. The composer that takes a tap marks it handled (preventDefault), so no second composer sends
 * it again and the tap learns it went: one nobody took gives its chips back. Free text stays: the box is unchanged.
 */
import { useEffect } from 'react'

export const REPLY_CHIP_EVENT = 'automatos:reply-chip'
/** The composer's own busy state: Send does nothing while it holds. */
const STREAMING = 'streaming'

export interface ReplyChipTap {
  readonly chatId: string
  readonly text: string
}

/** Dispatches a tap on a chip of chat `chatId`; true once a composer has taken it and sent its text. */
export function sendReplyChip(chatId: string, text: string): boolean {
  const event = new CustomEvent<ReplyChipTap>(REPLY_CHIP_EVENT, { detail: { chatId, text }, cancelable: true })
  return !window.dispatchEvent(event)
}

/** The tap an event carries, its text trimmed; null when it carries none. */
function tapOf(event: Event): ReplyChipTap | null {
  const detail = (event as CustomEvent<unknown>).detail as Partial<Record<keyof ReplyChipTap, unknown>> | null
  const text = typeof detail?.text === 'string' ? detail.text.trim() : ''
  return typeof detail?.chatId === 'string' && text !== '' ? { chatId: detail.chatId, text } : null
}

/** The person's own message for a chip: the shape the composer's Send hands the chat, with no attachment. */
export function replyChipMessage(text: string) {
  return { role: 'user', content: text, attachment_ids: [], parts: [{ type: 'text', text }] }
}

/** The composer of chat `chatId` sends each chip tapped there through `sendMessage`, except while it streams. */
export function useReplyChipsSent(chatId: string, status: unknown, sendMessage: (message: unknown) => void): void {
  useEffect(() => {
    if (status === STREAMING) return undefined
    const onTap = (event: Event) => {
      const tap = tapOf(event)
      if (tap === null || tap.chatId !== chatId || event.defaultPrevented) return
      event.preventDefault()
      sendMessage(replyChipMessage(tap.text))
    }
    window.addEventListener(REPLY_CHIP_EVENT, onTap)
    return () => window.removeEventListener(REPLY_CHIP_EVENT, onTap)
  }, [chatId, status, sendMessage])
}
