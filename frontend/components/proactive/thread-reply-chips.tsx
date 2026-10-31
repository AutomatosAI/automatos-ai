'use client'

/**
 * PRD-268 §6.6, §6.12.3, §6.12.4, K19 (US-W03a) — the reply chips under Auto's newest message.
 *
 * thread-cards.tsx draws this under an assistant message that offers chips (its live `replyChips`, else its saved
 * `reply_chips@1` part: lib/chat/turn-parts.ts). W2's ReplyChips shows them only while the message is the chat's
 * newest (newest-message.ts), so they are gone once a newer message renders. A tap goes to the chat's composer
 * (reply-chip-event.ts), which sends the chip's text as the person's own message. A tap nothing took (a turn is
 * streaming), or one no newer message followed within REPLY_FOLLOWS_MS (the chat passed over the send while it was
 * still finishing a turn), gives the chips back. Each chip is named "Reply: <chip>", its words carry data-ai-generated
 * and are drawn as text, in the secondary style: the composer's Send stays the view's ink primary.
 */
import { newerMessageWithin, useIsNewestMessage } from './newest-message'
import { sendReplyChip } from './reply-chip-event'
import { ReplyChips } from './reply-chips'

/** The avatar's width (w-8 + space-x-3), so the chips line up with the message's bubble. */
const ALIGNED_CLASS = 'max-w-[92%] pl-11'
const NOT_SENT = 'no composer sent the reply: a turn is streaming, or the chat has none'
/** A sent chip is the person's message, rendered at once; a tap nothing followed for this long gives the chips back. */
export const REPLY_FOLLOWS_MS = 2000

export interface ThreadReplyChipsProps {
  readonly chatId: string
  readonly messageId: string
  readonly chips: readonly string[]
}

/** The chips a message offers, while it is its chat's newest; a tap sends one as the person's message. */
export function ThreadReplyChips({ chatId, messageId, chips }: ThreadReplyChipsProps) {
  const latest = useIsNewestMessage(chatId, messageId)
  const reply = (text: string) =>
    sendReplyChip(chatId, text) ? newerMessageWithin(chatId, messageId, REPLY_FOLLOWS_MS) : Promise.reject(new Error(NOT_SENT))
  return <ReplyChips chips={chips} latest={latest} onReply={reply} className={ALIGNED_CLASS} />
}
