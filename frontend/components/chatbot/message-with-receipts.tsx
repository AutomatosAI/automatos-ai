'use client'

/**
 * PRD-256 US-001 (FX-001) — a reply with its receipts above it.
 *
 * The receipts block (what the turn's calls did, and the lines the platform puts above the
 * reply) sits directly above the assistant's bubble, aligned with it: live from the reply's
 * `receipts` (lib/chat/use-chat-with-receipts.ts), reloaded from its saved part. `Message`,
 * far over the component-length limit, is not touched; the chat page renders this in its place.
 * A message from before PRD-256, and every user message, renders exactly as `Message` does.
 *
 * PRD-264 EUAI-09 (Art. 50(2)): every assistant message is marked machine-readable as
 * AI-generated: `data-ai-generated="true"` on the plain block that holds it (no class, so
 * the list's spacing and the message's own layout are unchanged).
 *
 * PRD-268 US-W01b: an assistant message's `brief@1` and `suggestion@1` parts draw as cards under it
 * (components/proactive/thread-cards.tsx), outside the AI-generated block, so their controls sit outside it and
 * each card marks Auto's words itself. `Message` gets a copy without those parts, and without the text part when
 * the brief card holds the prose (lib/proactive/thread-parts.ts). A message with neither renders as before.
 *
 * PRD-268 US-W03a: each message rendered here is registered as its chat's newest until a newer one renders
 * (components/proactive/newest-message.ts), and an assistant message's reply chips (lib/chat/turn-parts.ts) go to
 * ThreadCards, which draws them while it is the newest. A read-only chat has no composer to send them, so no chips.
 *
 * PRD-268 US-W03b: the follow-ups an assistant message's turn scheduled (its live `followUps`, else its `follow_up@1`
 * parts) go to ThreadCards too, which draws a line for each that the caller's own list holds.
 *
 * PRD-268 US-W06: an assistant message's `calibration@1` part goes to ThreadCards as the calibration card, and its text
 * part steps aside (the card holds the lines: lib/proactive/thread-parts.ts).
 */
import { useMemo } from 'react'
import { Message, type MessageProps } from './message'
import { ReceiptsBlock } from './receipts-block'
import { receiptsOf } from '@/lib/chat/receipts'
import { followUpsOfMessage, replyChipsOfMessage } from '@/lib/chat/turn-parts'
import type { FollowUpLinesProps } from '@/components/proactive/follow-up-line'
import { useRenderedMessage } from '@/components/proactive/newest-message'
import { ThreadCards } from '@/components/proactive/thread-cards'
import type { ThreadCalibrationProps } from '@/components/proactive/thread-calibration'
import type { ThreadReplyChipsProps } from '@/components/proactive/thread-reply-chips'
import { threadPartsOf, type ThreadParts } from '@/lib/proactive/thread-parts'

function MessageAndReceipts(props: MessageProps) {
  const { message } = props
  const receipts = message.role === 'assistant' ? receiptsOf(message) : undefined
  if (!receipts) return <Message {...props} />
  return (
    <div className="space-y-2" data-testid="message-with-receipts">
      {/* the avatar's width and gap (w-8 + space-x-3), so the block lines up with the bubble */}
      <div className="max-w-[92%] space-y-2 pl-11">
        <ReceiptsBlock receipts={receipts} above={message.receiptsAbove} />
      </div>
      <Message {...props} />
    </div>
  )
}

/** The reply chips a message offers, and where it sits; none on a read-only chat. */
function threadReplyChips({ chatId, message, isReadonly }: MessageProps): ThreadReplyChipsProps | undefined {
  const chips = isReadonly ? undefined : replyChipsOfMessage(message)
  return chips ? { chatId, messageId: message.id, chips } : undefined
}

/** The follow-ups a message's turn scheduled, with its chat; none when it scheduled none. */
function threadFollowUps({ chatId, message }: MessageProps): FollowUpLinesProps | undefined {
  const followUps = followUpsOfMessage(message)
  return followUps.length > 0 ? { chatId, followUps } : undefined
}

/** The calibration a message carries, with its chat; none when it carries none. */
function threadCalibration({ chatId, isReadonly }: MessageProps, { calibration }: ThreadParts): ThreadCalibrationProps | undefined {
  return calibration ? { chatId, calibration, isReadonly } : undefined
}

export function MessageWithReceipts(props: MessageProps) {
  const thread = useMemo(() => threadPartsOf(props.message), [props.message])
  useRenderedMessage(props.chatId, props.message.id)
  if (props.message.role !== 'assistant') return <MessageAndReceipts {...props} />
  return (
    <>
      <div data-ai-generated="true" data-testid="ai-generated-message">
        <MessageAndReceipts {...props} message={thread.message} />
      </div>
      <ThreadCards
        brief={thread.brief}
        suggestionIds={thread.suggestionIds}
        replyChips={threadReplyChips(props)}
        followUps={threadFollowUps(props)}
        calibration={threadCalibration(props, thread)}
      />
    </>
  )
}
