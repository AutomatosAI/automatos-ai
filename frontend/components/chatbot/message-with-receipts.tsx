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
 */
import { useMemo } from 'react'
import { Message, type MessageProps } from './message'
import { ReceiptsBlock } from './receipts-block'
import { receiptsOf } from '@/lib/chat/receipts'
import { ThreadCards } from '@/components/proactive/thread-cards'
import { threadPartsOf } from '@/lib/proactive/thread-parts'

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

export function MessageWithReceipts(props: MessageProps) {
  const thread = useMemo(() => threadPartsOf(props.message), [props.message])
  if (props.message.role !== 'assistant') return <MessageAndReceipts {...props} />
  return (
    <>
      <div data-ai-generated="true" data-testid="ai-generated-message">
        <MessageAndReceipts {...props} message={thread.message} />
      </div>
      <ThreadCards brief={thread.brief} suggestionIds={thread.suggestionIds} />
    </>
  )
}
