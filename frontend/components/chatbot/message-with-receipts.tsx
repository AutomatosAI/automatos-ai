'use client'

/**
 * PRD-256 US-001 (FX-001) — a reply with its receipts above it.
 *
 * The receipts block (what the turn's calls did, and the lines the platform puts above the
 * reply) sits directly above the assistant's bubble, aligned with it: live from the reply's
 * `receipts` (lib/chat/use-chat-with-receipts.ts), reloaded from its saved part. `Message`,
 * far over the component-length limit, is not touched; the chat page renders this in its place.
 * A message from before PRD-256, and every user message, renders exactly as `Message` does.
 */
import { Message, type MessageProps } from './message'
import { ReceiptsBlock } from './receipts-block'
import { receiptsOf } from '@/lib/chat/receipts'

export function MessageWithReceipts(props: MessageProps) {
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
