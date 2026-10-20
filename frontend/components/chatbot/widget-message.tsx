'use client'

/**
 * One message in the floating AutoWidget's conversation (chat-widget.tsx): yours on the
 * right as plain text, Auto's on the left as markdown.
 *
 * PRD-264 EUAI-09 (Art. 50(2)): Auto's message is marked machine-readable as AI-generated,
 * `data-ai-generated="true"` on its row, the same mark the core chat (MessageWithReceipts)
 * and Play's AutoReply carry. A user message carries no mark.
 */
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

import type { ChatMessage } from '@/types'

const ASSISTANT_MARKDOWN =
  'prose prose-sm prose-invert max-w-none [&_p]:mb-1.5 [&_p]:last:mb-0 [&_ul]:mb-1.5 [&_li]:mb-0.5 [&_code]:text-xs [&_pre]:text-xs [&_h1]:text-sm [&_h2]:text-sm [&_h3]:text-xs'

export function WidgetMessage({ message }: { readonly message: ChatMessage }) {
  const isUser = message.role === 'user'
  const isAssistant = message.role === 'assistant'
  return (
    <div
      className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
      data-ai-generated={isAssistant ? 'true' : undefined}
      data-testid="widget-message"
    >
      <div
        className={`max-w-[85%] rounded-xl px-3 py-2 text-sm ${
          isUser ? 'bg-primary/15 text-foreground' : 'bg-secondary/40 text-foreground'
        }`}
      >
        {isAssistant ? (
          <div className={ASSISTANT_MARKDOWN}>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content || ''}</ReactMarkdown>
          </div>
        ) : (
          <p className="whitespace-pre-wrap">{message.content}</p>
        )}
      </div>
    </div>
  )
}
