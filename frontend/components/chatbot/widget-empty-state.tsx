'use client'

import { Bot } from 'lucide-react'
import type { PageQuickPrompt } from '@/lib/generated/page-manifest'

/** PRD-206 S3: the resume chip, always offered on an empty thread and answered by platform_resume_context. */
const RESUME_PROMPT = 'Where did we leave off?'

const CHIP_CLASS =
  'rounded-full border border-border bg-muted/50 px-2.5 py-1 text-xs text-foreground transition-colors hover:bg-muted disabled:opacity-50'

/**
 * The floating widget's empty thread: Auto's greeting for the page and the quick-prompt chips
 * (PRD-221 S7). Split out of chat-widget.tsx so the conversation component stays within the
 * code-shape limit.
 */
export function WidgetEmptyState({
  pageLabel,
  quickPrompts,
  isLoading,
  onSend,
}: {
  pageLabel: string
  quickPrompts: PageQuickPrompt[]
  isLoading: boolean
  onSend: (text: string) => void
}) {
  const send = (text: string) => {
    if (!isLoading) onSend(text)
  }
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-4">
      <div className="w-10 h-10 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center mb-3">
        <Bot className="w-5 h-5 text-primary" />
      </div>
      <p className="text-sm font-medium text-foreground mb-1">Hey, I&apos;m Auto</p>
      <p className="text-xs text-muted-foreground">
        Ask me anything about {pageLabel || 'this page'}, or tell me what you need help with.
      </p>
      <div className="mt-3 flex flex-wrap justify-center gap-1.5">
        <button type="button" onClick={() => send(RESUME_PROMPT)} disabled={isLoading} className={CHIP_CLASS}>
          {RESUME_PROMPT}
        </button>
        {quickPrompts.map((qp) => (
          <button key={qp.text} type="button" onClick={() => send(qp.text)} disabled={isLoading} className={CHIP_CLASS}>
            {qp.text}
          </button>
        ))}
      </div>
    </div>
  )
}
