'use client'

/**
 * PRD-238 S1 — the thinking channel.
 *
 * Fills live while the model reasons, opens by itself until the answer starts
 * typing, then folds to one line the reader can reopen. On reload the stored
 * reasoning part renders folded. Never rendered as the answer.
 */
import { useEffect, useState } from 'react'
import { Brain, ChevronRight, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface ReasoningBlockProps {
  text: string
  /** True while this message is still streaming (reasoning or answer). */
  streaming: boolean
  /** True once answer text has started — the block folds. */
  answerStarted: boolean
}

export function ReasoningBlock({ text, streaming, answerStarted }: ReasoningBlockProps) {
  const thinking = streaming && !answerStarted
  const [open, setOpen] = useState(thinking)

  // Auto-open while thinking, auto-fold the moment the answer takes over.
  useEffect(() => {
    setOpen(thinking)
  }, [thinking])

  if (!text) return null
  const label = thinking ? 'Thinking…' : 'Thought process'

  return (
    <details
      open={open}
      onToggle={(event) => setOpen((event.currentTarget as HTMLDetailsElement).open)}
      className="group/reason rounded-lg border border-border/40 bg-secondary/20 text-xs"
      data-testid="reasoning-block"
    >
      <summary className="flex cursor-pointer list-none items-center gap-1.5 px-2 py-1 text-muted-foreground [&::-webkit-details-marker]:hidden">
        {thinking ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin text-primary/70" aria-hidden />
        ) : (
          <Brain className="h-3.5 w-3.5 text-primary/60" aria-hidden />
        )}
        <span className={cn(thinking && 'text-foreground')}>{label}</span>
        <ChevronRight className="ml-auto h-3 w-3 transition-transform group-open/reason:rotate-90" aria-hidden />
      </summary>
      <div className="max-h-64 overflow-y-auto whitespace-pre-wrap px-2.5 pb-2 text-muted-foreground/90">
        {text}
      </div>
    </details>
  )
}
