'use client'

/**
 * PRD-252 R1 — a ticket's open questions, inside the ticket.
 *
 * A question row in Needs you, a question notification and the Questions tab's
 * ticket link open the ticket at its question, with the same card (and answer
 * controls) the Questions tab shows. Before, a ticket parked on a question said
 * only "Waiting for the operator" and the answer lived on another tab.
 */

import { useEffect, useRef } from 'react'
import { HelpCircle } from 'lucide-react'
import { useQuestions } from '@/hooks/use-approval-grants'
import { QuestionCard } from '@/components/command-center/question-card'
import { questionTicketId } from '@/lib/ticket-links'
import { cn } from '@/lib/utils'

interface TicketQuestionsProps {
  taskId: string
  focusQuestionId?: number | null
}

export function TicketQuestions({ taskId, focusQuestionId }: TicketQuestionsProps) {
  const { data } = useQuestions()
  const asks = (data?.grants ?? []).filter((q) => questionTicketId(q) === taskId)
  const focused = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    focused.current?.scrollIntoView?.({ block: 'nearest' })
  }, [focusQuestionId, asks.length])

  if (asks.length === 0) return null
  return (
    <section className="space-y-2 mb-6" aria-label="Questions on this ticket" data-testid="ticket-questions">
      <div className="flex items-center gap-1.5 text-muted-foreground">
        <HelpCircle className="w-3 h-3" />
        <h4 className="text-xs font-semibold uppercase tracking-wider">
          {asks.length === 1 ? 'Waiting for your answer' : `Waiting for ${asks.length} answers`}
        </h4>
      </div>
      {asks.map((q) => (
        <div
          key={q.id}
          ref={q.id === focusQuestionId ? focused : undefined}
          data-focused={q.id === focusQuestionId || undefined}
          className={cn('rounded', q.id === focusQuestionId && 'ring-2 ring-[hsl(var(--warning))]/60')}
        >
          <QuestionCard q={q} inTicket />
        </div>
      ))}
    </section>
  )
}
