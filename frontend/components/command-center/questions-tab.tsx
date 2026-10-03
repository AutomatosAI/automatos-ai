'use client'

/**
 * QuestionsTab — PRD-225 (G2). The one place every open agent question is
 * queued: each ask parked its subject, and answering it here resumes the work
 * automatically. Each question is a QuestionCard (its own module, since PRD-252
 * R1 renders the same card inside the ticket the question was asked on).
 */

import { Clock } from 'lucide-react'
import { useQuestions } from '@/hooks/use-approval-grants'
import { QuestionCard } from './question-card'

export function QuestionsTab() {
  const { data, isLoading, isError } = useQuestions()
  const questions = data?.grants ?? []

  if (isLoading) {
    return <p className="p-3 text-sm text-muted-foreground">Loading questions…</p>
  }
  if (isError) {
    return (
      <p className="p-3 text-sm text-muted-foreground">
        Could not load questions. You may not be a workspace admin.
      </p>
    )
  }
  if (questions.length === 0) {
    return (
      <p className="flex items-center gap-2 p-3 text-sm text-muted-foreground">
        <Clock className="h-4 w-4" /> No open questions. Agents are deciding on their own.
      </p>
    )
  }

  return (
    <div className="flex flex-col gap-2">
      {questions.map((q) => (
        <QuestionCard key={q.id} q={q} />
      ))}
    </div>
  )
}
