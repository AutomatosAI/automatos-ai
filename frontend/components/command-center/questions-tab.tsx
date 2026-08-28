'use client'

/**
 * QuestionsTab — PRD-225 (G2). The one place every open agent question is
 * queued: each ask parked its subject, and answering it here resumes the work
 * automatically. A card shows the question (markdown), who asked, the subject it
 * blocks, and THE CASCADE — the downstream work stuck behind it.
 *
 * Answer (free text ⌘/Ctrl-Enter, or an option button) posts to
 * POST /answer and the card leaves the list. Dismiss keeps the subject blocked
 * (the asker may re-ask) and shows the trail — answering "use your judgment" is
 * the one-click unblock instead.
 *
 * Reuses the ApprovalsInbox card shell and the chat markdown renderer — no rival
 * card, no rival markdown pipeline.
 */

import { useState } from 'react'
import Link from 'next/link'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Bot, Check, X, Clock, CornerDownRight } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { questionMarkdownComponents } from '@/components/chatbot/markdown-components'
import {
  useQuestions,
  useAnswerQuestion,
  useDenyApproval,
} from '@/hooks/use-approval-grants'
import type { ApprovalGrant } from '@/lib/api-client'

const DISMISS_HINT = 'Answer "use your judgment" to unblock instead.'

function subjectHref(q: ApprovalGrant): string | null {
  // Board tasks live on the Board tab; other subjects have no deep link yet.
  return q.subject_type === 'board_task' ? '/command-center?tab=board' : null
}

function askerLabel(q: ApprovalGrant): string {
  return q.asked_by_agent_id ? `Agent #${q.asked_by_agent_id}` : 'An agent'
}

function QuestionCard({ q }: { q: ApprovalGrant }) {
  const answerMut = useAnswerQuestion()
  const dismissMut = useDenyApproval()
  const [text, setText] = useState('')
  const [resolved, setResolved] = useState<null | 'answered' | 'dismissed'>(null)
  const busy = answerMut.isLoading || dismissMut.isLoading

  const submit = async (payload: { answer_text?: string; option?: string }) => {
    const value = (payload.answer_text ?? payload.option ?? '').trim()
    if (!value) return
    try {
      await answerMut.mutateAsync({ grantId: q.id, ...payload })
      toast.success('Answered — the agent is resuming')
      setResolved('answered')
    } catch {
      toast.error('Failed to send the answer')
    }
  }

  const dismiss = async () => {
    try {
      await dismissMut.mutateAsync(q.id)
      toast.info('Dismissed — the asker may re-ask')
      setResolved('dismissed')
    } catch {
      toast.error('Failed to dismiss the question')
    }
  }

  // Answering resumes the work — the card leaves the queue optimistically.
  if (resolved === 'answered') return null

  const href = subjectHref(q)
  const cascade = q.cascade
  const shownTasks = cascade?.tasks ?? []
  const overflow = cascade ? cascade.total - shownTasks.length : 0
  const options = Array.isArray(q.options) ? q.options : []

  return (
    <div className="flex flex-col gap-2 rounded border border-border bg-background/50 p-3">
      <div className="flex items-start justify-between gap-2">
        <div className="flex min-w-0 items-center gap-1.5 text-xs text-muted-foreground">
          <Bot className="h-3.5 w-3.5 shrink-0" />
          <span className="font-medium text-foreground">{askerLabel(q)}</span>
          <span aria-hidden>·</span>
          {href ? (
            <Link href={href as any} className="truncate underline-offset-2 hover:underline">
              {q.subject_type}:{q.subject_id}
            </Link>
          ) : (
            <span className="truncate">
              {q.subject_type}:{q.subject_id}
            </span>
          )}
        </div>
      </div>

      {/* The ask — markdown, via the shared chat renderer. */}
      <div className="prose prose-sm max-w-none dark:prose-invert" aria-label="Question">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={questionMarkdownComponents}>
          {q.question_md || ''}
        </ReactMarkdown>
      </div>

      {/* The blocked cascade — downstream work stuck behind this ask. */}
      {cascade && cascade.total > 0 && (
        <div
          className="rounded border border-border/70 bg-muted/40 px-2 py-1.5"
          aria-label="Blocked cascade"
        >
          <p className="mb-1 text-[11px] font-medium text-muted-foreground">
            Blocking {cascade.total} downstream task{cascade.total === 1 ? '' : 's'}
          </p>
          <ul className="flex flex-col gap-0.5">
            {shownTasks.map((t) => (
              <li key={t.id} className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                <CornerDownRight className="h-3 w-3 shrink-0" />
                <span className="truncate">{t.title}</span>
                <span className="shrink-0 opacity-70">· {t.status}</span>
              </li>
            ))}
          </ul>
          {overflow > 0 && (
            <p className="mt-0.5 text-[11px] text-muted-foreground opacity-70">+{overflow} more</p>
          )}
        </div>
      )}

      {resolved === 'dismissed' ? (
        <p
          role="note"
          className="rounded border border-border/70 bg-muted/40 px-2 py-1.5 text-xs text-muted-foreground"
        >
          Dismissed — the subject stays blocked and the asker may re-ask. {DISMISS_HINT}
        </p>
      ) : (
        <div className="flex flex-col gap-2">
          {options.length > 0 && (
            <div className="flex flex-wrap gap-2" aria-label="Answer options">
              {options.map((opt) => (
                <Button
                  key={opt}
                  size="sm"
                  variant="outline"
                  disabled={busy}
                  onClick={() => submit({ option: opt })}
                >
                  {opt}
                </Button>
              ))}
            </div>
          )}
          <textarea
            aria-label="Answer"
            value={text}
            disabled={busy}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                e.preventDefault()
                submit({ answer_text: text })
              }
            }}
            placeholder="Answer… (⌘/Ctrl-Enter to send)"
            rows={2}
            className="w-full resize-y rounded border border-border bg-background px-2 py-1.5 text-sm outline-none focus:border-primary"
          />
          <div className="flex gap-2">
            <Button
              size="sm"
              disabled={busy || !text.trim()}
              onClick={() => submit({ answer_text: text })}
              className="flex-1"
            >
              <Check className="mr-1 h-4 w-4" /> Answer
            </Button>
            <Button
              size="sm"
              variant="outline"
              disabled={busy}
              onClick={dismiss}
              title={DISMISS_HINT}
            >
              <X className="mr-1 h-4 w-4" /> Dismiss
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

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
