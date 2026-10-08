'use client'

/**
 * QuestionCard — PRD-225 (G2): one open agent question, with its answer
 * controls. A card shows the question (markdown), who asked, the subject it
 * blocks, and THE CASCADE — the downstream work stuck behind it.
 *
 * Answer (free text ⌘/Ctrl-Enter, or an option button) posts to
 * POST /answer and the card leaves the list. Dismiss keeps the subject blocked
 * (the asker may re-ask) and shows the trail — answering "use your judgment" is
 * the one-click unblock instead.
 *
 * Rendered by the Questions tab and, PRD-252 R1, inside the ticket the question
 * was asked on (the board's viewer), where the link back to the ticket is
 * dropped. Reuses the ApprovalsInbox card shell and the chat markdown renderer —
 * no rival card, no rival markdown pipeline.
 */

import { useState } from 'react'
import Link from 'next/link'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Bot, Check, X, CornerDownRight } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { questionMarkdownComponents } from '@/components/chatbot/markdown-components'
import { useAnswerQuestion, useDenyApproval } from '@/hooks/use-approval-grants'
import type { ApprovalGrant } from '@/lib/api-client'
import { agentLabel, ticketLabel } from '@/lib/grant-owner'
import { questionTicketId, ticketHref } from '@/lib/ticket-links'
import { CardActions, CARD_ACTION_BUTTON, CARD_ACTION_ICON } from './card-actions'

const DISMISS_HINT = 'Answer "use your judgment" to unblock instead.'

function subjectHref(q: ApprovalGrant): string | null {
  // PRD-252 R1: the ticket it was asked on, opened at this question. Other
  // subjects have no deep link yet.
  const ticket = questionTicketId(q)
  return ticket ? ticketHref(ticket, q.id) : null
}

function askerLabel(q: ApprovalGrant): string {
  return agentLabel(q.owner, q.asked_by_agent_id)
}

function subjectLabel(q: ApprovalGrant): string {
  // F091-E1: the ticket by number and title, not "board_task:612"
  return ticketLabel(q.owner) ?? `${q.subject_type}:${q.subject_id}`
}

/** Who asked, and the subject it blocks; inside that ticket, only who asked. */
function QuestionHeader({ q, inTicket }: { q: ApprovalGrant; inTicket: boolean }) {
  const href = subjectHref(q)
  return (
    <div className="flex min-w-0 items-center gap-1.5 text-xs text-muted-foreground">
      <Bot className="h-3.5 w-3.5 shrink-0" />
      <span className="font-medium text-foreground">{askerLabel(q)}</span>
      {!inTicket && <span aria-hidden>·</span>}
      {!inTicket && href && (
        <Link href={href as any} className="truncate underline-offset-2 hover:underline">
          {subjectLabel(q)}
        </Link>
      )}
      {!inTicket && !href && <span className="truncate">{subjectLabel(q)}</span>}
    </div>
  )
}

/** The blocked cascade — downstream work stuck behind this ask. */
function BlockedCascade({ cascade }: { cascade: ApprovalGrant['cascade'] }) {
  if (!cascade || cascade.total <= 0) return null
  const shownTasks = cascade.tasks ?? []
  const overflow = cascade.total - shownTasks.length
  return (
    <div className="rounded border border-border/70 bg-muted/40 px-2 py-1.5" aria-label="Blocked cascade">
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
      {overflow > 0 && <p className="mt-0.5 text-[11px] text-muted-foreground opacity-70">+{overflow} more</p>}
    </div>
  )
}

export function QuestionCard({ q, inTicket = false }: { q: ApprovalGrant; inTicket?: boolean }) {
  const answerMut = useAnswerQuestion()
  const dismissMut = useDenyApproval()
  const [text, setText] = useState('')
  const [showFreeText, setShowFreeText] = useState(!Array.isArray(q.options) || q.options.length === 0)
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

  const options = Array.isArray(q.options) ? q.options : []
  const hasOptions = options.length > 0

  return (
    <div className="flex flex-col gap-2 rounded border border-border bg-background/50 p-3">
      <QuestionHeader q={q} inTicket={inTicket} />

      {/* The ask — markdown, via the shared chat renderer. */}
      <div className="md-view md-view-compact" aria-label="Question">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={questionMarkdownComponents}>
          {q.question_md || ''}
        </ReactMarkdown>
      </div>

      <BlockedCascade cascade={q.cascade} />

      {resolved === 'dismissed' ? (
        <p
          role="note"
          className="rounded border border-border/70 bg-muted/40 px-2 py-1.5 text-xs text-muted-foreground"
        >
          Dismissed — the subject stays blocked and the asker may re-ask. {DISMISS_HINT}
        </p>
      ) : (
        <div className="flex flex-col gap-2">
          {/* When the ask carries options — an allow/deny hold — the chips ARE
              the answer, so they are the primary control. Night 1: the biggest
              button on the card was "Answer", disabled until you typed, on
              questions where typing was never the answer. #1045: at their own
              width and equal weight (no option is the default), a long option
              wraps inside its chip instead of stretching the row. */}
          {hasOptions && (
            <div className="flex flex-wrap gap-2" role="group" aria-label="Answer options">
              {options.map((opt) => (
                <Button
                  key={opt}
                  size="sm"
                  variant="outline"
                  disabled={busy}
                  onClick={() => submit({ option: opt })}
                  className={`${CARD_ACTION_BUTTON} h-auto min-h-8 max-w-full whitespace-normal py-1.5 text-left`}
                >
                  {opt}
                </Button>
              ))}
            </div>
          )}

          {hasOptions ? (
            <button
              type="button"
              className="self-start text-[11px] text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
              onClick={() => setShowFreeText((v) => !v)}
            >
              {showFreeText ? 'Hide' : 'Answer in your own words instead'}
            </button>
          ) : null}

          {showFreeText && (
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
          )}

          <CardActions>
            <Button
              size="sm"
              variant="outline"
              disabled={busy}
              onClick={dismiss}
              title={DISMISS_HINT}
              className={CARD_ACTION_BUTTON}
            >
              <X className={CARD_ACTION_ICON} /> Dismiss
            </Button>
            {showFreeText && (
              <Button
                size="sm"
                disabled={busy || !text.trim()}
                onClick={() => submit({ answer_text: text })}
                className={CARD_ACTION_BUTTON}
              >
                <Check className={CARD_ACTION_ICON} /> Answer
              </Button>
            )}
          </CardActions>
        </div>
      )}
    </div>
  )
}
