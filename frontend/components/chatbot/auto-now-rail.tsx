'use client'

/**
 * PRD-244 D5 — the "Auto now" rail: a live view of the floor beside the
 * conversation. Every section is a read the Command Centre already makes;
 * every row deep-links to its Command Centre tab; empty sections say so and
 * never fabricate a count. Built from the shared primitives and tokens, so it
 * renders in both styles: the Studio shell mounts it in its rail, the Classic
 * chat in an aside of its own.
 */
import { useState } from 'react'
import Link from 'next/link'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useAnswerQuestion } from '@/hooks/use-approval-grants'
import { useAutoNow, AUTO_NOW_ROWS } from '@/hooks/use-auto-now'
import { formatRelative } from '@/lib/format-relative'
import type { ApprovalGrant } from '@/lib/api-client'
import { cn } from '@/lib/utils'

export const AUTO_NOW_LINKS = {
  board: '/command-center?tab=board',
  questions: '/command-center?tab=questions',
  watchlist: '/command-center?tab=watchlist',
  governance: '/command-center?tab=governance',
} as const

const QUESTION_PREVIEW_CHARS = 140

/** The first line of the ask, markdown marks removed, for a one-line row. */
export function questionPreview(md: string | null | undefined): string {
  const line = (md ?? '').split('\n').map((l) => l.trim()).find(Boolean) ?? ''
  const plain = line.replace(/[#*_`>]+/g, '').trim()
  return plain.length > QUESTION_PREVIEW_CHARS ? `${plain.slice(0, QUESTION_PREVIEW_CHARS - 1)}…` : plain
}

function Eyebrow({ children, count, href }: { children: string; count?: number; href?: string }) {
  const label = (
    <span className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
      {children}
      {count !== undefined && count > 0 && (
        <span className="ml-1.5 rounded-full bg-accent/15 px-1.5 py-px text-[10px] font-medium text-accent">{count}</span>
      )}
    </span>
  )
  return (
    <div className="flex items-baseline justify-between gap-2">
      {label}
      {href && (
        <Link href={href as any} className="text-[11px] text-muted-foreground underline-offset-2 hover:text-foreground hover:underline">
          All →
        </Link>
      )}
    </div>
  )
}

function EmptyLine({ children }: { children: string }) {
  return <p className="text-[12px] leading-snug text-muted-foreground">{children}</p>
}

function Row({ href, primary, secondary }: { href: string; primary: string; secondary?: string }) {
  return (
    <Link
      href={href as any}
      className="block rounded-md border border-border bg-card/60 px-2.5 py-1.5 text-[12px] leading-snug text-foreground transition-colors hover:border-accent/40 hover:bg-card"
    >
      <span className="block truncate font-medium">{primary}</span>
      {secondary && <span className="block truncate text-[11px] text-muted-foreground">{secondary}</span>}
    </Link>
  )
}

function QuestionRow({ q }: { q: ApprovalGrant }) {
  const answer = useAnswerQuestion()
  const [text, setText] = useState('')
  const [done, setDone] = useState(false)
  const options = Array.isArray(q.options) ? q.options : []
  const asker = q.asked_by_agent_id ? `Agent #${q.asked_by_agent_id}` : 'An agent'

  const submit = async (payload: { answer_text?: string; option?: string }) => {
    const value = (payload.answer_text ?? payload.option ?? '').trim()
    if (!value) return
    try {
      await answer.mutateAsync({ grantId: q.id, ...payload })
      toast.success('Answered — the agent is resuming')
      setDone(true)
    } catch {
      toast.error('Failed to send the answer')
    }
  }

  if (done) return null

  return (
    <div className="flex flex-col gap-1.5 rounded-md border border-border bg-card/60 px-2.5 py-2 text-[12px] leading-snug" aria-label={`Question from ${asker}`}>
      <Link href={AUTO_NOW_LINKS.questions as any} className="block hover:underline">
        <span className="block text-[11px] text-muted-foreground">{asker} asks</span>
        <span className="block text-foreground">{questionPreview(q.question_md)}</span>
      </Link>
      {options.length > 0 ? (
        <div className="flex flex-wrap gap-1">
          {options.map((opt) => (
            <Button key={opt} type="button" size="sm" variant="outline" className="h-6 px-2 text-[11px]" disabled={answer.isLoading} onClick={() => void submit({ option: opt })}>
              {opt}
            </Button>
          ))}
        </div>
      ) : (
        <form
          className="flex gap-1"
          onSubmit={(e) => {
            e.preventDefault()
            void submit({ answer_text: text })
          }}
        >
          <Input value={text} onChange={(e) => setText(e.target.value)} placeholder="Answer…" aria-label="Answer" className="h-7 text-[12px]" />
          <Button type="submit" size="sm" className="h-7 px-2 text-[11px]" disabled={answer.isLoading || !text.trim()}>
            Send
          </Button>
        </form>
      )}
    </div>
  )
}

export function AutoNowRail({ className }: { className?: string }) {
  const now = useAutoNow()
  const stats = now.stats

  return (
    <div className={cn('flex flex-col gap-4', className)} aria-label="Auto now">
      <section className="flex flex-col gap-1.5">
        <Eyebrow href={AUTO_NOW_LINKS.board}>Working now</Eyebrow>
        <dl className="grid grid-cols-4 gap-1 text-center">
          {(
            [
              ['Working', stats?.working_now],
              ['Agents', stats?.agents_active],
              ['Queue', stats?.tasks_in_queue],
              ['Attention', stats?.needs_attention],
            ] as const
          ).map(([label, value]) => (
            <div key={label} className="rounded-md border border-border bg-card/60 px-1 py-1">
              <dt className="text-[9px] uppercase tracking-wider text-muted-foreground">{label}</dt>
              <dd className="font-mono text-[13px] text-foreground">{value ?? '—'}</dd>
            </div>
          ))}
        </dl>
        {now.working.length === 0 ? (
          <EmptyLine>No one is working right now.</EmptyLine>
        ) : (
          now.working.slice(0, AUTO_NOW_ROWS).map((a) => (
            <Row key={a.agent_id} href={AUTO_NOW_LINKS.board} primary={a.name} secondary={a.current?.title} />
          ))
        )}
      </section>

      <section className="flex flex-col gap-1.5">
        <Eyebrow count={now.questionCount} href={AUTO_NOW_LINKS.questions}>Questions</Eyebrow>
        {now.questionCount === 0 ? (
          <EmptyLine>Nothing waiting on you.</EmptyLine>
        ) : (
          now.questions.slice(0, AUTO_NOW_ROWS).map((q) => <QuestionRow key={q.id} q={q} />)
        )}
      </section>

      <section className="flex flex-col gap-1.5">
        <Eyebrow count={now.watches.length} href={AUTO_NOW_LINKS.watchlist}>Watchlist</Eyebrow>
        {now.watches.length === 0 ? (
          <EmptyLine>Nothing being watched.</EmptyLine>
        ) : (
          <Row
            href={AUTO_NOW_LINKS.watchlist}
            primary={now.nextWatch?.title ?? now.watches[0].title}
            secondary={now.nextWatch?.next_check_at ? `next check ${formatRelative(now.nextWatch.next_check_at)}` : `${now.watches.length} live`}
          />
        )}
      </section>

      <section className="flex flex-col gap-1.5">
        <Eyebrow count={now.decisionsTotal} href={AUTO_NOW_LINKS.governance}>Decisions</Eyebrow>
        {now.decisionsTotal === 0 ? (
          <EmptyLine>No decisions waiting.</EmptyLine>
        ) : (
          <Row href={AUTO_NOW_LINKS.governance} primary={`${now.decisionsTotal} waiting for a decision`} secondary="Open Governance" />
        )}
      </section>
    </div>
  )
}
