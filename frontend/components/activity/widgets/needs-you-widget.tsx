'use client'

/**
 * PRD-244 review batch 2 (Gerard, 2026-09-17) — "Needs you": the one widget
 * for everything only a human can move: open questions (PRD-225), missions
 * waiting for approval (approval gates) and decisions (reports + missions
 * escalated). Replaces the Decisions Needed and Approval Gates widgets. Each
 * row lands on the Command Centre tab that owns it.
 */
import Link from 'next/link'
import { AlertTriangle, CheckCircle2, ClipboardCheck, Clock, FileText, HelpCircle, Loader2, ShieldCheck, Target } from 'lucide-react'
import { useDecisionsNeeded, useApprovalGates } from '@/hooks/use-kpi-api'
import { useQuestions } from '@/hooks/use-approval-grants'
import { useBoardTasksList } from '@/hooks/use-board-tasks-api'
import { AUTO_NOW_LINKS, questionPreview } from '@/components/chatbot/auto-now-rail'
import { cn } from '@/lib/utils'

const ROWS = 3

const LEVEL_LABELS: Record<number, string> = { 4: 'L4 SECURITY', 3: 'L3 URGENT', 2: 'L2 APPROVAL', 1: 'L1 TASK', 0: 'L0 FYI' }
const LEVEL_TONES: Record<number, string> = {
  4: 'bg-red-500/15 text-red-500 border-red-500/30',
  3: 'bg-warning/15 text-warning border-warning/30',
  2: 'bg-amber-500/15 text-amber-500 border-amber-500/30',
  1: 'bg-muted text-muted-foreground border-border',
  0: 'bg-muted/50 text-muted-foreground border-border',
}

export function formatAge(iso: string | null | undefined): string {
  if (!iso) return ''
  const seconds = Math.floor((Date.now() - new Date(iso).getTime()) / 1000)
  if (seconds < 60) return `${seconds}s`
  const mins = Math.floor(seconds / 60)
  if (mins < 60) return `${mins}m`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}h`
  return `${Math.floor(hours / 24)}d`
}

function Section({ title, count, href, children }: { title: string; count: number; href: string; children: React.ReactNode }) {
  if (count === 0) return null
  return (
    <div className="space-y-1.5" aria-label={title}>
      <div className="flex items-center justify-between">
        <p className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
          {title} · {count}
        </p>
        <Link href={href as any} className="text-[10px] text-muted-foreground underline-offset-2 hover:text-foreground hover:underline">
          All →
        </Link>
      </div>
      {children}
    </div>
  )
}

const rowClass = 'block w-full rounded-lg border border-border/50 bg-muted/30 p-2 text-left transition-colors hover:bg-muted/60'

// F207: a list that could not be loaded is never "Nothing on your plate".
const DECISIONS_NOT_LOADED = 'The decisions waiting for you could not be loaded. Try again shortly.'

interface NeedsYouWidgetProps {
  period: string
  className?: string
}

export function NeedsYouWidget({ period, className }: NeedsYouWidgetProps) {
  const decisions = useDecisionsNeeded(10)
  const gates = useApprovalGates(period)
  const questions = useQuestions()
  // Tickets an agent finished and handed BACK for a verdict. They were missing
  // from this widget entirely: on night 1 four of them waited for the owner all
  // night on a panel whose whole job is to say what is waiting for the owner.
  const reviews = useBoardTasksList({ status: 'review', limit: '10' })

  const isLoading = decisions.isLoading || gates.isLoading || questions.isLoading || reviews.isLoading
  const asks = questions.data?.grants ?? []
  const pendingMissions = gates.data?.pending_missions ?? []
  const pendingCount = gates.data?.pending_count ?? 0
  const decisionItems = decisions.data?.items ?? []
  const decisionsTotal = decisions.data?.total ?? 0
  const decisionsError = decisions.data?.error ?? (decisions.isError ? DECISIONS_NOT_LOADED : null)
  const reviewTasks = reviews.data?.tasks ?? []
  const reviewTotal = reviews.data?.total ?? reviewTasks.length
  const total = asks.length + pendingCount + decisionsTotal + reviewTotal

  return (
    <div className={cn('h-full flex flex-col', className)}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <div className="flex items-center gap-2">
          <AlertTriangle className={cn('w-4 h-4', total > 0 ? 'text-warning' : 'text-success')} />
          <h3 className="text-sm font-semibold">Needs you</h3>
        </div>
        {!isLoading && total > 0 && (
          <span className="text-xs bg-warning/15 text-warning px-1.5 py-0.5 rounded-full font-medium">{total} waiting</span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4">
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : total === 0 && !decisionsError ? (
          <div className="flex flex-col items-center justify-center py-6 text-muted-foreground">
            <CheckCircle2 className="w-8 h-8 mb-2 opacity-30" />
            <p className="text-xs">Nothing on your plate. Grab a tea.</p>
          </div>
        ) : (
          <>
            <Section title="Questions" count={asks.length} href={AUTO_NOW_LINKS.questions}>
              {asks.slice(0, ROWS).map((q) => (
                <Link key={q.id} href={AUTO_NOW_LINKS.questions as any} className={rowClass}>
                  <div className="flex items-start gap-2">
                    <HelpCircle className="w-3 h-3 mt-0.5 shrink-0 text-muted-foreground" />
                    <span className="text-xs leading-snug line-clamp-2 flex-1">{questionPreview(q.question_md)}</span>
                  </div>
                  <div className="pl-5 text-[10px] text-muted-foreground">
                    {q.asked_by_agent_id ? `Agent #${q.asked_by_agent_id}` : 'An agent'}
                    {q.requested_at && ` · ${formatAge(q.requested_at)} ago`}
                  </div>
                </Link>
              ))}
            </Section>

            <Section title="In review" count={reviewTotal} href={AUTO_NOW_LINKS.board}>
              {reviewTasks.slice(0, ROWS).map((t: any) => (
                <Link key={t.id} href={AUTO_NOW_LINKS.board as any} className={rowClass}>
                  <div className="flex items-start gap-2">
                    <ClipboardCheck className="w-3 h-3 mt-0.5 shrink-0 text-muted-foreground" />
                    <span className="text-xs leading-snug line-clamp-2 flex-1">{t.title}</span>
                  </div>
                  <div className="pl-5 text-[10px] text-muted-foreground">
                    {t.agent_name ? t.agent_name : 'An agent'}
                    {t.completed_at && ` · ${formatAge(t.completed_at)} ago`}
                  </div>
                </Link>
              ))}
            </Section>

            <Section title="Approvals" count={pendingCount} href={AUTO_NOW_LINKS.governance}>
              {pendingMissions.slice(0, ROWS).map((m) => (
                <Link key={m.id} href={AUTO_NOW_LINKS.governance as any} className={cn(rowClass, 'border-warning/10 bg-warning/5')}>
                  <div className="flex items-start gap-2">
                    <ShieldCheck className="w-3 h-3 mt-0.5 shrink-0 text-warning" />
                    <span className="text-xs leading-snug line-clamp-2 flex-1">{m.goal}</span>
                  </div>
                  {m.waiting_since && (
                    <div className="flex items-center gap-1 pl-5 text-[10px] text-muted-foreground">
                      <Clock className="w-2.5 h-2.5" /> Waiting {formatAge(m.waiting_since)}
                    </div>
                  )}
                </Link>
              ))}
            </Section>

            {decisionsError && (
              <p role="status" className="text-xs text-warning">
                {decisionsError}
              </p>
            )}

            <Section title="Decisions" count={decisionsTotal} href={AUTO_NOW_LINKS.governance}>
              {decisionItems.slice(0, ROWS + 1).map((item) => {
                const level = item.escalation_level ?? 0
                const Icon = item.kind === 'report' ? FileText : Target
                return (
                  <Link key={`${item.kind}:${item.id}`} href={AUTO_NOW_LINKS.governance as any} className={rowClass}>
                    <div className="flex items-start gap-2">
                      <Icon className="w-3 h-3 mt-0.5 shrink-0 text-muted-foreground" />
                      <span className="text-xs leading-snug line-clamp-2 flex-1">{item.title}</span>
                      <span className={cn('text-[9px] px-1.5 py-0.5 rounded-full border font-medium shrink-0', LEVEL_TONES[level] || LEVEL_TONES[0])}>
                        {LEVEL_LABELS[level] || 'L0'}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 pl-5 text-[10px] text-muted-foreground">
                      <span className="capitalize">{item.kind}</span>
                      {item.agent_name && <span>· {item.agent_name}</span>}
                      {item.created_at && <span>· {formatAge(item.created_at)} ago</span>}
                    </div>
                  </Link>
                )
              })}
            </Section>
          </>
        )}
      </div>
    </div>
  )
}
