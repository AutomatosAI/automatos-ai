'use client'

/**
 * PRD-244 review batch 2 (Gerard, 2026-09-17) — "Needs you": the one widget
 * for everything only a human can move.
 *
 * PRD-252 R1: a row opens the thing itself, not the tab that lists it: the
 * ticket in review, the question inside its ticket, an approval at its ticket,
 * a mission at its plan. "All →" still lands on the tab that owns the family.
 *
 * PRD-252 R5: the number and the rows come from one endpoint (useNeedsYou), the
 * number the Board tab, ATTENTION and Auto's pill show. Every row is listed, so
 * the header's number is the rows below it; a family with more than the
 * endpoint lists says how many more.
 *
 * F246: a stuck ticket (one nothing will move until the owner does) is its own
 * family, each row saying why.
 */
import Link from 'next/link'
import { AlertTriangle, CheckCircle2, CircleSlash, ClipboardCheck, HelpCircle, Loader2, ShieldCheck, XCircle } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { useNeedsYou, type NeedsYou, type NeedsYouAskRow, type NeedsYouTicketRow, type StuckWhy } from '@/hooks/use-needs-you'
import { AUTO_NOW_LINKS, questionPreview } from '@/components/chatbot/auto-now-rail'
import { cn } from '@/lib/utils'
import { QUESTIONS_HREF, missionHref, ticketHref } from '@/lib/ticket-links'
import { numberedTitle } from '@/components/activity/board/ticket-kind'

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

/** A ticket opens in the board's viewer; a mission's own card opens the mission. */
export function ticketRowHref(row: NeedsYouTicketRow): string {
  return row.mission_id ? missionHref(row.mission_id) : ticketHref(row.ticket_id)
}

/** A question or approval opens inside its ticket; a mission at its plan; anything else on its tab. */
export function askRowHref(row: NeedsYouAskRow, fallback: string): string {
  if (row.source === 'mission') return missionHref(row.id)
  return row.ticket_id != null ? ticketHref(row.ticket_id, row.id) : fallback
}

// F207: a list that could not be loaded is never "Nothing on your plate".
export const NEEDS_YOU_NOT_LOADED = 'What needs you could not be loaded. Try again shortly.'

interface ShownRow {
  key: string
  href: string
  title: string
  meta: string
}

const by = (who: string | null, at: string | null) => [who ?? 'An agent', at && `${formatAge(at)} ago`].filter(Boolean).join(' · ')

// PRD-252 R4: a ticket is named by its number and title, so two alike can be told apart.
function ticketRows(rows: NeedsYouTicketRow[]): ShownRow[] {
  return rows.map((r) => ({
    key: `t${r.ticket_id}`, href: ticketRowHref(r),
    title: numberedTitle(r.number, r.title ?? `Ticket ${r.ticket_id}`), meta: by(r.agent_name, r.at),
  }))
}

/** F246: why nothing will move a stuck ticket, in the owner's words. */
export const STUCK_WHY: Record<StuckWhy, string> = {
  no_host: 'Waiting for a CLI host that is not online',
  no_agent: 'Assigned to no agent',
  not_picked_up: 'Waiting, but nothing will run it',
  mission_ended: 'Its mission has ended',
}

function stuckRows(rows: NeedsYouTicketRow[]): ShownRow[] {
  return ticketRows(rows).map((shown, i) => {
    const why = rows[i].why
    return why ? { ...shown, meta: `${STUCK_WHY[why]} · ${shown.meta}` } : shown
  })
}

// F246: a mission's plan names its card by number, as every ticket row does.
function askRows(rows: NeedsYouAskRow[], fallback: string): ShownRow[] {
  return rows.map((r) => ({
    key: `${r.source}${r.id}`,
    href: askRowHref(r, fallback),
    title: r.source === 'mission' ? numberedTitle(r.ticket_number, r.title ?? 'A mission plan') : questionPreview(r.title ?? ''),
    meta: r.source === 'mission' ? `Mission plan${r.at ? ` · waiting ${formatAge(r.at)}` : ''}`
      : [r.ticket_number, by(r.agent_name, r.at)].filter(Boolean).join(' · '),
  }))
}

interface Family {
  title: string
  href: string
  icon: LucideIcon
  count: number
  rows: ShownRow[]
}

/** The five families, in the order the owner acts on them. */
export function families(data: NeedsYou): Family[] {
  const { counts, rows } = data
  return [
    { title: 'In review', href: AUTO_NOW_LINKS.board, icon: ClipboardCheck, count: counts.review, rows: ticketRows(rows.review) },
    { title: 'Questions', href: AUTO_NOW_LINKS.questions, icon: HelpCircle, count: counts.question, rows: askRows(rows.question, QUESTIONS_HREF) },
    { title: 'Approvals', href: AUTO_NOW_LINKS.governance, icon: ShieldCheck, count: counts.approval, rows: askRows(rows.approval, AUTO_NOW_LINKS.governance) },
    { title: 'Stuck', href: AUTO_NOW_LINKS.board, icon: CircleSlash, count: counts.stuck, rows: stuckRows(rows.stuck) },
    { title: 'Failed', href: AUTO_NOW_LINKS.board, icon: XCircle, count: counts.failed, rows: ticketRows(rows.failed) },
  ]
}

const rowClass = 'block w-full rounded-lg border border-border/50 bg-muted/30 p-2 text-left transition-colors hover:bg-muted/60'
const linkClass = 'text-[10px] text-muted-foreground underline-offset-2 hover:text-foreground hover:underline'

function Section({ family }: { family: Family }) {
  const { title, href, icon: Icon, count, rows } = family
  if (count === 0) return null
  const more = count - rows.length
  return (
    <div className="space-y-1.5" aria-label={title}>
      <div className="flex items-center justify-between">
        <p className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
          {title} · {count}
        </p>
        <Link href={href as any} className={linkClass}>All →</Link>
      </div>
      {rows.map((row) => (
        <Link key={row.key} href={row.href as any} className={rowClass}>
          <div className="flex items-start gap-2">
            <Icon className="w-3 h-3 mt-0.5 shrink-0 text-muted-foreground" />
            <span className="text-xs leading-snug line-clamp-2 flex-1">{row.title}</span>
          </div>
          {row.meta && <div className="pl-5 text-[10px] text-muted-foreground">{row.meta}</div>}
        </Link>
      ))}
      {more > 0 && <Link href={href as any} className={cn(linkClass, 'block')}>{more} more →</Link>}
    </div>
  )
}

function Body({ query }: { query: ReturnType<typeof useNeedsYou> }) {
  if (query.isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
      </div>
    )
  }
  if (query.isError || !query.data) {
    return <p role="status" className="text-xs text-warning">{NEEDS_YOU_NOT_LOADED}</p>
  }
  if (query.data.total === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-6 text-muted-foreground">
        <CheckCircle2 className="w-8 h-8 mb-2 opacity-30" />
        <p className="text-xs">Nothing on your plate. Grab a tea.</p>
      </div>
    )
  }
  return <>{families(query.data).map((family) => <Section key={family.title} family={family} />)}</>
}

interface NeedsYouWidgetProps {
  className?: string
}

export function NeedsYouWidget({ className }: NeedsYouWidgetProps) {
  const query = useNeedsYou()
  const total = query.data?.total ?? 0

  return (
    <div className={cn('h-full flex flex-col', className)}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <div className="flex items-center gap-2">
          <AlertTriangle className={cn('w-4 h-4', total > 0 ? 'text-warning' : 'text-success')} />
          <h3 className="text-sm font-semibold">Needs you</h3>
        </div>
        {total > 0 && (
          <span className="text-xs bg-warning/15 text-warning px-1.5 py-0.5 rounded-full font-medium">{total} waiting</span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4">
        <Body query={query} />
      </div>
    </div>
  )
}
