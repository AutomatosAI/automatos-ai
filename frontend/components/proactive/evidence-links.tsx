'use client'

/**
 * PRD-268 §6.5.1 (US-W01a) — a suggestion's evidence, as links that open in the app.
 *
 * Each ref the check-in cited opens the thing it names (lib/proactive/evidence.ts): a ticket on the board,
 * a report in the feed, the Documents page, a Mission, a chat. A ticket is cited by its board number, and
 * the board opens a ticket by its row id, so a ticket's link asks the board's own search for that number
 * (GET /api/v1/tasks?search=, which matches a ticket's number, services/ticket_numbers.title_or_number)
 * and opens the ticket whose number it is. A mission step has no number of its own, so a step's ref opens
 * the step itself when the answer holds it, else its mission card. A ref the app has no page for is quoted
 * as text.
 *
 * The refs are data the check-in read: the words here are the app's own, drawn as text, never as markup.
 */
import { useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'
import { boardNumberOf, cardNumberOf, evidenceHref, evidenceLabel } from '@/lib/proactive/evidence'
import { ticketHref } from '@/lib/ticket-links'
import { ticketNumber, type Numbered } from '@/lib/ticket-number'
import type { EvidenceRef } from '@/lib/proactive/types'
import { cn } from '@/lib/utils'
import { TEXT_BUTTON_CLASS } from './card-parts'

export const EVIDENCE_LIST_LABEL = 'What Auto read'
const TASKS_PATH = '/api/v1/tasks'
// The board's search also windows Done and Cancelled tickets to the newest 200 (hooks/use-board-tasks.ts).
const FINISHED_LIMIT = '200'
const LINK_CLASS = 'text-xs font-medium text-foreground underline underline-offset-4 hover:text-muted-foreground'

type Listed = Numbered & { id?: unknown }

function isListed(task: unknown): task is Listed {
  return typeof task === 'object' && task !== null
}

/**
 * The row id of the ticket whose board number is `number` (as boardNumberOf writes it): the ticket itself, else
 * a step's mission card. Null when the board has neither.
 */
export async function ticketIdForNumber(number: string): Promise<string | null> {
  const card = cardNumberOf(number)
  const search = new URLSearchParams({ search: card.replace(/^#/, ''), finished_limit: FINISHED_LIMIT })
  const answer = await apiClient.request<{ tasks?: unknown }>(`${TASKS_PATH}?${search.toString()}`)
  const tasks = (Array.isArray(answer?.tasks) ? answer.tasks : []).filter(isListed)
  const found = tasks.find((task) => ticketNumber(task) === number) ?? tasks.find((task) => ticketNumber(task) === card)
  return found && (typeof found.id === 'number' || typeof found.id === 'string') ? String(found.id) : null
}

/** A ticket cited by its number: pressed, it finds the ticket on the board and opens it there. */
function TicketNumberLink({ number, label }: { readonly number: string; readonly label: string }) {
  const router = useRouter()
  const [finding, setFinding] = useState(false)
  const open = async () => {
    setFinding(true)
    try {
      const id = await ticketIdForNumber(number)
      if (id) router.push(ticketHref(id) as never)
      else toast.error(`Ticket ${number} isn't on the board any more.`)
    } catch (error) {
      console.error('[proactive] finding a cited ticket on the board failed:', error)
      toast.error(`Ticket ${number} couldn't be opened. Try again in a moment.`)
    } finally {
      setFinding(false)
    }
  }
  return (
    <button type="button" aria-busy={finding} disabled={finding} onClick={() => void open()} className={cn(TEXT_BUTTON_CLASS, 'text-xs underline')}>
      {label}
    </button>
  )
}

function EvidenceLink({ evidence }: { readonly evidence: EvidenceRef }) {
  const label = evidenceLabel(evidence)
  const number = boardNumberOf(evidence)
  if (number) return <TicketNumberLink number={number} label={label} />
  const href = evidenceHref(evidence)
  if (!href) return <span className="text-xs text-muted-foreground">{label}</span>
  return (
    <Link href={href as never} className={LINK_CLASS}>
      {label}
    </Link>
  )
}

/** The refs a suggestion cites, each opening what it names; nothing when it cites none. */
export function EvidenceLinks({ evidence }: { readonly evidence: readonly EvidenceRef[] }) {
  if (evidence.length === 0) return null
  return (
    <ul aria-label={EVIDENCE_LIST_LABEL} className="flex flex-wrap items-center gap-x-3 gap-y-1">
      {evidence.map((item, index) => (
        <li key={`${item.type}:${item.id}:${index}`}>
          <EvidenceLink evidence={item} />
        </li>
      ))}
    </ul>
  )
}
