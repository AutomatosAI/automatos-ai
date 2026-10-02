'use client'

/**
 * PRD-252 R7 in the ticket's viewer: Assign and Cancel under the ticket's
 * header, and, on a cancelled or closed ticket, who stopped it and when (it
 * opened to an empty viewer).
 */

import Link from 'next/link'
import { formatDistanceToNow } from 'date-fns'
import { Target, XCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { missionHref } from '@/lib/ticket-links'
import type { BoardTask } from '@/types/board'
import { canAssign, canCancel, isMissionTicket, whoStopped } from './ticket-actions'
import { useAgentChoices, useTicketActions } from './ticket-actions-menu'

export function TicketActionsBar({ task }: { task: BoardTask }) {
  const agents = useAgentChoices()
  const actions = useTicketActions(task)
  if (isMissionTicket(task)) {
    return task.mission_id ? (
      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground" data-testid="ticket-actions">
        <span>The mission runs this ticket.</span>
        <Link href={missionHref(task.mission_id) as any} className="inline-flex items-center gap-1 text-foreground underline-offset-2 hover:underline">
          <Target className="w-3 h-3" /> Open the mission
        </Link>
      </div>
    ) : null
  }
  if (!canAssign(task) && !canCancel(task)) return null
  return (
    <div className="flex flex-wrap items-center gap-2" data-testid="ticket-actions">
      {canAssign(task) && (
        <select
          aria-label="Assign to an agent"
          className="h-8 rounded-md border border-border bg-background px-2 text-xs"
          value=""
          disabled={actions.busy}
          onChange={(e) => {
            const agent = agents.find((a) => String(a.id) === e.target.value)
            if (agent) actions.assignTo(agent)
          }}
        >
          <option value="">{task.assignee ? 'Reassign to…' : 'Assign to…'}</option>
          {agents.map((a) => (
            <option key={a.id} value={a.id}>{a.name}</option>
          ))}
        </select>
      )}
      {canCancel(task) && (
        <Button size="sm" variant="outline" className="h-8 text-xs" disabled={actions.busy} onClick={actions.cancelIt}>
          <XCircle className="w-3.5 h-3.5 mr-1.5" /> Cancel ticket
        </Button>
      )}
    </div>
  )
}

/** Who cancelled (or closed) the ticket, when, and why. */
export function CancelledBanner({ task }: { task: BoardTask }) {
  const stop = whoStopped(task)
  if (!stop) return null
  const when = stop.at ? ` ${formatDistanceToNow(new Date(stop.at), { addSuffix: true })}` : ''
  return (
    <div className="flex items-start gap-3 px-4 py-3 rounded-lg bg-secondary/40 border border-border/40" data-testid="cancelled-banner">
      <XCircle className="w-5 h-5 text-muted-foreground shrink-0 mt-0.5" />
      <div className="space-y-0.5">
        <p className="text-sm font-medium">
          {stop.closed ? 'Closed' : 'Cancelled'}{stop.by ? ` by ${stop.by}` : ''}{when}
        </p>
        {stop.reason && <p className="text-xs text-muted-foreground">{stop.reason}</p>}
      </div>
    </div>
  )
}
