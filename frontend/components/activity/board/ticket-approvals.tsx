'use client'

/**
 * PRD-252 R1 — a ticket's pending approvals, inside the ticket.
 *
 * An approval row in Needs you opens the ticket at its approval
 * (`&question=<grant id>`, the one link format), and the card there is the one
 * the Governance inbox shows, with Grant and Deny. Before, a ticket blocked on
 * an approval said only that it was waiting, and the buttons lived on another
 * tab. Approvals are a workspace admin's to give: anyone else sees nothing here.
 */

import { useEffect, useRef } from 'react'
import { ShieldCheck } from 'lucide-react'
import { useApprovalGrants } from '@/hooks/use-approval-grants'
import { GrantCard } from '@/components/command-center/governance/approvals-inbox'
import { questionTicketId } from '@/lib/ticket-links'
import { cn } from '@/lib/utils'

interface TicketApprovalsProps {
  taskId: string
  focusGrantId?: number | null
}

export function TicketApprovals({ taskId, focusGrantId }: TicketApprovalsProps) {
  const { data } = useApprovalGrants('pending', 'approval')
  const grants = (data?.grants ?? []).filter((g) => questionTicketId(g) === taskId)
  const focused = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    focused.current?.scrollIntoView?.({ block: 'nearest' })
  }, [focusGrantId, grants.length])

  if (grants.length === 0) return null
  return (
    <section className="space-y-2 mb-6" aria-label="Approvals on this ticket" data-testid="ticket-approvals">
      <div className="flex items-center gap-1.5 text-muted-foreground">
        <ShieldCheck className="w-3 h-3" />
        <h4 className="text-xs font-semibold uppercase tracking-wider">
          {grants.length === 1 ? 'Waiting for your approval' : `Waiting for ${grants.length} approvals`}
        </h4>
      </div>
      {grants.map((g) => (
        <div
          key={g.id}
          ref={g.id === focusGrantId ? focused : undefined}
          data-focused={g.id === focusGrantId || undefined}
          className={cn('rounded', g.id === focusGrantId && 'ring-2 ring-[hsl(var(--warning))]/60')}
        >
          <GrantCard grant={g} />
        </div>
      ))}
    </section>
  )
}
