'use client'

/**
 * ApprovalsInbox — PRD-196 S1 (P2-15, governance I.1).
 *
 * The front door for durable approval grants: pending grants first (subject,
 * tool, risk tier, Art.14 oversight rationale, estimated cost, requested/expires),
 * decided grants below with a status filter. Grant / Deny / Revoke call the
 * ws-admin-gated API (PRD-196 S2) and optimistically refresh on decision.
 *
 * Reuses the MissionApprovalWidget oversight presentation (tier label + the
 * amber Art.14 note) rather than forking a rival card.
 */

import { useMemo, useState } from 'react'
import { ShieldAlert, Check, X, Ban, Clock } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { toast } from 'sonner'
import { oversightTierLabel } from '@/components/widgets/MissionApprovalWidget'
import {
  useApprovalGrants,
  useGrantApproval,
  useDenyApproval,
  useRevokeApproval,
} from '@/hooks/use-approval-grants'
import type { ApprovalGrant } from '@/lib/api-client'

const PENDING = 'pending'
const DECIDED_STATUSES = ['granted', 'denied', 'revoked', 'expired'] as const

function whenLabel(iso: string | null): string {
  if (!iso) return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return '—'
  return d.toLocaleString('en-GB', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })
}

function statusTone(status: string): string {
  switch (status) {
    case 'granted':
      return 'hsl(82 30% 33%)'
    case 'denied':
    case 'expired':
      return 'hsl(0 62% 38%)'
    case 'revoked':
      return 'hsl(38 78% 27%)'
    default:
      return 'hsl(213 51% 45%)'
  }
}

function GrantCard({ grant }: { grant: ApprovalGrant }) {
  const grantMut = useGrantApproval()
  const denyMut = useDenyApproval()
  const revokeMut = useRevokeApproval()
  const busy = grantMut.isLoading || denyMut.isLoading || revokeMut.isLoading

  const act = async (
    mut: { mutateAsync: (id: number) => Promise<unknown> },
    verb: string,
  ) => {
    try {
      await mut.mutateAsync(grant.id)
      toast.success(`Grant ${grant.id} ${verb}`)
    } catch {
      toast.error(`Failed to ${verb.replace(/ed$/, '')} grant ${grant.id}`)
    }
  }

  const oversight = grant.oversight
  const cost = grant.estimated_cost_usd

  return (
    <div className="flex flex-col gap-2 rounded border border-border bg-background/50 p-3">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="text-sm font-medium truncate">
            {grant.tool_name || grant.subject_type}
            <span className="text-xs text-muted-foreground"> · {grant.subject_type}:{grant.subject_id}</span>
          </p>
          <p className="text-xs text-muted-foreground mt-0.5">
            Requested {whenLabel(grant.requested_at)}
            {grant.expires_at && <> · expires {whenLabel(grant.expires_at)}</>}
            {cost && <> · est. ${cost}</>}
          </p>
        </div>
        <span
          className="shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide"
          style={{ color: statusTone(grant.status), border: `1px solid ${statusTone(grant.status)}` }}
        >
          {grant.status}
        </span>
      </div>

      {/* Art.14 oversight note — reused presentation from MissionApprovalWidget */}
      {oversight && (
        <div
          className="flex items-start gap-2 rounded border border-amber-300 bg-amber-50 px-2 py-1.5 dark:border-amber-800 dark:bg-amber-950/40"
          role="note"
          aria-label="Human oversight"
        >
          <ShieldAlert className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-600" />
          <div className="min-w-0">
            <p className="text-[11px] font-medium text-amber-800 dark:text-amber-300">
              {oversightTierLabel(oversight.tier)}
              {oversight.risk_class ? ` · ${oversight.risk_class.replace(/_/g, ' ')}` : ''}
            </p>
            {oversight.rationale && (
              <p className="mt-0.5 text-[10px] text-amber-700 dark:text-amber-400">{oversight.rationale}</p>
            )}
          </div>
        </div>
      )}

      {grant.reason && <p className="text-xs text-muted-foreground">{grant.reason}</p>}

      {grant.status === PENDING && (
        <div className="flex gap-2">
          <Button size="sm" disabled={busy} onClick={() => act(grantMut, 'granted')} className="flex-1">
            <Check className="h-4 w-4 mr-1" /> Grant
          </Button>
          <Button size="sm" variant="outline" disabled={busy} onClick={() => act(denyMut, 'denied')}>
            <X className="h-4 w-4 mr-1" /> Deny
          </Button>
        </div>
      )}
      {grant.status === 'granted' && (
        <div className="flex gap-2">
          <Button size="sm" variant="outline" disabled={busy} onClick={() => act(revokeMut, 'revoked')}>
            <Ban className="h-4 w-4 mr-1" /> Revoke
          </Button>
        </div>
      )}
    </div>
  )
}

export function ApprovalsInbox() {
  const { data, isLoading, isError } = useApprovalGrants()
  const [decidedFilter, setDecidedFilter] = useState<string>('all')

  const { pending, decided } = useMemo(() => {
    const grants = data?.grants ?? []
    const p = grants.filter((g) => g.status === PENDING)
    const d = grants.filter((g) => g.status !== PENDING)
    return { pending: p, decided: d }
  }, [data])

  const decidedShown = useMemo(
    () => (decidedFilter === 'all' ? decided : decided.filter((g) => g.status === decidedFilter)),
    [decided, decidedFilter],
  )

  if (isLoading) {
    return <p className="text-sm text-muted-foreground p-3">Loading approvals…</p>
  }
  if (isError) {
    return <p className="text-sm text-muted-foreground p-3">Could not load approvals. You may not be a workspace admin.</p>
  }

  return (
    <div className="flex flex-col gap-4">
      <section>
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">
          Pending {pending.length > 0 && <span className="text-foreground">({pending.length})</span>}
        </h3>
        {pending.length === 0 ? (
          <p className="flex items-center gap-2 text-sm text-muted-foreground">
            <Clock className="h-4 w-4" /> No approvals pending. Auto is running within its guardrails.
          </p>
        ) : (
          <div className="flex flex-col gap-2">
            {pending.map((g) => (
              <GrantCard key={g.id} grant={g} />
            ))}
          </div>
        )}
      </section>

      <section>
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Decided</h3>
          <select
            aria-label="Filter decided grants by status"
            value={decidedFilter}
            onChange={(e) => setDecidedFilter(e.target.value)}
            className="rounded border border-border bg-background px-2 py-0.5 text-xs"
          >
            <option value="all">All</option>
            {DECIDED_STATUSES.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
        {decidedShown.length === 0 ? (
          <p className="text-sm text-muted-foreground">No decided grants{decidedFilter !== 'all' ? ` (${decidedFilter})` : ''}.</p>
        ) : (
          <div className="flex flex-col gap-2">
            {decidedShown.map((g) => (
              <GrantCard key={g.id} grant={g} />
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
