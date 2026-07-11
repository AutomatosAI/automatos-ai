'use client'

/**
 * AuditPane — PRD-196 S3 (P2-15, governance I.2). The plane's window: a
 * filterable, paginated view of this workspace's audit log (ws-admin-gated,
 * ctx-scoped server-side). Non-empty today from GDPR + grant rows; the policy
 * verdict stream joins it when PRD-192 flips the flag.
 */

import { useState } from 'react'
import { useGovernanceAuditLog } from '@/hooks/use-governance'
import type { AuditLogRow } from '@/lib/api-client'

const PAGE = 25

const ACTION_PREFIXES = [
  { value: '', label: 'All actions' },
  { value: 'policy:', label: 'Policy verdicts' },
  { value: 'gdpr:', label: 'GDPR' },
  { value: 'approval_grant:', label: 'Approvals' },
  { value: 'audit:', label: 'Retention' },
]

const ACTOR_TYPES = [
  { value: '', label: 'Any actor' },
  { value: 'user', label: 'User' },
  { value: 'agent', label: 'Agent' },
  { value: 'system', label: 'System' },
]

function when(iso: string | null): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime())
    ? '—'
    : d.toLocaleString('en-GB', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })
}

function detailSummary(row: AuditLogRow): string {
  const d = row.details || {}
  return String(d.reason ?? d.verdict ?? d.scope ?? d.status ?? '')
}

export function AuditPane() {
  const [actionPrefix, setActionPrefix] = useState('')
  const [actorType, setActorType] = useState('')
  const [offset, setOffset] = useState(0)

  const { data, isLoading, isError } = useGovernanceAuditLog({
    action_prefix: actionPrefix || undefined,
    actor_type: actorType || undefined,
    limit: PAGE,
    offset,
  })

  const rows = data?.rows ?? []
  const total = data?.total ?? 0

  const onFilterChange = (setter: (v: string) => void) => (v: string) => {
    setter(v)
    setOffset(0)
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <select
          aria-label="Filter by action"
          value={actionPrefix}
          onChange={(e) => onFilterChange(setActionPrefix)(e.target.value)}
          className="rounded border border-border bg-background px-2 py-1 text-xs"
        >
          {ACTION_PREFIXES.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
        <select
          aria-label="Filter by actor type"
          value={actorType}
          onChange={(e) => onFilterChange(setActorType)(e.target.value)}
          className="rounded border border-border bg-background px-2 py-1 text-xs"
        >
          {ACTOR_TYPES.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
        <span className="ml-auto text-xs text-muted-foreground">{total} total</span>
      </div>

      {isLoading ? (
        <p className="text-sm text-muted-foreground">Loading audit log…</p>
      ) : isError ? (
        <p className="text-sm text-muted-foreground">Could not load the audit log. Workspace admin only.</p>
      ) : rows.length === 0 ? (
        <p className="text-sm text-muted-foreground">No audit entries match. As Auto acts, entries appear here.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="text-muted-foreground">
              <tr>
                <th className="py-1 pr-3 font-medium">Time</th>
                <th className="py-1 pr-3 font-medium">Actor</th>
                <th className="py-1 pr-3 font-medium">Action</th>
                <th className="py-1 pr-3 font-medium">Resource</th>
                <th className="py-1 font-medium">Detail</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id} className="border-t border-border/50 align-top">
                  <td className="py-1 pr-3 whitespace-nowrap text-muted-foreground">{when(r.created_at)}</td>
                  <td className="py-1 pr-3 whitespace-nowrap">{r.actor_type}</td>
                  <td className="py-1 pr-3 font-mono">{r.action}</td>
                  <td className="py-1 pr-3">
                    {r.resource_type}
                    {r.resource_id ? `:${r.resource_id}` : ''}
                  </td>
                  <td className="py-1 text-muted-foreground" title={JSON.stringify(r.details)}>
                    {detailSummary(r)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {total > PAGE && (
        <div className="flex items-center gap-2 text-xs">
          <button
            type="button"
            className="cc-btn"
            disabled={offset === 0}
            onClick={() => setOffset(Math.max(0, offset - PAGE))}
          >
            Prev
          </button>
          <span className="text-muted-foreground">
            {offset + 1}–{Math.min(offset + PAGE, total)} of {total}
          </span>
          <button
            type="button"
            className="cc-btn"
            disabled={offset + PAGE >= total}
            onClick={() => setOffset(offset + PAGE)}
          >
            Next
          </button>
        </div>
      )}
    </div>
  )
}
