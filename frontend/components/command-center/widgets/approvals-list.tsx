'use client'

/**
 * ApprovalsList — pending missions awaiting human approval.
 * Live data via `useApprovalGates('30d')`. Approve/Skip wired to existing
 * mission mutations.
 */

import { useRouter } from 'next/navigation'
import { useApprovalGates } from '@/hooks/use-kpi-api'
import { useApproveMission } from '@/hooks/use-missions-api'
import { formatDistanceToNowStrict } from 'date-fns'
import { toast } from 'sonner'

function waitingLabel(since: string | null): string {
  if (!since) return ''
  try {
    return `Waiting ${formatDistanceToNowStrict(new Date(since))}`
  } catch {
    return ''
  }
}

export function ApprovalsList({ limit = 4 }: { limit?: number }) {
  const router = useRouter()
  const { data, isLoading } = useApprovalGates('30d')
  const approve = useApproveMission()
  const items = data?.pending_missions?.slice(0, limit) ?? []

  if (isLoading) {
    return <div className="cc-panel-empty">Loading approval gates…</div>
  }

  if (items.length === 0) {
    return (
      <div className="cc-panel-empty">
        No pending approvals. Missions queued for human review will appear here.
      </div>
    )
  }

  const handleApprove = async (id: string) => {
    try {
      await approve.mutateAsync({ id })
      toast.success('Mission approved')
    } catch (err: any) {
      toast.error(err?.message ?? 'Failed to approve mission')
    }
  }

  return (
    <div>
      {items.map((m) => (
        <div className="cc-appr-row" key={m.id}>
          <div style={{ minWidth: 0 }}>
            <div className="ttl">{m.goal}</div>
            <div className="meta">awaiting your call</div>
          </div>
          <div className="meta" style={{ textAlign: 'right' }}>
            {waitingLabel(m.waiting_since ?? m.created_at)}
          </div>
          <div className="cc-appr-actions">
            <button
              type="button"
              className="cc-btn"
              style={{ height: 26, fontSize: 11.5, padding: '0 8px' }}
              onClick={() => router.push(`/missions/${m.id}` as any)}
            >
              Open
            </button>
            <button
              type="button"
              className="cc-btn primary"
              style={{ height: 26, fontSize: 11.5, padding: '0 8px' }}
              disabled={approve.isLoading}
              onClick={() => handleApprove(m.id)}
            >
              Approve
            </button>
          </div>
        </div>
      ))}
    </div>
  )
}
