'use client'

/**
 * AttentionList — "Needs your eyes" panel for the Summary tab.
 * Composed from `useDecisionsNeeded()` which already merges:
 *   - failed missions
 *   - stale approval gates
 *   - flagged reports (escalation_level > 0)
 *
 * Empty state is editorial — no fabricated items.
 */

import { useRouter } from 'next/navigation'
import { XCircle, AlertCircle, AlertTriangle } from 'lucide-react'
import { useDecisionsNeeded } from '@/hooks/use-kpi-api'
import { formatDistanceToNowStrict } from 'date-fns'

const TONE_MAP = {
  err: { Icon: XCircle, cls: 'err' as const },
  warn: { Icon: AlertTriangle, cls: 'warn' as const },
  info: { Icon: AlertCircle, cls: 'info' as const },
}

function escalationTone(level: number | null): keyof typeof TONE_MAP {
  if (level == null) return 'info'
  if (level >= 2) return 'err'
  if (level >= 1) return 'warn'
  return 'info'
}

function ageLabel(createdAt: string | null): string {
  if (!createdAt) return ''
  try {
    return formatDistanceToNowStrict(new Date(createdAt), { addSuffix: false })
  } catch {
    return ''
  }
}

export function AttentionList({ limit = 4 }: { limit?: number }) {
  const router = useRouter()
  const { data, isLoading } = useDecisionsNeeded(limit)
  const items = data?.items ?? []

  if (isLoading) {
    return <div className="cc-panel-empty">Loading…</div>
  }

  if (items.length === 0) {
    return (
      <div className="cc-panel-empty">
        Nothing flagged right now. Failed runs and stale approvals will surface here.
      </div>
    )
  }

  return (
    <div className="cc-attn">
      {items.slice(0, limit).map((it) => {
        const tone = escalationTone(it.escalation_level)
        const { Icon, cls } = TONE_MAP[tone]
        const age = ageLabel(it.created_at)
        const route =
          it.kind === 'mission'
            ? `/missions/${it.id}`
            : `/command-center?tab=activity&report=${it.id}`
        return (
          <div className="cc-attn-row" key={it.id}>
            <span className={`cc-attn-icn ${cls}`}>
              <Icon style={{ width: 14, height: 14, strokeWidth: 1.8 }} />
            </span>
            <div style={{ minWidth: 0 }}>
              <div className="ttl">{it.title}</div>
              <div className="sub">
                {it.kind === 'mission' ? 'mission' : 'report'}
                {it.agent_name ? ` · ${it.agent_name}` : ''}
                {age ? ` · ${age}` : ''}
                {it.status ? ` · ${it.status}` : ''}
              </div>
            </div>
            <button
              type="button"
              className="cc-btn"
              style={{ height: 26, fontSize: 11.5, padding: '0 10px' }}
              onClick={() => router.push(route as any)}
            >
              {it.kind === 'mission' ? 'Open mission' : 'Review'}
            </button>
          </div>
        )
      })}
    </div>
  )
}
