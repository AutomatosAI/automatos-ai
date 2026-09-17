'use client'

/**
 * StatsStrip — CD round-4 STATS-A: a six-cell editorial strip with
 * sparklines, mono numerals and semantic tones. The cells:
 *   WORKING · AGENTS · QUEUE · ATTENTION · CACHE-HIT · $ / REQ
 *
 * Live values come from `useActivityStats('1d')` for the workforce four,
 * `useCostAnalyticsUnified(7)` for CACHE-HIT (the share of input tokens the
 * providers served from prompt cache, PRD-231/#745) and $/REQ, and
 * `useHeartbeats()` for the scheduled count. The strip always renders — no
 * auto-swap to prose. Idle cells show their zero state honestly ("idle",
 * "0 high", "0 open", "no cached reads yet").
 */

import { useHeartbeats } from '@/hooks/use-heartbeats-api'
import { useActivityStats } from '@/hooks/use-activity-api'
import { useCostAnalyticsUnified } from '@/hooks/use-unified-analytics'
import { Sparkline, type SparklineTone } from './sparkline'

interface Cell {
  label: string
  value: string
  tone: SparklineTone
  delta: string
  spark?: number[]
}

export function StatsStrip() {
  const { data: stats } = useActivityStats('1d')
  const { data: cost } = useCostAnalyticsUnified(7)
  const { data: heartbeats } = useHeartbeats()

  const working = stats?.working_now ?? 0
  const agents = stats?.agents_active ?? 0
  const queue = stats?.tasks_in_queue ?? 0
  const attn = stats?.needs_attention ?? 0

  const scheduled = heartbeats?.heartbeats?.filter((h) => h.enabled).length ?? 0

  const costPerReq = (() => {
    const c = cost?.summary?.totalCost ?? 0
    const r = cost?.summary?.totalRequests ?? 0
    if (!r || !c) return '—'
    const v = c / r
    return `$${v.toFixed(4)}`
  })()

  const cacheShare = cost?.summary?.cacheShare ?? 0
  const cacheReadTokens = cost?.summary?.cacheReadTokens ?? 0

  const costSpark = cost?.costTrend
    ? cost.costTrend
        .slice(-10)
        .map((p: { date: string; total_cost: number }) => p.total_cost)
    : undefined

  const cells: Cell[] = [
    {
      label: 'WORKING',
      value: String(working),
      tone: working > 0 ? 'ok' : 'muted',
      delta: working > 0 ? 'running now' : 'idle',
    },
    {
      label: 'AGENTS',
      value: String(agents),
      tone: 'info',
      delta: `${scheduled} scheduled`,
    },
    {
      label: 'QUEUE',
      value: String(queue),
      tone: queue > 0 ? 'warn' : 'muted',
      delta: queue > 0 ? `${queue} in queue` : '0 high',
    },
    {
      label: 'ATTENTION',
      value: String(attn),
      tone: attn > 0 ? 'err' : 'muted',
      delta: attn > 0 ? 'needs you' : '0 open',
    },
    {
      label: 'CACHE-HIT',
      value: cacheShare > 0 ? `${Math.round(cacheShare * 100)}%` : cost?.summary?.totalRequests ? '0%' : '—',
      tone: cacheShare > 0 ? 'ok' : 'muted',
      delta: cacheShare > 0 ? `${formatTokens(cacheReadTokens)} cached · 7d` : cost?.summary?.totalRequests ? 'no cached reads · 7d' : 'no requests · 7d',
    },
    {
      label: '$ / REQ',
      value: costPerReq,
      tone: 'ok',
      delta: cost?.summary?.totalRequests
        ? `${cost.summary.totalRequests} req · 7d`
        : '— req',
      spark: costSpark && costSpark.length > 1 ? costSpark : undefined,
    },
  ]

  return (
    <div className="cc-stats">
      {cells.map((c) => (
        <div key={c.label} className="cell">
          <div className="l">
            <Dot tone={c.tone} />
            {c.label}
          </div>
          <div className={`v ${c.tone === 'muted' ? '' : c.tone}`}>{c.value}</div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
            <span className="delta" style={{ flex: 1, minWidth: 0 }}>
              {c.delta}
            </span>
            {c.spark && (
              <div className="spark" style={{ width: 60 }}>
                <Sparkline data={c.spark} tone={c.tone === 'muted' ? 'info' : c.tone} />
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M tok`
  if (n >= 1_000) return `${Math.round(n / 1_000)}k tok`
  return `${n} tok`
}

function Dot({ tone }: { tone: SparklineTone }) {
  if (tone === 'muted') return null
  const colors: Record<SparklineTone, string> = {
    ok: 'hsl(var(--olive))',
    err: 'hsl(var(--accent))',
    warn: 'hsl(var(--warn-ink))',
    info: 'hsl(var(--info))',
    muted: 'transparent',
  }
  return (
    <span
      style={{
        width: 6,
        height: 6,
        borderRadius: '50%',
        background: colors[tone],
        display: 'inline-block',
      }}
    />
  )
}

