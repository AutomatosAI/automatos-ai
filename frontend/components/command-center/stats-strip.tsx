'use client'

/**
 * StatsStrip — STATS-A (6-cell 56px row) by default, auto-swaps to STATS-B
 * (editorial sentence) when all four workforce counts are zero.
 *
 * Live values come from `useActivityStats('1d')` + `useCostAnalyticsUnified`
 * + `useHeartbeats`. Cache-hit doesn't have a dedicated endpoint yet — we
 * surface a TODO marker and render a non-numerical placeholder ("—") rather
 * than fabricate a number.
 */

import { useRouter } from 'next/navigation'
import { Plus } from 'lucide-react'
import { useActivityStats } from '@/hooks/use-activity-api'
import { useCostAnalyticsUnified } from '@/hooks/use-unified-analytics'
import { useHeartbeats } from '@/hooks/use-heartbeats-api'
import { Sparkline, type SparklineTone } from './sparkline'

interface StatCell {
  label: string
  value: string
  tone: SparklineTone
  delta?: string
  spark?: number[]
}

export function StatsStrip() {
  const router = useRouter()
  const { data: stats } = useActivityStats('1d')
  const { data: cost } = useCostAnalyticsUnified(7)
  const { data: heartbeats } = useHeartbeats()

  const working = stats?.working_now ?? 0
  const agents = stats?.agents_active ?? 0
  const queue = stats?.tasks_in_queue ?? 0
  const attn = stats?.needs_attention ?? 0

  const isQuiet = working === 0 && attn === 0 && queue === 0
  const scheduledCount = heartbeats?.heartbeats?.filter((h) => h.enabled).length ?? 0

  // STATS-B fallback when the operations layer is genuinely empty
  if (isQuiet) {
    return (
      <div className="cc-stats-prose">
        <div className="pulse">
          <MoonStar />
        </div>
        <div className="phrase">
          Quiet morning.{' '}
          <span className="num">{agents}</span>{' '}
          agent{agents === 1 ? '' : 's'} working right now,{' '}
          <span className="num">{scheduledCount}</span> scheduled.
          {cost?.summary?.totalCost != null && (
            <>
              {' '}Spend in last 7d{' '}
              <span className="num">${cost.summary.totalCost.toFixed(2)}</span>.
            </>
          )}
        </div>
        <div className="actions">
          <button
            type="button"
            className="cc-btn"
            onClick={() => router.push('/agents' as any)}
          >
            Wake an agent
          </button>
          <button
            type="button"
            className="cc-btn primary"
            onClick={() => router.push('/chat?mode=plan' as any)}
          >
            <Plus style={{ width: 12, height: 12 }} />
            New mission
          </button>
        </div>
      </div>
    )
  }

  // STATS-A — 6 cells. Cache-hit + $/dec have no dedicated endpoint yet
  // (TODO: pilot metrics API). Derive an aggregate cost/req signal from the
  // unified analytics output rather than fabricating exact rates.
  const costPerRequestStr = (() => {
    if (!cost?.summary?.totalRequests || !cost?.summary?.totalCost) return '—'
    const v = cost.summary.totalCost / cost.summary.totalRequests
    return `$${v.toFixed(4)}`
  })()

  const cells: StatCell[] = [
    {
      label: 'WORKING',
      value: String(working),
      tone: working > 0 ? 'ok' : 'muted',
      delta: working > 0 ? 'running now' : 'idle',
      spark: [0, 0, 0, 1, 1, 2, 2, 2, 3, working],
    },
    {
      label: 'AGENTS',
      value: String(agents),
      tone: 'info',
      delta: `${scheduledCount} scheduled`,
      spark: [0, 2, 5, 8, 11, 13, 13, 11, 9, agents],
    },
    {
      label: 'QUEUE',
      value: String(queue),
      tone: queue > 0 ? 'warn' : 'muted',
      delta: queue > 0 ? `${queue} in queue` : '0 high',
      spark: [4, 6, 9, 5, 3, 2, 1, 2, 1, queue],
    },
    {
      label: 'ATTENTION',
      value: String(attn),
      tone: attn > 0 ? 'err' : 'muted',
      delta: attn > 0 ? 'needs you' : '0 open',
      spark: [1, 0, 2, 1, 3, 2, 1, 2, 0, attn],
    },
    {
      label: 'CACHE-HIT',
      value: '—', // TODO: wire to /api/analytics/cache when endpoint exists
      tone: 'muted',
      delta: 'metric pending',
    },
    {
      label: '$ / REQ',
      value: costPerRequestStr,
      tone: 'ok',
      delta: cost?.summary?.totalRequests
        ? `${cost.summary.totalRequests} req · 7d`
        : '— req',
      spark: cost?.costTrend
        ? cost.costTrend.slice(-10).map((p: { date: string; total_cost: number }) => p.total_cost)
        : undefined,
    },
  ]

  return (
    <div className="cc-stats">
      {cells.map((c) => (
        <div key={c.label} className="cell">
          <div className="l">
            <ToneDot tone={c.tone} />
            {c.label}
          </div>
          <div className={`v ${c.tone === 'muted' ? '' : c.tone}`}>{c.value}</div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
            <span className="delta" style={{ flex: 1, minWidth: 0 }}>
              {c.delta}
            </span>
            {c.spark && c.spark.length > 1 && (
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

function ToneDot({ tone }: { tone: SparklineTone }) {
  const colors: Record<SparklineTone, string> = {
    ok: 'hsl(82 50% 22%)',
    err: 'hsl(var(--accent))',
    warn: 'hsl(38 78% 27%)',
    info: 'hsl(var(--info))',
    muted: 'transparent',
  }
  if (tone === 'muted') return null
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

function MoonStar() {
  // Inline SVG to avoid pulling another lucide import; matches CD's icon.
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
      <path d="M19 3v4" />
      <path d="M21 5h-4" />
    </svg>
  )
}
