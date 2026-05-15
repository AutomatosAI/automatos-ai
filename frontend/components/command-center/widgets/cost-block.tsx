'use client'

/**
 * CostBlock — 1d total + delta + sparkline + top spenders.
 * Live data via `useCostTracker('1d')` and `useCostAnalyticsUnified(7)`
 * for the cost trend sparkline.
 */

import { useMemo } from 'react'
import { useCostTracker } from '@/hooks/use-kpi-api'
import { useCostAnalyticsUnified } from '@/hooks/use-unified-analytics'
import { Sparkline } from '../sparkline'
import { toneFor } from '../agent-tones'

function fmtCurrency(n: number | null | undefined, digits: number = 2): string {
  if (typeof n !== 'number' || !isFinite(n)) return '$0.00'
  if (Math.abs(n) >= 1) return `$${n.toFixed(digits)}`
  return `$${n.toFixed(4)}`
}

export function CostBlock() {
  const { data: tracker, isLoading: trackerLoading } = useCostTracker('1d')
  const { data: unified } = useCostAnalyticsUnified(7)

  const trendData = useMemo(() => {
    if (tracker?.daily_trend && tracker.daily_trend.length > 1) {
      return tracker.daily_trend.map((p) => p.cost)
    }
    if (unified?.costTrend && unified.costTrend.length > 1) {
      return unified.costTrend.map((p: { date: string; total_cost: number }) => p.total_cost)
    }
    return null
  }, [tracker, unified])

  if (trackerLoading) {
    return <div className="cc-panel-empty">Loading cost data…</div>
  }

  const totalCost = tracker?.total_cost ?? 0
  const delta = tracker?.change_pct ?? 0
  const top = tracker?.top_agents?.slice(0, 4) ?? []

  return (
    <div className="cc-cost">
      <div className="total">
        <span className="v">{fmtCurrency(totalCost)}</span>
        {delta !== 0 && (
          <span className={`delta${delta > 0 ? ' up' : ''}`}>
            {delta > 0 ? '↑' : '↓'} {Math.abs(delta).toFixed(1)}% vs 7d avg
          </span>
        )}
      </div>

      {trendData && (
        <div className="spark-row">
          <Sparkline data={trendData} tone={delta > 0 ? 'err' : 'ok'} height={36} />
        </div>
      )}

      <p className="cc-eyebrow-sm" style={{ marginTop: 4 }}>
        Top spenders · today
      </p>

      {top.length === 0 ? (
        <div
          style={{
            fontFamily: 'var(--font-newsreader, serif)',
            fontSize: 12,
            color: 'hsl(var(--muted-foreground))',
            padding: '4px 0',
          }}
        >
          No spend recorded yet.
        </div>
      ) : (
        <div className="top">
          {top.map((row) => {
            const tone = toneFor(row.name)
            return (
              <div key={row.name} className="row">
                <span
                  aria-hidden
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: 2,
                    background: tone.bg,
                    display: 'inline-block',
                  }}
                />
                <span className="nm">{row.name}</span>
                <span className="v">{fmtCurrency(row.cost)}</span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
