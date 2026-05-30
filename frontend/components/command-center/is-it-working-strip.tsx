'use client'

/**
 * IsItWorkingStrip — PRD-142 Wave 0 (US-007). A second cc-stats strip,
 * rendered under StatsStrip in the Command Centre, answering "is the
 * platform working?" from the real Wave 0 analytics endpoints.
 *
 * Mirrors StatsStrip's idiom exactly: label + mono value + tone + delta.
 * Honest zero/pending states — the PRIMITIVES cell shows "—" / "metric
 * pending" (like StatsStrip's CACHE-HIT) rather than fabricate a health
 * signal; per-primitive health (US-006) lands in Wave 3.
 */

import type { SparklineTone } from './sparkline'
import {
  useActivationMetrics,
  useMissionSuccessRate,
  useErrorsBySubsystem,
  useWidgetEngagement,
} from '@/hooks/use-analytics-api'

interface Cell {
  label: string
  value: string
  tone: SparklineTone
  delta: string
}

export function IsItWorkingStrip() {
  const { data: activation } = useActivationMetrics()
  const { data: mission } = useMissionSuccessRate()
  const { data: errors } = useErrorsBySubsystem('24h')
  const { data: widget } = useWidgetEngagement('7d')

  const totalWs = activation?.total_workspaces ?? 0
  const activationPct = totalWs > 0 ? Math.round((activation?.rate ?? 0) * 100) : 0

  const missionTotal = mission?.total_executions ?? 0
  const missionPct = missionTotal > 0 ? Math.round(mission?.value ?? 0) : 0

  const errorTotal = errors?.total ?? 0
  const worst = errors?.by_subsystem?.length
    ? [...errors.by_subsystem].sort((a, b) => b.count - a.count)[0]
    : null

  const sessions = widget?.sessions ?? 0
  const widgetEvents = widget?.by_event_type?.reduce((sum, ev) => sum + ev.count, 0) ?? 0

  const cells: Cell[] = [
    {
      label: 'ACTIVATION',
      value: totalWs > 0 ? `${activationPct}%` : '—',
      tone: totalWs > 0 ? 'ok' : 'muted',
      delta: totalWs > 0 ? `${activation?.activated ?? 0}/${totalWs} workspaces` : 'no workspaces',
    },
    {
      label: 'MISSIONS',
      value: missionTotal > 0 ? `${missionPct}%` : '—',
      tone: missionTotal > 0 ? 'ok' : 'muted',
      delta: missionTotal > 0
        ? `${mission?.successful_executions ?? 0}/${missionTotal} ok`
        : 'no missions yet',
    },
    {
      label: 'ERRORS',
      value: String(errorTotal),
      tone: errorTotal > 0 ? 'err' : 'ok',
      delta: errorTotal > 0 && worst ? `${worst.subsystem} (${worst.count}) · 24h` : 'none · 24h',
    },
    {
      label: 'WIDGET',
      value: String(sessions),
      tone: sessions > 0 ? 'info' : 'muted',
      delta: sessions > 0 ? `${widgetEvents} events · 7d` : '— · 7d',
    },
    {
      label: 'PRIMITIVES',
      value: '—',
      tone: 'muted',
      delta: 'metric pending',
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
          </div>
        </div>
      ))}
    </div>
  )
}

function Dot({ tone }: { tone: SparklineTone }) {
  if (tone === 'muted') return null
  const colors: Record<SparklineTone, string> = {
    ok: 'hsl(82 50% 22%)',
    err: 'hsl(var(--accent))',
    warn: 'hsl(38 78% 27%)',
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
