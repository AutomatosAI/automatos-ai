'use client'

/**
 * IsItWorkingStrip — PRD-142 Wave 0 (US-007). A second cc-stats strip,
 * rendered under StatsStrip in the Command Centre, answering "is the
 * platform working?" from the real Wave 0 analytics endpoints.
 *
 * Mirrors StatsStrip's idiom exactly: label + mono value + tone + delta.
 * Honest zero/pending states — a cell shows "—" rather than fabricate a
 * signal. PRIMITIVES reads the W3-S2 /primitive-health endpoint (8 canonical
 * primitives); a primitive with no heartbeat finding renders as "awaiting
 * checks" rather than a fake green.
 *
 * PRD-185 S12: the whole strip is now reachable by a workspace admin (not just
 * the super-admin) via the own-workspace analytics tier. ACTIVATION is
 * workspace-scoped ("has THIS workspace reached first value?"), and two tiles
 * were added — SLOs (the 3 tracked PRD-180 objectives) and DELIVERABLES
 * (freshness of the last output, the signal that would have caught the 06-16
 * silent stop on day one).
 *
 * PRD-189 S2: COMMERCE — cross-sell persistence integrity. Shown only for
 * workspaces with a commerce sync history; reads reported-vs-present
 * frequently-bought-together edges (the "16 reported / 0 present" query that
 * would have caught the FBT wipe on day one).
 */

import type { SparklineTone } from './sparkline'
import {
  useWorkspaceActivation,
  useMissionSuccessRate,
  useErrorsBySubsystem,
  useWidgetEngagement,
  usePrimitiveHealth,
  useSLOs,
  useDeliverableFreshness,
  useCommerceIntegrity,
} from '@/hooks/use-analytics-api'

interface Cell {
  label: string
  value: string
  tone: SparklineTone
  delta: string
}

export function IsItWorkingStrip() {
  const { data: activation } = useWorkspaceActivation()
  const { data: mission } = useMissionSuccessRate()
  const { data: errors } = useErrorsBySubsystem('24h')
  const { data: widget } = useWidgetEngagement('7d')
  const { data: slos } = useSLOs('24h')
  const { data: fresh } = useDeliverableFreshness()
  const { data: commerce } = useCommerceIntegrity()

  const missionTotal = mission?.total_executions ?? 0
  const missionPct = missionTotal > 0 ? Math.round(mission?.value ?? 0) : 0

  const errorTotal = errors?.total ?? 0
  const worst = errors?.by_subsystem?.length
    ? [...errors.by_subsystem].sort((a, b) => b.count - a.count)[0]
    : null

  const sessions = widget?.sessions ?? 0
  const widgetEvents = widget?.by_event_type?.reduce((sum, ev) => sum + ev.count, 0) ?? 0

  const { data: health } = usePrimitiveHealth()
  const prim = health?.primitives ?? []
  const primTotal = prim.length
  const primKnown = prim.filter((p) => p.status !== 'unknown').length
  const primGreen = prim.filter((p) => p.status === 'green').length
  const primDegraded = prim.filter((p) => p.status === 'degraded').length
  const primDown = prim.filter((p) => p.status === 'down').length

  // SLOs (PRD-180) — met vs breached over the window; null meets_target = no data.
  const sloList = slos?.slos ?? []
  const sloTotal = sloList.length
  const sloMeasured = sloList.filter((s) => s.meets_target !== null).length
  const sloMet = sloList.filter((s) => s.meets_target === true).length
  const sloBreached = sloList.filter((s) => s.meets_target === false)

  // Deliverable freshness — the "is Auto still producing?" signal (S12).
  const freshTotal = fresh?.total ?? 0
  const freshAge = fresh?.age_seconds ?? null
  const freshHours = freshAge != null ? Math.floor(freshAge / 3600) : null
  const freshLabel =
    freshHours == null
      ? '—'
      : freshHours < 1
        ? '<1h'
        : freshHours < 48
          ? `${freshHours}h`
          : `${Math.floor(freshHours / 24)}d`

  const cells: Cell[] = [
    {
      label: 'ACTIVATION',
      value: activation && activation.completed_missions > 0 ? String(activation.completed_missions) : '—',
      tone: activation?.activated ? 'ok' : 'muted',
      delta: activation?.activated ? 'reached first value' : 'awaiting first value',
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
      label: 'SLOs',
      value: sloMeasured > 0 ? `${sloMet}/${sloTotal}` : '—',
      tone: sloBreached.length > 0 ? 'err' : sloMeasured > 0 ? 'ok' : 'muted',
      delta:
        sloMeasured === 0
          ? 'awaiting data'
          : sloBreached.length > 0
            ? `${sloBreached[0].sli.replace(/_/g, ' ')} off target`
            : 'all met · 24h',
    },
    {
      label: 'DELIVERABLES',
      value: freshTotal > 0 ? freshLabel : '—',
      tone: freshTotal === 0 ? 'muted' : freshHours != null && freshHours >= 48 ? 'warn' : 'ok',
      delta: freshTotal > 0 ? `${freshTotal} total · last output` : 'none produced yet',
    },
    // COMMERCE only exists for workspaces with a commerce sync history —
    // a workspace that never synced a catalog/orders keeps a 7-cell strip.
    ...(commerce?.synced
      ? [
          {
            label: 'COMMERCE',
            value:
              commerce.present_fbt_edges != null
                ? String(commerce.present_fbt_edges)
                : '—',
            tone: (commerce.ok === false
              ? 'err'
              : (commerce.present_fbt_edges ?? 0) > 0
                ? 'ok'
                : 'muted') as SparklineTone,
            delta:
              commerce.ok === false
                ? `cross-sell: ${commerce.present_fbt_edges} live · last computed ${commerce.reported_fbt_edges}`
                : commerce.reported_fbt_edges != null
                  ? `cross-sell: ${commerce.present_fbt_edges} pairs live · in sync`
                  : 'no orders sync yet',
          },
        ]
      : []),
    {
      label: 'WIDGET',
      value: String(sessions),
      tone: sessions > 0 ? 'info' : 'muted',
      delta: sessions > 0 ? `${widgetEvents} events · 7d` : '— · 7d',
    },
    {
      label: 'PRIMITIVES',
      value: primKnown > 0 ? `${primGreen}/${primTotal}` : '—',
      tone: primDown > 0 ? 'err' : primDegraded > 0 ? 'warn' : primKnown > 0 ? 'ok' : 'muted',
      delta:
        primKnown === 0
          ? 'awaiting checks'
          : primDown > 0
            ? `${primDown} down`
            : primDegraded > 0
              ? `${primDegraded} degraded`
              : 'all green',
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
