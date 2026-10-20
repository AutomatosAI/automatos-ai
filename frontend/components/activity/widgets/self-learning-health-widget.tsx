'use client'

import { Fragment } from 'react'
import { Brain, Gauge, Loader2, Sparkles, Zap } from 'lucide-react'
import {
  useSelfLearningHealth,
  type SelfLearningHealthData,
  type SelfLearningSection,
} from '@/hooks/use-learning-api'
import { cn } from '@/lib/utils'

interface SelfLearningHealthWidgetProps {
  className?: string
}

/** What the tile shows for a figure the backend does not have: never a fake 0. */
const ABSENT = '—'

// HARNESS status → tone, from the theme tokens, which follow the theme (the dark
// variant does nothing in this Tailwind build). A status not listed here reads muted.
const STATUS_TONES: Record<string, string> = {
  completed: 'text-success',
  running: 'text-info',
  failed: 'text-destructive',
  dormant_insufficient_agents: 'text-warning',
  scheduled_not_run_yet: 'text-muted-foreground',
  disabled: 'text-muted-foreground',
}

function statusTone(status?: string | null): string {
  return (status && STATUS_TONES[status]) || 'text-muted-foreground'
}

// The workspace's HARNESS mode as the owner reads it (PRD-263 section 8). Propose
// shows as itself: the backend reads a stored propose as propose.
const MODE_LABELS: Record<string, string> = {
  report_only: 'Report only',
  propose: 'Propose',
  disabled: 'Disabled',
}

// PRD-263 US-058: what became of HARNESS's proposals, its Approvals cards, as the
// ledger counts them. The backend calls a declined card `denied`.
const CARD_COUNTS = [
  { field: 'pending', label: 'Pending', tone: 'text-info' },
  { field: 'applied', label: 'Applied', tone: 'text-success' },
  { field: 'denied', label: 'Declined', tone: 'text-muted-foreground' },
  { field: 'expired', label: 'Expired', tone: 'text-warning' },
] as const

/** A status as words: "scheduled_not_run_yet" → "scheduled not run yet". */
function statusWords(status?: string | null): string {
  return status ? status.replace(/_/g, ' ') : ABSENT
}

/** Whether the backend named this section as one it could not read. */
function sectionDown(data: SelfLearningHealthData, section: SelfLearningSection): boolean {
  return data.unavailable?.includes(section) ?? false
}

function modeLabel(mode?: string | null): string {
  return mode ? MODE_LABELS[mode] ?? statusWords(mode) : ABSENT
}

function figure(value?: number | null): string {
  return typeof value === 'number' ? String(value) : ABSENT
}

function percent(value?: number | null): string {
  return typeof value === 'number' ? `${Math.round(value * 100)}%` : ABSENT
}

function Kpi({ value, label, tone }: { value: string; label: string; tone?: string }) {
  return (
    <div className="text-center flex-1 min-w-0">
      <span className={cn('text-2xl font-bold', value !== ABSENT && tone)}>{value}</span>
      <p className="text-[10px] text-muted-foreground">{label}</p>
    </div>
  )
}

function Divider() {
  return <div className="w-px h-8 bg-border/50" />
}

/** HARNESS's Approvals cards: pending, applied, declined and expired, "—" for any absent. */
function CardCounts({ data }: { data: SelfLearningHealthData }) {
  return (
    <div className="space-y-1">
      <p className="text-[10px] text-muted-foreground">Approvals cards</p>
      <div className="flex items-center gap-3">
        {CARD_COUNTS.map(({ field, label, tone }, index) => (
          <Fragment key={field}>
            {index > 0 && <Divider />}
            <Kpi value={figure(data[field])} label={label} tone={tone} />
          </Fragment>
        ))}
      </div>
    </div>
  )
}

function SelfLearningFigures({ data }: { data: SelfLearningHealthData }) {
  const harnessDown = sectionDown(data, 'harness')
  const mode = harnessDown ? ABSENT : modeLabel(data.mode)

  return (
    <>
      {/* KPI stats: coverage, the WATCH and REGRESSION counts, this workspace's 7-day tool calls */}
      <div className="flex items-center gap-3">
        <Kpi value={percent(data.coverage)} label="Coverage" />
        <Divider />
        <Kpi value={figure(data.watch)} label="Watch" tone="text-warning" />
        <Divider />
        <Kpi value={figure(data.regression)} label="Regression" tone="text-destructive" />
        <Divider />
        <Kpi value={figure(data.signals_7d)} label="Signals (7 days)" />
      </div>

      <CardCounts data={data} />

      {/* Detail lines: the HARNESS run and mode, and what Signals counts */}
      <div className="space-y-1.5 text-[11px] text-muted-foreground">
        <div className="flex items-center gap-2">
          <Sparkles className="w-3 h-3 shrink-0" />
          {harnessDown ? (
            <span className="text-warning">HARNESS section unavailable</span>
          ) : (
            <>
              <span>{`Iteration ${figure(data.iteration_count)}`}</span>
              <span>{`· Mode ${mode}`}</span>
            </>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Gauge className="w-3 h-3 shrink-0" />
          <span>{`Agents with not enough data: ${figure(data.insufficient_data)}`}</span>
        </div>
        <div className="flex items-center gap-2">
          <Zap className="w-3 h-3 shrink-0" />
          <span>Signals: this workspace&apos;s tool calls in the last 7 days</span>
        </div>
      </div>
    </>
  )
}

/**
 * Self-Learning health tile (PRD-142 Wave 4, W4-S16; PRD-263 US-022, US-058).
 *
 * Renders GET /api/harness/self-learning: the latest HARNESS run's status,
 * iteration, coverage and WATCH / REGRESSION / not-enough-data counts, the
 * workspace's HARNESS mode, this workspace's production tool calls over the
 * last 7 days (Signals), and how many of HARNESS's Approvals cards are pending,
 * applied, declined and expired (US-058). A figure the backend does not have
 * shows "—", never 0; a backend section that failed to load is named in
 * `unavailable` and shown as "unavailable" rather than breaking the tile.
 */
export function SelfLearningHealthWidget({ className }: SelfLearningHealthWidgetProps) {
  const { data, isLoading, isError } = useSelfLearningHealth()
  const harnessDown = !!data && sectionDown(data, 'harness')
  const status = harnessDown ? undefined : data?.status

  return (
    <div className={cn('h-full flex flex-col', className)}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <div className="flex items-center gap-2">
          <Brain className={cn('w-4 h-4', statusTone(status))} />
          <h3 className="text-sm font-semibold">Self-Learning</h3>
        </div>
        {!isLoading && !isError && data && (
          <span
            className={cn(
              'text-xs px-1.5 py-0.5 rounded-full font-medium',
              harnessDown ? 'bg-warning/15 text-warning' : cn('bg-secondary/60', statusTone(status)),
            )}
          >
            {harnessDown ? 'degraded' : statusWords(status)}
          </span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : isError || !data ? (
          <div className="flex flex-col items-center justify-center py-6 text-muted-foreground">
            <Brain className="w-8 h-8 mb-2 opacity-30" />
            <p className="text-xs">Self-learning health unavailable</p>
          </div>
        ) : (
          <SelfLearningFigures data={data} />
        )}
      </div>
    </div>
  )
}
