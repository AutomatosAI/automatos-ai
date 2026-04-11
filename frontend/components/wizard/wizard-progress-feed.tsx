'use client'

/**
 * PRD-130: Wizard Progress Feed
 * ==============================
 *
 * Terminal-styled live log for Step 5 of the wizard. Renders every
 * progress event emitted by the backend pipeline — scan, scrape, ingest,
 * graphify, profile — as coloured rows, auto-scrolling to the latest.
 *
 * Presentation only: the data comes from `useWizardProgress()`.
 */

import { useEffect, useRef } from 'react'
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  FileText,
  Loader2,
  Network,
  Sparkles,
  XCircle,
} from 'lucide-react'
import type {
  WizardProgressEvent,
  WizardProgressState,
  WizardStage,
} from '@/hooks/use-wizard-progress'

interface WizardProgressFeedProps {
  events: WizardProgressEvent[]
  state: WizardProgressState
  pageCount: number
}

const STAGE_META: Record<WizardStage, { label: string; icon: typeof FileText; accent: string }> = {
  scan: { label: 'Scan', icon: Network, accent: 'text-sky-400' },
  scrape: { label: 'Scrape', icon: FileText, accent: 'text-violet-400' },
  ingest: { label: 'Ingest', icon: Activity, accent: 'text-cyan-400' },
  graphify: { label: 'Graphify', icon: Sparkles, accent: 'text-fuchsia-400' },
  profile: { label: 'Profile', icon: CheckCircle2, accent: 'text-emerald-400' },
  plan: { label: 'Plan', icon: Sparkles, accent: 'text-amber-400' },
  complete: { label: 'Complete', icon: CheckCircle2, accent: 'text-emerald-400' },
  failed: { label: 'Failed', icon: XCircle, accent: 'text-red-400' },
}

const STAGE_ORDER: WizardStage[] = [
  'scan',
  'scrape',
  'ingest',
  'graphify',
  'profile',
]

export function WizardProgressFeed({
  events,
  state,
  pageCount,
}: WizardProgressFeedProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [events.length])

  // Derive which stages have been reached
  const reachedStages = new Set<WizardStage>(events.map((e) => e.stage))
  const currentStage = events.length > 0 ? events[events.length - 1].stage : null

  return (
    <div className="space-y-4">
      {/* Stage pill row */}
      <div className="flex flex-wrap items-center gap-2">
        {STAGE_ORDER.map((stage) => {
          const meta = STAGE_META[stage]
          const Icon = meta.icon
          const reached = reachedStages.has(stage)
          const active = currentStage === stage && state === 'streaming'
          const done =
            reached &&
            !active &&
            (currentStage === null ||
              STAGE_ORDER.indexOf(currentStage as WizardStage) > STAGE_ORDER.indexOf(stage) ||
              state === 'complete')
          return (
            <div
              key={stage}
              className={[
                'flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-medium transition-colors',
                active
                  ? 'border-primary bg-primary/10 text-primary'
                  : done
                  ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
                  : reached
                  ? 'border-border bg-secondary/50 text-foreground'
                  : 'border-border/40 bg-secondary/20 text-muted-foreground',
              ].join(' ')}
            >
              {active ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Icon className={`w-3.5 h-3.5 ${done ? 'text-emerald-400' : meta.accent}`} />
              )}
              <span>{meta.label}</span>
            </div>
          )
        })}
      </div>

      {/* Terminal log */}
      <div className="rounded-lg border border-border/50 bg-black/80 shadow-inner overflow-hidden">
        <div className="flex items-center justify-between px-3 py-2 border-b border-border/40 bg-black/40">
          <div className="flex items-center gap-2">
            <div className="flex gap-1.5">
              <div className="w-2.5 h-2.5 rounded-full bg-red-500/70" />
              <div className="w-2.5 h-2.5 rounded-full bg-amber-500/70" />
              <div className="w-2.5 h-2.5 rounded-full bg-emerald-500/70" />
            </div>
            <div className="text-[10px] uppercase tracking-wider text-muted-foreground font-mono">
              automatos ~ wizard intake ({pageCount} pages)
            </div>
          </div>
          <StateBadge state={state} />
        </div>

        <div
          ref={scrollRef}
          className="h-72 overflow-y-auto p-3 font-mono text-xs leading-relaxed space-y-0.5"
        >
          {events.length === 0 ? (
            <div className="text-muted-foreground flex items-center gap-2">
              <Loader2 className="w-3 h-3 animate-spin" />
              waiting for first event…
            </div>
          ) : (
            events.map((event, idx) => (
              <EventLine key={`${event.ts}-${idx}`} event={event} />
            ))
          )}
        </div>
      </div>
    </div>
  )
}

function EventLine({ event }: { event: WizardProgressEvent }) {
  const meta = STAGE_META[event.stage]
  const colour =
    event.level === 'error'
      ? 'text-red-400'
      : event.level === 'warn'
      ? 'text-amber-300'
      : meta?.accent || 'text-foreground'

  const time = new Date(event.ts * 1000).toLocaleTimeString('en-GB', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })

  return (
    <div className="flex gap-2">
      <span className="text-muted-foreground/70 shrink-0">{time}</span>
      <span className={`${colour} shrink-0 uppercase text-[10px] tracking-wider w-16`}>
        {event.stage}
      </span>
      <span
        className={[
          'break-all',
          event.level === 'error' && 'text-red-300',
          event.level === 'warn' && 'text-amber-200',
          event.level === 'info' && 'text-foreground/90',
        ]
          .filter(Boolean)
          .join(' ')}
      >
        {event.message}
      </span>
    </div>
  )
}

function StateBadge({ state }: { state: WizardProgressState }) {
  if (state === 'streaming') {
    return (
      <div className="flex items-center gap-1.5 text-[10px] font-mono text-primary">
        <Loader2 className="w-3 h-3 animate-spin" />
        STREAMING
      </div>
    )
  }
  if (state === 'complete') {
    return (
      <div className="flex items-center gap-1.5 text-[10px] font-mono text-emerald-400">
        <CheckCircle2 className="w-3 h-3" />
        COMPLETE
      </div>
    )
  }
  if (state === 'failed' || state === 'error') {
    return (
      <div className="flex items-center gap-1.5 text-[10px] font-mono text-red-400">
        <AlertTriangle className="w-3 h-3" />
        {state === 'failed' ? 'FAILED' : 'STREAM ERROR'}
      </div>
    )
  }
  return (
    <div className="text-[10px] font-mono text-muted-foreground">IDLE</div>
  )
}
