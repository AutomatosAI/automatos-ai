'use client'

import { Brain, Loader2, Sparkles, Zap } from 'lucide-react'
import { useSelfLearningHealth } from '@/hooks/use-learning-api'
import { cn } from '@/lib/utils'

interface SelfLearningHealthWidgetProps {
  className?: string
}

// HARNESS loop status → tone. The weekly loop reports completed / running /
// idle; a best-effort section failure on the backend surfaces as 'error'.
const STATUS_TONES: Record<string, string> = {
  completed: 'text-success',
  running: 'text-[hsl(var(--info))]',
  in_progress: 'text-[hsl(var(--info))]',
  idle: 'text-muted-foreground',
  failed: 'text-warning',
  error: 'text-warning',
}

function statusTone(status?: string): string {
  return (status && STATUS_TONES[status]) || 'text-muted-foreground'
}

/**
 * Self-Learning health tile (PRD-142 Wave 4, W4-S16 frontend half).
 *
 * Renders the three self-learning signals from GET /api/harness/self-learning:
 * HARNESS loop status + iteration, the tool-routing recorder's signal count, and
 * a prescription applied/queued summary. Each backend section is best-effort —
 * a section that failed to load arrives as `{ error }` and is shown as
 * "unavailable" rather than breaking the tile.
 */
export function SelfLearningHealthWidget({ className }: SelfLearningHealthWidgetProps) {
  const { data, isLoading, isError } = useSelfLearningHealth()

  const harness = data?.harness
  const toolRouting = data?.tool_routing
  const prescriptions = data?.prescriptions

  const harnessError = !!harness && 'error' in harness
  const routingError = !!toolRouting && 'error' in toolRouting
  const rxError = !!prescriptions && 'error' in prescriptions

  const status = harnessError ? 'error' : harness?.status
  const iteration = harnessError ? undefined : harness?.iteration
  const dropped = toolRouting?.dropped ?? 0

  const appliedDisplay = rxError ? '—' : (prescriptions?.applied ?? 0)
  const queuedDisplay = rxError ? '—' : (prescriptions?.queued ?? 0)
  const signalsDisplay = routingError ? '—' : (toolRouting?.recorded ?? 0)

  return (
    <div className={cn('h-full flex flex-col', className)}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <div className="flex items-center gap-2">
          <Brain className={cn('w-4 h-4', statusTone(status))} />
          <h3 className="text-sm font-semibold">Self-Learning</h3>
        </div>
        {!isLoading && !isError && status && (
          <span
            className={cn(
              'text-xs px-1.5 py-0.5 rounded-full font-medium capitalize',
              harnessError ? 'bg-warning/15 text-warning' : 'bg-secondary/60 text-muted-foreground',
            )}
          >
            {harnessError ? 'degraded' : status.replace('_', ' ')}
          </span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : isError ? (
          <div className="flex flex-col items-center justify-center py-6 text-muted-foreground">
            <Brain className="w-8 h-8 mb-2 opacity-30" />
            <p className="text-xs">Self-learning health unavailable</p>
          </div>
        ) : (
          <>
            {/* KPI stats — prescriptions applied / queued + tool-routing signals */}
            <div className="flex items-center gap-4">
              <div className="text-center flex-1">
                <span className="text-2xl font-bold text-success">{appliedDisplay}</span>
                <p className="text-[10px] text-muted-foreground">Applied</p>
              </div>
              <div className="w-px h-8 bg-border/50" />
              <div className="text-center flex-1">
                <span className="text-2xl font-bold text-warning">{queuedDisplay}</span>
                <p className="text-[10px] text-muted-foreground">Queued</p>
              </div>
              <div className="w-px h-8 bg-border/50" />
              <div className="text-center flex-1">
                <span className="text-2xl font-bold">{signalsDisplay}</span>
                <p className="text-[10px] text-muted-foreground">Signals</p>
              </div>
            </div>

            {/* Detail lines — HARNESS loop + tool-routing recorder */}
            <div className="space-y-1.5 text-[11px] text-muted-foreground">
              <div className="flex items-center gap-2">
                <Sparkles className="w-3 h-3 shrink-0" />
                <span>
                  HARNESS{' '}
                  {harnessError ? (
                    <span className="text-warning">section unavailable</span>
                  ) : (
                    <>
                      <span className={statusTone(status)}>{status ?? 'idle'}</span>
                      {typeof iteration === 'number' && <> · iteration {iteration}</>}
                    </>
                  )}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <Zap className="w-3 h-3 shrink-0" />
                <span>
                  Tool routing{' '}
                  {routingError ? (
                    <span className="text-warning">section unavailable</span>
                  ) : (
                    <>
                      {toolRouting?.recorded ?? 0} signals recorded
                      {dropped > 0 && <span className="text-warning"> · {dropped} dropped</span>}
                    </>
                  )}
                </span>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
