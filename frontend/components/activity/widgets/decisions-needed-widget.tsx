'use client'

import { AlertTriangle, FileText, Target, Loader2, CheckCircle2 } from 'lucide-react'
import { useDecisionsNeeded } from '@/hooks/use-kpi-api'
import { cn } from '@/lib/utils'

interface DecisionsNeededWidgetProps {
  className?: string
  onItemClick?: (kind: 'report' | 'mission', id: string) => void
}

const LEVEL_LABELS: Record<number, string> = {
  4: 'L4 SECURITY',
  3: 'L3 URGENT',
  2: 'L2 APPROVAL',
  1: 'L1 TASK',
  0: 'L0 FYI',
}

const LEVEL_TONES: Record<number, string> = {
  4: 'bg-red-500/15 text-red-500 border-red-500/30',
  3: 'bg-warning/15 text-warning border-warning/30',
  2: 'bg-amber-500/15 text-amber-500 border-amber-500/30',
  1: 'bg-muted text-muted-foreground border-border',
  0: 'bg-muted/50 text-muted-foreground border-border',
}

function formatAge(iso: string | null): string {
  if (!iso) return ''
  const seconds = Math.floor((Date.now() - new Date(iso).getTime()) / 1000)
  if (seconds < 60) return `${seconds}s`
  const mins = Math.floor(seconds / 60)
  if (mins < 60) return `${mins}m`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}h`
  return `${Math.floor(hours / 24)}d`
}

export function DecisionsNeededWidget({ className, onItemClick }: DecisionsNeededWidgetProps) {
  const { data, isLoading } = useDecisionsNeeded(10)

  const total = data?.total ?? 0
  const reportsCount = data?.reports_count ?? 0
  const missionsCount = data?.missions_count ?? 0
  const items = data?.items ?? []

  return (
    <div className={cn('h-full flex flex-col', className)}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <div className="flex items-center gap-2">
          <AlertTriangle className={cn('w-4 h-4', total > 0 ? 'text-warning' : 'text-success')} />
          <h3 className="text-sm font-semibold">Decisions Needed</h3>
        </div>
        {!isLoading && total > 0 && (
          <span className="text-xs bg-warning/15 text-warning px-1.5 py-0.5 rounded-full font-medium">
            {total} pending
          </span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : total === 0 ? (
          <div className="flex flex-col items-center justify-center py-6 text-muted-foreground">
            <CheckCircle2 className="w-8 h-8 mb-2 opacity-30" />
            <p className="text-xs">Nothing on your plate. Grab a tea.</p>
          </div>
        ) : (
          <>
            {/* Stats row */}
            <div className="flex items-center gap-4">
              <div className="text-center flex-1">
                <span className="text-2xl font-bold text-warning">{total}</span>
                <p className="text-[10px] text-muted-foreground">Awaiting</p>
              </div>
              <div className="w-px h-8 bg-border/50" />
              <div className="text-center flex-1">
                <span className="text-2xl font-bold">{reportsCount}</span>
                <p className="text-[10px] text-muted-foreground">Reports</p>
              </div>
              <div className="w-px h-8 bg-border/50" />
              <div className="text-center flex-1">
                <span className="text-2xl font-bold">{missionsCount}</span>
                <p className="text-[10px] text-muted-foreground">Missions</p>
              </div>
            </div>

            {/* Items list */}
            <div className="space-y-2">
              <p className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">
                Highest priority first
              </p>
              {items.map((item) => {
                const level = item.escalation_level ?? 0
                const Icon = item.kind === 'report' ? FileText : Target
                return (
                  <button
                    key={`${item.kind}:${item.id}`}
                    type="button"
                    onClick={() => onItemClick?.(item.kind, item.id)}
                    className={cn(
                      'w-full text-left p-2 rounded-lg border space-y-1 transition-colors',
                      'bg-muted/30 hover:bg-muted/60 border-border/50',
                      onItemClick ? 'cursor-pointer' : 'cursor-default',
                    )}
                  >
                    <div className="flex items-start gap-2">
                      <Icon className="w-3 h-3 text-muted-foreground mt-0.5 shrink-0" />
                      <span className="text-xs leading-snug line-clamp-2 flex-1">{item.title}</span>
                      <span
                        className={cn(
                          'text-[9px] px-1.5 py-0.5 rounded-full border font-medium shrink-0',
                          LEVEL_TONES[level] || LEVEL_TONES[0],
                        )}
                      >
                        {LEVEL_LABELS[level] || 'L0'}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 pl-5 text-[10px] text-muted-foreground">
                      <span className="capitalize">{item.kind}</span>
                      {item.agent_name && (
                        <>
                          <span>·</span>
                          <span>{item.agent_name}</span>
                        </>
                      )}
                      {item.created_at && (
                        <>
                          <span>·</span>
                          <span>{formatAge(item.created_at)} ago</span>
                        </>
                      )}
                    </div>
                  </button>
                )
              })}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
