'use client'

import { ShieldCheck, Loader2, Clock, AlertCircle } from 'lucide-react'
import { useApprovalGates } from '@/hooks/use-kpi-api'
import { cn } from '@/lib/utils'

interface ApprovalGatesWidgetProps {
  period: string
  className?: string
}

function formatWait(iso: string | null): string {
  if (!iso) return ''
  const seconds = Math.floor((Date.now() - new Date(iso).getTime()) / 1000)
  if (seconds < 60) return `${seconds}s`
  const mins = Math.floor(seconds / 60)
  if (mins < 60) return `${mins}m`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}h`
  return `${Math.floor(hours / 24)}d`
}

function formatSeconds(sec: number): string {
  if (sec < 60) return `${Math.round(sec)}s`
  const mins = Math.floor(sec / 60)
  if (mins < 60) return `${mins}m`
  return `${Math.floor(mins / 60)}h ${mins % 60}m`
}

export function ApprovalGatesWidget({ period, className }: ApprovalGatesWidgetProps) {
  const { data, isLoading } = useApprovalGates(period)

  const pendingCount = data?.pending_count ?? 0
  const pendingMissions = data?.pending_missions ?? []
  const avgApproval = data?.avg_approval_seconds ?? 0

  return (
    <div className={cn('h-full flex flex-col', className)}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-amber-400" />
          <h3 className="text-sm font-semibold">Approval Gates</h3>
        </div>
        {!isLoading && pendingCount > 0 && (
          <span className="text-xs bg-amber-500/15 text-amber-400 px-1.5 py-0.5 rounded-full font-medium">
            {pendingCount} pending
          </span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <>
            {/* Stats row */}
            <div className="flex items-center gap-4">
              <div className="text-center flex-1">
                <span className={cn(
                  'text-2xl font-bold',
                  pendingCount > 0 ? 'text-amber-400' : 'text-emerald-400'
                )}>
                  {pendingCount}
                </span>
                <p className="text-[10px] text-muted-foreground">Awaiting</p>
              </div>
              <div className="w-px h-8 bg-border/50" />
              <div className="text-center flex-1">
                <span className="text-2xl font-bold">{formatSeconds(avgApproval)}</span>
                <p className="text-[10px] text-muted-foreground">Avg approval</p>
              </div>
            </div>

            {/* Pending missions list */}
            {pendingMissions.length > 0 ? (
              <div className="space-y-1.5">
                <p className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">
                  Waiting for approval
                </p>
                {pendingMissions.map((mission) => (
                  <div
                    key={mission.id}
                    className="p-2 rounded-lg bg-amber-500/5 border border-amber-500/10 space-y-1"
                  >
                    <div className="flex items-start gap-2">
                      <AlertCircle className="w-3 h-3 text-amber-400 mt-0.5 shrink-0" />
                      <span className="text-xs leading-snug line-clamp-2">{mission.goal}</span>
                    </div>
                    {mission.waiting_since && (
                      <div className="flex items-center gap-1 pl-5">
                        <Clock className="w-2.5 h-2.5 text-muted-foreground" />
                        <span className="text-[10px] text-muted-foreground">
                          Waiting {formatWait(mission.waiting_since)}
                        </span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-4 text-muted-foreground">
                <ShieldCheck className="w-8 h-8 mb-2 opacity-30" />
                <p className="text-xs">All clear — no pending approvals</p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
