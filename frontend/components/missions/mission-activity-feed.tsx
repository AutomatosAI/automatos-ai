'use client'

import { useMemo } from 'react'
import {
  Play, CheckCircle2, XCircle, ShieldCheck, UserPlus, Search,
  Eye, Pause, Trophy, AlertTriangle, RefreshCw, SkipForward,
  Zap, Clock,
} from 'lucide-react'
import { ScrollArea } from '@/components/ui/scroll-area'
import { cn } from '@/lib/utils'
import type { EventResponse } from '@/types/missions'
import { formatDistanceToNow } from 'date-fns'

interface MissionActivityFeedProps {
  events: EventResponse[]
  className?: string
}

const EVENT_CONFIG: Record<string, { icon: typeof Play; color: string; label: string }> = {
  run_created: { icon: Zap, color: 'text-primary', label: 'Mission created' },
  run_planning_started: { icon: Search, color: 'text-[hsl(var(--info))]', label: 'Planning started' },
  run_plan_ready: { icon: CheckCircle2, color: 'text-[hsl(var(--success))]', label: 'Plan ready for review' },
  run_approved: { icon: CheckCircle2, color: 'text-[hsl(var(--success))]', label: 'Plan approved' },
  run_rejected: { icon: XCircle, color: 'text-destructive', label: 'Plan rejected' },
  run_started: { icon: Play, color: 'text-primary', label: 'Mission started' },
  run_paused: { icon: Pause, color: 'text-muted-foreground', label: 'Mission paused' },
  run_resumed: { icon: Play, color: 'text-primary', label: 'Mission resumed' },
  run_completed: { icon: Trophy, color: 'text-[hsl(var(--success))]', label: 'Mission completed' },
  run_failed: { icon: XCircle, color: 'text-destructive', label: 'Mission failed' },
  run_cancelled: { icon: XCircle, color: 'text-muted-foreground', label: 'Mission cancelled' },
  run_verifying: { icon: Search, color: 'text-[hsl(var(--info))]', label: 'Final verification' },
  run_awaiting_human: { icon: Eye, color: 'text-[hsl(var(--warning))]', label: 'Awaiting human review' },

  task_queued: { icon: Clock, color: 'text-muted-foreground', label: 'Task queued' },
  task_assigned: { icon: UserPlus, color: 'text-[hsl(var(--info))]', label: 'Agent assigned' },
  task_started: { icon: Play, color: 'text-primary', label: 'Task started' },
  task_output_submitted: { icon: CheckCircle2, color: 'text-[hsl(var(--info))]', label: 'Output submitted' },
  task_verification_started: { icon: Search, color: 'text-[hsl(var(--info))]', label: 'Verification started' },
  task_verification_passed: { icon: ShieldCheck, color: 'text-[hsl(var(--success))]', label: 'Verification passed' },
  task_verification_failed: { icon: AlertTriangle, color: 'text-[hsl(var(--warning))]', label: 'Verification failed' },
  task_human_review_requested: { icon: Eye, color: 'text-[hsl(var(--warning))]', label: 'Human review requested' },
  task_human_approved: { icon: CheckCircle2, color: 'text-[hsl(var(--success))]', label: 'Task approved by human' },
  task_human_rejected: { icon: XCircle, color: 'text-[hsl(var(--warning))]', label: 'Task rejected by human' },
  task_retrying: { icon: RefreshCw, color: 'text-[hsl(var(--warning))]', label: 'Task retrying' },
  task_failed: { icon: XCircle, color: 'text-destructive', label: 'Task failed' },
  task_skipped: { icon: SkipForward, color: 'text-muted-foreground', label: 'Task skipped' },
  task_stalled: { icon: AlertTriangle, color: 'text-[hsl(var(--warning))]', label: 'Task stalled' },

  stall_detected: { icon: AlertTriangle, color: 'text-[hsl(var(--warning))]', label: 'Stall detected' },
  run_stall_ledger: { icon: AlertTriangle, color: 'text-[hsl(var(--warning))]', label: 'Stall ledger verdict' },
  run_replanning: { icon: RefreshCw, color: 'text-[hsl(var(--warning))]', label: 'Replanning' },
  run_replanned: { icon: RefreshCw, color: 'text-[hsl(var(--info))]', label: 'Mission replanned' },
  model_fallback: { icon: RefreshCw, color: 'text-[hsl(var(--info))]', label: 'Model fallback' },
}

const DEFAULT_CONFIG = { icon: Zap, color: 'text-muted-foreground', label: 'Event' }

export function MissionActivityFeed({ events, className }: MissionActivityFeedProps) {
  const sortedEvents = useMemo(
    () => [...events].sort((a, b) => {
      if (!a.created_at || !b.created_at) return 0
      return new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    }),
    [events],
  )

  if (events.length === 0) {
    return (
      <div className={cn('flex items-center justify-center py-12 text-sm text-muted-foreground', className)}>
        No events yet
      </div>
    )
  }

  return (
    <ScrollArea className={cn('h-full', className)}>
      <div className="space-y-1 p-3">
        {sortedEvents.map((event) => {
          const config = EVENT_CONFIG[event.event_type] ?? DEFAULT_CONFIG
          const Icon = config.icon

          return (
            <div
              key={event.id}
              className="flex items-start gap-2.5 py-2 px-2 rounded-lg hover:bg-secondary/30 transition-colors"
            >
              <Icon className={cn('w-4 h-4 mt-0.5 shrink-0', config.color)} />

              <div className="flex-1 min-w-0">
                <p className="text-xs leading-relaxed">
                  <span className="font-medium">{config.label}</span>
                  {event.old_state && event.new_state && (
                    <span className="text-muted-foreground">
                      {' '}({event.old_state} → {event.new_state})
                    </span>
                  )}
                </p>
                {event.actor_type !== 'system' && event.actor_id && (
                  <p className="text-[10px] text-muted-foreground mt-0.5">
                    by {event.actor_type}: {event.actor_id}
                  </p>
                )}
              </div>

              {event.created_at && (
                <span className="text-[10px] text-muted-foreground shrink-0 mt-0.5">
                  {formatDistanceToNow(new Date(event.created_at), { addSuffix: true })}
                </span>
              )}
            </div>
          )
        })}
      </div>
    </ScrollArea>
  )
}
