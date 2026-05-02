'use client'

import { Calendar, Loader2 } from 'lucide-react'
import type { ScheduleItem } from '@/hooks/use-activity-api'
import { parseISO, formatDistanceToNowStrict } from 'date-fns'
import { cn } from '@/lib/utils'

// Same color hash as week grid for consistency
const AGENT_COLORS = [
  'text-info',
  'text-agent',
  'text-success',
  'text-warning',
  'text-rose-400',
  'text-cyan-400',
  'text-orange-400',
  'text-indigo-400',
]

function getAgentTextColor(agentName: string | null): string {
  if (!agentName) return AGENT_COLORS[0]
  let hash = 0
  for (let i = 0; i < agentName.length; i++) {
    hash = ((hash << 5) - hash + agentName.charCodeAt(i)) | 0
  }
  return AGENT_COLORS[Math.abs(hash) % AGENT_COLORS.length]
}

interface CalendarNextUpProps {
  items: ScheduleItem[]
  isLoading: boolean
  className?: string
}

export function CalendarNextUp({ items, isLoading, className }: CalendarNextUpProps) {
  // Sort by next_run_at ascending, filter out items without a next run
  const upcoming = items
    .filter((item) => item.next_run_at)
    .sort((a, b) => {
      const aTime = new Date(a.next_run_at!).getTime()
      const bTime = new Date(b.next_run_at!).getTime()
      return aTime - bTime
    })
    .slice(0, 8)

  return (
    <div className={cn('glass-card rounded-xl p-4', className)}>
      <div className="flex items-center gap-2 mb-3">
        <Calendar className="w-4 h-4 text-muted-foreground" />
        <h3 className="text-sm font-semibold">Next Up</h3>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-6">
          <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
        </div>
      ) : upcoming.length === 0 ? (
        <p className="text-xs text-muted-foreground text-center py-6">
          No upcoming tasks scheduled
        </p>
      ) : (
        <div className="divide-y divide-border/30">
          {upcoming.map((item) => {
            const nextRun = parseISO(item.next_run_at!)
            const now = new Date()
            const isPast = nextRun < now
            const relativeTime = isPast
              ? 'Now'
              : `In ${formatDistanceToNowStrict(nextRun)}`
            const textColor = getAgentTextColor(item.agent_name)

            return (
              <div
                key={item.id}
                className="flex items-center justify-between py-2.5 hover:bg-secondary/10 transition-colors rounded px-1"
              >
                <span className={cn('text-sm font-medium', textColor)}>
                  {item.name}
                </span>
                <span className="text-xs text-muted-foreground shrink-0 ml-4">
                  {relativeTime}
                </span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
