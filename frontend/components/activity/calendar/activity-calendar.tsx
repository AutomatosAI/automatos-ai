'use client'

import { useMemo, useRef, useState } from 'react'
import { AlertCircle, RefreshCw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useActivitySchedule } from '@/hooks/use-activity-api'
import type { ScheduleItem } from '@/hooks/use-activity-api'
import { CalendarWeekGrid } from './calendar-week-grid'
import { CalendarAlwaysRunning } from './calendar-always-running'
import { CalendarNextUp } from './calendar-next-up'
import { cn } from '@/lib/utils'

type ViewMode = 'week' | 'today'

interface ActivityCalendarProps {
  className?: string
}

/**
 * Parse frequency string to interval in minutes.
 * Handles "Every 30m", "Every 8h", "Every 1d", etc.
 * Returns null if not parseable (e.g. raw cron expressions).
 */
function parseIntervalMinutes(frequency: string): number | null {
  const match = frequency.match(/every\s+(\d+)\s*(m|min|h|hr|d|day)/i)
  if (!match) return null
  const value = parseInt(match[1], 10)
  const unit = match[2].toLowerCase()
  if (unit.startsWith('m')) return value
  if (unit.startsWith('h')) return value * 60
  if (unit.startsWith('d')) return value * 1440
  return null
}

export function ActivityCalendar({ className }: ActivityCalendarProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('week')
  const { data, isLoading, isError, refetch } = useActivitySchedule('7d')

  // Cache last successful data so calendar never blanks on transient errors
  const lastGoodData = useRef<ScheduleItem[]>([])
  const freshItems = data?.scheduled ?? []
  if (freshItems.length > 0) {
    lastGoodData.current = freshItems
  }
  const items = freshItems.length > 0 ? freshItems : lastGoodData.current

  // Split into "always running" (< 60 min interval) and regular scheduled
  const { alwaysRunning, scheduled } = useMemo(() => {
    const always: ScheduleItem[] = []
    const sched: ScheduleItem[] = []

    for (const item of items) {
      const intervalMin = parseIntervalMinutes(item.frequency)
      if (intervalMin !== null && intervalMin < 60) {
        always.push(item)
      } else {
        sched.push(item)
      }
    }

    return { alwaysRunning: always, scheduled: sched }
  }, [items])

  return (
    <div className={cn('space-y-4', className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Scheduled Tasks</h2>
          <p className="text-sm text-muted-foreground">Your automated routines</p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex rounded-lg border border-border/50 overflow-hidden">
            <button
              onClick={() => setViewMode('week')}
              className={cn(
                'px-3 py-1.5 text-xs font-medium transition-colors',
                viewMode === 'week'
                  ? 'bg-secondary text-foreground'
                  : 'text-muted-foreground hover:text-foreground'
              )}
            >
              Week
            </button>
            <button
              onClick={() => setViewMode('today')}
              className={cn(
                'px-3 py-1.5 text-xs font-medium transition-colors',
                viewMode === 'today'
                  ? 'bg-secondary text-foreground'
                  : 'text-muted-foreground hover:text-foreground'
              )}
            >
              Today
            </button>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={() => refetch()}
          >
            <RefreshCw className={cn('w-4 h-4', isLoading && 'animate-spin')} />
          </Button>
        </div>
      </div>

      {/* Error indicator (non-destructive — calendar still shows cached data) */}
      {isError && items.length > 0 && (
        <div className="flex items-center gap-2 text-xs text-warning/80 px-1">
          <AlertCircle className="h-3.5 w-3.5" />
          <span>Showing cached schedule — refresh to retry</span>
        </div>
      )}

      {/* Always Running Section */}
      {alwaysRunning.length > 0 && (
        <CalendarAlwaysRunning items={alwaysRunning} />
      )}

      {/* Week Grid or Today View */}
      {isLoading ? (
        <div className="space-y-4">
          <div className="h-[300px] bg-secondary/20 rounded-xl animate-pulse" />
          <div className="h-[200px] bg-secondary/20 rounded-xl animate-pulse" />
        </div>
      ) : (
        <CalendarWeekGrid
          items={scheduled}
          alwaysRunning={alwaysRunning}
          viewMode={viewMode}
        />
      )}

      {/* Next Up Section */}
      <CalendarNextUp items={items} isLoading={isLoading} />
    </div>
  )
}
