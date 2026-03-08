'use client'

import { useMemo } from 'react'
import { Calendar, Clock, RefreshCw, ChefHat } from 'lucide-react'
import { useActivitySchedule } from '@/hooks/use-activity-api'
import type { ScheduleItem } from '@/hooks/use-activity-api'
import { format, startOfWeek, addDays, isToday, isSameDay, parseISO } from 'date-fns'
import { cn } from '@/lib/utils'

interface ScheduleWidgetProps {
  className?: string
}

function WeekCalendar({ items }: { items: ScheduleItem[] }) {
  const weekStart = startOfWeek(new Date(), { weekStartsOn: 1 })
  const days = Array.from({ length: 7 }, (_, i) => addDays(weekStart, i))

  // Count items per day
  const countsByDay = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const item of items) {
      if (!item.next_run_at) continue
      const dateKey = format(parseISO(item.next_run_at), 'yyyy-MM-dd')
      counts[dateKey] = (counts[dateKey] || 0) + 1
    }
    return counts
  }, [items])

  return (
    <div className="grid grid-cols-7 gap-1 text-center">
      {days.map((day) => {
        const dateKey = format(day, 'yyyy-MM-dd')
        const count = countsByDay[dateKey] || 0
        const today = isToday(day)

        return (
          <div key={dateKey} className="flex flex-col items-center gap-1">
            <span className={cn(
              'text-[10px] font-medium',
              today ? 'text-primary' : 'text-muted-foreground'
            )}>
              {format(day, 'EEE')}
            </span>
            <div className={cn(
              'w-7 h-7 rounded-full flex items-center justify-center text-xs',
              today && 'bg-primary text-primary-foreground font-bold',
              !today && count > 0 && 'bg-secondary/80',
            )}>
              {format(day, 'd')}
            </div>
            <div className="flex gap-0.5 h-2">
              {count > 0 && Array.from({ length: Math.min(count, 3) }).map((_, i) => (
                <div key={i} className="w-1.5 h-1.5 rounded-full bg-[hsl(var(--info))]" />
              ))}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function UpcomingList({ items }: { items: ScheduleItem[] }) {
  if (items.length === 0) {
    return (
      <p className="text-xs text-muted-foreground text-center py-3">
        No upcoming schedules
      </p>
    )
  }

  return (
    <div className="space-y-1.5">
      {items.slice(0, 5).map((item) => {
        const nextRun = item.next_run_at ? parseISO(item.next_run_at) : null
        const timeStr = nextRun ? format(nextRun, 'HH:mm') : '--:--'
        const isItemToday = nextRun ? isToday(nextRun) : false
        const dayLabel = nextRun
          ? isItemToday
            ? 'Today'
            : isSameDay(nextRun, addDays(new Date(), 1))
              ? 'Tomorrow'
              : format(nextRun, 'EEE')
          : ''

        return (
          <div key={item.id} className="flex items-center gap-2 py-1 px-1 rounded hover:bg-secondary/30 transition-colors">
            <span className="text-xs font-mono text-muted-foreground w-10 shrink-0">
              {timeStr}
            </span>
            {item.type === 'routine' ? (
              <RefreshCw className="w-3 h-3 text-[hsl(var(--agent))] shrink-0" />
            ) : (
              <ChefHat className="w-3 h-3 text-[hsl(var(--info))] shrink-0" />
            )}
            <span className="text-xs truncate flex-1">{item.name}</span>
            <span className="text-[10px] text-muted-foreground shrink-0">
              {dayLabel}
            </span>
          </div>
        )
      })}
    </div>
  )
}

export function ScheduleWidget({ className }: ScheduleWidgetProps) {
  const { data, isLoading } = useActivitySchedule('7d')
  const items = data?.scheduled ?? []

  return (
    <div className={cn('h-full flex flex-col', className)}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <div className="flex items-center gap-2">
          <Calendar className="w-4 h-4 text-[hsl(var(--success))]" />
          <h3 className="text-sm font-semibold">Schedule</h3>
        </div>
        <span className="text-[10px] text-muted-foreground">This Week</span>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4">
        {isLoading ? (
          <div className="space-y-2">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="h-4 bg-secondary/50 rounded animate-pulse" />
            ))}
          </div>
        ) : (
          <>
            <WeekCalendar items={items} />
            <div>
              <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-2">
                Upcoming
              </p>
              <UpcomingList items={items} />
            </div>
          </>
        )}
      </div>
    </div>
  )
}
