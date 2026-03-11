'use client'

import { useMemo } from 'react'
import { startOfWeek, addDays, format, isToday, parseISO, isSameDay } from 'date-fns'
import type { ScheduleItem } from '@/hooks/use-activity-api'
import { cn } from '@/lib/utils'

// Stable colour palette for agents (hashed by name)
const AGENT_COLORS = [
  { bg: 'bg-blue-500/20', border: 'border-blue-500/40', text: 'text-blue-300' },
  { bg: 'bg-purple-500/20', border: 'border-purple-500/40', text: 'text-purple-300' },
  { bg: 'bg-emerald-500/20', border: 'border-emerald-500/40', text: 'text-emerald-300' },
  { bg: 'bg-amber-500/20', border: 'border-amber-500/40', text: 'text-amber-300' },
  { bg: 'bg-rose-500/20', border: 'border-rose-500/40', text: 'text-rose-300' },
  { bg: 'bg-cyan-500/20', border: 'border-cyan-500/40', text: 'text-cyan-300' },
  { bg: 'bg-orange-500/20', border: 'border-orange-500/40', text: 'text-orange-300' },
  { bg: 'bg-indigo-500/20', border: 'border-indigo-500/40', text: 'text-indigo-300' },
]

function getAgentColor(agentName: string | null) {
  if (!agentName) return AGENT_COLORS[0]
  let hash = 0
  for (let i = 0; i < agentName.length; i++) {
    hash = ((hash << 5) - hash + agentName.charCodeAt(i)) | 0
  }
  return AGENT_COLORS[Math.abs(hash) % AGENT_COLORS.length]
}

/**
 * Parse frequency to determine which days of the week a task runs.
 * Returns array of day indices (0=Sun, 1=Mon, ..., 6=Sat) or null if can't determine.
 */
function getRecurringDays(frequency: string): number[] | null {
  // Match "Every Xm" or "Every Xh" — runs daily
  if (/every\s+\d+\s*(m|min|h|hr)/i.test(frequency)) {
    return [0, 1, 2, 3, 4, 5, 6]
  }

  // Match cron: "0 9 * * 1-5" (weekdays) or "0 9 * * *" (daily)
  const cronMatch = frequency.match(/^[\d,/*-]+\s+[\d,/*-]+\s+[\d,/*-]+\s+[\d,/*-]+\s+([\d,/*-]+)$/)
  if (cronMatch) {
    const dowField = cronMatch[1]
    if (dowField === '*') return [0, 1, 2, 3, 4, 5, 6]

    // Parse ranges like "1-5"
    const rangeMatch = dowField.match(/^(\d)-(\d)$/)
    if (rangeMatch) {
      const start = parseInt(rangeMatch[1], 10)
      const end = parseInt(rangeMatch[2], 10)
      const days: number[] = []
      for (let i = start; i <= end; i++) days.push(i)
      return days
    }

    // Parse comma-separated like "1,3,5"
    if (/^[\d,]+$/.test(dowField)) {
      return dowField.split(',').map(Number)
    }
  }

  return null
}

/**
 * Parse the hour from a frequency/cron string.
 * Returns formatted time string or null.
 */
function getScheduledTime(item: ScheduleItem): string | null {
  // Try from next_run_at
  if (item.next_run_at) {
    try {
      return format(parseISO(item.next_run_at), 'h:mm a')
    } catch {
      return null
    }
  }

  // Try to extract from cron: "0 9 * * *" → 9:00 AM
  const cronMatch = item.frequency.match(/^(\d+)\s+(\d+)\s+/)
  if (cronMatch) {
    const mins = parseInt(cronMatch[1], 10)
    const hours = parseInt(cronMatch[2], 10)
    const d = new Date()
    d.setHours(hours, mins, 0, 0)
    return format(d, 'h:mm a')
  }

  return null
}

interface DayTask {
  item: ScheduleItem
  time: string | null
  color: ReturnType<typeof getAgentColor>
}

interface CalendarWeekGridProps {
  items: ScheduleItem[]
  alwaysRunning: ScheduleItem[]
  viewMode: 'week' | 'today'
  className?: string
}

export function CalendarWeekGrid({ items, alwaysRunning, viewMode, className }: CalendarWeekGridProps) {
  const weekStart = startOfWeek(new Date(), { weekStartsOn: 0 }) // Sunday
  const days = viewMode === 'today'
    ? [new Date()]
    : Array.from({ length: 7 }, (_, i) => addDays(weekStart, i))

  // Map tasks to days
  const tasksByDay = useMemo(() => {
    const map = new Map<string, DayTask[]>()

    for (const day of days) {
      map.set(format(day, 'yyyy-MM-dd'), [])
    }

    // Place items based on recurring pattern or next_run_at
    for (const item of items) {
      const color = getAgentColor(item.agent_name)
      const time = getScheduledTime(item)
      const recurringDays = getRecurringDays(item.frequency)

      if (recurringDays) {
        // Recurring: place on matching days
        for (const day of days) {
          const dow = day.getDay()
          if (recurringDays.includes(dow)) {
            const key = format(day, 'yyyy-MM-dd')
            const existing = map.get(key) ?? []
            map.set(key, [...existing, { item, time, color }])
          }
        }
      } else if (item.next_run_at) {
        // One-shot: place on the specific day
        try {
          const runDate = parseISO(item.next_run_at)
          for (const day of days) {
            if (isSameDay(runDate, day)) {
              const key = format(day, 'yyyy-MM-dd')
              const existing = map.get(key) ?? []
              map.set(key, [...existing, { item, time, color }])
            }
          }
        } catch {
          // ignore parse errors
        }
      }
    }

    // Sort tasks within each day by time
    for (const [key, tasks] of map) {
      map.set(key, tasks.sort((a, b) => {
        if (!a.time) return 1
        if (!b.time) return -1
        return a.time.localeCompare(b.time)
      }))
    }

    return map
  }, [items, days])

  return (
    <div className={cn('glass-card rounded-xl overflow-hidden', className)}>
      <div className={cn(
        'grid gap-px bg-border/20',
        viewMode === 'today' ? 'grid-cols-1' : 'grid-cols-7',
      )}>
        {days.map((day) => {
          const key = format(day, 'yyyy-MM-dd')
          const tasks = tasksByDay.get(key) ?? []
          const today = isToday(day)

          return (
            <div key={key} className="bg-background min-h-[200px] flex flex-col">
              {/* Day header */}
              <div className={cn(
                'px-3 py-2 border-b border-border/30 text-center',
                today && 'bg-primary/5',
              )}>
                <p className={cn(
                  'text-xs font-medium',
                  today ? 'text-primary' : 'text-muted-foreground',
                )}>
                  {format(day, 'EEE')}
                </p>
                <p className={cn(
                  'text-sm',
                  today && 'text-primary font-bold',
                )}>
                  {format(day, 'd')}
                </p>
              </div>

              {/* Task pills */}
              <div className="flex-1 p-1.5 space-y-1 overflow-y-auto">
                {tasks.map((task, idx) => (
                  <div
                    key={`${task.item.id}-${idx}`}
                    className={cn(
                      'px-2 py-1.5 rounded-md border text-xs cursor-default transition-opacity hover:opacity-80',
                      task.color.bg,
                      task.color.border,
                    )}
                  >
                    <p className={cn('font-medium truncate', task.color.text)}>
                      {task.item.name}
                    </p>
                    {task.time && (
                      <p className="text-[10px] text-muted-foreground mt-0.5">
                        {task.time}
                      </p>
                    )}
                  </div>
                ))}
                {tasks.length === 0 && (
                  <div className="h-full flex items-center justify-center">
                    <span className="text-[10px] text-muted-foreground/30">—</span>
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
