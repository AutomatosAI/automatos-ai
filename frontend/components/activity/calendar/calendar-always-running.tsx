'use client'

import { Zap } from 'lucide-react'
import type { ScheduleItem } from '@/hooks/use-activity-api'
import { cn } from '@/lib/utils'

interface CalendarAlwaysRunningProps {
  items: ScheduleItem[]
  className?: string
}

export function CalendarAlwaysRunning({ items, className }: CalendarAlwaysRunningProps) {
  if (items.length === 0) return null

  return (
    <div className={cn('glass-card rounded-xl p-4', className)}>
      <div className="flex items-center gap-2 mb-3">
        <Zap className="w-4 h-4 text-[hsl(var(--warning))]" />
        <h3 className="text-sm font-semibold">Always Running</h3>
      </div>
      <div className="flex flex-wrap gap-2">
        {items.map((item) => (
          <div
            key={item.id}
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-[hsl(var(--agent))]/30 bg-[hsl(var(--agent))]/10 text-sm"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-[hsl(var(--agent))] animate-pulse" />
            <span className="font-medium">{item.name}</span>
            <span className="text-xs text-muted-foreground">
              {item.frequency}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
