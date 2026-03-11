'use client'

import { Activity, Shield, AlertTriangle } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useMemoryHealth } from '@/hooks/use-memory-explorer-api'

export function HealthBanner() {
  const { data: health } = useMemoryHealth()
  if (!health) return null

  const statusColors: Record<string, string> = {
    healthy: 'text-[hsl(var(--success))]',
    degraded: 'text-[hsl(var(--warning))]',
    unavailable: 'text-destructive',
  }
  const statusIcons: Record<string, typeof Activity> = {
    healthy: Shield,
    degraded: AlertTriangle,
    unavailable: AlertTriangle,
  }
  const StatusIcon = statusIcons[health.health_status] || Activity

  return (
    <div className="flex flex-wrap gap-3">
      <div className="glass-card px-3 py-2 text-center min-w-[80px]">
        <div className="text-lg font-bold leading-none">{health.total_memories}</div>
        <div className="text-[10px] text-muted-foreground mt-1">Total</div>
      </div>
      <div className="glass-card px-3 py-2 text-center min-w-[80px]">
        <div className={cn('text-lg font-bold leading-none flex items-center justify-center gap-1', statusColors[health.health_status])}>
          <StatusIcon className="w-4 h-4" />
          <span className="capitalize">{health.health_status}</span>
        </div>
        <div className="text-[10px] text-muted-foreground mt-1">Status</div>
      </div>
      <div className="glass-card px-3 py-2 text-center min-w-[80px]">
        <div className="text-lg font-bold leading-none">
          {Math.round((health.search_effectiveness?.hit_rate ?? 0) * 100)}%
        </div>
        <div className="text-[10px] text-muted-foreground mt-1">Hit Rate</div>
      </div>
      <div className="glass-card px-3 py-2 text-center min-w-[80px]">
        <div className="text-lg font-bold leading-none">
          {health.search_effectiveness?.total_searches ?? 0}
        </div>
        <div className="text-[10px] text-muted-foreground mt-1">Searches</div>
      </div>
      {health.newest_memory && (
        <div className="glass-card px-3 py-2 text-center min-w-[80px]">
          <div className="text-sm font-bold leading-none">
            {new Date(health.newest_memory).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })}
          </div>
          <div className="text-[10px] text-muted-foreground mt-1">Latest</div>
        </div>
      )}
    </div>
  )
}
