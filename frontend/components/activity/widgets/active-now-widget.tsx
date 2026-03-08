'use client'

import { useRouter } from 'next/navigation'
import { Loader2, Clock, Play, ChefHat, RefreshCw } from 'lucide-react'
import { useActivityFeed } from '@/hooks/use-activity-api'
import type { ActivityFeedItem } from '@/hooks/use-activity-api'
import { cn } from '@/lib/utils'

function formatElapsed(iso: string | null): string {
  if (!iso) return ''
  const seconds = Math.floor((Date.now() - new Date(iso).getTime()) / 1000)
  if (seconds < 60) return `${seconds}s`
  const mins = Math.floor(seconds / 60)
  if (mins < 60) return `${mins}m`
  return `${Math.floor(mins / 60)}h ${mins % 60}m`
}

function ProgressBar({ item }: { item: ActivityFeedItem }) {
  if (!item.step_progress) return null
  const pct = Math.round((item.step_progress.current / item.step_progress.total) * 100)
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-secondary/50 rounded-full overflow-hidden">
        <div
          className="h-full bg-[hsl(var(--info))] rounded-full transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-[10px] text-muted-foreground shrink-0">{pct}%</span>
    </div>
  )
}

interface ActiveNowWidgetProps {
  period: string
  className?: string
}

export function ActiveNowWidget({ period, className }: ActiveNowWidgetProps) {
  const router = useRouter()
  const { data, isLoading } = useActivityFeed({
    status: 'working',
    period,
    limit: 5,
  })

  const items = data?.items ?? []

  const handleClick = (item: ActivityFeedItem) => {
    if (item.source_url) {
      router.push(item.source_url)
    }
  }

  return (
    <div className={cn('h-full flex flex-col', className)}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <div className="flex items-center gap-2">
          <Play className="w-4 h-4 text-[hsl(var(--info))]" />
          <h3 className="text-sm font-semibold">Active Now</h3>
          {items.length > 0 && (
            <span className="text-xs bg-[hsl(var(--info))]/15 text-[hsl(var(--info))] px-1.5 py-0.5 rounded-full font-medium">
              {items.length}
            </span>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-2 space-y-2">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : items.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
            <Clock className="w-8 h-8 mb-2 opacity-30" />
            <p className="text-xs">All quiet — nothing running</p>
          </div>
        ) : (
          items.map((item) => (
            <div
              key={item.id}
              onClick={() => handleClick(item)}
              className="p-2.5 rounded-lg bg-secondary/30 hover:bg-secondary/50 cursor-pointer transition-colors space-y-1.5"
            >
              <div className="flex items-center gap-2">
                <div className={cn(
                  'w-2 h-2 rounded-full shrink-0',
                  item.status === 'running' ? 'bg-[hsl(var(--info))] animate-pulse' : 'bg-muted-foreground/30'
                )} />
                {item.type === 'recipe' ? (
                  <ChefHat className="w-3 h-3 text-[hsl(var(--info))] shrink-0" />
                ) : (
                  <RefreshCw className="w-3 h-3 text-[hsl(var(--agent))] shrink-0" />
                )}
                <span className="text-sm font-medium truncate flex-1">{item.name}</span>
                <span className="text-[10px] text-muted-foreground shrink-0">
                  {item.status === 'running' ? formatElapsed(item.started_at) : 'Pending'}
                </span>
              </div>
              {item.step_progress && (
                <div className="pl-6">
                  <p className="text-[10px] text-muted-foreground mb-1">
                    Step {item.step_progress.current}/{item.step_progress.total}
                  </p>
                  <ProgressBar item={item} />
                </div>
              )}
              {!item.step_progress && item.status === 'pending' && (
                <p className="text-[10px] text-muted-foreground pl-6">
                  Starts soon
                </p>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  )
}
