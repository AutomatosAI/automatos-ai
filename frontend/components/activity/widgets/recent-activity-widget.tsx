'use client'

import { useRouter } from 'next/navigation'
import {
  CheckCircle2,
  XCircle,
  Clock,
  Eye,
  ChefHat,
  RefreshCw,
  History,
  Loader2,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useActivityFeed } from '@/hooks/use-activity-api'
import type { ActivityFeedItem } from '@/hooks/use-activity-api'
import { formatDistanceToNow } from 'date-fns'
import { cn } from '@/lib/utils'

function formatDuration(seconds: number | null): string {
  if (seconds == null) return '--'
  if (seconds < 60) return `${Math.round(seconds)}s`
  const mins = Math.floor(seconds / 60)
  const secs = Math.round(seconds % 60)
  if (mins < 60) return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`
  const hours = Math.floor(mins / 60)
  return `${hours}h ${mins % 60}m`
}

const STATUS_ICON = {
  completed: { icon: CheckCircle2, color: 'text-[hsl(var(--success))]' },
  failed: { icon: XCircle, color: 'text-destructive' },
  cancelled: { icon: Clock, color: 'text-muted-foreground' },
} as const

interface RecentActivityWidgetProps {
  period: string
  onViewAll?: () => void
  className?: string
}

export function RecentActivityWidget({ period, onViewAll, className }: RecentActivityWidgetProps) {
  const router = useRouter()
  const { data, isLoading } = useActivityFeed({
    status: 'done',
    period,
    limit: 5,
  })

  // Also fetch failed items and merge
  const { data: failedData } = useActivityFeed({
    status: 'attention',
    period,
    limit: 3,
  })

  // Merge and sort by started_at DESC
  const items = [
    ...(data?.items ?? []),
    ...(failedData?.items ?? []),
  ]
    .sort((a, b) => {
      const aDate = a.started_at ? new Date(a.started_at).getTime() : 0
      const bDate = b.started_at ? new Date(b.started_at).getTime() : 0
      return bDate - aDate
    })
    .slice(0, 5)

  const handleView = (item: ActivityFeedItem) => {
    if (item.type === 'chat' && item.source_id) {
      router.push(`/chat/${item.source_id}`)
    } else {
      // Navigate to Board tab with task focused
      const taskId = item.id.replace(/^(recipe|routine)-/, '')
      router.replace(`/activity?tab=board&task_id=${taskId}`, { scroll: false })
    }
  }

  return (
    <div className={cn('h-full flex flex-col', className)}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <div className="flex items-center gap-2">
          <History className="w-4 h-4 text-muted-foreground" />
          <h3 className="text-sm font-semibold">Recent Activity</h3>
        </div>
        {onViewAll && (
          <Button variant="ghost" size="sm" className="text-xs h-6" onClick={onViewAll}>
            View All
          </Button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : items.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
            <History className="w-8 h-8 mb-2 opacity-30" />
            <p className="text-xs">No recent activity</p>
          </div>
        ) : (
          <div className="divide-y divide-border/30">
            {items.map((item) => {
              const statusConf = STATUS_ICON[item.status as keyof typeof STATUS_ICON] ?? STATUS_ICON.completed
              const StatusIcon = statusConf.icon
              const timeAgo = item.started_at
                ? formatDistanceToNow(new Date(item.started_at), { addSuffix: true })
                : ''

              return (
                <div
                  key={item.id}
                  className="flex items-center gap-3 px-4 py-2.5 hover:bg-secondary/20 transition-colors"
                >
                  <StatusIcon className={cn('w-4 h-4 shrink-0', statusConf.color)} />
                  {item.type === 'recipe' ? (
                    <ChefHat className="w-3 h-3 text-[hsl(var(--info))] shrink-0" />
                  ) : (
                    <RefreshCw className="w-3 h-3 text-[hsl(var(--agent))] shrink-0" />
                  )}
                  <span className="text-sm truncate flex-1">{item.name}</span>
                  <span className="text-[10px] text-muted-foreground font-mono shrink-0 w-14 text-right">
                    {formatDuration(item.duration_seconds)}
                  </span>
                  <span className="text-[10px] text-muted-foreground shrink-0 w-16 text-right hidden sm:block">
                    {timeAgo}
                  </span>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 shrink-0"
                    onClick={() => handleView(item)}
                  >
                    <Eye className="w-3 h-3" />
                  </Button>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
