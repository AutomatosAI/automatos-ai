'use client'

/**
 * PRD-244 review batch 2 — "Activity": what is running now, then what just
 * finished or needs attention. Replaces the Active Now and Recent Activity
 * widgets (both read the same feed). Rows open the thing itself, through the
 * Activity tab's own link rules.
 */
import { useRouter } from 'next/navigation'
import { Activity, CheckCircle2, ChefHat, Clock, Eye, Loader2, RefreshCw, XCircle } from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'
import { Button } from '@/components/ui/button'
import { useActivityFeed, type ActivityFeedItem } from '@/hooks/use-activity-api'
import { rowHref } from '@/components/command-center/activity-tab'
import { cn } from '@/lib/utils'

const RUNNING_ROWS = 5
const RECENT_ROWS = 5

export function formatElapsed(iso: string | null): string {
  if (!iso) return ''
  const seconds = Math.floor((Date.now() - new Date(iso).getTime()) / 1000)
  if (seconds < 60) return `${seconds}s`
  const mins = Math.floor(seconds / 60)
  if (mins < 60) return `${mins}m`
  return `${Math.floor(mins / 60)}h ${mins % 60}m`
}

export function formatDuration(seconds: number | null): string {
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

const ts = (iso: string | null) => (iso ? new Date(iso).getTime() : 0)

interface ActivityWidgetProps {
  period: string
  onViewAll?: () => void
  className?: string
}

export function ActivityWidget({ period, onViewAll, className }: ActivityWidgetProps) {
  const router = useRouter()
  const running = useActivityFeed({ status: 'working', period, limit: RUNNING_ROWS })
  const done = useActivityFeed({ status: 'done', period, limit: RECENT_ROWS })
  const attention = useActivityFeed({ status: 'attention', period, limit: 3 })

  const runningItems = running.data?.items ?? []
  const recent = [...(done.data?.items ?? []), ...(attention.data?.items ?? [])]
    .sort((a, b) => ts(b.started_at) - ts(a.started_at))
    .slice(0, RECENT_ROWS)
  const isLoading = running.isLoading || done.isLoading

  const open = (item: ActivityFeedItem) => {
    const href = rowHref(item)
    if (href) router.push(href as any)
  }

  return (
    <div className={cn('h-full flex flex-col', className)}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-[hsl(var(--info))]" />
          <h3 className="text-sm font-semibold">Activity</h3>
          {runningItems.length > 0 && (
            <span className="text-xs bg-[hsl(var(--info))]/15 text-[hsl(var(--info))] px-1.5 py-0.5 rounded-full font-medium">
              {runningItems.length} running
            </span>
          )}
        </div>
        {onViewAll && (
          <Button variant="ghost" size="sm" className="text-xs h-6" onClick={onViewAll}>
            View all →
          </Button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <>
            <section className="px-4 py-2 space-y-1.5" aria-label="Running">
              <p className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">Running</p>
              {runningItems.length === 0 ? (
                <p className="text-xs text-muted-foreground py-1">All quiet — nothing running</p>
              ) : (
                runningItems.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => open(item)}
                    className="w-full text-left p-2 rounded-lg bg-secondary/30 hover:bg-secondary/50 transition-colors space-y-1.5"
                  >
                    <div className="flex items-center gap-2">
                      <div className={cn('w-2 h-2 rounded-full shrink-0', item.status === 'running' ? 'bg-[hsl(var(--info))] animate-pulse' : 'bg-muted-foreground/30')} />
                      {item.type === 'recipe' ? <ChefHat className="w-3 h-3 text-[hsl(var(--info))] shrink-0" /> : <RefreshCw className="w-3 h-3 text-[hsl(var(--agent))] shrink-0" />}
                      <span className="text-sm font-medium truncate flex-1">{item.name}</span>
                      <span className="text-[10px] text-muted-foreground shrink-0">{item.status === 'running' ? formatElapsed(item.started_at) : 'Pending'}</span>
                    </div>
                    {item.step_progress && (
                      <div className="pl-6 flex items-center gap-2">
                        <div className="flex-1 h-1.5 bg-secondary/50 rounded-full overflow-hidden">
                          <div className="h-full bg-[hsl(var(--info))] rounded-full" style={{ width: `${Math.round((item.step_progress.current / item.step_progress.total) * 100)}%` }} />
                        </div>
                        <span className="text-[10px] text-muted-foreground shrink-0">Step {item.step_progress.current}/{item.step_progress.total}</span>
                      </div>
                    )}
                  </button>
                ))
              )}
            </section>

            <section className="border-t border-border/30" aria-label="Recent">
              <p className="px-4 pt-2 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">Recent</p>
              {recent.length === 0 ? (
                <p className="px-4 py-2 text-xs text-muted-foreground">No recent activity</p>
              ) : (
                <div className="divide-y divide-border/30">
                  {recent.map((item) => {
                    const conf = STATUS_ICON[item.status as keyof typeof STATUS_ICON] ?? STATUS_ICON.completed
                    const StatusIcon = conf.icon
                    return (
                      <div key={item.id} className="flex items-center gap-3 px-4 py-2 hover:bg-secondary/20 transition-colors">
                        <StatusIcon className={cn('w-4 h-4 shrink-0', conf.color)} />
                        {item.type === 'recipe' ? <ChefHat className="w-3 h-3 text-[hsl(var(--info))] shrink-0" /> : <RefreshCw className="w-3 h-3 text-[hsl(var(--agent))] shrink-0" />}
                        <span className="text-sm truncate flex-1">{item.name}</span>
                        <span className="text-[10px] text-muted-foreground font-mono shrink-0 w-14 text-right">{formatDuration(item.duration_seconds)}</span>
                        <span className="text-[10px] text-muted-foreground shrink-0 w-16 text-right hidden sm:block">
                          {item.started_at ? formatDistanceToNow(new Date(item.started_at), { addSuffix: true }) : ''}
                        </span>
                        <Button variant="ghost" size="icon" className="h-6 w-6 shrink-0" aria-label={`Open ${item.name}`} onClick={() => open(item)}>
                          <Eye className="w-3 h-3" />
                        </Button>
                      </div>
                    )
                  })}
                </div>
              )}
            </section>
          </>
        )}
      </div>
    </div>
  )
}
