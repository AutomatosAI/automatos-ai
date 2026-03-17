'use client'

import { usePlaybookExecutions } from '@/hooks/use-playbook-api'

interface PlaybookRunDotsProps {
  recipeId: string
  useCount?: number
  compact?: boolean
  onClick?: () => void
}

function dotColor(status: string): string {
  switch (status) {
    case 'completed':
      return 'bg-[hsl(var(--success))]'
    case 'failed':
      return 'bg-destructive'
    default:
      return 'bg-muted-foreground/40'
  }
}

function shortTimeAgo(dateStr: string | undefined): string {
  if (!dateStr) return ''
  try {
    const diff = Date.now() - new Date(dateStr).getTime()
    const mins = Math.floor(diff / 60000)
    if (mins < 1) return 'now'
    if (mins < 60) return `${mins}m`
    const hours = Math.floor(mins / 60)
    if (hours < 24) return `${hours}h`
    const days = Math.floor(hours / 24)
    return `${days}d`
  } catch {
    return ''
  }
}

export function PlaybookRunDots({ recipeId, useCount, compact, onClick }: PlaybookRunDotsProps) {
  const { data } = usePlaybookExecutions(recipeId, { limit: 3 })
  const executions: any[] = (data as any)?.items || (Array.isArray(data) ? data : [])

  if (executions.length === 0) return null

  return (
    <div
      className={`flex items-center gap-2 cursor-pointer group/dots ${compact ? '' : 'py-0.5'}`}
      onClick={(e) => {
        e.stopPropagation()
        onClick?.()
      }}
      role="button"
      tabIndex={0}
      title="View run history"
    >
      <div className="flex items-center gap-1.5">
        {executions.map((exec: any, i: number) => (
          <div
            key={exec.id || exec.execution_id || i}
            className={compact ? '' : 'flex flex-col items-center gap-0.5'}
          >
            <div
              className={`w-2 h-2 rounded-full ${dotColor(exec.status)} transition-transform group-hover/dots:scale-110`}
            />
            {!compact && (
              <span className="text-[8px] text-muted-foreground leading-none">
                {shortTimeAgo(exec.started_at || exec.completed_at)}
              </span>
            )}
          </div>
        ))}
      </div>
      {useCount != null && useCount > 0 && (
        <span className="text-[10px] text-muted-foreground">
          Ran {useCount} times
        </span>
      )}
    </div>
  )
}
