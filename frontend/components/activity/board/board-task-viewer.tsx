'use client'

import { useEffect } from 'react'
import { X, Bot, Clock, CheckCircle2, AlertCircle, ArrowLeft, RotateCcw, Play, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { PremiumIcon } from '@/components/shared'
import { formatDistanceToNow, format } from 'date-fns'
import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import type { BoardTask } from '@/types/board'
import { PRIORITY_CONFIG, STATUS_CONFIG } from '@/types/board'
import { useUpdateTaskStatus } from '@/hooks/use-board-tasks'
import { cn } from '@/lib/utils'

interface BoardTaskViewerProps {
  task: BoardTask
  onClose: () => void
  className?: string
}

/**
 * Poll the backend for live task data when in_progress.
 * Falls back to the prop task when the query hasn't loaded yet.
 */
function useLiveTask(task: BoardTask) {
  const isActive = task.status === 'in_progress'

  const { data } = useQuery({
    queryKey: ['board-task-live', task.id],
    queryFn: () => apiClient.request<any>(`/api/v1/tasks/${task.source_id || task.id}`),
    enabled: isActive,
    refetchInterval: isActive ? 5000 : false,
    staleTime: 3000,
  })

  if (!data) return task

  // Merge live data into the task shape
  return {
    ...task,
    status: data.status ?? task.status,
    result: data.result ?? task.result,
    error_message: data.error_message ?? task.error_message,
    started_at: data.started_at ?? task.started_at,
    completed_at: data.completed_at ?? task.completed_at,
  } as BoardTask
}

function formatDuration(ms: number | undefined): string {
  if (!ms) return '--'
  const seconds = Math.floor(ms / 1000)
  if (seconds < 60) return `${seconds}s`
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  if (mins < 60) return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`
  const hours = Math.floor(mins / 60)
  return `${hours}h ${mins % 60}m`
}

function formatElapsed(startedAt: string): string {
  return formatDistanceToNow(new Date(startedAt), { addSuffix: false })
}

// ── Status-specific views ────────────────────────────────────────────

function InboxView({ task }: { task: BoardTask }) {
  const priorityConf = PRIORITY_CONFIG[task.priority]

  return (
    <div className="space-y-4">
      {task.description && (
        <div>
          <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Description</h4>
          <p className="text-sm whitespace-pre-wrap">{task.description}</p>
        </div>
      )}

      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1">
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Assignee</p>
          <div className="flex items-center gap-1.5">
            {task.assignee ? (
              <>
                {task.assignee.agent_icon ? (
                  <PremiumIcon name={task.assignee.agent_icon} size={16} />
                ) : (
                  <Bot className="w-4 h-4 text-primary" />
                )}
                <span className="text-sm">{task.assignee.agent_name}</span>
              </>
            ) : (
              <span className="text-sm text-muted-foreground">Unassigned</span>
            )}
          </div>
        </div>

        <div className="space-y-1">
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Priority</p>
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: priorityConf.color }} />
            <span className="text-sm capitalize">{task.priority}</span>
          </div>
        </div>

        <div className="space-y-1">
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Review Mode</p>
          <span className="text-sm capitalize">{task.review_mode}</span>
        </div>

        {task.due_date && (
          <div className="space-y-1">
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Due Date</p>
            <span className="text-sm">{format(new Date(task.due_date), 'MMM d, yyyy')}</span>
          </div>
        )}
      </div>

      {task.tags.length > 0 && (
        <div>
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Tags</p>
          <div className="flex flex-wrap gap-1">
            {task.tags.map((tag) => (
              <span key={tag} className="text-xs px-2 py-0.5 rounded-full bg-secondary/50 text-muted-foreground">
                {tag}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function InProgressView({ task }: { task: BoardTask }) {
  return (
    <div className="space-y-4">
      {/* Live status indicator */}
      <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[hsl(var(--info))]/10 border border-[hsl(var(--info))]/20">
        <Loader2 className="w-4 h-4 text-[hsl(var(--info))] animate-spin" />
        <span className="text-sm font-medium text-[hsl(var(--info))]">Agent is working...</span>
        {task.started_at && (
          <span className="text-xs text-muted-foreground ml-auto">{formatElapsed(task.started_at)} elapsed</span>
        )}
      </div>

      {/* Step progress if available */}
      {task.step_progress && (
        <div>
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Progress</h4>
            <span className="text-xs text-muted-foreground">
              Step {task.step_progress.current} of {task.step_progress.total}
            </span>
          </div>
          <div className="h-2 bg-secondary/50 rounded-full overflow-hidden">
            <div
              className="h-full bg-[hsl(var(--info))] rounded-full transition-all"
              style={{ width: `${(task.step_progress.current / task.step_progress.total) * 100}%` }}
            />
          </div>
        </div>
      )}

      {task.description && (
        <div>
          <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Task Description</h4>
          <p className="text-sm whitespace-pre-wrap text-muted-foreground">{task.description}</p>
        </div>
      )}

      {/* Partial result (streams in as task updates) */}
      {task.result && (
        <div>
          <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Partial Output</h4>
          <div className="glass-card rounded-lg p-3 text-sm whitespace-pre-wrap max-h-[300px] overflow-y-auto font-mono text-xs">
            {String(task.result)}
          </div>
        </div>
      )}
    </div>
  )
}

function ReviewView({ task, onStatusChange }: { task: BoardTask; onStatusChange: (status: string) => void }) {
  return (
    <div className="space-y-4">
      {/* Completion info */}
      <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[hsl(var(--warning))]/10 border border-[hsl(var(--warning))]/20">
        <Clock className="w-4 h-4 text-[hsl(var(--warning))]" />
        <span className="text-sm font-medium text-[hsl(var(--warning))]">Awaiting Review</span>
        {task.completed_at && (
          <span className="text-xs text-muted-foreground ml-auto">
            Completed {formatDistanceToNow(new Date(task.completed_at), { addSuffix: true })}
          </span>
        )}
      </div>

      {/* Agent result */}
      {task.result && (
        <div>
          <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Agent Result</h4>
          <div className="glass-card rounded-lg p-3 text-sm whitespace-pre-wrap max-h-[400px] overflow-y-auto">
            {String(task.result)}
          </div>
        </div>
      )}

      {/* Original description */}
      {task.description && (
        <div>
          <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Original Task</h4>
          <p className="text-sm text-muted-foreground whitespace-pre-wrap">{task.description}</p>
        </div>
      )}

      {/* Metadata */}
      <div className="grid grid-cols-2 gap-3">
        {task.started_at && (
          <div>
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Started</p>
            <p className="text-sm">{format(new Date(task.started_at), 'MMM d, HH:mm')}</p>
          </div>
        )}
        {task.completed_at && (
          <div>
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Completed</p>
            <p className="text-sm">{format(new Date(task.completed_at), 'MMM d, HH:mm')}</p>
          </div>
        )}
      </div>

      {/* Review actions */}
      <div className="flex gap-2 pt-2">
        <Button
          variant="outline"
          size="sm"
          className="text-xs flex-1"
          onClick={() => onStatusChange('inbox')}
        >
          <X className="w-3.5 h-3.5 mr-1" />
          Reject → Inbox
        </Button>
        <Button
          size="sm"
          className="text-xs flex-1"
          onClick={() => onStatusChange('done')}
        >
          <CheckCircle2 className="w-3.5 h-3.5 mr-1" />
          Approve → Done
        </Button>
      </div>
    </div>
  )
}

function DoneView({ task, onStatusChange }: { task: BoardTask; onStatusChange: (status: string) => void }) {
  const isFailed = task.error_message != null

  return (
    <div className="space-y-4">
      {/* Status */}
      <div className={cn(
        'flex items-center gap-2 px-3 py-2 rounded-lg border',
        isFailed
          ? 'bg-destructive/10 border-destructive/20'
          : 'bg-[hsl(var(--success))]/10 border-[hsl(var(--success))]/20',
      )}>
        {isFailed ? (
          <>
            <AlertCircle className="w-4 h-4 text-destructive" />
            <span className="text-sm font-medium text-destructive">Failed</span>
          </>
        ) : (
          <>
            <CheckCircle2 className="w-4 h-4 text-[hsl(var(--success))]" />
            <span className="text-sm font-medium text-[hsl(var(--success))]">Completed</span>
          </>
        )}
        {task.completed_at && (
          <span className="text-xs text-muted-foreground ml-auto">
            {formatDistanceToNow(new Date(task.completed_at), { addSuffix: true })}
          </span>
        )}
      </div>

      {/* Error */}
      {task.error_message && (
        <div>
          <h4 className="text-xs font-medium text-destructive uppercase tracking-wider mb-1">Error</h4>
          <div className="rounded-lg bg-destructive/10 border border-destructive/20 p-3">
            <p className="text-xs text-destructive font-mono whitespace-pre-wrap">{task.error_message}</p>
          </div>
        </div>
      )}

      {/* Agent result */}
      {task.result && (
        <div>
          <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Result</h4>
          <div className="glass-card rounded-lg p-3 text-sm whitespace-pre-wrap max-h-[400px] overflow-y-auto">
            {String(task.result)}
          </div>
        </div>
      )}

      {/* Original description */}
      {task.description && !task.result && (
        <div>
          <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Task Description</h4>
          <p className="text-sm text-muted-foreground whitespace-pre-wrap">{task.description}</p>
        </div>
      )}

      {/* Timestamps */}
      <div className="grid grid-cols-2 gap-3">
        {task.started_at && (
          <div>
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Started</p>
            <p className="text-sm">{format(new Date(task.started_at), 'MMM d, HH:mm')}</p>
          </div>
        )}
        {task.completed_at && (
          <div>
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Completed</p>
            <p className="text-sm">{format(new Date(task.completed_at), 'MMM d, HH:mm')}</p>
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="flex gap-2 pt-2">
        <Button
          variant="outline"
          size="sm"
          className="text-xs"
          onClick={() => onStatusChange('in_progress')}
        >
          <RotateCcw className="w-3.5 h-3.5 mr-1" />
          Re-run
        </Button>
      </div>
    </div>
  )
}

// ── Main Viewer ──────────────────────────────────────────────────────

export function BoardTaskViewer({ task: propTask, onClose, className }: BoardTaskViewerProps) {
  const task = useLiveTask(propTask)
  const statusConf = STATUS_CONFIG[task.status] ?? STATUS_CONFIG['inbox']
  const priorityConf = PRIORITY_CONFIG[task.priority]
  const updateStatus = useUpdateTaskStatus()

  const handleStatusChange = (newStatus: string) => {
    updateStatus.mutate({ taskId: task.id, status: newStatus as any })
  }

  return (
    <div className={cn(
      'fixed inset-y-0 right-0 w-[480px] max-w-full bg-background border-l border-border shadow-2xl z-50',
      'flex flex-col animate-in slide-in-from-right duration-200',
      className,
    )}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <button onClick={onClose} className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors">
          <ArrowLeft className="w-3.5 h-3.5" />
          Back
        </button>
        <div className="flex items-center gap-2">
          {task.status === 'in_progress' && (
            <Loader2 className="w-3 h-3 text-[hsl(var(--info))] animate-spin" />
          )}
          <div className={cn('w-2 h-2 rounded-full', statusConf.dotColor)} />
          <span className="text-xs font-medium text-muted-foreground">{statusConf.label}</span>
        </div>
      </div>

      {/* Title section */}
      <div className="px-4 py-3 border-b border-border/30">
        <h2 className="text-base font-semibold mb-1">{task.name}</h2>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          {task.assignee && (
            <>
              {task.assignee.agent_icon ? (
                <PremiumIcon name={task.assignee.agent_icon} size={14} />
              ) : (
                <Bot className="w-3.5 h-3.5 text-primary" />
              )}
              <span>{task.assignee.agent_name}</span>
              <span>·</span>
            </>
          )}
          <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: priorityConf.color }} />
          <span className="capitalize">{task.priority}</span>
          {task.started_at && (
            <>
              <span>·</span>
              <span>{format(new Date(task.started_at), 'MMM d, HH:mm')}</span>
            </>
          )}
        </div>
      </div>

      {/* Content — adapts based on status */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        {(task.status === 'inbox' || task.status === 'assigned') && <InboxView task={task} />}
        {task.status === 'in_progress' && <InProgressView task={task} />}
        {task.status === 'review' && <ReviewView task={task} onStatusChange={handleStatusChange} />}
        {task.status === 'done' && <DoneView task={task} onStatusChange={handleStatusChange} />}
      </div>
    </div>
  )
}
