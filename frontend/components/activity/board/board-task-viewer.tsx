'use client'

import { X, Bot, Clock, CheckCircle2, AlertCircle, ArrowLeft, RotateCcw, Download, Star } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { PremiumIcon } from '@/components/shared'
import { formatDistanceToNow, format } from 'date-fns'
import type { BoardTask } from '@/types/board'
import { PRIORITY_CONFIG, STATUS_CONFIG } from '@/types/board'
import { cn } from '@/lib/utils'

interface BoardTaskViewerProps {
  task: BoardTask
  onClose: () => void
  className?: string
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

function InboxView({ task }: { task: BoardTask }) {
  const priorityConf = PRIORITY_CONFIG[task.priority]

  return (
    <div className="space-y-4">
      {task.description && (
        <div>
          <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Description</h4>
          <p className="text-sm">{task.description}</p>
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

      {task.step_progress && (
        <div>
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Steps</p>
          <span className="text-sm">{task.step_progress.total} steps defined</span>
        </div>
      )}
    </div>
  )
}

function InProgressView({ task }: { task: BoardTask }) {
  return (
    <div className="space-y-4">
      {/* Progress */}
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

      {/* Runtime */}
      {task.started_at && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Clock className="w-4 h-4" />
          <span>Running for {formatDistanceToNow(new Date(task.started_at))}</span>
        </div>
      )}

      {task.description && (
        <div>
          <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Description</h4>
          <p className="text-sm">{task.description}</p>
        </div>
      )}
    </div>
  )
}

function ReviewView({ task }: { task: BoardTask }) {
  return (
    <div className="space-y-4">
      {/* Completion info */}
      <div className="flex items-center gap-2 text-sm">
        <CheckCircle2 className="w-4 h-4 text-[hsl(var(--success))]" />
        <span>Completed in {formatDuration(task.duration_ms)}</span>
        <span className="text-muted-foreground">— Awaiting Review</span>
      </div>

      {task.description && (
        <div>
          <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Report</h4>
          <div className="glass-card rounded-lg p-3 text-sm whitespace-pre-wrap">
            {task.description}
          </div>
        </div>
      )}

      {/* Review actions */}
      <div className="flex gap-2 pt-2">
        <Button variant="outline" size="sm" className="text-xs flex-1">
          <X className="w-3.5 h-3.5 mr-1" />
          Reject → Inbox
        </Button>
        <Button size="sm" className="text-xs flex-1">
          <CheckCircle2 className="w-3.5 h-3.5 mr-1" />
          Approve → Done
        </Button>
      </div>
    </div>
  )
}

function DoneView({ task }: { task: BoardTask }) {
  const isFailed = task.error_message != null

  return (
    <div className="space-y-4">
      {/* Status */}
      <div className="flex items-center gap-2 text-sm">
        {isFailed ? (
          <>
            <AlertCircle className="w-4 h-4 text-destructive" />
            <span className="text-destructive">Failed</span>
          </>
        ) : (
          <>
            <CheckCircle2 className="w-4 h-4 text-[hsl(var(--success))]" />
            <span>Completed</span>
          </>
        )}
        {task.duration_ms && (
          <span className="text-muted-foreground">in {formatDuration(task.duration_ms)}</span>
        )}
      </div>

      {/* Error */}
      {task.error_message && (
        <div className="rounded-lg bg-destructive/10 border border-destructive/20 p-3">
          <p className="text-xs text-destructive">{task.error_message}</p>
        </div>
      )}

      {/* Results */}
      {task.description && (
        <div>
          <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Results</h4>
          <div className="glass-card rounded-lg p-3 text-sm whitespace-pre-wrap">
            {task.description}
          </div>
        </div>
      )}

      {/* Execution summary */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Duration</p>
          <p className="text-sm">{formatDuration(task.duration_ms)}</p>
        </div>
        {task.step_progress && (
          <div>
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Steps</p>
            <p className="text-sm">{task.step_progress.current}/{task.step_progress.total} completed</p>
          </div>
        )}
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
        <Button variant="outline" size="sm" className="text-xs">
          <RotateCcw className="w-3.5 h-3.5 mr-1" />
          Re-run
        </Button>
        <Button variant="outline" size="sm" className="text-xs">
          <Download className="w-3.5 h-3.5 mr-1" />
          Download
        </Button>
      </div>
    </div>
  )
}

export function BoardTaskViewer({ task, onClose, className }: BoardTaskViewerProps) {
  const statusConf = STATUS_CONFIG[task.status]
  const priorityConf = PRIORITY_CONFIG[task.priority]

  return (
    <div className={cn(
      'fixed inset-y-0 right-0 w-[420px] max-w-full bg-background border-l border-border shadow-2xl z-50',
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
        {task.status === 'review' && <ReviewView task={task} />}
        {task.status === 'done' && <DoneView task={task} />}
      </div>
    </div>
  )
}
