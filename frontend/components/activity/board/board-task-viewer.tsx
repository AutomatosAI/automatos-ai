'use client'

import { Bot, Clock, CheckCircle2, AlertCircle, RotateCcw, Loader2, FileText, ExternalLink, Tag, Calendar, User, Shield, Workflow, Play, TerminalSquare } from 'lucide-react'
import { sessionDenials, denialLine, reviewReason } from './session-denials'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { PremiumIcon } from '@/components/shared'
import { formatDistanceToNow, format } from 'date-fns'
import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import type { BoardTask } from '@/types/board'
import { PRIORITY_CONFIG, STATUS_CONFIG } from '@/types/board'
import { useUpdateTaskStatus, useApproveTask, useRejectTask, useRunTask } from '@/hooks/use-board-tasks'
import { useRouter } from 'next/navigation'
import { cn } from '@/lib/utils'

interface BoardTaskViewerProps {
  task: BoardTask | null
  open: boolean
  onOpenChange: (open: boolean) => void
}

/**
 * Poll the backend for live task data when in_progress.
 * Falls back to the prop task when the query hasn't loaded yet.
 */
function useLiveTask(task: BoardTask | null) {
  const isActive = task?.status === 'in_progress'

  const { data } = useQuery({
    queryKey: ['board-task-live', task?.id],
    queryFn: () => apiClient.request<any>(`/api/v1/tasks/${task?.source_id || task?.id}`),
    enabled: !!task && isActive,
    refetchInterval: isActive ? 5000 : false,
    staleTime: 3000,
  })

  if (!task) return null
  if (!data) return task

  return {
    ...task,
    status: data.status ?? task.status,
    result: data.result ?? task.result,
    error_message: data.error_message ?? task.error_message,
    started_at: data.started_at ?? task.started_at,
    completed_at: data.completed_at ?? task.completed_at,
    step_progress: data.step_progress ?? task.step_progress,
    runtime_ref: data.runtime_ref ?? task.runtime_ref,
  } as BoardTask
}

function formatElapsed(startedAt: string): string {
  return formatDistanceToNow(new Date(startedAt), { addSuffix: false })
}

// ── PRD-234: the session behind a `runtime: cli` ticket ─────────────────────

function SessionBlock({ task }: { task: BoardTask }) {
  const ref = task.runtime_ref
  if (!ref || ref.runtime !== 'cli') return null
  const files: string[] = Array.isArray(ref.files_touched) ? ref.files_touched : []
  const usage = ref.usage || {}
  const takeover = ref.session_id
    ? `${ref.cwd ? `cd ${ref.cwd} && ` : ''}claude --resume ${ref.session_id}`
    : null
  return (
    <div>
      <SectionLabel icon={<TerminalSquare className="w-3 h-3" />}>Claude Code session</SectionLabel>
      <div className="glass-card rounded-lg p-4 text-sm space-y-2">
        <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs">
          <span className="text-muted-foreground">Provider · model</span>
          <span>{ref.provider || 'claude'}{ref.model ? ` · ${ref.model}` : ''}</span>
          <span className="text-muted-foreground">State</span>
          <span>{ref.exit_reason ? `finished (${ref.exit_reason})` : ref.live_tool ? `running · ${ref.live_tool}` : ref.last_event ? `running · ${ref.last_event}` : 'claimed'}</span>
          {usage.total_tokens != null && (
            <>
              <span className="text-muted-foreground">Tokens</span>
              <span>{Number(usage.total_tokens).toLocaleString()}{usage.model ? ` on ${usage.model}` : ''} · plan usage, no cost</span>
            </>
          )}
          {typeof ref.denials === 'number' && ref.denials > 0 && (
            <>
              <span className="text-muted-foreground">Refused tool calls</span>
              <span>{ref.denials}</span>
            </>
          )}
        </div>
        {reviewReason(ref) && (
          <div className="rounded-md border border-[hsl(var(--warning))]/40 bg-[hsl(var(--warning))]/10 p-2 text-xs space-y-1">
            <p>{reviewReason(ref)}</p>
            {sessionDenials(ref).length > 0 && (
              <ul className="font-mono space-y-0.5 max-h-32 overflow-y-auto">
                {sessionDenials(ref).map((d, i) => (
                  <li key={i} className="truncate" title={denialLine(d)}>{denialLine(d)}</li>
                ))}
              </ul>
            )}
          </div>
        )}
        {files.length > 0 && (
          <div>
            <p className="text-xs text-muted-foreground mb-1">Files touched</p>
            <ul className="text-xs font-mono space-y-0.5 max-h-32 overflow-y-auto">
              {files.map((f) => <li key={f} className="truncate" title={f}>{f}</li>)}
            </ul>
          </div>
        )}
        {takeover && (
          <div>
            <p className="text-xs text-muted-foreground mb-1">Take over in your terminal</p>
            <code className="block rounded bg-muted px-2 py-1.5 font-mono text-xs overflow-x-auto">{takeover}</code>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Metadata grid — shared across views ──────────────────────────────

function MetadataGrid({ task }: { task: BoardTask }) {
  const priorityConf = PRIORITY_CONFIG[task.priority]

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
      <MetaItem icon={<User className="w-3.5 h-3.5" />} label="Assignee">
        {task.assignee ? (
          <div className="flex items-center gap-1.5">
            {task.assignee.agent_icon ? (
              <PremiumIcon name={task.assignee.agent_icon} size={16} />
            ) : (
              <Bot className="w-4 h-4 text-primary" />
            )}
            <span className="text-sm font-medium">{task.assignee.agent_name}</span>
          </div>
        ) : (
          <span className="text-sm text-muted-foreground">Unassigned</span>
        )}
      </MetaItem>

      <MetaItem icon={<div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: priorityConf.color }} />} label="Priority">
        <span className="text-sm font-medium capitalize">{task.priority}</span>
      </MetaItem>

      <MetaItem icon={<Shield className="w-3.5 h-3.5" />} label="Review Mode">
        <span className="text-sm font-medium capitalize">{task.review_mode}</span>
      </MetaItem>

      {task.due_date ? (
        <MetaItem icon={<Calendar className="w-3.5 h-3.5" />} label="Due Date">
          <span className="text-sm font-medium">{format(new Date(task.due_date), 'MMM d, yyyy')}</span>
        </MetaItem>
      ) : (
        <MetaItem icon={<Calendar className="w-3.5 h-3.5" />} label="Created">
          <span className="text-sm font-medium">
            {task.started_at ? format(new Date(task.started_at), 'MMM d, yyyy') : '--'}
          </span>
        </MetaItem>
      )}
    </div>
  )
}

function MetaItem({ icon, label, children }: { icon: React.ReactNode; label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-1.5 text-muted-foreground">
        {icon}
        <span className="text-[10px] font-medium uppercase tracking-wider">{label}</span>
      </div>
      {children}
    </div>
  )
}

// ── Status-specific content panels ───────────────────────────────────

function AssignedContent({ task }: { task: BoardTask }) {
  return (
    <div className="space-y-6">
      <MetadataGrid task={task} />

      {task.description && (
        <div>
          <SectionLabel>Description</SectionLabel>
          <div className="glass-card rounded-lg p-4 text-sm whitespace-pre-wrap leading-relaxed">
            {task.description}
          </div>
        </div>
      )}

      {task.tags.length > 0 && (
        <div>
          <SectionLabel icon={<Tag className="w-3 h-3" />}>Tags</SectionLabel>
          <div className="flex flex-wrap gap-1.5">
            {task.tags.map((tag) => (
              <span key={tag} className="text-xs px-2.5 py-1 rounded-full bg-secondary/50 text-muted-foreground border border-border/30">
                {tag}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function InProgressContent({ task }: { task: BoardTask }) {
  return (
    <div className="space-y-6">
      {/* Live status banner */}
      <div className="flex items-center gap-3 px-4 py-3 rounded-lg bg-[hsl(var(--info))]/10 border border-[hsl(var(--info))]/20">
        <Loader2 className="w-5 h-5 text-[hsl(var(--info))] animate-spin shrink-0" />
        <div className="flex-1">
          <p className="text-sm font-medium text-[hsl(var(--info))]">Agent is working on this task...</p>
          {task.started_at && (
            <p className="text-xs text-muted-foreground mt-0.5">{formatElapsed(task.started_at)} elapsed</p>
          )}
        </div>
      </div>

      {/* Step progress */}
      {task.step_progress && (
        <div>
          <div className="flex items-center justify-between mb-2">
            <SectionLabel>Progress</SectionLabel>
            <span className="text-xs text-muted-foreground font-mono">
              {task.step_progress.current}/{task.step_progress.total}
            </span>
          </div>
          <div className="h-2.5 bg-secondary/50 rounded-full overflow-hidden">
            <div
              className="h-full bg-[hsl(var(--info))] rounded-full transition-all duration-300"
              style={{ width: `${(task.step_progress.current / task.step_progress.total) * 100}%` }}
            />
          </div>
        </div>
      )}

      <MetadataGrid task={task} />

      {task.description && (
        <div>
          <SectionLabel>Task Description</SectionLabel>
          <div className="glass-card rounded-lg p-4 text-sm whitespace-pre-wrap text-muted-foreground leading-relaxed">
            {task.description}
          </div>
        </div>
      )}

      {/* Live output */}
      <SessionBlock task={task} />

      {task.result && (
        <div>
          <SectionLabel>Live Output</SectionLabel>
          <div className="glass-card rounded-lg p-4 text-xs font-mono whitespace-pre-wrap max-h-[400px] overflow-y-auto border border-[hsl(var(--info))]/10">
            {String(task.result)}
          </div>
        </div>
      )}
    </div>
  )
}

function ReviewContent({ task, onStatusChange, onApprove, onReject }: { task: BoardTask; onStatusChange: (status: string) => void; onApprove: () => void; onReject: (feedback?: string) => void }) {
  const hasApprovalAction = !!task.planning_data?.approval_action
  return (
    <div className="space-y-6">
      {/* Review banner */}
      <div className="flex items-center gap-3 px-4 py-3 rounded-lg bg-[hsl(var(--warning))]/10 border border-[hsl(var(--warning))]/20">
        <Clock className="w-5 h-5 text-[hsl(var(--warning))] shrink-0" />
        <div className="flex-1">
          <p className="text-sm font-medium text-[hsl(var(--warning))]">Awaiting Review</p>
          {task.completed_at && (
            <p className="text-xs text-muted-foreground mt-0.5">
              Completed {formatDistanceToNow(new Date(task.completed_at), { addSuffix: true })}
            </p>
          )}
        </div>
      </div>

      <SessionBlock task={task} />

      {/* Agent result — the main attraction */}
      {task.result ? (
        <div>
          <SectionLabel icon={<FileText className="w-3 h-3" />}>Agent Result</SectionLabel>
          <div className="glass-card rounded-lg p-5 text-sm whitespace-pre-wrap max-h-[500px] overflow-y-auto leading-relaxed">
            {String(task.result)}
          </div>
        </div>
      ) : (
        <div className="glass-card rounded-lg p-5 text-center text-muted-foreground">
          <FileText className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">No result data available yet.</p>
          <p className="text-xs mt-1">The agent may still be processing or the result was empty.</p>
        </div>
      )}

      {/* Original task for reference */}
      {task.description && (
        <div>
          <SectionLabel>Original Task</SectionLabel>
          <div className="glass-card rounded-lg p-4 text-sm text-muted-foreground whitespace-pre-wrap leading-relaxed">
            {task.description}
          </div>
        </div>
      )}

      {/* Execution timeline */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
        {task.started_at && (
          <MetaItem icon={<Clock className="w-3.5 h-3.5" />} label="Started">
            <span className="text-sm font-medium">{format(new Date(task.started_at), 'MMM d, HH:mm')}</span>
          </MetaItem>
        )}
        {task.completed_at && (
          <MetaItem icon={<CheckCircle2 className="w-3.5 h-3.5" />} label="Completed">
            <span className="text-sm font-medium">{format(new Date(task.completed_at), 'MMM d, HH:mm')}</span>
          </MetaItem>
        )}
        {task.started_at && task.completed_at && (
          <MetaItem icon={<Clock className="w-3.5 h-3.5" />} label="Duration">
            <span className="text-sm font-medium">
              {formatDistanceToNow(new Date(task.started_at), { addSuffix: false })}
            </span>
          </MetaItem>
        )}
      </div>

      {/* Approval action info */}
      {hasApprovalAction && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-primary/5 border border-primary/20">
          <CheckCircle2 className="w-4 h-4 text-primary shrink-0" />
          <p className="text-xs text-muted-foreground">
            <span className="font-medium text-foreground">Approval action:</span>{' '}
            {task.planning_data?.approval_action?.type === 'publish_blog'
              ? 'Publish blog post to live site'
              : task.planning_data?.approval_action?.type ?? 'Execute action'}
          </p>
        </div>
      )}

      {/* Review actions */}
      <div className="flex gap-3 pt-2 border-t border-border/30">
        <Button
          variant="outline"
          className="flex-1"
          onClick={() => onReject()}
        >
          <AlertCircle className="w-4 h-4 mr-2" />
          Reject
        </Button>
        <Button
          className="flex-1"
          onClick={onApprove}
        >
          <CheckCircle2 className="w-4 h-4 mr-2" />
          {hasApprovalAction ? 'Approve & Publish' : 'Approve'}
        </Button>
      </div>
    </div>
  )
}

function DoneContent({ task, onStatusChange }: { task: BoardTask; onStatusChange: (status: string) => void }) {
  const isFailed = task.error_message != null

  return (
    <div className="space-y-6">
      {/* Status banner */}
      <div className={cn(
        'flex items-center gap-3 px-4 py-3 rounded-lg border',
        isFailed
          ? 'bg-destructive/10 border-destructive/20'
          : 'bg-[hsl(var(--success))]/10 border-[hsl(var(--success))]/20',
      )}>
        {isFailed ? (
          <AlertCircle className="w-5 h-5 text-destructive shrink-0" />
        ) : (
          <CheckCircle2 className="w-5 h-5 text-[hsl(var(--success))] shrink-0" />
        )}
        <div className="flex-1">
          <p className={cn('text-sm font-medium', isFailed ? 'text-destructive' : 'text-[hsl(var(--success))]')}>
            {isFailed ? 'Task Failed' : 'Task Completed'}
          </p>
          {task.completed_at && (
            <p className="text-xs text-muted-foreground mt-0.5">
              {formatDistanceToNow(new Date(task.completed_at), { addSuffix: true })}
            </p>
          )}
        </div>
      </div>

      {/* Error details */}
      {task.error_message && (
        <div>
          <SectionLabel>Error Details</SectionLabel>
          <div className="rounded-lg bg-destructive/10 border border-destructive/20 p-4">
            <p className="text-sm text-destructive font-mono whitespace-pre-wrap">{task.error_message}</p>
          </div>
        </div>
      )}

      {/* Result */}
      {task.result && (
        <div>
          <SectionLabel icon={<FileText className="w-3 h-3" />}>Result</SectionLabel>
          <div className="glass-card rounded-lg p-5 text-sm whitespace-pre-wrap max-h-[500px] overflow-y-auto leading-relaxed">
            {String(task.result)}
          </div>
        </div>
      )}

      {/* Original description (only if no result to show) */}
      {task.description && !task.result && (
        <div>
          <SectionLabel>Task Description</SectionLabel>
          <div className="glass-card rounded-lg p-4 text-sm text-muted-foreground whitespace-pre-wrap leading-relaxed">
            {task.description}
          </div>
        </div>
      )}

      {/* Timestamps */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
        {task.started_at && (
          <MetaItem icon={<Clock className="w-3.5 h-3.5" />} label="Started">
            <span className="text-sm font-medium">{format(new Date(task.started_at), 'MMM d, HH:mm')}</span>
          </MetaItem>
        )}
        {task.completed_at && (
          <MetaItem icon={<CheckCircle2 className="w-3.5 h-3.5" />} label="Completed">
            <span className="text-sm font-medium">{format(new Date(task.completed_at), 'MMM d, HH:mm')}</span>
          </MetaItem>
        )}
      </div>

      {/* Actions */}
      {task.type !== 'playbook' && (
        <div className="flex gap-3 pt-2 border-t border-border/30">
          <Button
            variant="outline"
            onClick={() => onStatusChange('in_progress')}
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Re-run Task
          </Button>
        </div>
      )}
    </div>
  )
}

// ── Helpers ──────────────────────────────────────────────────────────

function SectionLabel({ children, icon }: { children: React.ReactNode; icon?: React.ReactNode }) {
  return (
    <div className="flex items-center gap-1.5 mb-2">
      {icon && <span className="text-muted-foreground">{icon}</span>}
      <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">{children}</h4>
    </div>
  )
}

function ExecutionKitchenLink({ task, onClose }: { task: BoardTask; onClose: () => void }) {
  const router = useRouter()
  if (task.type !== 'playbook' || !task.planning_data?.execution_id) return null

  const execId = task.planning_data.execution_id
  const playbookId = task.planning_data.playbook_id

  return (
    <Button
      variant="outline"
      className="w-full"
      onClick={() => {
        onClose()
        router.push(`/activity/execution?id=${execId}&recipeId=${String(playbookId ?? '')}` as any)
      }}
    >
      <ExternalLink className="w-4 h-4 mr-2" />
      View in Execution Kitchen
    </Button>
  )
}

// ── Main Modal ──────────────────────────────────────────────────────

export function BoardTaskViewer({ task: propTask, open, onOpenChange }: BoardTaskViewerProps) {
  const task = useLiveTask(propTask)
  const updateStatus = useUpdateTaskStatus()
  const approveTask = useApproveTask()
  const rejectTask = useRejectTask()
  const runTask = useRunTask()

  if (!task) return null

  const statusConf = STATUS_CONFIG[task.status] ?? STATUS_CONFIG['inbox']
  const priorityConf = PRIORITY_CONFIG[task.priority]

  const handleStatusChange = (newStatus: string) => {
    updateStatus.mutate({ taskId: task.id, status: newStatus as any })
    if (newStatus === 'done') {
      onOpenChange(false)
    }
  }

  const handleApprove = () => {
    approveTask.mutate({ taskId: task.id }, {
      onSuccess: () => onOpenChange(false),
    })
  }

  const handleReject = (feedback?: string) => {
    rejectTask.mutate({ taskId: task.id, feedback })
  }

  // PRD-161 S5: re-dispatch a failed/idle/blocked task on demand.
  const handleRunNow = () => {
    runTask.mutate({ taskId: task.id })
  }
  const canRunNow =
    task.status === 'assigned' ||
    task.status === 'inbox' ||
    task.status === 'failed' ||
    task.status === 'blocked'

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[760px] md:max-w-[860px] max-h-[85vh] overflow-hidden flex flex-col p-0">
        {/* Header */}
        <div className="px-6 pt-5 pb-4 border-b border-border/30">
          <div className="flex items-start justify-between gap-4 pr-8">
            <div className="space-y-2 min-w-0 flex-1">
              <DialogTitle className="text-lg font-semibold leading-tight">
                {task.name}
              </DialogTitle>
              <DialogDescription className="flex items-center gap-2 flex-wrap">
                {/* Status badge */}
                <span className={cn(
                  'inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-0.5 rounded-full border',
                  task.status === 'in_progress' && 'bg-[hsl(var(--info))]/10 border-[hsl(var(--info))]/30 text-[hsl(var(--info))]',
                  task.status === 'review' && 'bg-[hsl(var(--warning))]/10 border-[hsl(var(--warning))]/30 text-[hsl(var(--warning))]',
                  task.status === 'done' && 'bg-[hsl(var(--success))]/10 border-[hsl(var(--success))]/30 text-[hsl(var(--success))]',
                  (task.status === 'inbox' || task.status === 'assigned') && 'bg-secondary/50 border-border/30 text-muted-foreground',
                )}>
                  {task.status === 'in_progress' && <Loader2 className="w-3 h-3 animate-spin" />}
                  <div className={cn('w-1.5 h-1.5 rounded-full', statusConf.dotColor)} />
                  {statusConf.label}
                </span>

                {/* Priority badge */}
                <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: priorityConf.color }} />
                  <span className="capitalize">{task.priority}</span>
                </span>

                {/* Playbook badge */}
                {task.type === 'playbook' && (
                  <span className="inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full bg-primary/10 border border-primary/20 text-primary">
                    <Workflow className="w-3 h-3" />
                    Playbook
                  </span>
                )}

                {/* Agent */}
                {task.assignee && (
                  <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
                    {task.assignee.agent_icon ? (
                      <PremiumIcon name={task.assignee.agent_icon} size={14} />
                    ) : (
                      <Bot className="w-3.5 h-3.5 text-primary" />
                    )}
                    {task.assignee.agent_name}
                  </span>
                )}
              </DialogDescription>
            </div>
          </div>
        </div>

        {/* Scrollable content area */}
        <div className="flex-1 overflow-y-auto px-6 py-5">
          {(task.status === 'inbox' || task.status === 'assigned') && <AssignedContent task={task} />}
          {task.status === 'in_progress' && <InProgressContent task={task} />}
          {task.status === 'review' && <ReviewContent task={task} onStatusChange={handleStatusChange} onApprove={handleApprove} onReject={handleReject} />}
          {task.status === 'done' && <DoneContent task={task} onStatusChange={handleStatusChange} />}

          {/* PRD-161 S5: failed tasks surface the error + a re-run affordance. */}
          {task.status === 'failed' && task.error_message && (
            <div className="rounded-lg border border-[hsl(var(--destructive))]/30 bg-[hsl(var(--destructive))]/5 p-3 text-sm text-[hsl(var(--destructive))]">
              {task.error_message}
            </div>
          )}

          {/* PRD-161 S5: Run Now — dispatch the task immediately through the board loop. */}
          {canRunNow && (
            <div className="mt-4">
              <Button
                onClick={handleRunNow}
                disabled={runTask.isPending || !task.assignee}
                className="w-full"
              >
                <Play className="w-4 h-4 mr-2" />
                {runTask.isPending ? 'Dispatching…' : 'Run Now'}
              </Button>
              {!task.assignee && (
                <p className="text-xs text-muted-foreground mt-1.5 text-center">
                  Assign an agent to run this task.
                </p>
              )}
            </div>
          )}

          {/* Deep link to Execution Kitchen for recipe tasks */}
          <div className="mt-4">
            <ExecutionKitchenLink task={task} onClose={() => onOpenChange(false)} />
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
