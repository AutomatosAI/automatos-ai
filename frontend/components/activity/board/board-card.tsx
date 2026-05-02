'use client'

import { Draggable } from '@hello-pangea/dnd'
import { Bot, AlertCircle, Workflow, ClipboardList, Trash2, FolderKanban, Target, Clock, ShieldAlert, Layers } from 'lucide-react'
import { PremiumIcon } from '@/components/shared'
import { formatDistanceToNow } from 'date-fns'
import type { BoardTask } from '@/types/board'
import { PRIORITY_CONFIG } from '@/types/board'
import { cn } from '@/lib/utils'

interface BoardCardProps {
  task: BoardTask
  index: number
  onOpen: (task: BoardTask) => void
  onDelete?: (taskId: string) => void
}

function SlaIndicator({ deadline }: { deadline: string }) {
  const now = Date.now()
  const sla = new Date(deadline).getTime()
  const remaining = sla - now
  const hoursLeft = remaining / (1000 * 60 * 60)

  if (remaining <= 0) {
    return (
      <span className="flex items-center gap-0.5 text-[10px] text-destructive font-medium">
        <Clock className="w-2.5 h-2.5" />
        Overdue
      </span>
    )
  }

  if (hoursLeft <= 2) {
    return (
      <span className="flex items-center gap-0.5 text-[10px] text-destructive">
        <Clock className="w-2.5 h-2.5" />
        {hoursLeft < 1 ? `${Math.round(hoursLeft * 60)}m` : `${Math.round(hoursLeft)}h`}
      </span>
    )
  }

  if (hoursLeft <= 8) {
    return (
      <span className="flex items-center gap-0.5 text-[10px] text-[hsl(var(--warning))]">
        <Clock className="w-2.5 h-2.5" />
        {Math.round(hoursLeft)}h
      </span>
    )
  }

  return null
}

export function BoardCard({ task, index, onOpen, onDelete }: BoardCardProps) {
  const priorityConf = PRIORITY_CONFIG[task.priority]
  const timeAgo = task.started_at
    ? formatDistanceToNow(new Date(task.started_at), { addSuffix: false })
    : task.completed_at
      ? formatDistanceToNow(new Date(task.completed_at), { addSuffix: false })
      : null

  const isFailed = task.error_message != null && task.status === 'done'
  const isBlocked = task.status === 'blocked'

  return (
    <Draggable draggableId={task.id} index={index}>
      {(provided, snapshot) => (
        <div
          ref={provided.innerRef}
          {...provided.draggableProps}
          {...provided.dragHandleProps}
          className={cn(
            'group glass-card card-glow rounded-lg p-3 mb-2 cursor-grab active:cursor-grabbing transition-shadow',
            'border-l-2',
            snapshot.isDragging && 'shadow-lg shadow-primary/10 ring-1 ring-primary/20',
            isBlocked && 'border-l-destructive bg-destructive/5',
            isFailed && !isBlocked && 'border-l-destructive',
            !isFailed && !isBlocked && `border-l-[${priorityConf.color}]`,
          )}
          style={{
            ...provided.draggableProps.style,
            borderLeftColor: isBlocked ? undefined : isFailed ? undefined : priorityConf.color,
          }}
          onClick={() => onOpen(task)}
        >
          {/* Type badge + delete button */}
          <div className="flex items-center justify-between mb-1">
            {task.type === 'mission' ? (
              <div className="flex items-center gap-1 min-w-0">
                <Target className="w-3 h-3 text-primary shrink-0" />
                <span className="text-[10px] font-medium text-primary uppercase tracking-wider shrink-0">Mission</span>
                {task.mission_name && (
                  <span className="text-[10px] text-primary/60 truncate" title={task.mission_name}>
                    · {task.mission_name}
                  </span>
                )}
              </div>
            ) : task.type === 'playbook' ? (
              <div className="flex items-center gap-1">
                <Workflow className="w-3 h-3 text-[hsl(var(--warning))]" />
                <span className="text-[10px] font-medium text-[hsl(var(--warning))] uppercase tracking-wider">Playbook</span>
              </div>
            ) : (task.type as string) === 'project' ? (
              <div className="flex items-center gap-1">
                <FolderKanban className="w-3 h-3 text-[hsl(var(--info))]" />
                <span className="text-[10px] font-medium text-[hsl(var(--info))] uppercase tracking-wider">
                  Project{task.project_id ? ` ${task.project_id}` : ''}
                </span>
              </div>
            ) : (
              <div className="flex items-center gap-1">
                <ClipboardList className="w-3 h-3 text-muted-foreground" />
                <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Task</span>
              </div>
            )}

            <div className="flex items-center gap-1">
              {/* Child count badge */}
              {(task.child_count ?? 0) > 0 && (
                <span className="flex items-center gap-0.5 text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary">
                  <Layers className="w-2.5 h-2.5" />
                  {task.child_count}
                </span>
              )}

              {onDelete && (
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    onDelete(task.id)
                  }}
                  className="opacity-0 group-hover:opacity-100 transition-opacity p-0.5 rounded hover:bg-destructive/10"
                  title="Delete task"
                >
                  <Trash2 className="w-3 h-3 text-muted-foreground hover:text-destructive" />
                </button>
              )}
            </div>
          </div>

          {/* Title */}
          <p className="text-sm font-medium line-clamp-2 mb-1">{task.name}</p>

          {/* Description */}
          {task.description && (
            <p className="text-xs text-muted-foreground line-clamp-2 mb-2">
              {task.description}
            </p>
          )}

          {/* Blocked reason */}
          {isBlocked && task.blocked_reason && (
            <div className="flex items-center gap-1 mb-2">
              <ShieldAlert className="w-3 h-3 text-destructive shrink-0" />
              <p className="text-[10px] text-destructive line-clamp-1">{task.blocked_reason}</p>
            </div>
          )}

          {/* Progress bar for in-progress tasks */}
          {task.step_progress && task.status === 'in_progress' && (
            <div className="mb-2">
              <div className="flex items-center justify-between text-[10px] text-muted-foreground mb-1">
                <span>Step {task.step_progress.current}/{task.step_progress.total}</span>
                <span>{Math.round((task.step_progress.current / task.step_progress.total) * 100)}%</span>
              </div>
              <div className="h-1 bg-secondary/50 rounded-full overflow-hidden">
                <div
                  className="h-full bg-[hsl(var(--info))] rounded-full transition-all"
                  style={{ width: `${(task.step_progress.current / task.step_progress.total) * 100}%` }}
                />
              </div>
            </div>
          )}

          {/* Error indicator */}
          {isFailed && (
            <div className="flex items-center gap-1 mb-2">
              <AlertCircle className="w-3 h-3 text-destructive shrink-0" />
              <p className="text-[10px] text-destructive line-clamp-1">{task.error_message}</p>
            </div>
          )}

          {/* Tags */}
          {task.tags.length > 0 && (
            <div className="flex flex-wrap gap-1 mb-2">
              {task.tags.slice(0, 3).map((tag) => (
                <span
                  key={tag}
                  className="text-[10px] px-1.5 py-0.5 rounded bg-secondary/50 text-muted-foreground"
                >
                  {tag}
                </span>
              ))}
              {task.tags.length > 3 && (
                <span className="text-[10px] text-muted-foreground">+{task.tags.length - 3}</span>
              )}
            </div>
          )}

          {/* Footer: agent + SLA + time */}
          <div className="flex items-center justify-between mt-1">
            <div className="flex items-center gap-1.5 min-w-0">
              {task.assignee ? (
                <>
                  {task.assignee.agent_icon ? (
                    <PremiumIcon name={task.assignee.agent_icon} size={14} className="shrink-0" />
                  ) : (
                    <Bot className="w-3.5 h-3.5 text-primary shrink-0" />
                  )}
                  <span className="text-[10px] text-muted-foreground truncate">
                    {task.assignee.agent_name}
                  </span>
                </>
              ) : (
                <span className="text-[10px] text-muted-foreground/50">Unassigned</span>
              )}
            </div>

            <div className="flex items-center gap-2">
              {task.sla_deadline && task.status !== 'done' && (
                <SlaIndicator deadline={task.sla_deadline} />
              )}
              {timeAgo && (
                <span className="text-[10px] text-muted-foreground shrink-0">{timeAgo}</span>
              )}
            </div>
          </div>
        </div>
      )}
    </Draggable>
  )
}
