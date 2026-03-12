'use client'

import { Draggable } from '@hello-pangea/dnd'
import { Bot, AlertCircle } from 'lucide-react'
import { PremiumIcon } from '@/components/shared'
import { formatDistanceToNow } from 'date-fns'
import type { BoardTask } from '@/types/board'
import { PRIORITY_CONFIG } from '@/types/board'
import { cn } from '@/lib/utils'

interface BoardCardProps {
  task: BoardTask
  index: number
  onOpen: (task: BoardTask) => void
}

export function BoardCard({ task, index, onOpen }: BoardCardProps) {
  const priorityConf = PRIORITY_CONFIG[task.priority]
  const timeAgo = task.started_at
    ? formatDistanceToNow(new Date(task.started_at), { addSuffix: false })
    : task.completed_at
      ? formatDistanceToNow(new Date(task.completed_at), { addSuffix: false })
      : null

  const isFailed = task.error_message != null && task.status === 'done'

  return (
    <Draggable draggableId={task.id} index={index}>
      {(provided, snapshot) => (
        <div
          ref={provided.innerRef}
          {...provided.draggableProps}
          {...provided.dragHandleProps}
          className={cn(
            'glass-card rounded-lg p-3 mb-2 cursor-grab active:cursor-grabbing transition-shadow',
            'border-l-2',
            snapshot.isDragging && 'shadow-lg shadow-primary/10 ring-1 ring-primary/20',
            isFailed && 'border-l-destructive',
            !isFailed && `border-l-[${priorityConf.color}]`,
          )}
          style={{
            ...provided.draggableProps.style,
            borderLeftColor: isFailed ? undefined : priorityConf.color,
          }}
          onClick={() => onOpen(task)}
        >
          {/* Title */}
          <p className="text-sm font-medium line-clamp-2 mb-1">{task.name}</p>

          {/* Description */}
          {task.description && (
            <p className="text-xs text-muted-foreground line-clamp-2 mb-2">
              {task.description}
            </p>
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

          {/* Footer: agent + time + open button */}
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

            {timeAgo && (
              <span className="text-[10px] text-muted-foreground shrink-0">{timeAgo}</span>
            )}
          </div>
        </div>
      )}
    </Draggable>
  )
}
