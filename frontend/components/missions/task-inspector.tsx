'use client'

import { X, Bot, Clock, Coins, AlertTriangle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import type { TaskResponse } from '@/types/missions'
import { TASK_STATE_CONFIG } from '@/types/missions'
import { formatDistanceToNow } from 'date-fns'

interface TaskInspectorProps {
  task: TaskResponse
  onClose: () => void
  className?: string
}

export function TaskInspector({ task, onClose, className }: TaskInspectorProps) {
  const stateConfig = TASK_STATE_CONFIG[task.state]

  const duration =
    task.started_at && task.completed_at
      ? formatDistanceToNow(new Date(task.started_at), { addSuffix: false })
      : task.started_at
        ? 'In progress'
        : null

  return (
    <div
      className={cn(
        'bg-background/95 backdrop-blur border border-border rounded-lg shadow-2xl',
        'flex flex-col overflow-hidden',
        className,
      )}
    >
      {/* Header */}
      <div className="p-4 border-b border-border flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-sm leading-tight">{task.title}</h3>
          <div className="flex items-center gap-2 mt-1.5">
            <Badge variant="outline" className={cn('text-[10px]', stateConfig.color)}>
              {stateConfig.label}
            </Badge>
            <span className="text-[10px] text-muted-foreground font-mono">
              #{task.sequence_number}
            </span>
          </div>
        </div>
        <Button variant="ghost" size="sm" className="h-7 w-7 p-0 shrink-0" onClick={onClose}>
          <X className="w-4 h-4" />
        </Button>
      </div>

      {/* Content */}
      <ScrollArea className="flex-1 p-4">
        <div className="space-y-4">
          {/* Agent */}
          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">Agent</p>
            <div className="flex items-center gap-2">
              <Bot className="w-4 h-4 text-[hsl(var(--agent))]" />
              <span className="text-sm">
                {task.agent_role ?? 'Unassigned'}
                {task.assigned_agent_id && (
                  <span className="text-muted-foreground"> (ID: {task.assigned_agent_id})</span>
                )}
              </span>
            </div>
          </div>

          {/* Description */}
          {task.description && (
            <div>
              <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">Description</p>
              <p className="text-sm text-muted-foreground leading-relaxed">{task.description}</p>
            </div>
          )}

          {/* Output excerpt */}
          {task.output_excerpt && (
            <div>
              <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">Output</p>
              <div className="rounded-lg bg-secondary/30 p-3 text-xs font-mono leading-relaxed whitespace-pre-wrap max-h-[200px] overflow-auto">
                {task.output_excerpt}
              </div>
            </div>
          )}

          {/* Failure info */}
          {task.failure_reason_code && (
            <div>
              <p className="text-[10px] uppercase tracking-wider text-destructive mb-1">Failure</p>
              <div className="flex items-start gap-2 rounded-lg bg-destructive/5 border border-destructive/20 p-3">
                <AlertTriangle className="w-4 h-4 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="text-xs font-medium text-destructive">{task.failure_reason_code}</p>
                  {task.failure_detail && (
                    <p className="text-xs text-muted-foreground mt-1">{task.failure_detail}</p>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Meta */}
          <div className="flex items-center gap-4 text-[11px] text-muted-foreground pt-2 border-t border-border">
            {duration && (
              <span className="flex items-center gap-1">
                <Clock className="w-3 h-3" /> {duration}
              </span>
            )}
            <span className="flex items-center gap-1">
              <Coins className="w-3 h-3" /> {task.tokens_used.toLocaleString()} tokens
            </span>
            {task.attempt_number > 1 && (
              <span>Attempt {task.attempt_number}</span>
            )}
          </div>
        </div>
      </ScrollArea>
    </div>
  )
}
