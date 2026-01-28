'use client'

/**
 * StepDetail Component for PRD-38.2 Extended Widgets
 *
 * Expanded view of a single workflow step with result/error details
 */

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  CheckCircle2,
  XCircle,
  Clock,
  Loader2,
  MinusCircle,
  Timer,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { WorkflowStep } from '../types'

interface StepDetailProps {
  step: WorkflowStep | null
  open: boolean
  onOpenChange: (open: boolean) => void
}

/**
 * Get status icon and color
 */
function getStepStatusConfig(status: WorkflowStep['status']) {
  switch (status) {
    case 'pending':
      return {
        icon: Clock,
        className: 'text-gray-400',
        label: 'Pending',
      }
    case 'running':
      return {
        icon: Loader2,
        className: 'text-blue-500 animate-spin',
        label: 'Running',
      }
    case 'completed':
      return {
        icon: CheckCircle2,
        className: 'text-green-500',
        label: 'Completed',
      }
    case 'failed':
      return {
        icon: XCircle,
        className: 'text-red-500',
        label: 'Failed',
      }
    case 'skipped':
      return {
        icon: MinusCircle,
        className: 'text-gray-400',
        label: 'Skipped',
      }
    default:
      return {
        icon: Clock,
        className: 'text-gray-400',
        label: 'Unknown',
      }
  }
}

/**
 * Format duration
 */
function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(2)}s`
  const mins = Math.floor(ms / 60000)
  const secs = ((ms % 60000) / 1000).toFixed(1)
  return `${mins}m ${secs}s`
}

export function StepDetail({ step, open, onOpenChange }: StepDetailProps) {
  if (!step) return null

  const config = getStepStatusConfig(step.status)
  const StatusIcon = config.icon

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <StatusIcon className={cn('h-5 w-5', config.className)} />
            {step.name}
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          {/* Step info */}
          <div className="flex items-center gap-2">
            <Badge variant="outline">{step.type}</Badge>
            <Badge variant="secondary" className={cn(
              step.status === 'completed' && 'bg-green-100 text-green-700',
              step.status === 'failed' && 'bg-red-100 text-red-700',
              step.status === 'running' && 'bg-blue-100 text-blue-700'
            )}>
              {config.label}
            </Badge>
            {step.duration !== undefined && (
              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                <Timer className="h-3.5 w-3.5" />
                {formatDuration(step.duration)}
              </div>
            )}
          </div>

          {/* Timing info */}
          {(step.startedAt || step.completedAt) && (
            <div className="text-sm text-muted-foreground space-y-1">
              {step.startedAt && (
                <div>
                  Started: {new Date(step.startedAt).toLocaleString()}
                </div>
              )}
              {step.completedAt && (
                <div>
                  Completed: {new Date(step.completedAt).toLocaleString()}
                </div>
              )}
            </div>
          )}

          {/* Error message */}
          {step.error && (
            <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
              <h4 className="text-sm font-medium text-red-800 dark:text-red-200 mb-1">
                Error
              </h4>
              <pre className="text-xs text-red-700 dark:text-red-300 whitespace-pre-wrap overflow-auto max-h-32">
                {step.error}
              </pre>
            </div>
          )}

          {/* Result */}
          {step.result !== undefined && (
            <div>
              <h4 className="text-sm font-medium mb-2">Result</h4>
              <ScrollArea className="h-[200px] rounded-md border">
                <pre className="p-3 text-xs font-mono whitespace-pre-wrap">
                  {typeof step.result === 'object'
                    ? JSON.stringify(step.result, null, 2)
                    : String(step.result)}
                </pre>
              </ScrollArea>
            </div>
          )}

          {/* Nested steps */}
          {step.children && step.children.length > 0 && (
            <div>
              <h4 className="text-sm font-medium mb-2">
                Sub-steps ({step.children.length})
              </h4>
              <div className="space-y-1">
                {step.children.map((child) => {
                  const childConfig = getStepStatusConfig(child.status)
                  const ChildIcon = childConfig.icon
                  return (
                    <div
                      key={child.id}
                      className="flex items-center gap-2 text-sm pl-4"
                    >
                      <ChildIcon className={cn('h-4 w-4', childConfig.className)} />
                      <span>{child.name}</span>
                      {child.duration !== undefined && (
                        <span className="text-xs text-muted-foreground">
                          {formatDuration(child.duration)}
                        </span>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
