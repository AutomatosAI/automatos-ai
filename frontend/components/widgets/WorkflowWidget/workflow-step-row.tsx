'use client'

/**
 * One row of WorkflowSteps (PRD-38.2 Extended Widgets): the step's number, status,
 * name, error preview and duration, as a button that opens its detail.
 *
 * P260-FX-05 (M41): moved out of the `steps.map` callback, which was over the
 * function limit. The running state is the info token at 10% in every tone (a
 * dark: palette background used to replace it, and the hover with it), and the
 * keyboard focus is a ring, so it shows on the running row too.
 */

import {
  Check,
  X,
  Clock,
  Loader2,
  SkipForward,
  ChevronRight,
  GitBranch,
  Repeat,
  Split,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { WorkflowStep } from '../types'

interface WorkflowStepRowProps {
  step: WorkflowStep
  index: number
  onSelect: (step: WorkflowStep) => void
}

/**
 * Get status icon component
 */
function getStepIcon(status: WorkflowStep['status']) {
  switch (status) {
    case 'pending':
      return Clock
    case 'running':
      return Loader2
    case 'completed':
      return Check
    case 'failed':
      return X
    case 'skipped':
      return SkipForward
    default:
      return Clock
  }
}

/**
 * Get step type icon
 */
function getTypeIcon(type: WorkflowStep['type']) {
  switch (type) {
    case 'condition':
      return GitBranch
    case 'loop':
      return Repeat
    case 'parallel':
      return Split
    default:
      return null
  }
}

/**
 * Format duration in ms to human-readable string
 */
function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  const mins = Math.floor(ms / 60000)
  const secs = Math.floor((ms % 60000) / 1000)
  if (mins < 60) return `${mins}m ${secs}s`
  const hours = Math.floor(mins / 60)
  const remainingMins = mins % 60
  return `${hours}h ${remainingMins}m`
}

/**
 * Compute elapsed ms from startedAt to now (or completedAt)
 */
function computeElapsed(startedAt: string, completedAt?: string): number {
  const start = new Date(startedAt).getTime()
  const end = completedAt ? new Date(completedAt).getTime() : Date.now()
  return Math.max(0, end - start)
}

export function WorkflowStepRow({ step, index, onSelect }: WorkflowStepRowProps) {
  const StatusIcon = getStepIcon(step.status)
  const TypeIcon = getTypeIcon(step.type)
  const isRunning = step.status === 'running'
  const isCompleted = step.status === 'completed'
  const isFailed = step.status === 'failed'
  const isPending = step.status === 'pending'
  const isSkipped = step.status === 'skipped'

  return (
    <button
      className={cn(
        'w-full text-left px-3 py-2.5 hover:bg-muted/50 transition-colors',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring',
        isRunning && 'bg-info/10'
      )}
      onClick={() => onSelect(step)}
    >
      <div className="flex items-start gap-3">
        {/* Step number and status icon */}
        <div className="flex items-center gap-2 mt-0.5">
          <span className="text-xs text-muted-foreground w-4">
            {index + 1}.
          </span>
          <StatusIcon
            className={cn(
              'h-4 w-4 flex-shrink-0',
              isPending && 'text-muted-foreground',
              isRunning && 'text-info animate-spin',
              isCompleted && 'text-success',
              isFailed && 'text-destructive',
              isSkipped && 'text-muted-foreground'
            )}
          />
        </div>

        {/* Step content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span
              className={cn(
                'text-sm',
                isCompleted && 'text-muted-foreground',
                isFailed && 'text-destructive'
              )}
            >
              {step.name}
            </span>

            {/* Type icon */}
            {TypeIcon && (
              <TypeIcon className="h-3.5 w-3.5 text-muted-foreground" />
            )}

            {/* Running indicator */}
            {isRunning && (
              <span className="text-xs text-blue-600 dark:text-info">
                ← Currently running
              </span>
            )}
          </div>

          {/* Error preview */}
          {step.error && (
            <p className="text-xs text-destructive truncate mt-0.5">
              {step.error}
            </p>
          )}

          {/* Nested steps indicator */}
          {step.children && step.children.length > 0 && (
            <p className="text-xs text-muted-foreground mt-0.5">
              {step.children.length} sub-step
              {step.children.length !== 1 ? 's' : ''}
            </p>
          )}
        </div>

        {/* Duration */}
        <div className="flex items-center gap-1 text-xs text-muted-foreground">
          {step.duration !== undefined ? (
            <span>{formatDuration(step.duration)}</span>
          ) : step.startedAt ? (
            <span>{formatDuration(computeElapsed(step.startedAt, step.completedAt))}</span>
          ) : null}
          <ChevronRight className="h-4 w-4" />
        </div>
      </div>
    </button>
  )
}
