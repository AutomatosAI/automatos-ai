'use client'

/**
 * WorkflowStatus Component for PRD-38.2 Extended Widgets
 *
 * Displays workflow status badge and control buttons
 */

import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Play,
  Pause,
  Square,
  Clock,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Loader2,
  Timer,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { WorkflowStatus as WorkflowStatusType } from '../types'

interface WorkflowStatusProps {
  status: WorkflowStatusType
  currentStepIndex?: number
  totalSteps: number
  startedAt?: string
  completedAt?: string
  onPause?: () => void
  onResume?: () => void
  onCancel?: () => void
  className?: string
}

/**
 * Status badge styling
 */
function getStatusConfig(status: WorkflowStatusType) {
  switch (status) {
    case 'pending':
      return {
        icon: Clock,
        label: 'Pending',
        variant: 'secondary' as const,
        className: 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300',
      }
    case 'running':
      return {
        icon: Loader2,
        label: 'Running',
        variant: 'default' as const,
        className: 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300',
        iconClassName: 'animate-spin',
      }
    case 'paused':
      return {
        icon: Pause,
        label: 'Paused',
        variant: 'secondary' as const,
        className: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300',
      }
    case 'completed':
      return {
        icon: CheckCircle2,
        label: 'Completed',
        variant: 'default' as const,
        className: 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300',
      }
    case 'failed':
      return {
        icon: XCircle,
        label: 'Failed',
        variant: 'destructive' as const,
        className: 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300',
      }
    case 'cancelled':
      return {
        icon: AlertCircle,
        label: 'Cancelled',
        variant: 'secondary' as const,
        className: 'bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300',
      }
    default:
      return {
        icon: Clock,
        label: 'Unknown',
        variant: 'secondary' as const,
        className: '',
      }
  }
}

/**
 * Calculate duration string
 */
function formatDuration(startedAt: string, completedAt?: string): string {
  const start = new Date(startedAt).getTime()
  const end = completedAt ? new Date(completedAt).getTime() : Date.now()
  const diffMs = end - start

  if (diffMs < 1000) return `${diffMs}ms`
  if (diffMs < 60000) return `${(diffMs / 1000).toFixed(1)}s`

  const mins = Math.floor(diffMs / 60000)
  const secs = Math.floor((diffMs % 60000) / 1000)
  if (mins < 60) return `${mins}m ${secs}s`

  const hours = Math.floor(mins / 60)
  const remainingMins = mins % 60
  return `${hours}h ${remainingMins}m`
}

export function WorkflowStatus({
  status,
  currentStepIndex,
  totalSteps,
  startedAt,
  completedAt,
  onPause,
  onResume,
  onCancel,
  className,
}: WorkflowStatusProps) {
  const config = getStatusConfig(status)
  const StatusIcon = config.icon

  const isRunning = status === 'running'
  const isPaused = status === 'paused'
  const canPause = isRunning && onPause
  const canResume = isPaused && onResume
  const canCancel = (isRunning || isPaused) && onCancel

  return (
    <div
      className={cn(
        'flex items-center justify-between px-3 py-2',
        'border-b border-border/30 bg-muted/20',
        className
      )}
    >
      {/* Status info */}
      <div className="flex items-center gap-3">
        {/* Status badge */}
        <Badge variant={config.variant} className={cn('gap-1', config.className)}>
          <StatusIcon className={cn('h-3.5 w-3.5', config.iconClassName)} />
          {config.label}
          {currentStepIndex !== undefined && totalSteps > 0 && (
            <span className="ml-1">
              (Step {currentStepIndex + 1}/{totalSteps})
            </span>
          )}
        </Badge>

        {/* Duration */}
        {startedAt && (
          <div className="flex items-center gap-1 text-xs text-muted-foreground">
            <Timer className="h-3.5 w-3.5" />
            <span>{formatDuration(startedAt, completedAt)}</span>
          </div>
        )}
      </div>

      {/* Control buttons */}
      <div className="flex items-center gap-1">
        {/* Pause button */}
        {canPause && (
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={onPause}
            title="Pause workflow"
          >
            <Pause className="h-4 w-4" />
          </Button>
        )}

        {/* Resume button */}
        {canResume && (
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={onResume}
            title="Resume workflow"
          >
            <Play className="h-4 w-4" />
          </Button>
        )}

        {/* Cancel button */}
        {canCancel && (
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 hover:text-destructive"
            onClick={onCancel}
            title="Cancel workflow"
          >
            <Square className="h-4 w-4" />
          </Button>
        )}
      </div>
    </div>
  )
}
