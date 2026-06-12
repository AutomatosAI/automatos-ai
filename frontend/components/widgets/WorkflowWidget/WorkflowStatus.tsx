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
  Check,
  X,
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
  actionInProgress?: 'pause' | 'resume' | 'cancel' | null
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
        className: 'bg-gray-100 text-foreground/70 dark:bg-secondary dark:text-foreground/90',
      }
    case 'running':
      return {
        icon: Loader2,
        label: 'Running',
        variant: 'default' as const,
        className: 'bg-blue-100 text-info dark:bg-blue-900 dark:text-info/80',
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
        icon: Check,
        label: 'Completed',
        variant: 'default' as const,
        className: 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300',
      }
    case 'failed':
      return {
        icon: X,
        label: 'Failed',
        variant: 'destructive' as const,
        className: 'bg-destructive/10 text-destructive dark:bg-destructive/20 dark:text-destructive/80',
      }
    case 'cancelled':
      return {
        icon: AlertCircle,
        label: 'Cancelled',
        variant: 'secondary' as const,
        className: 'bg-warning text-warning dark:bg-warning dark:text-warning',
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
  actionInProgress,
  className,
}: WorkflowStatusProps) {
  const config = getStatusConfig(status)
  const StatusIcon = config.icon

  const isRunning = status === 'running'
  const isPaused = status === 'paused'
  const canPause = isRunning && onPause
  const canResume = isPaused && onResume
  const canCancel = (isRunning || isPaused) && onCancel
  const isBusy = !!actionInProgress

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
        {/* Pause button — visible when running */}
        {canPause && (
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            disabled={isBusy}
            onClick={onPause}
            title="Pause workflow"
          >
            {actionInProgress === 'pause' ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Pause className="h-4 w-4" />
            )}
          </Button>
        )}

        {/* Resume button — visible when paused */}
        {canResume && (
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            disabled={isBusy}
            onClick={onResume}
            title="Resume workflow"
          >
            {actionInProgress === 'resume' ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Play className="h-4 w-4" />
            )}
          </Button>
        )}

        {/* Cancel button — visible when running or paused */}
        {canCancel && (
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 hover:text-destructive"
            disabled={isBusy}
            onClick={onCancel}
            title="Cancel workflow"
          >
            {actionInProgress === 'cancel' ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Square className="h-4 w-4" />
            )}
          </Button>
        )}
      </div>
    </div>
  )
}
