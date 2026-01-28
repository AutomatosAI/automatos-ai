'use client'

/**
 * WorkflowSteps Component for PRD-38.2 Extended Widgets
 *
 * Displays workflow steps as a progress list with expandable details
 */

import { useState } from 'react'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  CheckCircle2,
  XCircle,
  Clock,
  Loader2,
  MinusCircle,
  ChevronRight,
  GitBranch,
  Repeat,
  Split,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { StepDetail } from './StepDetail'
import type { WorkflowStep } from '../types'

interface WorkflowStepsProps {
  steps: WorkflowStep[]
  className?: string
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
      return CheckCircle2
    case 'failed':
      return XCircle
    case 'skipped':
      return MinusCircle
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
 * Format duration
 */
function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  const mins = Math.floor(ms / 60000)
  const secs = ((ms % 60000) / 1000).toFixed(0)
  return `${mins}m ${secs}s`
}

export function WorkflowSteps({ steps, className }: WorkflowStepsProps) {
  const [selectedStep, setSelectedStep] = useState<WorkflowStep | null>(null)
  const [detailOpen, setDetailOpen] = useState(false)

  const handleStepClick = (step: WorkflowStep) => {
    setSelectedStep(step)
    setDetailOpen(true)
  }

  return (
    <>
      <ScrollArea className={cn('flex-1', className)}>
        <div className="divide-y divide-border/30">
          {steps.map((step, index) => {
            const StatusIcon = getStepIcon(step.status)
            const TypeIcon = getTypeIcon(step.type)
            const isRunning = step.status === 'running'
            const isCompleted = step.status === 'completed'
            const isFailed = step.status === 'failed'
            const isPending = step.status === 'pending'
            const isSkipped = step.status === 'skipped'

            return (
              <button
                key={step.id}
                className={cn(
                  'w-full text-left px-3 py-2.5 hover:bg-muted/50 transition-colors',
                  'focus:outline-none focus:bg-muted/50',
                  isRunning && 'bg-blue-50/50 dark:bg-blue-950/20'
                )}
                onClick={() => handleStepClick(step)}
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
                        isPending && 'text-gray-400',
                        isRunning && 'text-blue-500 animate-spin',
                        isCompleted && 'text-green-500',
                        isFailed && 'text-red-500',
                        isSkipped && 'text-gray-400'
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
                          isFailed && 'text-red-600 dark:text-red-400'
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
                        <span className="text-xs text-blue-600 dark:text-blue-400">
                          ← Currently running
                        </span>
                      )}
                    </div>

                    {/* Error preview */}
                    {step.error && (
                      <p className="text-xs text-red-500 truncate mt-0.5">
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
                    {step.duration !== undefined && (
                      <span>{formatDuration(step.duration)}</span>
                    )}
                    <ChevronRight className="h-4 w-4" />
                  </div>
                </div>
              </button>
            )
          })}
        </div>
      </ScrollArea>

      {/* Step detail dialog */}
      <StepDetail
        step={selectedStep}
        open={detailOpen}
        onOpenChange={setDetailOpen}
      />
    </>
  )
}
