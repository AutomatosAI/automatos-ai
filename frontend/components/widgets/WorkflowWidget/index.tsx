'use client'

/**
 * WorkflowWidget Component for PRD-38.2 Extended Widgets
 *
 * Displays workflow execution status, steps, and provides control actions
 */

import { useMemo, useCallback } from 'react'
import { Workflow, FileJson } from 'lucide-react'
import { WidgetBase } from '../WidgetBase'
import { registerWidget } from '../registry'
import { WorkflowStatus } from './WorkflowStatus'
import { WorkflowSteps } from './WorkflowSteps'
import type {
  WidgetBaseProps,
  WorkflowWidgetData,
  WidgetDefinition,
} from '../types'
import { toast } from 'sonner'

export function WorkflowWidget({
  id,
  title,
  data,
  metadata,
  isActive,
  isLoading,
  error,
  onClose,
  onMaximize,
  onRefresh,
}: WidgetBaseProps<WorkflowWidgetData>) {
  // Find current running step
  const currentStepIndex = useMemo(() => {
    return data.steps.findIndex((step) => step.status === 'running')
  }, [data.steps])

  // Control handlers (would integrate with backend)
  const handlePause = useCallback(() => {
    toast.info('Pause functionality requires backend integration')
  }, [])

  const handleResume = useCallback(() => {
    toast.info('Resume functionality requires backend integration')
  }, [])

  const handleCancel = useCallback(() => {
    if (confirm('Are you sure you want to cancel this workflow?')) {
      toast.info('Cancel functionality requires backend integration')
    }
  }, [])

  // Calculate completion percentage
  const completionPercent = useMemo(() => {
    if (data.steps.length === 0) return 0
    const completed = data.steps.filter(
      (s) => s.status === 'completed' || s.status === 'skipped'
    ).length
    return Math.round((completed / data.steps.length) * 100)
  }, [data.steps])

  // Display title
  const displayTitle = title || data.workflowName || 'Workflow'

  return (
    <WidgetBase
      title={displayTitle}
      icon={<Workflow className="h-4 w-4" />}
      metadata={metadata}
      isActive={isActive}
      isLoading={isLoading}
      error={error}
      onClose={onClose}
      onMaximize={onMaximize}
      onRefresh={onRefresh}
      customActions={
        data.variables
          ? [
              {
                label: 'View Variables',
                icon: <FileJson className="h-4 w-4 mr-2" />,
                onClick: () => {
                  toast.info(
                    <div className="font-mono text-xs max-h-40 overflow-auto">
                      <pre>{JSON.stringify(data.variables, null, 2)}</pre>
                    </div>,
                    { duration: 5000 }
                  )
                },
              },
            ]
          : undefined
      }
    >
      <div className="flex flex-col h-full">
        {/* Status bar */}
        <WorkflowStatus
          status={data.status}
          currentStepIndex={currentStepIndex >= 0 ? currentStepIndex : undefined}
          totalSteps={data.steps.length}
          startedAt={data.startedAt}
          completedAt={data.completedAt}
          onPause={data.status === 'running' ? handlePause : undefined}
          onResume={data.status === 'paused' ? handleResume : undefined}
          onCancel={
            data.status === 'running' || data.status === 'paused'
              ? handleCancel
              : undefined
          }
        />

        {/* Error display */}
        {data.error && (
          <div className="px-3 py-2 bg-red-50 dark:bg-red-900/20 border-b border-red-200 dark:border-red-800">
            <p className="text-sm text-red-700 dark:text-red-300">
              {data.error}
            </p>
          </div>
        )}

        {/* Steps list */}
        <WorkflowSteps steps={data.steps} className="flex-1" />

        {/* Footer with progress */}
        <div className="px-3 py-1.5 border-t border-border/30 bg-muted/20">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>
              {data.steps.filter((s) => s.status === 'completed').length} of{' '}
              {data.steps.length} steps completed
            </span>
            <span>{completionPercent}%</span>
          </div>
          {/* Progress bar */}
          <div className="mt-1 h-1 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-primary transition-all duration-300"
              style={{ width: `${completionPercent}%` }}
            />
          </div>
        </div>
      </div>
    </WidgetBase>
  )
}

/**
 * Widget definition for registry
 */
export const WorkflowWidgetDef: WidgetDefinition<WorkflowWidgetData> = {
  type: 'workflow',
  displayName: 'Workflow',
  description: 'Display workflow execution status and steps',
  icon: Workflow,
  component: WorkflowWidget,
  defaultSize: { width: 5, height: 5 },
  minSize: { width: 4, height: 3 },
  capabilities: ['refreshable', 'fullscreen'],
}

// Register the widget
registerWidget(WorkflowWidgetDef)
