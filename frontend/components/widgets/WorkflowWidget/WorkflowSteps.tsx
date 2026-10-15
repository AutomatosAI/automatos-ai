'use client'

/**
 * WorkflowSteps Component for PRD-38.2 Extended Widgets
 *
 * Displays workflow steps as a progress list with expandable details
 */

import { useState } from 'react'
import { ScrollArea } from '@/components/ui/scroll-area'
import { cn } from '@/lib/utils'
import { StepDetail } from './StepDetail'
import { WorkflowStepRow } from './workflow-step-row'
import type { WorkflowStep } from '../types'

interface WorkflowStepsProps {
  steps: WorkflowStep[]
  className?: string
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
          {steps.map((step, index) => (
            <WorkflowStepRow key={step.id} step={step} index={index} onSelect={handleStepClick} />
          ))}
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
