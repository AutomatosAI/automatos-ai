'use client'

import { useState, useCallback } from 'react'
import { AnimatePresence } from 'framer-motion'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog'
import { useAssignableAgents } from '@/hooks/use-agent-api'
import type { CreateTaskPayload } from '@/hooks/use-board-tasks-api'
import type { TaskPriority, ReviewMode } from '@/types/board'
import { QuickCreateForm } from './create-task-steps'
import { PlanningForm } from './create-task-steps'
import { RefinedPreview } from './create-task-steps'
import { defaultScheduleAt, type ScheduleMode } from './schedule-choice'
import { AttachmentBar, AttachmentInput, useTaskAttachments } from './create-task-attachments'
import { useCreateTaskSubmit } from './create-task-submit'
import { useTaskPlanning, type Step } from './create-task-planning'

interface CreateTaskDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

const STEP_TEXT: Record<Step, { title: string; description: string }> = {
  quick: { title: 'Create Task', description: 'Create a task directly or plan it with AI.' },
  planning: { title: 'Plan with AI', description: 'Answer a few questions to refine the task.' },
  refined: { title: 'Review & Create', description: 'Review the AI-refined task before creating.' },
}

export function CreateTaskDialog({ open, onOpenChange }: CreateTaskDialogProps) {
  const [step, setStep] = useState<Step>('quick')

  // Form state
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [priority, setPriority] = useState<TaskPriority>('medium')
  const [agentId, setAgentId] = useState<string>('none')
  const [tags, setTags] = useState('')
  const [reviewMode, setReviewMode] = useState<ReviewMode>('auto')
  // When to file it: `now` is the existing create; anything else is a
  // scheduled task that waits on the calendar (POST /api/v1/scheduled-tasks).
  const [scheduleMode, setScheduleMode] = useState<ScheduleMode>('now')
  const [scheduleAt, setScheduleAt] = useState<string>(() => defaultScheduleAt())

  const files = useTaskAttachments()  // PRD-127
  const planning = useTaskPlanning({ title, description, setTitle, setPriority, setStep })
  const { data: agents = [] } = useAssignableAgents()
  const { refinedData } = planning

  const resetForm = useCallback(() => {
    setStep('quick')
    setTitle('')
    setDescription('')
    setPriority('medium')
    setAgentId('none')
    setTags('')
    setReviewMode('auto')
    setScheduleMode('now')
    setScheduleAt(defaultScheduleAt())
    planning.reset()
    files.reset()
  }, [planning, files])

  const handleOpenChange = useCallback((value: boolean) => {
    if (!value) resetForm()
    onOpenChange(value)
  }, [onOpenChange, resetForm])

  const buildPayload = useCallback((): CreateTaskPayload => ({
    title: refinedData?.title ?? title,
    description: refinedData?.description ?? description,
    priority: (refinedData?.priority as TaskPriority) ?? priority,
    assigned_agent_id: agentId && agentId !== 'none' ? Number(agentId) : undefined,
    tags: refinedData?.suggested_tags ?? (tags ? tags.split(',').map((t) => t.trim()).filter(Boolean) : []),
    review_mode: reviewMode,
    raw_prompt: description,
    planning_data: refinedData ?? undefined,
    attachment_ids: files.attachments.map((a) => a.attachment_id),  // PRD-127
  }), [title, description, priority, agentId, tags, reviewMode, refinedData, files.attachments])

  const assigned = agentId !== 'none' ? (agents as any[]).find((a) => String(a.id) === agentId) : null
  const { submit, isSubmitting } = useCreateTaskSubmit({
    buildPayload,
    scheduleMode,
    scheduleAt,
    attachmentCount: files.attachments.length,
    agentName: assigned?.name ?? null,
    onDone: () => handleOpenChange(false),
  })

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="glass-card sm:max-w-[640px]">
        <AttachmentInput files={files} />

        <DialogHeader>
          <DialogTitle className="text-base">{STEP_TEXT[step].title}</DialogTitle>
          <DialogDescription className="text-xs text-muted-foreground">{STEP_TEXT[step].description}</DialogDescription>
        </DialogHeader>

        {step === 'quick' && <AttachmentBar files={files} />}

        <AnimatePresence mode="wait">
          {step === 'quick' && (
            <QuickCreateForm
              key="quick"
              title={title}
              description={description}
              priority={priority}
              agentId={agentId}
              tags={tags}
              reviewMode={reviewMode}
              agents={agents}
              onTitleChange={setTitle}
              onDescriptionChange={setDescription}
              onPriorityChange={setPriority}
              onAgentIdChange={setAgentId}
              onTagsChange={setTags}
              onReviewModeChange={setReviewMode}
              scheduleMode={scheduleMode}
              scheduleAt={scheduleAt}
              onScheduleModeChange={setScheduleMode}
              onScheduleAtChange={setScheduleAt}
              onSubmit={submit}
              onPlan={planning.plan}
              isSubmitting={isSubmitting}
              isPlanning={planning.isPlanning}
            />
          )}

          {step === 'planning' && planning.planData && (
            <PlanningForm
              key="planning"
              planData={planning.planData}
              answers={planning.answers}
              onAnswerChange={(qId, idx) => planning.setAnswers({ ...planning.answers, [qId]: idx })}
              onRefine={planning.refine}
              onBack={() => setStep('quick')}
              isRefining={planning.isRefining}
            />
          )}

          {step === 'refined' && refinedData && (
            <RefinedPreview
              key="refined"
              data={refinedData}
              onBack={() => setStep('planning')}
              onSubmit={submit}
              isSubmitting={isSubmitting}
            />
          )}
        </AnimatePresence>
      </DialogContent>
    </Dialog>
  )
}
