'use client'

import { useState, useCallback } from 'react'
import { AnimatePresence } from 'framer-motion'
import { toast } from 'react-hot-toast'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog'
import { useAgents } from '@/hooks/use-agent-api'
import { useCreateTask, usePlanTask, useRefineTask } from '@/hooks/use-board-tasks-api'
import type { CreateTaskPayload, PlanResponse, RefineResponse } from '@/hooks/use-board-tasks-api'
import type { TaskPriority, ReviewMode } from '@/types/board'
import { QuickCreateForm } from './create-task-steps'
import { PlanningForm } from './create-task-steps'
import { RefinedPreview } from './create-task-steps'

interface CreateTaskDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

type Step = 'quick' | 'planning' | 'refined'

export function CreateTaskDialog({ open, onOpenChange }: CreateTaskDialogProps) {
  const [step, setStep] = useState<Step>('quick')

  // Form state
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [priority, setPriority] = useState<TaskPriority>('medium')
  const [agentId, setAgentId] = useState<string>('')
  const [tags, setTags] = useState('')
  const [reviewMode, setReviewMode] = useState<ReviewMode>('auto')

  // Planning state
  const [planData, setPlanData] = useState<PlanResponse | null>(null)
  const [answers, setAnswers] = useState<Record<string, number>>({})
  const [refinedData, setRefinedData] = useState<RefineResponse | null>(null)

  const { data: agents = [] } = useAgents()
  const createTask = useCreateTask()
  const planTask = usePlanTask()
  const refineTask = useRefineTask()

  const resetForm = useCallback(() => {
    setStep('quick')
    setTitle('')
    setDescription('')
    setPriority('medium')
    setAgentId('')
    setTags('')
    setReviewMode('auto')
    setPlanData(null)
    setAnswers({})
    setRefinedData(null)
  }, [])

  const handleOpenChange = useCallback((value: boolean) => {
    if (!value) resetForm()
    onOpenChange(value)
  }, [onOpenChange, resetForm])

  const buildPayload = useCallback((): CreateTaskPayload => {
    const finalTitle = refinedData?.title ?? title
    const finalDesc = refinedData?.description ?? description
    const finalPriority = (refinedData?.priority as TaskPriority) ?? priority
    const finalTags = refinedData?.suggested_tags
      ?? (tags ? tags.split(',').map((t) => t.trim()).filter(Boolean) : [])

    return {
      title: finalTitle,
      description: finalDesc,
      priority: finalPriority,
      assigned_agent_id: agentId ? Number(agentId) : undefined,
      tags: finalTags,
      review_mode: reviewMode,
      raw_prompt: description,
      planning_data: refinedData ?? undefined,
    }
  }, [title, description, priority, agentId, tags, reviewMode, refinedData])

  const handleCreate = useCallback(async () => {
    const payload = buildPayload()
    if (!payload.title.trim()) {
      toast.error('Title is required')
      return
    }
    try {
      await createTask.mutateAsync(payload)
      toast.success('Task created')
      handleOpenChange(false)
    } catch {
      toast.error('Failed to create task')
    }
  }, [buildPayload, createTask, handleOpenChange])

  const handlePlan = useCallback(async () => {
    const prompt = description || title
    if (!prompt.trim()) {
      toast.error('Enter a description to plan with AI')
      return
    }
    try {
      const result = await planTask.mutateAsync({ raw_prompt: prompt })
      setPlanData(result)
      if (result.suggested_title) setTitle(result.suggested_title)
      if (result.suggested_priority) setPriority(result.suggested_priority as TaskPriority)
      const defaultAnswers: Record<string, number> = {}
      for (const q of result.questions) {
        defaultAnswers[q.id] = q.default
      }
      setAnswers(defaultAnswers)
      setStep('planning')
    } catch {
      toast.error('Planning failed')
    }
  }, [description, title, planTask])

  const handleRefine = useCallback(async () => {
    const prompt = description || title
    try {
      const result = await refineTask.mutateAsync({ raw_prompt: prompt, answers })
      setRefinedData(result)
      setTitle(result.title)
      setPriority(result.priority as TaskPriority)
      setStep('refined')
    } catch {
      toast.error('Refinement failed')
    }
  }, [description, title, answers, refineTask])

  const isSubmitting = createTask.isLoading
  const isPlanning = planTask.isLoading
  const isRefining = refineTask.isLoading

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="glass-card sm:max-w-[540px]">
        <DialogHeader>
          <DialogTitle className="text-base">
            {step === 'quick' && 'Create Task'}
            {step === 'planning' && 'Plan with AI'}
            {step === 'refined' && 'Review & Create'}
          </DialogTitle>
          <DialogDescription className="text-xs text-muted-foreground">
            {step === 'quick' && 'Create a task directly or plan it with AI.'}
            {step === 'planning' && 'Answer a few questions to refine the task.'}
            {step === 'refined' && 'Review the AI-refined task before creating.'}
          </DialogDescription>
        </DialogHeader>

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
              onSubmit={handleCreate}
              onPlan={handlePlan}
              isSubmitting={isSubmitting}
              isPlanning={isPlanning}
            />
          )}

          {step === 'planning' && planData && (
            <PlanningForm
              key="planning"
              planData={planData}
              answers={answers}
              onAnswerChange={(qId, idx) => setAnswers({ ...answers, [qId]: idx })}
              onRefine={handleRefine}
              onBack={() => setStep('quick')}
              isRefining={isRefining}
            />
          )}

          {step === 'refined' && refinedData && (
            <RefinedPreview
              key="refined"
              data={refinedData}
              onBack={() => setStep('planning')}
              onSubmit={handleCreate}
              isSubmitting={isSubmitting}
            />
          )}
        </AnimatePresence>
      </DialogContent>
    </Dialog>
  )
}
