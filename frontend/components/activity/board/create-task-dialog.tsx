'use client'

import { useState, useCallback, useRef } from 'react'
import { useRouter } from 'next/navigation'
import { AnimatePresence } from 'framer-motion'
import { toast } from 'sonner'
import { Paperclip, X, FileText, Image } from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { useAssignableAgents } from '@/hooks/use-agent-api'
import { useCreateTask, usePlanTask, useRefineTask } from '@/hooks/use-board-tasks-api'
import type { CreateTaskPayload, PlanResponse, RefineResponse } from '@/hooks/use-board-tasks-api'
import type { TaskPriority, ReviewMode } from '@/types/board'
import { QuickCreateForm } from './create-task-steps'
import { PlanningForm } from './create-task-steps'
import { RefinedPreview } from './create-task-steps'
import { apiClient } from '@/lib/api-client'

// PRD-127: Attachment metadata
interface AttachmentMeta {
  attachment_id: string
  filename: string
  mime: string
  media_type: 'image' | 'document'
}

interface CreateTaskDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

type Step = 'quick' | 'planning' | 'refined'

export function CreateTaskDialog({ open, onOpenChange }: CreateTaskDialogProps) {
  const router = useRouter()
  const [step, setStep] = useState<Step>('quick')

  // Form state
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [priority, setPriority] = useState<TaskPriority>('medium')
  const [agentId, setAgentId] = useState<string>('none')
  const [tags, setTags] = useState('')
  const [reviewMode, setReviewMode] = useState<ReviewMode>('auto')

  // PRD-127: Attachment state
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [attachments, setAttachments] = useState<AttachmentMeta[]>([])
  const [isUploading, setIsUploading] = useState(false)

  // Planning state
  const [planData, setPlanData] = useState<PlanResponse | null>(null)
  const [answers, setAnswers] = useState<Record<string, number>>({})
  const [refinedData, setRefinedData] = useState<RefineResponse | null>(null)

  const { data: agents = [] } = useAssignableAgents()
  const createTask = useCreateTask()
  const planTask = usePlanTask()
  const refineTask = useRefineTask()

  // PRD-127: Handle file upload
  const handleFileSelect = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || [])
    if (files.length === 0) return

    setIsUploading(true)
    try {
      const results = await Promise.all(
        files.map((file) => apiClient.uploadAttachment(file))
      )
      setAttachments((prev) => [
        ...prev,
        ...results.map((r) => ({
          attachment_id: r.attachment_id,
          filename: r.filename,
          mime: r.mime,
          media_type: r.media_type,
        })),
      ])
      toast.success(`Uploaded ${files.length} file${files.length === 1 ? '' : 's'}`)
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : 'Upload failed'
      toast.error(msg)
    } finally {
      setIsUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }, [])

  const handleRemoveAttachment = useCallback(async (attachmentId: string) => {
    try {
      await apiClient.deleteAttachment(attachmentId)
      setAttachments((prev) => prev.filter((a) => a.attachment_id !== attachmentId))
    } catch {
      // Ignore delete errors — just remove from UI
      setAttachments((prev) => prev.filter((a) => a.attachment_id !== attachmentId))
    }
  }, [])

  const resetForm = useCallback(() => {
    setStep('quick')
    setTitle('')
    setDescription('')
    setPriority('medium')
    setAgentId('none')
    setTags('')
    setReviewMode('auto')
    setPlanData(null)
    setAnswers({})
    setRefinedData(null)
    setAttachments([])  // PRD-127
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
      assigned_agent_id: agentId && agentId !== 'none' ? Number(agentId) : undefined,
      tags: finalTags,
      review_mode: reviewMode,
      raw_prompt: description,
      planning_data: refinedData ?? undefined,
      attachment_ids: attachments.map((a) => a.attachment_id),  // PRD-127
    }
  }, [title, description, priority, agentId, tags, reviewMode, refinedData, attachments])

  const handleCreate = useCallback(async () => {
    const payload = buildPayload()
    if (!payload.title.trim()) {
      toast.error('Title is required')
      return
    }
    try {
      await createTask.mutateAsync(payload)

      // Build toast message with agent name when assigned
      const assignedAgent = agentId && agentId !== 'none'
        ? (agents as any[]).find((a) => String(a.id) === agentId)
        : null
      const msg = assignedAgent
        ? `Task assigned to ${assignedAgent.name}.`
        : 'Task created.'

      toast.success(msg, {
        duration: 5000,
        action: {
          label: 'View on Board',
          onClick: () => router.push('/command-center?tab=board'),
        },
      })
      handleOpenChange(false)
    } catch {
      toast.error('Failed to create task')
    }
  }, [buildPayload, createTask, handleOpenChange, agentId, agents, router])

  const handlePlan = useCallback(async () => {
    const prompt = description || title
    if (!prompt.trim()) {
      toast.error('Enter a description to plan with AI')
      return
    }
    try {
      const result = await planTask.mutateAsync({ raw_prompt: prompt })
      if (!result.questions || result.questions.length === 0) {
        toast.error('AI returned no planning questions — try a more detailed description')
        return
      }
      setPlanData(result)
      if (result.suggested_title) setTitle(result.suggested_title)
      if (result.suggested_priority) setPriority(result.suggested_priority as TaskPriority)
      const defaultAnswers: Record<string, number> = {}
      for (const q of result.questions) {
        defaultAnswers[q.id] = q.default
      }
      setAnswers(defaultAnswers)
      setStep('planning')
    } catch (err) {
      console.error('[CreateTask] Planning failed:', err)
      toast.error('Planning failed — check backend logs')
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
      <DialogContent className="glass-card sm:max-w-[640px]">
        {/* PRD-127: Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          multiple
          accept="image/*,.pdf,.doc,.docx,.xls,.xlsx,.txt,.csv,.md,.json,.py,.js,.ts,.tsx"
          onChange={handleFileSelect}
        />

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

        {/* PRD-127: Attachment bar */}
        {step === 'quick' && (
          <div className="flex items-center gap-2 pb-2 border-b border-border/50">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-8 gap-1.5 text-muted-foreground hover:text-foreground"
              disabled={isUploading}
              onClick={() => fileInputRef.current?.click()}
            >
              <Paperclip className="w-4 h-4" />
              <span className="text-xs">Attach</span>
            </Button>
            {isUploading && (
              <span className="text-xs text-muted-foreground animate-pulse">Uploading...</span>
            )}
            {attachments.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {attachments.map((att) => (
                  <div
                    key={att.attachment_id}
                    className="inline-flex items-center gap-1.5 rounded-full border border-border bg-muted/30 px-2.5 py-0.5 text-xs"
                  >
                    {att.media_type === 'image' ? (
                      <Image className="w-3 h-3 text-info" />
                    ) : (
                      <FileText className="w-3 h-3 text-warning" />
                    )}
                    <span className="max-w-[120px] truncate">{att.filename}</span>
                    <button
                      type="button"
                      onClick={() => handleRemoveAttachment(att.attachment_id)}
                      className="text-muted-foreground hover:text-destructive"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

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
