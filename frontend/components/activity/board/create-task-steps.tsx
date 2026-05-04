'use client'

import { motion } from 'framer-motion'
import { Sparkles, Loader2, ArrowLeft, Check } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { PremiumIcon } from '@/components/shared/premium-icon'
import type { PlanResponse, RefineResponse } from '@/hooks/use-board-tasks-api'
import type { TaskPriority, ReviewMode } from '@/types/board'
import { cn } from '@/lib/utils'

// ============= QUICK CREATE =============

export interface QuickCreateFormProps {
  title: string
  description: string
  priority: TaskPriority
  agentId: string
  tags: string
  reviewMode: ReviewMode
  agents: any[]
  onTitleChange: (v: string) => void
  onDescriptionChange: (v: string) => void
  onPriorityChange: (v: TaskPriority) => void
  onAgentIdChange: (v: string) => void
  onTagsChange: (v: string) => void
  onReviewModeChange: (v: ReviewMode) => void
  onSubmit: () => void
  onPlan: () => void
  isSubmitting: boolean
  isPlanning: boolean
}

export function QuickCreateForm(props: QuickCreateFormProps) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -10 }}
      className="space-y-3"
    >
      <div className="space-y-2">
        <Label htmlFor="task-title" className="text-xs">Title</Label>
        <Input
          id="task-title"
          value={props.title}
          onChange={(e) => props.onTitleChange(e.target.value)}
          placeholder="What needs to be done?"
          className="h-8 text-sm"
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="task-desc" className="text-xs">Description</Label>
        <Textarea
          id="task-desc"
          value={props.description}
          onChange={(e) => props.onDescriptionChange(e.target.value)}
          placeholder="Describe the task or paste a prompt for AI planning..."
          className="text-sm min-h-[120px] resize-y"
        />
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-2">
          <Label className="text-xs">Priority</Label>
          <Select value={props.priority} onValueChange={(v) => props.onPriorityChange(v as TaskPriority)}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="urgent">Urgent</SelectItem>
              <SelectItem value="high">High</SelectItem>
              <SelectItem value="medium">Medium</SelectItem>
              <SelectItem value="low">Low</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label className="text-xs">Assign Agent</Label>
          <Select value={props.agentId} onValueChange={props.onAgentIdChange}>
            <SelectTrigger className="h-8 text-xs"><SelectValue placeholder="Unassigned" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="none">Unassigned</SelectItem>
              {(props.agents as any[]).map((agent: any) => (
                <SelectItem key={agent.id} value={String(agent.id)}>
                  <span className="flex items-center gap-1.5">
                    <PremiumIcon name={agent.agent_icon ?? null} size={14} />
                    {agent.name}
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-2">
          <Label htmlFor="task-tags" className="text-xs">Tags (comma-separated)</Label>
          <Input
            id="task-tags"
            value={props.tags}
            onChange={(e) => props.onTagsChange(e.target.value)}
            placeholder="research, marketing"
            className="h-8 text-xs"
          />
        </div>

        <div className="space-y-2">
          <Label className="text-xs">Review Mode</Label>
          <Select value={props.reviewMode} onValueChange={(v) => props.onReviewModeChange(v as ReviewMode)}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="auto">Auto</SelectItem>
              <SelectItem value="human">Human</SelectItem>
              <SelectItem value="llm">LLM</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="flex justify-end gap-2 pt-2">
        <Button variant="outline" size="sm" onClick={props.onPlan} disabled={props.isPlanning || props.isSubmitting} className="text-xs">
          {props.isPlanning ? <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5 mr-1.5" />}
          Plan with AI
        </Button>
        <Button size="sm" onClick={props.onSubmit} disabled={props.isSubmitting || props.isPlanning} className="text-xs">
          {props.isSubmitting && <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />}
          Create
        </Button>
      </div>
    </motion.div>
  )
}

// ============= PLANNING =============

export interface PlanningFormProps {
  planData: PlanResponse
  answers: Record<string, number>
  onAnswerChange: (questionId: string, optionIndex: number) => void
  onRefine: () => void
  onBack: () => void
  isRefining: boolean
}

export function PlanningForm({ planData, answers, onAnswerChange, onRefine, onBack, isRefining }: PlanningFormProps) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 10 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 10 }}
      className="space-y-4"
    >
      {planData.questions.map((q) => (
        <div key={q.id} className="space-y-2">
          <p className="text-sm font-medium">{q.question}</p>
          <div className="space-y-1">
            {q.options.map((opt, idx) => (
              <label
                key={idx}
                className={cn(
                  'flex items-center gap-2 rounded-md border px-3 py-1.5 text-xs cursor-pointer transition-colors',
                  answers[q.id] === idx ? 'border-primary/50 bg-primary/5' : 'border-border/30 hover:border-border/60'
                )}
              >
                <input type="radio" name={q.id} checked={answers[q.id] === idx} onChange={() => onAnswerChange(q.id, idx)} className="sr-only" />
                <div className={cn(
                  'w-3 h-3 rounded-full border flex items-center justify-center',
                  answers[q.id] === idx ? 'border-primary bg-primary' : 'border-muted-foreground/40'
                )}>
                  {answers[q.id] === idx && <Check className="w-2 h-2 text-primary-foreground" />}
                </div>
                {opt}
              </label>
            ))}
          </div>
        </div>
      ))}

      <div className="flex justify-between pt-2">
        <Button variant="ghost" size="sm" onClick={onBack} className="text-xs">
          <ArrowLeft className="w-3.5 h-3.5 mr-1" /> Back
        </Button>
        <Button size="sm" onClick={onRefine} disabled={isRefining} className="text-xs">
          {isRefining ? <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5 mr-1.5" />}
          Refine
        </Button>
      </div>
    </motion.div>
  )
}

// ============= REFINED PREVIEW =============

export interface RefinedPreviewProps {
  data: RefineResponse
  onBack: () => void
  onSubmit: () => void
  isSubmitting: boolean
}

export function RefinedPreview({ data, onBack, onSubmit, isSubmitting }: RefinedPreviewProps) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 10 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 10 }}
      className="space-y-3"
    >
      <div className="glass-card rounded-lg p-3 space-y-2">
        <div>
          <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Title</span>
          <p className="text-sm font-medium">{data.title}</p>
        </div>
        <div>
          <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Description</span>
          <p className="text-xs text-muted-foreground whitespace-pre-wrap">{data.description}</p>
        </div>
        <div className="flex items-center gap-3">
          <div>
            <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Priority</span>
            <p className="text-xs capitalize">{data.priority}</p>
          </div>
          {data.suggested_tags.length > 0 && (
            <div>
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Tags</span>
              <div className="flex gap-1 flex-wrap mt-0.5">
                {data.suggested_tags.map((tag) => (
                  <span key={tag} className="text-[10px] px-1.5 py-0.5 rounded bg-secondary/50 text-muted-foreground">{tag}</span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="flex justify-between pt-1">
        <Button variant="ghost" size="sm" onClick={onBack} className="text-xs">
          <ArrowLeft className="w-3.5 h-3.5 mr-1" /> Back
        </Button>
        <Button size="sm" onClick={onSubmit} disabled={isSubmitting} className="text-xs">
          {isSubmitting && <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />}
          Create Task
        </Button>
      </div>
    </motion.div>
  )
}
