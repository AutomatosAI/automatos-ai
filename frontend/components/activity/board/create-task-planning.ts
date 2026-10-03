'use client'

/**
 * Planning a task with AI in the create dialog: the questions, then the refined
 * task. Split out of CreateTaskDialog (PRD-252); the behaviour is unchanged.
 */

import { useCallback, useState } from 'react'
import { toast } from 'sonner'
import { usePlanTask, useRefineTask, type PlanResponse, type RefineResponse } from '@/hooks/use-board-tasks-api'
import type { TaskPriority } from '@/types/board'

export type Step = 'quick' | 'planning' | 'refined'

interface PlanningOptions {
  title: string
  description: string
  setTitle: (title: string) => void
  setPriority: (priority: TaskPriority) => void
  setStep: (step: Step) => void
}

export function useTaskPlanning({ title, description, setTitle, setPriority, setStep }: PlanningOptions) {
  const [planData, setPlanData] = useState<PlanResponse | null>(null)
  const [answers, setAnswers] = useState<Record<string, number>>({})
  const [refinedData, setRefinedData] = useState<RefineResponse | null>(null)
  const planTask = usePlanTask()
  const refineTask = useRefineTask()

  const plan = useCallback(async () => {
    const prompt = description || title
    if (!prompt.trim()) return void toast.error('Enter a description to plan with AI')
    try {
      const result = await planTask.mutateAsync({ raw_prompt: prompt })
      if (!result.questions || result.questions.length === 0) {
        return void toast.error('AI returned no planning questions — try a more detailed description')
      }
      setPlanData(result)
      if (result.suggested_title) setTitle(result.suggested_title)
      if (result.suggested_priority) setPriority(result.suggested_priority as TaskPriority)
      setAnswers(Object.fromEntries(result.questions.map((q) => [q.id, q.default])))
      setStep('planning')
    } catch (err) {
      console.error('[CreateTask] Planning failed:', err)
      toast.error('Planning failed — check backend logs')
    }
  }, [description, title, planTask, setTitle, setPriority, setStep])

  const refine = useCallback(async () => {
    try {
      const result = await refineTask.mutateAsync({ raw_prompt: description || title, answers })
      setRefinedData(result)
      setTitle(result.title)
      setPriority(result.priority as TaskPriority)
      setStep('refined')
    } catch {
      toast.error('Refinement failed')
    }
  }, [description, title, answers, refineTask, setTitle, setPriority, setStep])

  const reset = useCallback(() => {
    setPlanData(null)
    setAnswers({})
    setRefinedData(null)
  }, [])

  return {
    planData, answers, setAnswers, refinedData, plan, refine, reset,
    isPlanning: planTask.isLoading, isRefining: refineTask.isLoading,
  }
}
