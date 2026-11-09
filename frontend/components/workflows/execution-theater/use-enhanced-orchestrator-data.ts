'use client'

// EnhancedOrchestratorView's state and the effects that fill it: which
// execution is shown, the list to pick one from, that execution's data, and a
// 2s poll while the workflow runs. Extracted from the component so it stays
// under the code-shape limit; no behaviour change.

import { useState, useEffect, useRef, type Dispatch, type SetStateAction } from 'react'
import { apiClient } from '@/lib/api-client'
import {
  EMPTY_METRICS,
  deriveCommunications,
  deriveLearningInsights,
  deriveMemoryActivities,
  deriveOrchestrationMetrics,
  liveCommunications,
  liveMemoryActivities,
  liveProgressCommunications,
  liveProgressMemoryActivities,
  normalizeExecutionOptions,
  randomLearningInsight,
  type CommunicationEvent,
  type ExecutionOption,
  type LearningInsight,
  type MemoryActivity,
  type OrchestrationMetrics
} from './enhanced-orchestrator-data'

interface OrchestrationState {
  memoryActivities: MemoryActivity[]
  setMemoryActivities: Dispatch<SetStateAction<MemoryActivity[]>>
  communications: CommunicationEvent[]
  setCommunications: Dispatch<SetStateAction<CommunicationEvent[]>>
  learningInsights: LearningInsight[]
  setLearningInsights: Dispatch<SetStateAction<LearningInsight[]>>
  metrics: OrchestrationMetrics
  setMetrics: Dispatch<SetStateAction<OrchestrationMetrics>>
  isLoading: boolean
  setIsLoading: Dispatch<SetStateAction<boolean>>
  historicalData: any
  setHistoricalData: Dispatch<SetStateAction<any>>
}

function useOrchestrationState(): OrchestrationState {
  const [memoryActivities, setMemoryActivities] = useState<MemoryActivity[]>([])
  const [communications, setCommunications] = useState<CommunicationEvent[]>([])
  const [learningInsights, setLearningInsights] = useState<LearningInsight[]>([])
  const [metrics, setMetrics] = useState<OrchestrationMetrics>(EMPTY_METRICS)
  const [isLoading, setIsLoading] = useState(false)
  const [historicalData, setHistoricalData] = useState<any>(null)
  return {
    memoryActivities, setMemoryActivities,
    communications, setCommunications,
    learningInsights, setLearningInsights,
    metrics, setMetrics,
    isLoading, setIsLoading,
    historicalData, setHistoricalData
  }
}

// Sync selectedExecutionId with executionId prop, but respect user's manual selection
// - If prop changes to a NEW value: always update (prop-driven navigation overrides user selection)
// - If prop stays the same but user manually changed: don't overwrite user's choice
// - If prop is cleared: only clear if user hasn't manually changed
function useExecutionSelection(executionId?: string) {
  const [selectedExecutionId, setSelectedExecutionId] = useState<string | null>(executionId ?? null)
  // Track whether user has manually changed selection to prevent prop from overwriting user choice
  const userChangedSelectionRef = useRef(false)
  const previousExecutionIdRef = useRef<string | undefined>(executionId)

  useEffect(() => {
    // Prop changed to a new value - always update (this is intentional navigation)
    if (executionId && executionId !== previousExecutionIdRef.current) {
      setSelectedExecutionId(executionId)
      userChangedSelectionRef.current = false // Reset flag since this is prop-driven
      previousExecutionIdRef.current = executionId
    }
    // Prop is set and user hasn't manually changed - allow prop to sync
    else if (executionId && !userChangedSelectionRef.current) {
      setSelectedExecutionId(executionId)
      previousExecutionIdRef.current = executionId
    }
    // Prop is cleared and user hasn't manually changed - clear selection
    else if (!executionId && !userChangedSelectionRef.current) {
      setSelectedExecutionId(null)
      previousExecutionIdRef.current = undefined
    }
    // If user has manually changed and prop hasn't changed to a new value, do nothing
    // (preserves user's manual selection)
  }, [executionId])

  const chooseExecution = (value: string) => {
    userChangedSelectionRef.current = true
    setSelectedExecutionId(value)
  }

  return { selectedExecutionId, setSelectedExecutionId, chooseExecution }
}

// Load execution list so historical runs can be selected
function useExecutionOptions(
  workflowId: number,
  executionId: string | undefined,
  isExecuting: boolean,
  selectedExecutionId: string | null,
  setSelectedExecutionId: (id: string) => void
): ExecutionOption[] {
  const [executionOptions, setExecutionOptions] = useState<ExecutionOption[]>([])

  useEffect(() => {
    let active = true

    const fetchExecutionOptions = async () => {
      try {
        const response = await apiClient.getWorkflowExecutions(workflowId.toString())
        const normalized = normalizeExecutionOptions(response, executionId)
        if (!active) {
          return
        }
        setExecutionOptions(normalized)
        if (!executionId && !isExecuting && !selectedExecutionId && normalized.length > 0) {
          setSelectedExecutionId(normalized[0].id)
        }
      } catch (error) {
        if (active) {
          console.error('Error loading workflow executions:', error)
        }
      }
    }

    fetchExecutionOptions()

    return () => {
      active = false
    }
  }, [workflowId, isExecuting, executionId, selectedExecutionId])

  return executionOptions
}

function useExecutionData(selectedExecutionId: string | null, state: OrchestrationState) {
  const loadExecutionData = async (execId: string) => {
    try {
      state.setIsLoading(true)

      // Get execution details
      const execution = await apiClient.getWorkflowExecution(execId.toString()) as any
      const subtasks = execution?.output_data?.subtasks || []

      const generatedMemoryActivities = deriveMemoryActivities(execution, subtasks)
      const generatedCommunications = deriveCommunications(execution, subtasks)

      const insights = deriveLearningInsights(execution)
      if (insights) {
        state.setLearningInsights(insights)
      }

      state.setMemoryActivities(generatedMemoryActivities.slice(0, 50))
      state.setCommunications(generatedCommunications.slice(0, 50))

      state.setMetrics(deriveOrchestrationMetrics(
        execution, subtasks, generatedMemoryActivities.length, generatedCommunications.length
      ))

      state.setHistoricalData({ execution })
    } catch (error) {
      console.error('Error loading execution data:', error)
    } finally {
      state.setIsLoading(false)
    }
  }

  useEffect(() => {
    if (!selectedExecutionId) {
      return
    }

    loadExecutionData(selectedExecutionId)
  }, [selectedExecutionId])
}

// Merge the live-progress endpoint's metrics, and put its new memory activities
// and messages first.
async function pollLiveProgress(workflowId: number, state: OrchestrationState) {
  try {
    const liveData = await apiClient.getWorkflowLiveProgress(workflowId.toString()) as any

    if (liveData.metrics) {
      state.setMetrics(prev => ({
        ...prev,
        ...liveData.metrics,
        completedTasks: liveData.completed_steps || prev.completedTasks,
        activeAgents: liveData.active_agents?.length || prev.activeAgents
      }))
    }
    if (liveData.memory_activities) {
      state.setMemoryActivities(prev => [...liveProgressMemoryActivities(liveData), ...prev].slice(0, 50))
    }
    if (liveData.communications) {
      state.setCommunications(prev => [...liveProgressCommunications(liveData), ...prev].slice(0, 50))
    }
  } catch (error) {
    console.error('Error fetching live progress:', error)
  }
}

// Re-read the execution and replace (not append to) its memory activities,
// messages and metrics. False when there is no execution to read: the poll then
// stops there, adding no insight either.
async function pollExecution(executionId: string | undefined, state: OrchestrationState): Promise<boolean> {
  let executionData: any = null
  let subtasks: any[] = []
  let realMemoryActivities: MemoryActivity[] = []
  let realCommunications: CommunicationEvent[] = []

  try {
    if (!executionId) return false
    executionData = await apiClient.getWorkflowExecution(executionId) as any
    if (!executionData) return false

    subtasks = executionData.output_data?.subtasks || []
    realMemoryActivities = liveMemoryActivities(executionData, subtasks)
    state.setMemoryActivities(realMemoryActivities.slice(0, 50))
    realCommunications = liveCommunications(subtasks)
    state.setCommunications(realCommunications.slice(0, 50))
  } catch (error) {
    console.error('Error fetching real execution data for orchestrator view:', error)
  }

  const insight = randomLearningInsight()
  if (insight) {
    state.setLearningInsights(prev => [insight, ...prev].slice(0, 20))
  }

  try {
    if (executionData && subtasks.length > 0) {
      state.setMetrics(deriveOrchestrationMetrics(
        executionData, subtasks, realMemoryActivities.length, realCommunications.length
      ))
    }
  } catch (error) {
    console.error('Error updating metrics:', error)
  }
  return true
}

function useLiveProgress(workflowId: number, executionId: string | undefined, isExecuting: boolean, state: OrchestrationState) {
  useEffect(() => {
    if (!isExecuting) return

    // Real-time updates during execution
    const interval = setInterval(async () => {
      if (executionId) {
        await pollLiveProgress(workflowId, state)
      }
      await pollExecution(executionId, state)
    }, 2000)

    return () => clearInterval(interval)
  }, [isExecuting, executionId, workflowId])
}

export function useEnhancedOrchestratorData(workflowId: number, executionId: string | undefined, isExecuting: boolean) {
  const selection = useExecutionSelection(executionId)
  const executionOptions = useExecutionOptions(
    workflowId, executionId, isExecuting, selection.selectedExecutionId, selection.setSelectedExecutionId
  )
  const state = useOrchestrationState()
  useExecutionData(selection.selectedExecutionId, state)
  useLiveProgress(workflowId, executionId, isExecuting, state)

  const selectedExecution = selection.selectedExecutionId
    ? executionOptions.find(option => option.id === selection.selectedExecutionId)
    : null

  return { ...state, ...selection, executionOptions, selectedExecution }
}
