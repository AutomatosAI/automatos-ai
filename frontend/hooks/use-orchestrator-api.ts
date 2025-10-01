/**
 * React hooks for Orchestrator API - All endpoints verified working ✅
 */

'use client'

import { useState, useCallback } from 'react'
import { apiClient } from "@/lib/api-client"

interface SubmitTaskParams {
  task_description: string
  priority?: string
  context?: any
}

interface ExecutePhaseParams {
  phase: string
  agents: string[]
  execution_type?: string
}

export function useTaskSubmission() {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [submittedTask, setSubmittedTask] = useState<any>(null)

  const submitTask = useCallback(async (params: SubmitTaskParams) => {
    setIsLoading(true)
    setError(null)

    try {
      // ✅ Verified working: /api/orchestrator/task/submit
      const response = await apiClient.submitTask(params)
      setSubmittedTask(response)
      return response
    } catch (err: any) {
      const errorMsg = err?.message || 'Failed to submit task'
      setError(errorMsg)
      throw err
    } finally {
      setIsLoading(false)
    }
  }, [])

  return {
    isLoading,
    error,
    submittedTask,
    submitTask,
  }
}

export function usePhaseExecution() {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [executionResult, setExecutionResult] = useState<any>(null)

  const executePhase = useCallback(async (params: ExecutePhaseParams) => {
    setIsLoading(true)
    setError(null)

    try {
      // ✅ Verified working: /api/orchestrator/execute-phase
      const response = await apiClient.executePhase(params)
      setExecutionResult(response)
      return response
    } catch (err: any) {
      const errorMsg = err?.message || 'Failed to execute phase'
      setError(errorMsg)
      throw err
    } finally {
      setIsLoading(false)
    }
  }, [])

  return {
    isLoading,
    error,
    executionResult,
    executePhase,
  }
}

/**
 * Combined hook for orchestrator operations
 */
export function useOrchestratorAPI() {
  const taskOps = useTaskSubmission()
  const phaseOps = usePhaseExecution()

  return {
    // Task operations
    submitTask: taskOps.submitTask,
    submittedTask: taskOps.submittedTask,
    taskLoading: taskOps.isLoading,
    taskError: taskOps.error,

    // Phase execution operations
    executePhase: phaseOps.executePhase,
    executionResult: phaseOps.executionResult,
    phaseLoading: phaseOps.isLoading,
    phaseError: phaseOps.error,
  }
}


