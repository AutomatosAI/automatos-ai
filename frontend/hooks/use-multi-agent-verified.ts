/**
 * React hooks for Multi-Agent API - All endpoints verified working ✅
 */

'use client'

import { useState, useCallback } from 'react'
import { apiClient } from "@/lib/api-client"

interface CoordinateAgentsParams {
  agents: string[]
  task: any
  strategy?: string
}

interface CollaborativeReasoningParams {
  problem: string
  agents: string[]
  strategy?: string
}

export function useMultiAgentHealth() {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [health, setHealth] = useState<any>(null)

  const checkHealth = useCallback(async () => {
    setIsLoading(true)
    setError(null)

    try {
      // ✅ Verified working: /api/multi-agent/health
      const response = await apiClient.getMultiAgentHealth()
      setHealth(response)
      return response
    } catch (err: any) {
      const errorMsg = err?.message || 'Failed to check multi-agent health'
      setError(errorMsg)
      throw err
    } finally {
      setIsLoading(false)
    }
  }, [])

  return {
    isLoading,
    error,
    health,
    checkHealth,
  }
}

export function useAgentCoordination() {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [coordinationResult, setCoordinationResult] = useState<any>(null)

  const coordinateAgents = useCallback(async (params: CoordinateAgentsParams) => {
    setIsLoading(true)
    setError(null)

    try {
      // ✅ Verified working: /api/multi-agent/coordination/coordinate
      const response = await apiClient.coordinateAgents(params)
      setCoordinationResult(response)
      return response
    } catch (err: any) {
      const errorMsg = err?.message || 'Failed to coordinate agents'
      setError(errorMsg)
      throw err
    } finally {
      setIsLoading(false)
    }
  }, [])

  return {
    isLoading,
    error,
    coordinationResult,
    coordinateAgents,
  }
}

export function useCollaborativeReasoning() {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [reasoningResult, setReasoningResult] = useState<any>(null)

  const performReasoning = useCallback(async (params: CollaborativeReasoningParams) => {
    setIsLoading(true)
    setError(null)

    try {
      // ✅ Verified working: /api/multi-agent/reasoning/collaborative
      const response = await apiClient.collaborativeReasoning(params)
      setReasoningResult(response)
      return response
    } catch (err: any) {
      const errorMsg = err?.message || 'Failed to perform collaborative reasoning'
      setError(errorMsg)
      throw err
    } finally {
      setIsLoading(false)
    }
  }, [])

  return {
    isLoading,
    error,
    reasoningResult,
    performReasoning,
  }
}

export function useOptimizationStatistics() {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [statistics, setStatistics] = useState<any>(null)

  const fetchStatistics = useCallback(async () => {
    setIsLoading(true)
    setError(null)

    try {
      // ✅ Verified working: /api/multi-agent/optimization/statistics
      const response = await apiClient.getOptimizationStatistics()
      setStatistics(response)
      return response
    } catch (err: any) {
      const errorMsg = err?.message || 'Failed to fetch optimization statistics'
      setError(errorMsg)
      throw err
    } finally {
      setIsLoading(false)
    }
  }, [])

  return {
    isLoading,
    error,
    statistics,
    fetchStatistics,
  }
}

/**
 * Combined hook for all multi-agent operations
 */
export function useMultiAgentAPI() {
  const healthOps = useMultiAgentHealth()
  const coordinationOps = useAgentCoordination()
  const reasoningOps = useCollaborativeReasoning()
  const statsOps = useOptimizationStatistics()

  return {
    // Health operations
    checkHealth: healthOps.checkHealth,
    health: healthOps.health,
    healthLoading: healthOps.isLoading,
    healthError: healthOps.error,

    // Coordination operations
    coordinateAgents: coordinationOps.coordinateAgents,
    coordinationResult: coordinationOps.coordinationResult,
    coordinationLoading: coordinationOps.isLoading,
    coordinationError: coordinationOps.error,

    // Reasoning operations
    performReasoning: reasoningOps.performReasoning,
    reasoningResult: reasoningOps.reasoningResult,
    reasoningLoading: reasoningOps.isLoading,
    reasoningError: reasoningOps.error,

    // Statistics operations
    fetchStatistics: statsOps.fetchStatistics,
    statistics: statsOps.statistics,
    statsLoading: statsOps.isLoading,
    statsError: statsOps.error,
  }
}


