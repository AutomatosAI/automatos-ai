
/**
 * React hooks for Multi-Agent Systems integration
 */

'use client'

import { useState, useEffect, useCallback } from 'react'
import { multiAgentClient, Agent } from '@/lib/api/multi-agent-client'

export function useMultiAgentReasoning() {
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  const runCollaborativeReasoning = useCallback(async (
    agents: Agent[],
    task: { description: string; type?: string },
    strategy?: 'majority_vote' | 'weighted_consensus' | 'expert_override' | 'iterative_refinement'
  ) => {
    setIsLoading(true)
    setError(null)

    const response = await multiAgentClient.collaborativeReasoning({
      agents,
      task,
      strategy,
    })

    if (response.error) {
      setError(response.error)
    } else {
      setResult(response.data)
    }

    setIsLoading(false)
    return response
  }, [])

  return {
    isLoading,
    result,
    error,
    runCollaborativeReasoning,
  }
}

export function useAgentCoordination() {
  const [isLoading, setIsLoading] = useState(false)
  const [coordinationResult, setCoordinationResult] = useState<any>(null)
  const [statistics, setStatistics] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  const coordinateAgents = useCallback(async (
    agents: Agent[],
    strategy?: 'sequential' | 'parallel' | 'hierarchical' | 'mesh' | 'adaptive'
  ) => {
    setIsLoading(true)
    setError(null)

    const response = await multiAgentClient.coordinateAgents({
      agents,
      strategy,
    })

    if (response.error) {
      setError(response.error)
    } else {
      setCoordinationResult(response.data)
    }

    setIsLoading(false)
    return response
  }, [])

  const rebalanceAgents = useCallback(async (agents: Agent[]) => {
    setIsLoading(true)
    setError(null)

    const response = await multiAgentClient.rebalanceAgents({ agents })

    if (response.error) {
      setError(response.error)
    } else {
      setCoordinationResult(response.data)
    }

    setIsLoading(false)
    return response
  }, [])

  const fetchStatistics = useCallback(async () => {
    const response = await multiAgentClient.getCoordinationStatistics()
    if (response.data) {
      setStatistics(response.data)
    }
    return response
  }, [])

  return {
    isLoading,
    coordinationResult,
    statistics,
    error,
    coordinateAgents,
    rebalanceAgents,
    fetchStatistics,
  }
}

export function useBehaviorMonitoring() {
  const [isLoading, setIsLoading] = useState(false)
  const [behaviorData, setBehaviorData] = useState<any>(null)
  const [realtimeData, setRealtimeData] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const [wsConnection, setWsConnection] = useState<WebSocket | null>(null)

  const monitorBehavior = useCallback(async (
    agents: Agent[],
    taskData: Record<string, any>
  ) => {
    setIsLoading(true)
    setError(null)

    const response = await multiAgentClient.monitorBehavior({
      agents,
      task_data: taskData,
    })

    if (response.error) {
      setError(response.error)
    } else {
      setBehaviorData(response.data)
    }

    setIsLoading(false)
    return response
  }, [])

  const connectRealtimeMonitoring = useCallback(() => {
    if (wsConnection) {
      wsConnection.close()
    }

    const ws = multiAgentClient.connectRealtimeMonitoring((data) => {
      setRealtimeData(data)
    })

    setWsConnection(ws)
    return ws
  }, [wsConnection])

  const disconnectRealtimeMonitoring = useCallback(() => {
    if (wsConnection) {
      wsConnection.close()
      setWsConnection(null)
    }
  }, [wsConnection])

  useEffect(() => {
    return () => {
      if (wsConnection) {
        wsConnection.close()
      }
    }
  }, [wsConnection])

  return {
    isLoading,
    behaviorData,
    realtimeData,
    error,
    monitorBehavior,
    connectRealtimeMonitoring,
    disconnectRealtimeMonitoring,
    isConnected: wsConnection?.readyState === WebSocket.OPEN,
  }
}

export function useMultiAgentOptimization() {
  const [isLoading, setIsLoading] = useState(false)
  const [optimizationResult, setOptimizationResult] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  const optimizeAgents = useCallback(async (
    agents: Agent[],
    objectives?: string[],
    strategy?: 'gradient_descent' | 'genetic_algorithm' | 'bayesian_optimization' | 'simulated_annealing'
  ) => {
    setIsLoading(true)
    setError(null)

    const response = await multiAgentClient.optimizeAgents({
      agents,
      objectives,
      strategy,
    })

    if (response.error) {
      setError(response.error)
    } else {
      setOptimizationResult(response.data)
    }

    setIsLoading(false)
    return response
  }, [])

  return {
    isLoading,
    optimizationResult,
    error,
    optimizeAgents,
  }
}

export function useMultiAgentStatistics() {
  const [statistics, setStatistics] = useState<{
    reasoning?: any
    coordination?: any
    behavior?: any
    optimization?: any
    health?: any
  }>({})
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchAllStatistics = useCallback(async () => {
    setIsLoading(true)
    setError(null)

    try {
      const [reasoning, coordination, behavior, optimization, health] = await Promise.all([
        multiAgentClient.getReasoningStatistics(),
        multiAgentClient.getCoordinationStatistics(),
        multiAgentClient.getBehaviorStatistics(),
        multiAgentClient.getOptimizationStatistics(),
        multiAgentClient.healthCheck(),
      ])

      setStatistics({
        reasoning: reasoning.data,
        coordination: coordination.data,
        behavior: behavior.data,
        optimization: optimization.data,
        health: health.data,
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch statistics')
    }

    setIsLoading(false)
  }, [])

  useEffect(() => {
    fetchAllStatistics()
  }, [fetchAllStatistics])

  return {
    statistics,
    isLoading,
    error,
    refetch: fetchAllStatistics,
  }
}
