import { useState, useEffect, useCallback } from 'react'
import { apiClient } from '@/lib/api-client'

export interface OrchestrationData {
  // Real-time metrics
  metrics: {
    totalTasks: number
    completedTasks: number
    activeAgents: number
    memoryOperations: number
    messagesExchanged: number
    patternsLearned: number
    avgResponseTime: number
    successRate: number
    tokenUsage: number
    estimatedCost: number
  }
  
  // Memory activities
  memoryActivities: Array<{
    id: string
    timestamp: string
    type: 'working' | 'short_term' | 'long_term' | 'collective'
    action: 'store' | 'retrieve' | 'consolidate'
    agentId?: number
    agentName?: string
    content: string
    relevance?: number
    itemCount?: number
  }>
  
  // Agent communications
  communications: Array<{
    id: string
    timestamp: string
    fromAgent: string
    toAgent: string
    messageType: 'query' | 'response' | 'broadcast' | 'coordination'
    priority: 'low' | 'normal' | 'high' | 'critical'
    content: string
    status: 'sent' | 'received' | 'acknowledged'
  }>
  
  // Learning insights
  learningInsights: Array<{
    id: string
    timestamp: string
    type: 'pattern' | 'optimization' | 'failure' | 'success'
    category: string
    description: string
    impact: 'low' | 'medium' | 'high'
    confidence: number
  }>
  
  // Historical data for charts
  historicalData?: {
    memoryTrend: Array<{ time: string; count: number; type: string }>
    communicationTrend: Array<{ time: string; messages: number; priority: string }>
    learningTrend: Array<{ time: string; patterns: number; confidence: number }>
    performanceTrend: Array<{ time: string; successRate: number; responseTime: number }>
  }
}

export function useOrchestrationData(workflowId?: number, executionId?: string) {
  const [data, setData] = useState<OrchestrationData>({
    metrics: {
      totalTasks: 0,
      completedTasks: 0,
      activeAgents: 0,
      memoryOperations: 0,
      messagesExchanged: 0,
      patternsLearned: 0,
      avgResponseTime: 0,
      successRate: 0,
      tokenUsage: 0,
      estimatedCost: 0
    },
    memoryActivities: [],
    communications: [],
    learningInsights: []
  })
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  const fetchOrchestrationData = useCallback(async () => {
    if (!workflowId) return

    try {
      setIsLoading(true)
      
      // If we have an execution ID, fetch specific execution data
      if (executionId) {
        const executionData = await apiClient.request(`/api/v1/workflows/executions/${executionId}`)
        
        // Parse orchestration events from execution data
        const parsedData = parseExecutionData(executionData)
        setData(parsedData)
      } else {
        // Fetch latest execution for the workflow
        const executions = await apiClient.request(`/api/v1/workflows/${workflowId}/executions?limit=1`)
        
        if (executions && executions.length > 0) {
          const latestExecution = executions[0]
          const executionData = await apiClient.request(`/api/v1/workflows/executions/${latestExecution.id}`)
          
          const parsedData = parseExecutionData(executionData)
          setData(parsedData)
        }
      }
      
      // Fetch additional analytics data
      const analyticsData = await apiClient.request(`/api/v1/workflows/${workflowId}/analytics`)
      
      if (analyticsData) {
        setData(prev => ({
          ...prev,
          historicalData: {
            memoryTrend: analyticsData.memory_trend || [],
            communicationTrend: analyticsData.communication_trend || [],
            learningTrend: analyticsData.learning_trend || [],
            performanceTrend: analyticsData.performance_trend || []
          }
        }))
      }
      
      setError(null)
    } catch (err) {
      console.error('Error fetching orchestration data:', err)
      setError(err as Error)
    } finally {
      setIsLoading(false)
    }
  }, [workflowId, executionId])

  // Parse execution data into orchestration format
  const parseExecutionData = (execution: any): OrchestrationData => {
    const inputData = typeof execution.input_data === 'string' 
      ? JSON.parse(execution.input_data) 
      : execution.input_data || {}
    
    const outputData = typeof execution.output_data === 'string'
      ? JSON.parse(execution.output_data)
      : execution.output_data || {}

    // Extract memory activities
    const memoryActivities = []
    if (inputData.memory_retrieval) {
      const memoryData = inputData.memory_retrieval
      
      // Add retrieval activities
      if (memoryData.working_memory?.length > 0) {
        memoryActivities.push({
          id: `mem-${Date.now()}-1`,
          timestamp: new Date().toISOString(),
          type: 'working',
          action: 'retrieve',
          content: `Retrieved ${memoryData.working_memory.length} working memory items`,
          itemCount: memoryData.working_memory.length
        })
      }
      
      if (memoryData.short_term_memory?.length > 0) {
        memoryActivities.push({
          id: `mem-${Date.now()}-2`,
          timestamp: new Date().toISOString(),
          type: 'short_term',
          action: 'retrieve',
          content: `Retrieved ${memoryData.short_term_memory.length} short-term memory items`,
          itemCount: memoryData.short_term_memory.length
        })
      }
      
      if (memoryData.long_term_memory?.length > 0) {
        memoryActivities.push({
          id: `mem-${Date.now()}-3`,
          timestamp: new Date().toISOString(),
          type: 'long_term',
          action: 'retrieve',
          content: `Retrieved ${memoryData.long_term_memory.length} long-term memory items`,
          itemCount: memoryData.long_term_memory.length
        })
      }
    }

    // Extract communications from logs
    const communications = []
    if (execution.execution_log) {
      const logLines = execution.execution_log.split('\n')
      logLines.forEach((line: string, index: number) => {
        if (line.includes('Agent communication') || line.includes('message')) {
          communications.push({
            id: `comm-${index}`,
            timestamp: new Date().toISOString(),
            fromAgent: 'Orchestrator',
            toAgent: 'Agent',
            messageType: 'coordination',
            priority: 'normal',
            content: line,
            status: 'sent'
          })
        }
      })
    }

    // Extract learning insights
    const learningInsights = []
    if (inputData.learning_system?.updates) {
      inputData.learning_system.updates.forEach((update: any, index: number) => {
        learningInsights.push({
          id: `learn-${index}`,
          timestamp: new Date().toISOString(),
          type: 'pattern',
          category: update.category || 'General',
          description: update.description || 'Learning update captured',
          impact: 'medium',
          confidence: update.confidence || 0.75
        })
      })
    }

    // Calculate metrics
    const metrics = {
      totalTasks: inputData.subtasks?.length || 0,
      completedTasks: outputData.completed_tasks || 0,
      activeAgents: inputData.agent_selection?.selected_agents?.length || 0,
      memoryOperations: memoryActivities.length,
      messagesExchanged: communications.length,
      patternsLearned: learningInsights.length,
      avgResponseTime: outputData.avg_response_time || 0,
      successRate: outputData.success_rate || 0,
      tokenUsage: outputData.total_tokens || 0,
      estimatedCost: outputData.total_cost || 0
    }

    return {
      metrics,
      memoryActivities,
      communications,
      learningInsights
    }
  }

  // Initial fetch
  useEffect(() => {
    fetchOrchestrationData()
  }, [fetchOrchestrationData])

  // Set up polling for real-time updates
  useEffect(() => {
    const interval = setInterval(fetchOrchestrationData, 5000) // Poll every 5 seconds
    return () => clearInterval(interval)
  }, [fetchOrchestrationData])

  return {
    data,
    isLoading,
    error,
    refetch: fetchOrchestrationData
  }
}

// Hook for aggregated orchestration data across all workflows
export function useAggregatedOrchestrationData(days: number = 7) {
  const [data, setData] = useState<{
    totalMemoryOperations: number
    totalCommunications: number
    totalPatternsLearned: number
    avgSuccessRate: number
    totalCostSaved: number
    topAgents: Array<{ name: string; executions: number; successRate: number }>
    memoryUtilization: { working: number; shortTerm: number; longTerm: number; collective: number }
    learningVelocity: number
  } | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    const fetchAggregatedData = async () => {
      try {
        setIsLoading(true)
        
        // Fetch benchmarking data
        const perfSummary = await apiClient.request(`/api/v1/benchmarking/performance-summary?days=${days}`)
        const learningData = await apiClient.request('/api/v1/benchmarking/learning-effectiveness')
        const agentData = await apiClient.request('/api/v1/benchmarking/agent-performance')
        
        // Aggregate the data
        const aggregated = {
          totalMemoryOperations: learningData?.summary?.total_memory_operations || 0,
          totalCommunications: perfSummary?.period?.total_executions * 10 || 0, // Estimate
          totalPatternsLearned: learningData?.summary?.total_patterns_learned || 0,
          avgSuccessRate: perfSummary?.improvements?.success_rate?.second_half_rate || 0,
          totalCostSaved: perfSummary?.improvements?.cost_efficiency?.total_savings || 0,
          topAgents: agentData?.agent_metrics?.slice(0, 5).map((agent: any) => ({
            name: agent.agent_name,
            executions: agent.total_executions,
            successRate: agent.success_rate
          })) || [],
          memoryUtilization: {
            working: 25,
            shortTerm: 35,
            longTerm: 30,
            collective: 10
          },
          learningVelocity: learningData?.summary?.learning_velocity || 0
        }
        
        setData(aggregated)
        setError(null)
      } catch (err) {
        console.error('Error fetching aggregated orchestration data:', err)
        setError(err as Error)
      } finally {
        setIsLoading(false)
      }
    }

    fetchAggregatedData()
    const interval = setInterval(fetchAggregatedData, 30000) // Refresh every 30s
    return () => clearInterval(interval)
  }, [days])

  return { data, isLoading, error }
}
