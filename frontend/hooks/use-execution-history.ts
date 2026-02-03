import { useState, useEffect, useCallback } from 'react'
import { apiClient } from '@/lib/api-client'

interface ExecutionStage {
  stage: number
  name: string
  status: 'completed' | 'skipped' | 'failed'
  data?: any
  metrics?: Record<string, any>
  quality_scores?: Record<string, number>
}

interface ExecutionDetails {
  id: number
  workflow: {
    id: number
    name: string
    description?: string
  }
  status: string
  started_at?: string
  completed_at?: string
  duration_seconds?: number
  execution_log?: string
  error_message?: string
  quality_scores: Record<string, number>
  metrics: {
    subtasks_total: number
    subtasks_completed: number
    agents_used: number
    agents_list: string[]
    tokens_used: number
    execution_time_ms: number
    cost: number
  }
  memory_integration: {
    memories_retrieved: number
    experiences_stored: number
    patterns_extracted: number
    knowledge_nodes_created: number
  }
  stages_completed: Record<string, boolean>
  subtasks: any[]
}

export function useExecutionHistory(workflowId: number, executionId?: string) {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [executionDetails, setExecutionDetails] = useState<ExecutionDetails | null>(null)
  const [executionStages, setExecutionStages] = useState<ExecutionStage[]>([])
  const [allExecutions, setAllExecutions] = useState<any[]>([])

  // Fetch latest execution if no executionId provided
  const fetchLatestExecution = useCallback(async () => {
    if (!workflowId) return
    
    try {
      setIsLoading(true)
      setError(null)
      
      const response = await apiClient.get(`/execution-history/workflow/${workflowId}/latest`)
      
      if (response.data && response.data.id) {
        setExecutionDetails(response.data)
        
        // Also fetch stages for this execution
        const stagesResponse = await apiClient.get(`/execution-history/execution/${response.data.id}/stages`)
        setExecutionStages(stagesResponse.data.stages || [])
      }
    } catch (err: any) {
      console.error('Error fetching latest execution:', err)
      setError(err.response?.data?.detail || 'Failed to fetch execution data')
    } finally {
      setIsLoading(false)
    }
  }, [workflowId])

  // Fetch specific execution details
  const fetchExecutionDetails = useCallback(async (execId: string) => {
    try {
      setIsLoading(true)
      setError(null)
      
      const [detailsResponse, stagesResponse] = await Promise.all([
        apiClient.get(`/execution-history/execution/${execId}/details`),
        apiClient.get(`/execution-history/execution/${execId}/stages`)
      ])
      
      setExecutionDetails(detailsResponse.data)
      setExecutionStages(stagesResponse.data.stages || [])
    } catch (err: any) {
      console.error('Error fetching execution details:', err)
      setError(err.response?.data?.detail || 'Failed to fetch execution details')
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Fetch all executions for workflow
  const fetchAllExecutions = useCallback(async (limit = 10) => {
    if (!workflowId) return
    
    try {
      const response = await apiClient.get(`/execution-history/workflow/${workflowId}/all?limit=${limit}`)
      setAllExecutions(response.data.executions || [])
    } catch (err: any) {
      console.error('Error fetching all executions:', err)
    }
  }, [workflowId])

  // Initial load
  useEffect(() => {
    if (executionId) {
      fetchExecutionDetails(executionId)
    } else {
      fetchLatestExecution()
    }
  }, [workflowId, executionId])

  // Convert execution data to UI format for Enhanced Orchestrator View
  const formatForOrchestratorView = () => {
    if (!executionDetails) return null
    
    return {
      memoryActivities: extractMemoryActivities(executionDetails),
      communications: extractCommunications(executionDetails),
      learningInsights: extractLearningInsights(executionDetails),
      metrics: {
        totalTasks: executionDetails.metrics.subtasks_total,
        completedTasks: executionDetails.metrics.subtasks_completed,
        activeAgents: executionDetails.metrics.agents_used,
        memoryOperations: executionDetails.memory_integration.memories_retrieved + 
                          executionDetails.memory_integration.experiences_stored,
        messagesExchanged: 0, // TODO: Extract from communication data
        patternsLearned: executionDetails.memory_integration.patterns_extracted,
        avgResponseTime: executionDetails.metrics.execution_time_ms / 
                        Math.max(executionDetails.metrics.subtasks_total, 1),
        successRate: executionDetails.quality_scores?.completeness || 0,
        tokenUsage: executionDetails.metrics.tokens_used,
        estimatedCost: executionDetails.metrics.cost
      }
    }
  }

  // Extract memory activities from execution data
  const extractMemoryActivities = (execution: ExecutionDetails) => {
    const activities = []
    const inputData = (execution as any).input_data || {}
    const memoryRetrieval = inputData.memory_retrieval || {}
    const memoryStorage = inputData.memory_storage || {}
    
    // Add retrieval activities
    if (memoryRetrieval.results?.agent_memories) {
      Object.entries(memoryRetrieval.results.agent_memories).forEach(([agentId, memories]: [string, any]) => {
        ['working_memory', 'short_term', 'long_term'].forEach(type => {
          const items = memories[type] || []
          if (items.length > 0) {
            activities.push({
              id: `retrieval_${agentId}_${type}`,
              timestamp: execution.started_at,
              type: type.replace('_memory', '').replace('_', '-') as any,
              action: 'retrieve',
              agentId: parseInt(agentId),
              agentName: `Agent ${agentId}`,
              content: `Retrieved ${items.length} ${type.replace('_', ' ')} memories`,
              itemCount: items.length,
              relevance: 0.85
            })
          }
        })
      })
    }
    
    // Add storage activities
    if (memoryStorage.results?.per_agent) {
      Object.entries(memoryStorage.results.per_agent).forEach(([agentName, count]: [string, any]) => {
        activities.push({
          id: `storage_${agentName}`,
          timestamp: execution.completed_at,
          type: 'experience',
          action: 'store',
          agentName,
          content: `Stored ${count} experiences`,
          itemCount: count,
          relevance: 0.9
        })
      })
    }
    
    return activities
  }

  // Extract communication events from execution data
  const extractCommunications = (execution: ExecutionDetails) => {
    const events = []
    const inputData = (execution as any).input_data || {}
    
    // Extract from agent execution data
    if (inputData.agent_execution?.results) {
      Object.entries(inputData.agent_execution.results).forEach(([subtaskId, result]: [string, any]) => {
        events.push({
          id: `exec_${subtaskId}`,
          timestamp: execution.started_at,
          fromAgent: result.agent_name,
          toAgent: 'Orchestrator',
          messageType: 'task_result',
          content: `Completed ${subtaskId}: ${result.status}`,
          priority: result.status === 'completed' ? 'low' : 'high'
        })
      })
    }
    
    return events
  }

  // Extract learning insights from execution data
  const extractLearningInsights = (execution: ExecutionDetails) => {
    const insights = []
    const inputData = (execution as any).input_data || {}
    const learningSystem = inputData.learning_system || {}
    
    if (learningSystem.updates?.agent_performance_updates) {
      learningSystem.updates.agent_performance_updates.forEach((update: any) => {
        insights.push({
          id: `learning_${update.agent_id}`,
          timestamp: execution.completed_at,
          type: 'performance' as any,
          agentId: update.agent_id,
          agentName: update.agent_name || `Agent ${update.agent_id}`,
          insight: update.improvement_summary || 'Performance metrics updated',
          impact: update.performance_delta || 0,
          confidence: 0.8
        })
      })
    }
    
    if (execution.memory_integration.patterns_extracted > 0) {
      insights.push({
        id: 'patterns_extracted',
        timestamp: execution.completed_at,
        type: 'pattern' as any,
        insight: `Extracted ${execution.memory_integration.patterns_extracted} patterns from execution`,
        impact: 0.7,
        confidence: 0.9
      })
    }
    
    return insights
  }

  return {
    isLoading,
    error,
    executionDetails,
    executionStages,
    allExecutions,
    orchestratorViewData: formatForOrchestratorView(),
    refresh: executionId ? () => fetchExecutionDetails(executionId) : fetchLatestExecution,
    fetchAllExecutions
  }
}
