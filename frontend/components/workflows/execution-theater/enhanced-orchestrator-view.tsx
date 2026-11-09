'use client'

import { useState, useEffect, useRef } from 'react'
import { Cpu } from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { apiClient } from '@/lib/api-client'
import {
  CommunicationTab,
  LearningTab,
  MemoryTab,
  MetricsOverviewCard,
  OverviewTab
} from './enhanced-orchestrator-sections'

export interface MemoryActivity {
  id: string
  timestamp: string
  type: 'working' | 'short_term' | 'long_term' | 'collective'
  action: 'store' | 'retrieve' | 'consolidate'
  agentId?: number
  agentName?: string
  content: string
  relevance?: number
  itemCount?: number
}

// CodeQL js/identity-replacement: the inline chain used to end in
// .replace('_', '_'), replacing "_" with itself — dead code. The memory
// kind's raw name ("working_memory", "short_term", "collective_memory", ...)
// keeps the underscore for a multi-word kind, so stripping only the
// "_memory" suffix is already the whole transform.
export function memoryActivityType(rawType: string): MemoryActivity['type'] {
  return rawType.replace('_memory', '') as MemoryActivity['type']
}

export interface CommunicationEvent {
  id: string
  timestamp: string
  fromAgent: string
  toAgent: string
  messageType: 'query' | 'response' | 'broadcast' | 'coordination'
  priority: 'low' | 'normal' | 'high' | 'critical'
  content: string
  status: 'sent' | 'received' | 'acknowledged'
}

export interface LearningInsight {
  id: string
  timestamp: string
  type: 'pattern' | 'optimization' | 'failure' | 'success'
  category: string
  description: string
  impact: 'low' | 'medium' | 'high'
  confidence: number
}

export interface OrchestrationMetrics {
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

export interface ExecutionOption {
  id: string
  label: string
  status?: string
  startedAt?: string
}

interface EnhancedOrchestratorViewProps {
  workflowId: number
  executionId?: string
  isExecuting: boolean
}

// Helper function to get agent name from execution data
const getAgentName = (executionData: any, agentId: number): string => {
  const subtasks = executionData.output_data?.subtasks || []
  const subtask = subtasks.find((t: any) => t.selected_agent?.agent_id === agentId)
  return subtask?.selected_agent?.agent_name || `Agent ${agentId}`
}

// --- loadExecutionData helpers -------------------------------------------
// Pure data-derivation steps, extracted so loadExecutionData itself stays a
// short orchestration of "fetch, derive, setState" (code-shape ratchet).

function buildMemoryActivitiesFromRetrieval(execution: any): MemoryActivity[] {
  const memActivities: MemoryActivity[] = []
  const memData = execution.input_data.memory_retrieval

  Object.entries(memData.agent_memories || {}).forEach(([agentId, memories]: [string, any]) => {
    ;['working_memory', 'short_term', 'long_term'].forEach(type => {
      const items = memories[type] || []
      if (items.length > 0) {
        memActivities.push({
          id: `mem_${agentId}_${type}`,
          timestamp: execution.created_at,
          type: memoryActivityType(type),
          action: 'retrieve',
          agentId: parseInt(agentId),
          agentName: `Agent ${agentId}`,
          content: `Retrieved ${items.length} ${type.replace('_', ' ')} items`,
          itemCount: items.length,
          relevance: 0.85
        })
      }
    })
  })

  if (memData.collective_memories?.length > 0) {
    memActivities.push({
      id: 'mem_collective',
      timestamp: execution.created_at,
      type: 'collective',
      action: 'retrieve',
      content: `Retrieved ${memData.collective_memories.length} collective memory items`,
      itemCount: memData.collective_memories.length,
      relevance: 0.9
    })
  }

  return memActivities
}

function synthesizeMemoryActivitiesFromSubtasks(execution: any, subtasks: any[]): MemoryActivity[] {
  return subtasks
    .filter((task: any) => task.selected_agent)
    .map((task: any, index: number) => ({
      id: `mem_subtask_${index}`,
      timestamp: execution.completed_at || execution.started_at || new Date().toISOString(),
      type: 'working',
      action: 'retrieve',
      agentId: task.selected_agent.agent_id,
      agentName: task.selected_agent.agent_name || `Agent ${task.selected_agent.agent_id}`,
      content: `Replaying context for subtask: ${task.description?.substring(0, 60)}...`,
      relevance: 0.8,
      itemCount: 1
    }))
}

// Extract memory activities from execution data, falling back to a
// subtask-derived synthesis when no memory records were found.
function deriveMemoryActivities(execution: any, subtasks: any[]): MemoryActivity[] {
  let generated: MemoryActivity[] = execution?.input_data?.memory_retrieval
    ? buildMemoryActivitiesFromRetrieval(execution)
    : []

  if (generated.length === 0 && subtasks.length > 0) {
    generated = synthesizeMemoryActivitiesFromSubtasks(execution, subtasks)
  }

  return generated
}

function buildCommunicationsFromLogs(execution: any): CommunicationEvent[] {
  const commEvents: CommunicationEvent[] = []
  execution.orchestrator_logs.forEach((log: string, index: number) => {
    if (log.includes('Message') || log.includes('communication')) {
      commEvents.push({
        id: `comm_${index}`,
        timestamp: execution.created_at,
        fromAgent: 'Orchestrator',
        toAgent: 'Agents',
        messageType: 'coordination',
        priority: 'normal',
        content: log,
        status: 'sent'
      })
    }
  })
  return commEvents
}

function synthesizeCommunicationsFromSubtasks(execution: any, subtasks: any[]): CommunicationEvent[] {
  const synthesizedComms: CommunicationEvent[] = []
  subtasks.forEach((task: any, index: number) => {
    if (!task.selected_agent) {
      return
    }
    synthesizedComms.push({
      id: `comm_assign_${index}`,
      timestamp: execution.started_at || new Date().toISOString(),
      fromAgent: 'Orchestrator',
      toAgent: task.selected_agent.agent_name || `Agent ${task.selected_agent.agent_id}`,
      messageType: 'query',
      priority: task.priority === 'high' ? 'high' : 'normal',
      content: `Task assigned: ${task.description?.substring(0, 60)}...`,
      status: 'sent'
    })

    if (task.execution_result) {
      synthesizedComms.push({
        id: `comm_result_${index}`,
        timestamp: execution.completed_at || new Date().toISOString(),
        fromAgent: task.selected_agent.agent_name || `Agent ${task.selected_agent.agent_id}`,
        toAgent: 'Orchestrator',
        messageType: 'response',
        priority: task.execution_result.status === 'completed' ? 'normal' : 'high',
        content: `Task completed: ${task.execution_result.status} (${task.execution_result.tokens_used || 0} tokens)`,
        status: 'sent'
      })
    }
  })
  return synthesizedComms
}

// Extract communication events from execution logs, falling back to a
// subtask-derived synthesis when no log-based events were found.
function deriveCommunications(execution: any, subtasks: any[]): CommunicationEvent[] {
  let generated: CommunicationEvent[] = execution?.orchestrator_logs
    ? buildCommunicationsFromLogs(execution)
    : []

  if (generated.length === 0 && subtasks.length > 0) {
    generated = synthesizeCommunicationsFromSubtasks(execution, subtasks)
  }

  return generated
}

// Extract learning insights, or null when the execution carries none (the
// caller leaves prior learningInsights state untouched in that case).
function deriveLearningInsights(execution: any): LearningInsight[] | null {
  if (!execution?.input_data?.learning_system?.updates) {
    return null
  }

  const insights: LearningInsight[] = []
  const learning = execution.input_data.learning_system.updates

  if (learning.agent_performance_updates) {
    Object.entries(learning.agent_performance_updates).forEach(([agentId, update]: [string, any]) => {
      insights.push({
        id: `learn_agent_${agentId}`,
        timestamp: execution.created_at,
        type: 'optimization',
        category: 'Performance',
        description: `Agent ${agentId} performance updated: Success rate ${(update.success_rate * 100).toFixed(1)}%`,
        impact: update.success_rate > 0.8 ? 'high' : 'medium',
        confidence: update.confidence || 0.75
      })
    })
  }

  if (learning.workflow_pattern_updates?.pattern_recorded) {
    insights.push({
      id: 'learn_pattern',
      timestamp: execution.created_at,
      type: 'pattern',
      category: 'Workflow',
      description: 'New workflow pattern recorded for future optimization',
      impact: 'medium',
      confidence: 0.8
    })
  }

  return insights
}

// Aggregate metrics from the execution's subtasks, or the all-zero shape
// when there are none.
function deriveOrchestrationMetrics(
  execution: any,
  subtasks: any[],
  memoryActivityCount: number,
  communicationCount: number
): OrchestrationMetrics {
  if (subtasks.length === 0) {
    return {
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
    }
  }

  const completedSubtasks = subtasks.filter((t: any) => t.execution_result?.status === 'completed')
  const uniqueAgents = new Set(subtasks.map((t: any) => t.selected_agent?.agent_id).filter(Boolean))
  const totalTokens = subtasks.reduce((sum: number, task: any) =>
    sum + (task.execution_result?.tokens_used || 0), 0)
  const successRate = subtasks.length > 0 ? completedSubtasks.length / subtasks.length : 0
  const avgTime = subtasks.length > 0
    ? subtasks.reduce((sum: number, t: any) => sum + (t.execution_result?.execution_time_ms || 0), 0) / subtasks.length
    : 0
  const estimatedCost = totalTokens * 0.00001

  return {
    totalTasks: subtasks.length,
    completedTasks: completedSubtasks.length,
    activeAgents: uniqueAgents.size,
    memoryOperations: memoryActivityCount,
    messagesExchanged: communicationCount,
    patternsLearned: execution.input_data?.memory_consolidation?.results?.patterns_extracted || 0,
    avgResponseTime: avgTime,
    successRate,
    tokenUsage: totalTokens,
    estimatedCost
  }
}

export function EnhancedOrchestratorView({ 
  workflowId, 
  executionId,
  isExecuting 
}: EnhancedOrchestratorViewProps) {
  const [memoryActivities, setMemoryActivities] = useState<MemoryActivity[]>([])
  const [communications, setCommunications] = useState<CommunicationEvent[]>([])
  const [learningInsights, setLearningInsights] = useState<LearningInsight[]>([])
  const [metrics, setMetrics] = useState<OrchestrationMetrics>({
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
  })
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['memory', 'communication']))
  const [selectedTab, setSelectedTab] = useState('overview')
  const [isLoading, setIsLoading] = useState(false)
  const [historicalData, setHistoricalData] = useState<any>(null)
  const [executionOptions, setExecutionOptions] = useState<ExecutionOption[]>([])
  const [selectedExecutionId, setSelectedExecutionId] = useState<string | null>(executionId ?? null)
  // Track whether user has manually changed selection to prevent prop from overwriting user choice
  const userChangedSelectionRef = useRef(false)
  const previousExecutionIdRef = useRef<string | undefined>(executionId)

  // Sync selectedExecutionId with executionId prop, but respect user's manual selection
  // - If prop changes to a NEW value: always update (prop-driven navigation overrides user selection)
  // - If prop stays the same but user manually changed: don't overwrite user's choice
  // - If prop is cleared: only clear if user hasn't manually changed
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

  // Load execution list so historical runs can be selected
  useEffect(() => {
    let active = true

    const fetchExecutionOptions = async () => {
      try {
        const response = await apiClient.getWorkflowExecutions(workflowId.toString())
        const rawItems = Array.isArray(response) ? response : response?.items || []

        const normalized: ExecutionOption[] = rawItems
          .map((item: any) => {
            const rawId = item?.id ?? item?.execution_id
            if (!rawId) {
              return null
            }
            const startedAt: string | undefined =
              item?.started_at ||
              item?.created_at ||
              item?.startedAt ||
              item?.createdAt ||
              undefined
            const status: string | undefined = item?.status || item?.execution_status || item?.state
            const formattedDate = startedAt ? new Date(startedAt).toLocaleString() : null
            const labelParts = [`#${rawId}`, status, formattedDate].filter(Boolean)

            return {
              id: String(rawId),
              label: labelParts.join(' • '),
              status,
              startedAt
            }
          })
          .filter((option): option is ExecutionOption => Boolean(option?.id))
          .sort((a, b) => {
            const aTime = a.startedAt ? new Date(a.startedAt).getTime() : 0
            const bTime = b.startedAt ? new Date(b.startedAt).getTime() : 0
            return bTime - aTime
          })

        const currentExecutionKey = executionId ? executionId.toString() : null
        if (
          currentExecutionKey &&
          !normalized.some((option) => option.id === currentExecutionKey)
        ) {
          normalized.unshift({
            id: currentExecutionKey,
            label: `#${currentExecutionKey} • current run`,
            status: 'running'
          })
        }

        if (!active) {
          return
        }

        setExecutionOptions(normalized)

        if (
          !executionId &&
          !isExecuting &&
          !selectedExecutionId &&
          normalized.length > 0
        ) {
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

  useEffect(() => {
    if (!selectedExecutionId) {
      return
    }

    loadExecutionData(selectedExecutionId)
  }, [selectedExecutionId])

  const selectedExecution = selectedExecutionId
    ? executionOptions.find(option => option.id === selectedExecutionId)
    : null

  const loadExecutionData = async (execId: string) => {
    try {
      setIsLoading(true)

      // Get execution details
      const execution = await apiClient.getWorkflowExecution(execId.toString()) as any
      const subtasks = execution?.output_data?.subtasks || []

      const generatedMemoryActivities = deriveMemoryActivities(execution, subtasks)
      const generatedCommunications = deriveCommunications(execution, subtasks)

      const insights = deriveLearningInsights(execution)
      if (insights) {
        setLearningInsights(insights)
      }

      setMemoryActivities(generatedMemoryActivities.slice(0, 50))
      setCommunications(generatedCommunications.slice(0, 50))

      setMetrics(deriveOrchestrationMetrics(
        execution, subtasks, generatedMemoryActivities.length, generatedCommunications.length
      ))

      setHistoricalData({ execution })
    } catch (error) {
      console.error('Error loading execution data:', error)
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    if (!isExecuting) return

    // Real-time updates during execution
    const interval = setInterval(async () => {
      // Fetch real-time data from API
      if (executionId) {
        try {
          const liveData = await apiClient.getWorkflowLiveProgress(workflowId.toString()) as any
          
          // Update metrics with live data
          if (liveData.metrics) {
            setMetrics(prev => ({
              ...prev,
              ...liveData.metrics,
              completedTasks: liveData.completed_steps || prev.completedTasks,
              activeAgents: liveData.active_agents?.length || prev.activeAgents
            }))
          }
          
          // Add new memory activities from live data
          if (liveData.memory_activities) {
            setMemoryActivities(prev => [
              ...liveData.memory_activities.map((activity: any) => ({
                id: `mem_live_${Date.now()}_${Math.random()}`,
                timestamp: new Date().toISOString(),
                type: activity.type || 'working',
                action: activity.action || 'store',
                agentId: activity.agent_id,
                agentName: activity.agent_name,
                content: activity.content,
                relevance: activity.relevance,
                itemCount: activity.item_count
              })),
              ...prev
            ].slice(0, 50))
          }
          
          // Add new communications from live data
          if (liveData.communications) {
            setCommunications(prev => [
              ...liveData.communications.map((comm: any) => ({
                id: `comm_live_${Date.now()}_${Math.random()}`,
                timestamp: new Date().toISOString(),
                fromAgent: comm.from_agent,
                toAgent: comm.to_agent,
                messageType: comm.type,
                priority: comm.priority,
                content: comm.content,
                status: comm.status
              })),
              ...prev
            ].slice(0, 50))
          }
        } catch (error) {
          console.error('Error fetching live progress:', error)
        }
      }

      // Fetch REAL execution data and extract memory/communication activities
      let executionData: any = null
      let subtasks: any[] = []
      let realMemoryActivities: MemoryActivity[] = []
      let realCommunications: CommunicationEvent[] = []
      
      try {
        if (!executionId) return
        executionData = await apiClient.getWorkflowExecution(executionId) as any
        if (!executionData) return
        
        // CLEAR OLD FAKE DATA - start fresh each time
        subtasks = executionData.output_data?.subtasks || []
        
        // Memory retrieval activities (from PRD 04/05)
        if (executionData.input_data?.memory_retrieval?.is_real) {
          const memRetrieval = executionData.input_data.memory_retrieval.results
          const agentMemories = memRetrieval?.agent_memories || {}
          
          Object.entries(agentMemories).forEach(([agentId, memories]: [string, any]) => {
            const workingMem = memories.working_memory || []
            const shortTerm = memories.short_term || []
            const longTerm = memories.long_term || []
            
            if (workingMem.length > 0) {
              realMemoryActivities.push({
                id: `mem_retrieve_working_${agentId}`,
                timestamp: new Date().toISOString(),
                type: 'working',
                action: 'retrieve',
                agentId: parseInt(agentId),
                agentName: getAgentName(executionData, parseInt(agentId)),
                content: `Memory operation: Processing context for subtask execution`,
                relevance: 0.9,
                itemCount: workingMem.length
              })
            }
            
            if (shortTerm.length > 0) {
              realMemoryActivities.push({
                id: `mem_retrieve_short_${agentId}`,
                timestamp: new Date().toISOString(),
                type: 'short_term',
                action: 'retrieve',
                agentId: parseInt(agentId),
                agentName: getAgentName(executionData, parseInt(agentId)),
                content: `Recent interactions and task history retrieved`,
                relevance: 0.75,
                itemCount: shortTerm.length
              })
            }
            
            if (longTerm.length > 0) {
              realMemoryActivities.push({
                id: `mem_retrieve_long_${agentId}`,
                timestamp: new Date().toISOString(),
                type: 'long_term',
                action: 'retrieve',
                agentId: parseInt(agentId),
                agentName: getAgentName(executionData, parseInt(agentId)),
                content: `Historical patterns and learned knowledge retrieved`,
                relevance: 0.85,
                itemCount: longTerm.length
              })
            }
          })
        }
        
        // Memory storage activities
        if (executionData.input_data?.memory_storage?.is_real) {
          const memStorage = executionData.input_data.memory_storage.results
          const perAgent = memStorage?.per_agent || {}
          
          Object.entries(perAgent).forEach(([agentId, storage]: [string, any]) => {
            if (storage.experiences_stored > 0) {
              realMemoryActivities.push({
                id: `mem_store_${agentId}`,
                timestamp: new Date().toISOString(),
                type: 'short_term',
                action: 'store',
                agentId: parseInt(agentId),
                agentName: getAgentName(executionData, parseInt(agentId)),
                content: `Stored ${storage.experiences_stored} task experiences`,
                relevance: 0.8,
                itemCount: storage.experiences_stored
              })
            }
          })
        }
        
        // Memory consolidation activities
        if (executionData.input_data?.memory_consolidation?.is_real) {
          const memConsol = executionData.input_data.memory_consolidation.results
          const agentsConsolidated = memConsol?.agents_consolidated || []
          
          agentsConsolidated.forEach((agentId: number) => {
            realMemoryActivities.push({
              id: `mem_consolidate_${agentId}`,
              timestamp: new Date().toISOString(),
              type: 'long_term',
              action: 'consolidate',
              agentId,
              agentName: getAgentName(executionData, agentId),
              content: `Consolidated learnings to long-term memory (${memConsol.patterns_extracted || 0} patterns extracted)`,
              relevance: 0.95,
              itemCount: memConsol.knowledge_nodes_created || 0
              })
          })
        }
        
        // IF NO MEMORY DATA from PRD 04/05, create activities from subtasks anyway
        if (realMemoryActivities.length === 0 && subtasks.length > 0) {
          subtasks.forEach((task: any, index: number) => {
            if (task.selected_agent) {
              realMemoryActivities.push({
                id: `mem_subtask_${index}`,
                timestamp: new Date().toISOString(),
                type: 'working',
                action: 'retrieve',
                agentId: task.selected_agent.agent_id,
                agentName: task.selected_agent.agent_name || `Agent ${task.selected_agent.agent_id}`,
                content: `Processing context for subtask: ${task.description?.substring(0, 60)}...`,
                relevance: 0.8,
                itemCount: 1
              })
            }
          })
        }
        
        // REPLACE old data entirely (don't append to prev)
        setMemoryActivities(realMemoryActivities.slice(0, 50))
        
        // Extract REAL inter-agent communication from subtasks
        
        subtasks.forEach((task: any, index: number) => {
          if (task.selected_agent) {
            // Task assignment is a form of communication from orchestrator to agent
            realCommunications.push({
              id: `comm_assign_${index}`,
              timestamp: new Date().toISOString(),
              fromAgent: 'Orchestrator',
              toAgent: task.selected_agent.agent_name || `Agent ${task.selected_agent.agent_id}`,
              messageType: 'query',
              priority: task.priority === 'high' ? 'high' : 'normal',
              content: `Task assigned: ${task.description?.substring(0, 60)}...`,
              status: 'sent'
            })
            
            // Task completion is a response from agent to orchestrator
            if (task.execution_result) {
              realCommunications.push({
                id: `comm_result_${index}`,
                timestamp: new Date().toISOString(),
                fromAgent: task.selected_agent.agent_name || `Agent ${task.selected_agent.agent_id}`,
                toAgent: 'Orchestrator',
                messageType: 'response',
                priority: task.execution_result.status === 'completed' ? 'normal' : 'high',
                content: `Task completed: ${task.execution_result.status} (${task.execution_result.tokens_used || 0} tokens)`,
                status: 'sent'
              })
            }
          }
        })
        
        // REPLACE old communication data entirely (don't append to prev)
        setCommunications(realCommunications.slice(0, 50))
        
      } catch (error) {
        console.error('Error fetching real execution data for orchestrator view:', error)
      }

      // Add learning insights
      if (Math.random() > 0.85) {
        const insightTypes: LearningInsight['type'][] = ['pattern', 'optimization', 'failure', 'success']
        const impacts: LearningInsight['impact'][] = ['low', 'medium', 'high']
        
        setLearningInsights(prev => [{
          id: `learn_${Date.now()}`,
          timestamp: new Date().toISOString(),
          type: insightTypes[Math.floor(Math.random() * insightTypes.length)],
          category: ['Performance', 'Quality', 'Efficiency', 'Collaboration'][Math.floor(Math.random() * 4)],
          description: `Identified optimization opportunity in task decomposition strategy`,
          impact: impacts[Math.floor(Math.random() * impacts.length)],
          confidence: Math.random()
        }, ...prev].slice(0, 20))
      }

      // Update metrics with REAL data (already have executionData and subtasks from above)
      try {
        if (executionData && subtasks.length > 0) {
          const completedSubtasks = subtasks.filter((t: any) => t.execution_result?.status === 'completed')
          const uniqueAgents = new Set(subtasks.map((t: any) => t.selected_agent?.agent_id).filter(Boolean))
          
          // Calculate real token usage
          const totalTokens = subtasks.reduce((sum: number, task: any) => 
            sum + (task.execution_result?.tokens_used || 0), 0)
          
          // Calculate success rate
          const successRate = subtasks.length > 0 
            ? completedSubtasks.length / subtasks.length 
            : 0
          
          // Calculate average response time
          const avgTime = subtasks.length > 0
            ? subtasks.reduce((sum: number, t: any) => sum + (t.execution_result?.execution_time_ms || 0), 0) / subtasks.length
            : 0
          
          // Calculate estimated cost (rough estimate: $0.01 per 1000 tokens for GPT-4)
          const estimatedCost = totalTokens * 0.00001
          
          setMetrics({
            totalTasks: subtasks.length,
            completedTasks: completedSubtasks.length,
            activeAgents: uniqueAgents.size,
            memoryOperations: realMemoryActivities.length,
            messagesExchanged: realCommunications.length,
            patternsLearned: executionData.input_data?.memory_consolidation?.results?.patterns_extracted || 0,
            avgResponseTime: avgTime,
            successRate: successRate,
            tokenUsage: totalTokens,
            estimatedCost: estimatedCost
          })
        }
      } catch (error) {
        console.error('Error updating metrics:', error)
      }
    }, 2000)

    return () => clearInterval(interval)
  }, [isExecuting, executionId, workflowId])

  const toggleSection = (section: string) => {
    setExpandedSections(prev => {
      const next = new Set(prev)
      if (next.has(section)) {
        next.delete(section)
      } else {
        next.add(section)
      }
      return next
    })
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <Cpu className="w-8 h-8 animate-spin mx-auto mb-4" />
          <p className="text-muted-foreground">Loading orchestration data...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <MetricsOverviewCard
        metrics={metrics}
        isExecuting={isExecuting}
        historicalData={historicalData}
        executionOptions={executionOptions}
        selectedExecutionId={selectedExecutionId}
        selectedExecution={selectedExecution}
        onSelectExecution={(value) => {
          userChangedSelectionRef.current = true
          setSelectedExecutionId(value)
        }}
      />

      {/* Tabbed Content */}
      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="space-y-4">
        <TabsList className="grid grid-cols-4 w-full">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="memory">Memory</TabsTrigger>
          <TabsTrigger value="communication">Communication</TabsTrigger>
          <TabsTrigger value="learning">Learning</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <OverviewTab
            memoryActivities={memoryActivities}
            communications={communications}
            expandedSections={expandedSections}
            onToggleSection={toggleSection}
          />
        </TabsContent>

        <TabsContent value="memory">
          <MemoryTab memoryActivities={memoryActivities} />
        </TabsContent>

        <TabsContent value="communication">
          <CommunicationTab communications={communications} />
        </TabsContent>

        <TabsContent value="learning">
          <LearningTab learningInsights={learningInsights} />
        </TabsContent>
      </Tabs>
    </div>
  )
}

