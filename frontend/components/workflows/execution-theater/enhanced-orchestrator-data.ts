// EnhancedOrchestratorView's data: its types, and the pure steps that turn an
// execution (loaded once, or polled every 2s while it runs) into what the view
// shows. Extracted from the component so it stays under the code-shape limit;
// no behaviour change.

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
export function deriveMemoryActivities(execution: any, subtasks: any[]): MemoryActivity[] {
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
export function deriveCommunications(execution: any, subtasks: any[]): CommunicationEvent[] {
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
export function deriveLearningInsights(execution: any): LearningInsight[] | null {
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
export function deriveOrchestrationMetrics(
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

export const EMPTY_METRICS: OrchestrationMetrics = {
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

// --- the execution list ---------------------------------------------------

function executionOption(item: any): ExecutionOption | null {
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
}

// The workflow's executions, newest first, with the current run added when the
// list does not have it yet.
export function normalizeExecutionOptions(response: any, executionId?: string): ExecutionOption[] {
  const rawItems = Array.isArray(response) ? response : response?.items || []

  const normalized: ExecutionOption[] = rawItems
    .map(executionOption)
    .filter((option: ExecutionOption | null): option is ExecutionOption => Boolean(option?.id))
    .sort((a: ExecutionOption, b: ExecutionOption) => {
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

  return normalized
}

// --- while it runs: the live-progress endpoint ------------------------------

export function liveProgressMemoryActivities(liveData: any): MemoryActivity[] {
  return liveData.memory_activities.map((activity: any) => ({
    id: `mem_live_${Date.now()}_${Math.random()}`,
    timestamp: new Date().toISOString(),
    type: activity.type || 'working',
    action: activity.action || 'store',
    agentId: activity.agent_id,
    agentName: activity.agent_name,
    content: activity.content,
    relevance: activity.relevance,
    itemCount: activity.item_count
  }))
}

export function liveProgressCommunications(liveData: any): CommunicationEvent[] {
  return liveData.communications.map((comm: any) => ({
    id: `comm_live_${Date.now()}_${Math.random()}`,
    timestamp: new Date().toISOString(),
    fromAgent: comm.from_agent,
    toAgent: comm.to_agent,
    messageType: comm.type,
    priority: comm.priority,
    content: comm.content,
    status: comm.status
  }))
}

// --- while it runs: the execution itself, re-read every poll ----------------

const LIVE_RETRIEVAL_KINDS = [
  { key: 'working_memory', id: 'working', type: 'working', relevance: 0.9,
    content: 'Memory operation: Processing context for subtask execution' },
  { key: 'short_term', id: 'short', type: 'short_term', relevance: 0.75,
    content: 'Recent interactions and task history retrieved' },
  { key: 'long_term', id: 'long', type: 'long_term', relevance: 0.85,
    content: 'Historical patterns and learned knowledge retrieved' }
] as const

function liveRetrievals(executionData: any): MemoryActivity[] {
  const activities: MemoryActivity[] = []
  const memRetrieval = executionData.input_data.memory_retrieval.results
  const agentMemories = memRetrieval?.agent_memories || {}

  Object.entries(agentMemories).forEach(([agentId, memories]: [string, any]) => {
    LIVE_RETRIEVAL_KINDS.forEach(kind => {
      const items = memories[kind.key] || []
      if (items.length > 0) {
        activities.push({
          id: `mem_retrieve_${kind.id}_${agentId}`,
          timestamp: new Date().toISOString(),
          type: kind.type,
          action: 'retrieve',
          agentId: parseInt(agentId),
          agentName: getAgentName(executionData, parseInt(agentId)),
          content: kind.content,
          relevance: kind.relevance,
          itemCount: items.length
        })
      }
    })
  })
  return activities
}

function liveStores(executionData: any): MemoryActivity[] {
  const activities: MemoryActivity[] = []
  const memStorage = executionData.input_data.memory_storage.results
  const perAgent = memStorage?.per_agent || {}

  Object.entries(perAgent).forEach(([agentId, storage]: [string, any]) => {
    if (storage.experiences_stored > 0) {
      activities.push({
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
  return activities
}

function liveConsolidations(executionData: any): MemoryActivity[] {
  const activities: MemoryActivity[] = []
  const memConsol = executionData.input_data.memory_consolidation.results
  const agentsConsolidated = memConsol?.agents_consolidated || []

  agentsConsolidated.forEach((agentId: number) => {
    activities.push({
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
  return activities
}

function liveSubtaskActivities(subtasks: any[]): MemoryActivity[] {
  const activities: MemoryActivity[] = []
  subtasks.forEach((task: any, index: number) => {
    if (task.selected_agent) {
      activities.push({
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
  return activities
}

// The memory records' retrievals, stores and consolidations; with none, one
// working-memory activity per assigned subtask.
export function liveMemoryActivities(executionData: any, subtasks: any[]): MemoryActivity[] {
  const input = executionData.input_data
  const activities = [
    ...(input?.memory_retrieval?.is_real ? liveRetrievals(executionData) : []),
    ...(input?.memory_storage?.is_real ? liveStores(executionData) : []),
    ...(input?.memory_consolidation?.is_real ? liveConsolidations(executionData) : [])
  ]
  if (activities.length === 0 && subtasks.length > 0) {
    return liveSubtaskActivities(subtasks)
  }
  return activities
}

// A subtask's assignment (orchestrator to agent) and, once it has one, its
// result (agent to orchestrator).
export function liveCommunications(subtasks: any[]): CommunicationEvent[] {
  const communications: CommunicationEvent[] = []
  subtasks.forEach((task: any, index: number) => {
    if (!task.selected_agent) {
      return
    }
    communications.push({
      id: `comm_assign_${index}`,
      timestamp: new Date().toISOString(),
      fromAgent: 'Orchestrator',
      toAgent: task.selected_agent.agent_name || `Agent ${task.selected_agent.agent_id}`,
      messageType: 'query',
      priority: task.priority === 'high' ? 'high' : 'normal',
      content: `Task assigned: ${task.description?.substring(0, 60)}...`,
      status: 'sent'
    })

    if (task.execution_result) {
      communications.push({
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
  })
  return communications
}

// One poll in seven or so adds an insight; its kind, category and impact are random.
export function randomLearningInsight(): LearningInsight | null {
  if (Math.random() <= 0.85) {
    return null
  }
  const insightTypes: LearningInsight['type'][] = ['pattern', 'optimization', 'failure', 'success']
  const impacts: LearningInsight['impact'][] = ['low', 'medium', 'high']

  return {
    id: `learn_${Date.now()}`,
    timestamp: new Date().toISOString(),
    type: insightTypes[Math.floor(Math.random() * insightTypes.length)],
    category: ['Performance', 'Quality', 'Efficiency', 'Collaboration'][Math.floor(Math.random() * 4)],
    description: `Identified optimization opportunity in task decomposition strategy`,
    impact: impacts[Math.floor(Math.random() * impacts.length)],
    confidence: Math.random()
  }
}
