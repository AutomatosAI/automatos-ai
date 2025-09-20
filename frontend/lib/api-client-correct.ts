/**
 * CORRECT API Client - Using REAL discovered endpoints
 * Based on comprehensive API testing results
 */

class ApiClient {
  private baseUrl: string
  private apiKey: string

  constructor() {
    this.baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://206.81.0.227:8000'
    this.apiKey = process.env.NEXT_PUBLIC_API_KEY || 'test_api_key_for_backend_validation_2025'
    console.log('🚀 API Client initialized:', this.baseUrl)
  }

  async request(endpoint: string, options: RequestInit = {}) {
    if (endpoint.match(/^\/(agents|documents|workflows)$/)) {
      endpoint = endpoint + '/'
    }
    
    const url = `${this.baseUrl}/api${endpoint}`
    console.log('📡 API Request:', url)
    
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': this.apiKey,
          ...options.headers,
        },
      })

      if (!response.ok) {
        console.error('❌ API Error:', response.status, response.statusText)
        throw new Error(`API Error: ${response.status}`)
      }

      const data = await response.json()
      console.log('✅ API Response:', endpoint, 'data:', data)
      return data
    } catch (error) {
      console.error('❌ Request failed:', error)
      throw error
    }
  }

  // ===== AGENTS - Using discovered endpoints =====
  getAgents = async () => this.request('/agents/')
  getAgent = async (id: string) => this.request(`/agents/${id}`)
  getAgentsAvailable = async () => this.request('/agents/available')
  createAgent = async (data: any) => this.request('/agents/', { method: 'POST', body: JSON.stringify(data) })
  updateAgent = async (id: string, data: any) => this.request(`/agents/${id}`, { method: 'PUT', body: JSON.stringify(data) })
  deleteAgent = async (id: string) => this.request(`/agents/${id}`, { method: 'DELETE' })
  getAgentSkills = async (agentId: string) => this.request(`/agents/${agentId}/skills`)
  getAgentPatterns = async (agentId: string) => this.request(`/agents/${agentId}/patterns`)
  getAgentPerformance = async (agentId: string) => this.request(`/agents/${agentId}/performance`)
  
  // CORRECT URLs from discovery:
  getAgentStats = async () => this.request('/system/agent-statistics')
  getAgentTypes = async () => this.request('/system/agent-types')
  getAgentStatus = async (agentId: string) => this.request(`/system/agent/${agentId}/status`)
  executeAgent = async (agentId: string, data: any) => this.request(`/system/agent/${agentId}/execute`, { method: 'POST', body: JSON.stringify(data) })

  // ===== DOCUMENTS - Using discovered endpoints =====
  getDocuments = async () => this.request('/documents/')
  getDocument = async (id: string) => this.request(`/documents/${id}`)
  deleteDocument = async (id: string) => this.request(`/documents/${id}`, { method: 'DELETE' })
  uploadDocument = async (formData: FormData) => {
    const url = `${this.baseUrl}/api/documents/upload`
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'X-API-Key': this.apiKey },
      body: formData
    })
    return response.json()
  }
  preprocessDocument = async (data: any) => this.request('/documents/preprocess', { method: 'POST', body: JSON.stringify(data) })
  reprocessDocument = async (id: string) => this.request(`/documents/${id}/reprocess`, { method: 'POST' })
  getDocumentContent = async (id: string) => this.request(`/documents/${id}/content`)

  // Document stats - use context stats as discovered
  getDocumentStats = async () => this.request('/context/stats')
  getDocumentCategories = async () => this.request('/skills/categories') // Skills categories can work for documents

  // ===== WORKFLOWS - Using discovered endpoints =====
  getWorkflows = async () => this.request('/workflows')
  getWorkflow = async (id: string) => this.request(`/workflows/${id}`)
  createWorkflow = async (data: any) => this.request('/workflows', { method: 'POST', body: JSON.stringify(data) })
  getActiveWorkflows = async () => this.request('/workflows/active')
  getWorkflowStats = async () => this.request('/workflows/stats/dashboard')
  getWorkflowLiveProgress = async (id: string) => this.request(`/workflows/${id}/live-progress`)
  executeWorkflow = async (id: string, data: any) => this.request(`/workflows/${id}/execute`, { method: 'POST', body: JSON.stringify(data) })
  executeWorkflowAdvanced = async (id: string, data: any) => this.request(`/workflows/${id}/execute-advanced`, { method: 'POST', body: JSON.stringify(data) })
  getWorkflowExecutions = async () => this.request('/workflows/executions/')
  getWorkflowExecution = async (id: string) => this.request(`/workflows/executions/${id}`)
  getWorkflowExecutionResults = async (id: string) => this.request(`/workflows/executions/${id}/results`)
  getRecommendedTemplates = async () => this.request('/workflows/templates/recommended')

  // ===== SYSTEM - Using discovered endpoints =====
  getSystemHealth = async () => this.request('/system/health')
  getSystemMetrics = async () => this.request('/system/metrics')
  getSystemConfig = async () => this.request('/system/config')
  getSystemConfigKey = async (key: string) => this.request(`/system/config/${key}`)
  updateSystemConfig = async (key: string, value: any) => this.request(`/system/config/${key}`, { method: 'PUT', body: JSON.stringify(value) })
  getRagConfig = async () => this.request('/system/rag')
  getRagConfigById = async (id: string) => this.request(`/system/rag/${id}`)
  testRagConfig = async (id: string, data: any) => this.request(`/system/rag/${id}/test`, { method: 'POST', body: JSON.stringify(data) })
  getPerformanceBaseline = async () => this.request('/system/performance-baseline')
  getPerformanceComparison = async () => this.request('/system/performance-comparison')
  getSystemStateSummary = async () => this.request('/system/state/summary')

  // ===== SKILLS - Using discovered endpoints =====
  getSkills = async () => this.request('/skills/')
  createSkill = async (data: any) => this.request('/skills/', { method: 'POST', body: JSON.stringify(data) })
  createSkillsBulk = async (data: any) => this.request('/skills/bulk', { method: 'POST', body: JSON.stringify(data) })
  updateSkill = async (id: string, data: any) => this.request(`/skills/${id}`, { method: 'PUT', body: JSON.stringify(data) })
  deleteSkill = async (id: string) => this.request(`/skills/${id}`, { method: 'DELETE' })
  getSkillCategories = async () => this.request('/skills/categories')
  getAgentSkillsList = async (agentId: string) => this.request(`/skills/agents/${agentId}/`)
  addSkillToAgent = async (agentId: string, data: any) => this.request(`/skills/agents/${agentId}/`, { method: 'POST', body: JSON.stringify(data) })
  removeSkillFromAgent = async (agentId: string, skillId: string) => this.request(`/skills/agents/${agentId}/${skillId}`, { method: 'DELETE' })

  // ===== PATTERNS - Using discovered endpoints =====
  getPatterns = async () => this.request('/patterns/')
  createPattern = async (data: any) => this.request('/patterns/', { method: 'POST', body: JSON.stringify(data) })
  getPattern = async (id: string) => this.request(`/patterns/${id}`)
  deletePattern = async (id: string) => this.request(`/patterns/${id}`, { method: 'DELETE' })
  
  // ===== MULTI-AGENT - Using discovered endpoints =====
  getAgentCoordination = async () => this.request('/multi-agent/coordination/statistics')
  coordinateAgents = async (data: any) => this.request('/multi-agent/coordination/coordinate', { method: 'POST', body: JSON.stringify(data) })
  collaborativeReasoning = async (data: any) => this.request('/multi-agent/reasoning/collaborative', { method: 'POST', body: JSON.stringify(data) })
  monitorBehavior = async (data: any) => this.request('/multi-agent/behavior/monitor', { method: 'POST', body: JSON.stringify(data) })
  optimizeAgents = async (data: any) => this.request('/multi-agent/optimization/optimize', { method: 'POST', body: JSON.stringify(data) })
  rebalanceCoordination = async (data: any) => this.request('/multi-agent/coordination/rebalance', { method: 'POST', body: JSON.stringify(data) })
  getReasoningStats = async () => this.request('/multi-agent/reasoning/statistics')
  getCoordinationStats = async () => this.request('/multi-agent/coordination/statistics')
  getBehaviorStats = async () => this.request('/multi-agent/behavior/statistics')
  getOptimizationStats = async () => this.request('/multi-agent/optimization/statistics')
  getMultiAgentHealth = async () => this.request('/multi-agent/health')

  // ===== CONTEXT ENGINEERING - Using discovered endpoints =====
  getContextStats = async () => this.request('/context/stats')
  getContextPerformance = async () => this.request('/context/performance')
  getContextSources = async () => this.request('/context/sources')
  getRecentQueries = async () => this.request('/context/queries/recent')
  getContextPatterns = async () => this.request('/context/patterns')
  initializeContext = async (data: any) => this.request('/context/initialize', { method: 'POST', body: JSON.stringify(data) })
  processQuery = async (data: any) => this.request('/context-engineering/process-query', { method: 'POST', body: JSON.stringify(data) })
  processContent = async (data: any) => this.request('/context-engineering/process-content', { method: 'POST', body: JSON.stringify(data) })
  getRetrievalStats = async () => this.request('/context-engineering/retrieval-stats')
  getContextHealth = async () => this.request('/context-engineering/health')

  // ===== ANALYTICS - Using discovered endpoints =====
  getAnalyticsSuccessRate = async () => this.request('/analytics/dashboard/success-rate')
  getTaskCompletionTime = async () => this.request('/analytics/dashboard/task-completion-time')
  getSystemLoadTrend = async () => this.request('/analytics/dashboard/system-load-trend')
  getErrorRateByType = async () => this.request('/analytics/dashboard/error-rate-by-type')
  getQueueDepth = async () => this.request('/analytics/dashboard/queue-depth')
  getEfficiencyScore = async () => this.request('/analytics/dashboard/efficiency-score')
  getCostPerExecution = async () => this.request('/analytics/performance/cost-per-execution')
  getPeakUsageHours = async () => this.request('/analytics/performance/peak-usage-hours')
  getBottlenecks = async () => this.request('/analytics/performance/bottlenecks')
  getPredictiveAlerts = async () => this.request('/analytics/performance/predictive-alerts')
  getAgentRanking = async () => this.request('/analytics/performance/agent-ranking')
  getSlaCompliance = async () => this.request('/analytics/performance/sla-compliance')
  getAllMetrics = async () => this.request('/analytics/dashboard/all-metrics')
  getAllEnhancements = async () => this.request('/analytics/performance/all-enhancements')

  // ===== EVALUATION - Using discovered endpoints =====
  evaluateSystem = async (data: any) => this.request('/evaluation/evaluate', { method: 'POST', body: JSON.stringify(data) })
  getEvaluationResults = async (id: string) => this.request(`/evaluation/results/${id}`)
  getEvaluationHistory = async () => this.request('/evaluation/history')
  getPerformanceMetrics = async () => this.request('/evaluation/performance-metrics')
  getEvaluationHealth = async () => this.request('/evaluation/health')

  // ===== ORCHESTRATOR - Using discovered endpoints =====
  submitTask = async (data: any) => this.request('/orchestrator/task/submit', { method: 'POST', body: JSON.stringify(data) })
  analyzeTask = async (data: any) => this.request('/orchestrator/task/analyze', { method: 'POST', body: JSON.stringify(data) })
  executePhase = async (data: any) => this.request('/orchestrator/execute-phase', { method: 'POST', body: JSON.stringify(data) })
  getTaskStatus = async (taskId: string) => this.request(`/orchestrator/task/${taskId}/status`)
  getActiveTasks = async () => this.request('/orchestrator/tasks/active')

  // ===== MEMORY - Using discovered endpoints =====
  storeMemory = async (data: any) => this.request('/v1/memory/store', { method: 'POST', body: JSON.stringify(data) })
  retrieveMemory = async (sessionId: string) => this.request(`/v1/memory/retrieve/${sessionId}`)
  getMemoryStats = async () => this.request('/v1/memory/stats')
  getMemoryHealth = async () => this.request('/v1/memory/health')
  getMemory = async (id: string) => this.request(`/v1/memory/${id}`)
  updateMemory = async (id: string, data: any) => this.request(`/v1/memory/${id}`, { method: 'PUT', body: JSON.stringify(data) })
  deleteMemory = async (id: string) => this.request(`/v1/memory/${id}`, { method: 'DELETE' })
  searchMemory = async (data: any) => this.request('/v1/memory/search', { method: 'POST', body: JSON.stringify(data) })
  clearMemory = async (data: any) => this.request('/v1/memory/clear', { method: 'POST', body: JSON.stringify(data) })

  // ===== TOOLS - Using discovered endpoints =====
  getTools = async () => this.request('/tools/')
  getTool = async (id: string) => this.request(`/tools/${id}`)
  installTool = async (id: string, data: any) => this.request(`/tools/${id}/install`, { method: 'POST', body: JSON.stringify(data) })
  configureTool = async (id: string, data: any) => this.request(`/tools/${id}/configure`, { method: 'POST', body: JSON.stringify(data) })
  uninstallTool = async (id: string) => this.request(`/tools/${id}/uninstall`, { method: 'DELETE' })
  getToolStatus = async (id: string) => this.request(`/tools/${id}/status`)
  getToolUsageStats = async (id: string) => this.request(`/tools/${id}/usage-stats`)
  getToolsHealth = async () => this.request('/tools/health')

  // ===== CREDENTIALS - Using discovered endpoints =====
  createCredential = async (data: any) => this.request('/credentials/', { method: 'POST', body: JSON.stringify(data) })
  getCredentials = async () => this.request('/credentials/')
  getCredential = async (toolId: string) => this.request(`/credentials/${toolId}`)
  updateCredential = async (id: string, data: any) => this.request(`/credentials/${id}`, { method: 'PUT', body: JSON.stringify(data) })
  deleteCredential = async (id: string) => this.request(`/credentials/${id}`, { method: 'DELETE' })
  testCredentialConnection = async (data: any) => this.request('/credentials/test-connection', { method: 'POST', body: JSON.stringify(data) })
  getCredentialsAuditLogs = async () => this.request('/credentials/audit/logs')
  getCredentialsHealth = async () => this.request('/credentials/health')

  // ===== PERMISSIONS - Using discovered endpoints =====
  getPermissionsMatrix = async () => this.request('/permissions/matrix')
  getPermissionsAssignments = async () => this.request('/permissions/assignments')
  assignPermission = async (data: any) => this.request('/permissions/assign', { method: 'POST', body: JSON.stringify(data) })
  bulkAssignPermissions = async (data: any) => this.request('/permissions/bulk-assign', { method: 'POST', body: JSON.stringify(data) })
  revokePermission = async (data: any) => this.request('/permissions/revoke', { method: 'DELETE', body: JSON.stringify(data) })
  getPermissionsAudit = async () => this.request('/permissions/audit')
  getPermissionsHealth = async () => this.request('/permissions/health')

  // ===== TEMPLATES - Using discovered endpoints =====
  getTemplates = async () => this.request('/templates/')
  getTemplatesByType = async (agentType: string) => this.request(`/templates/templates/${agentType}`)
  getSkillSuggestions = async () => this.request('/templates/templates/skills/suggestions')
  getCreationWizardConfig = async () => this.request('/templates/creation-wizard/config')

  // ===== PLAYBOOKS - Using discovered endpoints =====
  getPlaybooks = async () => this.request('/playbooks')
  minePlaybook = async (data: any) => this.request('/playbooks/mine', { method: 'POST', body: JSON.stringify(data) })

  // ===== FIELD THEORY - Using discovered endpoints =====
  updateField = async (data: any) => this.request('/field-theory/fields/update', { method: 'POST', body: JSON.stringify(data) })
  propagateField = async (data: any) => this.request('/field-theory/fields/propagate', { method: 'POST', body: JSON.stringify(data) })
  getFieldInteractions = async () => this.request('/field-theory/fields/interactions')
  getFieldStatistics = async () => this.request('/field-theory/fields/statistics')
  getFieldStates = async () => this.request('/field-theory/fields/states')
  getFieldTheoryHealth = async () => this.request('/field-theory/health')

  // ===== CHATBOT - Using discovered endpoints =====
  chatbotQuery = async (data: any) => this.request('/chatbot/query', { method: 'POST', body: JSON.stringify(data) })
  getChatbotSuggestions = async () => this.request('/chatbot/suggestions')
  chatbotExecute = async (data: any) => this.request('/chatbot/execute', { method: 'POST', body: JSON.stringify(data) })
  getChatHistory = async (sessionId: string) => this.request(`/chatbot/history/${sessionId}`)
  submitChatFeedback = async (data: any) => this.request('/chatbot/feedback', { method: 'POST', body: JSON.stringify(data) })

  // ===== LEARNING - Using discovered endpoints =====
  submitFeedback = async (data: any) => this.request('/learning/feedback', { method: 'POST', body: JSON.stringify(data) })
  processFeedback = async (data: any) => this.request('/learning/feedback/process', { method: 'POST', body: JSON.stringify(data) })
  getAggregateInsights = async () => this.request('/learning/insights/aggregate')
  getEffectivenessReport = async () => this.request('/learning/effectiveness-report')

  // ===== BACKWARD COMPATIBILITY =====
  getActivities = async () => this.request('/orchestrator/tasks/active')
  getProcessingStatus = async (documentId: string) => this.getDocument(documentId)
  getAgentConfig = async (agentId: string) => this.getAgent(agentId)
  updateAgentConfig = async (agentId: string, config: any) => this.updateAgent(agentId, { configuration: config })
  startAgent = async (agentId: string) => this.executeAgent(agentId, { action: 'start' })
  stopAgent = async (agentId: string) => this.executeAgent(agentId, { action: 'stop' })
  pauseAgent = async (agentId: string) => this.executeAgent(agentId, { action: 'pause' })
  getAgentLogs = async (agentId: string) => this.getAgentPerformance(agentId)
  getAgentMetrics = async (agentId: string) => this.getAgentPerformance(agentId)
}

export const apiClient = new ApiClient()
export default apiClient
