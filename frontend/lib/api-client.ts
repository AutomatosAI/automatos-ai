/**
 * API Client - Only calls endpoints that actually exist based on test results
 * Base URL: http://206.81.0.227:8000
 */

interface ApiResponse<T = any> {
  data: T
  success: boolean
  message?: string
  error?: string
}

interface ApiError {
  message: string
  status: number
  code?: string
}

class ApiClient {
  private baseUrl: string
  private defaultHeaders: Record<string, string>

  constructor() {
    this.baseUrl = (typeof window !== 'undefined' && (window as any).env?.NEXT_PUBLIC_API_URL) || ''
    this.defaultHeaders = {
      'Content-Type': 'application/json',
    }
    
    console.log('🚀 API Client initialized with baseUrl:', this.baseUrl)
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`
    
    console.log('🔍 API Call:', { url, method: options.method || 'GET' })
    
    const config: RequestInit = {
      ...options,
      headers: {
        ...this.defaultHeaders,
        ...options.headers,
      },
    }

    try {
      const response = await fetch(url, config)
      
      if (!response.ok) {
        console.error('❌ API Error:', response.status, response.statusText)
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const data = await response.json()
      console.log('✅ API Success:', endpoint, 'Data type:', Array.isArray(data) ? `array[${data.length}]` : typeof data)
      
      return data
    } catch (error: any) {
      console.error('🚨 API Failed:', endpoint, error.message)
      throw error
    }
  }

  // ===== SYSTEM ENDPOINTS =====
  async getSystemHealth() {
    return this.request('/api/system/health')
  }

  async getSystemMetrics() {
    return this.request('/api/system/metrics')
  }

  async getSystemConfig() {
    return this.request('/api/system/config')
  }

  async updateSystemConfig(data: any) {
    return this.request('/api/system/config', {
      method: 'PUT',
      body: JSON.stringify(data)
    })
  }

  async getSystemConfigKey(key: string) {
    return this.request(`/api/system/config/${key}`)
  }

  async updateSystemConfigKey(key: string, value: any) {
    return this.request(`/api/system/config/${key}`, {
      method: 'PUT',
      body: JSON.stringify({ value })
    })
  }

  async getSystemRAG() {
    return this.request('/api/system/rag')
  }

  async updateSystemRAG(data: any) {
    return this.request('/api/system/rag', {
      method: 'PUT',
      body: JSON.stringify(data)
    })
  }

  async getSystemRAGConfig(id: string) {
    return this.request(`/api/system/rag/${id}`)
  }

  async testSystemRAG(id: string) {
    return this.request(`/api/system/rag/${id}/test`, {
      method: 'POST'
    })
  }

  async testSystemRoute() {
    return this.request('/api/system/test-route')
  }

  async getSystemAgentTypes() {
    return this.request('/api/system/agent-types')
  }

  async getSystemAgentStatistics() {
    return this.request('/api/system/agent-statistics')
  }

  async getSystemAgentStatus(agentId: string) {
    return this.request(`/api/system/agent/${agentId}/status`)
  }

  async executeSystemAgent(agentId: string, data: any) {
    return this.request(`/api/system/agent/${agentId}/execute`, {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  async getSystemPerformanceBaseline() {
    return this.request('/api/system/performance-baseline')
  }

  async updateSystemLearningState(data: any) {
    return this.request('/api/system/learning-state/update', {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  async runSystemPerformanceTest(data: any) {
    return this.request('/api/system/performance-test', {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  async getSystemPerformanceComparison(data: any) {
    return this.request('/api/system/performance-comparison', {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  // ===== AGENT ENDPOINTS =====
  async getAgents(skip = 0, limit = 100) {
    return this.request(`/api/agents/?skip=${skip}&limit=${limit}`)
  }

  async getAgent(id: string) {
    return this.request(`/api/agents/${id}`)
  }

  async createAgent(data: any) {
    return this.request('/api/agents/', {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  async updateAgent(id: string, data: any) {
    return this.request(`/api/agents/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data)
    })
  }

  async deleteAgent(id: string) {
    return this.request(`/api/agents/${id}`, {
      method: 'DELETE'
    })
  }

  async getAvailableAgents() {
    return this.request('/api/agents/available')
  }

  async getAgentSkills(agentId: string) {
    return this.request(`/api/agents/${agentId}/skills`)
  }

  async updateAgentSkills(agentId: string, skills: any[]) {
    return this.request(`/api/agents/${agentId}/skills`, {
      method: 'PUT',
      body: JSON.stringify(skills)
    })
  }

  async getAgentPatterns(agentId: string) {
    return this.request(`/api/agents/${agentId}/patterns`)
  }

  async updateAgentPatterns(agentId: string, patterns: any[]) {
    return this.request(`/api/agents/${agentId}/patterns`, {
      method: 'PUT',
      body: JSON.stringify(patterns)
    })
  }

  async getAgentPerformance(agentId: string) {
    return this.request(`/api/agents/${agentId}/performance`)
  }

  // ===== WORKFLOW ENDPOINTS =====
  async getWorkflows() {
    return this.request('/api/workflows')
  }

  async createWorkflow(data: any) {
    return this.request('/api/workflows', {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  async getActiveWorkflows() {
    return this.request('/api/workflows/active')
  }

  async getWorkflowStatsDashboard() {
    return this.request('/api/workflows/stats/dashboard')
  }

  async getWorkflowLiveProgress(workflowId: string) {
    return this.request(`/api/workflows/${workflowId}/live-progress`)
  }

  async executeWorkflowAdvanced(workflowId: string, data: any) {
    return this.request(`/api/workflows/${workflowId}/execute-advanced`, {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  async executeWorkflow(workflowId: string, data: any) {
    return this.request(`/api/workflows/${workflowId}/execute`, {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  async executeWorkflowDirect(data: any) {
    return this.request('/api/workflows/execute', {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  async getWorkflowExecutions() {
    return this.request('/api/workflows/executions/')
  }

  async getWorkflowExecution(executionId: string) {
    return this.request(`/api/workflows/executions/${executionId}`)
  }

  async getWorkflowExecutionResults(executionId: string) {
    return this.request(`/api/workflows/executions/${executionId}/results`)
  }

  async getRecommendedWorkflowTemplates() {
    return this.request('/api/workflows/templates/recommended')
  }

  // ===== DOCUMENT ENDPOINTS =====
  async uploadDocument(file: File, metadata?: any) {
    const formData = new FormData()
    formData.append('file', file)
    if (metadata) {
      formData.append('metadata', JSON.stringify(metadata))
    }
    
    return fetch(`${this.baseUrl}/api/documents/upload`, {
      method: 'POST',
      body: formData
    }).then(response => response.json())
  }

  async preprocessDocument(data: any) {
    return this.request('/api/documents/preprocess', {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  async getDocuments() {
    return this.request('/api/documents/')
  }

  async getDocument(id: string) {
    return this.request(`/api/documents/${id}`)
  }

  async updateDocument(id: string, data: any) {
    return this.request(`/api/documents/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data)
    })
  }

  async deleteDocument(id: string) {
    return this.request(`/api/documents/${id}`, {
      method: 'DELETE'
    })
  }

  async reprocessDocument(id: string) {
    return this.request(`/api/documents/${id}/reprocess`, {
      method: 'POST'
    })
  }

  async getDocumentContent(id: string) {
    return this.request(`/api/documents/${id}/content`)
  }

  // ===== SKILLS ENDPOINTS =====
  async getSkills() {
    return this.request('/api/skills/')
  }

  async createSkill(data: any) {
    return this.request('/api/skills/single', {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  async getSkill(id: string) {
    return this.request(`/api/skills/${id}`)
  }

  async updateSkill(id: string, data: any) {
    return this.request(`/api/skills/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data)
    })
  }

  async deleteSkill(id: string) {
    return this.request(`/api/skills/${id}`, {
      method: 'DELETE'
    })
  }

  async getSkillCategories() {
    return this.request('/api/skills/categories')
  }

  async createSkillsBulk(skillsData: any[]) {
    return this.request('/api/skills/bulk', {
      method: 'POST',
      body: JSON.stringify(skillsData)
    })
  }

  async getAgentSkillsFromSkillsAPI(agentId: string) {
    return this.request(`/api/skills/agents/${agentId}/`)
  }

  async addSkillToAgent(agentId: string, skillId: string) {
    return this.request(`/api/skills/agents/${agentId}/`, {
      method: 'POST',
      body: JSON.stringify([parseInt(skillId)])
    })
  }

  async removeSkillFromAgent(agentId: string, skillId: string) {
    return this.request(`/api/skills/agents/${agentId}/${skillId}`, {
      method: 'DELETE'
    })
  }

  // ===== CHATBOT ENDPOINTS =====
  async sendChatbotQuery(message: string, context?: any) {
    return this.request('/api/chatbot/query', {
      method: 'POST',
      body: JSON.stringify({ message, context })
    })
  }

  async getChatbotSuggestions() {
    return this.request('/api/chatbot/suggestions')
  }

  async executeChatbotAction(data: any) {
    return this.request('/api/chatbot/execute', {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  async getChatbotHistory(sessionId: string) {
    return this.request(`/api/chatbot/history/${sessionId}`)
  }

  async sendChatbotFeedback(data: any) {
    return this.request('/api/chatbot/feedback', {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  // ===== LEGACY METHODS (for backward compatibility) =====
  async getSystemActivities(limit = 10) {
    console.log('⚠️ Activities endpoint not implemented, returning mock data')
    return []
  }

  async getAgentLogs(id: string) {
    console.log('⚠️ Agent logs endpoint not implemented')
    return []
  }

  async getAgentStats(id: string) {
    console.log('⚠️ Agent stats endpoint not implemented')
    return { status: 'active', performance: 85 }
  }

  async startAgent(id: string) {
    throw new Error('Start agent endpoint not implemented')
  }

  async stopAgent(id: string) {
    throw new Error('Stop agent endpoint not implemented')
  }

  async getDocumentAnalytics() {
    console.log('⚠️ Document analytics endpoint not implemented')
    return { totalDocuments: 0, totalSize: 0 }
  }

  async runWorkflow(id: string, inputs?: any) {
    return this.executeWorkflow(id, inputs || {})
  }

  async getWorkflowTemplates() {
    return this.getRecommendedWorkflowTemplates()
  }

  async getPerformanceAnalytics(timeRange: string) {
    console.log('⚠️ Performance analytics endpoint not implemented')
    return {}
  }

  async getUsageAnalytics(timeRange: string) {
    console.log('⚠️ Usage analytics endpoint not implemented')
    return {}
  }

  async getAgentAnalytics(timeRange: string) {
    console.log('⚠️ Agent analytics endpoint not implemented')
    return {}
  }

  async getSettings() {
    return this.getSystemConfig()
  }

  async updateSettings(data: any) {
    return this.updateSystemConfig(data)
  }


  async sendChatMessage(message: string, context?: any) {
    return this.sendChatbotQuery(message, context)
  }

  async testChat() {
    return this.testSystemRoute()
  }

  async processDocument(documentId: string) {
    return this.reprocessDocument(documentId)
  }

  // Analytics methods that return null if not implemented
  async getAnalyticsOverview() {
    try {
      return await this.request('/api/documents/analytics/overview')
    } catch {
      return null
    }
  }

  async getSearchPatterns() {
    try {
      return await this.request('/api/documents/analytics/search-patterns')
    } catch {
      return null
    }
  }

  async getProcessingPipeline() {
    try {
      return await this.request('/api/documents/processing/pipeline')
    } catch {
      return null
    }
  }

  async getLiveProcessingJobs() {
    try {
      return await this.request('/api/documents/processing/live')
    } catch {
      return null
    }
  }

  async reprocessAllDocuments() {
    try {
      return await this.request('/api/documents/processing/reprocess-all', {
        method: 'POST'
      })
    } catch {
      return null
    }
  }

  async getDocumentChunks(documentId: string) {
    try {
      return await this.request(`/api/documents/${documentId}/chunks`)
    } catch {
      return null
    }
  }

  async searchDocument(documentId: string, query: string) {
    try {
      return await this.request(`/api/documents/${documentId}/search`, {
        method: 'POST',
        body: JSON.stringify({ query })
      })
    } catch {
      return null
    }
  }
}

export const apiClient = new ApiClient()