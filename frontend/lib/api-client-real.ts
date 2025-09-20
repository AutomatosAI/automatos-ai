/**
 * Real API Client - NO MOCK DATA
 * Only real endpoints, no fallbacks
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

  // AGENTS - Real endpoints only
  getAgents = async () => this.request('/agents/')
  getAgent = async (id: string) => this.request(`/agents/${id}`)
  getAgentsAvailable = async () => this.request('/agents/available')
  createAgent = async (data: any) => this.request('/agents/', { method: 'POST', body: JSON.stringify(data) })
  updateAgent = async (id: string, data: any) => this.request(`/agents/${id}`, { method: 'PUT', body: JSON.stringify(data) })
  deleteAgent = async (id: string) => this.request(`/agents/${id}`, { method: 'DELETE' })
  getAgentSkills = async (agentId: string) => this.request(`/agents/${agentId}/skills`)
  getAgentPatterns = async (agentId: string) => this.request(`/agents/${agentId}/patterns`)
  getAgentPerformance = async (agentId: string) => this.request(`/agents/${agentId}/performance`)

  // DOCUMENTS - Real endpoints only
  getDocuments = async () => this.request('/documents/')
  getDocument = async (id: string) => this.request(`/documents/${id}`)
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

  // WORKFLOWS - Real endpoints only
  getWorkflows = async () => this.request('/workflows')
  getWorkflow = async (id: string) => this.request(`/workflows/${id}`)
  getActiveWorkflows = async () => this.request('/workflows/active')
  getWorkflowStats = async () => this.request('/workflows/stats/dashboard')
  getWorkflowLiveProgress = async (id: string) => this.request(`/workflows/${id}/live-progress`)
  executeWorkflow = async (id: string, data: any) => this.request(`/workflows/${id}/execute`, { method: 'POST', body: JSON.stringify(data) })
  executeWorkflowAdvanced = async (id: string, data: any) => this.request(`/workflows/${id}/execute-advanced`, { method: 'POST', body: JSON.stringify(data) })
  createWorkflow = async (data: any) => this.request('/workflows', { method: 'POST', body: JSON.stringify(data) })
  getWorkflowExecutions = async () => this.request('/workflows/executions/')
  getWorkflowExecution = async (id: string) => this.request(`/workflows/executions/${id}`)
  getWorkflowExecutionResults = async (id: string) => this.request(`/workflows/executions/${id}/results`)
  getRecommendedTemplates = async () => this.request('/workflows/templates/recommended')

  // SYSTEM - Real endpoints only
  getSystemHealth = async () => this.request('/system/health')
  getSystemMetrics = async () => this.request('/system/metrics')
  getSystemConfig = async () => this.request('/system/config')
  getSystemConfigKey = async (key: string) => this.request(`/system/config/${key}`)
  updateSystemConfig = async (key: string, value: any) => this.request(`/system/config/${key}`, { method: 'PUT', body: JSON.stringify(value) })
  getRagConfig = async () => this.request('/system/rag')
  getRagConfigById = async (id: string) => this.request(`/system/rag/${id}`)
  testRagConfig = async (id: string, data: any) => this.request(`/system/rag/${id}/test`, { method: 'POST', body: JSON.stringify(data) })

  // ENDPOINTS THAT NEED BACKEND IMPLEMENTATION
  // These will fail until backend implements them
  getAgentStats = async () => this.request('/agents/stats')
  getAgentTypes = async () => this.request('/agents/types')
  getDocumentStats = async () => this.request('/documents/stats')
  getDocumentCategories = async () => this.request('/documents/categories')
  getProcessingStatus = async (documentId: string) => this.request(`/documents/${documentId}/processing-status`)
  getSkills = async () => this.request('/skills')
  addSkillToAgent = async (agentId: string, skillId: string) => this.request(`/agents/${agentId}/skills`, { method: 'POST', body: JSON.stringify({ skill_id: skillId }) })
  removeSkillFromAgent = async (agentId: string, skillId: string) => this.request(`/agents/${agentId}/skills/${skillId}`, { method: 'DELETE' })
  createSkill = async (data: any) => this.request('/skills', { method: 'POST', body: JSON.stringify(data) })
  createSkillsBulk = async (data: any) => this.request('/skills/bulk', { method: 'POST', body: JSON.stringify(data) })
  getAgentConfig = async (agentId: string) => this.request(`/agents/${agentId}/configuration`)
  updateAgentConfig = async (agentId: string, config: any) => this.request(`/agents/${agentId}/configuration`, { method: 'PUT', body: JSON.stringify(config) })
  getAgentCoordination = async () => this.request('/coordination/status')
  coordinateAgents = async (data: any) => this.request('/coordination/coordinate', { method: 'POST', body: JSON.stringify(data) })
  collaborativeReasoning = async (data: any) => this.request('/reasoning/collaborative', { method: 'POST', body: JSON.stringify(data) })
  startAgent = async (agentId: string) => this.request(`/agents/${agentId}/start`, { method: 'POST' })
  stopAgent = async (agentId: string) => this.request(`/agents/${agentId}/stop`, { method: 'POST' })
  pauseAgent = async (agentId: string) => this.request(`/agents/${agentId}/pause`, { method: 'POST' })
  getAgentLogs = async (agentId: string) => this.request(`/agents/${agentId}/logs`)
  getAgentMetrics = async (agentId: string) => this.request(`/agents/${agentId}/metrics`)
  getActivities = async () => this.request('/activities')
}

export const apiClient = new ApiClient()
export default apiClient
