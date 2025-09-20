/**
 * Standardized API Client
 * Simple, reliable connection to backend
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
    // Ensure endpoint has trailing slash for list endpoints
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
      console.log('✅ API Response:', endpoint, 'count:', Array.isArray(data) ? data.length : 'object')
      return data
    } catch (error) {
      console.error('❌ Request failed:', error)
      throw error
    }
  }

  // Standardized endpoints - all with proper paths
  getAgents = () => this.request('/agents')
  getAgent = (id: string) => this.request(`/agents/${id}`)
  createAgent = (data: any) => this.request('/agents', {
    method: 'POST',
    body: JSON.stringify(data)
  })
  updateAgent = (id: string, data: any) => this.request(`/agents/${id}`, {
    method: 'PUT',
    body: JSON.stringify(data)
  })
  deleteAgent = (id: string) => this.request(`/agents/${id}`, {
    method: 'DELETE'
  })

  getDocuments = () => this.request('/documents')
  getDocument = (id: string) => this.request(`/documents/${id}`)
  
  getWorkflows = () => this.request('/workflows')
  getWorkflow = (id: string) => this.request(`/workflows/${id}`)
  createWorkflow = (data: any) => this.request('/workflows', {
    method: 'POST',
    body: JSON.stringify(data)
  })

  // System endpoints
  getSystemHealth = () => this.request('/system/health')
  getSystemMetrics = () => this.request('/system/metrics')
  
  // Placeholder for unimplemented endpoints
  getAgentStats = () => {
    console.log('⚠️ Agent stats endpoint not implemented')
    return Promise.resolve({ total: 0, active: 0, idle: 0 })
  }
  
  getAgentTypes = () => {
    console.log('⚠️ Agent types endpoint not implemented, returning defaults')
    return Promise.resolve([
      'reasoning', 'coding', 'planning', 'execution',
      'monitoring', 'analysis', 'synthesis', 'validation'
    ])
  }

  getActivities = () => {
    console.log('⚠️ Activities endpoint not implemented, returning mock data')
    return Promise.resolve([])
  }
}

export const apiClient = new ApiClient()
export default apiClient
