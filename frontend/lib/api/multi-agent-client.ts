
/**
 * Multi-Agent Systems API Client
 * Connects frontend to backend multi-agent capabilities
 */

interface ApiResponse<T> {
  data?: T
  error?: string
  status: number
}

interface Agent {
  id: string
  name: string
  type: string
  status: 'active' | 'inactive' | 'paused'
  performance?: number
  uptime?: number
  tasks_completed?: number
  error_rate?: number
}

interface CollaborativeReasoningRequest {
  agents: Agent[]
  task: {
    description: string
    type?: string
  }
  strategy?: 'majority_vote' | 'weighted_consensus' | 'expert_override' | 'iterative_refinement'
}

interface CoordinationRequest {
  agents: Agent[]
  strategy?: 'sequential' | 'parallel' | 'hierarchical' | 'mesh' | 'adaptive'
}

interface BehaviorMonitoringRequest {
  agents: Agent[]
  task_data: Record<string, any>
}

interface OptimizationRequest {
  agents: Agent[]
  objectives?: string[]
  strategy?: 'gradient_descent' | 'genetic_algorithm' | 'bayesian_optimization' | 'simulated_annealing'
}

class MultiAgentApiClient {
  private baseUrl: string

  constructor(baseUrl = 'http://localhost:8002/api/multi-agent') {
    this.baseUrl = baseUrl
  }

  private async makeRequest<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      })

      const data = await response.json()

      return {
        data: response.ok ? data : undefined,
        error: response.ok ? undefined : data.detail || 'Request failed',
        status: response.status,
      }
    } catch (error) {
      return {
        error: error instanceof Error ? error.message : 'Network error',
        status: 500,
      }
    }
  }

  // Collaborative Reasoning
  async collaborativeReasoning(request: CollaborativeReasoningRequest) {
    return this.makeRequest<any>('/reasoning/collaborative', {
      method: 'POST',
      body: JSON.stringify(request),
    })
  }

  async getReasoningStatistics() {
    return this.makeRequest<any>('/reasoning/statistics')
  }

  // Coordination Management
  async coordinateAgents(request: CoordinationRequest) {
    return this.makeRequest<any>('/coordination/coordinate', {
      method: 'POST',
      body: JSON.stringify(request),
    })
  }

  async rebalanceAgents(request: { agents: Agent[] }) {
    return this.makeRequest<any>('/coordination/rebalance', {
      method: 'POST',
      body: JSON.stringify(request),
    })
  }

  async getCoordinationStatistics() {
    return this.makeRequest<any>('/coordination/statistics')
  }

  // Behavior Monitoring
  async monitorBehavior(request: BehaviorMonitoringRequest) {
    return this.makeRequest<any>('/behavior/monitor', {
      method: 'POST',
      body: JSON.stringify(request),
    })
  }

  async getBehaviorStatistics() {
    return this.makeRequest<any>('/behavior/statistics')
  }

  // Multi-Agent Optimization
  async optimizeAgents(request: OptimizationRequest) {
    return this.makeRequest<any>('/optimization/optimize', {
      method: 'POST',
      body: JSON.stringify(request),
    })
  }

  async getOptimizationStatistics() {
    return this.makeRequest<any>('/optimization/statistics')
  }

  // Health Check
  async healthCheck() {
    return this.makeRequest<any>('/health')
  }

  // Real-time behavior monitoring WebSocket
  connectRealtimeMonitoring(onMessage: (data: any) => void) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/api/multi-agent/behavior/monitor/realtime`
    
    const ws = new WebSocket(wsUrl)
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        onMessage(data)
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error)
      }
    }
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }
    
    return ws
  }
}

export const multiAgentClient = new MultiAgentApiClient()

export type {
  Agent,
  CollaborativeReasoningRequest,
  CoordinationRequest,
  BehaviorMonitoringRequest,
  OptimizationRequest,
  ApiResponse,
}
