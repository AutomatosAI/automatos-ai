
/**
 * Field Theory Integration API Client
 * Connects frontend to backend field theory capabilities
 */

interface ApiResponse<T> {
  data?: T
  error?: string
  status: number
}

interface FieldUpdateRequest {
  session_id: string
  context: string
  field_type?: 'scalar' | 'vector' | 'tensor'
  influence_weights?: number[]
}

interface FieldPropagationRequest {
  session_id: string
  steps?: number
  alpha?: number
  beta?: number
}

interface FieldInteractionRequest {
  session_id_1: string
  session_id_2: string
  interaction_type?: 'similarity' | 'difference' | 'combination'
}

interface DynamicFieldRequest {
  session_id: string
  time_delta?: number
  learning_rate?: number
  momentum?: number
}

interface FieldOptimizationRequest {
  session_id: string
  objectives?: string[]
  constraints?: Record<string, any>
}

class FieldTheoryApiClient {
  private baseUrl: string

  constructor(baseUrl = 'http://localhost:8002/api/field-theory') {
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

  // Field Operations
  async updateField(request: FieldUpdateRequest) {
    return this.makeRequest<any>('/fields/update', {
      method: 'POST',
      body: JSON.stringify(request),
    })
  }

  async propagateField(request: FieldPropagationRequest) {
    return this.makeRequest<any>('/fields/propagate', {
      method: 'POST',
      body: JSON.stringify(request),
    })
  }

  async modelFieldInteractions(request: FieldInteractionRequest) {
    return this.makeRequest<any>('/fields/interactions', {
      method: 'POST',
      body: JSON.stringify(request),
    })
  }

  async dynamicFieldManagement(request: DynamicFieldRequest) {
    return this.makeRequest<any>('/fields/dynamic', {
      method: 'POST',
      body: JSON.stringify(request),
    })
  }

  async optimizeField(request: FieldOptimizationRequest) {
    return this.makeRequest<any>('/fields/optimize', {
      method: 'POST',
      body: JSON.stringify(request),
    })
  }

  // Field Data Retrieval
  async getFieldContext(sessionId: string) {
    return this.makeRequest<any>(`/fields/context/${sessionId}`)
  }

  async getFieldStatistics() {
    return this.makeRequest<any>('/fields/statistics')
  }

  async getFieldStates() {
    return this.makeRequest<any>('/fields/states')
  }

  async getFieldInteractions() {
    return this.makeRequest<any>('/fields/interactions', { method: 'GET' })
  }

  // Field Context Management
  async clearFieldContext(sessionId: string) {
    return this.makeRequest<any>(`/fields/context/${sessionId}`, {
      method: 'DELETE',
    })
  }

  // Batch Operations
  async batchUpdateFields(requests: FieldUpdateRequest[]) {
    return this.makeRequest<any>('/fields/batch/update', {
      method: 'POST',
      body: JSON.stringify({ requests }),
    })
  }

  async batchPropagateFields(requests: FieldPropagationRequest[]) {
    return this.makeRequest<any>('/fields/batch/propagate', {
      method: 'POST',
      body: JSON.stringify({ requests }),
    })
  }

  // Health Check
  async healthCheck() {
    return this.makeRequest<any>('/health')
  }
}

export const fieldTheoryClient = new FieldTheoryApiClient()

export type {
  FieldUpdateRequest,
  FieldPropagationRequest,
  FieldInteractionRequest,
  DynamicFieldRequest,
  FieldOptimizationRequest,
  ApiResponse,
}
