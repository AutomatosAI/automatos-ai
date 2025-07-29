
/**
 * API Client for Automotas AI Backend
 * ==================================
 * 
 * Centralized API client with TypeScript support and error handling.
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8002';
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8002/ws';

// Types
export interface Agent {
  id: number;
  name: string;
  description?: string;
  agent_type: 'code_architect' | 'security_expert' | 'performance_optimizer' | 'data_analyst' | 'infrastructure_manager' | 'custom';
  status: 'active' | 'inactive' | 'training';
  configuration?: Record<string, any>;
  performance_metrics?: Record<string, any>;
  created_at: string;
  updated_at: string;
  created_by?: string;
  skills: Skill[];
}

export interface Skill {
  id: number;
  name: string;
  description?: string;
  skill_type: 'analytical' | 'cognitive' | 'technical' | 'communication' | 'operational';
  implementation?: string;
  parameters?: Record<string, any>;
  performance_data?: Record<string, any>;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  created_by?: string;
}

export interface Pattern {
  id: number;
  name: string;
  description?: string;
  pattern_type: string;
  pattern_data: Record<string, any>;
  usage_count: number;
  effectiveness_score: number;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  created_by?: string;
}

export interface Workflow {
  id: number;
  name: string;
  description?: string;
  workflow_definition: Record<string, any>;
  status: 'draft' | 'active' | 'archived';
  created_at: string;
  updated_at: string;
  created_by?: string;
  agents: Agent[];
}

export interface WorkflowExecution {
  id: number;
  workflow_id: number;
  agent_id: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  input_data?: Record<string, any>;
  output_data?: Record<string, any>;
  execution_log?: string;
  started_at: string;
  completed_at?: string;
  error_message?: string;
}

export interface Document {
  id: number;
  filename: string;
  original_filename?: string;
  file_type?: string;
  file_size?: number;
  status: string;
  chunk_count: number;
  tags?: string[];
  description?: string;
  upload_date: string;
  processed_date?: string;
  created_by?: string;
}

export interface SystemConfig {
  id: number;
  config_key: string;
  config_value: Record<string, any>;
  description?: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  updated_by?: string;
}

export interface RAGConfig {
  id: number;
  name: string;
  embedding_model?: string;
  chunk_size: number;
  chunk_overlap: number;
  retrieval_strategy: string;
  top_k: number;
  similarity_threshold: number;
  configuration?: Record<string, any>;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  created_by?: string;
}

export interface SystemHealth {
  status: string;
  timestamp: string;
  services: Record<string, string>;
  metrics: Record<string, any>;
  version: string;
}

// API Client Class
class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${url}`, error);
      throw error;
    }
  }

  // Agent API methods
  async getAgents(params?: {
    skip?: number;
    limit?: number;
    status?: string;
    agent_type?: string;
    search?: string;
  }): Promise<Agent[]> {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          searchParams.append(key, value.toString());
        }
      });
    }
    
    return this.request<Agent[]>(`/api/agents/${searchParams.toString() ? '?' + searchParams.toString() : ''}`);
  }

  async getAgentTypes(): Promise<string[]> {
    try {
      // Try the direct endpoint first
      return this.request<string[]>('/api/agents/types');
    } catch (error) {
      // Fallback: extract unique agent types from existing agents
      console.warn('Agent types endpoint failed, using fallback method:', error);
      try {
        const agents = await this.getAgents();
        const uniqueTypes = [...new Set(agents.map(agent => agent.agent_type))];
        return uniqueTypes.filter(type => type) as string[];
      } catch (fallbackError) {
        console.error('Fallback method also failed:', fallbackError);
        // Return default agent types
        return ['code_architect', 'security_expert', 'performance_optimizer', 'data_analyst', 'infrastructure_manager', 'custom'];
      }
    }
  }

  async getAgent(id: number): Promise<Agent> {
    return this.request<Agent>(`/api/agents/${id}/`);
  }

  async createAgent(data: {
    name: string;
    description?: string;
    agent_type: 'code_architect' | 'security_expert' | 'performance_optimizer' | 'data_analyst' | 'infrastructure_manager' | 'custom';
    configuration?: Record<string, any>;
    skill_ids?: number[];
  }): Promise<Agent> {
    return this.request<Agent>('/api/agents/', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateAgent(id: number, data: Partial<Agent>): Promise<Agent> {
    return this.request<Agent>(`/api/agents/${id}/`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteAgent(id: number): Promise<{ message: string }> {
    return this.request<{ message: string }>(`/api/agents/${id}/`, {
      method: 'DELETE',
    });
  }

  // Skill API methods
  async getSkills(params?: {
    skip?: number;
    limit?: number;
    skill_type?: string;
    search?: string;
    active_only?: boolean;
  }): Promise<Skill[]> {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          searchParams.append(key, value.toString());
        }
      });
    }
    
    return this.request<Skill[]>(`/api/agents/skills/?${searchParams}`);
  }

  async getSkill(id: number): Promise<Skill> {
    return this.request<Skill>(`/api/agents/skills/${id}/`);
  }

  async createSkill(data: {
    name: string;
    description?: string;
    skill_type: 'analytical' | 'cognitive' | 'technical' | 'communication' | 'operational';
    implementation?: string;
    parameters?: Record<string, any>;
  }): Promise<Skill> {
    return this.request<Skill>('/api/agents/skills/', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateSkill(id: number, data: Partial<Skill>): Promise<Skill> {
    return this.request<Skill>(`/api/agents/skills/${id}/`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  // Pattern API methods
  async getPatterns(params?: {
    skip?: number;
    limit?: number;
    pattern_type?: string;
    search?: string;
    active_only?: boolean;
  }): Promise<Pattern[]> {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          searchParams.append(key, value.toString());
        }
      });
    }
    
    return this.request<Pattern[]>(`/api/agents/patterns?${searchParams}`);
  }

  async createPattern(data: {
    name: string;
    description?: string;
    pattern_type: string;
    pattern_data: Record<string, any>;
  }): Promise<Pattern> {
    return this.request<Pattern>('/api/agents/patterns', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // Workflow API methods
  async getWorkflows(params?: {
    skip?: number;
    limit?: number;
    status?: string;
    search?: string;
  }): Promise<Workflow[]> {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          searchParams.append(key, value.toString());
        }
      });
    }
    
    return this.request<Workflow[]>(`/api/workflows/?${searchParams}`);
  }

  async getWorkflow(id: number): Promise<Workflow> {
    return this.request<Workflow>(`/api/workflows/${id}/`);
  }

  async createWorkflow(data: {
    name: string;
    description?: string;
    workflow_definition: Record<string, any>;
    agent_ids?: number[];
  }): Promise<Workflow> {
    return this.request<Workflow>('/api/workflows/', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateWorkflow(id: number, data: Partial<Workflow>): Promise<Workflow> {
    return this.request<Workflow>(`/api/workflows/${id}/`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteWorkflow(id: number): Promise<{ message: string }> {
    return this.request<{ message: string }>(`/api/workflows/${id}/`, {
      method: 'DELETE',
    });
  }

  async executeWorkflow(workflowId: number, data: {
    agent_id: number;
    input_data?: Record<string, any>;
  }): Promise<WorkflowExecution> {
    return this.request<WorkflowExecution>(`/api/workflows/${workflowId}/execute`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getWorkflowExecutions(workflowId: number, params?: {
    skip?: number;
    limit?: number;
    status?: string;
  }): Promise<WorkflowExecution[]> {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          searchParams.append(key, value.toString());
        }
      });
    }
    
    return this.request<WorkflowExecution[]>(`/api/workflows/${workflowId}/executions?${searchParams}`);
  }

  async getWorkflowExecution(executionId: number): Promise<WorkflowExecution> {
    return this.request<WorkflowExecution>(`/api/workflows/executions/${executionId}`);
  }

  // Document API methods
  async getDocuments(params?: {
    skip?: number;
    limit?: number;
    status?: string;
    file_type?: string;
    search?: string;
  }): Promise<Document[]> {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          searchParams.append(key, value.toString());
        }
      });
    }
    
    return this.request<Document[]>(`/api/documents/?${searchParams}`);
  }

  async getDocument(id: number): Promise<Document> {
    return this.request<Document>(`/api/documents/${id}/`);
  }

  async uploadDocument(file: File, description?: string, tags?: string): Promise<{
    document_id: number;
    filename: string;
    status: string;
    message: string;
  }> {
    const formData = new FormData();
    formData.append('file', file);
    if (description) formData.append('description', description);
    if (tags) formData.append('tags', tags);

    const response = await fetch(`${this.baseUrl}/api/documents/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  }

  async deleteDocument(id: number): Promise<{ message: string }> {
    return this.request<{ message: string }>(`/api/documents/${id}/`, {
      method: 'DELETE',
    });
  }

  async reprocessDocument(id: number): Promise<{ message: string }> {
    return this.request<{ message: string }>(`/api/documents/${id}/reprocess/`, {
      method: 'POST',
    });
  }

  async getDocumentContent(id: number): Promise<{
    document_id: number;
    filename: string;
    chunk_count: number;
    chunks: any[];
  }> {
    return this.request(`/api/documents/${id}/content/`);
  }

  // System API methods
  async getSystemConfigs(params?: {
    skip?: number;
    limit?: number;
    search?: string;
    active_only?: boolean;
  }): Promise<SystemConfig[]> {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          searchParams.append(key, value.toString());
        }
      });
    }
    
    return this.request<SystemConfig[]>(`/api/system/config?${searchParams}`);
  }

  async getSystemConfig(key: string): Promise<SystemConfig> {
    return this.request<SystemConfig>(`/api/system/config/${key}`);
  }

  async createSystemConfig(data: {
    config_key: string;
    config_value: Record<string, any>;
    description?: string;
  }): Promise<SystemConfig> {
    return this.request<SystemConfig>('/api/system/config', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateSystemConfig(key: string, data: {
    config_key: string;
    config_value: Record<string, any>;
    description?: string;
  }): Promise<SystemConfig> {
    return this.request<SystemConfig>(`/api/system/config/${key}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  // RAG API methods
  async getRAGConfigs(params?: {
    skip?: number;
    limit?: number;
    active_only?: boolean;
  }): Promise<RAGConfig[]> {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          searchParams.append(key, value.toString());
        }
      });
    }
    
    return this.request<RAGConfig[]>(`/api/system/rag?${searchParams}`);
  }

  async getRAGConfig(id: number): Promise<RAGConfig> {
    return this.request<RAGConfig>(`/api/system/rag/${id}`);
  }

  async createRAGConfig(data: {
    name: string;
    embedding_model?: string;
    chunk_size?: number;
    chunk_overlap?: number;
    retrieval_strategy?: string;
    top_k?: number;
    similarity_threshold?: number;
    configuration?: Record<string, any>;
  }): Promise<RAGConfig> {
    return this.request<RAGConfig>('/api/system/rag', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async testRAGConfig(id: number, query: string): Promise<{
    config_id: number;
    query: string;
    results: any[];
    retrieval_time: string;
    total_results: number;
  }> {
    return this.request(`/api/context/rag/${id}/test?query=${encodeURIComponent(query)}`, {
      method: 'POST',
    });
  }

  // System Health
  async getSystemHealth(): Promise<SystemHealth> {
    return this.request<SystemHealth>('/api/system/health');
  }

  async getSystemMetrics(): Promise<Record<string, any>> {
    return this.request('/api/system/metrics');
  }
}

// Create and export API client instance
export const apiClient = new ApiClient();

// WebSocket connection manager
export class WebSocketManager {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private listeners: Map<string, Set<(data: any) => void>> = new Map();

  connect(clientId?: string): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        const url = clientId ? `${WS_URL}?client_id=${clientId}` : WS_URL;
        this.ws = new WebSocket(url);

        this.ws.onopen = () => {
          console.log('WebSocket connected');
          this.reconnectAttempts = 0;
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };

        this.ws.onclose = () => {
          console.log('WebSocket disconnected');
          this.attemptReconnect();
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          reject(error);
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  private handleMessage(message: any) {
    const { type, data } = message;
    const typeListeners = this.listeners.get(type);
    if (typeListeners) {
      typeListeners.forEach(listener => listener(data));
    }

    // Also notify wildcard listeners
    const wildcardListeners = this.listeners.get('*');
    if (wildcardListeners) {
      wildcardListeners.forEach(listener => listener(message));
    }
  }

  private attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        this.connect();
      }, this.reconnectDelay * this.reconnectAttempts);
    }
  }

  subscribe(eventType: string, callback: (data: any) => void) {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, new Set());
    }
    this.listeners.get(eventType)!.add(callback);

    // Return unsubscribe function
    return () => {
      const typeListeners = this.listeners.get(eventType);
      if (typeListeners) {
        typeListeners.delete(callback);
        if (typeListeners.size === 0) {
          this.listeners.delete(eventType);
        }
      }
    };
  }

  send(message: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected');
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.listeners.clear();
  }
}

// Create and export WebSocket manager instance
export const wsManager = new WebSocketManager();

// Export default API client
export default apiClient;
