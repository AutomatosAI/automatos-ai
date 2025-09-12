/* eslint-disable @typescript-eslint/no-explicit-any */

// API Configuration
const API_URL = process.env.NEXT_PUBLIC_API_URL || '/api'
const API_PREFIX = process.env.NEXT_PUBLIC_API_PREFIX || '/api'

// Types
export type Document = any;
export type Workflow = any;
export type Agent = any;

export interface SystemHealth {
  status: string;
  timestamp?: string;
  version?: string;
}

export interface SystemMetrics {
  cpu: { usage_percent?: number; average_usage?: number; cores?: number } | any;
  memory: { usage_percent?: number; percent?: number; used_gb?: number; total_gb?: number } | any;
  disk: { usage_percent?: number; percent?: number; used_gb?: number; total_gb?: number } | any;
  network?: { packets_sent?: number; packets_recv?: number; bytes_sent?: number; bytes_recv?: number };
  timestamp?: number | string;
}

class ApiClient {
  private baseURL: string;
  private headers: Record<string, string>;

  constructor() {
    // Handle case where API_URL already contains the API path
    if (API_URL.endsWith('/api') || API_URL.endsWith('/api/')) {
      this.baseURL = API_URL.replace(/\/api\/?$/, '') + API_PREFIX;
    } else {
      this.baseURL = API_URL + API_PREFIX;
    }
    this.headers = {
      'Content-Type': 'application/json',
    };
  }

  // Public flexible request helper that accepts absolute URLs or API-relative paths
  public async request<T = any>(endpoint: string, options: RequestInit = {}): Promise<T> {
    let url = endpoint
    if (!/^https?:\/\//i.test(endpoint)) {
      // Normalize: strip leading /api to avoid duplicate when baseURL already includes it
      const normalized = endpoint.startsWith('/api/') ? endpoint.replace(/^\/api/, '') : endpoint
      url = this.baseURL + (normalized.startsWith('/') ? normalized : `/${normalized}`)
    }

    // Debug logging (will be visible in browser console)
    if (typeof window !== 'undefined' && window.console) {
      console.log(`[API] ${endpoint} -> ${url}`)
    }

    const config: RequestInit = {
      ...options,
      headers: {
        ...this.headers,
        ...options.headers,
      },
    }

    const res = await fetch(url, config)
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
    return res.json()
  }

  private async http<T = any>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = this.baseURL + endpoint;
    const config: RequestInit = {
      ...options,
      headers: {
        ...this.headers,
        ...options.headers,
      },
    };

    const response = await fetch(url, config);
    if (response.ok === false) {
      throw new Error('HTTP error status: ' + response.status);
    }
    return response.json();
  }

  // Documents
  async getDocuments(params?: { limit?: number; offset?: number; q?: string }): Promise<Document[]> {
    const qs = new URLSearchParams();
    if (params?.limit != null) qs.set('limit', String(params.limit));
    if (params?.offset != null) qs.set('offset', String(params.offset));
    if (params?.q) qs.set('q', params.q);
    const url = '/documents' + (qs.toString() ? '?' + qs.toString() : '');
    const data: any = await this.http(url);
    if (Array.isArray(data)) return data;
    if (data?.items) return data.items;
    return [];
  }

  async uploadDocument(file: File, metadata?: { description?: string; tags?: string | string[] }): Promise<any> {
    const formData = new FormData()
    formData.append('file', file)
    if (metadata?.description) formData.append('description', String(metadata.description))
    if (metadata?.tags) {
      const tagsValue = Array.isArray(metadata.tags) ? metadata.tags.join(',') : metadata.tags
      formData.append('tags', tagsValue)
    }

    // Use the request method to handle URL properly
    const response = await fetch(this.baseURL + '/api/documents/upload', {
      method: 'POST',
      body: formData,
      // Don't set Content-Type - let browser set it for FormData
    })
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`)
    return response.json()
  }

  async deleteDocument(id: string | number): Promise<boolean> {
    await this.http('/documents/' + encodeURIComponent(String(id)), { method: 'DELETE' });
    return true;
  }

  async reprocessDocument(id: string | number): Promise<any> {
    return this.http('/documents/' + encodeURIComponent(String(id)) + '/reprocess', { method: 'POST' });
  }

  async getDocumentContent(id: string | number): Promise<any> {
    return this.http('/documents/' + encodeURIComponent(String(id)) + '/content');
  }

  // Processing methods with fallbacks
  async getProcessingPipeline(): Promise<any> {
    try {
      return await this.http('/documents/processing/pipeline');
    } catch (error) {
      return { stages: [], status: 'unavailable' };
    }
  }

  async getProcessingLiveStatus(): Promise<any> {
    try {
      return await this.http('/documents/processing/live-status');
    } catch (error) {
      return { active_jobs: [] };
    }
  }

  async reprocessAllDocuments(): Promise<any> {
    try {
      return await this.http('/documents/processing/reprocess-all', { method: 'POST' });
    } catch (error) {
      return { message: 'Feature not available' };
    }
  }

  async getAnalyticsOverview(): Promise<any> {
    try {
      return await this.http('/documents/analytics/overview');
    } catch (error) {
      return { totalDocuments: 0, processedDocuments: 0, failedDocuments: 0 };
    }
  }

  async getSearchPatterns(): Promise<any> {
    try {
      return await this.http('/documents/analytics/search-patterns');
    } catch (error) {
      return { patterns: [] };
    }
  }

  // Agents
  async getAgents(): Promise<Agent[]> {
    return this.http('/agents')
  }

  async getAgent(id: string | number): Promise<Agent> {
    return this.http(`/agents/${encodeURIComponent(String(id))}`)
  }

  async getAgentRuns(id: string | number, limit: number = 50): Promise<any> {
    return this.http(`/agents/${encodeURIComponent(String(id))}/runs?limit=${limit}`)
  }

  async createAgent(payload: any): Promise<any> {
    return this.http('/agents', {
      method: 'POST',
      body: JSON.stringify(payload)
    })
  }

  async updateAgent(agentId: string | number, body: any): Promise<any> {
    return this.http(`/agents/${encodeURIComponent(String(agentId))}`, {
      method: 'PUT',
      body: JSON.stringify(body),
    })
  }

  // Workflows
  async getWorkflows(params?: { limit?: number; offset?: number; q?: string }): Promise<Workflow[]> {
    const qs = new URLSearchParams()
    if (params?.limit != null) qs.set('limit', String(params.limit))
    if (params?.offset != null) qs.set('offset', String(params.offset))
    if (params?.q) qs.set('q', params.q)
    const url = '/workflows' + (qs.toString() ? `?${qs.toString()}` : '')
    const data: any = await this.http(url)
    if (Array.isArray(data)) return data
    if (data?.items) return data.items
    return [] as unknown as Workflow[]
  }

  // System
  async getSystemHealth(): Promise<SystemHealth> {
    return this.http('/system/health')
  }

  async getSystemMetrics(): Promise<SystemMetrics> {
    return this.http('/system/metrics')
  }

  async saveSystemConfig(configKey: string, configValue: any, description?: string): Promise<any> {
    return this.http('/system/config', {
      method: 'POST',
      body: JSON.stringify({ config_key: configKey, config_value: configValue, description }),
    })
  }
}

export const apiClient = new ApiClient();
export default apiClient;
