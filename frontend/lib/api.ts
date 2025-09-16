
/* eslint-disable @typescript-eslint/no-explicit-any */

// API Configuration - FIXED FOR PRODUCTION
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'https://api.automatos.app'
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
    // Clean baseURL construction - FIXED
    this.baseURL = API_URL.endsWith('/api') 
      ? API_URL 
      : `${API_URL.replace(/\/+$/, '')}${API_PREFIX}`;
    
    this.headers = {
      'Content-Type': 'application/json',
    };
    
    // Debug log for troubleshooting
    if (typeof window !== 'undefined' && window.console) {
      console.log(`[API Client] baseURL: ${this.baseURL}`);
    }
  }

  // UNIFIED request method - handles ALL API calls consistently
  public async request<T = any>(endpoint: string, options: RequestInit = {}): Promise<T> {
    let url: string;
    
    // Handle different endpoint formats
    if (endpoint.startsWith('http://') || endpoint.startsWith('https://')) {
      // Absolute URL - use as is
      url = endpoint;
    } else if (endpoint.startsWith('/api/')) {
      // Already has /api prefix - construct with base URL
      url = `${API_URL.replace(/\/+$/, '')}${endpoint}`;
    } else if (endpoint.startsWith('/')) {
      // Relative path - add to baseURL
      url = `${this.baseURL}${endpoint}`;
    } else {
      // No leading slash - add slash and append
      url = `${this.baseURL}/${endpoint}`;
    }

    // Debug logging
    if (typeof window !== 'undefined' && window.console) {
      console.log(`[API] ${endpoint} -> ${url}`);
    }

    const config: RequestInit = {
      ...options,
      headers: {
        ...this.headers,
        ...options.headers,
      },
    };

    try {
      const res = await fetch(url, config);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }
      return await res.json();
    } catch (error) {
      console.error(`[API Error] ${endpoint}:`, error);
      throw error;
    }
  }

  // Documents - FIXED to use unified request method
  async getDocuments(params?: { limit?: number; offset?: number; q?: string }): Promise<Document[]> {
    const qs = new URLSearchParams();
    if (params?.limit != null) qs.set('limit', String(params.limit));
    if (params?.offset != null) qs.set('offset', String(params.offset));
    if (params?.q) qs.set('q', params.q);
    const url = '/documents' + (qs.toString() ? '?' + qs.toString() : '');
    
    try {
      const data: any = await this.request(url);
      if (Array.isArray(data)) return data;
      if (data?.items) return data.items;
      return [];
    } catch (error) {
      console.error('Error fetching documents:', error);
      return []; // Return empty array instead of throwing
    }
  }

  async uploadDocument(file: File, metadata?: { description?: string; tags?: string | string[] }): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    if (metadata?.description) formData.append('description', String(metadata.description));
    if (metadata?.tags) {
      const tagsValue = Array.isArray(metadata.tags) ? metadata.tags.join(',') : metadata.tags;
      formData.append('tags', tagsValue);
    }

    // Use direct fetch for FormData uploads
    const url = `${this.baseURL}/documents/upload`;
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
      // Don't set Content-Type - let browser set it for FormData
    });
    
    if (!response.ok) {
      throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
    }
    return response.json();
  }

  async deleteDocument(id: string | number): Promise<boolean> {
    await this.request(`/documents/${encodeURIComponent(String(id))}`, { method: 'DELETE' });
    return true;
  }

  async reprocessDocument(id: string | number): Promise<any> {
    return this.request(`/documents/${encodeURIComponent(String(id))}/reprocess`, { method: 'POST' });
  }

  async getDocumentContent(id: string | number): Promise<any> {
    return this.request(`/documents/${encodeURIComponent(String(id))}/content`);
  }

  // Processing methods with fallbacks
  async getProcessingPipeline(): Promise<any> {
    try {
      return await this.request('/documents/processing/pipeline');
    } catch (error) {
      console.warn('Processing pipeline not available:', error);
      return { stages: [], status: 'unavailable' };
    }
  }

  async getProcessingLiveStatus(): Promise<any> {
    try {
      return await this.request('/documents/processing/live-status');
    } catch (error) {
      console.warn('Live status not available:', error);
      return { active_jobs: [] };
    }
  }

  async reprocessAllDocuments(): Promise<any> {
    try {
      return await this.request('/documents/processing/reprocess-all', { method: 'POST' });
    } catch (error) {
      console.warn('Reprocess all not available:', error);
      return { message: 'Feature not available' };
    }
  }

  async getAnalyticsOverview(): Promise<any> {
    try {
      return await this.request('/documents/analytics/overview');
    } catch (error) {
      console.warn('Analytics not available:', error);
      return { totalDocuments: 0, processedDocuments: 0, failedDocuments: 0 };
    }
  }

  async getSearchPatterns(): Promise<any> {
    try {
      return await this.request('/documents/analytics/search-patterns');
    } catch (error) {
      console.warn('Search patterns not available:', error);
      return { patterns: [] };
    }
  }

  // Agents
  async getAgents(): Promise<Agent[]> {
    return this.request('/agents');
  }

  async getAgent(id: string | number): Promise<Agent> {
    return this.request(`/agents/${encodeURIComponent(String(id))}`);
  }

  async getAgentRuns(id: string | number, limit: number = 50): Promise<any> {
    return this.request(`/agents/${encodeURIComponent(String(id))}/runs?limit=${limit}`);
  }

  async createAgent(payload: any): Promise<any> {
    return this.request('/agents', {
      method: 'POST',
      body: JSON.stringify(payload)
    });
  }

  async updateAgent(agentId: string | number, body: any): Promise<any> {
    return this.request(`/agents/${encodeURIComponent(String(agentId))}`, {
      method: 'PUT',
      body: JSON.stringify(body),
    });
  }

  // Workflows
  async getWorkflows(params?: { limit?: number; offset?: number; q?: string }): Promise<Workflow[]> {
    const qs = new URLSearchParams();
    if (params?.limit != null) qs.set('limit', String(params.limit));
    if (params?.offset != null) qs.set('offset', String(params.offset));
    if (params?.q) qs.set('q', params.q);
    const url = '/workflows' + (qs.toString() ? `?${qs.toString()}` : '');
    
    try {
      const data: any = await this.request(url);
      if (Array.isArray(data)) return data;
      if (data?.items) return data.items;
      return [];
    } catch (error) {
      console.error('Error fetching workflows:', error);
      return [];
    }
  }

  // System
  async getSystemHealth(): Promise<SystemHealth> {
    return this.request('/system/health');
  }

  async getSystemMetrics(): Promise<SystemMetrics> {
    return this.request('/system/metrics');
  }

  async saveSystemConfig(configKey: string, configValue: any, description?: string): Promise<any> {
    return this.request('/system/config', {
      method: 'POST',
      body: JSON.stringify({ config_key: configKey, config_value: configValue, description }),
    });
  }
}

export const apiClient = new ApiClient();
export default apiClient;
