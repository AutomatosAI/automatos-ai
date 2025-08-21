/* eslint-disable @typescript-eslint/no-explicit-any */

let BASE = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000').replace(/\/$/, '');
let PREFIX = (process.env.NEXT_PUBLIC_API_PREFIX || '/api').replace(/\/$/, '');
let API_KEY = process.env.NEXT_PUBLIC_API_KEY || '';
let AUTH = process.env.NEXT_PUBLIC_AUTH_TOKEN || '';
let TENANT_ID = process.env.NEXT_PUBLIC_TENANT_ID || '';

function readRuntimeOverrides() {
  if (process.env.NODE_ENV === 'production') return;
  if (typeof window === 'undefined') return;
  try {
    const overrides = JSON.parse(localStorage.getItem('automatos.api.overrides') || '{}');
    if (overrides.base) BASE = String(overrides.base).replace(/\/$/, '');
    if (overrides.prefix) PREFIX = String(overrides.prefix).replace(/\/$/, '');
    if (overrides.apiKey != null) API_KEY = String(overrides.apiKey);
    if (overrides.authToken != null) AUTH = String(overrides.authToken);
    if (overrides.tenantId != null) TENANT_ID = String(overrides.tenantId);
  } catch {}
}

function buildUrl(path: string): string {
  readRuntimeOverrides();
  if (path.startsWith('http://') || path.startsWith('https://')) return path;
  if (!path.startsWith('/')) path = '/' + path;
  // Allow absolute root paths like '/health' without prefix
  const isRootHealth = path === '/health';
  const full = isRootHealth ? `${BASE}${path}` : `${BASE}${PREFIX}${path}`;
  return full;
}

async function http<T = any>(path: string, init?: RequestInit): Promise<T> {
  readRuntimeOverrides();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(init?.headers as any || {}),
  };
  if (API_KEY && !headers['X-API-Key']) headers['X-API-Key'] = API_KEY;
  if (AUTH && !headers['Authorization']) headers['Authorization'] = `Bearer ${AUTH}`;
  if (TENANT_ID && !headers['X-Tenant-ID']) headers['X-Tenant-ID'] = TENANT_ID;

  const res = await fetch(buildUrl(path), { ...init, headers, cache: 'no-store' });
  const ct = res.headers.get('content-type') || '';
  const text = await res.text();
  let data: any = null;
  if (text) {
    try { data = JSON.parse(text); } catch { data = { __raw: text, __contentType: ct }; }
  }
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} :: ${path} :: ${typeof data === 'string' ? data : JSON.stringify(data).slice(0, 400)}`);
  return (data ?? null) as T;
}

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

export type Agent = any;
export type Document = any;
export type Workflow = any;

export const apiClient = {
  // System
  async getSystemHealth(): Promise<SystemHealth> {
    try { return await http<SystemHealth>('/system/health'); } catch { return http<SystemHealth>('/health'); }
  },
  async getSystemMetrics(): Promise<SystemMetrics> {
    return http<SystemMetrics>('/system/metrics');
  },

  // Agents
  async getAgents(params?: { q?: string; limit?: number; offset?: number; tenant_id?: string }): Promise<Agent[]> {
    const qs = new URLSearchParams();
    if (params?.q) qs.set('q', params.q);
    if (params?.limit != null) qs.set('limit', String(params.limit));
    if (params?.offset != null) qs.set('offset', String(params.offset));
    if (params?.tenant_id) qs.set('tenant_id', params.tenant_id);
    const url = `/agents${qs.toString() ? `?${qs.toString()}` : ''}`;
    // Some backends return { items, total }. Normalize to array.
    const data: any = await http(url);
    if (Array.isArray(data)) return data;
    if (data?.items) return data.items;
    return [];
  },
  async getAgent(id: string | number): Promise<Agent | null> {
    try { return await http<Agent>(`/agents/${encodeURIComponent(String(id))}`); } catch { return null; }
  },
  async getAgentRuns(agent_id: string | number, limit = 50): Promise<{ items: any[]; total?: number }> {
    const qs = new URLSearchParams({ agent_id: String(agent_id), limit: String(limit) }).toString();
    try { return await http<{ items: any[]; total?: number }>(`/runs?${qs}`); } catch { return { items: [] }; }
  },
  async createAgent(payload: any): Promise<Agent> {
    return http<Agent>(`/agents`, { method: 'POST', body: JSON.stringify(payload) });
  },
  async updateAgent(id: string | number, payload: any): Promise<Agent> {
    return http<Agent>(`/agents/${encodeURIComponent(String(id))}`, { method: 'PUT', body: JSON.stringify(payload) });
  },
  async deleteAgent(id: string | number): Promise<boolean> {
    await http<void>(`/agents/${encodeURIComponent(String(id))}`, { method: 'DELETE' });
    return true;
  },

  // Documents
  async getDocuments(params?: { limit?: number; offset?: number; q?: string }): Promise<Document[]> {
    const qs = new URLSearchParams();
    if (params?.limit != null) qs.set('limit', String(params.limit));
    if (params?.offset != null) qs.set('offset', String(params.offset));
    if (params?.q) qs.set('q', params.q);
    const url = `/documents${qs.toString() ? `?${qs.toString()}` : ''}`;
    const data: any = await http(url);
    if (Array.isArray(data)) return data;
    if (data?.items) return data.items;
    return [];
  },

  // Workflows
  async getWorkflows(params?: { q?: string; limit?: number; offset?: number; tenant_id?: string; owner?: string; tag?: string }): Promise<{ items: Workflow[]; total: number }>{
    const qs = new URLSearchParams();
    if (params?.q) qs.set('q', params.q);
    if (params?.limit != null) qs.set('limit', String(params.limit));
    if (params?.offset != null) qs.set('offset', String(params.offset));
    if (params?.tenant_id) qs.set('tenant_id', params.tenant_id);
    if (params?.owner) qs.set('owner', params.owner);
    if (params?.tag) qs.set('tag', params.tag);
    const url = `/workflows${qs.toString() ? `?${qs.toString()}` : ''}`;
    const data: any = await http(url);
    if (Array.isArray(data)) return { items: data, total: data.length };
    if (data?.items != null && data?.total != null) return data as { items: Workflow[]; total: number };
    return { items: [], total: 0 };
  },
  async getWorkflow(id: string | number): Promise<Workflow | null> {
    try { return await http<Workflow>(`/workflows/${encodeURIComponent(String(id))}`); } catch { return null }
  },
  async createWorkflow(body: Partial<Workflow>): Promise<Workflow> {
    return http<Workflow>(`/workflows`, { method: 'POST', body: JSON.stringify(body) });
  },
  async updateWorkflow(id: string | number, body: Partial<Workflow>): Promise<Workflow> {
    return http<Workflow>(`/workflows/${encodeURIComponent(String(id))}`, { method: 'PUT', body: JSON.stringify(body) });
  },
  async runWorkflow(id: string | number, input: any): Promise<{ run_id: string }> {
    return http<{ run_id: string }>(`/workflows/${encodeURIComponent(String(id))}/run`, { method: 'POST', body: JSON.stringify({ input }) });
  },
  async deleteWorkflow(id: string | number): Promise<boolean> {
    await http<void>(`/workflows/${encodeURIComponent(String(id))}`, { method: 'DELETE' });
    return true;
  },

  // Code Graph
  async codegraphIndex(payload: { project: string; root_dir: string }): Promise<any> {
    return http(`/codegraph/index`, { method: 'POST', body: JSON.stringify(payload) });
  },
  async codegraphSearch(params: { project: string; q: string; limit?: number }): Promise<{ prompt_block: string; count: number }>{
    const qs = new URLSearchParams({ project: params.project, q: params.q });
    if (params.limit != null) qs.set('limit', String(params.limit));
    return http(`/codegraph/search?${qs.toString()}`);
  },

  // Playbooks
  async listPlaybooks(params?: { tenant_id?: string }): Promise<{ items: any[] }>{
    const qs = new URLSearchParams();
    const effectiveTenant = params?.tenant_id || TENANT_ID || '';
    if (effectiveTenant) qs.set('tenant_id', effectiveTenant);
    const url = `/playbooks${qs.toString() ? `?${qs.toString()}` : ''}`;
    return http(url);
  },
  async minePlaybooks(body: { tenant_id?: string; min_support?: number; top_k?: number; name_prefix?: string }): Promise<{ generated: any[] }>{
    return http(`/playbooks/mine`, { method: 'POST', body: JSON.stringify(body || {}) });
  },
};

export default apiClient;

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://api.automatos.app';

export interface SystemHealth {
  status: string;
  timestamp: string;
  version: string;
}

export interface SystemMetrics {
  cpu: {
    usage_percent: number;
    cores: number;
  };
  memory: {
    usage_percent: number;
    used_gb: number;
    total_gb: number;
  };
  disk: {
    usage_percent: number;
    used_gb: number;
    total_gb: number;
  };
  network: {
    packets_sent: number;
    packets_recv: number;
    bytes_sent: number;
    bytes_recv: number;
  };
  timestamp: number;
}

      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${url}`, error);
      throw error;
    }
  }

  async getSystemHealth(): Promise<SystemHealth> {
    return this.request<SystemHealth>('/health');
  }

  async getSystemMetrics(): Promise<SystemMetrics> {
    return this.request<SystemMetrics>('/api/system/metrics');
  }
}

const apiClient = new ApiClient();
export default apiClient;
