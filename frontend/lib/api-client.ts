
/**
 * API Client — one transport (Clerk auth + workspace header + error handling)
 * over the backend API. Real failures surface honestly: PRD-168 S3 removed the
 * mock fallback/data system; dev uses the real PRD-153 local stack.
 * Base URL: NEXT_PUBLIC_API_URL.
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

export interface RAGConfig {
  id?: number
  name: string
  embedding_model: string
  chunk_size: number
  chunk_overlap: number
  retrieval_strategy: string
  top_k: number
  similarity_threshold: number
  configuration?: any
  is_active?: boolean
}

// PRD-142 Wave 0 — "Is it working?" dashboard tiles. Shapes mirror the
// analytics_real.py endpoints exactly; the request() helper returns the
// parsed JSON body directly (no ApiResponse envelope on these routes).
export interface ActivationMetric {
  activated: number
  total_workspaces: number
  rate: number
  generated_at: string
}

export interface MissionSuccessRateMetric {
  value: number
  trend: number
  total_executions: number
  successful_executions: number
  sources?: { workflows: number; missions: number }
}

export interface ErrorsBySubsystemMetric {
  window: string
  total: number
  by_subsystem: Array<{ subsystem: string; count: number; rate: number }>
  generated_at: string
}

export interface WidgetEngagementMetric {
  window: string
  by_event_type: Array<{ event_type: string; count: number }>
  sessions: number
  generated_at: string
}

export interface PrimitiveHealthMetric {
  primitives: Array<{
    name: string
    status: 'green' | 'degraded' | 'down' | 'unknown'
    last_checked: string | null
  }>
  generated_at: string
}

// PRD-197 S4 — per-seam retrieval substrate health (workspace-admin reachable).
export interface SubstrateSeamHealth {
  seam: string
  searches: number
  error_rate: number
  empty_rate: number
  avg_latency_ms: number | null
  p95_latency_ms: number | null
  status: 'green' | 'degraded' | 'down' | 'unknown'
}

export interface SubstrateHealthMetric {
  generated_at: string
  window_seconds: number
  seams: SubstrateSeamHealth[]
}

// PRD-185 S12 — own-workspace cockpit tiles (workspace-admin reachable).
export interface SloItem {
  sli: string
  description: string
  value: number | null
  unit: string
  target: number
  target_comparator: string
  window_seconds: number
  sample_size: number
  meets_target: boolean | null
}

export interface SlosMetric {
  generated_at: string
  window_seconds: number
  slos: SloItem[]
}

export interface WorkspaceActivationMetric {
  activated: boolean
  completed_missions: number
  generated_at: string
}

export interface DeliverableFreshnessMetric {
  last_produced_at: string | null
  age_seconds: number | null
  total: number
  generated_at: string
}

// PRD-189 S2 — cross-sell persistence integrity (the Commerce tile).
export interface CommerceIntegrityMetric {
  synced: boolean
  reported_fbt_edges: number | null
  present_fbt_edges: number | null
  drift: number | null
  ok: boolean | null
  last_orders_sync_at: number | null
  last_catalog_sync_at: number | null
  generated_at: string
}

// ─── Admin workspace override ────────────────────────────────────────
// Module-level override that takes priority over localStorage.
// Set by AdminWorkspaceSwitcher; reset on unmount.
let _adminWorkspaceOverride: string | null = null

export function setAdminWorkspaceOverride(wsId: string | null) {
  _adminWorkspaceOverride = wsId
}

export function getAdminWorkspaceOverride(): string | null {
  return _adminWorkspaceOverride
}

// ─── PRD-196 governance surface types ────────────────────────────────
export interface ApprovalGrantOversight {
  risk_class: string
  tier: 'monitor' | 'human_on_the_loop' | 'human_in_the_loop' | string
  rationale: string
  requires_approval: boolean
}

// PRD-225: the cascade of downstream work parked behind a question.
export interface QuestionCascadeTask {
  id: number
  title: string
  status: string
}
export interface QuestionCascade {
  total: number
  tasks: QuestionCascadeTask[]
}

export interface ApprovalGrant {
  id: number
  workspace_id: string | null
  subject_type: string
  subject_id: string
  tool_name: string | null
  risk_tier: string | null
  agent_id: number | null
  status: string
  reason: string | null
  estimated_cost_usd: string | null
  requested_at: string | null
  expires_at: string | null
  granted_at: string | null
  granted_by: string | null
  revoked_at: string | null
  revoked_by: string | null
  oversight?: ApprovalGrantOversight
  /** PRD-193 S4: params snapshot + executed_result land here on resume. */
  details?: Record<string, unknown>
  // PRD-225: a grant is a question when its decision is words, not a boolean.
  kind?: 'approval' | 'question' | string
  question_md?: string | null
  options?: string[] | null
  answer_text?: string | null
  answered_by?: string | null
  answered_at?: string | null
  asked_by_agent_id?: number | null
  channel_refs?: Record<string, unknown>
  /** Downstream tasks blocked behind a question (question rows only). */
  cascade?: QuestionCascade
}

export interface ApprovalGrantsResponse {
  grants: ApprovalGrant[]
}

export interface GovernanceStatus {
  policy_plane: { enforcing: boolean }
  grants: { by_status: Record<string, number>; total: number }
  audit: {
    policy_verdicts: { total: number; by_action: Record<string, number> }
    window_days: number
  }
  retention: { retention_days: number | null; floor_days: number | null; configured: boolean }
}

export interface AuditLogRow {
  id: number
  created_at: string | null
  actor_type: string
  user_id: number | null
  action: string
  resource_type: string | null
  resource_id: string | null
  resource_name: string | null
  details: Record<string, any>
}

export interface AuditLogResponse {
  rows: AuditLogRow[]
  total: number
  limit: number
  offset: number
}

export interface AuditLogFilters {
  action_prefix?: string
  actor_type?: string
  resource_type?: string
  since?: string
  until?: string
  limit?: number
  offset?: number
}

export interface PolicyDocument {
  posture: 'balanced' | 'strict' | 'permissive' | string
  agents_inherit_admin: boolean
  route_overrides: Record<string, 'auto' | 'ask'>
}

export interface BudgetConfig {
  window?: 'day' | 'month' | 'all' | string
  max_cost_usd?: number
  max_total_tokens?: number
}

export interface GdprErasureGap {
  store: string
  reason: string
}

export interface GdprUntaggedHistory {
  stores: string[]
  reason: string
}

export interface GdprSubjectErasureResult {
  workspace_id: string
  subject_id: string
  erased_at: string
  sql: { deleted: number }
  derived: { field_memory_deleted: number; durable_memory_deleted: number }
  gaps: GdprErasureGap[]
  untagged_history: GdprUntaggedHistory
}

export interface GdprWorkspaceErasureResult {
  workspace_id: string
  complete?: boolean
  derived?: { field_memory_deleted: number; durable_memory_deleted: number }
  [key: string]: any
}

// ===== PRD-204 Auto Watcher: watchlist types =====
export interface WatchRow {
  id: string
  title: string
  watch_type: string
  target_type: string
  target_id: string
  status: string
  policy: string
  success_criteria: string | null
  quality_threshold: number
  final_score: number | null
  /** x10 display convention from the backend, e.g. "8.3/10" or "unscored". */
  final_score_display: string
  final_verdict: string | null
  actions_taken: number
  action_budget: number
  last_checked_at: string | null
  next_check_at: string | null
  deadline_at: string | null
  created_at: string | null
  closed_at: string | null
  lineage?: Array<Record<string, string>>
}

export interface WatchEventRow {
  event_type: string
  summary: string | null
  score: number | null
  action_taken: string | null
  requires_attention: boolean
  created_at: string | null
}

export interface WatchesResponse {
  watches: WatchRow[]
  total: number
}

// ===== PRD-228 Fleet State: live floor read-model =====
export interface FleetCurrentWork {
  kind: 'board_task' | 'mission_task'
  id: number | string
  title: string
  since: string | null
}

export interface FleetAgentRow {
  agent_id: number
  name: string
  current: FleetCurrentWork | null
  queue_depth: number
  blocked: { count: number; open_asks: Array<number | string> }
  watches: { active: number; needs_attention: number }
  last_activity_at: string | null
  // Omitted when the cost source is unavailable (fail-soft).
  cost_24h?: { tokens: number; usd: number }
}

export interface FleetStateResponse {
  version: number
  generated_at: string | null
  window_hours: number
  cost_available: boolean
  cost_source: string | null
  // Source-availability flags (mirror cost_available): false means the source
  // failed and its fields carry fail-soft defaults, not real values — so a
  // degraded source is distinguishable from a genuine zero (P228-RVW-6).
  watches_available: boolean
  asks_available: boolean
  agents: FleetAgentRow[]
}

export interface WatchDetailResponse {
  watch: WatchRow
  recent_events: WatchEventRow[]
}

// ===== PRD-233 S6 Local profile: who Auto is talking to =====
export interface ProfileResponse {
  edition: 'local' | 'saas'
  // true only in the local edition — saas profiles are managed by Clerk
  editable: boolean
  id: number | null
  email: string | null
  name: string | null
  username: string | null
  avatar_url: string | null
  system_role: string
  // Why email is read-only (the operator lookup key / the identity provider)
  email_note: string
}

export interface ProfileUpdateRequest {
  name?: string
  username?: string
  avatar_url?: string
}

class ApiClient {
  private baseUrl: string
  private defaultHeaders: Record<string, string>
  private getClerkToken: (() => Promise<string | null>) | null = null

  constructor() {
    // CRITICAL: Point directly to production backend since Next.js proxy is disabled
    // Frontend runs locally on Mac, backend on remote server

    // Try multiple ways to get the API URL (build-time and runtime)
    this.baseUrl =
      (typeof window !== 'undefined' && (window as any).__NEXT_PUBLIC_API_URL__) || // Runtime injection
      process.env.NEXT_PUBLIC_API_URL || // Build-time env var
      (typeof window !== 'undefined' && (window as any).NEXT_PUBLIC_API_URL) || // Runtime fallback
      ''

    // Warn if baseUrl is not set (will cause 404s)
    if (!this.baseUrl && typeof window !== 'undefined') {
      console.error('❌ NEXT_PUBLIC_API_URL is not set! API calls will fail.')
      console.error('Set NEXT_PUBLIC_API_URL in Railway frontend service variables')
      console.error('Current env:', process.env.NEXT_PUBLIC_API_URL || 'NOT SET')
    }

    // Default headers - NO API KEY, will use Clerk JWT
    this.defaultHeaders = {
      'Content-Type': 'application/json',
    }

  }

  /**
   * Set the Clerk token getter function
   * Call this from a React component that has access to useAuth()
   */
  public setClerkTokenGetter(getter: () => Promise<string | null>) {
    this.getClerkToken = getter
    if (process.env.NODE_ENV !== 'production') console.log('✅ Clerk token getter configured')
  }

  // PRD-168 S3: the mock config/control system is removed — dev uses the real
  // PRD-153 local stack. request() no longer falls back to mock data.

  // PRD-168 S3: mock control removed. setCurrentPage kept as a no-op so existing
  // usePageAPI() callers don't break; it drives no behaviour now.
  public setCurrentPage(_pageName: string) {}



  /** Public base URL for building backend requests (e.g. skills API that uses raw fetch). */
  getBaseUrl(): string {
    return (this.baseUrl || '').replace(/\/$/, '')
  }

  /** Auth headers (Clerk JWT + workspace) for use with raw fetch. */
  async getAuthHeaders(): Promise<Record<string, string>> {
    const headers: Record<string, string> = {}
    if (typeof window !== 'undefined') {
      const workspaceId = _adminWorkspaceOverride || localStorage.getItem('last_active_workspace') || localStorage.getItem('last_active_org')
      if (workspaceId) headers['X-Workspace-ID'] = workspaceId
    }
    if (this.getClerkToken) {
      try {
        const token = await this.getClerkToken()
        if (token) headers['Authorization'] = `Bearer ${token}`
      } catch (_) { }
    }
    return headers
  }

  async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`

    // Get Clerk token if available (with timeout to prevent hanging)
    let token: string | null = null
    if (this.getClerkToken) {
      try {
        // Add 2 second timeout to prevent hanging indefinitely
        const tokenPromise = this.getClerkToken()
        const timeoutPromise = new Promise<null>((resolve) => setTimeout(() => resolve(null), 2000))
        token = await Promise.race([tokenPromise, timeoutPromise])
        if (!token) {
          console.warn('⚠️ Clerk token fetch timed out after 2s - proceeding without auth')
        }
      } catch (error) {
        console.warn('⚠️ Failed to get Clerk token:', error)
      }
    }

    // Auto-stringify body if it's an object and not FormData
    let body = options.body
    if (body && typeof body === 'object' && !(body instanceof FormData)) {
      body = JSON.stringify(body)
    }

    const headers: Record<string, string> = {
      ...this.defaultHeaders,
      ...(options.headers as Record<string, string>),
    }

    // FormData needs browser-set Content-Type with multipart boundary
    if (body instanceof FormData) {
      delete headers['Content-Type']
    }

    // Inject Workspace ID: admin override takes priority over localStorage
    if (typeof window !== 'undefined') {
      const workspaceId = _adminWorkspaceOverride
        || localStorage.getItem('last_active_workspace')
        || localStorage.getItem('last_active_org')
      if (workspaceId) {
        headers['X-Workspace-ID'] = workspaceId
      }
    }

    // Add Clerk JWT token if available
    if (token) {
      headers['Authorization'] = `Bearer ${token}`
    }

    const config: RequestInit = {
      ...options,
      body,
      headers,
    }

    try {
      const response = await fetch(url, {
        ...config,
        redirect: 'follow' // Follow redirects automatically
      })

      if (!response.ok) {
        console.error('❌ API Error:', response.status, response.statusText)
        // Try to extract detail message from response body
        let detail = response.statusText
        try {
          const errorBody = await response.json()
          if (errorBody?.detail) {
            detail = typeof errorBody.detail === 'string' ? errorBody.detail : JSON.stringify(errorBody.detail)
          }
        } catch {
          // Response body not JSON, use statusText
        }
        if (response.status === 401) {
          throw new Error(
            'HTTP 401: Unauthorized (missing/invalid Clerk token). ' +
            'Make sure you are signed in and the API client is configured with Clerk.'
          )
        }
        throw new Error(detail || `HTTP ${response.status}`)
      }

      // Handle empty bodies (204 No Content, or any successful response with no body)
      // without throwing a JSON parse error.
      const text = await response.text()
      const data = text ? JSON.parse(text) : null
      if (process.env.NODE_ENV !== 'production') console.log('✅ API Success:', endpoint, 'Data type:', Array.isArray(data) ? `array[${data.length}]` : typeof data)

      return data
    } catch (error: any) {
      // PRD-168 S3: no mock fallback. Real failures surface honestly instead of
      // being masked with hardcoded data (the dev local stack covers dev).
      if (process.env.NODE_ENV !== 'production') console.error('API request failed:', endpoint, error.message)
      throw error
    }
  }

  // ===== SYSTEM ENDPOINTS =====

  // Generic HTTP methods
  async get<T = any>(endpoint: string, options?: RequestInit) {
    return this.request<T>(endpoint, { ...options, method: 'GET' })
  }

  async post<T = any>(endpoint: string, body?: any, options?: RequestInit) {
    return this.request<T>(endpoint, { ...options, method: 'POST', body })
  }

  async put<T = any>(endpoint: string, body?: any, options?: RequestInit) {
    return this.request<T>(endpoint, { ...options, method: 'PUT', body })
  }

  async patch<T = any>(endpoint: string, body?: any, options?: RequestInit) {
    return this.request<T>(endpoint, { ...options, method: 'PATCH', body })
  }

  async delete<T = any>(endpoint: string, options?: RequestInit) {
    return this.request<T>(endpoint, { ...options, method: 'DELETE' })
  }

  async getSystemHealth() {
    return this.request('/api/system/health')
  }

  async getSystemMetrics() {
    return this.request('/api/system/metrics')
  }

  async getApiHealth() {
    return this.request('/api/health/endpoints')
  }

  // ===== PRD-233 S6 Local profile (GET both editions, PUT local only) =====
  async getProfile(): Promise<ProfileResponse> {
    return this.request<ProfileResponse>('/api/profile')
  }

  async updateProfile(data: ProfileUpdateRequest): Promise<ProfileResponse> {
    return this.request<ProfileResponse>('/api/profile', {
      method: 'PUT',
      body: JSON.stringify(data),
    })
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
      body: JSON.stringify({
        config_key: key,
        config_value: value,
        description: 'Updated via system configs interface'
      })
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
  async getAgents(
    skip = 0,
    limit = 100,
    options: { includeWorkspaceSystem?: boolean } = {},
  ) {
    const params = new URLSearchParams({ skip: String(skip), limit: String(limit) })
    if (options.includeWorkspaceSystem) params.set('include_workspace_system', 'true')
    return this.request(`/api/agents/?${params.toString()}`)
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

  // ===== MARKETPLACE ENDPOINTS =====
  async getMarketplaceItems(filters?: {
    type?: string
    category?: string
    search?: string
    featured?: boolean
    limit?: number
    offset?: number
  }) {
    const params = new URLSearchParams()
    if (filters?.type) params.append('type', filters.type)
    if (filters?.category) params.append('category', filters.category)
    if (filters?.search) params.append('search', filters.search)
    if (filters?.featured !== undefined) params.append('featured', String(filters.featured))
    if (filters?.limit) params.append('limit', String(filters.limit))
    if (filters?.offset) params.append('offset', String(filters.offset))

    const queryString = params.toString()
    return this.request(`/api/marketplace/items${queryString ? `?${queryString}` : ''}`)
  }

  async getMarketplaceItem(itemId: number) {
    return this.request(`/api/marketplace/items/${itemId}`)
  }

  async getFeaturedMarketplaceItems(limit: number = 8) {
    return this.request(`/api/marketplace/featured?limit=${limit}`)
  }

  async installMarketplaceItem(itemId: number) {
    return this.request(`/api/marketplace/items/${itemId}/install`, {
      method: 'POST'
    })
  }

  async submitToMarketplace(data: {
    item_type: string
    name?: string
    description?: string
    category?: string
    tags?: string[]
    metadata: any
  }) {
    return this.request('/api/marketplace/submit', {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  async checkMarketplaceUpdates() {
    return this.request('/api/marketplace/updates')
  }

  async toggleMarketplaceFeatured(itemId: number) {
    return this.request(`/api/marketplace/items/${itemId}/toggle-featured`, {
      method: 'POST'
    })
  }

  // ===== WORKFLOW ENDPOINTS =====
  async getWorkflows() {
    return this.request('/api/workflows')
  }

  async getWorkflow(workflowId: string) {
    return this.request(`/api/workflows/${workflowId}`)
  }

  async createWorkflow(data: any) {
    return this.request('/api/workflows', {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  async getActiveWorkflows() {
    return this.request<{
      active_workflows: any[];
      recipe_runs: any[];
      total_active: number;
      total_recipe_runs: number;
      system_load: number;
      last_updated: string;
    }>('/api/workflows/active')
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

  async deleteWorkflow(workflowId: number) {
    return this.request(`/api/workflows/${workflowId}`, {
      method: 'DELETE'
    })
  }

  async duplicateWorkflow(workflowId: number, data?: any) {
    return this.request(`/api/workflows/${workflowId}/duplicate`, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined
    })
  }

  async cleanupOldWorkflows(days: number = 30) {
    return this.request(`/api/workflows/cleanup/old?days=${days}`, {
      method: 'DELETE'
    })
  }

  async getWorkflowExecutions(workflowId?: string) {
    const url = workflowId
      ? `/api/workflows/executions/?workflow_id=${workflowId}`
      : '/api/workflows/executions/'
    return this.request(url)
  }

  async getWorkflowExecution(executionId: string) {
    return this.request(`/api/workflows/executions/${executionId}`)
  }

  async cancelWorkflowExecution(executionId: string) {
    return this.request(`/api/workflows/executions/${executionId}/cancel`, {
      method: 'POST'
    })
  }

  async getRecommendedWorkflowTemplates() {
    return this.request('/api/workflows/templates/recommended')
  }

  // Workflow Templates CRUD endpoints
  async listWorkflowTemplates(params?: {
    category?: string
    difficulty?: string
    is_featured?: boolean
    is_public?: boolean
    search?: string
    skip?: number
    limit?: number
    sort_by?: string
  }) {
    const queryParams = new URLSearchParams()
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          queryParams.append(key, String(value))
        }
      })
    }
    const query = queryParams.toString()
    return this.request(`/api/workflow-templates${query ? '?' + query : ''}`)
  }

  async getWorkflowTemplateById(templateId: string) {
    return this.request(`/api/workflow-templates/${templateId}`)
  }

  async createWorkflowTemplate(templateData: any) {
    return this.request('/api/workflow-templates', {
      method: 'POST',
      body: JSON.stringify(templateData)
    })
  }

  async updateWorkflowTemplate(templateId: string, templateData: any) {
    return this.request(`/api/workflow-templates/${templateId}`, {
      method: 'PUT',
      body: JSON.stringify(templateData)
    })
  }

  async deleteWorkflowTemplate(templateId: string) {
    return this.request(`/api/workflow-templates/${templateId}`, {
      method: 'DELETE'
    })
  }

  async recordTemplateUsage(templateId: string) {
    return this.request(`/api/workflow-templates/${templateId}/use`, {
      method: 'POST'
    })
  }

  async getFeaturedTemplates(limit?: number) {
    return this.request(`/api/workflow-templates/featured/list${limit ? '?limit=' + limit : ''}`)
  }

  async getTemplateCategories() {
    return this.request('/api/workflow-templates/categories/list')
  }

  // ===== WORKFLOW RECIPES ENDPOINTS =====

  async listWorkflowRecipes(params?: {
    category?: string
    difficulty?: string
    is_featured?: boolean
    is_public?: boolean
    search?: string
    skip?: number
    limit?: number
    sort_by?: string
  }) {
    const queryParams = new URLSearchParams()
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          queryParams.append(key, value.toString())
        }
      })
    }
    const query = queryParams.toString()
    return this.request(`/api/workflow-recipes${query ? '?' + query : ''}`)
  }

  async getWorkflowRecipeById(recipeId: string) {
    return this.request(`/api/workflow-recipes/${recipeId}`)
  }

  async createWorkflowRecipe(recipeData: any) {
    return this.request('/api/workflow-recipes', {
      method: 'POST',
      body: JSON.stringify(recipeData)
    })
  }

  async updateWorkflowRecipe(recipeId: string, recipeData: any) {
    return this.request(`/api/workflow-recipes/${recipeId}`, {
      method: 'PUT',
      body: JSON.stringify(recipeData)
    })
  }

  async deleteWorkflowRecipe(recipeId: string) {
    return this.request(`/api/workflow-recipes/${recipeId}`, {
      method: 'DELETE'
    })
  }

  async recordRecipeUsage(recipeId: string) {
    return this.request(`/api/workflow-recipes/${recipeId}/use`, {
      method: 'POST'
    })
  }

  async getFeaturedRecipes(limit?: number) {
    return this.request(`/api/workflow-recipes/featured/list${limit ? '?limit=' + limit : ''}`)
  }

  async getRecipeCategories() {
    return this.request('/api/workflow-recipes/categories/list')
  }

  // Marketplace recipe endpoints
  async submitRecipeToMarketplace(params: { recipe_id: string; category?: string; icon?: string }) {
    return this.request('/api/workflow-recipes/submit', {
      method: 'POST',
      body: JSON.stringify(params)
    })
  }

  async installRecipeFromMarketplace(recipeId: number) {
    return this.request(`/api/workflow-recipes/install/${recipeId}`, {
      method: 'POST'
    })
  }

  async executeRecipe(recipeId: string, inputData?: Record<string, any>) {
    return this.request(`/api/workflow-recipes/${recipeId}/execute`, {
      method: 'POST',
      body: JSON.stringify({ input_data: inputData || {} })
    })
  }

  async getRecipeSuggestions(recipeId: string) {
    return this.request(`/api/workflow-recipes/${recipeId}/suggestions`)
  }

  async getRecipeExecutions(recipeId: string, params?: { status?: string; skip?: number; limit?: number }) {
    const queryParams = new URLSearchParams()
    if (params?.status) queryParams.append('status', params.status)
    if (params?.skip !== undefined) queryParams.append('skip', params.skip.toString())
    if (params?.limit !== undefined) queryParams.append('limit', params.limit.toString())
    const query = queryParams.toString()
    return this.request(`/api/workflow-recipes/${recipeId}/executions${query ? '?' + query : ''}`)
  }

  async getRecipeExecution(recipeId: string, executionId: string) {
    return this.request(`/api/workflow-recipes/${recipeId}/executions/${executionId}`)
  }

  async cancelRecipeExecution(recipeId: string, executionId: string) {
    return this.request(`/api/workflow-recipes/${recipeId}/executions/${executionId}/cancel`, {
      method: 'POST'
    })
  }

  async getRecipeStepFullLogs(recipeId: string, executionId: string, stepOrder: number) {
    return this.request(`/api/workflow-recipes/${recipeId}/executions/${executionId}/steps/${stepOrder}/logs`)
  }

  // ===== CODEGRAPH ENDPOINTS (PRD-11) =====

  /** Index a GitHub repository */
  async codegraphIndexGithub(data: {
    project_name: string
    github_url: string
    branch?: string
    auth_token?: string
    exclude_patterns?: string[]
  }) {
    return this.request('/api/code-graph/index/github', {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  /** Search symbols by name (fuzzy matching) */
  async codegraphSearchSymbols(params: {
    project: string
    q: string
    symbol_type?: string
    limit?: number
  }) {
    const searchParams = new URLSearchParams({
      project: params.project,
      q: params.q,
      ...(params.symbol_type && { symbol_type: params.symbol_type }),
      ...(params.limit && { limit: params.limit.toString() })
    })
    return this.request(`/api/code-graph/search/symbols?${searchParams}`)
  }

  /** Semantic search using vector similarity */
  async codegraphSearchSemantic(params: {
    project: string
    q: string
    limit?: number
  }) {
    const searchParams = new URLSearchParams({
      project: params.project,
      q: params.q,
      ...(params.limit && { limit: params.limit.toString() })
    })
    return this.request(`/api/code-graph/search/semantic?${searchParams}`)
  }

  /** List all indexed projects */
  async codegraphListProjects() {
    return this.request('/api/code-graph/projects')
  }

  /** Get project details */
  async codegraphGetProject(projectId: number) {
    return this.request(`/api/code-graph/projects/${projectId}`)
  }

  /** Delete a project */
  async codegraphDeleteProject(projectId: number) {
    return this.request(`/api/code-graph/projects/${projectId}`, {
      method: 'DELETE'
    })
  }

  /** Re-index a project */
  async codegraphReindexProject(projectId: number) {
    return this.request(`/api/code-graph/projects/${projectId}/reindex`, {
      method: 'POST'
    })
  }

  /** Ask a natural language question about code */
  async codegraphAskQuestion(projectId: number, question: string) {
    return this.request(`/api/code-graph/projects/${projectId}/ask`, {
      method: 'POST',
      body: JSON.stringify({ question }),
    })
  }

  /** Get call graph for a symbol */
  async codegraphGetCallGraph(params: { project: string; symbol: string; depth?: number; direction?: string }) {
    const searchParams = new URLSearchParams({
      project: params.project,
      symbol: params.symbol,
      ...(params.depth && { depth: params.depth.toString() }),
      ...(params.direction && { direction: params.direction }),
    })
    return this.request(`/api/code-graph/call-graph?${searchParams}`)
  }

  /** Get architecture analysis for a project */
  async codegraphGetArchitecture(projectId: number) {
    return this.request(`/api/code-graph/projects/${projectId}/architecture`)
  }

  /** Health check */
  async codegraphHealth() {
    return this.request('/api/code-graph/health')
  }

  // ===== KNOWLEDGE GRAPH ENDPOINTS =====

  // PRD-165 S2 — cluster-first drill-in. Server-side subgraph queries so big
  // graphs never ship the full graph.json to the browser (Q28).

  /** Community overview — the cluster-first entry point (id, size, title?). */
  async graphCommunitiesOverview() {
    return this.request('/api/knowledge/graph/communities')
  }

  /** Induced subgraph for one community's members ({nodes, links}, capped). */
  async graphCommunitySubgraph(communityId: number, maxNodes = 300) {
    return this.request(`/api/knowledge/graph/community/${communityId}?max_nodes=${maxNodes}`)
  }

  /** A node + its 1-hop neighbourhood ('expand from here'). */
  async graphExpandNode(nodeId: string, maxNodes = 150) {
    return this.request(`/api/knowledge/graph/node/${encodeURIComponent(nodeId)}/neighbors?max_nodes=${maxNodes}`)
  }

  /** Shortest path between two node ids ({found, path, nodes, links}). */
  async graphPath(source: string, target: string) {
    return this.request(`/api/knowledge/graph/path?source=${encodeURIComponent(source)}&target=${encodeURIComponent(target)}`)
  }

  /** Label search for search-to-focus ({matches: [{id,label,...}]}). */
  async graphSearchNodes(query: string, limit = 25) {
    return this.request(`/api/knowledge/graph/search?q=${encodeURIComponent(query)}&limit=${limit}`)
  }

  /** Rename a community / edit its summary (PRD-165 S3 — editable labels). */
  async graphSetCommunityLabel(communityId: number, title: string, summary?: string) {
    return this.request(`/api/knowledge/graph/community/${communityId}/label`, {
      method: 'PATCH',
      body: JSON.stringify(summary !== undefined ? { title, summary } : { title }),
      headers: { 'Content-Type': 'application/json' },
    })
  }

  async importBusinessGraph(file: File, merge: boolean = false) {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('merge', String(merge))

    // Use raw fetch (same pattern as uploadDocument) — this.request() has
    // Content-Type issues with FormData and may route through Next.js proxy.
    const headers: any = { ...this.defaultHeaders }
    delete headers['Content-Type'] // Let browser set multipart/form-data with boundary

    if (typeof window !== 'undefined') {
      const workspaceId = _adminWorkspaceOverride || localStorage.getItem('last_active_workspace') || localStorage.getItem('last_active_org')
      if (workspaceId) headers['X-Workspace-ID'] = workspaceId
    }

    if (this.getClerkToken) {
      try {
        const token = await this.getClerkToken()
        if (token) headers['Authorization'] = `Bearer ${token}`
      } catch (error) {
        console.warn('[GraphImport] Failed to get Clerk token:', error)
      }
    }

    const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || ''
    const url = `${BACKEND_URL}/api/knowledge/graph/import`
    console.log('[GraphImport] Uploading to:', url, 'file:', file.name, file.size, 'bytes')

    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: formData,
    })

    console.log('[GraphImport] Response:', response.status, response.statusText)

    if (!response.ok) {
      const errorText = await response.text()
      console.error('[GraphImport] Error:', errorText)
      throw new Error(errorText || `HTTP ${response.status}`)
    }

    return response.json()
  }

  async buildBusinessGraph() {
    const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || ''
    const url = `${BACKEND_URL}/api/knowledge/graph/build`

    const headers: any = { ...this.defaultHeaders }
    if (typeof window !== 'undefined') {
      const workspaceId = _adminWorkspaceOverride || localStorage.getItem('last_active_workspace') || localStorage.getItem('last_active_org')
      if (workspaceId) headers['X-Workspace-ID'] = workspaceId
    }
    if (this.getClerkToken) {
      try {
        const token = await this.getClerkToken()
        if (token) headers['Authorization'] = `Bearer ${token}`
      } catch (_) {}
    }

    const response = await fetch(url, { method: 'POST', headers })
    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(errorText || `HTTP ${response.status}`)
    }
    return response.json()
  }

  // ===== ATTACHMENT ENDPOINTS (PRD-127: Ephemeral Attachments) =====

  /**
   * Upload an ephemeral attachment for chat, missions, tasks, or channels.
   * Files are stored with a 7-day TTL — use uploadDocument for persistent storage.
   */
  async uploadAttachment(file: File): Promise<{
    attachment_id: string
    filename: string
    mime: string
    media_type: 'image' | 'document'
    size_bytes: number
  }> {
    const formData = new FormData()
    formData.append('file', file)

    // Use fetch directly for file upload (don't use this.request which sets Content-Type)
    const headers: Record<string, string> = { ...this.defaultHeaders }
    delete headers['Content-Type'] // Let browser set multipart/form-data with boundary

    // Inject Workspace ID from LocalStorage
    if (typeof window !== 'undefined') {
      const workspaceId = _adminWorkspaceOverride || localStorage.getItem('last_active_workspace') || localStorage.getItem('last_active_org')
      if (workspaceId) {
        headers['X-Workspace-ID'] = workspaceId
      }
    }

    // Add Clerk JWT token if available
    if (this.getClerkToken) {
      try {
        const token = await this.getClerkToken()
        if (token) {
          headers['Authorization'] = `Bearer ${token}`
        }
      } catch {
        // Token retrieval failed — continue without auth
      }
    }

    // Upload directly to backend, bypassing Next.js proxy
    const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || ''
    const url = `${BACKEND_URL}/api/attachments`

    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: formData,
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(`Attachment upload failed (${response.status}): ${errorText || response.statusText}`)
    }

    return response.json()
  }

  async getAttachment(attachmentId: string) {
    return this.request(`/api/attachments/${attachmentId}`)
  }

  async deleteAttachment(attachmentId: string) {
    return this.request(`/api/attachments/${attachmentId}`, { method: 'DELETE' })
  }

  // ===== DOCUMENT ENDPOINTS =====
  async uploadDocument(file: File, metadata?: any) {
    console.log('[Upload] Starting document upload:', {
      filename: file.name,
      size: file.size,
      type: file.type,
      metadata: metadata
    })

    const formData = new FormData()
    formData.append('file', file)

    // Add metadata as separate form fields (backend expects description and tags separately)
    if (metadata?.description) {
      formData.append('description', metadata.description)
    }
    if (metadata?.tags) {
      const tags = Array.isArray(metadata.tags) ? metadata.tags.join(',') : metadata.tags
      formData.append('tags', tags)
    }
    if (metadata?.team_access) {
      const teamAccess = Array.isArray(metadata.team_access) ? metadata.team_access.join(',') : metadata.team_access
      if (teamAccess) {
        formData.append('team_access', teamAccess)
      }
    }

    // Use fetch directly for file upload (don't use this.request which sets Content-Type)
    const headers: any = { ...this.defaultHeaders }
    delete headers['Content-Type'] // Let browser set multipart/form-data with boundary

    // Inject Workspace ID from LocalStorage (same as request method)
    if (typeof window !== 'undefined') {
      const workspaceId = _adminWorkspaceOverride || localStorage.getItem('last_active_workspace') || localStorage.getItem('last_active_org')
      if (workspaceId) {
        headers['X-Workspace-ID'] = workspaceId
        console.log('[Upload] 🏢 Injected workspace context:', workspaceId)
      }
    }

    // Add Clerk JWT token if available
    if (this.getClerkToken) {
      try {
        const token = await this.getClerkToken()
        if (token) {
          headers['Authorization'] = `Bearer ${token}`
          console.log('[Upload] 🔐 Added Clerk JWT to upload request')
        }
      } catch (error) {
        console.warn('[Upload] ⚠️ Failed to get Clerk token:', error)
      }
    }

    // CRITICAL: Upload must go DIRECTLY to backend, bypassing Next.js proxy
    const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || ''
    const url = `${BACKEND_URL}/api/documents/upload`
    console.log('[Upload] Uploading DIRECTLY to backend:', url)
    console.log('[Upload] FormData entries:', Array.from(formData.entries()).map(([k, v]) => [k, typeof v === 'string' ? v : 'File']))

    const response = await fetch(url, {
      method: 'POST',
      headers: headers,
      body: formData
    })

    console.log('[Upload] Response status:', response.status, response.statusText)

    if (!response.ok) {
      const errorText = await response.text()
      console.error('[Upload] Error response:', errorText)
      throw new Error(`Upload failed (${response.status}): ${errorText || response.statusText}`)
    }

    const data = await response.json()
    console.log('[Upload] Success! Response data:', data)
    return data
  }

  async preprocessDocument(documentId: string) {
    // Backend expects GET, not POST
    return this.request(`/api/documents/${documentId}/preprocess`, {
      method: 'GET'
    })
  }

  async getDocuments(team?: string) {
    // PRD-158 S3: optional server-side team filter.
    const qs = team ? `?team=${encodeURIComponent(team)}` : ''
    return this.request(`/api/documents/${qs}`)
  }

  // PRD-157 S6: real processing-queue status (replaces the FALLBACK_DATA placebo).
  async getProcessingQueue() {
    return this.request('/api/documents/queue/status')
  }

  // PRD-158: Teams entity.
  async getTeams(): Promise<Array<{ id: number; name: string; normalized_name: string }>> {
    return this.request('/api/teams')
  }

  async createTeam(name: string): Promise<{ id: number; name: string; normalized_name: string }> {
    return this.request('/api/teams', {
      method: 'POST',
      body: JSON.stringify({ name }),
    })
  }

  // PRD-158 S3: per-team document counts (server-side aggregate).
  async getDocumentTeamCounts(): Promise<{ counts: Record<string, number>; untagged: number; total: number }> {
    return this.request('/api/documents/team-counts')
  }

  // PRD-158 S4: team-access edits (backend PATCH/bulk endpoints already exist).
  async updateDocumentTeamAccess(documentId: number, teamAccess: string[]) {
    return this.request(`/api/documents/${documentId}/team-access`, {
      method: 'PATCH',
      body: JSON.stringify({ team_access: teamAccess }),
    })
  }

  async bulkUpdateTeamAccess(documentIds: number[], teamAccess: string[]) {
    return this.request('/api/documents/bulk-team-access', {
      method: 'POST',
      body: JSON.stringify({ document_ids: documentIds, team_access: teamAccess }),
    })
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

  async semanticSearch(query: string, options: { limit?: number; min_similarity?: number } = {}) {
    const params = new URLSearchParams({
      query,
      limit: String(options.limit ?? 10),
      min_similarity: String(options.min_similarity ?? 0.7)
    })
    return this.request(`/api/documents/search?${params}`, {
      method: 'POST'
    })
  }

  async ragRetrieve(params: { query: string; max_chunks?: number; max_tokens?: number; diversity?: number }) {
    const searchParams = new URLSearchParams({ query: params.query })
    if (params.max_chunks != null) searchParams.set('max_chunks', String(params.max_chunks))
    if (params.max_tokens != null) searchParams.set('max_tokens', String(params.max_tokens))
    if (params.diversity != null) searchParams.set('diversity', String(params.diversity))
    return this.request(`/api/documents/rag/retrieve?${searchParams}`, { method: 'POST' })
  }

  // ===== SKILLS ENDPOINTS =====
  async getSkills() {
    const skills = await this.request('/api/v1/skills') as any[]
    // Add default difficulty field if missing
    return skills.map((skill: any) => ({
      ...skill,
      difficulty: skill.difficulty || 'intermediate' // Default difficulty
    }))
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
    return this.request(`/api/v1/skills/agents/${agentId}/skills`)
  }

  async addSkillToAgent(agentId: string, skillId: string) {
    return this.request(`/api/v1/skills/agents/${agentId}/skills`, {
      method: 'POST',
      body: JSON.stringify([parseInt(skillId)])
    })
  }

  async removeSkillFromAgent(agentId: string, skillId: string) {
    return this.request(`/api/v1/skills/agents/${agentId}/skills?skill_ids=${skillId}`, {
      method: 'DELETE'
    })
  }

  // ===== CONTEXT ENGINEERING ENDPOINTS =====
  async getContextStats() {
    return this.request('/api/context/stats')
  }

  async getContextSources() {
    return this.request('/api/context/sources')
  }

  async getContextPatterns() {
    return this.request('/api/context/patterns')
  }

  async getContextPerformance(timeRange: string = '24h') {
    return this.request(`/api/context/performance?time_range=${timeRange}`)
  }

  async getRecentContextQueries() {
    return this.request('/api/context/queries/recent')
  }

  async getOptimizationRecommendations() {
    return this.request('/api/context/optimize')
  }

  // ===== ANALYTICS ENDPOINTS =====
  async getPerformanceEnhancements() {
    return this.request('/api/analytics/performance/all-enhancements')
  }

  async getAnalyticsMetrics() {
    return this.request('/api/analytics/metrics')
  }

  // ===== TOOLS / INTEGRATIONS ENDPOINTS =====
  async getTools(params?: { status?: string; category?: string; provider?: string; search?: string; skip?: number; limit?: number }) {
    // Tools UI reads from rewrite `/api/tools/*` endpoints.
    const status = params?.status
    const category = params?.category
    const search = params?.search
    const skip = params?.skip ?? 0
    const limit = params?.limit ?? 20

    // Connected tools only (active connections)
    if (status === 'active') {
      const connected = (await this.request('/api/tools/connected')) as any
      const apps: any[] = connected?.apps || []

      const stableId = (s: string) => {
        // Simple deterministic hash -> negative int (avoids collisions with DB ids)
        let h = 0
        for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0
        return h === 0 ? -1 : -Math.abs(h)
      }

      let normalized = apps.map((a: any) => ({
        id: a.id ?? stableId(String(a.app_name || '')),
        name: a.app_name,
        description: a.description || '',
        display_name: a.display_name,  // Keep display_name for search
        integration_url: 'composio://',
        capabilities: {},
        credentials_schema: {},
        status: a.status || 'active',  // Use actual status from API (active/added/pending)
        enabled: true,
        provider: 'Composio',
        version: '1.0.0',
        icon: a.logo_url,
        logo: a.logo_url,
        category: (a.categories || [])[0] || 'Integration',
        tags: a.categories || [],
        metadata: {
          action_count: a.action_count || 0,
          trigger_count: a.trigger_count || 0,
          auth_schemes: a.auth_schemes || [],
          triggers: a.triggers || [],
        },
        updated_at: null,
      }))

      // Apply search filter if provided (same logic as backend)
      if (search) {
        const searchLower = search.toLowerCase()
        normalized = normalized.filter((tool: any) =>
          (tool.display_name || '').toLowerCase().includes(searchLower) ||
          (tool.name || '').toLowerCase().includes(searchLower) ||
          (tool.description || '').toLowerCase().includes(searchLower)
        )
      }

      const paged = normalized.slice(skip, skip + limit)
      const total = normalized.length
      const pages = limit ? Math.ceil(total / limit) : 1
      return {
        data: paged,
        pagination: {
          total,
          skip,
          limit,
          pages,
          current_page: limit ? Math.floor(skip / limit) + 1 : 1,
        },
      }
    }

    // Marketplace (cached) tools
    const q = new URLSearchParams()
    if (category) q.append('category', category)
    if (search) q.append('search', search)
    q.append('limit', String(Math.min(limit, 500)))
    q.append('offset', String(skip))

    const marketplace = (await this.request(`/api/tools/marketplace?${q.toString()}`)) as any
    const apps: any[] = marketplace?.apps || []

    const normalized = apps.map((a: any) => ({
      id: a.id,
      name: a.app_name,
      description: a.description || '',
      integration_url: 'composio://',
      capabilities: {},
      credentials_schema: {},
      status: a.is_connected ? 'active' : 'available',
      enabled: !!a.is_connected,
      provider: 'Composio',
      version: '1.0.0',
      icon: a.logo_url,
      logo: a.logo_url,
      category: (a.categories || [])[0] || 'Integration',
      tags: a.categories || [],
      metadata: {
        action_count: a.action_count || 0,
        trigger_count: a.trigger_count || 0,
        auth_schemes: a.auth_schemes || [],
        triggers: a.triggers || [],  // Include triggers array from API
      },
      updated_at: marketplace?.last_synced || null,
    }))

    const total = marketplace?.total_apps ?? normalized.length
    const pages = limit ? Math.ceil(total / limit) : 1

    return {
      data: normalized,
      pagination: {
        total,
        skip,
        limit,
        pages,
        current_page: limit ? Math.floor(skip / limit) + 1 : 1,
      },
    }
  }

  async getToolCategories() {
    const stats = (await this.request('/api/tools/stats')) as any
    const categories = stats?.categories || {}
    // Return top 15 categories by app count (sorted descending)
    // This keeps the category filter dropdown manageable (734 categories is unusable)
    // Note: All apps remain fully searchable - this only limits the category filter options
    const allCategories = Object.entries(categories)
      .map(([name, count]) => ({ id: name, name, count: count as number }))
      .sort((a, b) => b.count - a.count) // Sort by count descending (most popular first)
      .slice(0, 15) // Take top 15 most popular categories
    return allCategories
  }

  async getWorkspaceTools() {
    // All workspace tools (connected + not connected)
    return this.request('/api/tools/workspace')
  }

  async getToolsStats() {
    const stats = (await this.request('/api/tools/stats')) as any
    return {
      total_tools: stats?.total_apps ?? 0,
      tools_available: stats?.total_actions ?? 0,
      connected_apps: stats?.connected_apps ?? 0,
      categories: stats?.categories ? Object.keys(stats.categories).length : 0,
      last_synced: stats?.last_synced ?? null,
    }
  }

  async syncToolsCache(syncType: 'full' | 'incremental' = 'full') {
    return this.request(`/api/tools/sync?sync_type=${syncType}`, { method: 'POST' })
  }

  async getIntegrationsStatus() {
    // PRD-233 S2: honest integrations state — the same predicate the tool
    // router uses to decide whether Composio tools are offered at all.
    return this.request<{
      available: boolean
      reason: string | null
      key_configured: boolean
      apps_cached: number
      last_sync: string | null
      sync_status: 'running' | 'completed' | 'failed' | string | null
    }>('/api/tools/integrations/status')
  }

  async syncOpenRouterCache() {
    return this.request('/api/openrouter/sync', { method: 'POST' })
  }

  // ===== LEGACY METHODS (for backward compatibility) =====
  // These methods now properly throw errors so mock fallback system handles them
  async getSystemActivities(limit = 10) {
    // Will use mock fallback automatically if endpoint fails
    return this.request(`/api/system/activities?limit=${limit}`)
  }

  async getAgentLogs(id: string) {
    // Will use mock fallback automatically if endpoint fails
    return this.request(`/api/agents/${id}/logs`)
  }

  async getAgentStats(id: string) {
    // Will use mock fallback automatically if endpoint fails
    return this.request(`/api/agents/${id}/stats`)
  }

  async startAgent(id: string) {
    // Agent "start/stop" is a status toggle on the record — no dedicated
    // route exists (and never did). Reuse the canonical PUT /api/agents/{id}.
    return this.updateAgent(id, { status: 'active' })
  }

  async stopAgent(id: string) {
    return this.updateAgent(id, { status: 'inactive' })
  }

  async getDocumentAnalytics() {
    // Will use mock fallback automatically if endpoint fails
    return this.request('/api/documents/analytics')
  }

  async runWorkflow(id: string, inputs?: any) {
    return this.executeWorkflow(id, inputs || {})
  }

  async getWorkflowTemplates() {
    return this.getRecommendedWorkflowTemplates()
  }

  async getPerformanceAnalytics(timeRange: string) {
    // Use existing /api/system/metrics with timeRange parameter
    return this.request(`/api/system/metrics?timeRange=${timeRange}`)
  }

  async getUsageAnalytics(timeRange: string) {
    // Will use mock fallback automatically if endpoint fails
    return this.request(`/api/usage/analytics?timeRange=${timeRange}`)
  }

  async getAllMetrics() {
    return this.request('/api/metrics/all')
  }

  // ===== PRD-142 Wave 0 "Is it working?" tiles =====
  async getActivationMetrics(): Promise<ActivationMetric> {
    return this.request<ActivationMetric>('/api/analytics/activation')
  }

  async getMissionSuccessRate(): Promise<MissionSuccessRateMetric> {
    return this.request<MissionSuccessRateMetric>('/api/analytics/dashboard/success-rate')
  }

  async getErrorsBySubsystem(window: string = '24h'): Promise<ErrorsBySubsystemMetric> {
    return this.request<ErrorsBySubsystemMetric>(`/api/analytics/errors/by-subsystem?window=${encodeURIComponent(window)}`)
  }

  async getWidgetEngagement(window: string = '7d'): Promise<WidgetEngagementMetric> {
    return this.request<WidgetEngagementMetric>(`/api/analytics/widget-engagement?window=${encodeURIComponent(window)}`)
  }

  async getPrimitiveHealth(): Promise<PrimitiveHealthMetric> {
    return this.request<PrimitiveHealthMetric>('/api/analytics/primitive-health')
  }

  // ===== PRD-185 S12 own-workspace cockpit tiles =====
  async getSLOs(window: string = '24h'): Promise<SlosMetric> {
    return this.request<SlosMetric>(`/api/analytics/slos?window=${encodeURIComponent(window)}`)
  }

  // PRD-197 S4 — per-seam retrieval substrate health
  async getSubstrateHealth(window: string = '24h'): Promise<SubstrateHealthMetric> {
    return this.request<SubstrateHealthMetric>(`/api/analytics/substrate-health?window=${encodeURIComponent(window)}`)
  }

  async getWorkspaceActivation(): Promise<WorkspaceActivationMetric> {
    return this.request<WorkspaceActivationMetric>('/api/analytics/activation/workspace')
  }

  async getDeliverableFreshness(): Promise<DeliverableFreshnessMetric> {
    return this.request<DeliverableFreshnessMetric>('/api/analytics/deliverable-freshness')
  }

  async getCommerceIntegrity(): Promise<CommerceIntegrityMetric> {
    return this.request<CommerceIntegrityMetric>('/api/analytics/commerce-integrity')
  }

  async getAgentAnalytics(timeRange: string) {
    // Will use mock fallback automatically if endpoint fails
    return this.request(`/api/agents/analytics?timeRange=${timeRange}`)
  }

  async getSettings() {
    return this.getSystemConfig()
  }

  async updateSettings(data: any) {
    return this.updateSystemConfig(data)
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

  // ===== MARKETPLACE PLUGINS ENDPOINTS =====

  async listPlugins(params?: {
    category?: string
    search?: string
    tags?: string[]
    sort?: string
    page?: number
    limit?: number
  }) {
    const q = new URLSearchParams()
    if (params?.category) q.append('category', params.category)
    if (params?.search) q.append('search', params.search)
    if (params?.tags) params.tags.forEach(t => q.append('tags', t))
    if (params?.sort) q.append('sort', params.sort)
    if (params?.page) q.append('page', String(params.page))
    if (params?.limit) q.append('limit', String(params.limit))
    const qs = q.toString()
    return this.request(`/api/marketplace/plugins${qs ? `?${qs}` : ''}`)
  }

  async getPluginDetail(pluginId: string) {
    return this.request(`/api/marketplace/plugins/${pluginId}`)
  }

  async getPluginContent(pluginId: string) {
    return this.request(`/api/marketplace/plugins/${pluginId}/content`)
  }

  async listPluginCategories() {
    return this.request('/api/marketplace/plugins/categories')
  }

  async getWorkspacePlugins(workspaceId: string) {
    return this.request(`/api/workspaces/${workspaceId}/plugins`)
  }

  async enablePlugin(workspaceId: string, pluginId: string) {
    return this.request(`/api/workspaces/${workspaceId}/plugins`, {
      method: 'POST',
      body: JSON.stringify({ plugin_id: pluginId })
    })
  }

  async disablePlugin(workspaceId: string, pluginId: string) {
    return this.request(`/api/workspaces/${workspaceId}/plugins/${pluginId}`, {
      method: 'DELETE'
    })
  }

  async getAgentPlugins(agentId: string) {
    return this.request(`/api/agents/${agentId}/plugins`)
  }

  async updateAgentPlugins(agentId: string, pluginIds: string[]) {
    return this.request(`/api/agents/${agentId}/plugins`, {
      method: 'PUT',
      body: JSON.stringify({ plugin_ids: pluginIds })
    })
  }

  // ===== ADMIN PLUGIN MANAGEMENT ENDPOINTS =====

  async uploadPlugin(formData: FormData) {
    // Use raw fetch for multipart upload — don't set Content-Type header
    const headers = await this.getAuthHeaders()
    const url = `${this.baseUrl}/api/admin/plugins/upload`
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: formData
    })
    if (!response.ok) {
      let detail = response.statusText
      try {
        const errorBody = await response.json()
        if (errorBody?.detail) {
          detail = typeof errorBody.detail === 'string' ? errorBody.detail : JSON.stringify(errorBody.detail)
        }
      } catch { /* not JSON */ }
      throw new Error(detail || `HTTP ${response.status}`)
    }
    return response.json()
  }

  async approvePlugin(pluginId: string) {
    return this.request(`/api/admin/plugins/${pluginId}/approve`, {
      method: 'POST'
    })
  }

  async rejectPlugin(pluginId: string, reason: string) {
    return this.request(`/api/admin/plugins/${pluginId}/reject`, {
      method: 'POST',
      body: JSON.stringify({ reason })
    })
  }

  async getScanResults(pluginId: string) {
    return this.request(`/api/admin/plugins/${pluginId}/scan`)
  }

  async getPendingPlugins(page: number = 1, limit: number = 20) {
    return this.request(`/api/admin/plugins/pending?page=${page}&limit=${limit}`)
  }

  // ===== PERSONA ENDPOINTS =====

  async listPersonas(params?: { category?: string; scope?: string }) {
    const q = new URLSearchParams()
    if (params?.category) q.append('category', params.category)
    if (params?.scope) q.append('scope', params.scope)
    const qs = q.toString()
    return this.request(`/api/personas${qs ? '?' + qs : ''}`)
  }

  async getPersona(personaId: string) {
    return this.request(`/api/personas/${personaId}`)
  }

  async createWorkspacePersona(workspaceId: string, data: {
    name: string
    description?: string
    system_prompt: string
    voice_description?: string
    category?: string
    suggested_temperature?: number
  }) {
    return this.request(`/api/workspaces/${workspaceId}/personas`, {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  async updateWorkspacePersona(workspaceId: string, personaId: string, data: {
    name?: string
    description?: string
    system_prompt?: string
    voice_description?: string
    category?: string
    suggested_temperature?: number
  }) {
    return this.request(`/api/workspaces/${workspaceId}/personas/${personaId}`, {
      method: 'PUT',
      body: JSON.stringify(data)
    })
  }

  async deleteWorkspacePersona(workspaceId: string, personaId: string) {
    return this.request(`/api/workspaces/${workspaceId}/personas/${personaId}`, {
      method: 'DELETE'
    })
  }

  async getAgentPersona(agentId: number) {
    return this.request(`/api/agents/${agentId}/persona`)
  }

  async setAgentPersona(agentId: number, data: {
    persona_id?: string | null
    custom_prompt?: string | null
    use_custom?: boolean
  }) {
    return this.request(`/api/agents/${agentId}/persona`, {
      method: 'PUT',
      body: JSON.stringify(data)
    })
  }

  // ===== MULTI-AGENT ENDPOINTS (Working ✅) =====
  async getMultiAgentHealth() {
    return this.request('/api/multi-agent/health')
  }

  async coordinateAgents(data: { agents: string[], task: any, strategy?: string }) {
    return this.request('/api/multi-agent/coordination/coordinate', {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  async collaborativeReasoning(data: { problem: string, agents: string[], strategy?: string }) {
    return this.request('/api/multi-agent/reasoning/collaborative', {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  async getOptimizationStatistics() {
    return this.request('/api/multi-agent/optimization/statistics')
  }

  // ===== MEMORY ENDPOINTS =====
  async getMemoryStats() {
    return this.request('/api/v1/memory/stats/real')  // REAL data from database!
  }

  // ===== CONTEXT ENDPOINTS (All Working ✅) =====
  async getContextSystemHealth() {
    return this.request('/api/context/system/health')
  }

  async initializeContext(data?: any) {
    return this.request('/api/context/initialize', {
      method: 'POST',
      body: JSON.stringify(data || {})
    })
  }

  // PRD-192 S6: the PRD-56 workspace-task client methods (submit/list/get/
  // cancel against the deleted tasks router) were removed with the backend —
  // zero component/hook callers existed (grep-proven).

  async getWorkspaceFiles(workspaceId: string, path?: string) {
    const qs = path ? `?path=${encodeURIComponent(path)}` : ''
    return this.request(`/api/workspaces/${workspaceId}/files${qs}`)
  }

  async getWorkspaceFileContent(workspaceId: string, path: string) {
    return this.request(`/api/workspaces/${workspaceId}/files/content?path=${encodeURIComponent(path)}`)
  }

  async saveWorkspaceFile(workspaceId: string, path: string, content: string) {
    return this.request(`/api/workspaces/${workspaceId}/files/content`, {
      method: 'PUT',
      body: JSON.stringify({ path, content }),
    })
  }

  /**
   * Build a raw-bytes URL for a workspace file. Used by FilePreview for
   * binary types (PDF/DOCX/XLSX/image/video/audio) which fetch as arrayBuffer.
   * Returns a relative path; FilePreview will prepend baseUrl and add auth headers.
   */
  getWorkspaceFileRawUrl(workspaceId: string, path: string): string {
    return `/api/workspaces/${workspaceId}/files/raw?path=${encodeURIComponent(path)}`
  }

  async listGithubRepos(workspaceId: string, page = 1, perPage = 30) {
    return this.request(`/api/workspaces/${workspaceId}/github/repos?page=${page}&per_page=${perPage}`)
  }

  async cloneGithubRepo(workspaceId: string, repoUrl: string, branch?: string) {
    return this.request(`/api/workspaces/${workspaceId}/github/clone`, {
      method: 'POST',
      body: JSON.stringify({ repo_url: repoUrl, ...(branch ? { branch } : {}) }),
    })
  }

  // ===== ORCHESTRATOR ENDPOINTS (Working ✅) =====
  async submitTask(data: { task_description: string, priority?: string, context?: any }) {
    return this.request('/api/orchestrator/task/submit', {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  async executePhase(data: { phase: string, agents: string[], execution_type?: string }) {
    return this.request('/api/orchestrator/execute-phase', {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  // ===== EVALUATION ENDPOINTS (Working ✅) =====
  async getPerformanceMetrics() {
    return this.request('/api/evaluation/performance-metrics')
  }

  // ===== ASSIGNMENTS ENDPOINTS =====

  async getRecommendedAssignments(limit: number = 8) {
    return this.request(`/api/assignments/recommended?limit=${limit}`)
  }

  // ===== RAG CONFIGURATION ENDPOINTS =====
  async createRAGConfig(config: Omit<RAGConfig, 'id'>): Promise<RAGConfig> {
    return this.request('/api/context/rag/config', {
      method: 'POST',
      body: JSON.stringify(config)
    })
  }

  async updateRAGConfig(id: number, config: Partial<RAGConfig>): Promise<RAGConfig> {
    return this.request(`/api/context/rag/config/${id}`, {
      method: 'PUT',
      body: JSON.stringify(config)
    })
  }

  async deleteRAGConfig(id: number): Promise<void> {
    return this.request(`/api/context/rag/config/${id}`, {
      method: 'DELETE'
    })
  }

  async testRAGConfig(id: number, query: string): Promise<any> {
    console.log('[API] testRAGConfig called:', { id, query })
    const result = await this.request(`/api/context/rag/${id}/test?query=${encodeURIComponent(query)}`, {
      method: 'POST'
    })
    console.log('[API] testRAGConfig result:', result)
    return result
  }

  // ===== PRD-196 S1/S2 governance: approval grants (ws-admin gated) =====
  async listApprovalGrants(status?: string, kind?: string): Promise<ApprovalGrantsResponse> {
    // PRD-225: the Questions tab reuses this route with kind=question.
    const params = new URLSearchParams()
    if (status) params.set('status', status)
    if (kind) params.set('kind', kind)
    const qs = params.toString()
    return this.request<ApprovalGrantsResponse>(
      `/api/v1/approval-grants${qs ? `?${qs}` : ''}`,
    )
  }

  async grantApproval(grantId: number): Promise<{ grant: ApprovalGrant }> {
    return this.request(`/api/v1/approval-grants/${grantId}/grant`, { method: 'POST' })
  }

  async denyApproval(grantId: number): Promise<{ grant: ApprovalGrant }> {
    return this.request(`/api/v1/approval-grants/${grantId}/deny`, { method: 'POST' })
  }

  async revokeApproval(grantId: number): Promise<{ grant: ApprovalGrant }> {
    return this.request(`/api/v1/approval-grants/${grantId}/revoke`, { method: 'POST' })
  }

  // PRD-225: answer a pending question — records the answer and resumes the
  // parked subject through the grant resume machinery.
  async answerQuestion(
    grantId: number,
    body: { answer_text?: string; option?: string }
  ): Promise<{ grant: ApprovalGrant }> {
    return this.request(`/api/v1/approval-grants/${grantId}/answer`, {
      method: 'POST',
      body: JSON.stringify(body),
    })
  }

  // ===== PRD-196 S3 governance: status + audit log (ws-admin gated) =====
  async getGovernanceStatus(): Promise<GovernanceStatus> {
    return this.request<GovernanceStatus>('/api/v1/governance/status')
  }

  async getGovernanceAuditLog(filters: AuditLogFilters = {}): Promise<AuditLogResponse> {
    const params = new URLSearchParams()
    if (filters.action_prefix) params.set('action_prefix', filters.action_prefix)
    if (filters.actor_type) params.set('actor_type', filters.actor_type)
    if (filters.resource_type) params.set('resource_type', filters.resource_type)
    if (filters.since) params.set('since', filters.since)
    if (filters.until) params.set('until', filters.until)
    if (filters.limit != null) params.set('limit', String(filters.limit))
    if (filters.offset != null) params.set('offset', String(filters.offset))
    const qs = params.toString()
    return this.request<AuditLogResponse>(`/api/v1/governance/audit-log${qs ? `?${qs}` : ''}`)
  }

  // ===== PRD-196 S4 governance: policy posture + budget editors =====
  async getGovernancePolicy(): Promise<PolicyDocument> {
    return this.request<PolicyDocument>('/api/v1/governance/policy')
  }

  async putGovernancePolicy(body: Partial<PolicyDocument>): Promise<PolicyDocument> {
    return this.request<PolicyDocument>('/api/v1/governance/policy', {
      method: 'PUT',
      body: JSON.stringify(body),
    })
  }

  async getGovernanceBudget(): Promise<BudgetConfig> {
    return this.request<BudgetConfig>('/api/v1/governance/budget')
  }

  async putGovernanceBudget(body: BudgetConfig): Promise<BudgetConfig> {
    return this.request<BudgetConfig>('/api/v1/governance/budget', {
      method: 'PUT',
      body: JSON.stringify(body),
    })
  }

  // ===== PRD-196 S7 GDPR self-service (ws-admin gated) =====
  async getGdprExport(): Promise<any> {
    return this.request('/api/v1/gdpr/export')
  }

  async eraseGdprSubject(subjectId: string): Promise<GdprSubjectErasureResult> {
    return this.request<GdprSubjectErasureResult>('/api/v1/gdpr/erase-subject', {
      method: 'POST',
      body: JSON.stringify({ subject_id: subjectId }),
    })
  }

  async eraseGdprWorkspace(confirmWorkspaceId: string): Promise<GdprWorkspaceErasureResult> {
    return this.request<GdprWorkspaceErasureResult>('/api/v1/gdpr/erase', {
      method: 'POST',
      body: JSON.stringify({ confirm_workspace_id: confirmWorkspaceId }),
    })
  }

  // ===== PRD-204 S11 Auto Watcher: watchlist =====
  async listWatches(includeClosed: boolean = false): Promise<WatchesResponse> {
    const qs = includeClosed ? '?include_closed=true' : ''
    return this.request<WatchesResponse>(`/api/v1/watches${qs}`)
  }

  async getWatch(watchId: string): Promise<WatchDetailResponse> {
    return this.request<WatchDetailResponse>(`/api/v1/watches/${watchId}`)
  }

  async cancelWatch(watchId: string): Promise<{ watch: WatchRow }> {
    return this.request(`/api/v1/watches/${watchId}/cancel`, { method: 'POST' })
  }

  // ===== PRD-228 Fleet State: live floor read-model =====
  async getFleetState(): Promise<FleetStateResponse> {
    return this.request<FleetStateResponse>('/api/v1/fleet')
  }
}

export const apiClient = new ApiClient()
export default apiClient
