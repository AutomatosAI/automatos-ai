
/**
 * API Client - Only calls endpoints that actually exist based on test results
 * Base URL: Configured via NEXT_PUBLIC_API_URL environment variable (set by Docker Compose or .env.local)
 * 
 * MOCK SYSTEM:
 * - Tries real API first, falls back to mock data on failure
 * - Can be controlled globally or per-endpoint
 * - Use window.automatos.mocks to control from console
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

// Mock configuration interface
interface MockConfig {
  enabled: boolean
  endpoints: {
    [key: string]: boolean
  }
  logMockUsage: boolean
}

// ====================================================================
// PAGE-LEVEL MOCK CONFIGURATION - SIMPLE ON/OFF SWITCH PER PAGE
// ====================================================================
// Set to 'false' to use REAL APIs, 'true' to use MOCK data
// 
// Usage: When calling API from a page, pass the page name
// Example: apiClient.setCurrentPage('dashboard') 
//
const PAGE_MOCK_CONFIG: Record<string, boolean> = {
  // Core Pages
  'dashboard': false,        // ✅ Use real APIs - working endpoints
  'agents': false,           // ✅ Use real APIs - working endpoints (FIXED: recursive call bug)
  'workflows': false,        // ✅ Use real APIs - working endpoints
  'activity': false,         // ✅ Use real APIs - PRD-72 Activity Command Centre
  'documents': false,        // false = REAL APIs ✅ | true = MOCK data ❌
  'analytics': false,        // ✅ Use real APIs - working endpoints
  'context': false,          // ✅ Use real APIs - all context endpoints working
  'memory': false,           // ✅ Use real APIs - all memory endpoints working
  'field-theory': false,     // ✅ Use real APIs - all field theory endpoints working
  'multi-agent': false,      // ✅ Use real APIs - coordination/reasoning working
  'orchestrator': false,     // ✅ Use real APIs - task submission working

  // Settings/Admin Pages
  'settings': false,         // ✅ Use real APIs - credentials system ready
  'tools': false,            // ✅ Use real APIs - tools endpoints working
  'credentials': false,      // ✅ Use real APIs - credentials system ready
  'marketplace': false,      // ✅ Use real APIs - marketplace/composio endpoints working
  'admin': false,            // ✅ Use real APIs - admin plugin management

  // Testing/Development
  'test': true,              // 🧪 Always use mocks for testing
  'demo': true,              // 🧪 Always use mocks for demos
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

class ApiClient {
  private baseUrl: string
  private defaultHeaders: Record<string, string>
  private mockConfig: MockConfig
  private mockData: Record<string, () => any>
  private currentPage: string = '' // Track which page is making requests
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

    // Initialize mock config (dev only)
    const isDev = process.env.NODE_ENV !== 'production'
    this.mockConfig = isDev ? this.loadMockConfig() : { enabled: false, endpoints: {}, logMockUsage: false }
    this.mockData = isDev ? this.initializeMockData() : {}

    // Expose mock control to window for easy debugging (dev only)
    if (isDev && typeof window !== 'undefined') {
      (window as any).automatos = {
        ...(window as any).automatos,
        mocks: {
          enable: () => this.enableMocks(),
          disable: () => this.disableMocks(),
          toggle: (endpoint?: string) => this.toggleMock(endpoint),
          status: () => this.getMockStatus(),
          config: () => this.mockConfig
        }
      }
    }

    if (isDev) {
      console.log('🚀 API Client initialized')
      console.log(`📍 Base URL: ${this.baseUrl || 'relative URLs (Next.js)'}`)
      console.log(`📍 NEXT_PUBLIC_API_URL env: ${process.env.NEXT_PUBLIC_API_URL || 'NOT SET'}`)
      console.log('🔐 Auth: Clerk JWT (workspace-aware)')
      if (this.mockConfig.enabled) {
        console.warn('⚠️  Mock mode is ENABLED - Disable for production!')
      } else {
        console.log('✅ Real API mode enabled')
      }
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

  // Load mock config from localStorage or use defaults
  private loadMockConfig(): MockConfig {
    if (typeof window === 'undefined') {
      return { enabled: false, endpoints: {}, logMockUsage: true }
    }

    const stored = localStorage.getItem('automatos-mock-config')
    if (stored) {
      try {
        return JSON.parse(stored)
      } catch (e) {
        console.error('Failed to parse mock config:', e)
      }
    }

    // Default config - mocks DISABLED for working APIs
    return {
      enabled: false, // Disabled by default - use real APIs
      endpoints: {
        // Only enable mocks for endpoints that consistently fail
        '/api/insights/extract': false, // Requires auth - will work after backend restart
        '/api/recommendations/generate': false, // Requires auth - will work after backend restart
        '/api/learning/feedback': false, // Requires auth - will work after backend restart
        '/api/knowledge/share': false, // Requires auth - will work after backend restart
        '/api/system/performance-baseline': false, // Requires auth - will work after backend restart
        '/api/documents/upload': false, // Schema needs fixing
        '/api/multi-agent/behavior/learn': true, // 404 - not implemented
        '/api/multi-agent/optimization/adaptive': true, // 404 - not implemented
      },
      logMockUsage: true // Enable logging to see what's being used
    }
  }

  // Save mock config to localStorage
  private saveMockConfig() {
    if (typeof window !== 'undefined') {
      localStorage.setItem('automatos-mock-config', JSON.stringify(this.mockConfig))
    }
  }

  // Mock control methods
  private enableMocks() {
    this.mockConfig.enabled = true
    this.saveMockConfig()
    console.log('🎭 Mocks ENABLED globally')
  }

  private disableMocks() {
    this.mockConfig.enabled = false
    this.saveMockConfig()
    console.log('🌐 Mocks DISABLED - Using real APIs only')
  }

  private toggleMock(endpoint?: string) {
    if (!endpoint) {
      this.mockConfig.enabled = !this.mockConfig.enabled
      console.log(this.mockConfig.enabled ? '🎭 Mocks ENABLED' : '🌐 Mocks DISABLED')
    } else {
      const current = this.mockConfig.endpoints[endpoint] ?? true
      this.mockConfig.endpoints[endpoint] = !current
      console.log(`${endpoint}: ${!current ? '🎭 Mock' : '🌐 Real API'}`)
    }
    this.saveMockConfig()
  }

  private getMockStatus() {
    return {
      global: this.mockConfig.enabled,
      endpoints: this.mockConfig.endpoints,
      totalMocked: Object.values(this.mockConfig.endpoints).filter(v => v).length
    }
  }

  // Check if mock should be used for endpoint
  private shouldUseMock(endpoint: string): boolean {
    // Never use mocks in production
    if (process.env.NODE_ENV === 'production') return false

    // 1. CHECK PAGE-LEVEL CONFIG FIRST (highest priority)
    if (this.currentPage && this.currentPage in PAGE_MOCK_CONFIG) {
      // Page config overrides everything - return it directly
      // true = use mocks, false = use real APIs
      const useMock = PAGE_MOCK_CONFIG[this.currentPage]
      if (this.mockConfig.logMockUsage) {
        console.log(`🔍 ${endpoint} → Page config: ${useMock ? '🔄 MOCK' : '🌐 REAL API'}`)
      }
      return useMock
    }

    // 2. Check if mocks are globally disabled
    if (!this.mockConfig.enabled) return false

    // 3. Check specific endpoint config
    if (endpoint in this.mockConfig.endpoints) {
      return this.mockConfig.endpoints[endpoint]
    }

    // 4. Default to true if mocks are globally enabled
    return true
  }

  /**
   * Set the current page/feature to control mock behavior per page
   * @param pageName - Name of the page (e.g., 'dashboard', 'agents', 'workflows')
   * 
   * @example
   * // In your page component:
   * useEffect(() => {
   *   apiClient.setCurrentPage('dashboard')
   *   return () => apiClient.setCurrentPage('') // Clear on unmount
   * }, [])
   */
  public setCurrentPage(pageName: string) {
    this.currentPage = pageName.toLowerCase()
    if (process.env.NODE_ENV !== 'production') {
      const mockStatus = PAGE_MOCK_CONFIG[this.currentPage] ? 'MOCKS ON' : 'REAL APIs'
      console.log(`📄 Page: ${pageName} → ${mockStatus}`)
    }
  }

  /**
   * Get the current page mock status
   */
  public getPageMockStatus(pageName?: string): boolean {
    const page = pageName || this.currentPage
    return PAGE_MOCK_CONFIG[page.toLowerCase()] ?? false
  }

  /**
   * Override page mock setting temporarily (useful for testing)
   */
  public setPageMockOverride(pageName: string, useMocks: boolean) {
    PAGE_MOCK_CONFIG[pageName.toLowerCase()] = useMocks
    if (process.env.NODE_ENV !== 'production') console.log(`🔧 Mock override for ${pageName}: ${useMocks ? 'ENABLED' : 'DISABLED'}`)
  }


  // Initialize all mock data
  private initializeMockData(): Record<string, () => any> {
    const now = new Date().toISOString()
    return {
      // System endpoints
      '/api/system/health': () => ({
        status: 'healthy',
        version: '2.0.0',
        timestamp: new Date().toISOString(),
        services: {
          database: 'connected',
          redis: 'connected',
          llm: 'connected',
          workers: 'active'
        }
      }),

      '/api/system/metrics': () => ({
        cpu: { usage: 45 + Math.random() * 20 },
        memory: { percent: 60 + Math.random() * 15 },
        disk: { percent: 35 + Math.random() * 10 },
        api_calls_count: Math.floor(1000 + Math.random() * 500),
        average_response_time: 145 + Math.random() * 50,
        error_rate: Math.random() * 5
      }),

      '/api/system/agent-statistics': () => ({
        total: 12,
        active: 5,
        idle: 6,
        failed: 1,
        performance: {
          average_response_time: 1.2,
          success_rate: 0.94,
          error_rate: 0.06
        }
      }),

      // Agents endpoints
      // NOTE: These mocks should NEVER be used if API is working correctly
      // Agents endpoint - should use real API
      '/api/agents/': () => {
        console.warn('⚠️ USING MOCK AGENTS - This should not happen! API call failed.')
        return [
          {
            id: 1,
            name: 'MOCK: Data Analyst',
            type: 'analysis',
            status: 'active',
            description: 'Specialized in data analysis and insights',
            created_at: now,
            capabilities: ['data-processing', 'visualization', 'reporting']
          },
          {
            id: 2,
            name: 'MOCK: Content Creator',
            type: 'content',
            status: 'active',
            description: 'AI-powered content generation',
            created_at: now,
            capabilities: ['writing', 'editing', 'seo']
          },
          {
            id: 3,
            name: 'MOCK: Code Assistant',
            type: 'development',
            status: 'idle',
            description: 'Automated code review and generation',
            created_at: now,
            capabilities: ['code-review', 'refactoring', 'documentation']
          }
        ]
      },

      // Workflows endpoints
      '/api/workflows': () => [
        {
          id: 1,
          name: 'Data Processing Pipeline',
          status: 'running',
          description: 'Automated ETL pipeline',
          steps: 8,
          currentStep: 5,
          last_run: now
        },
        {
          id: 2,
          name: 'Content Generation',
          status: 'completed',
          description: 'Blog post creation workflow',
          steps: 5,
          currentStep: 5,
          last_run: now
        }
      ],

      // Documents endpoints
      '/api/documents': () => [
        {
          id: 1,
          name: 'Report_Q3.pdf',
          type: 'pdf',
          size: 2456789,
          status: 'processed',
          uploaded_at: now,
          tags: ['finance', 'quarterly']
        },
        {
          id: 2,
          name: 'Analysis.xlsx',
          type: 'excel',
          size: 567890,
          status: 'processing',
          uploaded_at: now,
          tags: ['data', 'analysis']
        }
      ],

      // Skills endpoints
      '/api/skills/': () => [
        { id: 1, name: 'Data Analysis', category: 'technical' },
        { id: 2, name: 'Report Writing', category: 'communication' },
        { id: 3, name: 'API Integration', category: 'technical' }
      ],

      // Analytics endpoints  
      '/api/analytics/performance/all-enhancements': () => ({
        enhancements: [
          {
            id: 1,
            name: 'Response Time Optimization',
            impact: 0.34,
            status: 'active',
            metrics: { before: 450, after: 297 }
          },
          {
            id: 2,
            name: 'Memory Optimization',
            impact: 0.28,
            status: 'active',
            metrics: { before: 78, after: 56 }
          },
          {
            id: 3,
            name: 'Query Optimization',
            impact: 0.45,
            status: 'testing',
            metrics: { before: 234, after: 128 }
          }
        ],
        overall_improvement: 0.35,
        last_updated: new Date().toISOString()
      }),

      // Legacy/Additional endpoints
      '/api/system/activities': () => {
        const activities = []
        const titles = [
          'Agent Processing Complete',
          'Workflow Execution Started',
          'Document Upload Successful',
          'System Health Check',
          'API Rate Limit Updated',
          'Memory Optimization Complete',
          'New Agent Configuration Applied',
          'Context Assembly Finished',
          'Knowledge Graph Updated',
          'Performance Metrics Calculated'
        ]
        for (let i = 0; i < 10; i++) {
          activities.push({
            id: `activity-${i}`,
            type: ['agent', 'workflow', 'document', 'system'][i % 4],
            title: titles[i],
            description: `Activity ${i + 1} completed successfully with ${85 + i}% efficiency`,
            timestamp: new Date(Date.now() - i * 3600000).toISOString(),
            status: ['success', 'warning', 'info', 'error'][i % 4]
          })
        }
        return activities
      },

      '/api/documents/analytics': () => ({
        totalDocuments: 156,
        totalSize: 45678901,
        averageSize: 292818,
        processingRate: 0.94,
        typeBreakdown: {
          pdf: 45,
          docx: 38,
          txt: 23,
          xlsx: 50
        }
      }),

      '/api/performance/analytics': () => ({
        responseTime: [120, 135, 115, 145, 130],
        throughput: [450, 480, 520, 490, 510],
        errorRate: [0.02, 0.01, 0.03, 0.02, 0.01],
        uptime: 99.95
      }),

      '/api/usage/analytics': () => ({
        apiCalls: 12456,
        uniqueUsers: 234,
        peakHour: '14:00',
        averageSessionDuration: 1820
      }),

      '/api/agents/analytics': () => ({
        totalAgents: 12,
        activeAgents: 8,
        completedTasks: 456,
        averageCompletionTime: 234
      }),

      '/api/metrics/all': () => ({
        system: { cpu: 45, memory: 67, disk: 32 },
        performance: { avgResponseTime: 234, throughput: 1567, errorRate: 0.02 },
        usage: { apiCalls: 12456, uniqueUsers: 234, activeAgents: 8 },
        health: { status: 'healthy', uptime: 99.9, lastCheck: new Date().toISOString() }
      }),

      // Enhanced Analytics Mock Data
      '/api/analytics/cost-analysis': () => ({
        average_cost_per_execution: 0.003,
        monthly_data: [
          { date: '2024-01-01', total_executions: 12450, total_cost: 37.35, cost_per_execution: 0.003 },
          { date: '2024-01-15', total_executions: 14200, total_cost: 42.60, cost_per_execution: 0.003 }
        ],
        cost_trend: 'decreasing',
        savings_this_month: 125.50
      }),

      // Context Engineering - NO MOCK DATA - Use real API only
      // Removed mock data to ensure real backend data is shown

      '/api/analytics/peak-usage': () => ({
        hourly_pattern: Array.from({ length: 24 }, (_, hour) => ({
          hour,
          usage_percent: Math.max(30, Math.sin(hour / 24 * Math.PI * 2) * 50 + 50),
          api_calls: Math.floor(Math.random() * 1000) + 100,
          active_agents: Math.floor(Math.random() * 10) + 1,
          category: hour >= 9 && hour <= 17 ? 'business_hours' : 'off_hours'
        })),
        peak_hours: [10, 11, 14, 15, 16],
        peak_period: '2PM-4PM',
        peak_usage_percent: 87,
        recommendation: 'Consider scaling resources during 2PM-4PM'
      }),

      '/api/analytics/bottlenecks': () => ({
        bottlenecks_detected: 3,
        bottlenecks: [
          {
            type: 'memory',
            severity: 'high',
            current_usage: 85,
            threshold: 80,
            description: 'Memory usage exceeding threshold',
            recommendation: 'Increase memory allocation or optimize usage',
            impact: 'Performance degradation'
          },
          {
            type: 'api_rate_limit',
            severity: 'medium',
            current_usage: 75,
            threshold: 80,
            description: 'Approaching API rate limits',
            recommendation: 'Implement request batching',
            impact: 'Potential request failures'
          },
          {
            type: 'database_connections',
            severity: 'low',
            current_usage: 65,
            threshold: 80,
            description: 'Database connection pool usage',
            recommendation: 'Monitor connection pool',
            impact: 'None currently'
          }
        ],
        overall_health: 'moderate',
        last_check: new Date().toISOString()
      }),

      '/api/analytics/predictive-alerts': () => ({
        predictive_alerts: [
          {
            type: 'resource_exhaustion',
            severity: 'warning',
            prediction: 'Memory will reach 90% in 2 hours',
            current_usage: 78,
            recommended_action: 'Scale resources or optimize memory usage',
            confidence: 0.85
          },
          {
            type: 'traffic_spike',
            severity: 'info',
            prediction: 'Expected 30% traffic increase tomorrow',
            current_usage: 60,
            recommended_action: 'Pre-scale infrastructure',
            confidence: 0.72
          }
        ],
        alerts_count: 2,
        forecast_period: '24 hours',
        confidence_level: 'high'
      }),

      '/api/analytics/agent-ranking': () => ({
        agent_rankings: [
          {
            agent_id: '1',
            name: 'Data Analyst',
            agent_type: 'analysis',
            performance_score: 95,
            success_rate: 0.98,
            avg_response_time: 120,
            tasks_completed: 450,
            uptime_percent: 99.5,
            rank: 1
          },
          {
            agent_id: '2',
            name: 'Code Generator',
            agent_type: 'coding',
            performance_score: 92,
            success_rate: 0.95,
            avg_response_time: 150,
            tasks_completed: 380,
            uptime_percent: 98.8,
            rank: 2
          },
          {
            agent_id: '3',
            name: 'Document Processor',
            agent_type: 'document',
            performance_score: 88,
            success_rate: 0.92,
            avg_response_time: 200,
            tasks_completed: 320,
            uptime_percent: 97.5,
            rank: 3
          }
        ],
        total_agents: 8,
        top_performer: { name: 'Data Analyst', score: 95 },
        average_score: 85.3
      }),

      '/api/analytics/sla-compliance': () => ({
        overall_compliance: 94.5,
        overall_status: 'compliant',
        status_color: 'green',
        sla_metrics: {
          response_time: {
            sla_target: 200,
            current_average: 145,
            compliance_rate: 98.5,
            status: 'compliant'
          },
          uptime: {
            sla_target: 99.5,
            current_uptime: 99.8,
            compliance_rate: 100,
            status: 'compliant'
          },
          error_rate: {
            sla_target: 0.05,
            current_rate: 0.02,
            compliance_rate: 100,
            status: 'compliant'
          },
          throughput: {
            sla_target: 1000,
            current_average: 1567,
            compliance_rate: 100,
            status: 'compliant'
          }
        },
        reporting_period: 'Last 30 days',
        next_review: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString()
      }),

      '/api/analytics/success-rate': () => ({
        current_rate: 94.5,
        trend: 'increasing',
        historical: Array.from({ length: 30 }, (_, i) => ({
          date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString(),
          rate: 90 + Math.random() * 8
        }))
      }),

      '/api/analytics/completion-time': () => ({
        average_time: 234,
        median_time: 210,
        p95_time: 450,
        p99_time: 890,
        by_type: {
          analysis: 120,
          generation: 250,
          processing: 180,
          optimization: 340
        }
      }),

      '/api/analytics/load-trend': () => ({
        current_load: 67,
        trend: 'stable',
        forecast_24h: 72,
        historical: Array.from({ length: 24 }, (_, i) => ({
          hour: i,
          load: 50 + Math.random() * 40
        }))
      }),

      '/api/analytics/error-rate': () => ({
        overall_rate: 0.02,
        by_type: {
          timeout: 0.008,
          validation: 0.005,
          api_error: 0.004,
          system_error: 0.003
        },
        trend: 'decreasing',
        critical_errors: 2
      }),

      // Agent specific endpoints
      '/api/agents/1/logs': () => [
        { timestamp: new Date().toISOString(), level: 'info', message: 'Agent started' },
        { timestamp: new Date().toISOString(), level: 'info', message: 'Processing task' }
      ],
      '/api/agents/1/stats': () => ({
        status: 'active',
        performance: 92,
        tasksCompleted: 45,
        uptime: 3600
      }),

      // Add pattern matching for dynamic endpoints
      // This is a fallback for any agent ID
      'default': () => {
        console.log('Using default mock data')
        return {}
      }
    }
  }

  // Override getMockDataForEndpoint to handle dynamic paths
  public getMockDataForEndpoint(endpoint: string): any {
    // Strip query parameters for matching
    const endpointWithoutQuery = endpoint.split('?')[0]

    // Try exact match first (without query params)
    const mockFunction = this.mockData[endpointWithoutQuery]
    if (mockFunction) {
      return typeof mockFunction === 'function' ? mockFunction() : mockFunction
    }

    // Try with full endpoint (with query params)
    const fullMockFunction = this.mockData[endpoint]
    if (fullMockFunction) {
      return typeof fullMockFunction === 'function' ? fullMockFunction() : fullMockFunction
    }

    // Handle dynamic agent endpoints
    if (endpoint.match(/^\/api\/agents\/\d+\/logs/)) {
      return [
        { timestamp: new Date().toISOString(), level: 'info', message: 'Agent activity log' }
      ]
    }
    if (endpoint.match(/^\/api\/agents\/\d+\/stats/)) {
      return { status: 'active', performance: 85 }
    }
    if (endpoint.match(/^\/api\/agents\/\d+\/(start|stop)/)) {
      return { success: true, message: 'Operation completed' }
    }

    // Handle other query param endpoints
    if (endpointWithoutQuery === '/api/performance/analytics') {
      return this.mockData['/api/performance/analytics']()
    }
    if (endpointWithoutQuery === '/api/usage/analytics') {
      return this.mockData['/api/usage/analytics']()
    }
    if (endpointWithoutQuery === '/api/agents/analytics') {
      return this.mockData['/api/agents/analytics']()
    }

    // Return generic mock if no specific mock exists
    console.warn(`No mock data defined for ${endpoint}, returning empty array for safety`)
    return [] // Return empty array instead of empty object to prevent .slice errors
  }

  /** Public base URL for building backend requests (e.g. skills API that uses raw fetch). */
  getBaseUrl(): string {
    return (this.baseUrl || '').replace(/\/$/, '')
  }

  /** Auth headers (Clerk JWT + workspace) for use with raw fetch. */
  async getAuthHeaders(): Promise<Record<string, string>> {
    const headers: Record<string, string> = {}
    if (typeof window !== 'undefined') {
      const workspaceId = localStorage.getItem('last_active_workspace') || localStorage.getItem('last_active_org')
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

    console.log('🔍 API Call:', { url, method: options.method || 'GET' })

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
      console.log('🔐 Added Clerk JWT to request')
    } else {
      console.warn('⚠️ No Clerk token available - request may fail')
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

      const data = await response.json()
      if (process.env.NODE_ENV !== 'production') console.log('✅ API Success:', endpoint, 'Data type:', Array.isArray(data) ? `array[${data.length}]` : typeof data)

      return data
    } catch (error: any) {
      // Check if we should use mock fallback
      if (this.shouldUseMock(endpoint)) {
        if (this.mockConfig.logMockUsage) {
          console.warn(`⚠️ API failed for ${endpoint}, falling back to mock data`, error.message)
        }

        const mockData = this.getMockDataForEndpoint(endpoint)

        if (this.mockConfig.logMockUsage) {
          console.log('🎭 Using mock data for:', endpoint, mockData)
        }

        // Emit event for UI to show mock indicator
        if (typeof window !== 'undefined') {
          window.dispatchEvent(new CustomEvent('mock-used', {
            detail: { endpoint, data: mockData }
          }))
        }

        // Add slight delay to simulate network
        await new Promise(resolve => setTimeout(resolve, 100))

        return mockData as T
      }

      // If mocks are disabled, throw the original error
      if (process.env.NODE_ENV !== 'production') console.error('🚨 API Failed:', endpoint, error.message)
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

  async getWorkflowExecutionResults(executionId: string) {
    return this.request(`/api/workflows/executions/${executionId}/results`)
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

  // ===== BUSINESS GRAPH ENDPOINTS =====

  async importBusinessGraph(file: File, merge: boolean = false) {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('merge', String(merge))

    // Use raw fetch (same pattern as uploadDocument) — this.request() has
    // Content-Type issues with FormData and may route through Next.js proxy.
    const headers: any = { ...this.defaultHeaders }
    delete headers['Content-Type'] // Let browser set multipart/form-data with boundary

    if (typeof window !== 'undefined') {
      const workspaceId = localStorage.getItem('last_active_workspace') || localStorage.getItem('last_active_org')
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
      const workspaceId = localStorage.getItem('last_active_workspace') || localStorage.getItem('last_active_org')
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
      const workspaceId = localStorage.getItem('last_active_workspace') || localStorage.getItem('last_active_org')
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
      const workspaceId = localStorage.getItem('last_active_workspace') || localStorage.getItem('last_active_org')
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

  async syncOpenRouterCache() {
    return this.request('/api/openrouter/sync', { method: 'POST' })
  }

  // ===== CHATBOT ENDPOINTS =====
  async sendChatbotQuery(params: {
    query: string
    context?: any
    sessionId?: string
    provider?: string
    model?: string
  }) {
    const payload: Record<string, any> = {
      query: params.query,
      context: params.context,
      session_id: params.sessionId,
      provider: params.provider,
      model: params.model
    }

    Object.keys(payload).forEach((key) => {
      if (payload[key] === undefined || payload[key] === null) {
        delete payload[key]
      }
    })

    return this.request('/api/chatbot/query', {
      method: 'POST',
      body: JSON.stringify(payload)
    })
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
    return this.request(`/api/agents/${id}/start`, { method: 'POST' })
  }

  async stopAgent(id: string) {
    return this.request(`/api/agents/${id}/stop`, { method: 'POST' })
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


  async sendChatMessage(message: string, context?: any) {
    return this.sendChatbotQuery({ query: message, context })
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

  // Enhanced Analytics Methods
  // ===== FIELD THEORY ENDPOINTS (All Working ✅) =====
  async getFieldTheoryHealth() {
    return this.request('/api/field-theory/health')
  }

  async updateFieldContext(data: { session_id: string, context_data: any, field_type?: string }) {
    return this.request('/api/field-theory/fields/update', {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  async propagateField(data: { source: string, targets?: string[], propagation_steps?: number }) {
    return this.request('/api/field-theory/fields/propagate', {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  async modelFieldInteractions(data: { task_id: number, user_id: number, similarity_threshold?: number }) {
    return this.request('/api/field-theory/fields/interactions', {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  async manageDynamicFields(data: { session_id: string, context?: any }) {
    return this.request('/api/field-theory/fields/dynamic', {
      method: 'POST',
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

  // ===== MEMORY ENDPOINTS (All Working ✅) =====
  async storeMemory(data: { session_id: string, content: any, memory_type?: string, importance?: number, tags?: string[] }) {
    return this.request('/api/v1/memory/store', {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  async retrieveMemory(sessionId: string, query?: string, maxItems: number = 20, includeAugmented: boolean = true) {
    const params = new URLSearchParams()
    if (query) params.append('query', query)
    params.append('max_items', maxItems.toString())
    params.append('include_augmented', includeAugmented.toString())

    return this.request(`/api/v1/memory/retrieve/${sessionId}?${params.toString()}`)
  }

  async consolidateMemory(data?: { session_id?: string, memory_level?: string, strategy?: string }) {
    return this.request('/api/v1/memory/consolidate', {
      method: 'POST',
      body: JSON.stringify(data || {})
    })
  }

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

  // ===== WORKSPACE TASK ENDPOINTS (PRD-56 Phase 2) =====
  async submitWorkspaceTask(data: {
    steps: Array<{
      action: string
      command?: string
      repo?: string
      branch?: string
      path?: string
      content?: string
      cwd?: string
      timeout?: number
      description?: string
    }>
    priority?: string
    timeout_seconds?: number
  }) {
    return this.request('/api/tasks/submit', {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  async listWorkspaceTasks(params?: { status?: string; limit?: number; offset?: number }) {
    const qs = new URLSearchParams()
    if (params?.status) qs.set('status', params.status)
    if (params?.limit) qs.set('limit', String(params.limit))
    if (params?.offset) qs.set('offset', String(params.offset))
    const query = qs.toString()
    return this.request(`/api/tasks${query ? '?' + query : ''}`)
  }

  async getWorkspaceTask(taskId: string) {
    return this.request(`/api/tasks/${taskId}`)
  }

  async cancelWorkspaceTask(taskId: string) {
    return this.request(`/api/tasks/${taskId}/cancel`, { method: 'POST' })
  }

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
}

export const apiClient = new ApiClient()
export default apiClient
