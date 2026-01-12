
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
  'documents': false,        // false = REAL APIs ✅ | true = MOCK data ❌
  'analytics': false,        // ✅ Use real APIs - working endpoints
  'context': false,          // ✅ Use real APIs - all context endpoints working
  'memory': false,           // ✅ Use real APIs - all memory endpoints working
  'field-theory': false,     // ✅ Use real APIs - all field theory endpoints working
  'multi-agent': false,      // ✅ Use real APIs - coordination/reasoning working
  'orchestrator': false,     // ✅ Use real APIs - task submission working

  // Settings/Admin Pages
  'settings': false,         // ✅ Use real APIs - credentials system ready
  'tools': false,            // ✅ Use real APIs - MCP tools endpoints working
  'credentials': false,      // ✅ Use real APIs - credentials system ready

  // Testing/Development
  'test': true,              // 🧪 Always use mocks for testing
  'demo': true,              // 🧪 Always use mocks for demos
}

class ApiClient {
  private baseUrl: string
  private defaultHeaders: Record<string, string>
  private mockConfig: MockConfig
  private mockData: Record<string, () => any>
  private currentPage: string = '' // Track which page is making requests

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

    // Get API key from environment variables
    const apiKey = typeof window !== 'undefined'
      ? (window as any).NEXT_PUBLIC_API_KEY || process.env.NEXT_PUBLIC_API_KEY
      : process.env.NEXT_PUBLIC_API_KEY

    this.defaultHeaders = {
      'Content-Type': 'application/json',
      ...(apiKey && { 'x-api-key': apiKey }), // Add API key if available
    }

    // Initialize mock config
    this.mockConfig = this.loadMockConfig()
    this.mockData = this.initializeMockData()

    // Expose mock control to window for easy debugging
    if (typeof window !== 'undefined') {
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

    console.log('🚀 API Client initialized')
    console.log(`📍 Base URL: ${this.baseUrl || 'relative URLs (Next.js)'}`)
    console.log(`📍 NEXT_PUBLIC_API_URL env: ${process.env.NEXT_PUBLIC_API_URL || 'NOT SET'}`)
    console.log(`🔐 API Key: ${apiKey ? '✅ Configured' : '❌ Missing'}`)
    if (this.mockConfig.enabled) {
      console.warn('⚠️  Mock mode is ENABLED - Disable for production!')
    } else {
      console.log('✅ Real API mode enabled')
    }
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
    const mockStatus = PAGE_MOCK_CONFIG[this.currentPage] ? 'MOCKS ON' : 'REAL APIs'
    console.log(`📄 Page: ${pageName} → ${mockStatus}`)
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
    console.log(`🔧 Mock override for ${pageName}: ${useMocks ? 'ENABLED' : 'DISABLED'}`)
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

      // Tools endpoints
      '/api/tools': () => [
        {
          id: 1,
          name: 'API Tester',
          type: 'debugging',
          status: 'available',
          description: 'Test API endpoints with custom payloads',
          icon: 'bug'
        },
        {
          id: 2,
          name: 'Log Analyzer',
          type: 'monitoring',
          status: 'available',
          description: 'Analyze and search through system logs',
          icon: 'file-text'
        },
        {
          id: 3,
          name: 'Performance Profiler',
          type: 'optimization',
          status: 'maintenance',
          description: 'Profile system performance and identify bottlenecks',
          icon: 'activity'
        },
        {
          id: 4,
          name: 'Database Explorer',
          type: 'data',
          status: 'available',
          description: 'Browse and query database tables',
          icon: 'database'
        }
      ],
      '/api/tools/health': () => ({
        status: 'healthy',
        available: 3,
        maintenance: 1,
        total: 4,
        last_check: new Date().toISOString()
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

  async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`

    console.log('🔍 API Call:', { url, method: options.method || 'GET' })

    // Auto-stringify body if it's an object and not FormData
    let body = options.body
    if (body && typeof body === 'object' && !(body instanceof FormData)) {
      body = JSON.stringify(body)
    }

    const config: RequestInit = {
      ...options,
      body,
      headers: {
        ...this.defaultHeaders,
        ...options.headers,
      },
    }

    try {
      const response = await fetch(url, {
        ...config,
        redirect: 'follow' // Follow redirects automatically
      })

      if (!response.ok) {
        console.error('❌ API Error:', response.status, response.statusText)
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const data = await response.json()
      console.log('✅ API Success:', endpoint, 'Data type:', Array.isArray(data) ? `array[${data.length}]` : typeof data)

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
      console.error('🚨 API Failed:', endpoint, error.message)
      throw error
    }
  }

  // ===== SYSTEM ENDPOINTS =====
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
      body: JSON.stringify({ value })
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
    return this.request('/api/workflows/active')
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

  /** Health check */
  async codegraphHealth() {
    return this.request('/api/code-graph/health')
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

    // Use fetch directly for file upload (don't use this.request which sets Content-Type)
    const headers: any = { ...this.defaultHeaders }
    delete headers['Content-Type'] // Let browser set multipart/form-data with boundary

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

  // ===== TOOLS ENDPOINTS =====
  async getTools() {
    return this.request('/api/tools')
  }

  async getTool(id: string) {
    return this.request(`/api/tools/${id}`)
  }

  async getToolsHealth() {
    return this.request('/api/tools/health')
  }

  async executeToolAction(toolId: string, action: string, data?: any) {
    return this.request(`/api/tools/${toolId}/${action}`, {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  // ===== MCP TOOLS ENDPOINTS (Phase 3) =====
  async getMCPTools(params?: { status?: string; category?: string; provider?: string; search?: string; skip?: number; limit?: number }) {
    const queryParams = new URLSearchParams()
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) queryParams.append(key, String(value))
      })
    }
    const url = queryParams.toString() ? `/api/mcp-tools/?${queryParams}` : '/api/mcp-tools/'
    return this.request(url)
  }

  async getMCPTool(id: number) {
    return this.request(`/api/mcp-tools/${id}`)
  }

  async createMCPTool(data: any) {
    return this.request('/api/mcp-tools/', {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  async updateMCPTool(id: number, data: any) {
    return this.request(`/api/mcp-tools/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data)
    })
  }

  async deleteMCPTool(id: number) {
    return this.request(`/api/mcp-tools/${id}`, {
      method: 'DELETE'
    })
  }

  async testMCPToolConnection(id: number, params?: any) {
    return this.request(`/api/mcp-tools/${id}/test`, {
      method: 'POST',
      body: JSON.stringify(params || {})
    })
  }

  async getMCPToolCategories() {
    return this.request('/api/mcp-tools/categories/list')
  }

  async getMCPToolsStats() {
    return this.request('/api/mcp-tools/stats/summary')
  }

  // Agent-Tool Assignment Endpoints
  async getMCPToolAssignments(enabledOnly: boolean = true) {
    return this.request(`/api/mcp-tools/assignments?enabled_only=${enabledOnly}`)
  }

  async getAgentTools(agentId: number, enabledOnly: boolean = true) {
    return this.request(`/api/mcp-tools/agents/${agentId}/tools?enabled_only=${enabledOnly}`)
  }

  async assignToolToAgent(agentId: number, toolId: number, data?: { enabled?: boolean; permissions?: any; configuration?: any }) {
    return this.request(`/api/mcp-tools/agents/${agentId}/tools/${toolId}`, {
      method: 'POST',
      body: JSON.stringify(data || { tool_id: toolId, enabled: true, permissions: {}, configuration: {} })
    })
  }

  async removeToolFromAgent(agentId: number, toolId: number) {
    return this.request(`/api/mcp-tools/agents/${agentId}/tools/${toolId}`, {
      method: 'DELETE'
    })
  }

  async updateToolPermissions(agentId: number, toolId: number, permissions: any) {
    return this.request(`/api/mcp-tools/agents/${agentId}/tools/${toolId}/permissions`, {
      method: 'PUT',
      body: JSON.stringify(permissions)
    })
  }

  async getToolUsageLogs(params?: { tool_id?: number; agent_id?: number; success_only?: boolean; skip?: number; limit?: number }) {
    const queryParams = new URLSearchParams()
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) queryParams.append(key, String(value))
      })
    }
    const url = queryParams.toString() ? `/api/mcp-tools/usage/logs?${queryParams}` : '/api/mcp-tools/usage/logs'
    return this.request(url)
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
