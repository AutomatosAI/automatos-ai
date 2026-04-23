
/**
 * API Configuration for Enhanced Automotas AI Platform
 */

// Environment configuration
export const API_CONFIG = {
  BASE_URL: process.env.NEXT_PUBLIC_API_URL || '',
  TIMEOUT: 10000, // 10 seconds
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000, // 1 second
}

// API endpoints
export const API_ENDPOINTS = {
  // System
  SYSTEM_HEALTH: '/api/system/health',
  SYSTEM_METRICS: '/api/system/metrics',
  SYSTEM_ACTIVITIES: '/api/system/activities',
  
  // Agents
  AGENTS: '/api/agents',
  AGENT_DETAILS: (id: string) => `/api/agents/${id}`,
  AGENT_LOGS: (id: string) => `/api/agents/${id}/logs`,
  AGENT_STATS: (id: string) => `/api/agents/${id}/stats`,
  
  // Documents  
  DOCUMENTS: '/api/documents',
  DOCUMENT_DETAILS: (id: string) => `/api/documents/${id}`,
  DOCUMENT_UPLOAD: '/api/documents/upload',
  DOCUMENT_ANALYTICS: '/api/documents/analytics',
  DOCUMENT_PROCESSING: (id: string) => `/api/documents/${id}/processing`,
  
  // Workflows
  WORKFLOWS: '/api/workflows',
  WORKFLOW_DETAILS: (id: string) => `/api/workflows/${id}`,
  WORKFLOW_RUN: (id: string) => `/api/workflows/${id}/run`,
  WORKFLOW_STATUS: (id: string) => `/api/workflows/${id}/status`,
  WORKFLOW_LOGS: (id: string) => `/api/workflows/${id}/logs`,
  WORKFLOW_TEMPLATES: '/api/workflows/templates',
  
  // Analytics
  PERFORMANCE_ANALYTICS: '/api/analytics/performance',
  USAGE_ANALYTICS: '/api/analytics/usage',
  DOCUMENT_ANALYTICS_TIME: '/api/analytics/documents',
  ERROR_ANALYTICS: '/api/analytics/errors',
  
  // Settings
  SETTINGS: '/api/settings',
  USER_PREFERENCES: '/api/settings/user',
  SYSTEM_BACKUP: '/api/settings/backup',
  SYSTEM_RESTORE: '/api/settings/restore',
  SYSTEM_LOGS: '/api/settings/logs',
}

// Request interceptor configuration
export const REQUEST_INTERCEPTORS = {
  // Add auth headers if needed
  addAuthHeaders: (headers: Headers) => {
    const token = localStorage.getItem('auth_token')
    if (token) {
      headers.set('Authorization', `Bearer ${token}`)
    }
    return headers
  },
  
  // Add request ID for tracking
  addRequestId: (headers: Headers) => {
    headers.set('X-Request-ID', generateRequestId())
    return headers
  },
}

// Response interceptor configuration
export const RESPONSE_INTERCEPTORS = {
  // Handle common error responses
  handleErrors: (response: Response) => {
    if (response.status === 401) {
      // Handle unauthorized - redirect to login
      window.location.href = '/login'
      return
    }
    
    if (response.status >= 500) {
      // Handle server errors
      console.error('Server error:', response.status, response.statusText)
    }
    
    return response
  },
}

// Utility functions
function generateRequestId(): string {
  return Math.random().toString(36).substring(2) + Date.now().toString(36)
}

// WebSocket configuration for real-time updates
// Derive WebSocket URL from API URL if WS_URL not explicitly set
const getWebSocketUrl = () => {
  if (process.env.NEXT_PUBLIC_WS_URL) {
    return process.env.NEXT_PUBLIC_WS_URL
  }
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || ''
  if (apiUrl) {
    return apiUrl.replace(/^https?/, 'ws') + '/ws'
  }
  return ''
}

export const WEBSOCKET_CONFIG = {
  URL: getWebSocketUrl(),
  RECONNECT_INTERVAL: 5000, // 5 seconds
  MAX_RECONNECT_ATTEMPTS: 5,
  PING_INTERVAL: 30000, // 30 seconds
}

// Feature flags
export const FEATURE_FLAGS = {
  REAL_TIME_UPDATES: true,
  ADVANCED_ANALYTICS: true,
  WORKFLOW_BUILDER: true,
  AGENT_TEMPLATES: true,
  DOCUMENT_PROCESSING: true,
  THEME_SWITCHING: true,
}

// Performance monitoring
export const PERFORMANCE_CONFIG = {
  ENABLE_METRICS: process.env.NODE_ENV === 'production',
  SLOW_QUERY_THRESHOLD: 2000, // 2 seconds
  ERROR_TRACKING: true,
  USER_ANALYTICS: false, // Respect privacy
}
