/**
 * API Client - Only calls endpoints that actually exist based on test results
 * Base URL: Configured via NEXT_PUBLIC_API_URL environment variable (set by Docker Compose or .env.local)
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

class ApiClient {
  private baseUrl: string
  private defaultHeaders: Record<string, string>

  constructor() {
    this.baseUrl = process.env.NEXT_PUBLIC_API_URL || ''
    const apiKey = process.env.NEXT_PUBLIC_API_KEY || 'test_api_key_for_backend_validation_2025'
    this.defaultHeaders = {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
    }
    
    console.log('🚀 API Client initialized with baseUrl:', this.baseUrl, 'and API key:', apiKey ? '***' + apiKey.slice(-4) : 'none')
  }
