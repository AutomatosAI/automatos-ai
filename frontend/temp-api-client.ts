/**
 * API Client - Only calls endpoints that actually exist based on test results
 * Base URL: http://206.81.0.227:8000
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
    this.baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://206.81.0.227:8000'
    const apiKey = process.env.NEXT_PUBLIC_API_KEY || 'test_api_key_for_backend_validation_2025'
    this.defaultHeaders = {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
    }
    
    console.log('🚀 API Client initialized with baseUrl:', this.baseUrl, 'and API key:', apiKey ? '***' + apiKey.slice(-4) : 'none')
  }
