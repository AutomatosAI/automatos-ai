/**
 * API Client with enhanced logging and error handling
 */

const LOG_PREFIX = '🔌 [API]';

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
  private logLevel: 'debug' | 'info' | 'warn' | 'error' = 'debug';

  constructor() {
    // Get API URL from environment variables or use fallbacks
    this.baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://206.81.0.227:8000'
    const apiKey = process.env.NEXT_PUBLIC_API_KEY || 'test_api_key_for_backend_validation_2025'
    
    this.defaultHeaders = {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
    }
    
    console.log()
    console.log()
    console.log()
  }

  private log(level: 'debug' | 'info' | 'warn' | 'error', message: string, data?: any) {
    const levels = {
      debug: 0,
      info: 1,
      warn: 2,
      error: 3
    };

    if (levels[level] >= levels[this.logLevel]) {
      const timestamp = new Date().toISOString();
      const logPrefix = ;
      
      if (data) {
        console[level](, data);
      } else {
        console[level]();
      }
      
      // If we're in the browser, also write to a dedicated log div if it exists
      if (typeof window !== 'undefined') {
        try {
          const logContainer = document.getElementById('api-log-container');
          if (logContainer) {
            const logEntry = document.createElement('div');
            logEntry.className = ;
            logEntry.innerHTML = ;
            logContainer.appendChild(logEntry);
            logContainer.scrollTop = logContainer.scrollHeight;
          }
        } catch (e) {
          // Ignore errors in log DOM manipulation
        }
      }
    }
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = 
    
    this.log('info', );
    
    const config: RequestInit = {
      ...options,
      headers: {
        ...this.defaultHeaders,
        ...options.headers,
      },
    }

    try {
      this.log('debug', );
      const response = await fetch(url, config)
      
      if (!response.ok) {
        const errorText = await response.text().catch(() => 'No error text available');
        this.log('error', , {
          endpoint,
          status: response.status,
          statusText: response.statusText,
          errorText: errorText.substring(0, 500) // Truncate long error messages
        });
        
        throw new Error();
      }

      const contentType = response.headers.get('content-type');
      let data;
      
      if (contentType && contentType.includes('application/json')) {
        data = await response.json();
      } else {
        const text = await response.text();
        this.log('warn', , {
          endpoint, 
          contentType,
          textSample: text.substring(0, 100)
        });
        try {
          // Try to parse as JSON anyway
          data = JSON.parse(text);
        } catch (e) {
          this.log('error', , {
            endpoint,
            error: e
          });
          // Return text as is
          return text as unknown as T;
        }
      }
      
      this.log('info', , {
        dataType: Array.isArray(data) ?  : typeof data,
        dataSample: JSON.stringify(data).substring(0, 100)
      });
      
      return data;
    } catch (error: any) {
      this.log('error', , {
        error: error.message,
        stack: error.stack
      });
      throw error;
    }
  }

  // Keep the rest of the methods the same as the original api-client.ts
  // ...
