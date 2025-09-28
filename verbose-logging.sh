#!/bin/bash

# Create log files with generous permissions
touch /root/automatos-ai/verbose-frontend.log
chmod 666 /root/automatos-ai/verbose-frontend.log
touch /root/automatos-ai/api-calls.log
chmod 666 /root/automatos-ai/api-calls.log

# Update the API client to write detailed logs to a file
cat > /root/automatos-ai/frontend/lib/api-client.ts << 'EOF'
/**
 * API Client with file logging
 */

import fs from 'fs';
import path from 'path';

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

function logToFile(message: string) {
  if (typeof window === 'undefined') {
    // Only log on server side
    try {
      fs.appendFileSync('/root/automatos-ai/api-calls.log', );
    } catch (err) {
      console.error('Failed to write to log file:', err);
    }
  }
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
    
    const initMessage = ;
    console.log('🚀', initMessage);
    logToFile(initMessage);
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = this.baseUrl + endpoint
    
    const requestMessage = ;
    console.log('🔍', requestMessage);
    logToFile(requestMessage);
    
    const config: RequestInit = {
      ...options,
      headers: {
        ...this.defaultHeaders,
        ...options.headers,
      },
    }

    try {
      const response = await fetch(url, config)
      
      if (!response.ok) {
        const errorMessage = ;
        console.error('❌', errorMessage);
        logToFile();
        throw new Error()
      }

      const data = await response.json()
      const successMessage = ;
      console.log('✅', successMessage);
      logToFile(successMessage);
      
      return data
    } catch (error: any) {
      const failMessage = ;
      console.error('🚨', failMessage);
      logToFile();
      throw error
    }
  }

  // Simple methods for API tests
  async getSystemHealth() {
    return this.request('/api/system/health')
  }

  async getAgentStats(id?: string) {
    return this.request(id ?  : '/api/system/agent-statistics')
  }
  
  async getAgentTypes() {
    return this.request('/api/system/agent-types')
  }

  async getAgents() {
    logToFile('Calling getAgents() method');
    return this.request('/api/agents/')
  }
  
  async getDocuments() {
    logToFile('Calling getDocuments() method');
    return this.request('/api/documents/')
  }
  
  async getWorkflows() {
    logToFile('Calling getWorkflows() method');
    return this.request('/api/workflows/')
  }
  
  async getChatbotSuggestions() {
    return this.request('/api/chatbot/suggestions?page=dashboard&user_role=user')
  }
  
  async testSystemRoute() {
    return this.request('/api/system/test-route')
  }
}

export const apiClient = new ApiClient()
EOF

# Rebuild the frontend
cd /root/automatos-ai/frontend
pkill -9 -f next || true 
sleep 2
netstat -tulpn | grep 3000
rm -rf .next node_modules/.cache
npm run build && npm start > /root/automatos-ai/verbose-frontend.log 2>&1 &
sleep 5

echo ==============================================================
echo Verbose logging enabled! Check these files:
echo - /root/automatos-ai/verbose-frontend.log - Frontend logs
echo - /root/automatos-ai/api-calls.log - API call logs
echo ==============================================================
echo Recent API logs:
tail -n 10 /root/automatos-ai/api-calls.log || echo No API logs yet
