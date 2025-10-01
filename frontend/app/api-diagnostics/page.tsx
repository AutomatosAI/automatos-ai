'use client'

import React, { useState, useEffect } from 'react'
import { apiClient } from '@/lib/api-client'

export default function ApiDiagnosticsPage() {
  const [endpoints, setEndpoints] = useState([
    { name: 'System Health', endpoint: '/system/health', status: 'pending' },
    { name: 'Agents', endpoint: '/agents', status: 'pending' },
    { name: 'Documents', endpoint: '/documents', status: 'pending' },
    { name: 'Workflows', endpoint: '/workflows', status: 'pending' }
  ])
  
  const [loading, setLoading] = useState(false)

  const testEndpoint = async (endpoint: string, index: number) => {
    setEndpoints(prev => {
      const updated = [...prev]
      updated[index] = { ...updated[index], status: 'loading' }
      return updated
    })

    try {
      await apiClient.getSystemHealth()
      setEndpoints(prev => {
        const updated = [...prev]
        updated[index] = { ...updated[index], status: 'success' }
        return updated
      })
    } catch (error) {
      setEndpoints(prev => {
        const updated = [...prev]
        updated[index] = { 
          ...updated[index], 
          status: 'error',
          error: error instanceof Error ? error.message : 'Unknown error'
        }
        return updated
      })
    }
  }

  const testAll = async () => {
    setLoading(true)
    
    for (let i = 0; i < endpoints.length; i++) {
      await testEndpoint(endpoints[i].endpoint, i)
    }
    
    setLoading(false)
  }

  return (
    <div className="container mx-auto py-8 px-4">
      <h1 className="text-2xl font-bold mb-6">API Diagnostics</h1>
      
      <div className="mb-6">
        <button 
          onClick={testAll}
          disabled={loading}
          className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50"
        >
          {loading ? 'Testing...' : 'Test All Endpoints'}
        </button>
      </div>
      
      <div className="space-y-4">
        {endpoints.map((endpoint, i) => (
          <div key={i} className="border rounded-lg p-4">
            <div className="flex justify-between items-center">
              <div>
                <h3 className="font-medium">{endpoint.name}</h3>
                <p className="text-sm text-muted-foreground">{endpoint.endpoint}</p>
              </div>
              <div className="flex items-center">
                <span className="mr-2">
                  {endpoint.status === 'pending' && 'Not tested'}
                  {endpoint.status === 'loading' && 'Testing...'}
                  {endpoint.status === 'success' && '✅ Success'}
                  {endpoint.status === 'error' && '❌ Failed'}
                </span>
                <button
                  onClick={() => testEndpoint(endpoint.endpoint, i)}
                  disabled={endpoint.status === 'loading' || loading}
                  className="px-3 py-1 bg-secondary text-secondary-foreground text-sm rounded hover:bg-secondary/80 disabled:opacity-50"
                >
                  Test
                </button>
              </div>
            </div>
            
            {endpoint.status === 'error' && (
              <div className="mt-2 p-2 bg-destructive/10 text-destructive rounded text-sm">
                {endpoint.error}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
