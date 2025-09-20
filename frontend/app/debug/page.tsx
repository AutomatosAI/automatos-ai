'use client'

import { useEffect, useState } from 'react'
import { useAgents } from '@/hooks/use-agent-api'
import { apiClient } from '@/lib/api-client'

export default function DebugPage() {
  const [directData, setDirectData] = useState<any>(null)
  const [directError, setDirectError] = useState<any>(null)
  const { data: hookData, isLoading, error: hookError } = useAgents()

  useEffect(() => {
    // Direct API call to compare
    apiClient.getAgents()
      .then(data => {
        console.log('Direct API call success:', data)
        setDirectData(data)
      })
      .catch(err => {
        console.error('Direct API call error:', err)
        setDirectError(err.message)
      })
  }, [])

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-4">API Debug Page</h1>
      
      <div className="grid grid-cols-2 gap-4">
        <div className="border p-4 rounded">
          <h2 className="font-bold mb-2">Direct API Call</h2>
          <p>Status: {directData ? 'Success' : directError ? 'Error' : 'Loading...'}</p>
          {directData && (
            <div>
              <p>Data type: {typeof directData}</p>
              <p>Is Array: {Array.isArray(directData) ? 'Yes' : 'No'}</p>
              <p>Length: {Array.isArray(directData) ? directData.length : 'N/A'}</p>
              <pre className="text-xs mt-2 overflow-auto max-h-40">
                {JSON.stringify(directData, null, 2).slice(0, 500)}
              </pre>
            </div>
          )}
          {directError && <p className="text-red-500">{directError}</p>}
        </div>
        
        <div className="border p-4 rounded">
          <h2 className="font-bold mb-2">React Query Hook</h2>
          <p>Status: {isLoading ? 'Loading' : hookError ? 'Error' : 'Success'}</p>
          <p>isLoading: {isLoading ? 'true' : 'false'}</p>
          {hookData && (
            <div>
              <p>Data type: {typeof hookData}</p>
              <p>Is Array: {Array.isArray(hookData) ? 'Yes' : 'No'}</p>
              <p>Length: {Array.isArray(hookData) ? hookData.length : 'N/A'}</p>
              <pre className="text-xs mt-2 overflow-auto max-h-40">
                {JSON.stringify(hookData, null, 2).slice(0, 500)}
              </pre>
            </div>
          )}
          {hookError && <p className="text-red-500">{String(hookError)}</p>}
        </div>
      </div>
    </div>
  )
}
