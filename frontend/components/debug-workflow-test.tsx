
'use client'

import { useState } from 'react'

export function DebugWorkflowTest() {
  const [status, setStatus] = useState('')
  const [result, setResult] = useState('')

  const testWorkflowCreation = async () => {
    setStatus('Testing...')
    setResult('')
    
    const testWorkflow = {
      name: "Debug Test Workflow",
      description: "Testing workflow creation from React component",
      category: "automation",
      priority: "medium",
      config: {
        autoRun: false,
        timeout: 3600,
        retryAttempts: 3,
        parallelExecution: false
      },
      steps: [
        {
          id: 'start',
          type: 'trigger',
          name: 'Start',
          config: {}
        }
      ],
      agents: [],
      tags: []
    }

    try {
      console.log('🔄 Testing workflow creation from React component...')
      console.log('📝 Environment check:', {
        NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
        NODE_ENV: process.env.NODE_ENV,
        window_location: typeof window !== 'undefined' ? window.location.href : 'SSR'
      })

      const url = 'http://localhost:8002/api/workflows'
      console.log('📡 Making request to:', url)

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(testWorkflow),
        mode: 'cors', // Explicitly set CORS mode
      })

      console.log('📡 Response status:', response.status)
      console.log('📡 Response ok:', response.ok)

      if (!response.ok) {
        const errorText = await response.text()
        console.error('❌ Response error:', errorText)
        throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`)
      }

      const data = await response.json()
      console.log('✅ Success:', data)
      setStatus('Success!')
      setResult(JSON.stringify(data, null, 2))
    } catch (error) {
      console.error('❌ Error:', error)
      setStatus('Failed!')
      setResult(error instanceof Error ? error.message : 'Unknown error')
    }
  }

  return (
    <div className="p-4 border rounded">
      <h3 className="text-lg font-semibold mb-4">Debug: Workflow Creation Test</h3>
      <button 
        onClick={testWorkflowCreation}
        className="mb-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
      >
        Test Workflow Creation
      </button>
      
      {status && (
        <div className="mb-2">
          <strong>Status:</strong> {status}
        </div>
      )}
      
      {result && (
        <div>
          <strong>Result:</strong>
          <pre className="mt-2 p-2 bg-gray-100 rounded overflow-auto text-xs">
            {result}
          </pre>
        </div>
      )}
    </div>
  )
}
