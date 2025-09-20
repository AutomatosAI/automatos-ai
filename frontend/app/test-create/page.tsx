'use client'
import { useState } from 'react'

export default function TestCreate() {
  const [result, setResult] = useState('')
  
  const createAgent = async () => {
    try {
      const res = await fetch('https://api.automatos.app/api/agents/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': 'test_api_key_for_backend_validation_2025'
        },
        body: JSON.stringify({
          name: 'Test Agent ' + Date.now(),
          agent_type: 'task',
          description: 'Created from test page'
        })
      })
      const data = await res.json()
      setResult('SUCCESS! Agent ' + data.id + ' created')
    } catch (e) {
      setResult('ERROR: ' + e.message)
    }
  }
  
  return (
    <div style={{padding: 50}}>
      <h1>Direct Agent Create Test</h1>
      <button 
        onClick={createAgent}
        style={{padding: 10, fontSize: 20, background: 'blue', color: 'white'}}
      >
        CREATE AGENT
      </button>
      <p>{result}</p>
    </div>
  )
}
