'use client'

import { useEffect, useState } from 'react'

export default function TestAPI() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    console.log('Starting fetch to https://api.automatos.app/api/agents/')
    fetch('https://api.automatos.app/api/agents/', {
      headers: {
        'X-API-Key': 'test_api_key_for_backend_validation_2025'
      }
    })
    .then(res => {
      console.log('Response received:', res.status)
      return res.json()
    })
    .then(data => {
      console.log('Data parsed:', data)
      setData(data)
      setLoading(false)
    })
    .catch(err => {
      console.error('Fetch error:', err)
      setError(err.message)
      setLoading(false)
    })
  }, [])

  return (
    <div style={{ padding: '20px' }}>
      <h1>HTTPS API Test (api.automatos.app)</h1>
      {loading && <p>Loading...</p>}
      {error && <p style={{color: 'red'}}>Error: {error}</p>}
      {data && <p style={{color: 'green'}}>Success! Got {data.length} agents</p>}
    </div>
  )
}
