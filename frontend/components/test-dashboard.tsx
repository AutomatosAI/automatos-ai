'use client'

import { useState, useEffect } from 'react'

export function TestDashboard() {
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        console.log('🚀 Fetching system health...')
        const response = await fetch('https://api.automatos.app/api/system/health')
        const result = await response.json()
        console.log('✅ Got data:', result)
        setData(result)
      } catch (err: any) {
        console.error('❌ Error:', err)
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  if (loading) return <div>Loading...</div>
  if (error) return <div>Error: {error}</div>

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Test Dashboard</h1>
      <div className="bg-gray-100 p-4 rounded">
        <h2 className="text-lg font-semibold mb-2">System Health:</h2>
        <pre>{JSON.stringify(data, null, 2)}</pre>
      </div>
    </div>
  )
}
