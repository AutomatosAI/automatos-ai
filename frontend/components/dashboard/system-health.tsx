'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Server, CheckCircle, AlertCircle } from 'lucide-react'
import { apiClient } from '@/lib/api'

export function SystemHealth() {
  const [healthData, setHealthData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchHealthData = async () => {
      try {
        setLoading(true)
        const data = await apiClient.getSystemHealth()
        setHealthData(data)
        setError(null)
      } catch (err) {
        console.error('Failed to fetch health data:', err)
        setError(err.message)
        setHealthData(null)
      } finally {
        setLoading(false)
      }
    }

    fetchHealthData()
    const interval = setInterval(fetchHealthData, 30000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="glass-card p-6">
        <h3 className="text-lg font-semibold mb-4">System Health</h3>
        <div className="animate-pulse">Loading...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="glass-card p-6">
        <h3 className="text-lg font-semibold mb-4">System Health</h3>
        <div className="text-red-400">Error: {error}</div>
      </div>
    )
  }

  return (
    <div className="glass-card p-6">
      <h3 className="text-lg font-semibold mb-4">System Health</h3>
      <div className="space-y-4">
        <div className="flex items-center space-x-3">
          {healthData?.status === 'healthy' ? (
            <CheckCircle className="w-6 h-6 text-green-400" />
          ) : (
            <AlertCircle className="w-6 h-6 text-red-400" />
          )}
          <div>
            <div className="font-medium">Status: {healthData?.status}</div>
            <div className="text-sm text-muted-foreground">Version: {healthData?.version}</div>
            <div className="text-sm text-muted-foreground">
              Last checked: {healthData?.timestamp ? new Date(healthData.timestamp).toLocaleTimeString() : 'Unknown'}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
