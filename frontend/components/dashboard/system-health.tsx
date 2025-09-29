'use client'

import { motion } from 'framer-motion'
import { Server, CheckCircle, AlertCircle } from 'lucide-react'
import { useSystemHealth } from '@/hooks/use-system-config-api'
import dynamic from 'next/dynamic'

// Client-side only component to avoid hydration issues with time
const TimeDisplay = dynamic(() => Promise.resolve(({ timestamp }: { timestamp: string }) => {
  return <span>{timestamp ? new Date(timestamp).toLocaleTimeString() : 'Unknown'}</span>;
}), { ssr: false });

export function SystemHealth() {
  const { data: healthData, isLoading: loading, error } = useSystemHealth()

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
        <div className="text-red-400">Error: {error.message || 'Unknown error'}</div>
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
              Last checked: <TimeDisplay timestamp={healthData?.timestamp} />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
