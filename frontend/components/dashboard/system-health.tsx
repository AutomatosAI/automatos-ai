
'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Server, Database, Network, Shield, AlertCircle, CheckCircle, Clock } from 'lucide-react'
import { apiClient, SystemHealth as SystemHealthType } from '@/lib/api'

interface HealthMetric {
  name: string
  status: string
  uptime: string
  icon: any
  color: string
}

export function SystemHealth() {
  const [healthData, setHealthData] = useState<SystemHealthType | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchHealthData = async () => {
      try {
        setLoading(true)
        const data = await apiClient.getSystemHealth()
        setHealthData(data)
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch system health')
        console.error('Error fetching system health:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchHealthData()
    
    // Refresh health data every 30 seconds
    const interval = setInterval(fetchHealthData, 30000)
    return () => clearInterval(interval)
  }, [])

  const getHealthMetrics = (): HealthMetric[] => {
    if (!healthData) return []

    const serviceMapping: Record<string, { name: string; icon: any }> = {
      api: { name: 'API Gateway', icon: Network },
      database: { name: 'Database', icon: Database },
      document_processor: { name: 'Document Processor', icon: Server },
      rag_system: { name: 'RAG System', icon: Shield }
    }

    return Object.entries(healthData.services).map(([key, status]) => {
      const mapping = serviceMapping[key] || { name: key.replace('_', ' '), icon: Server }
      const isHealthy = status === 'healthy'
      
      return {
        name: mapping.name,
        status: status,
        uptime: isHealthy ? '99.9%' : '0%',
        icon: mapping.icon,
        color: isHealthy ? 'text-green-400' : status === 'degraded' ? 'text-yellow-400' : 'text-red-400'
      }
    })
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
      case 'degraded':
        return <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse" />
      case 'unhealthy':
        return <div className="w-2 h-2 bg-red-400 rounded-full animate-pulse" />
      default:
        return <div className="w-2 h-2 bg-gray-400 rounded-full" />
    }
  }

  const getOverallStatusColor = () => {
    if (!healthData) return 'text-gray-400'
    switch (healthData.status) {
      case 'healthy':
        return 'text-green-400'
      case 'degraded':
        return 'text-yellow-400'
      case 'unhealthy':
        return 'text-red-400'
      default:
        return 'text-gray-400'
    }
  }

  if (loading) {
    return (
      <div className="glass-card p-6">
        <h3 className="text-lg font-semibold mb-4">System Health</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, index) => (
            <div key={index} className="p-4 rounded-lg border border-border/50 animate-pulse">
              <div className="h-4 bg-secondary rounded mb-2"></div>
              <div className="h-3 bg-secondary rounded mb-1"></div>
              <div className="h-2 bg-secondary rounded"></div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="glass-card p-6">
        <h3 className="text-lg font-semibold mb-4">System Health</h3>
        <div className="flex items-center justify-center p-8">
          <div className="text-center">
            <AlertCircle className="h-8 w-8 text-red-500 mx-auto mb-2" />
            <p className="text-red-500 mb-1">Failed to load system health</p>
            <p className="text-sm text-muted-foreground">{error}</p>
          </div>
        </div>
      </div>
    )
  }

  const healthMetrics = getHealthMetrics()

  return (
    <div className="glass-card p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">System Health</h3>
        <div className="flex items-center space-x-2">
          <span className={`text-sm font-medium ${getOverallStatusColor()}`}>
            {healthData?.status?.toUpperCase() || 'UNKNOWN'}
          </span>
          {healthData && getStatusIcon(healthData.status)}
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
        {healthMetrics.map((metric, index) => (
          <motion.div
            key={metric.name}
            className="p-4 rounded-lg border border-border/50 hover:border-primary/20 transition-colors"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
          >
            <div className="flex items-center justify-between mb-2">
              <metric.icon className={`w-5 h-5 ${metric.color}`} />
              {getStatusIcon(metric.status)}
            </div>
            <h4 className="font-medium text-sm">{metric.name}</h4>
            <p className="text-muted-foreground text-xs">Status: {metric.status}</p>
          </motion.div>
        ))}
      </div>

      {/* System Metrics */}
      {healthData?.metrics && (
        <div className="border-t border-border/30 pt-4">
          <h4 className="text-sm font-medium mb-3">System Metrics</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-lg font-bold">{healthData.metrics.cpu_usage || 'N/A'}</div>
              <div className="text-xs text-muted-foreground">CPU Usage</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold">{healthData.metrics.memory_usage || 'N/A'}</div>
              <div className="text-xs text-muted-foreground">Memory Usage</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold">{healthData.metrics.disk_usage || 'N/A'}</div>
              <div className="text-xs text-muted-foreground">Disk Usage</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold">{healthData.version || 'N/A'}</div>
              <div className="text-xs text-muted-foreground">Version</div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
