'use client'

import { motion } from 'framer-motion'
import { Server, CheckCircle, AlertCircle, Database, Cpu, Activity } from 'lucide-react'
import { useSystemHealth } from '@/hooks/use-system-config-api'
import dynamic from 'next/dynamic'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'
import { Badge } from '../ui/badge'

// Client-side only component to avoid hydration issues with time
const TimeDisplay = dynamic(() => Promise.resolve(({ timestamp }: { timestamp: string }) => {
  return <span>{timestamp ? new Date(timestamp).toLocaleTimeString() : 'Unknown'}</span>;
}), { ssr: false });

export function SystemHealth() {
  const { data: healthData, isLoading: loading, error } = useSystemHealth()

  if (loading) {
    return (
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-brand-primary" />
            System Health
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-2">
            <div className="h-4 bg-muted rounded w-3/4" />
            <div className="h-4 bg-muted rounded w-1/2" />
          </div>
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-brand-primary" />
            System Health
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 text-destructive">
            <AlertCircle className="w-5 h-5" />
            <span>Error loading health data</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  const isHealthy = healthData?.status === 'healthy'
  const services = healthData?.services || {}

  return (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-brand-primary" />
            System Health
          </div>
          <Badge variant={isHealthy ? "default" : "destructive"} className="ml-auto">
            {healthData?.status?.toUpperCase()}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Overall Status */}
          <div className="flex items-center space-x-3 pb-3 border-b border-border/50">
            {isHealthy ? (
              <CheckCircle className="w-6 h-6 text-success" />
            ) : (
              <AlertCircle className="w-6 h-6 text-destructive" />
            )}
            <div className="flex-1">
              <div className="font-medium text-base">
                {isHealthy ? 'All Systems Operational' : 'System Degraded'}
              </div>
              <div className="text-sm text-muted-foreground flex items-center gap-3 mt-1">
                <span>Version {healthData?.version}</span>
                <span>•</span>
                <span><TimeDisplay timestamp={healthData?.timestamp} /></span>
              </div>
            </div>
          </div>

          {/* Services Status */}
          <div className="space-y-2">
            <div className="text-sm font-medium mb-2">Core Services</div>
            {Object.entries(services).map(([service, status]) => (
              <div key={service} className="flex items-center justify-between py-1">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${
                    status === 'healthy' ? 'bg-success' : 'bg-destructive'
                  }`} />
                  <span className="text-sm capitalize">{service.replace('_', ' ')}</span>
                </div>
                <Badge 
                  variant={status === 'healthy' ? "outline" : "destructive"} 
                  className="text-xs"
                >
                  {status}
                </Badge>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
