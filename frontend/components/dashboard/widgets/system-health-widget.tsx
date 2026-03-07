'use client'

/**
 * System Health Widget - Shows real-time system status
 */

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { 
  Activity, 
  Database, 
  HardDrive, 
  Cpu,
  AlertTriangle,
  CheckCircle 
} from 'lucide-react'

interface SystemHealthData {
  overall_status: string
  status_color: string
  health_score: number
  cpu: {
    usage_percent: number
    cores: number
    status: string
  }
  memory: {
    usage_percent: number
    used_gb: number
    total_gb: number
    status: string
  }
  disk: {
    usage_percent: number
    used_gb: number
    total_gb: number
    status: string
  }
  database: {
    status: string
    active_connections: number
    response_time_ms: number
  }
  redis: {
    status: string
    memory_mb: number
  }
  queue: {
    depth: number
    status: string
  }
  error_rate: number
}

interface SystemHealthWidgetProps {
  data: SystemHealthData
}

export function SystemHealthWidget({ data }: SystemHealthWidgetProps) {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'excellent':
        return <CheckCircle className="w-4 h-4 text-[hsl(var(--success))]" />
      case 'warning':
        return <AlertTriangle className="w-4 h-4 text-[hsl(var(--warning))]" />
      case 'critical':
        return <AlertTriangle className="w-4 h-4 text-[hsl(var(--destructive))]" />
      default:
        return null
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'excellent':
        return 'text-[hsl(var(--success))]'
      case 'warning':
        return 'text-[hsl(var(--warning))]'
      case 'critical':
        return 'text-[hsl(var(--destructive))]'
      default:
        return 'text-gray-500'
    }
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5" />
            System Health
          </CardTitle>
          <Badge 
            variant={data.overall_status === 'healthy' ? 'default' : 'destructive'}
            className={getStatusColor(data.overall_status)}
          >
            {data.overall_status.toUpperCase()} • {data.health_score}%
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* CPU Usage */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="flex items-center gap-2">
              <Cpu className="w-4 h-4" />
              CPU ({data.cpu.cores} cores)
            </span>
            <span className="flex items-center gap-1">
              {data.cpu.usage_percent}%
              {getStatusIcon(data.cpu.status)}
            </span>
          </div>
          <Progress value={data.cpu.usage_percent} className="h-2" />
        </div>

        {/* Memory Usage */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="flex items-center gap-2">
              <Activity className="w-4 h-4" />
              Memory
            </span>
            <span className="flex items-center gap-1">
              {data.memory.used_gb}/{data.memory.total_gb}GB
              {getStatusIcon(data.memory.status)}
            </span>
          </div>
          <Progress value={data.memory.usage_percent} className="h-2" />
        </div>

        {/* Disk Usage */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="flex items-center gap-2">
              <HardDrive className="w-4 h-4" />
              Disk
            </span>
            <span className="flex items-center gap-1">
              {data.disk.used_gb}/{data.disk.total_gb}GB
              {getStatusIcon(data.disk.status)}
            </span>
          </div>
          <Progress value={data.disk.usage_percent} className="h-2" />
        </div>

        {/* Services Status */}
        <div className="grid grid-cols-2 gap-4 pt-2 border-t">
          <div>
            <p className="text-xs text-muted-foreground">Database</p>
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4" />
              <Badge variant={data.database.status === 'healthy' ? 'default' : 'destructive'}>
                {data.database.status}
              </Badge>
            </div>
            <p className="text-xs mt-1">
              {data.database.active_connections} connections • {data.database.response_time_ms}ms
            </p>
          </div>

          <div>
            <p className="text-xs text-muted-foreground">Redis</p>
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4" />
              <Badge variant={data.redis.status === 'healthy' ? 'default' : 'destructive'}>
                {data.redis.status}
              </Badge>
            </div>
            <p className="text-xs mt-1">
              {data.redis.memory_mb}MB memory
            </p>
          </div>
        </div>

        {/* Queue and Error Rate */}
        <div className="grid grid-cols-2 gap-4 pt-2 border-t">
          <div>
            <p className="text-xs text-muted-foreground">Queue Depth</p>
            <p className="text-lg font-semibold">{data.queue.depth}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Error Rate</p>
            <p className="text-lg font-semibold">{data.error_rate.toFixed(1)}%</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

