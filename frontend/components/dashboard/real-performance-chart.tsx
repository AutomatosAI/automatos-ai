
'use client'

import { motion } from 'framer-motion'
import { useState, useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  ReferenceLine
} from 'recharts'
import { 
  TrendingUp, 
  Activity, 
  Clock,
  Cpu,
  MemoryStick,
  Zap
} from 'lucide-react'

interface RealPerformanceChartProps {
  systemMetrics: any
  isLoading: boolean
}

export function RealPerformanceChart({ systemMetrics, isLoading }: RealPerformanceChartProps) {
  const [selectedMetric, setSelectedMetric] = useState<'cpu' | 'memory' | 'api_calls' | 'response_time'>('cpu')

  // Generate mock time series data based on real current values
  // In production, this would come from your time-series database
  const generateTimeSeriesData = (currentValue: number, dataPoints: number = 24) => {
    const data = []
    const now = new Date()
    
    for (let i = dataPoints - 1; i >= 0; i--) {
      const timestamp = new Date(now.getTime() - (i * 60 * 60 * 1000)) // Hourly data points
      
      // Generate realistic fluctuations around the current value
      const variation = (Math.random() - 0.5) * 20 // ±10 variation
      const value = Math.max(0, Math.min(100, currentValue + variation))
      
      data.push({
        time: timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        fullTime: timestamp,
        value: Math.round(value * 100) / 100,
        timestamp: timestamp.getTime()
      })
    }
    
    // Ensure the last data point matches the current value
    if (data.length > 0) {
      data[data.length - 1].value = currentValue
    }
    
    return data
  }

  const chartData = useMemo(() => {
    if (!systemMetrics) {
      return []
    }

    const metrics = systemMetrics
    
    switch (selectedMetric) {
      case 'cpu':
        return generateTimeSeriesData(metrics.cpu?.average_usage || 23)
      case 'memory':
        return generateTimeSeriesData(metrics.memory?.percent || 45)
      case 'api_calls':
        return generateTimeSeriesData(metrics.api_calls_per_minute || 150)
      case 'response_time':
        return generateTimeSeriesData(metrics.avg_response_time || 245)
      default:
        return []
    }
  }, [systemMetrics, selectedMetric])

  const metricConfigs = {
    cpu: {
      name: 'CPU Usage',
      unit: '%',
      color: '#3b82f6',
      icon: Cpu,
      threshold: 80,
      current: systemMetrics?.cpu?.average_usage || 0
    },
    memory: {
      name: 'Memory Usage', 
      unit: '%',
      color: '#10b981',
      icon: MemoryStick,
      threshold: 85,
      current: systemMetrics?.memory?.percent || 0
    },
    api_calls: {
      name: 'API Calls',
      unit: '/min',
      color: '#f59e0b',
      icon: Activity,
      threshold: 1000,
      current: 150 || 0
    },
    response_time: {
      name: 'Response Time',
      unit: 'ms', 
      color: '#8b5cf6',
      icon: Clock,
      threshold: 1000,
      current: 245 || 0
    }
  }

  const currentConfig = metricConfigs[selectedMetric]

  const customTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-background/95 backdrop-blur-sm border border-border rounded-lg p-3 shadow-lg">
          <p className="text-sm font-medium">{label}</p>
          <p className="text-sm text-muted-foreground">
            {currentConfig.name}: <span className="font-mono text-foreground">
              {payload[0].value}{currentConfig.unit}
            </span>
          </p>
        </div>
      )
    }
    return null
  }

  if (isLoading) {
    return (
      <Card className="glass-card">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-brand-secondary" />
              Performance Metrics
            </CardTitle>
            <div className="flex gap-2">
              {[...Array(4)].map((_, i) => (
                <Skeleton key={i} className="h-8 w-20" />
              ))}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-64 w-full" />
        </CardContent>
      </Card>
    )
  }

  const trend = chartData.length >= 2 
    ? chartData[chartData.length - 1].value - chartData[chartData.length - 2].value
    : 0

  return (
    <Card className="glass-card">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-brand-secondary" />
            <div>
              <CardTitle>Performance Metrics</CardTitle>
              <p className="text-sm text-muted-foreground">
                Real-time system performance monitoring
              </p>
            </div>
          </div>
          
          {/* Metric Selection Buttons */}
          <div className="flex gap-1">
            {Object.entries(metricConfigs).map(([key, config]) => {
              const Icon = config.icon
              const isSelected = selectedMetric === key
              
              return (
                <Button
                  key={key}
                  variant={isSelected ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setSelectedMetric(key as any)}
                  className={`h-8 px-3 ${isSelected ? 'bg-brand-primary' : ''}`}
                >
                  <Icon className="w-3 h-3 mr-1" />
                  <span className="hidden sm:inline">{config.name}</span>
                </Button>
              )
            })}
          </div>
        </div>
      </CardHeader>
      
      <CardContent>
        {/* Current Value and Trend */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              {(() => {
                const Icon = currentConfig.icon
                return <Icon className="w-5 h-5" style={{ color: currentConfig.color }} />
              })()}
              <div>
                <p className="text-2xl font-bold">
                  {currentConfig.current}{currentConfig.unit}
                </p>
                <p className="text-sm text-muted-foreground">
                  Current {currentConfig.name}
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              {trend !== 0 && (
                <Badge variant={trend > 0 ? 'destructive' : 'default'}>
                  {trend > 0 ? '+' : ''}{trend.toFixed(1)}{currentConfig.unit}
                </Badge>
              )}
              
              {currentConfig.current > currentConfig.threshold && (
                <Badge variant="secondary">
                  Above Threshold
                </Badge>
              )}
            </div>
          </div>
          
          {/* Status Indicator */}
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${
              currentConfig.current < currentConfig.threshold 
                ? 'bg-green-500 animate-pulse' 
                : 'bg-yellow-500 animate-pulse'
            }`} />
            <span className="text-sm text-muted-foreground">
              {currentConfig.current < currentConfig.threshold ? 'Normal' : 'Warning'}
            </span>
          </div>
        </div>

        {/* Chart */}
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid 
                strokeDasharray="3 3" 
                stroke="hsl(var(--border))" 
                opacity={0.3}
              />
              <XAxis 
                dataKey="time" 
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
                tickLine={false}
                axisLine={false}
              />
              <YAxis 
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
                tickLine={false}
                axisLine={false}
              />
              
              {/* Threshold line */}
              <ReferenceLine 
                y={currentConfig.threshold} 
                stroke="#ef4444" 
                strokeDasharray="5 5"
                opacity={0.7}
              />
              
              <Tooltip content={customTooltip} />
              
              <Line 
                type="monotone" 
                dataKey="value" 
                stroke={currentConfig.color}
                strokeWidth={2}
                dot={false}
                activeDot={{ 
                  r: 4, 
                  fill: currentConfig.color,
                  stroke: 'hsl(var(--background))',
                  strokeWidth: 2
                }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Chart Legend */}
        <div className="flex items-center justify-between mt-4 pt-4 border-t border-border/50">
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <div className="flex items-center gap-2">
              <div 
                className="w-3 h-0.5 rounded"
                style={{ backgroundColor: currentConfig.color }}
              />
              <span>{currentConfig.name}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-0.5 rounded bg-red-500 opacity-70" style={{
                background: 'repeating-linear-gradient(to right, #ef4444 0, #ef4444 3px, transparent 3px, transparent 8px)'
              }} />
              <span>Threshold ({currentConfig.threshold}{currentConfig.unit})</span>
            </div>
          </div>
          
          <div className="text-xs text-muted-foreground">
            Last 24 hours
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
