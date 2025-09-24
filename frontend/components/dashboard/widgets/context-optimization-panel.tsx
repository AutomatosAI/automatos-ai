'use client'

/**
 * Context Optimization Panel - Shows ACTUAL token savings
 */

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { 
  Zap, 
  TrendingDown, 
  BarChart3,
  ArrowDown,
  ArrowUp,
  Info
} from 'lucide-react'
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts'

interface ContextOptimizationPanelProps {
  contextData: any
  overview: any
}

export function ContextOptimizationPanel({ contextData, overview }: ContextOptimizationPanelProps) {
  // Map API response to expected structure
  const data = {
    total_optimizations: contextData?.totalOptimizations || 0,
    avg_tokens_saved: contextData?.tokensSaved ? Math.floor(contextData.tokensSaved / Math.max(contextData.totalOptimizations, 1)) : 0,
    total_tokens_saved: contextData?.tokensSaved || 0,
    avg_information_density: contextData?.efficiency || 0,
    compression_ratio: contextData?.avgCompressionRatio || 1.0,
    optimization_success_rate: contextData?.efficiency || 0,
    patterns_used: contextData?.patterns_used || {}
  };
  const [timeRange, setTimeRange] = useState<'24h' | '7d' | '30d'>('24h')
  const [trendData, setTrendData] = useState<any[]>([])
  const [isLoading, setIsLoading] = useState(false)

  // Fetch trend data
  useEffect(() => {
    const fetchTrendData = async () => {
      setIsLoading(true)
      try {
        const response = await fetch(`/api/analytics/trends?metric=token_usage&period=${timeRange}`)
        const result = await response.json()
        setTrendData(result.data || [])
      } catch (error) {
        console.error('Error fetching trend data:', error)
      } finally {
        setIsLoading(false)
      }
    }

    fetchTrendData()
  }, [timeRange])

  // Calculate savings percentage
  const savingsPercent = data.compression_ratio > 1 
    ? ((data.compression_ratio - 1) / data.compression_ratio * 100)
    : 0

  // Get top patterns
  const topPatterns = Object.entries(data.patterns_used)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5)

  return (
    <div className="space-y-4">
      {/* Header Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground">Total Saved</p>
                <p className="text-2xl font-bold">
                  {data.total_tokens_saved > 1000000 
                    ? `${(data.total_tokens_saved / 1000000).toFixed(1)}M`
                    : data.total_tokens_saved > 1000
                    ? `${(data.total_tokens_saved / 1000).toFixed(1)}k`
                    : data.total_tokens_saved
                  }
                </p>
                <p className="text-xs text-green-500 flex items-center mt-1">
                  <ArrowDown className="w-3 h-3" />
                  tokens
                </p>
              </div>
              <Zap className="w-8 h-8 text-yellow-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground">Compression</p>
                <p className="text-2xl font-bold">{data.compression_ratio.toFixed(1)}x</p>
                <p className="text-xs text-blue-500 flex items-center mt-1">
                  <TrendingDown className="w-3 h-3" />
                  {savingsPercent.toFixed(1)}%
                </p>
              </div>
              <BarChart3 className="w-8 h-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground">Info Density</p>
                <p className="text-2xl font-bold">{data.avg_information_density.toFixed(3)}</p>
                <p className="text-xs text-purple-500 flex items-center mt-1">
                  <ArrowUp className="w-3 h-3" />
                  bits/token
                </p>
              </div>
              <Info className="w-8 h-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground">Success Rate</p>
                <p className="text-2xl font-bold">{data.optimization_success_rate.toFixed(1)}%</p>
                <Progress value={data.optimization_success_rate} className="mt-2 h-1" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Token Usage Trend */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Token Optimization Trend</CardTitle>
            <div className="flex gap-2">
              <Button
                variant={timeRange === '24h' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setTimeRange('24h')}
              >
                24h
              </Button>
              <Button
                variant={timeRange === '7d' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setTimeRange('7d')}
              >
                7d
              </Button>
              <Button
                variant={timeRange === '30d' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setTimeRange('30d')}
              >
                30d
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="h-64 flex items-center justify-center">
              <p className="text-muted-foreground">Loading trend data...</p>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={trendData}>
                <defs>
                  <linearGradient id="tokenGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                <XAxis 
                  dataKey="date" 
                  tickFormatter={(value) => new Date(value).toLocaleDateString()}
                />
                <YAxis />
                <Tooltip 
                  labelFormatter={(value) => new Date(value).toLocaleString()}
                  formatter={(value: any) => [`${value} tokens`, 'Used']}
                />
                <Area
                  type="monotone"
                  dataKey="total"
                  stroke="#3b82f6"
                  fill="url(#tokenGradient)"
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>

      {/* Pattern Usage */}
      <Card>
        <CardHeader>
          <CardTitle>Optimization Patterns Used</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {topPatterns.map(([pattern, count]) => (
              <div key={pattern} className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Badge variant="outline">{pattern}</Badge>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-sm text-muted-foreground">
                    {count} uses
                  </span>
                  <Progress 
                    value={(count / data.total_optimizations) * 100} 
                    className="w-20 h-2"
                  />
                </div>
              </div>
            ))}
            {topPatterns.length === 0 && (
              <p className="text-sm text-muted-foreground text-center py-4">
                No pattern data available yet
              </p>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Optimization Stats */}
      <Card>
        <CardContent className="pt-6">
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <p className="text-lg font-bold">{data.total_optimizations}</p>
              <p className="text-xs text-muted-foreground">Total Optimizations</p>
            </div>
            <div>
              <p className="text-lg font-bold">{data.avg_tokens_saved.toFixed(0)}</p>
              <p className="text-xs text-muted-foreground">Avg Tokens/Optimization</p>
            </div>
            <div>
              <p className="text-lg font-bold">
                ${(data.total_tokens_saved * 0.00002).toFixed(2)}
              </p>
              <p className="text-xs text-muted-foreground">Estimated Savings</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

