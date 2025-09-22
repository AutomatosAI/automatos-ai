'use client'

/**
 * Token Usage Trends - Shows token usage over time
 */

import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Zap } from 'lucide-react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts'

export function TokenUsageTrends() {
  const [trendData, setTrendData] = useState<any[]>([])
  const [timeRange, setTimeRange] = useState<'7d' | '30d'>('7d')
  const [isLoading, setIsLoading] = useState(false)

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

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Zap className="w-5 h-5" />
            Token Usage Trends
          </CardTitle>
          <div className="flex gap-2">
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
            <p className="text-muted-foreground">Loading trends...</p>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={trendData}>
              <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
              <XAxis 
                dataKey="date" 
                tickFormatter={(value) => new Date(value).toLocaleDateString()}
              />
              <YAxis />
              <Tooltip 
                labelFormatter={(value) => new Date(value).toLocaleDateString()}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="total"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
                name="Total Tokens"
              />
              <Line
                type="monotone"
                dataKey="average"
                stroke="#10b981"
                strokeWidth={2}
                dot={false}
                name="Avg per Execution"
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  )
}

