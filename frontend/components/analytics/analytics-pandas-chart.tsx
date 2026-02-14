'use client'

import { useEffect } from 'react'
import { BarChart3, AlertCircle, Loader2 } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { useAnalyticsChart, useChartPresets } from '@/hooks/use-unified-analytics'

interface AnalyticsPandasChartProps {
  presetId?: string
  query?: string
  chartType?: string
}

export function AnalyticsPandasChart({ presetId, query, chartType }: AnalyticsPandasChartProps) {
  const { data: presets } = useChartPresets()
  const chartMutation = useAnalyticsChart()

  // Resolve preset config if presetId provided
  const preset = presetId && presets ? presets.find((p) => p.id === presetId) : null
  const resolvedQuery = query || preset?.query
  const resolvedChartType = chartType || preset?.chart_type
  const title = preset?.title || 'AI Chart'

  // Auto-generate chart when query is resolved and mutation hasn't fired
  useEffect(() => {
    if (resolvedQuery && !chartMutation.isLoading && !chartMutation.isSuccess && !chartMutation.isError) {
      chartMutation.mutate({ query: resolvedQuery, chartType: resolvedChartType })
    }
    // Only trigger on resolvedQuery change — mutation object ref changes on every render
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resolvedQuery, resolvedChartType])

  // Loading skeleton
  if (chartMutation.isLoading || (!resolvedQuery && !chartMutation.data)) {
    return (
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            <BarChart3 className="w-4 h-4 text-purple-400" />
            {title}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Loader2 className="w-3 h-3 animate-spin" />
            Generating chart...
          </div>
          <Skeleton className="h-48 w-full" />
          <Skeleton className="h-4 w-3/4" />
        </CardContent>
      </Card>
    )
  }

  // Error state
  if (chartMutation.isError) {
    return (
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            <BarChart3 className="w-4 h-4 text-purple-400" />
            {title}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <AlertCircle className="w-8 h-8 mx-auto mb-3 text-red-400 opacity-70" />
            <p className="text-sm text-muted-foreground">
              Failed to generate chart
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              {chartMutation.error.message}
            </p>
          </div>
        </CardContent>
      </Card>
    )
  }

  const chartData = chartMutation.data
  if (!chartData) return null

  return (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm">
          <BarChart3 className="w-4 h-4 text-purple-400" />
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Rendered chart images */}
        {chartData.charts && chartData.charts.length > 0 && (
          <div className="space-y-3">
            {chartData.charts.map((base64, idx) => (
              <img
                key={idx}
                src={`data:image/png;base64,${base64}`}
                alt={`${title} chart ${idx + 1}`}
                className="w-full rounded-lg border border-border/30"
              />
            ))}
          </div>
        )}

        {/* NL summary text */}
        {chartData.summary && (
          <p className="text-sm text-muted-foreground leading-relaxed">
            {chartData.summary}
          </p>
        )}

        {/* Fallback: no charts generated */}
        {(!chartData.charts || chartData.charts.length === 0) && !chartData.summary && (
          <div className="text-center py-6">
            <BarChart3 className="w-8 h-8 mx-auto mb-2 opacity-30 text-muted-foreground" />
            <p className="text-xs text-muted-foreground">No chart data available</p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
