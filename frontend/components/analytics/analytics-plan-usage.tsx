'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { Gauge } from 'lucide-react'

interface PlanUsageData {
  planName: string
  planTier: string
  usage: Record<string, { used: number; limit: number | null; label: string }>
}

interface Props {
  data: PlanUsageData | undefined
  isLoading: boolean
}

function getUsageColor(percentage: number): string {
  if (percentage >= 90) return 'bg-[hsl(var(--destructive))]'
  if (percentage >= 70) return 'bg-[hsl(var(--warning))]'
  return 'bg-[hsl(var(--success))]'
}

function getUsageTextColor(percentage: number): string {
  if (percentage >= 90) return 'text-[hsl(var(--destructive))]'
  if (percentage >= 70) return 'text-[hsl(var(--warning))]'
  return 'text-[hsl(var(--success))]'
}

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return n.toString()
}

export function AnalyticsPlanUsage({ data, isLoading }: Props) {
  if (isLoading) {
    return (
      <Card className="glass-card">
        <CardHeader>
          <Skeleton className="h-5 w-32" />
        </CardHeader>
        <CardContent className="space-y-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="space-y-2">
              <Skeleton className="h-4 w-40" />
              <Skeleton className="h-3 w-full" />
            </div>
          ))}
        </CardContent>
      </Card>
    )
  }

  if (!data) return null

  return (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <Gauge className="w-5 h-5 text-[hsl(var(--info))]" />
            Plan Usage
          </span>
          <Badge variant="outline" className="text-xs">
            {data.planName} Plan
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Object.entries(data.usage).map(([key, item]) => {
            const hasLimit = item.limit != null
            const percentage = hasLimit
              ? item.limit! > 0
                ? Math.min(100, (item.used / item.limit!) * 100)
                : item.used > 0 ? 100 : 0
              : null

            return (
              <div key={key} className="space-y-2 p-3 rounded-lg bg-secondary/20">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">{item.label}</span>
                  <span className="text-sm text-muted-foreground">
                    {formatNumber(item.used)}
                    {hasLimit ? ` / ${formatNumber(item.limit!)}` : ''}
                  </span>
                </div>
                {percentage !== null ? (
                  <>
                    <div className="h-2 rounded-full bg-secondary overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-300 ${getUsageColor(percentage)}`}
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                    <p className={`text-xs ${getUsageTextColor(percentage)}`}>
                      {percentage.toFixed(0)}% used
                    </p>
                  </>
                ) : (
                  <p className="text-xs text-muted-foreground">
                    No limit set (pilot)
                  </p>
                )}
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
