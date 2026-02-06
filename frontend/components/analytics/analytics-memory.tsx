'use client'

import { Brain, Edit3, Check } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'

interface MemoryData {
  items: Array<{
    key: string
    value: string
    source?: string
    updated_at?: string
  }>
  totalMemories: number
  hitRate: number
}

interface Props {
  data: MemoryData | undefined
  isLoading: boolean
}

export function AnalyticsMemory({ data, isLoading }: Props) {
  if (isLoading) {
    return (
      <Card className="glass-card">
        <CardHeader>
          <Skeleton className="h-5 w-48" />
        </CardHeader>
        <CardContent className="space-y-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-12 w-full" />
          ))}
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <Brain className="w-5 h-5 text-purple-400" />
            What Automatos Knows
          </span>
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="text-xs">
              {data?.totalMemories || 0} memories
            </Badge>
            <Badge variant="outline" className="text-xs">
              {((data?.hitRate || 0) * 100).toFixed(0)}% hit rate
            </Badge>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {data?.items && data.items.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {data.items.slice(0, 8).map((item, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-3 rounded-lg bg-secondary/20 border border-border/50"
              >
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-muted-foreground">{item.key}</p>
                  <p className="text-sm font-medium truncate">{item.value}</p>
                </div>
                <Badge
                  variant="outline"
                  className={`text-xs shrink-0 ml-2 ${
                    item.source === 'user' ? 'text-blue-400 border-blue-400/30' : 'text-purple-400 border-purple-400/30'
                  }`}
                >
                  {item.source === 'user' ? 'You set this' : 'Learned'}
                </Badge>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-6 text-muted-foreground">
            <Brain className="w-10 h-10 mx-auto mb-3 opacity-50" />
            <p className="text-sm">Automatos is still learning about your workspace.</p>
            <p className="text-xs mt-1">Preferences will appear here as you use the platform.</p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
