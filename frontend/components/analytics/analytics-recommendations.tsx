'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Sparkles,
  DollarSign,
  AlertTriangle,
  FileText,
  TrendingUp,
  X,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'

interface Recommendation {
  id: string
  type: 'cost' | 'performance' | 'document' | 'quota'
  title: string
  description: string
  impact: string
  action?: string
}

interface Props {
  recommendations: Recommendation[]
  isLoading: boolean
}

const typeConfig = {
  cost: { icon: DollarSign, color: 'text-green-400', bg: 'bg-green-500/10', border: 'border-green-500/20' },
  performance: { icon: AlertTriangle, color: 'text-yellow-400', bg: 'bg-yellow-500/10', border: 'border-yellow-500/20' },
  document: { icon: FileText, color: 'text-blue-400', bg: 'bg-blue-500/10', border: 'border-blue-500/20' },
  quota: { icon: TrendingUp, color: 'text-orange-400', bg: 'bg-orange-500/10', border: 'border-orange-500/20' },
}

export function AnalyticsRecommendations({ recommendations, isLoading }: Props) {
  const [dismissed, setDismissed] = useState<Set<string>>(new Set())

  const visible = recommendations.filter(r => !dismissed.has(r.id))

  if (isLoading) {
    return (
      <Card className="glass-card">
        <CardHeader>
          <Skeleton className="h-5 w-48" />
        </CardHeader>
        <CardContent className="space-y-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-20 w-full" />
          ))}
        </CardContent>
      </Card>
    )
  }

  if (visible.length === 0) {
    return (
      <Card className="glass-card">
        <CardContent className="p-6 text-center">
          <Sparkles className="w-10 h-10 mx-auto text-green-400 mb-3" />
          <p className="font-medium">Everything looks good!</p>
          <p className="text-sm text-muted-foreground mt-1">
            No optimization suggestions right now. We'll check back later.
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-orange-400" />
          Recommendations
          <Badge variant="secondary" className="text-xs">{visible.length}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <AnimatePresence>
          {visible.map((rec) => {
            const config = typeConfig[rec.type]
            const Icon = config.icon

            return (
              <motion.div
                key={rec.id}
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className={`p-4 rounded-lg ${config.bg} border ${config.border}`}
              >
                <div className="flex items-start gap-3">
                  <Icon className={`w-5 h-5 mt-0.5 shrink-0 ${config.color}`} />
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-sm">{rec.title}</p>
                    <p className="text-sm text-muted-foreground mt-1">{rec.description}</p>
                    {rec.impact && (
                      <Badge variant="outline" className="mt-2 text-xs">
                        {rec.impact}
                      </Badge>
                    )}
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="shrink-0 h-6 w-6 p-0"
                    aria-label={`Dismiss ${rec.title || 'recommendation'}`}
                    onClick={() => setDismissed(prev => new Set([...prev, rec.id]))}
                  >
                    <X className="w-3 h-3" />
                  </Button>
                </div>
              </motion.div>
            )
          })}
        </AnimatePresence>
      </CardContent>
    </Card>
  )
}
