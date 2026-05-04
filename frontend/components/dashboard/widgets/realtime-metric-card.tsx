'use client'

/**
 * Realtime Metric Card - Animated metric display with live updates
 */

import { useEffect, useRef } from 'react'
import { motion, useAnimation } from 'framer-motion'
import { Card, CardContent } from '@/components/ui/card'
import { TrendingUp, TrendingDown, Minus, LucideIcon } from 'lucide-react'

interface RealtimeMetricCardProps {
  title: string
  value: number | string
  total?: number
  subtitle?: string
  icon: LucideIcon
  color: string
  trend?: 'up' | 'down' | 'neutral'
  animate?: boolean
}

export function RealtimeMetricCard({
  title,
  value,
  total,
  subtitle,
  icon: Icon,
  color,
  trend = 'neutral',
  animate = true
}: RealtimeMetricCardProps) {
  const controls = useAnimation()
  const previousValue = useRef(value)

  useEffect(() => {
    if (animate && value !== previousValue.current) {
      controls.start({
        scale: [1, 1.05, 1],
        transition: { duration: 0.3 }
      })
      previousValue.current = value
    }
  }, [value, animate, controls])

  const getTrendIcon = () => {
    switch (trend) {
      case 'up':
        return <TrendingUp className="w-4 h-4 text-success" />
      case 'down':
        return <TrendingDown className="w-4 h-4 text-destructive" />
      default:
        return <Minus className="w-4 h-4 text-muted-foreground" />
    }
  }

  const displayValue = total ? `${value}/${total}` : value

  return (
    <Card>
      <CardContent className="pt-6">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">{title}</p>
            <motion.div animate={controls}>
              <p className="text-2xl font-bold">{displayValue}</p>
            </motion.div>
            {subtitle && (
              <p className="text-xs text-muted-foreground">{subtitle}</p>
            )}
          </div>
          <div className="flex flex-col items-center gap-2">
            <Icon className={`w-8 h-8 ${color}`} />
            {getTrendIcon()}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

