
'use client'

import { motion } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
import { LucideIcon, TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

interface EnhancedMetricCardProps {
  title: string
  value: string | number
  unit?: string
  change?: number | string
  changeType?: 'up' | 'down' | 'neutral' | 'good' | 'warning' | 'critical'
  icon: LucideIcon
  gradient: string
  subtitle?: string
  badge?: {
    text: string
    variant: 'default' | 'secondary' | 'destructive' | 'outline'
  }
  trend_data?: Array<{hour?: number, load?: number, value?: number}>
  onClick?: () => void
}

export function EnhancedMetricCard({ 
  title, 
  value, 
  unit = '',
  change, 
  changeType = 'neutral',
  icon: Icon, 
  gradient,
  subtitle,
  badge,
  trend_data,
  onClick
}: EnhancedMetricCardProps) {
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  })

  const getChangeIcon = () => {
    switch (changeType) {
      case 'up':
      case 'good':
        return <TrendingUp className="w-3 h-3" />
      case 'down':
      case 'warning':
      case 'critical':
        return <TrendingDown className="w-3 h-3" />
      default:
        return <Minus className="w-3 h-3" />
    }
  }

  const getChangeColor = () => {
    switch (changeType) {
      case 'good':
        return 'text-green-400'
      case 'warning':
        return 'text-yellow-400'
      case 'critical':
        return 'text-red-400'
      case 'up':
        return 'text-green-400'
      case 'down':
        return 'text-red-400'
      default:
        return 'text-gray-400'
    }
  }

  const getBadgeColor = () => {
    switch (changeType) {
      case 'good':
        return 'bg-green-500/10 text-green-400 border-green-500/20'
      case 'warning':
        return 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20'
      case 'critical':
        return 'bg-red-500/10 text-red-400 border-red-500/20'
      default:
        return 'bg-blue-500/10 text-blue-400 border-blue-500/20'
    }
  }

  // Create mini trend visualization
  const renderMiniTrend = () => {
    if (!trend_data || trend_data.length === 0) return null

    const maxValue = Math.max(...trend_data.map(d => d?.load || d?.value || 0))
    const minValue = Math.min(...trend_data.map(d => d?.load || d?.value || 0))
    const range = maxValue - minValue || 1

    const points = trend_data.map((d, i) => {
      const value = d?.load || d?.value || 0
      const x = (i / (trend_data.length - 1)) * 100
      const y = 100 - ((value - minValue) / range) * 80 // 80% of height, leaving margin
      return `${x},${y}`
    }).join(' ')

    return (
      <div className="absolute bottom-2 right-2 w-16 h-8">
        <svg className="w-full h-full" viewBox="0 0 100 100">
          <polyline
            points={points}
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            className={getChangeColor()}
            opacity={0.6}
          />
        </svg>
      </div>
    )
  }

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 20 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.5, delay: 0.1 }}
      className={onClick ? "cursor-pointer" : ""}
      onClick={onClick}
    >
      <Card className="relative overflow-hidden bg-gray-900/50 backdrop-blur-sm border-gray-800/50 hover:bg-gray-900/60 transition-all duration-300 group">
        <CardContent className="p-6">
          <div className="flex items-start justify-between">
            <div className="space-y-2 flex-1">
              <div className="flex items-center justify-between">
                <p className="text-sm font-medium text-gray-300">{title}</p>
                {badge && (
                  <Badge 
                    variant="outline" 
                    className={`text-xs ${getBadgeColor()}`}
                  >
                    {badge.text}
                  </Badge>
                )}
              </div>
              
              <div className="flex items-baseline space-x-2">
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={inView ? { opacity: 1 } : {}}
                  transition={{ duration: 1, delay: 0.5 }}
                  className="text-2xl font-bold text-white"
                >
                  {typeof value === 'number' ? (
                    <motion.span
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 2 }}
                    >
                      {value}
                    </motion.span>
                  ) : (
                    value
                  )}
                  {unit && <span className="text-lg text-gray-400 ml-1">{unit}</span>}
                </motion.div>
                
                {change !== undefined && (
                  <div className={`flex items-center space-x-1 text-sm ${getChangeColor()}`}>
                    {getChangeIcon()}
                    <span>{change}</span>
                  </div>
                )}
              </div>

              {subtitle && (
                <p className="text-xs text-gray-400">{subtitle}</p>
              )}
            </div>

            <div className="relative">
              <div className={`p-3 rounded-lg bg-gradient-to-br ${gradient} group-hover:scale-110 transition-transform duration-300`}>
                <Icon className="w-6 h-6 text-white" />
              </div>
            </div>
          </div>

          {/* Mini trend chart */}
          {renderMiniTrend()}

          {/* Hover effect overlay */}
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
        </CardContent>
      </Card>
    </motion.div>
  )
}
