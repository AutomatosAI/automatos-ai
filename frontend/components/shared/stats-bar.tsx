'use client'

import { motion } from 'framer-motion'
import { Card, CardContent } from '@/components/ui/card'
import { cn } from '@/lib/utils'
import type { LucideIcon } from 'lucide-react'

export interface StatItem {
  label: string
  value: string | number
  change?: string
  icon: LucideIcon
  iconColor?: string
}

export interface StatsBarProps {
  stats: StatItem[]
  loading?: boolean
  glow?: boolean
  className?: string
}

const iconColors = [
  'text-primary',
  'text-[hsl(var(--success))]',
  'text-[hsl(var(--info))]',
  'text-[hsl(var(--agent))]',
]

export function StatsBar({ stats, loading = false, glow = true, className }: StatsBarProps) {
  return (
    <div className={cn('grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6', className)}>
      {stats.map((stat, index) => (
        <motion.div
          key={stat.label}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: index * 0.1 }}
        >
          <Card className={cn('glass-card hover:border-primary/20 transition-all duration-300', glow && 'card-glow')}>
            <CardContent className="p-4">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-3 min-w-0">
                  <div className="w-10 h-10 rounded-2xl bg-black/20 border border-primary/10 flex items-center justify-center shrink-0">
                    <stat.icon
                      className={cn('w-5 h-5', stat.iconColor || iconColors[index % iconColors.length])}
                    />
                  </div>
                  <div className="min-w-0">
                    <div className="text-2xl font-bold leading-none">
                      {loading ? '...' : stat.value}
                    </div>
                    <div className="text-sm text-muted-foreground truncate">{stat.label}</div>
                  </div>
                </div>
                {stat.change && (
                  <div className="shrink-0 text-xs text-muted-foreground">
                    {stat.change}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      ))}
    </div>
  )
}
