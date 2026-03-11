'use client'

import { motion } from 'framer-motion'
import { Card, CardContent } from '@/components/ui/card'
import { cn } from '@/lib/utils'
import { PremiumIcon } from '@/components/shared'
import { useSystemIcons } from '@/hooks/use-system-config-api'
import type { LucideIcon } from 'lucide-react'

export interface StatItem {
  label: string
  value: string | number
  change?: string
  icon: LucideIcon
  iconColor?: string
  /** Optional global icon key (e.g. 'global_agent'). If set in system config, PremiumIcon renders instead of Lucide. */
  globalIconKey?: string
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
  const { data: iconMappings = {} } = useSystemIcons()

  return (
    <div className={cn('hidden md:grid grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6', className)}>
      {stats.map((stat, index) => {
        const premiumName = stat.globalIconKey ? iconMappings[stat.globalIconKey] : null

        return (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: index * 0.1 }}
          >
            <Card className={cn('glass-card hover:border-primary/20 transition-all duration-300', glow && 'card-glow')}>
              <CardContent className="p-4">
                <div className="flex items-start gap-3">
                  <div className="w-10 h-10 rounded-2xl bg-black/20 border border-primary/10 flex items-center justify-center shrink-0 mt-0.5">
                    {premiumName ? (
                      <PremiumIcon name={premiumName} size={22} />
                    ) : (
                      <stat.icon
                        className={cn('w-5 h-5', stat.iconColor || iconColors[index % iconColors.length])}
                      />
                    )}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-baseline justify-between gap-2">
                      <span className="text-2xl font-bold leading-none truncate">
                        {loading ? '...' : stat.value}
                      </span>
                      {stat.change && (
                        <span className="shrink-0 text-xs text-muted-foreground whitespace-nowrap">
                          {stat.change}
                        </span>
                      )}
                    </div>
                    <div className="text-sm text-muted-foreground mt-1">
                      <span className="truncate">{stat.label}</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )
      })}
    </div>
  )
}
