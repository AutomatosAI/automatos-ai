
'use client'

import { motion } from 'framer-motion'
import { LucideIcon } from 'lucide-react'
import { TrendingUp, TrendingDown } from 'lucide-react'

interface MetricCardProps {
  title: string
  value: string
  change: string
  changeType: 'positive' | 'negative'
  icon: LucideIcon
  gradient: string
}

export function MetricCard({
  title,
  value,
  change,
  changeType,
  icon: Icon,
  gradient
}: MetricCardProps) {
  return (
    <motion.div
      className="metric-card group"
      whileHover={{ scale: 1.02 }}
      transition={{ duration: 0.2 }}
    >
      <div className="flex items-center justify-between mb-4">
        <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${gradient} flex items-center justify-center`}>
          <Icon className="w-6 h-6 text-white" />
        </div>
        
        <div className={`flex items-center space-x-1 text-sm ${
          changeType === 'positive' ? 'text-green-400' : 'text-red-400'
        }`}>
          {changeType === 'positive' ? (
            <TrendingUp className="w-4 h-4" />
          ) : (
            <TrendingDown className="w-4 h-4" />
          )}
          <span>{change}</span>
        </div>
      </div>
      
      <div className="space-y-1">
        <h3 className="text-3xl font-bold">{value}</h3>
        <p className="text-muted-foreground text-sm">{title}</p>
      </div>
    </motion.div>
  )
}
