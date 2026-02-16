'use client'

import type { ReactNode } from 'react'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'

export interface PageHeaderProps {
  title: string
  titleAccent: string
  subtitle?: string
  actions?: ReactNode
  className?: string
}

export function PageHeader({
  title,
  titleAccent,
  subtitle,
  actions,
  className,
}: PageHeaderProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={cn('flex flex-col sm:flex-row justify-between items-start gap-3', className)}
    >
      <div className="min-w-0">
        <h1 data-testid="page-title" className="text-2xl md:text-3xl font-bold mb-1 md:mb-2">
          {title} <span className="gradient-text">{titleAccent}</span>
        </h1>
        {subtitle && (
          <p className="text-sm md:text-base text-muted-foreground">{subtitle}</p>
        )}
      </div>
      {actions && <div className="flex items-center gap-2 md:gap-3 shrink-0 flex-wrap">{actions}</div>}
    </motion.div>
  )
}
