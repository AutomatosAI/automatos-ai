'use client'

import type { ReactNode } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import { Card, CardHeader, CardContent } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { cn } from '@/lib/utils'

export interface ItemCardProps {
  icon?: ReactNode
  title: ReactNode
  subtitle?: ReactNode
  titleBadges?: ReactNode
  description?: ReactNode
  meta?: ReactNode
  actions?: ReactNode
  animated?: boolean
  animationDelay?: number
  onClick?: () => void
  className?: string
  children?: ReactNode
}

export function ItemCard({
  icon,
  title,
  subtitle,
  titleBadges,
  description,
  meta,
  actions,
  animated = true,
  animationDelay = 0,
  onClick,
  className,
  children,
}: ItemCardProps) {
  const content = (
    <Card
      className={cn(
        'glass-card card-glow hover:border-primary/20 transition-all duration-300',
        onClick && 'cursor-pointer',
        className
      )}
      onClick={onClick}
    >
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-3 min-w-0">
            {icon && (
              <div className="shrink-0">{icon}</div>
            )}
            <div className="min-w-0">
              <div className="font-semibold truncate">{title}</div>
              {subtitle && (
                <div className="text-xs text-muted-foreground">{subtitle}</div>
              )}
            </div>
          </div>
          {titleBadges && (
            <div className="flex items-center gap-2 shrink-0">{titleBadges}</div>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-3 pt-0">
        {description && (
          <p className="text-sm text-muted-foreground line-clamp-2">{description}</p>
        )}
        {meta}
        {children}
        {actions && (
          <>
            <Separator />
            <div className="flex items-center justify-between">{actions}</div>
          </>
        )}
      </CardContent>
    </Card>
  )

  const prefersReducedMotion = useReducedMotion()

  if (!animated || prefersReducedMotion) return content

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      transition={{ duration: 0.4, delay: animationDelay }}
    >
      {content}
    </motion.div>
  )
}
