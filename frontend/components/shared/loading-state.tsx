'use client'

import { Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Skeleton } from '@/components/ui/skeleton'

export type LoadingVariant = 'list' | 'cards' | 'page' | 'inline' | 'spinner'

export interface LoadingStateProps {
  /** Shape of the placeholder. Defaults to a skeleton list. */
  variant?: LoadingVariant
  /** Number of skeleton rows/cards to render (list & cards variants). */
  count?: number
  /** Accessible label announced to screen readers. */
  label?: string
  className?: string
}

/**
 * Canonical loading surface (PRD-169 S2). Collapses the six divergent loading
 * patterns into one Skeleton-based primitive. Use `variant` to match context.
 */
export function LoadingState({
  variant = 'list',
  count = 3,
  label = 'Loading…',
  className,
}: LoadingStateProps) {
  if (variant === 'spinner' || variant === 'inline') {
    return (
      <div
        role="status"
        aria-label={label}
        className={cn(
          'flex items-center justify-center gap-2 text-muted-foreground',
          variant === 'spinner' ? 'py-16' : 'py-2',
          className,
        )}
      >
        <Loader2 className="w-4 h-4 animate-spin" />
        <span className="text-sm">{label}</span>
      </div>
    )
  }

  if (variant === 'cards') {
    return (
      <div
        role="status"
        aria-label={label}
        className={cn('grid gap-4 sm:grid-cols-2 lg:grid-cols-3', className)}
      >
        {Array.from({ length: count }).map((_, i) => (
          <Skeleton key={i} className="h-40 w-full" />
        ))}
      </div>
    )
  }

  if (variant === 'page') {
    return (
      <div role="status" aria-label={label} className={cn('space-y-6', className)}>
        <Skeleton className="h-10 w-1/3" />
        <Skeleton className="h-4 w-2/3" />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: Math.max(count, 3) }).map((_, i) => (
            <Skeleton key={i} className="h-40 w-full" />
          ))}
        </div>
      </div>
    )
  }

  // list (default)
  return (
    <div role="status" aria-label={label} className={cn('space-y-3', className)}>
      {Array.from({ length: count }).map((_, i) => (
        <Skeleton key={i} className="h-16 w-full" />
      ))}
    </div>
  )
}
