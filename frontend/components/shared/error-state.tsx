'use client'

import type { ReactNode } from 'react'
import { AlertTriangle, type LucideIcon } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'

export interface ErrorStateProps {
  title?: string
  /** Explicit message; falls back to `error`'s message, then a generic line. */
  description?: string
  /** Raw error to derive a message from when `description` is absent. */
  error?: unknown
  icon?: LucideIcon
  onRetry?: () => void
  retryLabel?: string
  /** Custom action node; overrides the default retry button when provided. */
  action?: ReactNode
  className?: string
}

function messageFrom(error: unknown): string | undefined {
  if (error instanceof Error) return error.message
  if (typeof error === 'string') return error
  return undefined
}

/**
 * Canonical error surface (PRD-169 S2). One look for every failed load —
 * replaces the divergent inline "failed to fetch" blocks across focus pages.
 */
export function ErrorState({
  title = 'Something went wrong',
  description,
  error,
  icon: Icon = AlertTriangle,
  onRetry,
  retryLabel = 'Try again',
  action,
  className,
}: ErrorStateProps) {
  const message =
    description ?? messageFrom(error) ?? 'An unexpected error occurred. Please try again.'

  return (
    <div
      role="alert"
      className={cn('flex flex-col items-center justify-center py-16 px-6 text-center', className)}
    >
      <div className="w-16 h-16 rounded-2xl bg-destructive/10 border border-destructive/20 flex items-center justify-center mb-4">
        <Icon className="w-8 h-8 text-destructive" />
      </div>
      <h3 className="text-lg font-semibold mb-1">{title}</h3>
      <p className="text-sm text-muted-foreground max-w-sm mb-6">{message}</p>
      {action ?? (onRetry ? (
        <Button variant="outline" onClick={onRetry}>
          {retryLabel}
        </Button>
      ) : null)}
    </div>
  )
}
