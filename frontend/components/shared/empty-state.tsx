'use client'

import type { ReactNode } from 'react'
import type { LucideIcon } from 'lucide-react'
import { cn } from '@/lib/utils'
import {
  EMPTY_STATE_MASCOT_SLOT,
  PlacementSlot,
  usePlacementBase,
} from '@/components/theme-host/placement-slot'

export interface EmptyStateProps {
  icon: LucideIcon
  title: string
  description: string
  action?: ReactNode
  className?: string
}

/**
 * EmptyState — the icon tile, a title, a description and an optional action.
 *
 * PRD-261 A-12b: the icon tile is the fallback of the placement slot `empty-state.mascot@1`, so
 * an active theme may stand its mascot there; with no theme the tile draws exactly as before.
 */
export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
  className,
}: EmptyStateProps) {
  const base = usePlacementBase()
  const iconTile = (
    <div className="w-16 h-16 rounded-2xl bg-secondary/50 border border-border/30 flex items-center justify-center mb-4">
      <Icon className="w-8 h-8 text-muted-foreground/60" />
    </div>
  )
  return (
    <div className={cn('flex flex-col items-center justify-center py-16 px-6 text-center', className)}>
      <PlacementSlot slot={EMPTY_STATE_MASCOT_SLOT} props={{ base, title, description }} fallback={iconTile} />
      <h3 className="text-lg font-semibold mb-1">{title}</h3>
      <p className="text-sm text-muted-foreground max-w-sm mb-6">{description}</p>
      {action}
    </div>
  )
}
