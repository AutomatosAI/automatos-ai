'use client'

import { StatusBadge, type StatusVariant } from '@/components/shared/status-badge'
import type { RunState } from '@/types/missions'
import { RUN_STATE_CONFIG } from '@/types/missions'

const STATE_TO_VARIANT: Record<RunState, StatusVariant> = {
  pending: 'neutral',
  planning: 'info',
  awaiting_approval: 'warning',
  running: 'primary',
  paused: 'neutral',
  verifying: 'info',
  awaiting_human: 'warning',
  completed: 'success',
  failed: 'error',
  cancelled: 'neutral',
}

interface MissionStatusBadgeProps {
  state: RunState
  size?: 'sm' | 'default'
  className?: string
}

export function MissionStatusBadge({ state, size = 'default', className }: MissionStatusBadgeProps) {
  const config = RUN_STATE_CONFIG[state]
  const variant = STATE_TO_VARIANT[state]
  const showDot = config.pulse ?? false

  return (
    <StatusBadge status={variant} dot={showDot} size={size} className={className}>
      {config.label}
    </StatusBadge>
  )
}
