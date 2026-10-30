'use client'

import { StatusBadge, type StatusVariant } from '@/components/shared/status-badge'
import { RowStatusLabel } from '@/components/proactive/row-status'
import { missionRetryTarget } from '@/components/proactive/use-row-retry'
import { displayStateLabel } from '@/lib/proactive/display-state'
import type { MissionResponse, RunState } from '@/types/missions'
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

/**
 * PRD-268 US-W04: a Mission card's status. When the server says where the Mission stands, it is drawn in the words
 * every surface uses (StatusLabel: glyph, word, ▸ next step, Try again, in Classic's badge); otherwise, for a state
 * the vocabulary leaves unnamed (pending, paused, cancelled), its run state's badge as before.
 */
export function MissionRowStatus({ mission, title }: { mission: MissionResponse; title?: string }) {
  if (displayStateLabel(mission.display_state) === null) return <MissionStatusBadge state={mission.state} size="sm" />
  return <RowStatusLabel row={mission} target={missionRetryTarget(mission)} title={title ?? mission.goal} />
}
