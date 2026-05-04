import type { StatusVariant } from '@/components/shared/status-badge'

const STATUS_MAP: Record<string, StatusVariant> = {
  // success
  completed: 'success',
  complete: 'success',
  done: 'success',
  passed: 'success',
  healthy: 'success',
  resolved: 'success',
  approved: 'success',
  verified: 'success',
  connected: 'success',
  online: 'success',
  enabled: 'success',
  installed: 'success',
  synced: 'success',

  // active / info
  active: 'active',
  running: 'active',
  in_progress: 'active',
  'in-progress': 'active',
  processing: 'active',
  executing: 'active',
  live: 'active',
  streaming: 'active',
  started: 'active',
  open: 'info',
  available: 'info',
  ready: 'info',
  deployed: 'info',

  // warning
  pending: 'warning',
  waiting: 'warning',
  queued: 'warning',
  paused: 'warning',
  delayed: 'warning',
  degraded: 'warning',
  attention: 'warning',
  stale: 'warning',
  expiring: 'warning',
  draft: 'warning',

  // error
  failed: 'error',
  error: 'error',
  offline: 'error',
  disconnected: 'error',
  rejected: 'error',
  blocked: 'error',
  critical: 'error',
  expired: 'error',
  broken: 'error',
  crashed: 'error',
  terminated: 'error',

  // neutral
  idle: 'neutral',
  inactive: 'neutral',
  disabled: 'neutral',
  archived: 'neutral',
  stopped: 'neutral',
  unknown: 'neutral',
  cancelled: 'neutral',
  canceled: 'neutral',
  skipped: 'neutral',
  none: 'neutral',

  // purple / agent
  agent: 'purple',
  skill: 'purple',
  tool: 'purple',
  plugin: 'purple',
  custom: 'purple',

  // primary
  featured: 'primary',
  premium: 'primary',
  highlighted: 'primary',
  pinned: 'primary',
}

export function getStatusVariant(status: string): StatusVariant {
  const normalized = status.toLowerCase().trim().replace(/\s+/g, '_')
  return STATUS_MAP[normalized] ?? 'neutral'
}

export const TRANSITION_CLASS = 'transition-all duration-[220ms]' as const
