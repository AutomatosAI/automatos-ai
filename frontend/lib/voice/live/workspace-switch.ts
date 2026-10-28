/**
 * Settings → Voice → Auto Live — who may flip the workspace's own switch: the same
 * `workspace:manage` gate `PUT /api/workspaces/current/voice-live` enforces (owner and
 * admin hold it; see `modules/policy/roles.py`'s `ROLE_PERMISSIONS`), on the
 * `canTurnOnSocials` pattern (components/deliverables/socials/socials-status.ts).
 */
import type { Workspace } from '@/components/workspace-provider'

const MANAGE_ROLES: ReadonlySet<Workspace['role']> = new Set<Workspace['role']>(['owner', 'admin'])

export function canManageVoiceLive(role: Workspace['role'] | undefined): boolean {
  return !!role && MANAGE_ROLES.has(role)
}
