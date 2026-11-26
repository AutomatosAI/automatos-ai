'use client'

/**
 * PRD-277 US-504 — the Team page's waiting list in enterprise. TeamManagement renders this only
 * when the edition is enterprise (a build-time constant), so the waiting hook and, through it, the
 * step-up hook run in no other edition. The role is the API's (`GET /api/v1/auth/me`, through the
 * enterprise session); anyone but an install admin sees nothing and sends no request.
 */
import { isInstallAdmin, systemRoleOf, useEnterpriseSession } from '@/lib/enterprise/session-context'

import { EnterpriseWaitingPeople } from './enterprise-waiting-people'
import { useEnterpriseWaitingPeople } from './use-enterprise-waiting-people'

interface EnterpriseWaitingSectionProps {
  readonly workspaceId: string | null
  /** Called once a person is placed, so the members list shows them. */
  readonly onPlaced: () => void
}

export function EnterpriseWaitingSection({ workspaceId, onPlaced }: EnterpriseWaitingSectionProps) {
  const installAdmin = isInstallAdmin(systemRoleOf(useEnterpriseSession()))
  const waiting = useEnterpriseWaitingPeople(workspaceId, installAdmin, onPlaced)
  return installAdmin ? <EnterpriseWaitingPeople waiting={waiting} /> : null
}
