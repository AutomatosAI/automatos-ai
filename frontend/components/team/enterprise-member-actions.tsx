'use client'

/**
 * PRD-277 US-504 — an install admin's two buttons on a member's row, in enterprise:
 *   End sessions  → POST /api/v1/auth/people/{user_id}/sessions/revoke (signed out everywhere)
 *   Deactivate    → POST /api/v1/auth/people/{user_id}/deactivate (no sign-in until reactivated)
 * Each asks first, and each needs a recent sign-in (step-up: the browser signs in again, and the
 * admin presses the button once more on return). TeamManagement renders this only in enterprise;
 * anyone but an install admin (the API's role, through the enterprise session) sees nothing. The
 * API checks the admin; the buttons only spare a request it would refuse.
 */
import { LogOut, UserX } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { deactivatePerson, endSessionsOf } from '@/lib/enterprise/auth-api'
import { actionErrorMessage } from '@/lib/enterprise/messages'
import { isInstallAdmin, systemRoleOf, useEnterpriseSession } from '@/lib/enterprise/session-context'
import { useEnterpriseStepUp } from '@/lib/enterprise/step-up'

import { displayName } from './team-format'
import type { TeamMember } from './team-types'

interface EnterpriseMemberActionsProps {
  readonly member: TeamMember
  readonly onChanged: () => void
}

export function EnterpriseMemberActions({ member, onChanged }: EnterpriseMemberActionsProps) {
  const installAdmin = isInstallAdmin(systemRoleOf(useEnterpriseSession()))
  const endSessions = useEnterpriseStepUp(endSessionsOf)
  const deactivate = useEnterpriseStepUp(deactivatePerson)
  const userId = Number(member.user_id)
  const who = displayName(member)
  if (!installAdmin) return null

  const run = async (question: string, action: (id: number) => Promise<unknown>) => {
    if (!Number.isSafeInteger(userId) || !confirm(question)) return
    try {
      await action(userId)
      onChanged()
    } catch (error) {
      console.error('[team] the access change failed', error)
      alert(actionErrorMessage(error))
    }
  }

  return (
    <>
      <Button
        variant="ghost"
        size="icon"
        title="End sessions"
        aria-label={`End ${who}'s sessions`}
        onClick={() => void run(`Sign ${who} out everywhere?`, endSessions)}
        className="opacity-0 group-hover:opacity-100 focus:opacity-100 text-muted-foreground hover:text-foreground transition-all"
      >
        <LogOut className="w-4 h-4" />
      </Button>
      <Button
        variant="ghost"
        size="icon"
        title="Deactivate"
        aria-label={`Deactivate ${who}`}
        onClick={() => void run(`Deactivate ${who}? They are signed out and cannot sign in until reactivated.`, deactivate)}
        className="opacity-0 group-hover:opacity-100 focus:opacity-100 text-muted-foreground hover:text-red-400 hover:bg-red-400/10 transition-all"
      >
        <UserX className="w-4 h-4" />
      </Button>
    </>
  )
}
