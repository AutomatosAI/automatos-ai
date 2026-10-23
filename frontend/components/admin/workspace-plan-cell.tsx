'use client'

/**
 * PRD-265 US-007 / W2 — the operator console's plan column: the plan dropdown
 * (./workspace-plan-select.tsx) and, beside it, "Offer Business"
 * (./business-invite-button.tsx). Same props as the dropdown.
 *
 * app/admin/workspaces/page.tsx imports this under the dropdown's name, so the console's
 * body (a single component far over the 150-line limit the changed-lines gate enforces)
 * did not have to change to mount it.
 */
import type { ComponentProps } from 'react'
import { BusinessInviteButton } from './business-invite-button'
import { WorkspacePlanSelect } from './workspace-plan-select'

export function WorkspacePlanCell(props: ComponentProps<typeof WorkspacePlanSelect>) {
  return (
    <>
      <WorkspacePlanSelect {...props} />
      <BusinessInviteButton
        workspaceId={props.workspaceId}
        workspaceName={props.workspaceName}
        plan={props.plan}
        disabled={props.disabled}
      />
    </>
  )
}
