'use client'

import { useWorkspace } from '@automatos/theme-sdk'

/**
 * Whether the platform offers Socials to this workspace (`socials.available` on GET
 * /api/workspaces/current), for the shelf's links into the Deliverables page's tabs (PRD-251 D1,
 * through the SDK's visibleDeliverableTabs). The SDK's workspace carries it as `socialsAvailable`
 * (lib/theme-sdk/context.ts), read as the host's own Deliverables page reads it; false until the
 * workspace is known.
 */
export function useSocialsOffered(): boolean {
  return useWorkspace().workspace?.socialsAvailable === true
}
