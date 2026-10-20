'use client'

// PRD-261 A-14b, seam item: the one core import left in Play's shelf. lib/__tests__/prd261-port-deliverables.test.ts
// allows exactly this import in exactly this file, and fails when the allowance outlives the import.
import { useWorkspace } from '@/components/workspace-provider'

/**
 * Whether the platform offers Socials to this workspace (`socials.available` on GET
 * /api/workspaces/current), for the shelf's links into the Deliverables page's tabs (PRD-251 D1,
 * through the SDK's visibleDeliverableTabs).
 *
 * The theme SDK has no name for it yet: `context.useWorkspace` hands a theme the workspace's id,
 * name, slug and plan only (lib/theme-sdk/context.ts), and no other planned name carries it
 * (lib/theme-sdk/surface/*.json). A-14b adds no SDK name of its own (PA-16), so the shelf reads the
 * host's provider here, in one place, until the seam owner plans one (for example a
 * `socialsAvailable` field on `HostWorkspace`, with its sdk-surface-map.md row and .d.ts).
 */
export function useSocialsOffered(): boolean {
  const { workspace } = useWorkspace()
  return workspace?.socials?.available === true
}
