/**
 * PRD-222 W2·S1b (US-024) — nav exposure gating.
 *
 * A nav item may declare `requiredExposure` (an exposure.nav key, e.g.
 * 'analytics' or 'team'). The server derives exposure from the workspace's plan
 * tier (PLAN_TIERS) and returns it on GET /api/workspaces/current; the sidebars
 * hide items the tier gates. Hidden ≠ deleted (D5): a hidden item is simply
 * absent from the rail — its route still resolves for deep links.
 */
import type { WorkspaceExposure } from '@/components/workspace-provider'

/**
 * Should a nav item be shown for this exposure?
 *
 * Fail-open: an item with no `requiredExposure` always shows, and an unknown
 * exposure (still loading, or a solo/local session with no profile) shows
 * everything — nav must never flash gated. Only an explicit `false` on the
 * required nav key hides the item.
 */
export function isNavItemVisible(
  requiredExposure: string | undefined,
  exposure: WorkspaceExposure | null | undefined,
): boolean {
  if (!requiredExposure) return true
  const nav = exposure?.nav
  if (!nav) return true
  return nav[requiredExposure] !== false
}
