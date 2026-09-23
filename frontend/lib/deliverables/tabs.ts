/** The Deliverables page's tabs — the values `?tab=` accepts and the Studio page tabs link to (PRD-244 W0). */
export const DELIVERABLE_TABS = ['outputs', 'blogs', 'templates', 'socials'] as const
export type DeliverableTab = (typeof DELIVERABLE_TABS)[number]

/**
 * PRD-251 D1: the Socials tab exists only while the platform offers Socials
 * (`socials.available` on GET /api/workspaces/current). Both shells render
 * exactly these tabs.
 */
export function visibleDeliverableTabs(socialsAvailable: boolean): DeliverableTab[] {
  return DELIVERABLE_TABS.filter((tab) => tab !== 'socials' || socialsAvailable)
}

/** `?tab=` → a tab the caller can see. Anything else — an unknown value, or
 * `socials` while Socials is unavailable — is Outputs. */
export function resolveDeliverableTab(param: string | null, socialsAvailable: boolean): DeliverableTab {
  const visible: readonly string[] = visibleDeliverableTabs(socialsAvailable)
  return param && visible.includes(param) ? (param as DeliverableTab) : 'outputs'
}
