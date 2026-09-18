/** The Deliverables page's tabs — the values `?tab=` accepts and the Studio page tabs link to (PRD-244 W0). */
export const DELIVERABLE_TABS = ['outputs', 'blogs', 'templates'] as const
export type DeliverableTab = (typeof DELIVERABLE_TABS)[number]
