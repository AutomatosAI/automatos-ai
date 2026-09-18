/** The Agents page's tabs — the values `?tab=` accepts and the Studio page tabs link to (PRD-244 W0). */
export const AGENT_TAB_VALUES = ['roster', 'org-chart', 'configuration', 'skills'] as const
export type AgentTab = (typeof AGENT_TAB_VALUES)[number]
