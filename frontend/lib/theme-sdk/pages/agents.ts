/**
 * PRD-261 A-10a — SDK area `pages/agents`: the data the /agents page reads. Stubbed here; A-12d fills the bodies.
 * Until then every name throws "not in this SDK build" (internal/stub.ts), so a theme built
 * against a later SDK fails loudly. The names are the plan's (surface/pages-agents.json): the
 * area's lane replaces bodies, never names, unless the A-10c surface map changes first.
 */
import { notInThisBuild } from '../internal/stub'

const AREA = 'pages/agents'

export const useAgents = notInThisBuild(AREA, 'useAgents')
export const useAgentStats = notInThisBuild(AREA, 'useAgentStats')
