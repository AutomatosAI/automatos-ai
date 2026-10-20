// @ts-check
// PRD-261 A-12d: the SDK agents-page hello theme. Besides its frame, which only wraps the page, it
// registers its own page on /agents and draws it from the SDK's pages/agents area alone: the
// roster from useAgents() and the counts from useAgentStats(), and its Refresh button asks the
// roster again through the query's refetch. It imports nothing else of the host. It reaches the
// browser only through the lane's route interception (e2e/browser/specs/pages-agents.spec.ts,
// through support/hello-theme.ts), and its bare imports resolve through the host's import map.
import { jsx, jsxs } from 'react/jsx-runtime'
import { defineTheme, useAgentStats, useAgents } from '@automatos/theme-sdk'

const AGENTS_ROUTE = '/agents'

/** @param {{ children?: import('react').ReactNode }} props */
function AgentsFrame({ children }) {
  return jsx('div', { 'data-testid': 'hello-a12d-frame', className: 'hello-a12d-frame', children })
}

/** @param {{ total: number | null, active: number | null }} props */
function Counts({ total, active }) {
  return jsxs('p', {
    children: [
      jsx('span', { 'data-testid': 'hello-a12d-total', children: total === null ? '' : String(total) }),
      ' agents, ',
      jsx('span', { 'data-testid': 'hello-a12d-active', children: active === null ? '' : String(active) }),
      ' active',
    ],
  })
}

/** @param {{ agents: ReadonlyArray<{ id: number, name: string, status: string }> }} props */
function Roster({ agents }) {
  return jsx('ul', {
    'data-testid': 'hello-a12d-roster',
    children: agents.map((agent) =>
      jsx('li', { 'data-agent-id': String(agent.id), 'data-status': agent.status, children: agent.name }, agent.id),
    ),
  })
}

function HelloAgentsPage() {
  const agents = useAgents()
  const stats = useAgentStats()
  return jsxs('section', {
    'data-testid': 'hello-a12d-page',
    'data-loading': agents.isLoading ? 'true' : 'false',
    className: 'hello-a12d-page',
    'aria-label': 'Hello agents',
    children: [
      jsx('h2', { children: 'Hello agents' }),
      jsx(Counts, { total: stats.data ? stats.data.total_agents : null, active: stats.data ? stats.data.active_agents : null }),
      jsx(Roster, { agents: agents.data ?? [] }),
      jsx('button', {
        type: 'button',
        'data-testid': 'hello-a12d-refresh',
        onClick: () => void agents.refetch(),
        children: 'Refresh',
      }),
    ],
  })
}

export default defineTheme({
  register(host) {
    host.registerSlot('frame@1', AgentsFrame)
    host.registerPage(AGENTS_ROUTE, HelloAgentsPage)
  },
  unregister() {},
})
