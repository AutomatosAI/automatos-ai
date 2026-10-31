/**
 * One Marketplace page for everyone (PRD-270 §5.7) — the member cards, signed in and signed out.
 *
 * Signed in (no provider: the member's access) every card is today's: its counts show and its Add
 * installs through the member route. Signed out, a card shows none of the counts the public reads
 * don't carry (installs, enables, steps, prices, max output) and its Add goes through
 * `requireSignIn` instead of installing.
 */
import type { ReactNode } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }))
vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('sonner', () => ({ toast: Object.assign(vi.fn(), { success: vi.fn(), error: vi.fn() }) }))

import { MarketplaceAccessProvider } from '../access/marketplace-access'
import type { MarketplaceAccess } from '../access/marketplace-access'
import { MEMBER_SOURCE } from '../access/member-source'
import { LLMModelCard, routeActionPath } from '../llm-model-card'
import type { LLMModel } from '../llm-model-card'
import { AgentGridCard } from '../marketplace-agent-cards'
import type { MarketplaceAgent } from '../marketplace-agent-cards'
import { PlaybookGridCard } from '../marketplace-playbook-cards'
import { PluginCard } from '../marketplace-plugins-tab'
import type { PluginSummary } from '../marketplace-plugins-tab'

const requireSignIn = vi.fn(() => true)
const VISITOR: MarketplaceAccess = { signedIn: false, requireSignIn, source: MEMBER_SOURCE }

const MODEL: LLMModel = {
  id: 1, provider: 'anthropic', serving_provider: 'anthropic', model_id: 'claude-opus', display_name: 'Claude Opus',
  model_family: null, description: null, context_window: 200000, max_output_tokens: 32000, input_cost_per_1k: 0.015,
  output_cost_per_1k: 0.075, capabilities: {}, recommended_for: [], supports_functions: true, supports_vision: false,
  supports_streaming: true, status: 'active', category: null, tags: [], is_featured: false, is_default: false,
  requires_plan: null, install_count: 1234, is_installed: false,
}
const AGENT: MarketplaceAgent = {
  id: 7, name: 'Research Assistant', description: 'Finds sources.', creator_name: 'Automatos', category: 'research',
  install_count: 42, metadata: {},
}
const PLUGIN: PluginSummary = {
  id: 'p-1', slug: 'code-review', name: 'Code review', version: '1.2.0', description: null, category_id: null,
  category_name: null, tags: [], skills_count: 3, commands_count: 0, agents_count: 0, hooks_count: 0, token_estimate: 0,
  enable_count: 57, is_featured: false, security_status: null, approval_status: null, original_author: null, license: null,
  created_at: null,
}
const RECIPE = { id: 12, name: 'Weekly report', install_count: 1500, steps: [{ agent_id: 1 }, { agent_id: 2 }, { agent_id: 1 }] }
const noop = () => undefined

function renderAs(access: MarketplaceAccess | null, ui: ReactNode) {
  const client = new QueryClient()
  const tree = access ? <MarketplaceAccessProvider access={access}>{ui}</MarketplaceAccessProvider> : ui
  return render(<QueryClientProvider client={client}>{tree}</QueryClientProvider>)
}

const agentCard = () => (
  <AgentGridCard
    agent={AGENT} iconMappings={{}} isAdmin={false} onOpen={noop}
    admin={{ approvingId: null, deletingId: null, approve: noop, remove: noop }}
    installState={{ installingId: null, installedIds: new Set(), install: noop }}
  />
)
const pluginCard = () => (
  <PluginCard
    plugin={PLUGIN} index={0} isEnabled={false} isEnabling={false} onEnable={noop} onClick={noop} isAdmin={false}
    isApproving={false} isDeleting={false} isDeactivating={false} onApprove={noop} onDeactivate={noop} onDelete={noop}
  />
)
const playbookCard = () => (
  <PlaybookGridCard recipe={RECIPE} iconMappings={{}} isAdmin={false} installing={false} onView={noop} onInstall={noop} />
)

beforeEach(() => {
  api.post.mockReset()
  api.post.mockResolvedValue({})
  requireSignIn.mockClear()
})

afterEach(cleanup)

describe('signed in: today’s cards', () => {
  it('an LLM route shows its max output, prices and installs, and Add installs that route', async () => {
    renderAs(null, <LLMModelCard model={MODEL} />)

    expect(screen.getByText('Max Output')).toBeInTheDocument()
    expect(screen.getByText('Cost / 1M tokens')).toBeInTheDocument()
    expect(screen.getByText('1.2k')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Add to Workspace' }))
    await vi.waitFor(() => expect(api.post).toHaveBeenCalledWith(routeActionPath(MODEL, 'install')))
  })

  it('an agent, a plugin and a Playbook show their installs, enables and steps', () => {
    renderAs(null, <>{agentCard()}{pluginCard()}{playbookCard()}</>)

    expect(screen.getByText('42 installs')).toBeInTheDocument()
    expect(screen.getByText('57')).toBeInTheDocument()
    expect(screen.getByText('3 Steps')).toBeInTheDocument()
    expect(screen.getByText('2 Agents')).toBeInTheDocument()
    expect(screen.getByText('1.5k installs')).toBeInTheDocument()
  })
})

describe('signed out: what the public reads carry, and Add goes to sign-in', () => {
  it('an LLM route shows no max output, prices or installs, and Add asks for sign-in with its route', () => {
    renderAs(VISITOR, <LLMModelCard model={MODEL} />)

    expect(screen.getByText('Context')).toBeInTheDocument()
    expect(screen.queryByText('Max Output')).not.toBeInTheDocument()
    expect(screen.queryByText('Cost / 1M tokens')).not.toBeInTheDocument()
    expect(screen.queryByText('1.2k')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Add to Workspace' }))
    expect(requireSignIn).toHaveBeenCalledWith('llms', 'anthropic:claude-opus')
    expect(api.post).not.toHaveBeenCalled()
  })

  it('a free route still says Free', () => {
    renderAs(VISITOR, <LLMModelCard model={{ ...MODEL, is_free: true }} />)
    expect(screen.getAllByText('Free').length).toBeGreaterThan(0)
  })

  it('an agent, a plugin and a Playbook show no installs, enables or steps', () => {
    renderAs(VISITOR, <>{agentCard()}{pluginCard()}{playbookCard()}</>)

    expect(screen.getByText('Research Assistant')).toBeInTheDocument()
    expect(screen.queryByText(/installs/)).not.toBeInTheDocument()
    expect(screen.queryByText('57')).not.toBeInTheDocument()
    expect(screen.queryByText(/Steps/)).not.toBeInTheDocument()
  })
})
