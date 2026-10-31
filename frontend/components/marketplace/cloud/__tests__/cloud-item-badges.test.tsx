/**
 * PRD-270 W2 US-270-14b (§5.5, §8-Q3a, K19) — the From Automatos Cloud and Reviewed on Automatos Cloud
 * badges.
 *
 * - a Cloud agent, Playbook, package and plugin (by the key its install takes) carries both;
 * - a local item carries neither, and neither does any item while the read fails;
 * - on the hosted edition nothing renders and nothing is asked, with no QueryClientProvider needed;
 * - the word Official never appears;
 * - the seams carry them: the agent list row and grid card, the package card and the agent and
 *   Playbook detail header;
 * - a Sync with Automatos Cloud reads the Cloud items again.
 *
 * `@clerk/nextjs` is the double sync-button.test.tsx uses (the Sync's error words come from
 * lib/theme-host/api.ts, which imports it).
 */
import type { ReactElement } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, render, renderHook, screen, waitFor, within } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }))
const edition = vi.hoisted(() => ({ local: true }))
const toast = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn() }))

vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('@/lib/auth-edition', () => ({
  get isLocal() {
    return edition.local
  },
  get isSaaS() {
    return !edition.local
  },
}))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'ws-1', name: 'Acme', role: 'owner' } }),
}))
vi.mock('@/components/shared', () => ({ PremiumIcon: () => null }))
vi.mock('@/hooks/use-playbook-api', () => ({ useInstallPlaybookFromMarketplace: () => ({ mutateAsync: vi.fn() }) }))
vi.mock('sonner', () => ({ toast }))
vi.mock('@clerk/nextjs', () => ({
  useReverification: (fetcher: (...args: unknown[]) => Promise<unknown>) => fetcher,
}))

import { Dialog } from '@/components/ui/dialog'
import { cloudItemsKeys } from '@/hooks/use-cloud-items'
import { useCloudSync } from '@/hooks/use-cloud-sync'
import type { CloudItems } from '@/lib/marketplace/cloud-item-rows'

import { AgentGridCard, AgentListRow } from '../../marketplace-agent-cards'
import type { MarketplaceAgent } from '../../marketplace-agent-cards'
import type { MarketplaceItem } from '../../marketplace-homepage'
import { ItemModalHeader } from '../../marketplace-item-sections'
import { MarketplacePackagesTab } from '../../marketplace-packages-tab'
import type { MarketplacePackage } from '../../package-types'
import { CloudItemBadges } from '../cloud-item-badges'
import { CLOUD_BADGE, REVIEWED_BADGE } from '../item-copy'

const ITEMS_PATH = '/api/marketplace/cloud-sync/items'
const PLUGIN_ID = '3f2a8c1e-0b4d-4e6f-9a7b-1c2d3e4f5a6b'

function row(name: string) {
  return { externalId: `ext-${name}`, version: '1.0.0', revision: 'a'.repeat(64), name }
}

const ITEMS: CloudItems = {
  agents: [{ id: 7, ...row('Ledger Helper') }],
  playbooks: [{ id: 12, ...row('Weekly Close') }],
  packages: [{ slug: 'shop-starter', ...row('Shop Starter') }],
  plugins: [{ id: PLUGIN_ID, ...row('Invoice Tools') }],
}

function newClient() {
  return new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
}

function renderWithClient(ui: ReactElement, client = newClient()) {
  render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
  return client
}

function agent(id: number): MarketplaceAgent {
  return {
    id, name: `Agent ${id}`, description: 'Keeps the books', creator_name: 'Cloud Publisher',
    category: 'custom', install_count: 0, is_approved: true, metadata: {},
  }
}

function pkg(slug: string): MarketplacePackage {
  return {
    id: `id-${slug}`, slug, name: `Package ${slug}`, description: 'A team', vertical_tags: [],
    matching: {}, members: [], setup_manifest: {}, showcase: false,
  }
}

function item(id: number, type: string): MarketplaceItem {
  return {
    id, type, name: `Item ${id}`, description: 'Detail', creator_name: 'Cloud Publisher', tags: [],
    install_count: 0, is_featured: false, version: '1.0.0', metadata: {},
    created_at: '2026-10-30T10:00:00Z', updated_at: '2026-10-30T10:00:00Z',
  }
}

const noAdmin = { approvingId: null, deletingId: null, approve: vi.fn(), remove: vi.fn() }
const noInstall = { installingId: null, installedIds: new Set<number>(), install: vi.fn() }

beforeEach(() => {
  vi.clearAllMocks()
  edition.local = true
  api.get.mockResolvedValue(ITEMS)
})

async function expectBoth(container: HTMLElement) {
  expect(await within(container).findByText(CLOUD_BADGE)).toBeInTheDocument()
  expect(within(container).getByText(REVIEWED_BADGE)).toBeInTheDocument()
  expect(within(container).queryByText(/official/i)).not.toBeInTheDocument()
}

describe('the words', () => {
  it('are the PRD’s, and neither says Official', () => {
    expect(CLOUD_BADGE).toBe('From Automatos Cloud')
    expect(REVIEWED_BADGE).toBe('Reviewed on Automatos Cloud')
    expect(`${CLOUD_BADGE} ${REVIEWED_BADGE}`).not.toMatch(/official/i)
  })
})

describe('CloudItemBadges', () => {
  it.each([
    ['agent', 7], ['playbook', 12], ['package', 'shop-starter'], ['plugin', PLUGIN_ID],
  ] as const)('shows both badges on a Cloud %s', async (kind, id) => {
    renderWithClient(<div data-testid="item"><CloudItemBadges kind={kind} id={id} /></div>)

    await expectBoth(screen.getByTestId('item'))
    expect(api.get).toHaveBeenCalledWith(ITEMS_PATH)
  })

  it.each([
    ['agent', 8], ['agent', 12], ['playbook', 7], ['package', 'local-team'], ['plugin', 'shop-starter'],
  ] as const)('shows nothing on a local item (%s %s)', async (kind, id) => {
    renderWithClient(
      <>
        <div data-testid="cloud"><CloudItemBadges kind="agent" id={7} /></div>
        <div data-testid="local"><CloudItemBadges kind={kind} id={id} /></div>
      </>,
    )

    await expectBoth(screen.getByTestId('cloud'))
    expect(screen.getByTestId('local')).toBeEmptyDOMElement()
  })

  it('reads the items once for every card', async () => {
    renderWithClient(
      <>
        <CloudItemBadges kind="agent" id={7} />
        <CloudItemBadges kind="package" id="shop-starter" />
        <CloudItemBadges kind="plugin" id={PLUGIN_ID} />
      </>,
    )

    expect(await screen.findAllByText(CLOUD_BADGE)).toHaveLength(3)
    expect(api.get).toHaveBeenCalledTimes(1)
  })

  it('shows nothing while the read fails', async () => {
    api.get.mockRejectedValue(new Error('Not Found'))
    renderWithClient(<div data-testid="item"><CloudItemBadges kind="agent" id={7} /></div>)

    await waitFor(() => expect(api.get).toHaveBeenCalledWith(ITEMS_PATH))
    expect(screen.getByTestId('item')).toBeEmptyDOMElement()
    expect(screen.queryByText(CLOUD_BADGE)).not.toBeInTheDocument()
  })

  it('shows nothing and asks nothing on the hosted edition, with no QueryClientProvider', () => {
    edition.local = false
    render(<div data-testid="item"><CloudItemBadges kind="agent" id={7} /></div>)

    expect(screen.getByTestId('item')).toBeEmptyDOMElement()
    expect(api.get).not.toHaveBeenCalled()
  })
})

describe('the seams', () => {
  it('the agent list row and grid card carry them for a Cloud agent only', async () => {
    renderWithClient(
      <>
        <div data-testid="row"><AgentListRow agent={agent(7)} iconMappings={{}} isAdmin={false} onOpen={vi.fn()} /></div>
        <div data-testid="grid">
          <AgentGridCard agent={agent(7)} iconMappings={{}} isAdmin={false} admin={noAdmin}
            installState={noInstall} onOpen={vi.fn()} />
        </div>
        <div data-testid="local">
          <AgentGridCard agent={agent(8)} iconMappings={{}} isAdmin={false} admin={noAdmin}
            installState={noInstall} onOpen={vi.fn()} />
        </div>
      </>,
    )

    await expectBoth(screen.getByTestId('row'))
    await expectBoth(screen.getByTestId('grid'))
    expect(within(screen.getByTestId('local')).queryByText(CLOUD_BADGE)).not.toBeInTheDocument()
  })

  it('the package card carries them for a Cloud package only', async () => {
    renderWithClient(<MarketplacePackagesTab packages={[pkg('shop-starter'), pkg('local-team')]} />)

    const cards = await screen.findAllByTestId('package-card')
    const cloudCard = cards.find((card) => within(card).queryByText('Package shop-starter'))!
    const localCard = cards.find((card) => within(card).queryByText('Package local-team'))!
    await expectBoth(cloudCard)
    expect(within(localCard).queryByText(CLOUD_BADGE)).not.toBeInTheDocument()
  })

  it.each([
    ['an agent', 7, 'agent'],
    ['a Playbook', 12, 'recipe'],
  ])('the detail header carries them for %s from Automatos Cloud', async (_label, id, type) => {
    renderWithClient(
      <Dialog open>
        <div data-testid="header"><ItemModalHeader item={item(id, type)} installing={false} onInstall={vi.fn()} /></div>
      </Dialog>,
    )

    await expectBoth(screen.getByTestId('header'))
  })

  it('the detail header reads a Playbook’s id as a Playbook’s, never an agent’s', async () => {
    renderWithClient(
      <Dialog open>
        <div data-testid="cloud"><ItemModalHeader item={item(7, 'agent')} installing={false} onInstall={vi.fn()} /></div>
        <div data-testid="other"><ItemModalHeader item={item(7, 'recipe')} installing={false} onInstall={vi.fn()} /></div>
      </Dialog>,
    )

    await expectBoth(screen.getByTestId('cloud'))
    expect(within(screen.getByTestId('other')).queryByText(CLOUD_BADGE)).not.toBeInTheDocument()
  })
})

describe('after a Sync with Automatos Cloud', () => {
  it('the badges read the Cloud items again', async () => {
    const status = {
      enabled: true, cloudUrl: 'https://cloud.example.test', lastAttemptAt: null, lastSuccessAt: null,
      lastError: null, generatedAt: null, counts: { new: 1, updated: 0, unchanged: 0, refused: 0 },
      themes: 0, items: { agent: 0, playbook: 0, package: 0, plugin: 0 }, stale: false,
    }
    api.post.mockResolvedValue({ counts: status.counts, updatesApplied: 0, updatesFailed: 0, status })
    const client = renderWithClient(<div data-testid="item"><CloudItemBadges kind="agent" id={7} /></div>)
    await expectBoth(screen.getByTestId('item'))
    expect(api.get).toHaveBeenCalledTimes(1)
    const invalidate = vi.spyOn(client, 'invalidateQueries')
    const { result } = renderHook(() => useCloudSync(), {
      wrapper: ({ children }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>,
    })

    act(() => result.current.mutate())

    await waitFor(() => expect(api.get).toHaveBeenCalledTimes(2))
    expect(api.get).toHaveBeenLastCalledWith(ITEMS_PATH)
    expect(invalidate).toHaveBeenCalledWith({ queryKey: cloudItemsKeys.all })
  })
})
