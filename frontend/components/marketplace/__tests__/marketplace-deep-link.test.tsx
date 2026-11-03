/**
 * PRD-270 US-270-09b (§5.7, §8-Q8, K22) — the member Marketplace's one deep link: a
 * `?tab=<tab>&item=<key>&add=1` link opens that tab's own detail modal on the item, read through
 * the route the tab already calls; a skill's link sets the search instead; an unknown item opens
 * nothing; closing strips `item` and `add`; nothing installs until the click.
 *
 * The modals are doubles that show what they were opened on and expose their add and close: this
 * test is about which modal opens on what, and what the deep link itself sends. The modals' own
 * installs are theirs, unchanged, and run on a click only.
 */
import type { ReactNode } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const nav = vi.hoisted(() => ({ params: new URLSearchParams(), replace: vi.fn() }))
const api = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }))
const toast = vi.hoisted(() => Object.assign(vi.fn(), { error: vi.fn(), success: vi.fn() }))

vi.mock('next/navigation', () => ({
  useSearchParams: () => nav.params,
  useRouter: () => ({ replace: nav.replace }),
}))
vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('sonner', () => ({ toast }))

interface Closable {
  onClose: () => void
}

function CloseButton({ onClose }: Closable) {
  return <button onClick={() => onClose()}>Close</button>
}

vi.mock('@/components/marketplace/marketplace-item-modal', () => ({
  MarketplaceItemModal: ({ itemId, itemType, onClose }: Closable & { itemId: number; itemType: string }) => (
    <div data-testid="item-modal">
      {`${itemType} ${itemId}`}
      <CloseButton onClose={onClose} />
    </div>
  ),
}))
vi.mock('@/components/marketplace/package-detail-modal', () => ({
  PackageDetailModal: ({ pkg, onClose }: Closable & { pkg: { slug: string } }) => (
    <div data-testid="package-modal">
      {pkg.slug}
      <CloseButton onClose={onClose} />
    </div>
  ),
}))
vi.mock('@/components/marketplace/marketplace-app-details-modal', () => ({
  MarketplaceAppDetailsModal: (props: Closable & { open: boolean; app: { name: string } | null; onConnect: () => void; isConnected: boolean }) =>
    props.open && props.app ? (
      <div data-testid="app-modal">
        {`${props.app.name} ${props.isConnected ? 'connected' : 'not connected'}`}
        <button onClick={props.onConnect}>Add to Workspace</button>
        <CloseButton onClose={props.onClose} />
      </div>
    ) : null,
}))
vi.mock('@/components/marketplace/marketplace-plugin-detail-modal', () => ({
  MarketplacePluginDetailModal: (props: Closable & { open: boolean; pluginId: string; isEnabled: boolean; onEnable: () => void }) =>
    props.open ? (
      <div data-testid="plugin-modal">
        {`${props.pluginId} ${props.isEnabled ? 'enabled' : 'not enabled'}`}
        <button onClick={props.onEnable}>Enable</button>
        <CloseButton onClose={props.onClose} />
      </div>
    ) : null,
}))
vi.mock('@/components/marketplace/llm-model-detail-modal', () => ({
  LLMModelDetailModal: (props: { model: { model_id: string } | null; open: boolean; onOpenChange: (open: boolean) => void }) =>
    props.open && props.model ? (
      <div data-testid="llm-modal">
        {props.model.model_id}
        <CloseButton onClose={() => props.onOpenChange(false)} />
      </div>
    ) : null,
}))

import { MarketplaceDeepLink } from '../deep-link/marketplace-deep-link'
import { COULD_NOT_OPEN } from '../deep-link/deep-link-openers'

const WORKSPACE = 'ws-1'
const PLUGIN_ID = '0f8fad5b-d9cb-469f-a165-70867728950e'
const FREE_ROUTE = 'meta-llama/llama-3.1-70b:free'

/** The member routes the tabs call, as the server answers them; anything else is a 404. */
function answer(path: string): unknown {
  const url = new URL(path, 'http://core.test')
  const query = url.searchParams
  switch (url.pathname) {
    case '/api/marketplace/items/12':
      return query.get('type') === 'agent' ? { id: 12, name: 'Writer' } : undefined
    case '/api/marketplace/items/7':
      return query.get('type') === 'recipe' ? { id: 7, name: 'Weekly digest' } : undefined
    case '/api/marketplace/packages/growth-kit':
      return { id: 'p-1', slug: 'growth-kit', name: 'Growth kit', members: [], setup_manifest: {} }
    case '/api/tools/marketplace':
      return query.get('search') === 'github'
        ? { apps: [{ app_name: 'GITHUB_ENTERPRISE', display_name: 'GitHub Enterprise' }, { app_name: 'GITHUB', display_name: 'GitHub', is_connected: false }] }
        : { apps: [] }
    case '/api/tools/connected':
      return { apps: [] }
    case `/api/marketplace/plugins/${PLUGIN_ID}`:
      return { id: PLUGIN_ID, name: 'Linear' }
    case `/api/workspaces/${WORKSPACE}/plugins`:
      return { items: [] }
    case '/api/marketplace/llm/catalog':
      return query.get('provider') === 'openrouter'
        ? {
            models: [
              { model_id: 'meta-llama/llama-3.1-70b', serving_provider: 'openrouter', provider: 'meta-llama' },
              { model_id: FREE_ROUTE, serving_provider: 'openrouter', provider: 'meta-llama' },
            ],
          }
        : { models: [] }
    default:
      return undefined
  }
}

function notFound(): Error & { status: number } {
  return Object.assign(new Error('Not found'), { status: 404 })
}

function renderLink(link: Record<string, string>, onSearch = vi.fn()) {
  nav.params = new URLSearchParams(link)
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )
  render(<MarketplaceDeepLink workspaceId={WORKSPACE} onSearch={onSearch} />, { wrapper })
  return onSearch
}

/** Lets every pending read answer. */
async function settle() {
  for (let round = 0; round < 3; round += 1) {
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
  }
}

const MODALS = ['item-modal', 'package-modal', 'app-modal', 'plugin-modal', 'llm-modal']

function expectNoModal() {
  for (const id of MODALS) expect(screen.queryByTestId(id)).not.toBeInTheDocument()
}

beforeEach(() => {
  nav.params = new URLSearchParams()
  nav.replace.mockReset()
  api.get.mockReset()
  api.post.mockReset()
  toast.mockReset()
  toast.error.mockReset()
  api.get.mockImplementation(async (path: string) => {
    const found = answer(path)
    if (found === undefined) throw notFound()
    return found
  })
})

afterEach(() => {
  // Nothing installs on load: every request a link sent was a read.
  expect(api.post).not.toHaveBeenCalled()
})

describe('each tab’s item opens its modal', () => {
  it.each([
    ['agents', '12', 'item-modal', 'agent 12'],
    ['recipes', '7', 'item-modal', 'recipe 7'],
    ['playbooks', '7', 'item-modal', 'recipe 7'],
    ['packages', 'growth-kit', 'package-modal', 'growth-kit'],
    ['tools', 'github', 'app-modal', 'GITHUB not connected'],
    ['plugins', PLUGIN_ID, 'plugin-modal', `${PLUGIN_ID} not enabled`],
    ['llms', `openrouter:${FREE_ROUTE}`, 'llm-modal', FREE_ROUTE],
  ])('?tab=%s&item=%s opens %s on it', async (tab, item, modal, shown) => {
    renderLink({ tab, item, add: '1' })

    expect(await screen.findByTestId(modal)).toHaveTextContent(shown)
    expect(toast.error).not.toHaveBeenCalled()
  })

  it('reads an LLM route through the LLMs tab’s catalogue read, its key split and encoded', async () => {
    renderLink({ tab: 'llms', item: `openrouter:${FREE_ROUTE}`, add: '1' })
    await screen.findByTestId('llm-modal')

    expect(api.get.mock.calls.map(([path]) => path)).toEqual([
      '/api/marketplace/llm/catalog?provider=openrouter&search=meta-llama%2Fllama-3.1-70b%3Afree&limit=200',
    ])
  })

  it('reads an application through the Applications tab’s catalogue and workspace-tools reads', async () => {
    renderLink({ tab: 'tools', item: 'github', add: '1' })
    await screen.findByTestId('app-modal')
    await settle()

    expect(api.get.mock.calls.map(([path]) => path)).toEqual([
      '/api/tools/marketplace?search=github&limit=1000',
      '/api/tools/connected',
    ])
  })

  it('an application active in this workspace opens as connected, as its tab shows it', async () => {
    api.get.mockImplementation(async (path: string) =>
      path === '/api/tools/connected' ? { apps: [{ app_name: 'github', status: 'active' }] } : answer(path),
    )
    renderLink({ tab: 'tools', item: 'github', add: '1' })

    expect(await screen.findByText('GITHUB connected')).toBeInTheDocument()
  })

  it('opens on the item alone, without add=1 (a shared link)', async () => {
    renderLink({ tab: 'packages', item: 'growth-kit' })

    expect(await screen.findByTestId('package-modal')).toHaveTextContent('growth-kit')
  })

  it('a skill’s link sets the Marketplace search to its name, once, then strips the link', async () => {
    const onSearch = renderLink({ tab: 'skills', item: 'pdf-tools', add: '1' })

    await waitFor(() => expect(onSearch).toHaveBeenCalledWith('pdf-tools'))
    expect(onSearch).toHaveBeenCalledTimes(1)
    expect(nav.replace).toHaveBeenCalledWith('/marketplace?tab=skills')
    expect(api.get).not.toHaveBeenCalled()
    expectNoModal()
  })
})

interface UnopenedCase {
  readonly name: string
  readonly tab: string
  readonly item: string
  /** Whether the key was well-formed, so its tab's route was asked (and answered not found). */
  readonly asked: boolean
}

const UNOPENED: readonly UnopenedCase[] = [
  { name: 'an agent id that is not a number', tab: 'agents', item: 'writer', asked: false },
  { name: 'an agent that is not listed', tab: 'agents', item: '999', asked: true },
  { name: 'a package that is not listed', tab: 'packages', item: 'nope', asked: true },
  { name: 'a package key with a path in it', tab: 'packages', item: '../admin', asked: false },
  { name: 'an application the catalogue does not hold', tab: 'tools', item: 'gitlab', asked: true },
  { name: 'a plugin key that is not a UUID', tab: 'plugins', item: '../../admin/plugins', asked: false },
  { name: 'a plugin that is not listed', tab: 'plugins', item: '11111111-2222-4333-8444-555555555555', asked: true },
  { name: 'an LLM key that is not a route', tab: 'llms', item: 'gpt-4o', asked: false },
  { name: 'an LLM route the catalogue does not hold', tab: 'llms', item: 'openrouter:not/here', asked: true },
]

interface UnreadCase {
  readonly name: string
  readonly link: Record<string, string>
}

const UNREAD: readonly UnreadCase[] = [
  { name: 'no item', link: { tab: 'agents', add: '1' } },
  { name: 'a Themes link (the Themes tab opens its own)', link: { tab: 'themes', item: 'acme.aurora', add: '1' } },
  { name: 'a link with no tab (Themes)', link: { item: 'acme.aurora', add: '1' } },
  { name: 'an overlong item', link: { tab: 'packages', item: 'a'.repeat(257), add: '1' } },
]

describe('what opens nothing', () => {
  it.each(UNOPENED)('$name', async ({ tab, item, asked }) => {
    renderLink({ tab, item, add: '1' })
    await settle()

    expectNoModal()
    expect(api.get.mock.calls.length > 0).toBe(asked)
    expect(toast.error).not.toHaveBeenCalled()
  })

  it.each(UNREAD)('$name: no read at all', async ({ link }) => {
    renderLink(link)
    await settle()

    expectNoModal()
    expect(api.get).not.toHaveBeenCalled()
    expect(nav.replace).not.toHaveBeenCalled()
  })

  it('a failed read says so and opens nothing', async () => {
    api.get.mockRejectedValue(Object.assign(new Error('Internal server error'), { status: 500 }))
    renderLink({ tab: 'packages', item: 'growth-kit', add: '1' })

    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(COULD_NOT_OPEN))
    expectNoModal()
  })
})

describe('closing the modal', () => {
  it.each([
    ['agents', '12', 'item-modal'],
    ['packages', 'growth-kit', 'package-modal'],
    ['tools', 'github', 'app-modal'],
    ['plugins', PLUGIN_ID, 'plugin-modal'],
    ['llms', `openrouter:${FREE_ROUTE}`, 'llm-modal'],
  ])('strips item and add from a ?tab=%s link, the tab kept', async (tab, item, modal) => {
    renderLink({ tab, item, add: '1' })
    fireEvent.click(await screen.findByRole('button', { name: 'Close' }))

    expect(nav.replace).toHaveBeenCalledWith(`/marketplace?tab=${tab}`)
    expect(screen.queryByTestId(modal)).not.toBeInTheDocument()
  })
})

describe('nothing installs until the click', () => {
  it('an application is added only when Add to Workspace is clicked', async () => {
    renderLink({ tab: 'tools', item: 'github', add: '1' })
    await screen.findByTestId('app-modal')
    await settle()
    expect(api.post).not.toHaveBeenCalled()

    api.post.mockResolvedValue({ status: 'added' })
    fireEvent.click(screen.getByRole('button', { name: 'Add to Workspace' }))

    await waitFor(() => expect(toast).toHaveBeenCalledWith('Added to Workspace', expect.anything()))
    expect(api.post).toHaveBeenCalledTimes(1)
    expect(api.post).toHaveBeenCalledWith('/api/tools/add-to-workspace', { app_name: 'GITHUB' })
    // The workspace's tools are read again, so the Applications tab behind shows the add too.
    await waitFor(() => expect(api.get.mock.calls.filter(([path]) => path === '/api/tools/connected')).toHaveLength(2))
    api.post.mockClear() // the click's request, made; afterEach checks the load sent none
  })

  it('a plugin is enabled only when Enable is clicked, then shows as enabled', async () => {
    renderLink({ tab: 'plugins', item: PLUGIN_ID, add: '1' })
    await screen.findByTestId('plugin-modal')
    await settle()
    expect(api.post).not.toHaveBeenCalled()

    api.post.mockResolvedValue({ success: true })
    fireEvent.click(screen.getByRole('button', { name: 'Enable' }))

    expect(await screen.findByText(`${PLUGIN_ID} enabled`)).toBeInTheDocument()
    expect(api.post).toHaveBeenCalledTimes(1)
    expect(api.post).toHaveBeenCalledWith(`/api/workspaces/${WORKSPACE}/plugins`, { plugin_id: PLUGIN_ID })
    api.post.mockClear()
  })

  it('a plugin already enabled here opens as enabled', async () => {
    api.get.mockImplementation(async (path: string) =>
      path === `/api/workspaces/${WORKSPACE}/plugins` ? { items: [{ plugin_id: PLUGIN_ID }] } : answer(path),
    )
    renderLink({ tab: 'plugins', item: PLUGIN_ID, add: '1' })

    expect(await screen.findByText(`${PLUGIN_ID} enabled`)).toBeInTheDocument()
  })
})
