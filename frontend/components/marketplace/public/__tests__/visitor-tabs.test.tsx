/**
 * PRD-270 US-270-08b (§5.7, K20) — the signed-out visitor's tabs.
 *
 * - every tab of the member strip renders from the public answers (doubles of the hosted
 *   edition's catalogue and its public reads): the Themes storefront's hero from the catalogue with
 *   its web and phone screenshots, and each other tab's grid, its item detail and Add to my
 *   workspace, which goes to sign-in carrying the item;
 * - nothing workspace-scoped: no "In this workspace", update badge, "Publish theme", submit or
 *   "Installed in this workspace", and the workspace is never read;
 * - every request is a public read through apiClient.get (no install, no member route);
 * - the search filters on the client; Show more reads the next page; a shared link opens its item.
 *
 * The apiClient double has only `get`, `getBaseUrl` and a `post` that must stay uncalled, so any
 * member call would fail the render. The Clerk, style and toast doubles are the storefront test's.
 */
import type { ReactNode } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const nav = vi.hoisted(() => ({ params: new URLSearchParams() }))
const api = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn(), getBaseUrl: vi.fn() }))
const useWorkspace = vi.hoisted(() => vi.fn())

vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('next/navigation', () => ({ useSearchParams: () => nav.params }))
vi.mock('next/link', () => ({
  default: ({ href, children, ...rest }: { href: string; children: ReactNode }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}))
vi.mock('@/components/workspace-provider', () => ({ useWorkspace }))
vi.mock('@/contexts/ui-style-context', () => ({ useUiStyleOptional: () => ({ setStyle: vi.fn() }) }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@clerk/nextjs', () => ({ useReverification: (fetcher: unknown) => fetcher }))

import { SIGNED_BY_AUTOMATOS } from '@/components/settings/themes/theme-labels'
import { CATALOG_ENDPOINT } from '@/lib/marketplace/public-catalog'
import { marketplaceItemPath, signInUrl } from '@/lib/marketplace/return-after-sign-in'
import {
  ARCADE_DIGEST,
  ARCADE_NEW,
  ARCADE_OLD,
  BASE,
  HUD,
  catalog,
  filesOf,
  publicAnswer,
  publicEntry,
} from '@/lib/marketplace/__tests__/public-fixtures'

import {
  ADD_TO_WORKSPACE,
  INSTALLED_BADGE,
  INSTALLED_HEADING,
  MANAGERS_ADD,
  UPDATE_BADGE,
} from '../../themes/listing-format'
import { PUBLISH_THEME } from '../../themes/publish-theme'
import {
  DETAILS,
  MAKE_YOUR_OWN_BODY,
  MAKE_YOUR_OWN_VISITOR_BODY,
  OPEN_DESIGN_KIT,
  SUBMIT_PACKAGE,
} from '../../themes/storefront/copy'
import { NO_ITEM_MATCHES, SHOW_MORE, itemDetails } from '../copy'
import { openedMarketplaceTab } from '../../category-tabs'
import { VisitorTabs } from '../visitor-tabs'

/** The visitor's Add link: sign-in, then back on the item (US-270-09a). */
const signInToAdd = (tab: string, item: string) => signInUrl(marketplaceItemPath(tab, item))
const ARCADE_FILES = `${BASE}${filesOf(ARCADE_DIGEST)}`
const PUBLIC_REQUEST = /^\/api\/(marketplace\/catalog|public\/marketplace\/[a-z]+\?limit=100&offset=\d+)$/
const READ = /^\/api\/public\/marketplace\/([a-z]+)\?limit=100&offset=(\d+)$/

const WRITER = publicEntry({ key: 'writer', name: 'Writer', description: 'Drafts posts.', category: 'Content', tags: ['writing'], requiredApps: [] })

const PAGES: Readonly<Record<string, unknown>> = {
  'agents:0': publicAnswer('agents', [publicEntry(), WRITER]),
  'playbooks:0': publicAnswer('playbooks', [
    publicEntry({ key: 'weekly-report', name: 'Weekly report', category: 'Reporting', icon: null, requiredApps: ['gmail'] }),
  ]),
  'packages:0': publicAnswer('packages', [
    publicEntry({ key: 'socials', name: 'Socials', members: [{ type: 'agent', name: 'Social Writer' }] }),
  ]),
  'applications:0': publicAnswer('applications', [
    publicEntry({ key: 'notion', name: 'Notion', icon: 'https://logos.composio.dev/api/notion', actionCount: 12, triggerCount: 2 }),
  ]),
  'llms:0': publicAnswer('llms', [
    publicEntry({ key: 'anthropic:claude-opus', name: 'Claude Opus', publisher: 'Anthropic', contextWindow: 200000, priceTier: 'high' }),
  ]),
  'plugins:0': publicAnswer('plugins', [publicEntry({ key: 'code-review', name: 'Code review', version: '1.2.0', skillsCount: 3 })]),
  'skills:0': publicAnswer('skills', [publicEntry({ key: 'pdf', name: 'PDF', version: '1.0.0' })]),
}

function answerFrom(pages: Readonly<Record<string, unknown>>, url: string): Promise<unknown> {
  if (url === CATALOG_ENDPOINT) return Promise.resolve(catalog([ARCADE_NEW, ARCADE_OLD, HUD]))
  const read = READ.exec(url)
  const page = read ? pages[`${read[1]}:${read[2]}`] : undefined
  return page ? Promise.resolve(page) : Promise.reject(new Error(`no double for ${url}`))
}

function renderTabs(searchQuery = '') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const view = render(
    <QueryClientProvider client={client}>
      <VisitorTabs searchQuery={searchQuery} />
    </QueryClientProvider>,
  )
  const rerenderWith = (query: string) =>
    view.rerender(
      <QueryClientProvider client={client}>
        <VisitorTabs searchQuery={query} />
      </QueryClientProvider>,
    )
  return { ...view, rerenderWith }
}

function chooseTab(name: string) {
  const tab = screen.getByRole('tab', { name })
  fireEvent.mouseDown(tab) // Radix Tabs activates on mouse down
  fireEvent.click(tab)
}

function requestedUrls(): string[] {
  return api.get.mock.calls.map(([url]) => String(url))
}

function expectNothingWorkspaceScoped() {
  for (const words of [INSTALLED_BADGE, UPDATE_BADGE, PUBLISH_THEME, SUBMIT_PACKAGE, INSTALLED_HEADING, MANAGERS_ADD]) {
    expect(screen.queryByText(words)).not.toBeInTheDocument()
  }
}

beforeEach(() => {
  nav.params = new URLSearchParams()
  api.get.mockReset()
  api.post.mockReset()
  api.getBaseUrl.mockReset()
  api.get.mockImplementation((url: string) => answerFrom(PAGES, url))
  api.getBaseUrl.mockReturnValue(BASE)
  useWorkspace.mockReset()
})

afterEach(() => {
  // No member route, no install, no workspace read, on any tab.
  for (const url of requestedUrls()) expect(url).toMatch(PUBLIC_REQUEST)
  expect(api.post).not.toHaveBeenCalled()
  expect(useWorkspace).not.toHaveBeenCalled()
  cleanup()
})

describe('the visitor’s tabs (PRD-270 US-270-08b)', () => {
  it('draws the member strip: Themes, Packages, Applications, Agents, Playbooks, LLMs, Capabilities', () => {
    renderTabs()

    expect(screen.getAllByRole('tab').map((tab) => tab.textContent)).toEqual([
      'Themes', 'Packages', 'Applications', 'Agents', 'Playbooks', 'LLMs', 'Capabilities',
    ])
    expect(screen.getByRole('tab', { name: 'Themes' })).toHaveAttribute('aria-selected', 'true')
  })
})

describe('Themes: the storefront over the catalogue', () => {
  it('shows the hero from the catalogue with its web and phone screenshots, and the strip of every theme', async () => {
    renderTabs()

    const hero = await screen.findByTestId('storefront-hero')
    expect(within(hero).getByTestId('storefront-hero-name')).toHaveTextContent('Arcade')
    expect(within(hero).getByRole('img', { name: 'Arcade on the web, Default' })).toHaveAttribute(
      'src', `${ARCADE_FILES}/assets/previews/default.png`,
    )
    expect(within(hero).getByRole('img', { name: 'Arcade on Automatos Go, Default' })).toHaveAttribute(
      'src', `${ARCADE_FILES}/assets/previews/phone/default.png`,
    )
    expect(screen.getByTestId('theme-listing-automatos.arcade')).toBeInTheDocument()
    expect(screen.getByTestId('theme-listing-automatos.hud')).toBeInTheDocument()
    expect(requestedUrls()).toEqual([CATALOG_ENDPOINT])
  })

  it('shows nothing workspace-scoped: no install state, update badge, Publish form, submit or installed list', async () => {
    renderTabs()

    await screen.findByTestId('storefront-hero')
    expectNothingWorkspaceScoped()
    expect(screen.getByRole('link', { name: OPEN_DESIGN_KIT })).toBeInTheDocument()
    expect(screen.getByText(MAKE_YOUR_OWN_VISITOR_BODY)).toBeInTheDocument()
    expect(screen.queryByText(MAKE_YOUR_OWN_BODY)).not.toBeInTheDocument()
  })

  it('with nothing listed, the empty stage says to sign in to submit, and offers no submit', async () => {
    api.get.mockImplementation((url: string) =>
      url === CATALOG_ENDPOINT ? Promise.resolve(catalog([])) : answerFrom(PAGES, url),
    )
    renderTabs()

    const stage = await screen.findByTestId('storefront-empty')
    expect(within(stage).getByText(MAKE_YOUR_OWN_VISITOR_BODY)).toBeInTheDocument()
    expectNothingWorkspaceScoped()
  })

  it('Add to my workspace goes to sign-in, carrying the theme', async () => {
    renderTabs()

    const hero = await screen.findByTestId('storefront-hero')
    const expected = `/sign-in?redirect_url=${encodeURIComponent('/marketplace?tab=themes&item=automatos.arcade&add=1')}`
    expect(within(hero).getByRole('link', { name: ADD_TO_WORKSPACE })).toHaveAttribute('href', expected)
    expect(signInToAdd('themes', 'automatos.arcade')).toBe(expected)
  })

  it('Details opens the theme’s detail from the catalogue: facts, every version, the Add link', async () => {
    renderTabs()

    const hero = await screen.findByTestId('storefront-hero')
    fireEvent.click(within(hero).getByRole('button', { name: DETAILS }))
    const dialog = await screen.findByTestId('theme-listing-detail')
    expect(within(dialog).getByText(SIGNED_BY_AUTOMATOS)).toBeInTheDocument()
    expect(within(dialog).getByText('studio')).toBeInTheDocument()
    expect(within(dialog).getByTestId('theme-version-1.1.0')).toBeInTheDocument()
    expect(within(dialog).getByTestId('theme-version-1.0.0')).toBeInTheDocument()
    expect(within(dialog).getByRole('link', { name: ADD_TO_WORKSPACE })).toHaveAttribute(
      'href', signInToAdd('themes', 'automatos.arcade'),
    )
    expectNothingWorkspaceScoped()
    expect(requestedUrls()).toEqual([CATALOG_ENDPOINT])
  })

  it('a shared link’s item is the hero', async () => {
    nav.params = new URLSearchParams('tab=themes&item=automatos.hud')
    renderTabs()

    await waitFor(() => expect(screen.getByTestId('storefront-hero-name')).toHaveTextContent('HUD'))
  })

  it('a refusal shows the server’s own sentence', async () => {
    const detail = "The Marketplace can't be read right now. Try again in a moment."
    api.get.mockRejectedValue(Object.assign(new Error('Service Unavailable'), { body: { detail } }))
    renderTabs()

    expect(await screen.findByRole('alert')).toHaveTextContent(detail)
  })
})

describe('the read-only tabs, from the public reads', () => {
  // [the tab, its ?tab= value, the public type, an item's name and key, a line its detail shows]
  it.each([
    ['Packages', 'packages', 'packages', 'Socials', 'socials', 'Agent · Social Writer'],
    ['Applications', 'tools', 'applications', 'Notion', 'notion', '12'],
    ['Agents', 'agents', 'agents', 'Research Assistant', 'research-assistant', 'notion'],
    ['Playbooks', 'recipes', 'playbooks', 'Weekly report', 'weekly-report', 'gmail'],
    ['LLMs', 'llms', 'llms', 'Claude Opus', 'anthropic:claude-opus', '200,000'],
  ])('%s: the grid, the item’s detail and its Add link', async (label, tab, type, name, key, line) => {
    renderTabs()
    chooseTab(label)

    const card = await screen.findByTestId(`visitor-item-${key}`)
    expect(card).toHaveTextContent(name)
    fireEvent.click(screen.getByRole('button', { name: itemDetails(name) }))
    const dialog = await screen.findByTestId('visitor-item-detail')
    expect(within(dialog).getByRole('heading', { name })).toBeInTheDocument()
    expect(within(dialog).getByText(line)).toBeInTheDocument()
    expect(within(dialog).getByRole('link', { name: ADD_TO_WORKSPACE })).toHaveAttribute('href', signInToAdd(tab, key))
    expect(requestedUrls()).toContain(`/api/public/marketplace/${type}?limit=100&offset=0`)
    expectNothingWorkspaceScoped()
  })

  it('Capabilities: Plugins first, then Skills, each its own read', async () => {
    renderTabs()
    chooseTab('Capabilities')

    expect(await screen.findByTestId('visitor-item-code-review')).toHaveTextContent('Code review')
    chooseTab('Skills')
    expect(await screen.findByTestId('visitor-item-pdf')).toHaveTextContent('PDF')
    fireEvent.click(screen.getByRole('button', { name: itemDetails('PDF') }))
    const dialog = await screen.findByTestId('visitor-item-detail')
    expect(within(dialog).getByRole('link', { name: ADD_TO_WORKSPACE })).toHaveAttribute('href', signInToAdd('skills', 'pdf'))
    expect(requestedUrls()).toEqual(expect.arrayContaining([
      '/api/public/marketplace/plugins?limit=100&offset=0', '/api/public/marketplace/skills?limit=100&offset=0',
    ]))
  })

  it('a ?tab=skills link opens Capabilities on Skills', async () => {
    nav.params = new URLSearchParams('tab=skills')
    renderTabs()

    expect(screen.getByRole('tab', { name: 'Capabilities' })).toHaveAttribute('aria-selected', 'true')
    expect(await screen.findByTestId('visitor-item-pdf')).toBeInTheDocument()
    expect(openedMarketplaceTab('plugins')).toEqual({ tab: 'capabilities', capability: 'plugins' })
    expect(openedMarketplaceTab('Playbooks')).toEqual({ tab: 'recipes', capability: 'plugins' })
    expect(openedMarketplaceTab(null)).toEqual({ tab: 'themes', capability: 'plugins' })
  })

  it('the search filters on the client, with no new request', async () => {
    nav.params = new URLSearchParams('tab=agents')
    const view = renderTabs('drafts')

    expect(await screen.findByTestId('visitor-item-writer')).toBeInTheDocument()
    expect(screen.queryByTestId('visitor-item-research-assistant')).not.toBeInTheDocument()
    view.rerenderWith('nothing like it')
    expect(screen.getByText(NO_ITEM_MATCHES)).toBeInTheDocument()
    view.rerenderWith('')
    expect(screen.getByTestId('visitor-item-research-assistant')).toBeInTheDocument()
    expect(requestedUrls()).toEqual(['/api/public/marketplace/agents?limit=100&offset=0'])
  })

  it('Show more reads the next page', async () => {
    nav.params = new URLSearchParams('tab=agents')
    const pages = {
      ...PAGES,
      'agents:0': publicAnswer('agents', [publicEntry(), WRITER], true),
      'agents:2': publicAnswer('agents', [publicEntry({ key: 'analyst', name: 'Analyst' })], false, 2),
    }
    api.get.mockImplementation((url: string) => answerFrom(pages, url))
    renderTabs()
    fireEvent.click(await screen.findByRole('button', { name: SHOW_MORE }))

    expect(await screen.findByTestId('visitor-item-analyst')).toBeInTheDocument()
    expect(screen.getByTestId('visitor-item-writer')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: SHOW_MORE })).not.toBeInTheDocument()
    expect(requestedUrls()).toEqual([
      '/api/public/marketplace/agents?limit=100&offset=0', '/api/public/marketplace/agents?limit=100&offset=2',
    ])
  })

  it('a shared link’s item opens its detail once: not again after the visitor changes tab', async () => {
    nav.params = new URLSearchParams('tab=agents&item=writer')
    renderTabs()

    const dialog = await screen.findByTestId('visitor-item-detail')
    expect(within(dialog).getByRole('heading', { name: 'Writer' })).toBeInTheDocument()
    fireEvent.click(within(dialog).getByRole('button', { name: 'Close' }))
    await waitFor(() => expect(screen.queryByTestId('visitor-item-detail')).not.toBeInTheDocument())
    chooseTab('Packages')
    await screen.findByTestId('visitor-item-socials')
    chooseTab('Agents')

    expect(await screen.findByTestId('visitor-item-writer')).toBeInTheDocument()
    expect(screen.queryByTestId('visitor-item-detail')).not.toBeInTheDocument()
  })

  it('pages that repeat an item show it once', async () => {
    nav.params = new URLSearchParams('tab=agents')
    const pages = {
      ...PAGES,
      'agents:0': publicAnswer('agents', [publicEntry(), WRITER], true),
      'agents:2': publicAnswer('agents', [WRITER, publicEntry({ key: 'analyst', name: 'Analyst' })], false, 2),
    }
    api.get.mockImplementation((url: string) => answerFrom(pages, url))
    renderTabs()
    fireEvent.click(await screen.findByRole('button', { name: SHOW_MORE }))

    expect(await screen.findByTestId('visitor-item-analyst')).toBeInTheDocument()
    expect(screen.getAllByTestId('visitor-item-writer')).toHaveLength(1)
  })
})
