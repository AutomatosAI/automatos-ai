/**
 * One Marketplace page for everyone (PRD-270 §5.7) — /marketplace signed out on SaaS is the member
 * page, MarketplaceHomepage, under the visitor header, drawn from the public reads.
 *
 * - the header's counts, the Featured hero and rail, and Popular in place of "Recommended for You";
 * - every tab renders its real member cards from the public reads, read-only;
 * - every add (an agent's, a Playbook's, an application's, an LLM route's, a plugin's, a skill's,
 *   a featured item's, a package's and a theme's) goes to sign-in carrying the item's link, and
 *   nothing is installed;
 * - nothing workspace-scoped shows, and every request is a public read: the apiClient double answers
 *   only the public routes, refuses anything else, and its `post` must stay uncalled.
 *
 * The edition is SaaS (the build default) and Clerk says signed out, so the real auth seam, the real
 * workspace hook and the real page run as a visitor's would.
 */
import type { ReactNode } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const nav = vi.hoisted(() => ({ params: new URLSearchParams(), push: vi.fn(), replace: vi.fn() }))
const api = vi.hoisted(() => ({
  get: vi.fn(), post: vi.fn(), delete: vi.fn(), getBaseUrl: vi.fn(), setCurrentPage: vi.fn(),
  // The member reads the page used to make: each must stay uncalled signed out.
  request: vi.fn(), getMarketplaceItems: vi.fn(), getFeaturedMarketplaceItems: vi.fn(), getSystemConfigKey: vi.fn(),
  getToolCategories: vi.fn(), getIntegrationsStatus: vi.fn(),
}))
const MEMBER_READS = [
  'request', 'getMarketplaceItems', 'getFeaturedMarketplaceItems', 'getSystemConfigKey', 'getToolCategories',
  'getIntegrationsStatus',
] as const
const clerkUseAuth = vi.hoisted(() => vi.fn())
const providerWorkspace = vi.hoisted(() => vi.fn())

vi.mock('@/lib/api-client', () => ({ apiClient: api, getAdminWorkspaceOverride: () => null }))
vi.mock('next/navigation', () => {
  const router = { push: nav.push, replace: nav.replace }
  return { useSearchParams: () => nav.params, usePathname: () => '/marketplace', useRouter: () => router }
})
vi.mock('next/link', () => ({
  default: ({ href, children, ...rest }: { href: string; children: ReactNode }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}))
vi.mock('@clerk/nextjs', () => ({
  useAuth: clerkUseAuth,
  useUser: vi.fn(),
  useSession: vi.fn(),
  useOrganization: vi.fn(),
  useClerk: vi.fn(),
  useReverification: (fetcher: unknown) => fetcher,
}))
vi.mock('@/components/workspace-provider', () => ({ useWorkspace: providerWorkspace }))
vi.mock('@/contexts/role-context', () => ({ useSystemRole: () => ({ systemRole: 'user', isAdmin: false, isLoading: false }) }))
vi.mock('@/contexts/ui-style-context', () => ({ useUiStyleOptional: () => ({ setStyle: vi.fn() }) }))
vi.mock('@/hooks/use-page-api', () => ({ usePageAPI: vi.fn() }))
vi.mock('@/components/layout/main-layout', () => ({ MainLayout: () => <div data-testid="main-layout" /> }))
vi.mock('sonner', () => ({ toast: Object.assign(vi.fn(), { success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() }) }))

import { RETURN_PATH_STORAGE_KEY, STUDIO_SITE_URL, marketplaceItemPath, signInUrl } from '@/lib/marketplace/return-after-sign-in'

import { ADD_TO_WORKSPACE, INSTALLED_BADGE, INSTALLED_HEADING, UPDATE_BADGE } from '../../themes/listing-format'
import { PUBLISH_THEME } from '../../themes/publish-theme'
import { POPULAR_TITLE, VISITOR_LEDE } from '../copy'
import { MarketplacePage } from '../marketplace-page'
import { LLM_KEY, PLUGIN_ID, PUBLIC_REQUEST, publicAnswerFor } from './one-page-fixtures'

const ADD = 'Add to Workspace'
const MEMBER_ONLY = [
  'Recommended for You', 'Import from GitHub', 'Sync with Automatos Cloud', PUBLISH_THEME, INSTALLED_HEADING,
  INSTALLED_BADGE, UPDATE_BADGE, 'Enabled for Workspace',
]

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MarketplacePage />
    </QueryClientProvider>,
  )
}

function chooseTab(name: string) {
  const tab = screen.getByRole('tab', { name })
  fireEvent.mouseDown(tab) // Radix Tabs activates on mouse down
  fireEvent.click(tab)
}

/** The innermost open tab panel (Capabilities holds its own Plugins / Skills panels). */
function openPanel(): HTMLElement {
  const panels = screen.getAllByRole('tabpanel')
  return panels[panels.length - 1]
}

/** The item dialog once its item is read (its loading dialog is another element). */
async function loadedDialog(text: string): Promise<HTMLElement> {
  await waitFor(() => expect(screen.getByRole('dialog')).toHaveTextContent(text))
  return screen.getByRole('dialog')
}

function requestedUrls(): string[] {
  return api.get.mock.calls.map(([url]) => String(url))
}

function expectSignInFor(tab: string, item: string) {
  const returnPath = marketplaceItemPath(tab, item)
  expect(nav.push).toHaveBeenLastCalledWith(signInUrl(returnPath))
  expect(JSON.parse(window.sessionStorage.getItem(RETURN_PATH_STORAGE_KEY) ?? '{}').path).toBe(returnPath)
}

beforeEach(() => {
  nav.params = new URLSearchParams()
  nav.push.mockReset()
  nav.replace.mockReset()
  api.get.mockReset()
  api.get.mockImplementation((url: string) => publicAnswerFor(url))
  for (const read of MEMBER_READS) api[read].mockReset()
  api.getBaseUrl.mockReturnValue('https://api.automatos.app')
  clerkUseAuth.mockReturnValue({ isLoaded: true, isSignedIn: false })
  providerWorkspace.mockReset()
  window.sessionStorage.clear()
})

afterEach(() => {
  // A public read only, no install, no workspace read, on any tab.
  for (const url of requestedUrls()) expect(url).toMatch(PUBLIC_REQUEST)
  expect(api.post).not.toHaveBeenCalled()
  expect(api.delete).not.toHaveBeenCalled()
  for (const read of MEMBER_READS) expect(api[read]).not.toHaveBeenCalled()
  expect(providerWorkspace).not.toHaveBeenCalled()
  for (const words of MEMBER_ONLY) expect(screen.queryByText(words)).not.toBeInTheDocument()
  cleanup()
})

describe('the page signed out (one page for everyone)', () => {
  it('is the member homepage under the visitor header: its header line, counts, Featured and Popular', async () => {
    renderPage()

    expect(screen.queryByTestId('main-layout')).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Sign in' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Start free' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Studio' })).toHaveAttribute('href', STUDIO_SITE_URL)
    expect(screen.getByText(VISITOR_LEDE)).toBeInTheDocument()

    expect(await screen.findByText('100 items')).toBeInTheDocument()
    expect(screen.getByText(/113 installs/)).toBeInTheDocument()
    expect(await screen.findByRole('heading', { name: 'Featured' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Research Assistant' })).toBeInTheDocument()
    expect(screen.getByText('v1.0.0')).toBeInTheDocument()
    expect(screen.getByText('40 installs')).toBeInTheDocument()
    expect(screen.getByText('Writer')).toBeInTheDocument()

    expect(await screen.findByRole('heading', { name: POPULAR_TITLE })).toBeInTheDocument()
    expect(screen.getByText('Weekly report')).toBeInTheDocument()
    expect(screen.getByText('61')).toBeInTheDocument()
  })

  it('opens on the Themes storefront from the catalogue; its Add goes to sign-in carrying the theme', async () => {
    renderPage()

    const hero = await screen.findByTestId('storefront-hero')
    expect(within(hero).getByTestId('storefront-hero-name')).toHaveTextContent('Arcade')
    expect(within(hero).getByRole('link', { name: ADD_TO_WORKSPACE })).toHaveAttribute(
      'href', signInUrl(marketplaceItemPath('themes', 'automatos.arcade')),
    )
  })

  it('a featured item opens the item dialog from the public reads; its Add goes to sign-in', async () => {
    renderPage()

    fireEvent.click(await screen.findByRole('button', { name: 'Install' }))
    const dialog = await loadedDialog('by Automatos')
    expect(within(dialog).getByText('NOTION')).toBeInTheDocument() // its required app, as the member dialog lists tools
    expect(within(dialog).queryByText(/installs/)).not.toBeInTheDocument()
    fireEvent.click(within(dialog).getByRole('button', { name: ADD }))
    expectSignInFor('agents', '7')
  })
})

describe('every tab renders its real cards from the public reads, and every add goes to sign-in', () => {
  // [the tab (and Capabilities' sub-tab), lines its card shows, lines it must not, the item's ?tab= and key]
  it.each([
    [['Agents'], ['Research Assistant', 'by Automatos'], ['installs'], 'agents', '7'],
    [['Playbooks'], ['Weekly report', 'Reporting'], ['Steps', 'installs'], 'recipes', '12'],
    [['Applications'], ['Notion', '12 Tools', '2 Triggers', '1 Available'], ['Connected', 'Added'], 'tools', 'notion'],
    [['LLMs'], ['Claude Opus', 'claude-opus', '200K', '1 models'], ['Max Output', 'Cost / 1M tokens', 'Added'], 'llms', LLM_KEY],
    [['Capabilities', 'Plugins'], ['Code review', 'v1.2.0', '1 Available'], ['Enabled', 'On'], 'plugins', PLUGIN_ID],
    [['Capabilities', 'Skills'], ['PDF', 'v1.0.0', '1 Available'], ['Enabled', 'Added'], 'skills', 'pdf'],
  ])('%s', async (tabs, shows, hides, tab, key) => {
    renderPage()
    for (const name of tabs) chooseTab(name)

    const panel = openPanel()
    await waitFor(() => expect(within(panel).getAllByRole('button', { name: ADD }).length).toBeGreaterThan(0))
    for (const words of shows) expect(within(panel).getAllByText(words, { exact: false }).length).toBeGreaterThan(0)
    for (const words of hides) expect(within(panel).queryByText(new RegExp(`\\b${words}\\b`))).not.toBeInTheDocument()

    fireEvent.click(within(panel).getAllByRole('button', { name: ADD })[0])
    expectSignInFor(tab, key)
  })

  it('Packages: the real card, and its detail’s Add goes to sign-in carrying the package', async () => {
    renderPage()
    chooseTab('Packages')

    const card = await screen.findByTestId('package-card')
    expect(card).toHaveTextContent('Socials')
    expect(card).toHaveTextContent('1 Agent')
    fireEvent.click(card)
    const dialog = await screen.findByTestId('visitor-item-detail')
    expect(within(dialog).getByText('Agent · Social Writer')).toBeInTheDocument()
    expect(within(dialog).getByText('instagram')).toBeInTheDocument()
    expect(within(dialog).getByRole('link', { name: ADD_TO_WORKSPACE })).toHaveAttribute(
      'href', signInUrl(marketplaceItemPath('packages', 'socials')),
    )
  })

  it('an agent’s Details is the member item dialog, read from the public reads', async () => {
    renderPage()
    chooseTab('Agents')

    const panel = openPanel()
    fireEvent.click((await within(panel).findAllByRole('button', { name: 'Details' }))[0])
    const dialog = await loadedDialog('Research Assistant')
    expect(within(dialog).getByRole('heading', { name: 'Research Assistant' })).toBeInTheDocument()
    expect(within(dialog).getByText('Finds, reads and summarises sources.')).toBeInTheDocument()
  })

  it('an LLM route’s card opens the public detail with its public facts', async () => {
    renderPage()
    chooseTab('LLMs')

    fireEvent.click(await screen.findByText('Claude Opus'))
    const dialog = await screen.findByTestId('visitor-item-detail')
    expect(within(dialog).getByText('200,000')).toBeInTheDocument()
    expect(within(dialog).getByText('reasoning · excellent')).toBeInTheDocument()
  })
})

describe('a shared link, signed out', () => {
  it('?tab=llms&item=<route> opens that route’s public detail', async () => {
    nav.params = new URLSearchParams(`tab=llms&item=${LLM_KEY}`)
    renderPage()

    const dialog = await screen.findByTestId('visitor-item-detail')
    expect(within(dialog).getByText('Claude Opus')).toBeInTheDocument()
  })

  it('?tab=agents&item=<id> opens the agent’s item dialog', async () => {
    nav.params = new URLSearchParams('tab=agents&item=8')
    renderPage()

    const dialog = await loadedDialog('Drafts posts.')
    expect(within(dialog).getByRole('heading', { name: 'Writer' })).toBeInTheDocument()
  })
})
