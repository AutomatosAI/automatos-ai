/**
 * PRD-270 US-270-05b — Settings › Automatos Cloud (§5.5, K18). The panel shows the URL in use, the
 * last success, check and error, the last Sync's counts and the themes held, Sync now (the
 * Marketplace header's mutation and toast) and how to turn it off; it says plainly when the Sync is
 * off. SettingsPanel gains the tab on the local edition only, for an owner or admin while the status
 * says the Sync is on; the hosted edition and a member never see it and ask nothing.
 *
 * Every other Settings tab body is stubbed (settings-panel-edition.test.tsx's doubles); the tab strip
 * is a stand-in that lists the tabs and the open one. `@clerk/nextjs` is the double
 * sync-button.test.tsx uses (the error words come from lib/theme-host/api.ts, which imports it).
 */
import type { ReactElement } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }))
const edition = vi.hoisted(() => ({ local: true }))
const workspace = vi.hoisted(() => ({ role: 'owner' as string }))
const nav = vi.hoisted(() => ({ search: '' }))
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
  useWorkspace: () => ({ workspace: { id: 'ws-1', name: 'Acme', role: workspace.role } }),
}))
vi.mock('@/contexts/ui-style-context', () => ({ useUiStyleOptional: () => null }))
vi.mock('sonner', () => ({ toast }))
vi.mock('@clerk/nextjs', () => ({
  useReverification: (fetcher: (...args: unknown[]) => Promise<unknown>) => fetcher,
}))

vi.mock('next/link', () => ({
  default: ({ href, children }: any) => <a href={href}>{children}</a>,
}))
vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
  useSearchParams: () => new URLSearchParams(nav.search),
  usePathname: () => '/settings',
}))
vi.mock('@/contexts/role-context', () => ({
  useSystemRole: () => ({ isAdmin: true, systemRole: 'super_admin', isLoading: false }),
}))
vi.mock('@/components/shared', () => ({
  PageHeader: () => null,
  FilterTabs: ({ tabs, value, children }: any) => (
    <div>
      <output data-testid="active-tab">{value}</output>
      <nav aria-label="settings tabs">
        {tabs.map((tab: any) => (
          <button key={tab.value} type="button">
            {tab.label}
          </button>
        ))}
      </nav>
      {children}
    </div>
  ),
  TabsContent: ({ children }: any) => <section>{children}</section>,
}))
vi.mock('../CredentialsTab', () => ({ CredentialsTab: () => null }))
vi.mock('../SystemSettingsTab', () => ({ default: () => null }))
vi.mock('../SystemLLMSettingsTab', () => ({ default: () => null }))
vi.mock('../WebhooksSettingsTab', () => ({ default: () => null }))
vi.mock('../ApiKeysSettingsTab', () => ({ ApiKeysSettingsTab: () => null }))
vi.mock('../ChannelsSettingsTab', () => ({ ChannelsSettingsTab: () => null }))
vi.mock('../ApiKeyManager', () => ({ ApiKeyManager: () => null }))
vi.mock('../WidgetSdkTab', () => ({ WidgetSdkTab: () => null }))
vi.mock('../NotificationsSettingsTab', () => ({ NotificationsSettingsTab: () => null }))
vi.mock('../SessionModeTab', () => ({ SessionModeTab: () => null }))
vi.mock('../WorkspaceSocialsCard', () => ({ WorkspaceSocialsCard: () => null }))
vi.mock('../WorkspaceVoiceLiveCard', () => ({ WorkspaceVoiceLiveCard: () => null }))
vi.mock('../themes/appearance-tab', () => ({ AppearanceTab: () => null }))
vi.mock('../phones-devices/phones-devices-tab', () => ({ PhonesDevicesTab: () => null }))
vi.mock('../billing/billing-tab', () => ({ BillingTab: () => null }))

import { cloudSyncKeys } from '@/hooks/use-cloud-sync'
import type { CloudSyncResult, CloudSyncStatus } from '@/lib/marketplace/cloud-sync'

import { AutomatosCloudPanel } from '../automatos-cloud-panel'
import { LOCAL_EDITION_SETTINGS_TABS, SettingsPanel } from '../SettingsPanel'

const STATUS_PATH = '/api/marketplace/cloud-sync/status'
const SYNC_PATH = '/api/marketplace/cloud-sync'
const CLOUD_URL = 'https://api.automatos.app'
const TURN_OFF = 'To turn it off, set AUTOMATOS_CLOUD_URL= (empty) in envs/api.local and restart.'

const minutesAgo = (minutes: number) => new Date(Date.now() - minutes * 60_000).toISOString()

function status(overrides: Partial<CloudSyncStatus> = {}): CloudSyncStatus {
  return {
    enabled: true,
    cloudUrl: CLOUD_URL,
    lastAttemptAt: null,
    lastSuccessAt: null,
    lastError: null,
    generatedAt: null,
    counts: { new: 0, updated: 0, unchanged: 0, refused: 0 },
    themes: 0,
    items: { agent: 0, playbook: 0, package: 0, plugin: 0 },
    stale: true,
    ...overrides,
  }
}

function renderWithClient(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
  return client
}

const tabLabels = () => screen.getAllByRole('button').map((button) => button.textContent ?? '')
const settle = () => new Promise((resolve) => setTimeout(resolve, 0))

beforeEach(() => {
  vi.clearAllMocks()
  edition.local = true
  workspace.role = 'owner'
  nav.search = ''
  api.get.mockResolvedValue(status())
})

describe('the Automatos Cloud panel', () => {
  it('shows the facts from the status: the URL in use, the last Sync, its counts and the themes held', async () => {
    api.get.mockResolvedValue(
      status({
        lastSuccessAt: minutesAgo(2),
        lastAttemptAt: minutesAgo(1),
        lastError: 'Automatos Cloud did not answer in time.',
        counts: { new: 2, updated: 1, unchanged: 5, refused: 0 },
        themes: 3,
        stale: false,
      }),
    )
    renderWithClient(<AutomatosCloudPanel />)

    expect(await screen.findByTestId('automatos-cloud-url')).toHaveTextContent(CLOUD_URL)
    expect(screen.getByTestId('automatos-cloud-success')).toHaveTextContent('2m ago')
    expect(screen.getByTestId('automatos-cloud-attempt')).toHaveTextContent('1m ago')
    expect(screen.getByTestId('automatos-cloud-error')).toHaveTextContent('Automatos Cloud did not answer in time.')
    expect(screen.getByTestId('automatos-cloud-counts')).toHaveTextContent('2 new, 1 updated, 5 unchanged, 0 refused')
    expect(screen.getByTestId('automatos-cloud-themes')).toHaveTextContent(/^3$/)
    expect(screen.getByTestId('automatos-cloud-turn-off')).toHaveTextContent(TURN_OFF)
    expect(screen.getByRole('button', { name: 'Sync now' })).toBeEnabled()
    expect(api.get).toHaveBeenCalledWith(STATUS_PATH)
  })

  it('says never and none before any Sync, and renders the error as text', async () => {
    const error = 'The catalogue answered 404. <b>Try again</b> later.'
    api.get.mockResolvedValue(status({ lastError: error }))
    renderWithClient(<AutomatosCloudPanel />)

    const line = await screen.findByTestId('automatos-cloud-error')
    expect(line).toHaveTextContent(error)
    expect(line.querySelector('b')).toBeNull()
    expect(screen.getByTestId('automatos-cloud-success')).toHaveTextContent('never')
    expect(screen.getByTestId('automatos-cloud-counts')).toHaveTextContent(/^Not synced yet$/)
    expect(screen.getByTestId('automatos-cloud-themes')).toHaveTextContent(/^0$/)
  })

  it('keeps the facts and Sync now when a re-read of the status fails', async () => {
    const client = renderWithClient(<AutomatosCloudPanel />)
    expect(await screen.findByRole('button', { name: 'Sync now' })).toBeEnabled()

    api.get.mockRejectedValueOnce(Object.assign(new Error('HTTP 503'), { body: { detail: 'Try again shortly.' } }))
    await act(() => client.refetchQueries({ queryKey: cloudSyncKeys.status('ws-1') }))

    await waitFor(() => expect(client.getQueryState(cloudSyncKeys.status('ws-1'))?.status).toBe('error'))
    expect(screen.getByRole('button', { name: 'Sync now' })).toBeEnabled()
    expect(screen.getByTestId('automatos-cloud-url')).toHaveTextContent(CLOUD_URL)
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('names the agents, Playbooks, packages and plugins held here, a type with none left out, or none', async () => {
    api.get.mockResolvedValue(status({ lastSuccessAt: minutesAgo(2), themes: 3, items: { agent: 2, playbook: 0, package: 1, plugin: 1 } }))
    const { unmount } = render(<QueryClientProvider client={new QueryClient()}><AutomatosCloudPanel /></QueryClientProvider>)
    expect(await screen.findByTestId('automatos-cloud-items')).toHaveTextContent(/^2 agents, 1 package, 1 plugin$/)
    unmount()

    api.get.mockResolvedValue(status({ lastSuccessAt: minutesAgo(2), themes: 3 }))
    renderWithClient(<AutomatosCloudPanel />)
    expect(await screen.findByTestId('automatos-cloud-items')).toHaveTextContent(/^none$/)
  })

  it('Sync now posts once, is disabled with a spinner while it runs, then shows the toast and the new facts', async () => {
    let answer: (value: CloudSyncResult) => void = () => {}
    api.post.mockReturnValue(new Promise<CloudSyncResult>((resolve) => { answer = resolve }))
    renderWithClient(<AutomatosCloudPanel />)

    fireEvent.click(await screen.findByRole('button', { name: 'Sync now' }))
    await waitFor(() => expect(screen.getByRole('button', { name: 'Sync now' })).toBeDisabled())
    expect(screen.getByTestId('automatos-cloud-spinner')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Sync now' }))

    answer({
      counts: { new: 2, updated: 1, unchanged: 0, refused: 0 },
      updatesApplied: 0,
      updatesFailed: 0,
      status: status({ lastSuccessAt: new Date().toISOString(), themes: 3, counts: { new: 2, updated: 1, unchanged: 0, refused: 0 } }),
    })

    await waitFor(() => expect(toast.success).toHaveBeenCalledWith('2 new, 1 updated from Automatos Cloud'))
    expect(api.post).toHaveBeenCalledTimes(1)
    expect(api.post).toHaveBeenCalledWith(SYNC_PATH)
    await waitFor(() => expect(screen.getByTestId('automatos-cloud-themes')).toHaveTextContent(/^3$/))
    expect(screen.getByTestId('automatos-cloud-success')).toHaveTextContent('just now')
    await waitFor(() => expect(screen.getByRole('button', { name: 'Sync now' })).toBeEnabled())
    expect(screen.queryByTestId('automatos-cloud-spinner')).not.toBeInTheDocument()
  })

  it('says plainly when the Sync is off, and offers no Sync now', async () => {
    api.get.mockResolvedValue(status({ enabled: false, cloudUrl: null }))
    renderWithClient(<AutomatosCloudPanel />)

    expect(await screen.findByTestId('automatos-cloud-off')).toHaveTextContent(
      'Sync with Automatos Cloud is off on this install: AUTOMATOS_CLOUD_URL is empty.',
    )
    expect(screen.queryByRole('button', { name: 'Sync now' })).not.toBeInTheDocument()
    expect(screen.queryByTestId('automatos-cloud-turn-off')).not.toBeInTheDocument()
  })

  it('renders nothing and asks nothing for a member or on the hosted edition', async () => {
    workspace.role = 'member'
    const { unmount } = render(<QueryClientProvider client={new QueryClient()}><AutomatosCloudPanel /></QueryClientProvider>)
    await settle()
    expect(screen.queryByTestId('automatos-cloud')).not.toBeInTheDocument()
    unmount()

    workspace.role = 'owner'
    edition.local = false
    renderWithClient(<AutomatosCloudPanel />)
    await settle()
    expect(screen.queryByTestId('automatos-cloud')).not.toBeInTheDocument()
    expect(api.get).not.toHaveBeenCalled()
  })
})

describe('the Automatos Cloud tab in Settings', () => {
  it.each(['owner', 'admin'])('a local %s gets the tab, and ?tab=automatos-cloud opens it', async (role) => {
    workspace.role = role
    nav.search = 'tab=automatos-cloud'
    renderWithClient(<SettingsPanel />)

    expect(await screen.findByRole('button', { name: 'Automatos Cloud' })).toBeInTheDocument()
    expect(tabLabels().indexOf('Automatos Cloud')).toBe(tabLabels().indexOf('Appearance') + 1)
    expect(screen.getByTestId('active-tab')).toHaveTextContent('automatos-cloud')
    expect(await screen.findByTestId('automatos-cloud-url')).toHaveTextContent(CLOUD_URL)
    expect(LOCAL_EDITION_SETTINGS_TABS.has('automatos-cloud')).toBe(true)
  })

  it.each(['member', 'viewer'])('a local %s never gets the tab, and nothing is asked', async (role) => {
    workspace.role = role
    renderWithClient(<SettingsPanel />)

    await settle()
    expect(tabLabels()).not.toContain('Automatos Cloud')
    expect(screen.queryByTestId('automatos-cloud')).not.toBeInTheDocument()
    expect(api.get).not.toHaveBeenCalled()
  })

  it('the hosted edition never has the tab, and asks nothing', async () => {
    edition.local = false
    renderWithClient(<SettingsPanel />)

    await settle()
    expect(tabLabels()).toContain('Billing')
    expect(tabLabels()).not.toContain('Automatos Cloud')
    expect(api.get).not.toHaveBeenCalled()
  })

  it('AUTOMATOS_CLOUD_URL= (empty) hides the tab with the Marketplace button', async () => {
    api.get.mockResolvedValue(status({ enabled: false, cloudUrl: null }))
    renderWithClient(<SettingsPanel />)

    await waitFor(() => expect(api.get).toHaveBeenCalledWith(STATUS_PATH))
    await settle()
    expect(tabLabels()).not.toContain('Automatos Cloud')
    expect(screen.queryByTestId('automatos-cloud')).not.toBeInTheDocument()
  })
})
