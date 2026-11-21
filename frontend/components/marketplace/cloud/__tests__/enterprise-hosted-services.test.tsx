/**
 * PRD-277 US-704 (D-12) — the Marketplace's Automatos Cloud pieces on an enterprise build, where each hosted
 * service is the operator's own setting (GET /api/hosted-services, hooks/use-hosted-services.ts).
 *
 * - with Cloud sync and Cloud items off, Sync with Automatos Cloud, the Cloud badges and the Cloud items
 *   layer render nothing, and the only request is the one that reads the settings;
 * - while that answer loads, or when it can't be read, nothing renders and nothing else is asked;
 * - with both on, the button shows (its status read) and a Cloud item carries its badges;
 * - Cloud items on with Cloud sync off shows the badges and no button: each setting decides its own piece.
 *
 * The build is enterprise through lib/edition-name.ts, the one place that reads it; lib/auth-edition.ts
 * follows it, so `isLocal` and `isSaaS` are both false.
 */
import type { ReactElement } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }))

vi.mock('@/lib/edition-name', () => ({ isEnterprise: true, resolveEdition: () => 'enterprise' }))
vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'ws-1', name: 'Acme', role: 'owner' } }),
}))
vi.mock('@/contexts/ui-style-context', () => ({ useUiStyleOptional: () => null }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@clerk/nextjs', () => ({
  useReverification: (fetcher: (...args: unknown[]) => Promise<unknown>) => fetcher,
}))

import { isLocal, isSaaS } from '@/lib/auth-edition'
import { HOSTED_SERVICES_ENDPOINT } from '@/lib/hosted-services'

import { CloudItemBadges } from '../cloud-item-badges'
import { CloudItemsLayer } from '../cloud-items-layer'
import { SYNC_WITH_CLOUD } from '../copy'
import { CLOUD_BADGE } from '../item-copy'
import { SyncButton } from '../sync-button'

const STATUS_PATH = '/api/marketplace/cloud-sync/status'
const ITEMS_PATH = '/api/marketplace/cloud-sync/items'

const STATUS = {
  enabled: true,
  cloudUrl: 'https://cloud.example.test',
  lastAttemptAt: null,
  lastSuccessAt: null,
  lastError: null,
  generatedAt: null,
  counts: { new: 0, updated: 0, unchanged: 0, refused: 0 },
  themes: 0,
  items: { agent: 1, playbook: 0, package: 0, plugin: 0 },
  stale: true,
}
const ITEMS = {
  agents: [{ id: 7, externalId: 'ext-ledger', version: '1.0.0', revision: 'a'.repeat(64), name: 'Ledger Helper' }],
  playbooks: [],
  packages: [],
  plugins: [],
}

const UNREADABLE = 'unreadable'

function answering(services: Record<string, boolean> | typeof UNREADABLE) {
  api.get.mockImplementation(async (path: string) => {
    if (path === HOSTED_SERVICES_ENDPOINT) {
      if (services === UNREADABLE) throw new Error('offline')
      return services
    }
    if (path === STATUS_PATH) return STATUS
    if (path === ITEMS_PATH) return ITEMS
    throw new Error(`unexpected request ${path}`)
  })
}

function renderAll(ui: ReactElement = (
  <>
    <SyncButton />
    <CloudItemBadges kind="agent" id={7} />
    <CloudItemsLayer />
  </>
)) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}

const paths = () => api.get.mock.calls.map(([path]) => path)
const tick = () => new Promise((resolve) => setTimeout(resolve, 0))

beforeEach(() => {
  vi.clearAllMocks()
})

describe('an enterprise build', () => {
  it('is neither local nor hosted', () => {
    expect(isLocal).toBe(false)
    expect(isSaaS).toBe(false)
  })

  it('with Cloud sync and Cloud items off shows nothing and asks only for the settings', async () => {
    answering({ themeRevocations: true, cloudSync: false, cloudItems: false })
    renderAll()

    await waitFor(() => expect(paths()).toEqual([HOSTED_SERVICES_ENDPOINT]))
    await tick()
    expect(screen.queryByRole('button', { name: SYNC_WITH_CLOUD })).not.toBeInTheDocument()
    expect(screen.queryByText(CLOUD_BADGE)).not.toBeInTheDocument()
    expect(paths()).toEqual([HOSTED_SERVICES_ENDPOINT])
  })

  it('shows nothing and asks nothing else while the settings load or when they cannot be read', async () => {
    answering(UNREADABLE)
    renderAll()

    await waitFor(() => expect(paths()).toEqual([HOSTED_SERVICES_ENDPOINT]))
    await tick()
    expect(screen.queryByRole('button', { name: SYNC_WITH_CLOUD })).not.toBeInTheDocument()
    expect(screen.queryByText(CLOUD_BADGE)).not.toBeInTheDocument()
    expect(paths()).toEqual([HOSTED_SERVICES_ENDPOINT])
  })

  it('with both on shows the button and the badges', async () => {
    answering({ themeRevocations: true, cloudSync: true, cloudItems: true })
    renderAll(
      <>
        <SyncButton />
        <CloudItemBadges kind="agent" id={7} />
      </>,
    )

    expect(await screen.findByRole('button', { name: SYNC_WITH_CLOUD })).toBeEnabled()
    expect(await screen.findByText(CLOUD_BADGE)).toBeInTheDocument()
    expect(paths()).toEqual(expect.arrayContaining([HOSTED_SERVICES_ENDPOINT, STATUS_PATH, ITEMS_PATH]))
  })

  it('Cloud items on with Cloud sync off shows the badges and no button', async () => {
    answering({ themeRevocations: true, cloudSync: false, cloudItems: true })
    renderAll(
      <>
        <SyncButton />
        <CloudItemBadges kind="agent" id={7} />
      </>,
    )

    expect(await screen.findByText(CLOUD_BADGE)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: SYNC_WITH_CLOUD })).not.toBeInTheDocument()
    expect(paths()).not.toContain(STATUS_PATH)
  })
})
