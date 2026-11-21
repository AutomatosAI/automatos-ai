/**
 * PRD-277 US-704 (D-12) — the Automatos Cloud pieces on an enterprise build. A setting decides only what offers
 * or describes a request to Automatos; what the install already holds is shown whatever the settings say.
 *
 * - Sync with Automatos Cloud follows Cloud sync (GET /api/hosted-services, hooks/use-hosted-services.ts): off,
 *   unreadable or loading shows no button and reads no Sync status; on shows it;
 * - a held Cloud item's badges and the Cloud items layer (the review before an install, the update notices)
 *   show as on the local edition, with every setting off, and when the settings can't be read;
 * - Settings › Automatos Cloud, shown while Cloud sync is on, says a Sync brings themes only while Cloud items
 *   are off, and names the operator's settings as the way to turn it off.
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

import { AutomatosCloudPanel } from '@/components/settings/automatos-cloud-panel'
import { isLocal, isSaaS } from '@/lib/auth-edition'
import { HOSTED_SERVICES_ENDPOINT } from '@/lib/hosted-services'
import { CLOUD_UPDATES_ENDPOINT } from '@/lib/marketplace/cloud-items'

import { CloudItemBadges } from '../cloud-item-badges'
import { CloudItemsLayer } from '../cloud-items-layer'
import { CLOUD_LEDE, CLOUD_LEDE_THEMES_ONLY, SYNC_WITH_CLOUD, TURN_OFF, TURN_OFF_ENTERPRISE } from '../copy'
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
const ALL_OFF = { themeRevocations: false, cloudSync: false, cloudItems: false }

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

function renderAll(ui: ReactElement) {
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

  it.each([
    ['every setting off', ALL_OFF],
    ['the settings unreadable', UNREADABLE],
  ] as const)('with %s shows a held Cloud item and no Sync button', async (_label, services) => {
    answering(services)
    renderAll(
      <>
        <SyncButton />
        <CloudItemBadges kind="agent" id={7} />
      </>,
    )

    expect(await screen.findByText(CLOUD_BADGE)).toBeInTheDocument()
    await waitFor(() => expect(paths()).toContain(HOSTED_SERVICES_ENDPOINT))
    await tick()
    expect(screen.queryByRole('button', { name: SYNC_WITH_CLOUD })).not.toBeInTheDocument()
    expect(paths()).not.toContain(STATUS_PATH)
  })

  it('mounts the Cloud items layer with every setting off: its update notices are read', async () => {
    answering(ALL_OFF)
    renderAll(<CloudItemsLayer />)

    await waitFor(() => expect(paths()).toContain(CLOUD_UPDATES_ENDPOINT))
  })

  it('with Cloud sync on shows the button and reads its status', async () => {
    answering({ themeRevocations: true, cloudSync: true, cloudItems: false })
    renderAll(<SyncButton />)

    expect(await screen.findByRole('button', { name: SYNC_WITH_CLOUD })).toBeEnabled()
    expect(paths()).toEqual(expect.arrayContaining([HOSTED_SERVICES_ENDPOINT, STATUS_PATH]))
  })

  it('Settings › Automatos Cloud says a Sync brings themes only while Cloud items are off', async () => {
    answering({ themeRevocations: true, cloudSync: true, cloudItems: false })
    renderAll(<AutomatosCloudPanel />)

    expect(await screen.findByText(CLOUD_LEDE_THEMES_ONLY)).toBeInTheDocument()
    expect(screen.queryByText(CLOUD_LEDE)).not.toBeInTheDocument()
    expect(await screen.findByText(TURN_OFF_ENTERPRISE)).toBeInTheDocument()
    expect(screen.queryByText(TURN_OFF)).not.toBeInTheDocument()
  })

  it('Settings › Automatos Cloud says a Sync brings items too while Cloud items are on', async () => {
    answering({ themeRevocations: true, cloudSync: true, cloudItems: true })
    renderAll(<AutomatosCloudPanel />)

    expect(await screen.findByText(CLOUD_LEDE)).toBeInTheDocument()
    expect(screen.queryByText(CLOUD_LEDE_THEMES_ONLY)).not.toBeInTheDocument()
  })
})
