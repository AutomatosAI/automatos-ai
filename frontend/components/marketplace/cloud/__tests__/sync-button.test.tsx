/**
 * PRD-270 US-270-05a — Sync with Automatos Cloud in the Marketplace header (§5.5, K18). A local owner
 * or admin sees it while the Sync is on; a member, the hosted edition and a Sync that is off never do,
 * and the first two ask nothing. While a Sync runs the button is disabled with a spinner; after it the
 * toast says what it brought and the workspace's theme queries are read again. The status line shows
 * the last success and the themes held, or the last error in plain words.
 *
 * `@clerk/nextjs` is the double revocations-panel.test.tsx uses (the error words come from
 * lib/theme-host/api.ts, which imports it).
 */
import type { ReactElement } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }))
const edition = vi.hoisted(() => ({ local: true }))
const workspace = vi.hoisted(() => ({ role: 'owner' as string }))
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

import { cloudSyncKeys } from '@/hooks/use-cloud-sync'
import { themeQueryKeys } from '@/hooks/use-installed-themes'
import type { CloudSyncResult, CloudSyncStatus } from '@/lib/marketplace/cloud-sync'

import { NOTHING_NEW, SYNC_WITH_CLOUD } from '../copy'
import { SyncButton } from '../sync-button'

const STATUS_PATH = '/api/marketplace/cloud-sync/status'
const SYNC_PATH = '/api/marketplace/cloud-sync'

function status(overrides: Partial<CloudSyncStatus> = {}): CloudSyncStatus {
  return {
    enabled: true,
    cloudUrl: 'https://api.automatos.app',
    lastAttemptAt: null,
    lastSuccessAt: null,
    lastError: null,
    generatedAt: null,
    counts: { new: 0, updated: 0, unchanged: 0, refused: 0 },
    themes: 0,
    stale: true,
    ...overrides,
  }
}

function result(counts: { new: number; updated: number }, overrides: Partial<CloudSyncResult> = {}): CloudSyncResult {
  return {
    counts: { ...counts, unchanged: 0, refused: 0 },
    updatesApplied: 0,
    updatesFailed: 0,
    status: status({ lastSuccessAt: new Date().toISOString(), themes: 3 }),
    ...overrides,
  }
}

function renderButton(ui: ReactElement = <SyncButton />) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  const invalidate = vi.spyOn(client, 'invalidateQueries')
  render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
  return { client, invalidate }
}

const syncButton = () => screen.findByRole('button', { name: SYNC_WITH_CLOUD })

beforeEach(() => {
  vi.clearAllMocks()
  edition.local = true
  workspace.role = 'owner'
  api.get.mockResolvedValue(status())
})

describe('who sees Sync with Automatos Cloud', () => {
  it.each(['owner', 'admin'])('a local %s sees it, with the status line', async (role) => {
    workspace.role = role
    renderButton()

    expect(await syncButton()).toBeEnabled()
    expect(screen.getByTestId('cloud-sync-status')).toHaveTextContent('Not synced yet')
    expect(api.get).toHaveBeenCalledWith(STATUS_PATH)
  })

  it.each(['member', 'viewer'])('a %s never sees it, and nothing is asked', async (role) => {
    workspace.role = role
    renderButton()

    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(screen.queryByTestId('cloud-sync')).not.toBeInTheDocument()
    expect(api.get).not.toHaveBeenCalled()
  })

  it('the hosted edition never shows it, and asks nothing', async () => {
    edition.local = false
    renderButton()

    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(screen.queryByTestId('cloud-sync')).not.toBeInTheDocument()
    expect(api.get).not.toHaveBeenCalled()
  })

  it('is hidden while the status says the Sync is off', async () => {
    api.get.mockResolvedValue(status({ enabled: false, cloudUrl: null }))
    const { client } = renderButton()

    await waitFor(() => expect(client.getQueryData(cloudSyncKeys.status('ws-1'))).toMatchObject({ enabled: false }))
    expect(api.get).toHaveBeenCalledWith(STATUS_PATH)
    expect(screen.queryByTestId('cloud-sync')).not.toBeInTheDocument()
  })
})

describe('the status line', () => {
  it('says when the last Sync succeeded and how many themes it holds', async () => {
    const twoMinutesAgo = new Date(Date.now() - 2 * 60_000).toISOString()
    api.get.mockResolvedValue(status({ lastSuccessAt: twoMinutesAgo, themes: 3, stale: false }))
    renderButton()

    expect(await screen.findByTestId('cloud-sync-status')).toHaveTextContent('Last synced 2m ago · 3 themes')
  })

  it('shows the last error in plain words, as text', async () => {
    const error = 'Automatos Cloud did not answer in time. <b>Try again</b> later.'
    api.get.mockResolvedValue(status({ lastError: error }))
    renderButton()

    const line = await screen.findByTestId('cloud-sync-error')
    expect(line).toHaveTextContent(error)
    expect(line.querySelector('b')).toBeNull()
    expect(screen.queryByTestId('cloud-sync-status')).not.toBeInTheDocument()
  })
})

describe('a Sync', () => {
  it('is disabled with a spinner while it runs, then says what it brought and reads the themes again', async () => {
    let answer: (value: CloudSyncResult) => void = () => {}
    api.post.mockReturnValue(new Promise<CloudSyncResult>((resolve) => { answer = resolve }))
    const { invalidate } = renderButton()

    fireEvent.click(await syncButton())

    await waitFor(() => expect(screen.getByRole('button', { name: SYNC_WITH_CLOUD })).toBeDisabled())
    expect(screen.getByTestId('cloud-sync-spinner')).toBeInTheDocument()
    expect(api.post).toHaveBeenCalledWith(SYNC_PATH)

    answer(result({ new: 2, updated: 1 }))

    await waitFor(() => expect(toast.success).toHaveBeenCalledWith('2 new, 1 updated from Automatos Cloud'))
    expect(invalidate).toHaveBeenCalledWith({ queryKey: themeQueryKeys.all('ws-1') })
    expect(await screen.findByTestId('cloud-sync-status')).toHaveTextContent('Last synced just now · 3 themes')
    expect(screen.getByRole('button', { name: SYNC_WITH_CLOUD })).toBeEnabled()
    expect(screen.queryByTestId('cloud-sync-spinner')).not.toBeInTheDocument()
  })

  it('says nothing new when both counts are 0', async () => {
    api.post.mockResolvedValue(result({ new: 0, updated: 0 }))
    renderButton()

    fireEvent.click(await syncButton())

    await waitFor(() => expect(toast.success).toHaveBeenCalledWith(NOTHING_NEW))
    expect(toast.success).toHaveBeenCalledWith('Nothing new from Automatos Cloud')
    expect(toast.error).not.toHaveBeenCalled()
  })

  it('names the themes whose update failed', async () => {
    api.post.mockResolvedValue(result({ new: 0, updated: 1 }, { updatesFailed: 1 }))
    renderButton()

    fireEvent.click(await syncButton())

    await waitFor(() => expect(toast.success).toHaveBeenCalledWith('0 new, 1 updated from Automatos Cloud'))
    expect(toast.error).toHaveBeenCalledWith(expect.stringContaining('1 theme could not be updated from Automatos Cloud'))
  })

  it('a failed Sync shows the server’s sentence and reads the status again', async () => {
    const detail = 'The Automatos Cloud catalogue could not be read.'
    api.post.mockRejectedValue(Object.assign(new Error('HTTP 502'), { body: { detail } }))
    renderButton()

    fireEvent.click(await syncButton())

    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(detail))
    expect(toast.success).not.toHaveBeenCalled()
    await waitFor(() => expect(api.get).toHaveBeenCalledTimes(2))
  })
})
