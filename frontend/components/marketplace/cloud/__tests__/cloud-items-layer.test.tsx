/**
 * PRD-270 W2 US-270-14d (§5.6, K19) — the Cloud items layer with the real apiClient and the real install
 * gate: only fetch is a double, answering the Cloud items, the disclosure and the install.
 *
 * - an install of a Cloud item opens its disclosure and sends nothing until Add, then exactly one install
 *   request; Close sends none and the request fails with the gate's sentence;
 * - a local item installs at once, with no dialog;
 * - one review at a time, and a review still open when the layer unmounts is refused;
 * - on the hosted edition the layer renders nothing and the install goes straight out;
 * - MarketplaceHomepage mounts the layer once.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const edition = vi.hoisted(() => ({ local: true }))

vi.mock('@/lib/auth-edition', () => ({
  get isLocal() {
    return edition.local
  },
  get isSaaS() {
    return !edition.local
  },
}))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'ws-1', name: 'Acme', role: 'member' } }),
}))

import { apiClient } from '@/lib/api-client'
import type { AgentDisclosure } from '@/lib/marketplace/cloud-items'
import type { CloudItems } from '@/lib/marketplace/cloud-item-rows'
import { CLOUD_INSTALL_BUSY, CLOUD_INSTALL_CLOSED } from '@/lib/marketplace/cloud-install-gate'

import { CloudItemsLayer } from '../cloud-items-layer'

const CLOUD_INSTALL = '/api/marketplace/items/7/install'
const LOCAL_INSTALL = '/api/marketplace/items/8/install'

function row(name: string) {
  return { externalId: `ext-${name}`, version: '1.2.0', revision: 'a'.repeat(64), name }
}

const ITEMS: CloudItems = {
  agents: [{ id: 7, ...row('Ledger Helper') }],
  playbooks: [],
  packages: [],
  plugins: [],
}

const AGENT: AgentDisclosure = {
  kind: 'agent', name: 'Ledger Helper', version: '1.2.0', syncedAt: '2026-10-30T10:00:00+00:00', trust: 'reviewed',
  publisher: 'Cloud Publisher', unresolved: [],
  prompt: { source: 'custom', text: 'You keep the books.', personaName: null },
  model: null, apps: [{ name: 'GMAIL', needsConnection: true }], skills: [], plugins: [],
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

/** The backend: the Cloud items, the agent's disclosure, no update notices (US-270-14e) and any install. */
function backend() {
  return vi.fn(async (url: string, init?: RequestInit) => {
    const path = String(url)
    if (path.endsWith('/api/marketplace/cloud-sync/updates')) return json({ updates: [] })
    if (path.endsWith('/api/marketplace/cloud-sync/items')) return json(ITEMS)
    if (path.endsWith('/api/marketplace/cloud-sync/items/agent/7/disclosure')) return json(AGENT)
    if (init?.method === 'POST' && path.endsWith('/install')) return json({ success: true, message: 'Added' })
    return json({ detail: 'Not found' }, 404)
  })
}

let fetchMock: ReturnType<typeof backend>

/** The install requests fetch received. */
function installsSent(): string[] {
  return fetchMock.mock.calls
    .filter(([, init]) => init?.method === 'POST')
    .map(([url]) => String(url))
}

function renderLayer() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <CloudItemsLayer />
    </QueryClientProvider>,
  )
}

/** Starts an install as a card would, its outcome kept for the test to read. */
function install(endpoint: string): Promise<unknown> {
  const answer = apiClient.post(endpoint)
  answer.catch(() => {})
  return answer
}

beforeEach(() => {
  edition.local = true
  fetchMock = backend()
  vi.stubGlobal('fetch', fetchMock)
  vi.spyOn(console, 'log').mockImplementation(() => {})
  vi.spyOn(console, 'error').mockImplementation(() => {})
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('an install of a Cloud item', () => {
  it('opens its disclosure, and Add sends exactly one install request', async () => {
    renderLayer()

    let answer: Promise<unknown> = Promise.resolve()
    await act(async () => {
      answer = install(CLOUD_INSTALL)
    })
    const add = await screen.findByTestId('cloud-disclosure-add')

    expect(screen.getByText('Ledger Helper')).toBeInTheDocument()
    expect(screen.getByText('You keep the books.')).toBeInTheDocument()
    expect(installsSent()).toEqual([])

    fireEvent.click(add)

    await expect(answer).resolves.toEqual({ success: true, message: 'Added' })
    expect(installsSent()).toHaveLength(1)
    expect(installsSent()[0].endsWith(CLOUD_INSTALL)).toBe(true)
    await waitFor(() => expect(screen.queryByTestId('cloud-disclosure')).not.toBeInTheDocument())
  })

  it('sends nothing on Close, and the install fails with the gate sentence', async () => {
    renderLayer()

    let answer: Promise<unknown> = Promise.resolve()
    await act(async () => {
      answer = install(CLOUD_INSTALL)
    })
    fireEvent.click(await screen.findByTestId('cloud-disclosure-close'))

    await expect(answer).rejects.toThrow(CLOUD_INSTALL_CLOSED)
    expect(installsSent()).toEqual([])
    await waitFor(() => expect(screen.queryByTestId('cloud-disclosure')).not.toBeInTheDocument())
  })

  it('refuses a second install while a review is open, and the first still waits for its answer', async () => {
    renderLayer()

    let first: Promise<unknown> = Promise.resolve()
    let second: Promise<unknown> = Promise.resolve()
    await act(async () => {
      first = install(CLOUD_INSTALL)
    })
    await screen.findByTestId('cloud-disclosure-add')
    await act(async () => {
      second = install(CLOUD_INSTALL)
    })

    await expect(second).rejects.toThrow(CLOUD_INSTALL_BUSY)
    expect(screen.getAllByTestId('cloud-disclosure')).toHaveLength(1)
    fireEvent.click(screen.getByTestId('cloud-disclosure-add'))
    await expect(first).resolves.toEqual({ success: true, message: 'Added' })
    expect(installsSent()).toHaveLength(1)
  })

  it('refuses a review still open when the layer unmounts', async () => {
    const { unmount } = renderLayer()

    let answer: Promise<unknown> = Promise.resolve()
    await act(async () => {
      answer = install(CLOUD_INSTALL)
    })
    await screen.findByTestId('cloud-disclosure-add')
    unmount()

    await expect(answer).rejects.toThrow(CLOUD_INSTALL_CLOSED)
    expect(installsSent()).toEqual([])
  })
})

describe('everything else', () => {
  it('installs a local item at once, with no dialog', async () => {
    renderLayer()

    await act(async () => {
      await expect(install(LOCAL_INSTALL)).resolves.toEqual({ success: true, message: 'Added' })
    })

    expect(screen.queryByTestId('cloud-disclosure')).not.toBeInTheDocument()
    expect(installsSent()).toHaveLength(1)
  })

  it('renders nothing on the hosted edition, and the install goes straight out', async () => {
    edition.local = false
    const { container } = renderLayer()

    await expect(install(CLOUD_INSTALL)).resolves.toEqual({ success: true, message: 'Added' })

    expect(container).toBeEmptyDOMElement()
    expect(screen.queryByTestId('cloud-disclosure')).not.toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(installsSent()).toHaveLength(1)
  })
})

describe('its mount', () => {
  it('is mounted once in MarketplaceHomepage', () => {
    const homepage = readFileSync(path.resolve(__dirname, '../../marketplace-homepage.tsx'), 'utf8')

    expect(homepage.match(/<CloudItemsLayer \/>/g)).toHaveLength(1)
    expect(homepage).toContain("import { CloudItemsLayer } from './cloud/cloud-items-layer'")
  })
})
