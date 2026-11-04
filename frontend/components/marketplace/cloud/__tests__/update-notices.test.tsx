/**
 * PRD-270 W2 US-270-14e (§5.6, K17, K19) — the update notices, rendered by the Cloud items layer with the real
 * apiClient and the real install gate: only fetch is a double, answering the updates, the Cloud items, the
 * disclosures and the installs.
 *
 * - a notice per update, with its name: a copy's newer version, a plugin updated in place;
 * - Review opens the disclosure for the Cloud row; its Add sends one install request for that row, and the
 *   gate doesn't ask twice (it takes the allowance and never reads the items); the notices are read again;
 *   a failed Add keeps the review open; one review at a time, whichever opened first;
 * - an install of the Cloud item from a card (through the gate's review) clears its notice too;
 * - a plugin's notice has no Add, and its Dismiss hides it for that revision only, also with storage blocked;
 * - no notice when the list is empty or unreadable, and nothing on the hosted edition.
 */
import type { ReactNode } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const edition = vi.hoisted(() => ({ local: true }))
const toast = vi.hoisted(() => Object.assign(vi.fn(), { success: vi.fn(), error: vi.fn() }))

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
vi.mock('sonner', () => ({ toast }))

import { cloudItemsKeys } from '@/hooks/use-cloud-items'
import { useInstallPlaybookFromMarketplace } from '@/hooks/use-playbook-api'
import { apiClient } from '@/lib/api-client'
import type { AgentDisclosure, CloudUpdate, PluginDisclosure } from '@/lib/marketplace/cloud-items'
import type { CloudItems } from '@/lib/marketplace/cloud-item-rows'
import { CLOUD_INSTALL_BUSY } from '@/lib/marketplace/cloud-install-gate'

import { CloudItemsLayer } from '../cloud-items-layer'
import { newerVersionLine, pluginUpdatedLine } from '../item-copy'
import { DISMISSED_KEY_PREFIX, dismissedKey } from '../update-notices'

const PLUGIN_ID = '1b4e28ba-2fa1-11d2-883f-0016d3cca427'
const SYNCED = '2026-10-30T10:00:00+00:00'

const AGENT_UPDATE: CloudUpdate = {
  kind: 'agent', cloudId: 7, name: 'Ledger Helper',
  installed: [{ id: 70, name: 'Ledger Helper', version: '1.1.0+aaaaaaaaaaaa' }],
  available: { version: '1.2.0', revision: 'b'.repeat(64), syncedAt: SYNCED },
}
const PLAYBOOK_UPDATE: CloudUpdate = {
  kind: 'playbook', cloudId: 9, name: 'Month End Close',
  installed: [{ id: 90, name: 'Month End Close', version: '1.0.0+aaaaaaaaaaaa' }],
  available: { version: '1.1.0', revision: 'c'.repeat(64), syncedAt: SYNCED },
}
const PLUGIN_UPDATE: CloudUpdate = {
  kind: 'plugin', cloudId: PLUGIN_ID, name: 'Invoice Tools', installed: [],
  available: { version: '2.0.0', revision: 'd'.repeat(64), syncedAt: SYNCED },
}

function row(name: string) {
  return { externalId: `ext-${name}`, version: '1.2.0', revision: 'a'.repeat(64), name }
}

const ITEMS: CloudItems = {
  agents: [{ id: 7, ...row('Ledger Helper') }],
  playbooks: [{ id: 9, ...row('Month End Close') }],
  packages: [],
  plugins: [{ id: PLUGIN_ID, ...row('Invoice Tools') }],
}

const HEADER = {
  version: '1.2.0', syncedAt: SYNCED, trust: 'reviewed' as const, publisher: 'Cloud Publisher', unresolved: [],
}

const AGENT: AgentDisclosure = {
  ...HEADER, kind: 'agent', name: 'Ledger Helper',
  prompt: { source: 'custom', text: 'You keep the books.', personaName: null },
  model: null, apps: [], skills: [], plugins: [],
}

const PLUGIN: PluginDisclosure = {
  ...HEADER, kind: 'plugin', name: 'Invoice Tools', version: '2.0.0',
  scan: null, localCheck: null, contents: { skills: 1, commands: 0, agents: 0, hooks: 0 }, skills: ['Invoice Reader'],
  updatesInPlace: true,
}

const PLAYBOOK = {
  ...HEADER, kind: 'playbook', name: 'Month End Close', steps: [], agents: [], apps: [],
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

/** What GET /updates answers; an install takes its item's notice off, as a copy at the new version would. */
let updatesAnswer: unknown
/** When set, every install fails on the server. */
let installsFail = false

function disclosureOf(path: string): unknown {
  if (path.endsWith('/items/agent/7/disclosure')) return AGENT
  if (path.endsWith('/items/playbook/9/disclosure')) return PLAYBOOK
  if (path.endsWith(`/items/plugin/${PLUGIN_ID}/disclosure`)) return PLUGIN
  return null
}

function installed(path: string): Response {
  if (installsFail) return json({ detail: 'The install failed.' }, 500)
  const kind = path.includes('/workflow-recipes/') ? 'playbook' : 'agent'
  const { updates } = updatesAnswer as { updates: CloudUpdate[] }
  updatesAnswer = { updates: updates.filter((update) => update.kind !== kind) }
  return json({ success: true, message: 'Added', warnings: [] })
}

function backend() {
  return vi.fn(async (url: string, init?: RequestInit) => {
    const path = String(url)
    if (path.endsWith('/api/marketplace/cloud-sync/updates')) return json(updatesAnswer)
    if (path.endsWith('/api/marketplace/cloud-sync/items')) return json(ITEMS)
    const disclosure = disclosureOf(path)
    if (disclosure) return json(disclosure)
    if (init?.method === 'POST') return installed(path)
    return json({ detail: 'Not found' }, 404)
  })
}

let fetchMock: ReturnType<typeof backend>

function requested(suffix: string, method = 'GET'): string[] {
  return fetchMock.mock.calls
    .filter(([url, init]) => String(url).endsWith(suffix) && (init?.method ?? 'GET') === method)
    .map(([url]) => String(url))
}

function installsSent(): string[] {
  return fetchMock.mock.calls.filter(([, init]) => init?.method === 'POST').map(([url]) => String(url))
}

function renderLayer(beside?: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const view = render(
    <QueryClientProvider client={client}>
      <CloudItemsLayer />
      {beside}
    </QueryClientProvider>,
  )
  return { ...view, client }
}

/** A Playbook card's Add, as the Playbooks tab makes it: the install hook, through the gate. */
function PlaybookCard() {
  const install = useInstallPlaybookFromMarketplace()
  return <button type="button" onClick={() => install.mutate(9)}>Add Month End Close</button>
}

/** The notice that names `name`. */
async function noticeFor(name: string): Promise<HTMLElement> {
  const notices = await screen.findAllByTestId('cloud-update-notice')
  const notice = notices.find((element) => element.textContent?.includes(name))
  if (!notice) throw new Error(`no notice for ${name}`)
  return notice
}

/** Waits until the updates read settled as `status`, so a test can then say nothing is shown. */
async function updatesSettled(client: QueryClient, status: 'success' | 'error'): Promise<void> {
  await waitFor(() => expect(client.getQueryState(cloudItemsKeys.updates('ws-1'))?.status).toBe(status))
  await act(async () => {})
}

beforeEach(() => {
  edition.local = true
  updatesAnswer = { updates: [AGENT_UPDATE, PLAYBOOK_UPDATE, PLUGIN_UPDATE] }
  installsFail = false
  fetchMock = backend()
  vi.stubGlobal('fetch', fetchMock)
  window.localStorage.clear()
  vi.spyOn(console, 'log').mockImplementation(() => {})
  vi.spyOn(console, 'error').mockImplementation(() => {})
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  toast.mockClear()
  toast.success.mockClear()
  toast.error.mockClear()
})

describe('the notices', () => {
  it('shows one notice per update, with its name', async () => {
    renderLayer()

    expect(await screen.findAllByTestId('cloud-update-notice')).toHaveLength(3)
    expect(screen.getByText(newerVersionLine('Ledger Helper'))).toBeInTheDocument()
    expect(screen.getByText(newerVersionLine('Month End Close'))).toBeInTheDocument()
    expect(screen.getByText(pluginUpdatedLine('Invoice Tools', '2.0.0'))).toBeInTheDocument()

    const agent = await noticeFor('Ledger Helper')
    expect(within(agent).getByRole('button', { name: 'Review the new version' })).toBeInTheDocument()
    expect(within(agent).queryByTestId('cloud-update-dismiss')).not.toBeInTheDocument()
    const plugin = await noticeFor('Invoice Tools')
    expect(within(plugin).getByRole('button', { name: 'Review' })).toBeInTheDocument()
    expect(within(plugin).getByRole('button', { name: 'Dismiss' })).toBeInTheDocument()
  })

  it('opens the disclosure for the Cloud row on Review, and sends nothing', async () => {
    renderLayer()

    fireEvent.click(within(await noticeFor('Ledger Helper')).getByTestId('cloud-update-review'))

    expect(await screen.findByTestId('cloud-disclosure-add')).toBeInTheDocument()
    expect(within(screen.getByTestId('cloud-disclosure')).getByText('You keep the books.')).toBeInTheDocument()
    expect(requested('/items/agent/7/disclosure')).toHaveLength(1)
    expect(installsSent()).toEqual([])
  })
})

describe('Add from a notice', () => {
  it('sends one install request for the Cloud agent, the gate does not ask twice, and the notice clears', async () => {
    renderLayer()

    fireEvent.click(within(await noticeFor('Ledger Helper')).getByTestId('cloud-update-review'))
    fireEvent.click(await screen.findByTestId('cloud-disclosure-add'))

    await waitFor(() => expect(installsSent()).toHaveLength(1))
    expect(installsSent()[0].endsWith('/api/marketplace/items/7/install')).toBe(true)
    // The gate took the allowance: it never read the Cloud items to ask again.
    expect(requested('/api/marketplace/cloud-sync/items')).toEqual([])
    await waitFor(() => expect(screen.queryByTestId('cloud-disclosure')).not.toBeInTheDocument())
    await waitFor(() => expect(screen.queryByText(newerVersionLine('Ledger Helper'))).not.toBeInTheDocument())
    expect(requested('/api/marketplace/cloud-sync/updates').length).toBeGreaterThan(1)
    expect(installsSent()).toHaveLength(1)
  })

  it('installs a new copy of a Cloud Playbook through its own install call, once', async () => {
    renderLayer()

    fireEvent.click(within(await noticeFor('Month End Close')).getByTestId('cloud-update-review'))
    fireEvent.click(await screen.findByTestId('cloud-disclosure-add'))

    await waitFor(() => expect(installsSent()).toHaveLength(1))
    expect(installsSent()[0].endsWith('/api/workflow-recipes/install/9')).toBe(true)
    expect(requested('/api/marketplace/cloud-sync/items')).toEqual([])
    await waitFor(() => expect(screen.queryByTestId('cloud-disclosure')).not.toBeInTheDocument())
    expect(toast.success).toHaveBeenCalledWith('Playbook added to your workspace', { description: 'Added' })
  })

  it('keeps the review open when the install fails, and Add can be pressed again', async () => {
    installsFail = true
    renderLayer()

    fireEvent.click(within(await noticeFor('Ledger Helper')).getByTestId('cloud-update-review'))
    fireEvent.click(await screen.findByTestId('cloud-disclosure-add'))

    await waitFor(() => expect(installsSent()).toHaveLength(1))
    await waitFor(() => expect(screen.getByTestId('cloud-disclosure-add')).not.toBeDisabled())
    expect(screen.getByTestId('cloud-disclosure')).toBeInTheDocument()
    expect(screen.getByText(newerVersionLine('Ledger Helper'))).toBeInTheDocument()
    expect(requested('/api/marketplace/cloud-sync/items')).toEqual([])
  })
})

describe('one review at a time', () => {
  it('refuses an install from a card while a notice review is open', async () => {
    renderLayer()
    fireEvent.click(within(await noticeFor('Ledger Helper')).getByTestId('cloud-update-review'))
    await screen.findByTestId('cloud-disclosure-add')

    let card: Promise<unknown> = Promise.resolve()
    await act(async () => {
      card = apiClient.post('/api/marketplace/items/7/install')
      card.catch(() => {})
    })

    await expect(card).rejects.toThrow(CLOUD_INSTALL_BUSY)
    expect(screen.getAllByTestId('cloud-disclosure')).toHaveLength(1)
    expect(installsSent()).toEqual([])
  })

  it('says so, and opens nothing, when Review is pressed while a card review is open', async () => {
    renderLayer()
    const notice = await noticeFor('Ledger Helper')
    await act(async () => {
      apiClient.post('/api/marketplace/items/7/install').catch(() => {})
    })
    await screen.findByTestId('cloud-disclosure-add')

    fireEvent.click(within(notice).getByTestId('cloud-update-review'))

    expect(toast.error).toHaveBeenCalledWith(CLOUD_INSTALL_BUSY)
    expect(screen.getAllByTestId('cloud-disclosure')).toHaveLength(1)
    expect(installsSent()).toEqual([])
  })
})

describe('an install from a card', () => {
  it('asks through the review, and its install clears the notice', async () => {
    renderLayer(<PlaybookCard />)
    await noticeFor('Month End Close')

    fireEvent.click(screen.getByRole('button', { name: 'Add Month End Close' }))
    fireEvent.click(await screen.findByTestId('cloud-disclosure-add'))

    await waitFor(() => expect(installsSent()).toHaveLength(1))
    expect(installsSent()[0].endsWith('/api/workflow-recipes/install/9')).toBe(true)
    await waitFor(() => expect(screen.queryByText(newerVersionLine('Month End Close'))).not.toBeInTheDocument())
    expect(screen.getByText(newerVersionLine('Ledger Helper'))).toBeInTheDocument()
  })
})

describe('an updated plugin', () => {
  it('opens its disclosure with no Add', async () => {
    renderLayer()

    fireEvent.click(within(await noticeFor('Invoice Tools')).getByTestId('cloud-update-review'))

    const dialog = await screen.findByTestId('cloud-disclosure')
    expect(await within(dialog).findByText('Invoice Reader')).toBeInTheDocument()
    expect(screen.queryByTestId('cloud-disclosure-add')).not.toBeInTheDocument()
    fireEvent.click(screen.getByTestId('cloud-disclosure-close'))
    await waitFor(() => expect(screen.queryByTestId('cloud-disclosure')).not.toBeInTheDocument())
    expect(installsSent()).toEqual([])
  })

  it('is hidden by Dismiss for that revision only', async () => {
    updatesAnswer = { updates: [PLUGIN_UPDATE] }
    const first = renderLayer()

    fireEvent.click(within(await noticeFor('Invoice Tools')).getByTestId('cloud-update-dismiss'))

    expect(screen.queryByTestId('cloud-update-notices')).not.toBeInTheDocument()
    expect(window.localStorage.getItem(dismissedKey('ws-1', PLUGIN_UPDATE))).toBe('1')
    first.unmount()

    // The same revision stays dismissed for this viewer (beside a notice that shows).
    updatesAnswer = { updates: [AGENT_UPDATE, PLUGIN_UPDATE] }
    const second = renderLayer()
    expect(await noticeFor('Ledger Helper')).toBeInTheDocument()
    expect(screen.getAllByTestId('cloud-update-notice')).toHaveLength(1)
    expect(screen.queryByText(pluginUpdatedLine('Invoice Tools', '2.0.0'))).not.toBeInTheDocument()
    second.unmount()

    // A later update, a new revision, shows again.
    const moved: CloudUpdate = {
      ...PLUGIN_UPDATE, available: { ...PLUGIN_UPDATE.available, version: '2.1.0', revision: 'e'.repeat(64) },
    }
    updatesAnswer = { updates: [moved] }
    renderLayer()
    expect(await screen.findByText(pluginUpdatedLine('Invoice Tools', '2.1.0'))).toBeInTheDocument()
  })

  it('still renders, and Dismiss still hides it for the page, when localStorage throws', async () => {
    // The notices' reads and writes throw; apiClient's own workspace header read is left working.
    const { getItem, setItem } = Storage.prototype
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(function (this: Storage, key: string) {
      if (key.startsWith(DISMISSED_KEY_PREFIX)) throw new Error('storage blocked')
      return getItem.call(this, key)
    })
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(function (this: Storage, key: string, value: string) {
      if (key.startsWith(DISMISSED_KEY_PREFIX)) throw new Error('storage blocked')
      setItem.call(this, key, value)
    })
    renderLayer()

    const plugin = await noticeFor('Invoice Tools')
    expect(screen.getAllByTestId('cloud-update-notice')).toHaveLength(3)
    expect(Storage.prototype.getItem).toHaveBeenCalledWith(dismissedKey('ws-1', PLUGIN_UPDATE))
    fireEvent.click(within(plugin).getByTestId('cloud-update-dismiss'))

    expect(Storage.prototype.setItem).toHaveBeenCalledWith(dismissedKey('ws-1', PLUGIN_UPDATE), '1')
    expect(screen.queryByText(pluginUpdatedLine('Invoice Tools', '2.0.0'))).not.toBeInTheDocument()
    expect(screen.getAllByTestId('cloud-update-notice')).toHaveLength(2)
  })
})

describe('no notice', () => {
  it('when the list is empty', async () => {
    updatesAnswer = { updates: [] }
    const { client } = renderLayer()

    await updatesSettled(client, 'success')

    expect(screen.queryByTestId('cloud-update-notices')).not.toBeInTheDocument()
  })

  it('when the answer is unreadable (an agent keyed by a string)', async () => {
    updatesAnswer = { updates: [{ ...AGENT_UPDATE, cloudId: '7' }] }
    const { client } = renderLayer()

    await updatesSettled(client, 'error')

    expect(screen.queryByTestId('cloud-update-notices')).not.toBeInTheDocument()
  })

  it('on the hosted edition: nothing renders and nothing is asked', async () => {
    edition.local = false
    const { container } = renderLayer()

    await act(async () => {})

    expect(container).toBeEmptyDOMElement()
    expect(fetchMock).not.toHaveBeenCalled()
  })
})
