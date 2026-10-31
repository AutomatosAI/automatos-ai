/**
 * PRD-270 W2 US-270-14d (§5.6, §5.8, K19) — the install gate apiClient.request() consults.
 *
 * - each of K19's four install calls for a Cloud item waits for the disclosure and goes on Add (the
 *   plugin call by its body's plugin_id, a JSON string or an object);
 * - Close throws CloudInstallDeclined, so the request is never sent;
 * - a local item, a GET and any other request pass, without the opener: every other request returns at
 *   once, with nothing to await;
 * - the hosted edition passes everything and reads nothing;
 * - no opener registered, a failed items read, an unreadable items answer, an unreadable plugin body and
 *   a key no install route could take refuse with K19's sentences;
 * - another spelling of an install call (query, trailing slash, escapes, dot segments, backslashes, a tab,
 *   "07", a uuid's other forms, a lower-case method) is still that call;
 * - an allowance lets one install through, once, and expires;
 * - the gate's own read of the Cloud items is kept a minute and dropped when a layer registers;
 * - the gate's files import nothing from lib/api-client.ts.
 */
import { readFileSync } from 'fs'
import path from 'path'
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

import type { CloudInstallCall } from '../cloud-install-call'
import type { CloudItems } from '../cloud-item-rows'

type Gate = typeof import('../cloud-install-gate')

const PLUGIN_ID = '3f2a8c1e-0b4d-4e6f-9a7b-1c2d3e4f5a6b'
const ITEMS_PATH = '/api/marketplace/cloud-sync/items'

function row(name: string) {
  return { externalId: `ext-${name}`, version: '1.0.0', revision: 'a'.repeat(64), name }
}

const ITEMS: CloudItems = {
  agents: [{ id: 7, ...row('Ledger Helper') }],
  playbooks: [{ id: 12, ...row('Weekly Close') }],
  packages: [{ slug: 'shop-starter', ...row('Shop Starter') }],
  plugins: [{ id: PLUGIN_ID, ...row('Invoice Tools') }],
}

const POST = { method: 'POST' }

/** K19's four install calls, each naming its Cloud row. */
const CLOUD_INSTALLS = [
  ['agent', 7, '/api/marketplace/items/7/install', POST],
  ['playbook', 12, '/api/workflow-recipes/install/12', POST],
  ['package', 'shop-starter', '/api/marketplace/packages/shop-starter/install', POST],
  ['plugin', PLUGIN_ID, '/api/workspaces/ws-1/plugins', { method: 'POST', body: JSON.stringify({ plugin_id: PLUGIN_ID }) }],
  ['plugin', PLUGIN_ID, '/api/workspaces/ws-1/plugins', { method: 'POST', body: { plugin_id: PLUGIN_ID } as unknown as BodyInit }],
] as const

let gate: Gate
let client: { get: ReturnType<typeof vi.fn> }

/** An opener whose answer the test gives: `answer(true)` is Add, `answer(false)` is Close. */
function deferredOpener(items: () => Promise<CloudItems> = async () => ITEMS) {
  let answer: (added: boolean) => void = () => {}
  const open = vi.fn((_call: CloudInstallCall) => new Promise<boolean>((resolve) => {
    answer = resolve
  }))
  const unregister = gate.registerCloudInstallOpener({ open, items })
  return { open, unregister, answer: (added: boolean) => answer(added) }
}

/** Whether `promise` has settled by the time pending work has run. */
async function settled(promise: Promise<unknown>): Promise<boolean> {
  let done = false
  promise.then(() => { done = true }, () => { done = true })
  await new Promise((resolve) => setTimeout(resolve, 0))
  return done
}

beforeEach(async () => {
  edition.local = true
  vi.resetModules()
  gate = await import('../cloud-install-gate')
  client = { get: vi.fn(async () => ITEMS) }
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('a Cloud item install on the local edition', () => {
  it.each(CLOUD_INSTALLS)('waits for the %s disclosure and goes on Add (%s)', async (kind, key, endpoint, options) => {
    const opener = deferredOpener()

    const waiting = gate.cloudInstallGate(endpoint, options as RequestInit, client) as Promise<void>

    expect(waiting).toBeInstanceOf(Promise)
    expect(await settled(waiting)).toBe(false)
    expect(opener.open).toHaveBeenCalledWith({ kind, key })
    opener.answer(true)
    await expect(waiting).resolves.toBeUndefined()
    expect(opener.open).toHaveBeenCalledTimes(1)
  })

  it.each(CLOUD_INSTALLS)('refuses the %s install on Close, and nothing is sent (%s)', async (_kind, _key, endpoint, options) => {
    const opener = deferredOpener()
    const send = vi.fn()

    const request = (async () => {
      await gate.cloudInstallGate(endpoint, options as RequestInit, client)
      send()
    })()
    await settled(request)
    opener.answer(false)

    await expect(request).rejects.toBeInstanceOf(gate.CloudInstallDeclined)
    await expect(request).rejects.toThrow('Nothing was added: you closed the review.')
    expect(send).not.toHaveBeenCalled()
  })

  it.each([
    ['a query and a trailing slash', '/api/marketplace/items/7/install/?from=card', POST, 'agent', 7],
    ['an escaped path', '/api/marketplace/%69tems/7/install', POST, 'agent', 7],
    ['dot segments', '/api/marketplace/items/1/../7/install', POST, 'agent', 7],
    ['escaped dot segments', '/api/x/%2e%2e/marketplace/items/7/install', POST, 'agent', 7],
    ['backslashes', '/api\\marketplace\\items\\7\\install', POST, 'agent', 7],
    ['a tab in the path', '/api/workflow-recipes/inst\tall/12', POST, 'playbook', 12],
    ['a padded id', '/api/workflow-recipes/install/012', POST, 'playbook', 12],
    ['a lower-case method', '/api/marketplace/packages/shop-starter/install', { method: 'post' }, 'package', 'shop-starter'],
    ['an upper-case uuid', '/api/workspaces/ws-1/plugins', { method: 'POST', body: JSON.stringify({ plugin_id: PLUGIN_ID.toUpperCase() }) }, 'plugin', PLUGIN_ID],
    ['a braced uuid', '/api/workspaces/ws-1/plugins', { method: 'POST', body: { plugin_id: `{${PLUGIN_ID}}` } }, 'plugin', PLUGIN_ID],
    ['a uuid without hyphens', '/api/workspaces/ws-1/plugins/', { method: 'POST', body: { plugin_id: PLUGIN_ID.replace(/-/g, '') } }, 'plugin', PLUGIN_ID],
  ] as const)('reads %s as the same install call', async (_label, endpoint, options, kind, key) => {
    const opener = deferredOpener()

    const waiting = gate.cloudInstallGate(endpoint, options as unknown as RequestInit, client) as Promise<void>
    await settled(waiting)

    expect(opener.open).toHaveBeenCalledWith({ kind, key })
    opener.answer(true)
    await expect(waiting).resolves.toBeUndefined()
  })
})

describe('what passes', () => {
  it('lets a local item install go without asking', async () => {
    const opener = deferredOpener()

    await expect(gate.cloudInstallGate('/api/marketplace/items/8/install', POST, client)).resolves.toBeUndefined()
    await expect(gate.cloudInstallGate('/api/marketplace/packages/my-own/install', POST, client)).resolves.toBeUndefined()
    await expect(gate.cloudInstallGate('/api/workspaces/ws-1/plugins', {
      method: 'POST', body: JSON.stringify({ plugin_id: '00000000-0000-4000-8000-000000000000' }),
    }, client)).resolves.toBeUndefined()

    expect(opener.open).not.toHaveBeenCalled()
  })

  it.each([
    ['a GET of an install path', '/api/marketplace/items/7/install', { method: 'GET' }],
    ['a request with no method', '/api/marketplace/items/7/install', {}],
    ['another POST', '/api/marketplace/items/7/review', POST],
    ['a plugin disable', `/api/workspaces/ws-1/plugins/${PLUGIN_ID}`, { method: 'DELETE' }],
    ['a plugin POST naming no plugin', '/api/workspaces/ws-1/plugins', { method: 'POST', body: JSON.stringify({}) }],
    ['a theme install', '/api/marketplace/themes/abc/install', POST],
  ] as const)('returns at once for %s, reading nothing', (_label, endpoint, options) => {
    const opener = deferredOpener()

    expect(gate.cloudInstallGate(endpoint, options as RequestInit, client)).toBeUndefined()
    expect(client.get).not.toHaveBeenCalled()
    expect(opener.open).not.toHaveBeenCalled()
  })

  it.each(CLOUD_INSTALLS)('passes every %s install on the hosted edition, reading nothing (%s)', (_kind, _key, endpoint, options) => {
    edition.local = false
    const opener = deferredOpener()

    expect(gate.cloudInstallGate(endpoint, options as RequestInit, client)).toBeUndefined()
    expect(client.get).not.toHaveBeenCalled()
    expect(opener.open).not.toHaveBeenCalled()
  })
})

describe('what refuses', () => {
  it("refuses a Cloud item with no layer to show its disclosure, and lets a local one go", async () => {
    await expect(gate.cloudInstallGate('/api/marketplace/items/7/install', POST, client)).rejects.toThrow(
      'Open it in the Marketplace to see what it will use, then add it there.',
    )
    await expect(gate.cloudInstallGate('/api/marketplace/items/8/install', POST, client)).resolves.toBeUndefined()
    expect(client.get).toHaveBeenCalledWith(ITEMS_PATH)
  })

  it('refuses when its own read of the Cloud items fails, or answers what it cannot read', async () => {
    client.get.mockRejectedValueOnce(new Error('offline'))
    await expect(gate.cloudInstallGate('/api/marketplace/items/8/install', POST, client)).rejects.toThrow(
      "Couldn't check where this item comes from. Try again.",
    )

    client.get.mockResolvedValueOnce({ agents: 'none' })
    await expect(gate.cloudInstallGate('/api/marketplace/items/8/install', POST, client)).rejects.toBeInstanceOf(
      gate.CloudInstallRefused,
    )
  })

  it("refuses when the layer's read of the Cloud items fails, and never opens", async () => {
    const opener = deferredOpener(async () => {
      throw new Error('offline')
    })

    await expect(gate.cloudInstallGate('/api/marketplace/items/8/install', POST, client)).rejects.toThrow(
      "Couldn't check where this item comes from. Try again.",
    )
    expect(opener.open).not.toHaveBeenCalled()
    expect(client.get).not.toHaveBeenCalled()
  })

  it('refuses a plugin call whose body it cannot read', async () => {
    await expect(gate.cloudInstallGate('/api/workspaces/ws-1/plugins', {
      method: 'POST', body: `{"plugin_id": "${PLUGIN_ID}", "n": NaN}`,
    }, client)).rejects.toThrow("Couldn't check where this item comes from. Try again.")
  })

  it.each([
    ['an id that is not an integer', '/api/marketplace/items/seven/install', POST],
    ['an id in other digits', '/api/workflow-recipes/install/\u0661\u0662', POST],
    ['a plugin_id that is not a uuid', '/api/workspaces/ws-1/plugins', { method: 'POST', body: JSON.stringify({ plugin_id: `0x${'0'.repeat(30)}` }) }],
  ] as const)('refuses an install call naming %s, reading nothing', async (_label, endpoint, options) => {
    const opener = deferredOpener()

    await expect(gate.cloudInstallGate(endpoint, options as RequestInit, client)).rejects.toThrow(
      "Couldn't check where this item comes from. Try again.",
    )
    expect(opener.open).not.toHaveBeenCalled()
    expect(client.get).not.toHaveBeenCalled()
  })
})

describe('allowances', () => {
  it('lets the next install of that item through without asking, once', async () => {
    const opener = deferredOpener()
    gate.allowNext('agent', 7)

    await expect(gate.cloudInstallGate('/api/marketplace/items/7/install', POST, client)).resolves.toBeUndefined()
    expect(opener.open).not.toHaveBeenCalled()

    const again = gate.cloudInstallGate('/api/marketplace/items/7/install', POST, client) as Promise<void>
    await settled(again)
    expect(opener.open).toHaveBeenCalledTimes(1)
    opener.answer(false)
    await expect(again).rejects.toBeInstanceOf(gate.CloudInstallDeclined)
  })

  it('is for that item only, matched as its call is read', async () => {
    const opener = deferredOpener()
    gate.allowNext('plugin', PLUGIN_ID.toUpperCase())

    const other = gate.cloudInstallGate('/api/marketplace/items/7/install', POST, client) as Promise<void>
    await settled(other)
    expect(opener.open).toHaveBeenCalledWith({ kind: 'agent', key: 7 })
    opener.answer(true)
    await other

    await expect(gate.cloudInstallGate('/api/workspaces/ws-1/plugins', {
      method: 'POST', body: { plugin_id: PLUGIN_ID } as unknown as BodyInit,
    }, client)).resolves.toBeUndefined()
    expect(opener.open).toHaveBeenCalledTimes(1)
  })

  it('expires when its install call does not come', async () => {
    const now = Date.now()
    const clock = vi.spyOn(Date, 'now').mockReturnValue(now)
    const opener = deferredOpener()
    gate.allowNext('playbook', 12)
    clock.mockReturnValue(now + gate.ALLOWANCE_MS + 1)

    const waiting = gate.cloudInstallGate('/api/workflow-recipes/install/12', POST, client) as Promise<void>
    await settled(waiting)

    expect(opener.open).toHaveBeenCalledTimes(1)
    opener.answer(true)
    await waiting
  })
})

describe('the Cloud items it reads', () => {
  it('keeps its own read a minute, then reads again', async () => {
    const now = Date.now()
    const clock = vi.spyOn(Date, 'now').mockReturnValue(now)

    await gate.cloudInstallGate('/api/marketplace/items/8/install', POST, client)
    await gate.cloudInstallGate('/api/marketplace/items/9/install', POST, client)
    expect(client.get).toHaveBeenCalledTimes(1)

    clock.mockReturnValue(now + gate.GATE_ITEMS_CACHE_MS + 1)
    await gate.cloudInstallGate('/api/marketplace/items/8/install', POST, client)
    expect(client.get).toHaveBeenCalledTimes(2)
  })

  it("reads through the layer while one is registered, and drops its own read when one comes or goes", async () => {
    await gate.cloudInstallGate('/api/marketplace/items/8/install', POST, client)
    expect(client.get).toHaveBeenCalledTimes(1)

    const items = vi.fn(async () => ITEMS)
    const opener = deferredOpener(items)
    await gate.cloudInstallGate('/api/marketplace/items/8/install', POST, client)
    expect(items).toHaveBeenCalledTimes(1)
    expect(client.get).toHaveBeenCalledTimes(1)

    opener.unregister()
    await gate.cloudInstallGate('/api/marketplace/items/8/install', POST, client)
    expect(client.get).toHaveBeenCalledTimes(2)
  })

  it("an earlier layer's unregister leaves a later layer registered", async () => {
    const first = deferredOpener()
    const second = deferredOpener()
    first.unregister()

    const waiting = gate.cloudInstallGate('/api/marketplace/items/7/install', POST, client) as Promise<void>
    await settled(waiting)
    expect(second.open).toHaveBeenCalledTimes(1)
    expect(first.open).not.toHaveBeenCalled()
    second.answer(true)
    await waiting
  })
})

describe('its imports', () => {
  it.each(['cloud-install-gate.ts', 'cloud-install-call.ts', 'cloud-item-rows.ts'])(
    '%s imports nothing from lib/api-client.ts, nor anything that does',
    (file) => {
      const source = readFileSync(path.resolve(__dirname, '..', file), 'utf8')
      expect(source).not.toMatch(/from ['"][^'"]*api-client['"]/)
      expect(source).not.toMatch(/from ['"][^'"]*\/cloud-items['"]/)
      expect(source).not.toMatch(/from ['"]@\/hooks\//)
    },
  )
})
