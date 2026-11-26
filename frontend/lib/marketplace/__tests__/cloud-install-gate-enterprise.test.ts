/**
 * PRD-277 US-704 (D-12) — the install gate on an enterprise build. ENTERPRISE_CALLS_CLOUD_ITEMS stops the Sync
 * from requesting items; it never stops the review of an item the install already holds, so the gate behaves
 * as on the local edition, always (lib/hosted-services.ts `holdsCloudItems`):
 *
 * - any request that isn't one of K19's four install calls returns at once and reads nothing;
 * - a Cloud item's install waits for its review (refused here, with no layer mounted), after reading only the
 *   Cloud items: GET /api/hosted-services is never asked, so an unreadable answer there can refuse nothing;
 * - a local item's install passes;
 * - when the Cloud items can't be read, the install is refused ("Couldn't check …"), as on the local edition;
 * - lib/hosted-services.ts, which the gate imports, imports nothing from lib/api-client.ts.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/edition-name', () => ({ isEnterprise: true, resolveEdition: () => 'enterprise' }))

import { HOSTED_SERVICES_ENDPOINT } from '@/lib/hosted-services'

import type { CloudGateClient } from '../cloud-install-gate'

type Gate = typeof import('../cloud-install-gate')

let gate: Gate
/** A fresh gate for each test: its own read of the Cloud items is kept a minute (K19). */
const cloudInstallGate: Gate['cloudInstallGate'] = (...args) => gate.cloudInstallGate(...args)

const ITEMS_PATH = '/api/marketplace/cloud-sync/items'
const POST = { method: 'POST' }
const CLOUD_AGENT_INSTALL = '/api/marketplace/items/7/install'
const LOCAL_AGENT_INSTALL = '/api/marketplace/items/8/install'
const ITEMS = {
  agents: [{ id: 7, externalId: 'ext-ledger', version: '1.0.0', revision: 'a'.repeat(64), name: 'Ledger Helper' }],
  playbooks: [],
  packages: [],
  plugins: [],
}

let get: ReturnType<typeof vi.fn>
const client = (): CloudGateClient => ({ get } as unknown as CloudGateClient)

/** The install answers its Cloud items (or can't); the settings read always fails, and must never be asked. */
function answering(items: 'held' | 'unreadable') {
  get = vi.fn(async (endpoint: string) => {
    if (endpoint === HOSTED_SERVICES_ENDPOINT) throw new Error('the gate must not ask for the settings')
    if (endpoint === ITEMS_PATH) {
      if (items === 'unreadable') throw new Error('offline')
      return ITEMS
    }
    throw new Error(`unexpected request ${endpoint}`)
  })
}

const reads = () => get.mock.calls.map(([endpoint]) => endpoint)

beforeEach(async () => {
  vi.resetModules()
  gate = await import('../cloud-install-gate')
  answering('held')
})

describe('the install gate on an enterprise build', () => {
  it('returns at once for a request that is not an install, reading nothing', () => {
    expect(cloudInstallGate('/api/agents', { method: 'GET' }, client())).toBeUndefined()
    expect(cloudInstallGate(CLOUD_AGENT_INSTALL, { method: 'GET' }, client())).toBeUndefined()
    expect(get).not.toHaveBeenCalled()
  })

  it('makes a Cloud item wait for its review whatever the settings say (refused with no layer mounted)', async () => {
    const waiting = cloudInstallGate(CLOUD_AGENT_INSTALL, POST, client())

    await expect(waiting).rejects.toBeInstanceOf(gate.CloudInstallRefused)
    await expect(waiting).rejects.toThrow(gate.CLOUD_INSTALL_ELSEWHERE)
    expect(reads()).toEqual([ITEMS_PATH])
  })

  it('lets a local item pass, having read only the Cloud items', async () => {
    await expect(cloudInstallGate(LOCAL_AGENT_INSTALL, POST, client())).resolves.toBeUndefined()
    expect(reads()).toEqual([ITEMS_PATH])
  })

  it('refuses the install when the Cloud items cannot be read', async () => {
    answering('unreadable')
    const waiting = cloudInstallGate(CLOUD_AGENT_INSTALL, POST, client())

    await expect(waiting).rejects.toThrow(gate.CLOUD_INSTALL_UNCHECKED)
    expect(reads()).toEqual([ITEMS_PATH])
  })
})

describe('its imports', () => {
  it('lib/hosted-services.ts imports nothing from lib/api-client.ts, nor any hook', () => {
    const source = readFileSync(path.resolve(__dirname, '..', '..', 'hosted-services.ts'), 'utf8')
    expect(source).not.toMatch(/from ['"][^'"]*api-client['"]/)
    expect(source).not.toMatch(/from ['"]@\/hooks\//)
  })

  it('the gate no longer reads the hosted-service settings', () => {
    const source = readFileSync(path.resolve(__dirname, '..', 'cloud-install-gate.ts'), 'utf8')
    expect(source).not.toMatch(/readHostedServices|editionHostedServices/)
  })
})
