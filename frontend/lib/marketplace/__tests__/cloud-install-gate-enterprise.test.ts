/**
 * PRD-277 US-704 (D-12) — the install gate on an enterprise build, where Cloud items are the operator's
 * own setting (GET /api/hosted-services, lib/hosted-services.ts).
 *
 * - any request that isn't one of K19's four install calls returns at once and reads nothing;
 * - with Cloud items off, an install passes after reading only the settings, never the Cloud items;
 * - with Cloud items on, a Cloud item's install waits for its review as on the local edition (refused here,
 *   with no layer mounted), and a local item's install passes;
 * - when the settings can't be read, the install is refused ("Couldn't check …"), as for the Cloud items;
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

function answering(cloudItems: boolean | 'unreadable') {
  get = vi.fn(async (endpoint: string) => {
    if (endpoint === HOSTED_SERVICES_ENDPOINT) {
      if (cloudItems === 'unreadable') throw new Error('offline')
      return { themeRevocations: true, cloudSync: true, cloudItems }
    }
    if (endpoint === ITEMS_PATH) return ITEMS
    throw new Error(`unexpected request ${endpoint}`)
  })
}

const reads = () => get.mock.calls.map(([endpoint]) => endpoint)

beforeEach(async () => {
  vi.resetModules()
  gate = await import('../cloud-install-gate')
  answering(false)
})

describe('the install gate on an enterprise build', () => {
  it('returns at once for a request that is not an install, reading nothing', () => {
    expect(cloudInstallGate('/api/agents', { method: 'GET' }, client())).toBeUndefined()
    expect(cloudInstallGate(CLOUD_AGENT_INSTALL, { method: 'GET' }, client())).toBeUndefined()
    expect(get).not.toHaveBeenCalled()
  })

  it('with Cloud items off, lets an install go after reading only the settings', async () => {
    await expect(cloudInstallGate(CLOUD_AGENT_INSTALL, POST, client())).resolves.toBeUndefined()
    expect(reads()).toEqual([HOSTED_SERVICES_ENDPOINT])
  })

  it('with Cloud items on, a Cloud item waits for its review (refused with no layer mounted)', async () => {
    answering(true)
    const waiting = cloudInstallGate(CLOUD_AGENT_INSTALL, POST, client())

    await expect(waiting).rejects.toBeInstanceOf(gate.CloudInstallRefused)
    await expect(waiting).rejects.toThrow(gate.CLOUD_INSTALL_ELSEWHERE)
    expect(reads()).toEqual([HOSTED_SERVICES_ENDPOINT, ITEMS_PATH])
  })

  it('with Cloud items on, a local item passes', async () => {
    answering(true)
    await expect(cloudInstallGate(LOCAL_AGENT_INSTALL, POST, client())).resolves.toBeUndefined()
  })

  it('refuses the install when the settings cannot be read', async () => {
    answering('unreadable')
    const waiting = cloudInstallGate(CLOUD_AGENT_INSTALL, POST, client())

    await expect(waiting).rejects.toThrow(gate.CLOUD_INSTALL_UNCHECKED)
    expect(reads()).toEqual([HOSTED_SERVICES_ENDPOINT])
  })
})

describe('its imports', () => {
  it('lib/hosted-services.ts imports nothing from lib/api-client.ts, nor any hook', () => {
    const source = readFileSync(path.resolve(__dirname, '..', '..', 'hosted-services.ts'), 'utf8')
    expect(source).not.toMatch(/from ['"][^'"]*api-client['"]/)
    expect(source).not.toMatch(/from ['"]@\/hooks\//)
  })
})
