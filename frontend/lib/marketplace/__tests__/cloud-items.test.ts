/**
 * PRD-270 W2 US-270-14b (K18, K19) — the three local reads of Automatos Cloud items.
 *
 * - each read asks apiClient.get on its own path: the items, one item's disclosure (kind and key in the
 *   path, encoded), the caller's workspace's updates;
 * - a failed read throws the client's error, never an invented answer;
 * - an items answer that isn't four lists of rows carrying their install keys and texts throws too, so
 *   a caller never reads it as "no Cloud items";
 * - isCloudItem matches a row by its kind's list and install key only.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ get: vi.fn() }))
vi.mock('@/lib/api-client', () => ({ apiClient: api }))

import {
  CLOUD_ITEMS_ENDPOINT,
  CLOUD_ITEMS_UNREADABLE,
  CLOUD_UPDATES_ENDPOINT,
  cloudItemsOf,
  getCloudDisclosure,
  getCloudItems,
  getCloudUpdates,
  isCloudItem,
} from '../cloud-items'
import type { CloudItems, CloudUpdates } from '../cloud-items'

const PLUGIN_ID = '3f2a8c1e-0b4d-4e6f-9a7b-1c2d3e4f5a6b'

function row(name: string) {
  return { externalId: `ext-${name}`, version: '1.0.0', revision: 'a'.repeat(64), name }
}

const ITEMS: CloudItems = {
  agents: [{ id: 7, ...row('Ledger Helper') }],
  playbooks: [{ id: 12, ...row('Weekly Close') }],
  packages: [{ slug: 'shop-starter', ...row('Shop Starter') }],
  plugins: [{ id: PLUGIN_ID, ...row('Invoice Tools') }],
}

beforeEach(() => {
  api.get.mockReset()
})

describe('the three reads', () => {
  it('reads the Cloud items from GET /api/marketplace/cloud-sync/items', async () => {
    api.get.mockResolvedValue(ITEMS)

    await expect(getCloudItems()).resolves.toEqual(ITEMS)
    expect(CLOUD_ITEMS_ENDPOINT).toBe('/api/marketplace/cloud-sync/items')
    expect(api.get).toHaveBeenCalledWith('/api/marketplace/cloud-sync/items')
  })

  it.each([
    ['agent', 7, '/api/marketplace/cloud-sync/items/agent/7/disclosure'],
    ['playbook', 12, '/api/marketplace/cloud-sync/items/playbook/12/disclosure'],
    ['package', 'shop-starter', '/api/marketplace/cloud-sync/items/package/shop-starter/disclosure'],
    ['plugin', PLUGIN_ID, `/api/marketplace/cloud-sync/items/plugin/${PLUGIN_ID}/disclosure`],
    ['package', 'a/b?c', '/api/marketplace/cloud-sync/items/package/a%2Fb%3Fc/disclosure'],
  ] as const)('reads a %s disclosure by its key (%s)', async (kind, key, path) => {
    const answer = { kind, name: 'Item', version: '1.0.0', syncedAt: null, trust: 'reviewed', publisher: null, unresolved: [] }
    api.get.mockResolvedValue(answer)

    await expect(getCloudDisclosure(kind, key)).resolves.toBe(answer)
    expect(api.get).toHaveBeenCalledWith(path)
  })

  it('reads the update notices from GET /api/marketplace/cloud-sync/updates', async () => {
    const answer: CloudUpdates = {
      updates: [{
        kind: 'agent', cloudId: 7, name: 'Ledger Helper',
        installed: [{ id: 40, name: 'Ledger Helper', version: '0.9.0+bbbbbbbbbbbb' }],
        available: { version: '1.0.0', revision: 'a'.repeat(64), syncedAt: '2026-10-30T10:00:00+00:00' },
      }],
    }
    api.get.mockResolvedValue(answer)

    await expect(getCloudUpdates()).resolves.toBe(answer)
    expect(CLOUD_UPDATES_ENDPOINT).toBe('/api/marketplace/cloud-sync/updates')
    expect(api.get).toHaveBeenCalledWith('/api/marketplace/cloud-sync/updates')
  })
})

describe('a failed read', () => {
  const failure = new Error('Not Found')

  it.each([
    ['the items', () => getCloudItems()],
    ['a disclosure', () => getCloudDisclosure('agent', 7)],
    ['the updates', () => getCloudUpdates()],
  ])('throws the client’s error for %s', async (_label, read) => {
    api.get.mockRejectedValue(failure)

    await expect(read()).rejects.toBe(failure)
  })

  it.each([
    ['nothing', null],
    ['a list', []],
    ['an empty object', {}],
    ['a missing list', { agents: [], playbooks: [], packages: [] }],
    ['a list that is not one', { ...ITEMS, plugins: {} }],
    ['an agent keyed by text', { ...ITEMS, agents: [{ ...ITEMS.agents[0], id: '7' }] }],
    ['a package with no slug', { ...ITEMS, packages: [row('No Slug')] }],
    ['a plugin keyed by a number', { ...ITEMS, plugins: [{ ...ITEMS.plugins[0], id: 3 }] }],
    ['a row that is not an object', { ...ITEMS, playbooks: [12] }],
    ['a row with no name', { ...ITEMS, agents: [{ id: 7, externalId: 'ext', version: '1.0.0', revision: 'a'.repeat(64) }] }],
    ['a row whose version is not text', { ...ITEMS, plugins: [{ ...ITEMS.plugins[0], version: 1 }] }],
  ])('throws for an items answer that is %s, never reading it as no Cloud items', async (_label, answer) => {
    api.get.mockResolvedValue(answer)

    await expect(getCloudItems()).rejects.toThrow(CLOUD_ITEMS_UNREADABLE)
    expect(() => cloudItemsOf(answer)).toThrow(CLOUD_ITEMS_UNREADABLE)
  })
})

describe('isCloudItem', () => {
  it('is false while the items are not read', () => {
    expect(isCloudItem(undefined, 'agent', 7)).toBe(false)
  })

  it.each([
    ['agent', 7], ['agent', '7'], ['playbook', 12], ['package', 'shop-starter'],
    ['plugin', PLUGIN_ID], ['plugin', PLUGIN_ID.toUpperCase()],
  ] as const)('finds the %s %s by its install key', (kind, key) => {
    expect(isCloudItem(ITEMS, kind, key)).toBe(true)
  })

  it.each([
    ['agent', 12], ['playbook', 7], ['agent', 8], ['package', 'Shop-Starter'],
    ['package', 'shop'], ['plugin', 'shop-starter'],
  ] as const)('does not find the %s %s: another kind or another key', (kind, key) => {
    expect(isCloudItem(ITEMS, kind, key)).toBe(false)
  })
})
