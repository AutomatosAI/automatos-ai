/**
 * PRD-270 W2 US-270-14d (K3, K19) — apiClient.request() consults the Cloud install gate once, before it
 * reads a token or sends anything.
 *
 * - the gate is called with the endpoint, the options and the client itself, and fetch waits for it;
 * - a gate that refuses means no fetch, no token read, and the gate's error is the request's;
 * - a gate that returns at once lets the request go unchanged;
 * - the generic methods (post) and the named install methods reach it the same way.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const gate = vi.hoisted(() => vi.fn())

vi.mock('@/lib/marketplace/cloud-install-gate', () => ({ cloudInstallGate: gate }))

import { apiClient } from '@/lib/api-client'

const INSTALL_PATH = '/api/marketplace/items/7/install'

function stubFetch() {
  const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) => new Response(JSON.stringify({ success: true }), {
    status: 200, headers: { 'Content-Type': 'application/json' },
  }))
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

function deferred() {
  let resolve: () => void = () => {}
  const promise = new Promise<void>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

async function flush(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 0))
}

beforeEach(() => {
  gate.mockReset()
  vi.spyOn(console, 'log').mockImplementation(() => {})
  vi.spyOn(console, 'error').mockImplementation(() => {})
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('request() and the Cloud install gate', () => {
  it('asks the gate with the endpoint, the options and itself, and sends only once it allows', async () => {
    const fetchMock = stubFetch()
    const allowed = deferred()
    gate.mockReturnValue(allowed.promise)

    const answer = apiClient.post(INSTALL_PATH)
    await flush()

    expect(gate).toHaveBeenCalledTimes(1)
    expect(gate).toHaveBeenCalledWith(INSTALL_PATH, expect.objectContaining({ method: 'POST' }), apiClient)
    expect(fetchMock).not.toHaveBeenCalled()

    allowed.resolve()
    await expect(answer).resolves.toEqual({ success: true })
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(String(fetchMock.mock.calls[0][0]).endsWith(INSTALL_PATH)).toBe(true)
  })

  it('lets a request the gate passes at once go unchanged', async () => {
    const fetchMock = stubFetch()
    gate.mockReturnValue(undefined)

    await apiClient.get('/api/marketplace/items?limit=20')

    expect(gate).toHaveBeenCalledWith('/api/marketplace/items?limit=20', expect.objectContaining({ method: 'GET' }), apiClient)
    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [url, init] = fetchMock.mock.calls[0]
    expect(String(url).endsWith('/api/marketplace/items?limit=20')).toBe(true)
    expect(init?.method).toBe('GET')
  })

  it('reaches the gate from the named install methods with their bodies', async () => {
    stubFetch()
    gate.mockReturnValue(undefined)

    await apiClient.enablePlugin('ws-1', 'plugin-1')
    await apiClient.installRecipeFromMarketplace(12)

    expect(gate).toHaveBeenNthCalledWith(1, '/api/workspaces/ws-1/plugins', expect.objectContaining({
      method: 'POST', body: JSON.stringify({ plugin_id: 'plugin-1' }),
    }), apiClient)
    expect(gate).toHaveBeenNthCalledWith(2, '/api/workflow-recipes/install/12', expect.objectContaining({ method: 'POST' }), apiClient)
  })

  it('sends nothing, and reads no token, when the gate refuses', async () => {
    const fetchMock = stubFetch()
    const token = vi.fn(async () => 'token')
    apiClient.setClerkTokenGetter(token)
    const refusal = new Error('Nothing was added: you closed the review.')
    gate.mockRejectedValue(refusal)

    await expect(apiClient.installMarketplaceItem(7)).rejects.toBe(refusal)

    expect(fetchMock).not.toHaveBeenCalled()
    expect(token).not.toHaveBeenCalled()
  })
})
