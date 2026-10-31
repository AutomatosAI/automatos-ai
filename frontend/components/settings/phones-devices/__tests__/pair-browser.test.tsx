/**
 * PRD-266 US-005b — "Pair this browser" (§4.3, K8).
 *
 * On a local install in LOCAL_REMOTE_ACCESS=paired mode a browser with no device token pairs with
 * a one-time code, like a phone, before the app shows. The token is kept in browser storage and
 * rides every API call through the existing seam (LocalAuthProvider → apiClient's token getter);
 * the theme files proxy gets it from a SameSite=Strict cookie on its own path. A 401
 * device_revoked or device_pairing_required brings the screen back. /go/ stays public (W3's
 * /go/pair). SaaS, and a local install that is off or open, are unchanged.
 *
 * The real apiClient runs over a stubbed fetch, so every refusal travels the real error path
 * (api-client.ts httpErrorOf → lib/api-unauthorized.ts → the gate).
 */
import React from 'react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const nav = vi.hoisted(() => ({ pathname: '/' as string | null }))
vi.mock('next/navigation', () => ({ usePathname: () => nav.pathname }))

const TOKEN = `amd_${'t'.repeat(43)}`
const TOKEN_KEY = 'automatos.device_token'
const INSTALL_ID_KEY = 'automatos.browser_install_id'
const WORKSPACE_KEY = 'last_active_workspace'
const COOKIE = 'automatos_device_token'
const COOKIE_PATH = '/api/themes/files'
const WORKSPACE = '6f1c2b0e-4a7d-4c1e-9b2a-1d3e5f7a9c0b'
const DISCOVERY = 'GET /.well-known/automatos.json'
const PAIR = 'POST /api/v1/mobile/pair'
const AGENTS = 'GET /api/agents'
const PAIRED = { device_id: '1b9d6bcd-bbfd-4b2d-9b5d-ab8dfbbd4bed', workspace_id: WORKSPACE, token: TOKEN, rotate_after: '2027-01-19T10:00:00+00:00' }
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/
const APP = 'the app'
const SCREEN = { name: 'Pair this browser' }

type Answer = { readonly status: number; readonly body: unknown }
interface Call { readonly route: string; readonly headers: Record<string, string>; readonly body: unknown }
type ApiClient = typeof import('@/lib/api-client').apiClient
type LocalAuthProvider = typeof import('@/components/local-auth-provider').LocalAuthProvider

const ok = (body: unknown): Answer => ({ status: 200, body })
const refused = (code: string): Answer => ({ status: 401, body: { detail: { code } } })
const discovery = (enabled: boolean): Answer =>
  ok({ schema: 'automatos-discovery/v1', edition: 'local', pairing: { enabled } })

/** The API over a stubbed fetch: each "METHOD path" answers as scripted (200 {} otherwise); every call is kept. */
function stubApi(answers: Readonly<Record<string, Answer>>): Call[] {
  const calls: Call[] = []
  vi.stubGlobal('fetch', vi.fn(async (url: string, init: RequestInit = {}) => {
    const route = `${init.method ?? 'GET'} ${new URL(String(url), 'http://app.test').pathname}`
    const body = typeof init.body === 'string' ? JSON.parse(init.body) : undefined
    calls.push({ route, headers: { ...(init.headers as Record<string, string>) }, body })
    const answer = answers[route] ?? ok({})
    return new Response(JSON.stringify(answer.body), { status: answer.status, headers: { 'Content-Type': 'application/json' } })
  }))
  return calls
}

function lastCall(calls: Call[], route: string): Call | undefined {
  return calls.filter((call) => call.route === route).pop()
}

interface Mounted {
  readonly apiClient: ApiClient
  readonly getter: () => Promise<string | null>
  readonly Provider: LocalAuthProvider
  readonly rerender: (ui: React.ReactElement) => void
}

/** The local edition's auth boundary, fresh from import (a reload), around the app. */
async function mountLocal(): Promise<Mounted> {
  vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', 'local')
  vi.stubEnv('NEXT_PUBLIC_API_URL', '')
  vi.resetModules()
  const { apiClient } = await import('@/lib/api-client')
  const install = vi.spyOn(apiClient, 'setClerkTokenGetter')
  const { LocalAuthProvider: Provider } = await import('@/components/local-auth-provider')
  const { rerender } = render(<Provider><p>{APP}</p></Provider>)
  return { apiClient, getter: install.mock.calls[0][0], Provider, rerender }
}

/** The cookies a request to the theme files proxy carries (the cookie's Path is the proxy's). */
function themeFilesCookies(): string {
  window.history.pushState({}, '', `${COOKIE_PATH}/${'a'.repeat(64)}/dist/entry.js`)
  const carried = document.cookie
  window.history.pushState({}, '', '/')
  return carried
}

function typeCode(code: string, name = ''): void {
  fireEvent.change(screen.getByLabelText('Pairing code'), { target: { value: code } })
  fireEvent.change(screen.getByLabelText('Name this browser'), { target: { value: name } })
  fireEvent.click(screen.getByRole('button', { name: 'Pair' }))
}

async function refuse(apiClient: ApiClient): Promise<void> {
  await act(async () => {
    await apiClient.get('/api/agents').catch(() => undefined)
  })
}

beforeEach(() => {
  vi.spyOn(console, 'error').mockImplementation(() => undefined)
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
  vi.spyOn(console, 'log').mockImplementation(() => undefined)
})

afterEach(() => {
  cleanup()
  localStorage.clear()
  document.cookie = `${COOKIE}=; Path=${COOKIE_PATH}; Expires=Thu, 01 Jan 1970 00:00:00 GMT`
  nav.pathname = '/'
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('PRD-266 US-005b — local, paired mode', () => {
  it('shows Pair this browser (a code field and a name) instead of the app when no token is stored', async () => {
    stubApi({ [DISCOVERY]: discovery(true) })
    await mountLocal()
    expect(await screen.findByRole('heading', SCREEN)).toBeInTheDocument()
    expect(screen.getByLabelText('Pairing code')).toBeInTheDocument()
    expect(screen.getByLabelText('Name this browser')).toBeInTheDocument()
    expect(screen.queryByText(APP)).toBeNull()
  })

  it('pairs a good code as platform web, keeps the token and its workspace, and the getter returns it', async () => {
    const calls = stubApi({ [DISCOVERY]: discovery(true), [PAIR]: ok(PAIRED) })
    const { apiClient, getter } = await mountLocal()
    await screen.findByRole('heading', SCREEN)
    await expect(getter()).resolves.toBeNull()

    typeCode(' abcd efgh ', 'Office laptop')

    expect(await screen.findByText(APP)).toBeInTheDocument()
    const pair = lastCall(calls, PAIR)
    expect(pair?.body).toEqual({ code: 'abcd efgh', name: 'Office laptop', platform: 'web', install_id: expect.stringMatching(UUID) })
    expect(localStorage.getItem(INSTALL_ID_KEY)).toBe((pair?.body as { install_id: string }).install_id)
    expect(localStorage.getItem(TOKEN_KEY)).toBe(TOKEN)
    expect(localStorage.getItem(WORKSPACE_KEY)).toBe(WORKSPACE)
    await expect(getter()).resolves.toBe(TOKEN)
    await apiClient.get('/api/agents')
    expect(lastCall(calls, AGENTS)?.headers).toMatchObject({ Authorization: `Bearer ${TOKEN}`, 'X-Workspace-ID': WORKSPACE })
    expect(themeFilesCookies()).toContain(`${COOKIE}=${TOKEN}`)
  })

  it('pairs over plain http too: without crypto.randomUUID the install id comes from getRandomValues', async () => {
    const real = globalThis.crypto
    vi.stubGlobal('crypto', { getRandomValues: real.getRandomValues.bind(real) })
    const calls = stubApi({ [DISCOVERY]: discovery(true), [PAIR]: ok(PAIRED) })
    await mountLocal()
    await screen.findByRole('heading', SCREEN)

    typeCode('ABCD-EFGH')

    expect(await screen.findByText(APP)).toBeInTheDocument()
    const installId = (lastCall(calls, PAIR)?.body as { install_id: string }).install_id
    expect(installId).toMatch(/^[0-9a-f]{32}$/)
    expect(localStorage.getItem(INSTALL_ID_KEY)).toBe(installId)
  })

  it('keeps no half pairing when the browser cannot save it, and says so', async () => {
    stubApi({ [DISCOVERY]: discovery(true), [PAIR]: ok(PAIRED) })
    await mountLocal()
    await screen.findByRole('heading', SCREEN)
    const save = Storage.prototype.setItem
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation((key: string, value: string) => {
      if (key === TOKEN_KEY) throw new DOMException('The quota has been exceeded.', 'QuotaExceededError')
      save.call(localStorage, key, value)
    })

    typeCode('ABCD-EFGH')

    expect(await screen.findByRole('alert')).toHaveTextContent('could not save its pairing')
    expect(localStorage.getItem(TOKEN_KEY)).toBeNull()
    expect(screen.getByRole('heading', SCREEN)).toBeInTheDocument()
    expect(themeFilesCookies()).not.toContain(COOKIE)
  })

  it('says why a wrong code was refused and stays on the screen', async () => {
    stubApi({ [DISCOVERY]: discovery(true), [PAIR]: refused('invalid_pairing_code') })
    await mountLocal()
    await screen.findByRole('heading', SCREEN)
    const { PAIRING_MESSAGES } = await import('@/hooks/use-mobile-devices')

    typeCode('WRONG-CODE')

    expect(await screen.findByRole('alert')).toHaveTextContent(PAIRING_MESSAGES.invalidCode)
    expect(screen.getByRole('heading', SCREEN)).toBeInTheDocument()
    expect(localStorage.getItem(TOKEN_KEY)).toBeNull()
  })

  it('a 401 device_revoked clears the stored token and its cookie and returns to the screen', async () => {
    localStorage.setItem(TOKEN_KEY, TOKEN)
    const calls = stubApi({ [DISCOVERY]: discovery(true), [AGENTS]: refused('device_revoked') })
    const { apiClient, getter } = await mountLocal()
    expect(screen.getByText(APP)).toBeInTheDocument()
    await waitFor(() => expect(themeFilesCookies()).toContain(`${COOKIE}=${TOKEN}`))

    await refuse(apiClient)

    expect(await screen.findByRole('heading', SCREEN)).toBeInTheDocument()
    expect(screen.queryByText(APP)).toBeNull()
    expect(lastCall(calls, AGENTS)?.headers.Authorization).toBe(`Bearer ${TOKEN}`)
    expect(localStorage.getItem(TOKEN_KEY)).toBeNull()
    expect(themeFilesCookies()).not.toContain(COOKIE)
    await expect(getter()).resolves.toBeNull()
  })

  it('a 401 device_pairing_required shows the screen, even when discovery could not be read', async () => {
    stubApi({ [DISCOVERY]: { status: 503, body: { detail: 'down' } }, [AGENTS]: refused('device_pairing_required') })
    const { apiClient } = await mountLocal()
    expect(screen.getByText(APP)).toBeInTheDocument()

    await refuse(apiClient)

    expect(await screen.findByRole('heading', SCREEN)).toBeInTheDocument()
  })

  it('keeps a held token when a late device_pairing_required answers a request sent before it was stored', async () => {
    localStorage.setItem(TOKEN_KEY, TOKEN)
    stubApi({ [DISCOVERY]: discovery(true), [AGENTS]: refused('device_pairing_required') })
    const { apiClient, getter } = await mountLocal()
    await waitFor(() => expect(themeFilesCookies()).toContain(`${COOKIE}=${TOKEN}`))

    await refuse(apiClient)

    expect(screen.getByText(APP)).toBeInTheDocument()
    expect(screen.queryByRole('heading', SCREEN)).toBeNull()
    await expect(getter()).resolves.toBe(TOKEN)
  })

  it('stays paired after a reload and a server restart: the token comes from browser storage, never memory', async () => {
    localStorage.setItem(TOKEN_KEY, TOKEN)
    const calls = stubApi({ [DISCOVERY]: discovery(true) })
    await mountLocal()
    await waitFor(() => expect(themeFilesCookies()).toContain(`${COOKIE}=${TOKEN}`))
    cleanup()

    const reloaded = await mountLocal()

    await expect(reloaded.getter()).resolves.toBe(TOKEN)
    await reloaded.apiClient.get('/api/agents')
    expect(lastCall(calls, AGENTS)?.headers.Authorization).toBe(`Bearer ${TOKEN}`)
    expect(screen.getByText(APP)).toBeInTheDocument()
    expect(screen.queryByRole('heading', SCREEN)).toBeNull()
  })
})

describe('PRD-266 US-005b — the public paths', () => {
  it('names them in one constant: /go/', async () => {
    const gate = await import('@/components/settings/phones-devices/pair-browser-gate')
    expect(gate.PAIRING_GATE_PUBLIC_PREFIXES).toEqual(['/go/'])
    expect(gate.isPairingGatePublicPath('/go/pair')).toBe(true)
    for (const gated of ['/', '/go', '/gopher', '/settings', '/api/go/pair', null]) {
      expect(gate.isPairingGatePublicPath(gated)).toBe(false)
    }
  })

  it('renders /go/pair for an unpaired browser, and every other route still shows the gate', async () => {
    stubApi({ [DISCOVERY]: discovery(true) })
    const { Provider, rerender } = await mountLocal()
    await screen.findByRole('heading', SCREEN)

    nav.pathname = '/go/pair'
    rerender(<Provider><p>{APP}</p></Provider>)
    expect(screen.getByText(APP)).toBeInTheDocument()
    expect(screen.queryByRole('heading', SCREEN)).toBeNull()

    for (const path of ['/', '/chat', '/settings', '/go', '/gopher']) {
      nav.pathname = path
      rerender(<Provider><p>{APP}</p></Provider>)
      expect(screen.getByRole('heading', SCREEN)).toBeInTheDocument()
      expect(screen.queryByText(APP)).toBeNull()
    }
  })
})

describe('PRD-266 US-005b — SaaS, and local off or open: nothing changes', () => {
  it('SaaS: the gate is the app, and discovery is never read', async () => {
    vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', 'saas')
    vi.resetModules()
    const calls = stubApi({ [DISCOVERY]: discovery(true) })
    const { PairBrowserGate } = await import('@/components/settings/phones-devices/pair-browser-gate')
    render(<PairBrowserGate><p>{APP}</p></PairBrowserGate>)
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 20))
    })
    expect(screen.getByText(APP)).toBeInTheDocument()
    expect(screen.queryByRole('heading', SCREEN)).toBeNull()
    expect(calls).toEqual([])
  })

  it('local off or open: the app with no screen, no stored token and no bearer, a leftover token cleared', async () => {
    localStorage.setItem(TOKEN_KEY, TOKEN)
    const calls = stubApi({ [DISCOVERY]: discovery(false) })
    const { apiClient, getter } = await mountLocal()

    await waitFor(() => expect(localStorage.getItem(TOKEN_KEY)).toBeNull())

    expect(screen.getByText(APP)).toBeInTheDocument()
    expect(screen.queryByRole('heading', SCREEN)).toBeNull()
    await expect(getter()).resolves.toBeNull()
    await apiClient.get('/api/agents')
    expect(lastCall(calls, AGENTS)?.headers).not.toHaveProperty('Authorization')
    expect(themeFilesCookies()).not.toContain(COOKIE)
  })
})
