/**
 * PRD-277 US-601 — Settings → Phones & devices on an enterprise install.
 *
 * Pair a phone shows while the discovery document says the install lets people pair
 * (`edition: "enterprise"`, `pairing.enabled`), and mints a code for the signed-in person, with
 * its QR image; there is no Pair a browser, because an enterprise browser signs in. With pairing
 * off it is not shown. The Expo switch shows only to a caller the route answers (an install admin):
 * anyone else's read is 403, so they see no switch and are not asked before a code. An install
 * admin's answer that needs a fresh sign-in sends the browser to sign in again at the API's own
 * start address, and is not shown as a failure.
 *
 * The real apiClient and hooks run over a stubbed fetch, so every call travels the real path. Where
 * the browser is sent is read from `signInStartUrl`, whose answer here is a hash, so jsdom can follow it.
 */
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const DISCOVERY = 'GET /.well-known/automatos.json'
const DEVICES = 'GET /api/v1/mobile/devices'
const MINT = 'POST /api/v1/mobile/pairing-codes'
const READ_CONSENT = 'GET /api/v1/mobile/push-consent'
const SAVE_CONSENT = 'PUT /api/v1/mobile/push-consent'
const WEB_URL = 'https://automatos.example.com'
const CODE = 'ABCD-EFGH'
const LINK = `${WEB_URL}/go/pair?api=https%3A%2F%2Fautomatos.example.com&code=${CODE}`
const TEN_MINUTES_MS = 600_000
const START = '/api/v1/auth/oidc/start?step_up=1'
const SWITCH_LABEL = 'Send notifications through Expo’s push service'

const signIn = vi.hoisted(() => ({ startUrl: vi.fn((_returnTo: string | null | undefined, _start?: string) => '#signing-in-again') }))

vi.mock('@/lib/enterprise/auth-api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/lib/enterprise/auth-api')>()),
  signInStartUrl: signIn.startUrl,
}))

type Answer = { readonly status: number; readonly body: unknown }
type Client = import('@tanstack/react-query').QueryClient

const ok = (body: unknown): Answer => ({ status: 200, body })
const discovery = (enabled: boolean): Answer => ok({
  schema: 'automatos-discovery/v1', edition: 'enterprise', web_url: WEB_URL,
  auth: { modes: enabled ? ['oidc', 'device'] : ['oidc'], clerk: null, oidc: { start_url: `${WEB_URL}/api/v1/auth/oidc/start` } },
  pairing: { enabled }, push: { expo: false },
})
const MINTED = ok({ code: CODE, expires_at: new Date(Date.now() + TEN_MINUTES_MS).toISOString(), qr_payload: LINK })
const FORBIDDEN: Answer = { status: 403, body: { detail: 'Only an install admin signed in to Automatos decides how notifications are sent.' } }
// orchestrator/core/auth/step_up.py sign_in_again_body(10).
const STEP_UP: Answer = {
  status: 403,
  body: { code: 'step_up_required', detail: 'Confirm it’s you to continue: sign in again.', step_up: { start: START, max_age_minutes: 10 } },
}

/** The API over a stubbed fetch: each "METHOD path" answers as scripted (200 {} otherwise); every route is kept. */
function stubApi(answers: Readonly<Record<string, Answer>>): string[] {
  const calls: string[] = []
  vi.stubGlobal('fetch', vi.fn(async (url: string, init: RequestInit = {}) => {
    const route = `${init.method ?? 'GET'} ${new URL(String(url), 'http://app.test').pathname}`
    calls.push(route)
    const answer = answers[route] ?? ok({})
    return new Response(JSON.stringify(answer.body), { status: answer.status, headers: { 'Content-Type': 'application/json' } })
  }))
  return calls
}

/** The tab in an enterprise build, fresh from import, under its own React Query client. */
async function renderEnterpriseTab(): Promise<Client> {
  vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', 'enterprise')
  vi.stubEnv('NEXT_PUBLIC_API_URL', '')
  vi.resetModules()
  const { QueryClient, QueryClientProvider } = await import('@tanstack/react-query')
  const { PhonesDevicesTab } = await import('../phones-devices-tab')
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <PhonesDevicesTab />
    </QueryClientProvider>,
  )
  return client
}

async function settled(client: Client, key: readonly string[]): Promise<void> {
  await waitFor(() => expect(['success', 'error']).toContain(client.getQueryState(key)?.status))
}

afterEach(() => {
  cleanup()
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  window.history.replaceState(null, '', '/')
})

describe('Settings → Phones & devices in enterprise (PRD-277 US-601)', () => {
  it('a member pairs their own phone: the code and its QR image, no Pair a browser, no switch and no question', async () => {
    const calls = stubApi({ [DEVICES]: ok({ devices: [] }), [DISCOVERY]: discovery(true), [READ_CONSENT]: FORBIDDEN, [MINT]: MINTED })
    const client = await renderEnterpriseTab()

    fireEvent.click(await screen.findByRole('button', { name: 'Get a pairing code' }))

    expect(await screen.findByTestId('pairing-code')).toHaveTextContent(CODE)
    expect(screen.getByRole('img', { name: 'QR code of the pairing link' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Pair a browser' })).not.toBeInTheDocument()
    await settled(client, ['mobile-push-consent'])
    expect(screen.queryByRole('switch')).not.toBeInTheDocument()
    expect(calls).toContain(MINT)
    expect(calls).not.toContain(SAVE_CONSENT)
  })

  it('with pairing off the devices are listed and there is no Pair a phone', async () => {
    const calls = stubApi({ [DEVICES]: ok({ devices: [] }), [DISCOVERY]: discovery(false), [READ_CONSENT]: FORBIDDEN })
    const client = await renderEnterpriseTab()

    await settled(client, ['mobile-discovery'])
    expect(screen.queryByRole('heading', { name: 'Pair a phone' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Get a pairing code' })).not.toBeInTheDocument()
    expect(calls).not.toContain(MINT)
  })

  it('an install admin sees the switch, and an answer that needs a fresh sign-in sends the browser to sign in again', async () => {
    window.history.replaceState(null, '', '/settings?tab=phones-devices')
    const calls = stubApi({
      [DEVICES]: ok({ devices: [] }), [DISCOVERY]: discovery(true),
      [READ_CONSENT]: ok({ agreed: false, answered: true, answered_at: '2026-11-20T09:00:00+00:00' }), [SAVE_CONSENT]: STEP_UP,
    })
    vi.spyOn(console, 'warn').mockImplementation(() => {})
    await renderEnterpriseTab()

    fireEvent.click(await screen.findByRole('switch', { name: SWITCH_LABEL }))

    await waitFor(() => expect(signIn.startUrl).toHaveBeenCalledWith('/settings?tab=phones-devices', START))
    await waitFor(() => expect(window.location.hash).toBe('#signing-in-again'))
    expect(calls.filter((call) => call === SAVE_CONSENT)).toHaveLength(1)
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })
})
