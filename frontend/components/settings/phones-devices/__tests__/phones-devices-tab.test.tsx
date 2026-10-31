/**
 * PRD-266 US-005c — Settings → Phones & devices (§4.3 "Web").
 *
 * Both editions list the devices GET /api/v1/mobile/devices returns (name, platform, last seen,
 * push on or off) with Revoke, which calls DELETE /devices/{id} only after the confirm. Only a
 * local install in paired mode (the discovery document says so) shows Pair a phone, which mints a
 * code (POST /pairing-codes) and shows it, its ten-minute countdown and its pairing link as text
 * with a copy button (the QR image is US-019, K9), and Pair a browser. SaaS never reads discovery
 * and never shows pairing.
 *
 * The real apiClient and hooks run over a stubbed fetch, so every call travels the real path.
 */
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const DEVICE_ID = '1b9d6bcd-bbfd-4b2d-9b5d-ab8dfbbd4bed'
const DISCOVERY = 'GET /.well-known/automatos.json'
const DEVICES = 'GET /api/v1/mobile/devices'
const MINT = 'POST /api/v1/mobile/pairing-codes'
const REVOKE = `DELETE /api/v1/mobile/devices/${DEVICE_ID}`
const WEB_URL = 'https://studio.example.com'
const CODE = 'ABCD-EFGH'
const LINK = `${WEB_URL}/go/pair?api=https%3A%2F%2Fapi.example.com&code=${CODE}`
const TEN_MINUTES_MS = 600_000
const PHONE_NAME = 'Ana’s iPhone'

const PHONE = {
  id: DEVICE_ID,
  user_id: 1,
  kind: 'paired',
  name: PHONE_NAME,
  platform: 'ios',
  status: 'active',
  push_enabled: true,
  last_seen_at: new Date().toISOString(),
  paired_at: '2026-10-20T09:00:00+00:00',
}
const OLD_BROWSER = { ...PHONE, id: '7c4a8d09-ca37-4b2e-8f9a-2d1e0b3c4a5f', name: 'Old laptop', platform: 'web', status: 'revoked', push_enabled: false, last_seen_at: null }

type Answer = { readonly status: number; readonly body: unknown }
type Client = import('@tanstack/react-query').QueryClient

const ok = (body: unknown): Answer => ({ status: 200, body })
const discovery = (enabled: boolean): Answer =>
  ok({ schema: 'automatos-discovery/v1', edition: 'local', web_url: WEB_URL, pairing: { enabled } })
const mintAnswer = (expiresInMs: number): Answer =>
  ok({ code: CODE, expires_at: new Date(Date.now() + expiresInMs).toISOString(), qr_payload: LINK })

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

/** The tab in `edition`, fresh from import, under its own React Query client. */
async function renderTab(edition: 'local' | 'saas'): Promise<Client> {
  vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', edition)
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

/** Wait until the discovery document has been read, so a missing pairing panel means "not paired". */
async function discoveryRead(client: Client): Promise<void> {
  await waitFor(() => expect(client.getQueryState(['mobile-discovery'])?.status).toBe('success'))
}

const pairPhoneHeading = () => screen.queryByRole('heading', { name: 'Pair a phone' })

const originalClipboard = Object.getOwnPropertyDescriptor(window.navigator, 'clipboard')

/** This browser's clipboard, or none at all (plain http on the LAN is not a secure context). */
function setClipboard(clipboard: { writeText: (text: string) => Promise<void> } | undefined): void {
  Object.defineProperty(window.navigator, 'clipboard', { configurable: true, value: clipboard })
}

afterEach(() => {
  cleanup()
  if (originalClipboard) Object.defineProperty(window.navigator, 'clipboard', originalClipboard)
  else delete (window.navigator as { clipboard?: unknown }).clipboard
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('Settings → Phones & devices: the list (PRD-266 US-005c)', () => {
  it('renders a device: its name, platform, last seen and push state, with Revoke; a revoked one says so', async () => {
    stubApi({ [DEVICES]: ok({ devices: [PHONE, OLD_BROWSER] }), [DISCOVERY]: discovery(false) })
    await renderTab('local')

    expect(await screen.findByText(PHONE_NAME)).toBeInTheDocument()
    expect(screen.getByText('iOS · Last seen Just now · Push on')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: `Revoke ${PHONE_NAME}` })).toBeInTheDocument()

    expect(screen.getByText('Old laptop')).toBeInTheDocument()
    expect(screen.getByText('Browser · Not seen yet · Push off')).toBeInTheDocument()
    expect(screen.getByText('revoked')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Revoke Old laptop' })).not.toBeInTheDocument()
  })

  it('Revoke calls DELETE only after the confirm, then reads the list again', async () => {
    const calls = stubApi({ [DEVICES]: ok({ devices: [PHONE] }), [REVOKE]: ok({ ...PHONE, status: 'revoked' }) })
    await renderTab('local')
    const revoke = await screen.findByRole('button', { name: `Revoke ${PHONE_NAME}` })

    fireEvent.click(revoke)
    const dialog = await screen.findByRole('dialog')
    expect(dialog).toHaveTextContent(PHONE_NAME)
    expect(dialog).toHaveTextContent('pairs again with a new code')
    fireEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }))
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
    expect(calls).not.toContain(REVOKE)

    fireEvent.click(screen.getByRole('button', { name: `Revoke ${PHONE_NAME}` }))
    fireEvent.click(within(await screen.findByRole('dialog')).getByRole('button', { name: 'Revoke' }))
    await waitFor(() => expect(calls).toContain(REVOKE))
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
    await waitFor(() => expect(calls.filter((route) => route === DEVICES)).toHaveLength(2))
  })

  it('a refused revoke keeps the dialog open and says so', async () => {
    stubApi({ [DEVICES]: ok({ devices: [PHONE] }), [REVOKE]: { status: 503, body: { detail: 'down' } } })
    await renderTab('saas')

    fireEvent.click(await screen.findByRole('button', { name: `Revoke ${PHONE_NAME}` }))
    const dialog = await screen.findByRole('dialog')
    fireEvent.click(within(dialog).getByRole('button', { name: 'Revoke' }))
    expect(await within(dialog).findByRole('alert')).toHaveTextContent('Revoking didn’t go through')
  })
})

describe('Settings → Phones & devices: pairing only in local paired mode (PRD-266 US-005c)', () => {
  it('Pair a phone shows the code, the ten-minute countdown and the pairing link; Pair a browser names the web address', async () => {
    const calls = stubApi({ [DEVICES]: ok({ devices: [] }), [DISCOVERY]: discovery(true), [MINT]: mintAnswer(TEN_MINUTES_MS) })
    await renderTab('local')

    fireEvent.click(await screen.findByRole('button', { name: 'Get a pairing code' }))
    expect(await screen.findByTestId('pairing-code')).toHaveTextContent(CODE)
    expect(screen.getByTestId('pairing-countdown').textContent).toMatch(/^Expires in (10:00|9:5\d)$/)
    expect(screen.getByTestId('pairing-link')).toHaveTextContent(LINK)
    expect(screen.getByRole('button', { name: 'Copy the pairing link' })).toBeInTheDocument()
    expect(calls).toContain(MINT)

    expect(screen.getByRole('heading', { name: 'Pair a browser' })).toBeInTheDocument()
    expect(screen.getByText(WEB_URL)).toBeInTheDocument()
    expect(screen.getByText('No phones or paired browsers yet.')).toBeInTheDocument()
  })

  it('Copy puts the link on the clipboard, and says to select it where the browser has no clipboard', async () => {
    stubApi({ [DEVICES]: ok({ devices: [] }), [DISCOVERY]: discovery(true), [MINT]: mintAnswer(TEN_MINUTES_MS) })
    const writeText = vi.fn().mockResolvedValue(undefined)
    setClipboard({ writeText })
    await renderTab('local')
    fireEvent.click(await screen.findByRole('button', { name: 'Get a pairing code' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Copy the pairing link' }))
    await waitFor(() => expect(writeText).toHaveBeenCalledWith(LINK))

    setClipboard(undefined)
    fireEvent.click(screen.getByRole('button', { name: 'Copy the pairing link' }))
    expect(await screen.findByRole('status')).toHaveTextContent('select the link and copy it')
  })

  it('an expired code is not shown: it says to get a new one', async () => {
    stubApi({ [DEVICES]: ok({ devices: [] }), [DISCOVERY]: discovery(true), [MINT]: mintAnswer(-1000) })
    await renderTab('local')

    fireEvent.click(await screen.findByRole('button', { name: 'Get a pairing code' }))
    expect(await screen.findByText('This code has expired. Get a new one.')).toBeInTheDocument()
    expect(screen.queryByTestId('pairing-code')).not.toBeInTheDocument()
  })

  it('a local install that is not in paired mode lists its devices without the pairing panel', async () => {
    const calls = stubApi({ [DEVICES]: ok({ devices: [PHONE] }), [DISCOVERY]: discovery(false) })
    const client = await renderTab('local')

    expect(await screen.findByText(PHONE_NAME)).toBeInTheDocument()
    await discoveryRead(client)
    expect(pairPhoneHeading()).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Pair a browser' })).not.toBeInTheDocument()
    expect(calls).not.toContain(MINT)
  })

  it('saas shows the list without the pairing panel, and never reads discovery', async () => {
    const calls = stubApi({ [DEVICES]: ok({ devices: [{ ...PHONE, kind: 'clerk' }] }), [DISCOVERY]: discovery(true) })
    await renderTab('saas')

    expect(await screen.findByText(PHONE_NAME)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: `Revoke ${PHONE_NAME}` })).toBeInTheDocument()
    expect(pairPhoneHeading()).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Pair a browser' })).not.toBeInTheDocument()
    expect(calls).not.toContain(DISCOVERY)
    expect(calls).not.toContain(MINT)
  })
})

describe('formatCountdown (PRD-266 US-005c)', () => {
  it('reads seconds as m:ss and never goes below zero', async () => {
    const { formatCountdown } = await import('../pair-phone-panel')
    expect(formatCountdown(600)).toBe('10:00')
    expect(formatCountdown(65)).toBe('1:05')
    expect(formatCountdown(0)).toBe('0:00')
    expect(formatCountdown(-3)).toBe('0:00')
  })
})
