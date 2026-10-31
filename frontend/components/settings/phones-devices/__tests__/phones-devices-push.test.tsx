/**
 * PRD-266 US-019b (§4.3 "Web", K15) — each phone's push state in Settings → Phones & devices.
 *
 * A row whose push is off for a recorded reason says it in words (DeviceNotRegistered, token_moved,
 * device_revoked). The caller's own active phone (no owner_label) has a push switch that sends PATCH
 * /api/v1/mobile/devices/{id} {push_enabled}; a phone that never registered an Expo token answers
 * 409 push_token_missing and the row says to open Automatos Go on it. Another person's row shows
 * whose it is (owner_label) and its state, with no switch; so does a paired browser or a revoked phone.
 *
 * The real apiClient and hooks run over a stubbed fetch, so every call travels the real path.
 */
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const MINE_ID = '1b9d6bcd-bbfd-4b2d-9b5d-ab8dfbbd4bed'
const THEIRS_ID = '7c4a8d09-ca37-4b2e-8f9a-2d1e0b3c4a5f'
const DEVICES = 'GET /api/v1/mobile/devices'
const PATCH_MINE = `PATCH /api/v1/mobile/devices/${MINE_ID}`
const MY_PHONE = 'iPhone 16'
const THEIR_PHONE = 'Pixel 9'
const OWNER = 'Ana Lopez'
const NOT_REGISTERED_WORDS = 'The phone no longer has the app (Expo reported it)'
const MOVED_WORDS = 'Moved to this phone’s newer sign-in'
const REVOKED_WORDS = 'Revoked'
const TOKEN_MISSING_LINE = 'Open Automatos Go on this phone to turn notifications on'

type Answer = { readonly status: number; readonly body: unknown }
type Call = { readonly route: string; readonly body: unknown }

const ok = (body: unknown): Answer => ({ status: 200, body })
const MINE = {
  id: MINE_ID,
  user_id: 1,
  kind: 'clerk',
  name: MY_PHONE,
  platform: 'ios',
  status: 'active',
  push_enabled: true,
  push_disabled_reason: null,
  last_seen_at: null,
}
const THEIRS = { ...MINE, id: THEIRS_ID, user_id: 2, name: THEIR_PHONE, platform: 'android', owner_label: OWNER }

/** The API over a stubbed fetch: each "METHOD path" answers as scripted (200 {} otherwise); every call is kept with its body. */
function stubApi(answers: Readonly<Record<string, Answer>>): Call[] {
  const calls: Call[] = []
  vi.stubGlobal('fetch', vi.fn(async (url: string, init: RequestInit = {}) => {
    const route = `${init.method ?? 'GET'} ${new URL(String(url), 'http://app.test').pathname}`
    calls.push({ route, body: typeof init.body === 'string' ? JSON.parse(init.body) : undefined })
    const answer = answers[route] ?? ok({})
    return new Response(JSON.stringify(answer.body), { status: answer.status, headers: { 'Content-Type': 'application/json' } })
  }))
  return calls
}

const listing = (...devices: readonly object[]): Record<string, Answer> => ({ [DEVICES]: ok({ devices }) })
const bodiesOf = (calls: readonly Call[], route: string): unknown[] =>
  calls.filter((call) => call.route === route).map((call) => call.body)

/** The tab in SaaS (no discovery, no pairing), fresh from import, under its own React Query client. */
async function renderTab(): Promise<void> {
  vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', 'saas')
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
}

const switchFor = (name: string) => screen.queryByRole('switch', { name: `Notifications on ${name}` })

afterEach(() => {
  cleanup()
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('Settings → Phones & devices: why push is off (PRD-266 US-019b)', () => {
  it('says each recorded reason in words', async () => {
    stubApi(listing(
      { ...THEIRS, push_enabled: false, push_disabled_reason: 'DeviceNotRegistered' },
      { ...THEIRS, id: 'a1', name: 'Old phone', push_enabled: false, push_disabled_reason: 'token_moved' },
      { ...THEIRS, id: 'a2', name: 'Lost phone', status: 'revoked', push_enabled: false, push_disabled_reason: 'device_revoked' },
    ))
    await renderTab()

    expect(await screen.findByText(NOT_REGISTERED_WORDS)).toBeInTheDocument()
    expect(screen.getByText(MOVED_WORDS)).toBeInTheDocument()
    expect(screen.getByText(REVOKED_WORDS)).toBeInTheDocument()
    expect(screen.getAllByText(/· Push off$/)).toHaveLength(3)
  })

  it('says no reason while push is on, or for a reason it does not know', async () => {
    stubApi(listing(
      { ...THEIRS, push_enabled: true, push_disabled_reason: 'DeviceNotRegistered' },
      { ...THEIRS, id: 'a1', name: 'Old phone', push_enabled: false, push_disabled_reason: 'something_new' },
    ))
    await renderTab()

    expect(await screen.findByText(THEIR_PHONE)).toBeInTheDocument()
    expect(screen.queryByText(NOT_REGISTERED_WORDS)).not.toBeInTheDocument()
    expect(screen.queryByText('something_new')).not.toBeInTheDocument()
  })
})

describe('Settings → Phones & devices: the push switch on your own phone (PRD-266 US-019b)', () => {
  it('turning your own phone off sends PATCH {push_enabled: false}', async () => {
    const calls = stubApi({ ...listing(MINE), [PATCH_MINE]: ok({ ...MINE, push_enabled: false }) })
    await renderTab()

    const own = await screen.findByRole('switch', { name: `Notifications on ${MY_PHONE}` })
    expect(own).toHaveAttribute('aria-checked', 'true')
    fireEvent.click(own)

    await waitFor(() => expect(bodiesOf(calls, PATCH_MINE)).toEqual([{ push_enabled: false }]))
  })

  it('a phone with no Expo token answers push_token_missing, and the row says to open the app', async () => {
    const calls = stubApi({
      ...listing({ ...MINE, push_enabled: false }),
      [PATCH_MINE]: { status: 409, body: { detail: { code: 'push_token_missing' } } },
    })
    await renderTab()

    fireEvent.click(await screen.findByRole('switch', { name: `Notifications on ${MY_PHONE}` }))

    expect(await screen.findByRole('alert')).toHaveTextContent(TOKEN_MISSING_LINE)
    expect(bodiesOf(calls, PATCH_MINE)).toEqual([{ push_enabled: true }])
  })

  it('any other failure says the switch did not change', async () => {
    stubApi({ ...listing(MINE), [PATCH_MINE]: { status: 503, body: { detail: 'Pairing is unavailable right now.' } } })
    await renderTab()

    fireEvent.click(await screen.findByRole('switch', { name: `Notifications on ${MY_PHONE}` }))

    expect(await screen.findByRole('alert')).toHaveTextContent('The switch didn’t change')
    expect(screen.queryByText(TOKEN_MISSING_LINE)).not.toBeInTheDocument()
  })

  it('a paired browser and a revoked phone of your own get no switch', async () => {
    stubApi(listing(
      { ...MINE, name: 'Work laptop', platform: 'web' },
      { ...MINE, id: 'a3', name: 'Old iPhone', status: 'revoked', push_enabled: false },
    ))
    await renderTab()

    expect(await screen.findByText('Work laptop')).toBeInTheDocument()
    expect(screen.getByText('Old iPhone')).toBeInTheDocument()
    expect(screen.queryAllByRole('switch')).toHaveLength(0)
  })
})

describe('Settings → Phones & devices: whose phone it is (PRD-266 US-019b)', () => {
  it('another person’s phone shows its owner and its state, with no switch; yours has a switch and no owner', async () => {
    const calls = stubApi(listing(MINE, THEIRS))
    await renderTab()

    expect(await screen.findByText(THEIR_PHONE)).toBeInTheDocument()
    expect(screen.getByText(OWNER)).toBeInTheDocument()
    expect(screen.getByText('Android · Not seen yet · Push on')).toBeInTheDocument()
    expect(switchFor(THEIR_PHONE)).not.toBeInTheDocument()
    expect(switchFor(MY_PHONE)).toBeInTheDocument()
    expect(screen.getAllByText(OWNER)).toHaveLength(1)
    expect(calls.map((call) => call.route).filter((route) => route.startsWith('PATCH'))).toEqual([])
  })
})
