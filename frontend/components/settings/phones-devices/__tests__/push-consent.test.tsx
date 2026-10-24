/**
 * PRD-266 US-019a (§4.3 "Web", §4.4, D21, K13) — the operator's Expo answer in Settings → Phones
 * & devices.
 *
 * On a local install in paired mode, "Pair a phone" asks once, before its first code, while GET
 * /api/v1/mobile/push-consent says the answer was never given: Yes saves {agreed: true}, Not now
 * {agreed: false}, and the code is minted after the save; once answered it is not asked again. The
 * page's switch reads the same answer and writes PUT /push-consent. SaaS reads neither route and
 * shows neither. The code is shown as the server returns it (XXXX-XXXX) and copied in that form.
 *
 * The real apiClient and hooks run over a stubbed fetch, so every call travels the real path.
 */
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const DISCOVERY = 'GET /.well-known/automatos.json'
const DEVICES = 'GET /api/v1/mobile/devices'
const MINT = 'POST /api/v1/mobile/pairing-codes'
const READ_CONSENT = 'GET /api/v1/mobile/push-consent'
const SAVE_CONSENT = 'PUT /api/v1/mobile/push-consent'
const WEB_URL = 'https://studio.example.com'
const CODE = 'ABCD-EFGH'
const LINK = `${WEB_URL}/go/pair?api=https%3A%2F%2Fapi.example.com&code=${CODE}`
const TEN_MINUTES_MS = 600_000
const ANSWERED_AT = '2026-10-24T09:00:00+00:00'
const QUESTION = 'Send notifications to phones through Expo’s push service?'
const SWITCH_LABEL = 'Send notifications through Expo’s push service'
/** A failed read is tried once more after react-query's first backoff (one second). */
const READ_RETRY_WAIT_MS = 4000

type Answer = { readonly status: number; readonly body: unknown }
type Call = { readonly route: string; readonly body: unknown }

const ok = (body: unknown): Answer => ({ status: 200, body })
const discovery = (enabled: boolean): Answer =>
  ok({ schema: 'automatos-discovery/v1', edition: 'local', web_url: WEB_URL, pairing: { enabled }, push: { expo: false } })
const consent = (agreed: boolean, answered: boolean): Answer =>
  ok({ agreed, answered, answered_at: answered ? ANSWERED_AT : null })
const MINTED = ok({ code: CODE, expires_at: new Date(Date.now() + TEN_MINUTES_MS).toISOString(), qr_payload: LINK })

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

const routes = (calls: readonly Call[]): string[] => calls.map((call) => call.route)
const bodiesOf = (calls: readonly Call[], route: string): unknown[] =>
  calls.filter((call) => call.route === route).map((call) => call.body)

/** The tab in `edition`, fresh from import, under its own React Query client. */
async function renderTab(edition: 'local' | 'saas'): Promise<void> {
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
}

/** A local install in paired mode whose operator gave `answer` (or never answered). */
function pairedInstall(answer: Answer, extra: Readonly<Record<string, Answer>> = {}): Call[] {
  return stubApi({ [DEVICES]: ok({ devices: [] }), [DISCOVERY]: discovery(true), [READ_CONSENT]: answer, [MINT]: MINTED, ...extra })
}

const getCode = async () => fireEvent.click(await screen.findByRole('button', { name: 'Get a pairing code' }))

const originalClipboard = Object.getOwnPropertyDescriptor(window.navigator, 'clipboard')

afterEach(() => {
  cleanup()
  if (originalClipboard) Object.defineProperty(window.navigator, 'clipboard', originalClipboard)
  else delete (window.navigator as { clipboard?: unknown }).clipboard
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('Pair a phone asks the Expo question once (PRD-266 US-019a)', () => {
  it('not answered: asks first, and Yes saves {agreed: true} and then mints; the next code asks nothing', async () => {
    const calls = pairedInstall(consent(false, false), { [SAVE_CONSENT]: consent(true, true) })
    await renderTab('local')

    await getCode()
    const question = await screen.findByRole('group', { name: QUESTION })
    expect(within(question).getByText(/Expo relays each notification/)).toHaveTextContent('unless this workspace redacts questions')
    expect(routes(calls)).not.toContain(MINT)

    fireEvent.click(within(question).getByRole('button', { name: 'Yes' }))
    expect(await screen.findByTestId('pairing-code')).toHaveTextContent(CODE)
    expect(bodiesOf(calls, SAVE_CONSENT)).toEqual([{ agreed: true }])
    expect(routes(calls).indexOf(SAVE_CONSENT)).toBeLessThan(routes(calls).indexOf(MINT))
    expect(screen.queryByText(QUESTION)).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Get a new code' }))
    await waitFor(() => expect(routes(calls).filter((route) => route === MINT)).toHaveLength(2))
    expect(screen.queryByText(QUESTION)).not.toBeInTheDocument()
    expect(bodiesOf(calls, SAVE_CONSENT)).toHaveLength(1)
  })

  it('not answered: Not now saves {agreed: false} and then mints', async () => {
    const calls = pairedInstall(consent(false, false), { [SAVE_CONSENT]: consent(false, true) })
    await renderTab('local')

    await getCode()
    fireEvent.click(await screen.findByRole('button', { name: 'Not now' }))
    expect(await screen.findByTestId('pairing-code')).toHaveTextContent(CODE)
    expect(bodiesOf(calls, SAVE_CONSENT)).toEqual([{ agreed: false }])
    expect(routes(calls).indexOf(SAVE_CONSENT)).toBeLessThan(routes(calls).indexOf(MINT))
  })

  it('an answer that is not saved mints nothing: the question stays and says so', async () => {
    const calls = pairedInstall(consent(false, false), { [SAVE_CONSENT]: { status: 503, body: { detail: 'down' } } })
    await renderTab('local')

    await getCode()
    fireEvent.click(await screen.findByRole('button', { name: 'Yes' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Couldn’t save your answer')
    expect(screen.getByText(QUESTION)).toBeInTheDocument()
    expect(routes(calls)).not.toContain(MINT)
  })

  it('answered (yes or no): no question, the code is minted at once', async () => {
    const calls = pairedInstall(consent(false, true))
    await renderTab('local')

    await getCode()
    expect(await screen.findByTestId('pairing-code')).toHaveTextContent(CODE)
    expect(screen.queryByText(QUESTION)).not.toBeInTheDocument()
    expect(routes(calls)).not.toContain(SAVE_CONSENT)
  })

  it('a caller the route refuses (403) is not asked and sees no switch: the code is minted at once', async () => {
    const calls = pairedInstall({ status: 403, body: { detail: 'Forbidden' } })
    await renderTab('local')

    await getCode()
    expect(await screen.findByTestId('pairing-code')).toHaveTextContent(CODE)
    expect(screen.queryByText(QUESTION)).not.toBeInTheDocument()
    expect(screen.queryByRole('switch')).not.toBeInTheDocument()
    expect(routes(calls)).not.toContain(SAVE_CONSENT)
  })

  it('an answer that cannot be read is never taken for "not answered": no question, and the switch says so', async () => {
    const calls = pairedInstall({ status: 503, body: { detail: 'down' } })
    await renderTab('local')

    expect(await screen.findByText('Couldn’t read this install’s answer right now.', undefined, { timeout: READ_RETRY_WAIT_MS }))
      .toBeInTheDocument()
    await getCode()
    expect(await screen.findByTestId('pairing-code')).toHaveTextContent(CODE)
    expect(screen.queryByText(QUESTION)).not.toBeInTheDocument()
    expect(screen.getByRole('switch', { name: SWITCH_LABEL })).toBeDisabled()
    expect(routes(calls)).not.toContain(SAVE_CONSENT)
  })

  it('an answer given on the switch while the question is open ends the question', async () => {
    const calls = pairedInstall(consent(false, false), { [SAVE_CONSENT]: consent(true, true) })
    await renderTab('local')

    await getCode()
    expect(await screen.findByRole('group', { name: QUESTION })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('switch', { name: SWITCH_LABEL }))
    await waitFor(() => expect(screen.queryByText(QUESTION)).not.toBeInTheDocument())
    expect(bodiesOf(calls, SAVE_CONSENT)).toEqual([{ agreed: true }])
    expect(routes(calls)).not.toContain(MINT)
  })
})

describe('the Expo switch on Phones & devices (PRD-266 US-019a)', () => {
  it('reads the answer and writes a change, in paired mode or not', async () => {
    const calls = stubApi({
      [DEVICES]: ok({ devices: [] }),
      [DISCOVERY]: discovery(false),
      [READ_CONSENT]: consent(true, true),
      [SAVE_CONSENT]: consent(false, true),
    })
    await renderTab('local')

    const toggle = await screen.findByRole('switch', { name: SWITCH_LABEL })
    await waitFor(() => expect(toggle).toHaveAttribute('aria-checked', 'true'))
    await waitFor(() => expect(toggle).toBeEnabled())

    fireEvent.click(toggle)
    await waitFor(() => expect(bodiesOf(calls, SAVE_CONSENT)).toEqual([{ agreed: false }]))
    await waitFor(() => expect(toggle).toHaveAttribute('aria-checked', 'false'))
    expect(routes(calls)).not.toContain(MINT)
  })

  it('a refused change says so and keeps the stored answer', async () => {
    stubApi({
      [DEVICES]: ok({ devices: [] }),
      [DISCOVERY]: discovery(false),
      [READ_CONSENT]: consent(false, true),
      [SAVE_CONSENT]: { status: 503, body: { detail: 'down' } },
    })
    await renderTab('local')

    const toggle = await screen.findByRole('switch', { name: SWITCH_LABEL })
    await waitFor(() => expect(toggle).toBeEnabled())
    fireEvent.click(toggle)
    expect(await screen.findByRole('alert')).toHaveTextContent('Couldn’t save your answer')
    expect(toggle).toHaveAttribute('aria-checked', 'false')
  })
})

describe('SaaS has neither (PRD-266 US-019a)', () => {
  it('never reads the answer, never asks, shows no switch', async () => {
    const calls = stubApi({ [DEVICES]: ok({ devices: [] }), [DISCOVERY]: discovery(true), [READ_CONSENT]: consent(false, false) })
    await renderTab('saas')

    expect(await screen.findByText('No phones or paired browsers yet.')).toBeInTheDocument()
    expect(screen.queryByRole('switch')).not.toBeInTheDocument()
    expect(screen.queryByText(QUESTION)).not.toBeInTheDocument()
    expect(routes(calls)).not.toContain(READ_CONSENT)
    expect(routes(calls)).not.toContain(SAVE_CONSENT)
  })
})

describe('the pairing code as the server returns it (PRD-266 US-019a)', () => {
  it('renders as XXXX-XXXX with its hyphen, and Copy copies that form', async () => {
    pairedInstall(consent(true, true))
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(window.navigator, 'clipboard', { configurable: true, value: { writeText } })
    await renderTab('local')

    await getCode()
    const code = await screen.findByTestId('pairing-code')
    expect(code.textContent).toBe(CODE)
    expect(code.textContent).toMatch(/^[A-Z2-9]{4}-[A-Z2-9]{4}$/)

    fireEvent.click(screen.getByRole('button', { name: 'Copy the code' }))
    await waitFor(() => expect(writeText).toHaveBeenCalledWith(CODE))
  })
})
