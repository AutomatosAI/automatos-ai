/**
 * PRD-260 T-009 — the edges of the app under Play, ported from Automatos Play's
 * tests/edges/{NotFound,Asleep}.test.tsx onto the core: the 404
 * (app/not-found.tsx: Play's lost screen under Play at >= 1024 px, Next's plain
 * words everywhere else) and "Automatos is asleep" (Play's frame over a mocked
 * apiClient and fake timers: GET /health failing on every retry shows it in
 * place of the page, one failed ask whose retry answers never does
 * (P260-RVW-1), answering mounts the page again, Check again asks again, the
 * automatic recheck counts down). Roles, attributes and text only (CSS modules
 * are proxies under css: false).
 */
import React, { useEffect, useState } from 'react'
import { act, fireEvent, render, renderHook, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  type BackendConnection,
  BACKEND_HEALTH_POLL_MS,
  BACKEND_HEALTH_RETRIES,
  BACKEND_HEALTH_RETRY_DELAY_MS,
} from '@/hooks/use-backend-health'

const look = vi.hoisted(() => ({ play: true, narrow: false }))
vi.mock('@/hooks/use-studio-theme', () => ({ useIsStudio: () => true, useIsPlay: () => look.play }))
vi.mock('@/hooks/use-mobile', () => ({ useIsTabletOrBelow: () => look.narrow, useIsMobile: () => look.narrow }))

const api = vi.hoisted(() => ({ request: vi.fn(), getBaseUrl: () => 'http://localhost:8000' }))
vi.mock('@/lib/api-client', () => ({ apiClient: api }))

// The frame's own parts have their tests (play-frame.test.tsx, play-sidebar.test.tsx).
vi.mock('@/components/play/shell/play-sidebar', () => ({ PlaySidebar: () => <nav aria-label="Main" /> }))
vi.mock('@/components/layout/studio-header', () => ({ StudioHeaderUtilities: () => null }))

vi.mock('@/components/mascot/auto-mascot', () => ({
  AutoMascot: ({ state }: { state: string }) => <div data-testid="auto" data-state={state} />,
}))

import NotFound from '@/app/not-found'
import { PlayFrame } from '@/components/play/shell/play-frame'
import { useBackendMascot } from '@/hooks/use-backend-mascot'
import { SECOND_MS } from '@/hooks/use-recheck-countdown'
import { ASLEEP_RECHECK_MS } from '../asleep-gate'
import {
  ASLEEP_TITLE,
  AsleepScreen,
  asleepDetail,
  CHECK_AGAIN,
  CHECKING,
  countdownLine,
  HOSTED_LINE,
  NOT_CONNECTED,
} from '../asleep-screen'
import { BACK_TO_CHAT, LOST_LINE, LOST_TITLE } from '../lost-screen'
import { COPIED_LABEL, COPY_LABEL, START_COMMAND } from '../start-command'

/** Whether GET /health answers, and how often the page has mounted. */
const health = { up: false }
const mounts = { page: 0 }

const DRAFT_LABEL = 'Draft'
const TYPED = 'Half a message, not sent yet'

/** The page inside the frame, counting its mounts; its draft lives only while it stays mounted. */
function Page() {
  const [draft, setDraft] = useState('')
  useEffect(() => {
    mounts.page += 1
  }, [])
  return (
    <>
      <p>The page</p>
      <input aria-label={DRAFT_LABEL} value={draft} onChange={(event) => setDraft(event.target.value)} />
    </>
  )
}

function primeHealth() {
  api.request.mockImplementation((endpoint: string) => {
    if (endpoint !== '/health') return Promise.reject(new Error(`unexpected call to ${endpoint}`))
    return health.up ? Promise.resolve({ status: 'healthy' }) : Promise.reject(new Error('Failed to fetch'))
  })
}

const healthCalls = () => api.request.mock.calls.filter(([endpoint]) => endpoint === '/health').length

/** Play's frame around the page; the hook's own retry options win over the client's default. */
function renderFrame() {
  primeHealth()
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <PlayFrame>
        <Page />
      </PlayFrame>
    </QueryClientProvider>,
  )
}

function renderAsleep(props: { onCheck?: () => void; recheckSeconds?: number; local?: boolean } = {}) {
  const { onCheck = vi.fn(), recheckSeconds = 3, local = false } = props
  render(
    <AsleepScreen
      mascot="sleeping"
      detail={asleepDetail('localhost')}
      onCheck={onCheck}
      recheckSeconds={recheckSeconds}
      local={local}
    />,
  )
  return onCheck
}

const tick = (ms: number) =>
  act(() => {
    vi.advanceTimersByTime(ms)
  })

/** Let `ms` of fake time pass, with the promises and React updates it starts. */
const elapse = (ms: number) =>
  act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })

/** Let every retry of the ask in flight run: each fails or answers as `health.up` says. */
async function throughRetries() {
  for (let retry = 0; retry < BACKEND_HEALTH_RETRIES; retry += 1) await elapse(BACKEND_HEALTH_RETRY_DELAY_MS)
}

const asleepHeading = () => screen.queryByRole('heading', { level: 1, name: ASLEEP_TITLE })

/** Play's frame with /health failing on the first ask and every retry: the asleep screen. */
async function renderAsleepFrame() {
  renderFrame()
  await elapse(0)
  await throughRetries()
  expect(asleepHeading()).toBeInTheDocument()
}

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
  api.request.mockReset()
  Object.assign(look, { play: true, narrow: false })
  health.up = false
  mounts.page = 0
})

describe('the 404 (app/not-found.tsx)', () => {
  it("under Play at >= 1024 px: Auto lost, 'This page wandered off' and the way back to /chat", () => {
    render(<NotFound />)
    expect(screen.getByRole('heading', { level: 1, name: LOST_TITLE })).toBeInTheDocument()
    expect(LOST_TITLE).toBe('This page wandered off')
    expect(screen.getByText(LOST_LINE)).toBeInTheDocument()
    expect(screen.getByTestId('auto')).toHaveAttribute('data-state', 'lost')
    expect(screen.getByRole('link', { name: BACK_TO_CHAT })).toHaveAttribute('href', '/chat')
    expect(screen.queryByText('This page could not be found.')).not.toBeInTheDocument()
  })

  it("under Classic or Studio: the plain 404 in Next's own words, a link home, no Play design", () => {
    look.play = false
    render(<NotFound />)
    expect(screen.getByRole('heading', { level: 1, name: '404' })).toBeInTheDocument()
    expect(screen.getByText('This page could not be found.')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Go home' })).toHaveAttribute('href', '/')
    expect(screen.queryByRole('heading', { name: LOST_TITLE })).not.toBeInTheDocument()
    expect(screen.queryByTestId('auto')).not.toBeInTheDocument()
  })

  it('under Play below 1024 px: the plain 404 too (D7)', () => {
    look.narrow = true
    render(<NotFound />)
    expect(screen.getByText('This page could not be found.')).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: LOST_TITLE })).not.toBeInTheDocument()
  })
})

describe("Play's frame while Automatos is asleep (GET /health)", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
  })

  it('shows the asleep screen in place of the page only once /health has failed on every retry, Auto asleep', async () => {
    renderFrame()
    await elapse(0)
    for (let retry = 1; retry <= BACKEND_HEALTH_RETRIES; retry += 1) {
      // A failed ask is not the answer yet: the page stays while /health is asked again.
      expect(asleepHeading()).not.toBeInTheDocument()
      expect(screen.getByText('The page')).toBeInTheDocument()
      await elapse(BACKEND_HEALTH_RETRY_DELAY_MS)
    }
    expect(healthCalls()).toBe(1 + BACKEND_HEALTH_RETRIES)
    expect(api.request).toHaveBeenCalledWith('/health')
    expect(asleepHeading()).toBeInTheDocument()
    expect(screen.queryByText('The page')).not.toBeInTheDocument()
    expect(screen.queryByRole('navigation', { name: 'Main' })).not.toBeInTheDocument()
    expect(screen.getByTestId('auto')).toHaveAttribute('data-state', 'sleeping')
    expect(screen.getByRole('status')).toHaveTextContent(NOT_CONNECTED)
    expect(screen.getByText(asleepDetail('localhost'))).toBeInTheDocument()
    expect(screen.getByText(countdownLine(ASLEEP_RECHECK_MS / SECOND_MS))).toBeInTheDocument()
  })

  it('one failed poll whose retry answers never shows the asleep screen, and the page keeps what was typed', async () => {
    health.up = true
    renderFrame()
    await elapse(0)
    fireEvent.change(screen.getByRole('textbox', { name: DRAFT_LABEL }), { target: { value: TYPED } })
    health.up = false
    await elapse(BACKEND_HEALTH_POLL_MS)
    expect(healthCalls()).toBe(2)
    expect(asleepHeading()).not.toBeInTheDocument()
    health.up = true
    await elapse(BACKEND_HEALTH_RETRY_DELAY_MS)
    expect(healthCalls()).toBe(3)
    expect(asleepHeading()).not.toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: DRAFT_LABEL })).toHaveValue(TYPED)
    expect(mounts.page).toBe(1)
    expect(screen.getByRole('navigation', { name: 'Main' })).toBeInTheDocument()
  })

  it('after an answer, a poll that fails on every retry shows the asleep screen, and only then', async () => {
    health.up = true
    renderFrame()
    await elapse(0)
    health.up = false
    await elapse(BACKEND_HEALTH_POLL_MS)
    for (let retry = 1; retry <= BACKEND_HEALTH_RETRIES; retry += 1) {
      expect(asleepHeading()).not.toBeInTheDocument()
      expect(screen.getByText('The page')).toBeInTheDocument()
      await elapse(BACKEND_HEALTH_RETRY_DELAY_MS)
    }
    expect(healthCalls()).toBe(2 + BACKEND_HEALTH_RETRIES)
    expect(asleepHeading()).toBeInTheDocument()
    expect(screen.queryByText('The page')).not.toBeInTheDocument()
    expect(screen.getByTestId('auto')).toHaveAttribute('data-state', 'sleeping')
  })

  it('Check again asks /health again, and the page stays away while it asks and retries', async () => {
    await renderAsleepFrame()
    const before = healthCalls()
    const mountsWhileAsleep = mounts.page
    fireEvent.click(screen.getByRole('button', { name: CHECK_AGAIN }))
    await elapse(0)
    expect(healthCalls()).toBe(before + 1)
    expect(asleepHeading()).toBeInTheDocument()
    await throughRetries()
    expect(healthCalls()).toBe(before + 1 + BACKEND_HEALTH_RETRIES)
    // A recheck of a query that never succeeded reads as loading in react-query: not an answer.
    expect(asleepHeading()).toBeInTheDocument()
    expect(mounts.page).toBe(mountsWhileAsleep)
    expect(screen.queryByText('The page')).not.toBeInTheDocument()
  })

  it('mounts the frame and the page again once /health answers', async () => {
    await renderAsleepFrame()
    const mountsWhileAsleep = mounts.page
    health.up = true
    fireEvent.click(screen.getByRole('button', { name: CHECK_AGAIN }))
    await elapse(0)
    expect(screen.getByText('The page')).toBeInTheDocument()
    expect(mounts.page).toBe(mountsWhileAsleep + 1)
    expect(asleepHeading()).not.toBeInTheDocument()
    expect(screen.getByRole('navigation', { name: 'Main' })).toBeInTheDocument()
  })

  it('the automatic recheck mounts the page again once /health answers', async () => {
    await renderAsleepFrame()
    health.up = true
    await elapse(ASLEEP_RECHECK_MS)
    // The countdown asks as React renders its last second; the answer reaches the gate on the next turn.
    await elapse(0)
    expect(healthCalls()).toBe(2 + BACKEND_HEALTH_RETRIES)
    expect(screen.getByText('The page')).toBeInTheDocument()
    expect(asleepHeading()).not.toBeInTheDocument()
  })

  it("a recheck's retries finish before the next automatic recheck asks", () => {
    expect(BACKEND_HEALTH_RETRIES * BACKEND_HEALTH_RETRY_DELAY_MS).toBeLessThan(ASLEEP_RECHECK_MS)
  })

  it('shows the frame and the page, never the asleep screen, while /health answers', async () => {
    health.up = true
    renderFrame()
    expect(screen.getByText('The page')).toBeInTheDocument()
    await elapse(0)
    expect(healthCalls()).toBe(1)
    await elapse(BACKEND_HEALTH_POLL_MS)
    expect(healthCalls()).toBe(2)
    expect(asleepHeading()).not.toBeInTheDocument()
    expect(screen.getByText('The page')).toBeInTheDocument()
    expect(mounts.page).toBe(1)
  })
})

describe('Auto on the asleep screen (useBackendMascot)', () => {
  it('takes backend-down to sleep and backend-up to wake', () => {
    const { result, rerender } = renderHook(({ connection }) => useBackendMascot(connection), {
      initialProps: { connection: 'checking' as BackendConnection },
    })
    expect(result.current).toMatchObject({ mascot: 'idle', asleep: false })
    rerender({ connection: 'down' })
    expect(result.current).toMatchObject({ mascot: 'sleeping', asleep: true })
    rerender({ connection: 'connected' })
    expect(result.current).toMatchObject({ mascot: 'idle', asleep: false })
  })
})

describe('the asleep screen', () => {
  it('counts down to the automatic recheck, asks, and starts the count over', () => {
    vi.useFakeTimers()
    const onCheck = renderAsleep({ recheckSeconds: 3 })
    expect(screen.getByText('Checking again in 3 seconds.')).toBeInTheDocument()
    tick(SECOND_MS)
    expect(screen.getByText('Checking again in 2 seconds.')).toBeInTheDocument()
    tick(SECOND_MS)
    expect(screen.getByText('Checking again in 1 second.')).toBeInTheDocument()
    expect(onCheck).not.toHaveBeenCalled()
    tick(SECOND_MS)
    expect(onCheck).toHaveBeenCalledTimes(1)
    expect(screen.getByText('Checking again in 3 seconds.')).toBeInTheDocument()
  })

  it('Check again asks at once and starts the count over', () => {
    vi.useFakeTimers()
    const onCheck = renderAsleep({ recheckSeconds: 3 })
    tick(2 * SECOND_MS)
    fireEvent.click(screen.getByRole('button', { name: CHECK_AGAIN }))
    expect(onCheck).toHaveBeenCalledTimes(1)
    expect(screen.getByText('Checking again in 3 seconds.')).toBeInTheDocument()
    tick(2 * SECOND_MS)
    expect(onCheck).toHaveBeenCalledTimes(1)
  })

  it('in the local edition names the command that starts Automatos, with Copy', async () => {
    const user = userEvent.setup()
    renderAsleep({ local: true })
    expect(screen.getByText(START_COMMAND)).toBeInTheDocument()
    expect(START_COMMAND).toBe('docker compose up')
    expect(screen.getByText('automatos-ai')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: COPY_LABEL }))
    expect(await screen.findByRole('button', { name: COPIED_LABEL })).toBeInTheDocument()
    expect(await navigator.clipboard.readText()).toBe(START_COMMAND)
  })

  it('in the hosted edition offers no command, only that it will come back', () => {
    renderAsleep({ local: false })
    expect(screen.getByText(HOSTED_LINE)).toBeInTheDocument()
    expect(screen.queryByText(START_COMMAND)).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: COPY_LABEL })).not.toBeInTheDocument()
  })

  it('says the countdown and the detail in words', () => {
    expect(countdownLine(10)).toBe('Checking again in 10 seconds.')
    expect(countdownLine(1)).toBe('Checking again in 1 second.')
    expect(countdownLine(0)).toBe(CHECKING)
    expect(asleepDetail('localhost')).toBe("Automatos didn't answer at localhost.")
    expect(asleepDetail(undefined)).toBe("Automatos didn't answer.")
  })
})
