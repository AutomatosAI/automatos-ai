/**
 * PRD-260 T-009 — the edges of the app under Play, ported from Automatos Play's
 * tests/edges/{NotFound,Asleep}.test.tsx onto the core: the 404
 * (app/not-found.tsx: Play's lost screen under Play at >= 1024 px, Next's plain
 * words everywhere else) and "Automatos is asleep" (Play's frame over a mocked
 * apiClient: GET /health failing shows it in place of the page, answering
 * mounts the page again, Check again asks again, the automatic recheck counts
 * down). Roles, attributes and text only (CSS modules are proxies under
 * css: false).
 */
import React, { useEffect } from 'react'
import { act, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { BackendConnection } from '@/hooks/use-backend-health'

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

/** The page inside the frame, counting its mounts. */
function Page() {
  useEffect(() => {
    mounts.page += 1
  }, [])
  return <p>The page</p>
}

function primeHealth() {
  api.request.mockImplementation((endpoint: string) => {
    if (endpoint !== '/health') return Promise.reject(new Error(`unexpected call to ${endpoint}`))
    return health.up ? Promise.resolve({ status: 'healthy' }) : Promise.reject(new Error('Failed to fetch'))
  })
}

const healthCalls = () => api.request.mock.calls.filter(([endpoint]) => endpoint === '/health').length

function renderFrame() {
  primeHealth()
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const user = userEvent.setup()
  render(
    <QueryClientProvider client={client}>
      <PlayFrame>
        <Page />
      </PlayFrame>
    </QueryClientProvider>,
  )
  return user
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
  it('shows the asleep screen in place of the page while /health fails, Auto asleep', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    renderFrame()
    expect(await screen.findByRole('heading', { level: 1, name: ASLEEP_TITLE })).toBeInTheDocument()
    expect(api.request).toHaveBeenCalledWith('/health')
    expect(screen.queryByText('The page')).not.toBeInTheDocument()
    expect(screen.queryByRole('navigation', { name: 'Main' })).not.toBeInTheDocument()
    expect(screen.getByTestId('auto')).toHaveAttribute('data-state', 'sleeping')
    expect(screen.getByRole('status')).toHaveTextContent(NOT_CONNECTED)
    expect(screen.getByText(asleepDetail('localhost'))).toBeInTheDocument()
    expect(screen.getByText(countdownLine(ASLEEP_RECHECK_MS / SECOND_MS))).toBeInTheDocument()
  })

  it('Check again asks /health again, and the page stays away while it asks', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const user = renderFrame()
    const check = await screen.findByRole('button', { name: CHECK_AGAIN })
    const before = healthCalls()
    const mountsWhileAsleep = mounts.page
    await user.click(check)
    await waitFor(() => expect(healthCalls()).toBe(before + 1))
    await act(async () => undefined)
    expect(screen.getByRole('heading', { name: ASLEEP_TITLE })).toBeInTheDocument()
    // A recheck of a query that never succeeded reads as loading in react-query: not an answer.
    expect(mounts.page).toBe(mountsWhileAsleep)
    expect(screen.queryByText('The page')).not.toBeInTheDocument()
  })

  it('mounts the frame and the page again once /health answers', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const user = renderFrame()
    await screen.findByRole('heading', { name: ASLEEP_TITLE })
    const mountsWhileAsleep = mounts.page
    health.up = true
    await user.click(screen.getByRole('button', { name: CHECK_AGAIN }))
    expect(await screen.findByText('The page')).toBeInTheDocument()
    expect(mounts.page).toBe(mountsWhileAsleep + 1)
    expect(screen.queryByRole('heading', { name: ASLEEP_TITLE })).not.toBeInTheDocument()
    expect(screen.getByRole('navigation', { name: 'Main' })).toBeInTheDocument()
  })

  it('shows the frame and the page, never the asleep screen, while /health answers', async () => {
    health.up = true
    renderFrame()
    expect(screen.getByText('The page')).toBeInTheDocument()
    await waitFor(() => expect(healthCalls()).toBe(1))
    await act(async () => undefined)
    expect(screen.queryByRole('heading', { name: ASLEEP_TITLE })).not.toBeInTheDocument()
    expect(screen.getByText('The page')).toBeInTheDocument()
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
