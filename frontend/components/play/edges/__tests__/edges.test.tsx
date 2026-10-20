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
 *
 * P260-FX-21: a frozen /health times out into the asleep screen (M43); a frame
 * mounted while it is down opens on the asleep screen, its page never mounted
 * (M89); a check in flight says Checking… (M94); offline says so (M83); the
 * local start command is `make up` with its hint (M84); the 404 keeps Next's
 * title (M68).
 *
 * P260-FX-22 (M21): the swap moves keyboard focus (the asleep heading, then the
 * frame's <main> once it answers) and the gate's own live region, outside the
 * swap, says each change.
 *
 * P260-FX-25 (M69): the 404 reads Play's width from useThemeLayout (stood in for
 * here); under Play while it isn't known (the server, hydration) it renders
 * nothing, so neither the lost screen nor the plain page shows first.
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
  BACKEND_HEALTH_TIMEOUT_MS,
  connectionOf,
} from '@/hooks/use-backend-health'

const look = vi.hoisted(() => ({ play: true, narrow: false, widthUnknown: false }))
vi.mock('@/hooks/use-studio-theme', async () => {
  const { BUILTIN_PLAY_ENTRY } = await import('@/lib/theme-host/builtin/play-entry')
  return {
    useIsStudio: () => true,
    useActiveTheme: () => (look.play ? BUILTIN_PLAY_ENTRY : null),
    useThemeLayout: () => {
      if (!look.play || look.narrow) return 'fallback'
      return look.widthUnknown ? 'pending' : 'theme'
    },
  }
})

const api = vi.hoisted(() => ({ request: vi.fn(), getBaseUrl: () => 'http://localhost:8000' }))
vi.mock('@/lib/api-client', () => ({ apiClient: api }))

// The frame's own parts have their tests (play-frame.test.tsx, play-sidebar.test.tsx).
vi.mock('@/components/play/shell/play-sidebar', () => ({ PlaySidebar: () => <nav aria-label="Main" /> }))
vi.mock('@/components/layout/studio-header', () => ({ StudioHeaderUtilities: () => null }))

vi.mock('@/components/mascot/auto-mascot', () => ({
  AutoMascot: ({ state }: { state: string }) => <div data-testid="auto" data-state={state} />,
}))

import NotFound, { metadata, NOT_FOUND_TITLE } from '@/app/not-found'
import { PlayFrame } from '@/components/play/shell/play-frame'
import { useBackendMascot } from '@/hooks/use-backend-mascot'
import { SECOND_MS } from '@/hooks/use-recheck-countdown'
import { ASLEEP_RECHECK_MS, AWAKE_LINE } from '../asleep-gate'
import {
  ASLEEP_TITLE,
  AsleepScreen,
  asleepDetail,
  CHECK_AGAIN,
  CHECKING,
  countdownLine,
  HOSTED_LINE,
  NOT_CONNECTED,
  OFFLINE_LINE,
} from '../asleep-screen'
import { BACK_TO_CHAT, LOST_LINE, LOST_TITLE } from '../lost-screen'
import { COPIED_LABEL, COPY_LABEL, START_COMMAND, START_HINT } from '../start-command'

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

/** A frozen backend: /health never answers, and an ask fails only when its signal gives up on it. */
function freezeHealth() {
  api.request.mockImplementation(
    (_endpoint: string, options?: RequestInit) =>
      new Promise((_answer, fail) => {
        const signal = options?.signal
        if (!signal) return
        signal.addEventListener('abort', () => fail(signal.reason))
      }),
  )
}

const healthCalls = () => api.request.mock.calls.filter(([endpoint]) => endpoint === '/health').length

const newClient = () => new QueryClient({ defaultOptions: { queries: { retry: false } } })

/**
 * Play's frame around the page; the hook's own retry options win over the client's default.
 * Pass the client of an earlier frame to mount a new one over its cache, as browser Back does.
 */
function renderFrame({ prime = primeHealth, client = newClient() } = {}) {
  prime()
  const { unmount } = render(
    <QueryClientProvider client={client}>
      <PlayFrame>
        <Page />
      </PlayFrame>
    </QueryClientProvider>,
  )
  return { client, unmount }
}

interface AsleepOptions {
  readonly onCheck?: () => void
  readonly recheckSeconds?: number
  readonly local?: boolean
  readonly checking?: boolean
}

function asleepScreen({ onCheck = vi.fn(), recheckSeconds = 3, local = false, checking = false }: AsleepOptions = {}) {
  return (
    <AsleepScreen
      mascot="sleeping"
      detail={asleepDetail('localhost')}
      onCheck={onCheck}
      checking={checking}
      recheckSeconds={recheckSeconds}
      local={local}
    />
  )
}

function renderAsleep(props: AsleepOptions = {}) {
  const onCheck = props.onCheck ?? vi.fn()
  render(asleepScreen({ ...props, onCheck }))
  return onCheck
}

/** The browser says it is offline (navigator.onLine) until the spy is restored. */
const goOffline = () => vi.spyOn(window.navigator, 'onLine', 'get').mockReturnValue(false)

/** Tell the page the connection changed, as the browser does. */
const connectivityEvent = (name: 'online' | 'offline') =>
  act(() => {
    window.dispatchEvent(new Event(name))
  })

const tick = (ms: number) =>
  act(() => {
    vi.advanceTimersByTime(ms)
  })

/** Let `ms` of fake time pass, with the promises and React updates it starts. */
const elapse = (ms: number) =>
  act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })

/**
 * Let react-query hand a settled ask to React. Its notify is a setTimeout(0), and a timer set
 * during a fake tick fires 1 ms later, so the last retry's answer lands one fake ms after it.
 */
const settle = () => elapse(1)

/** Let every retry of the ask in flight run, each failing or answering as `health.up` says, and settle. */
async function throughRetries() {
  for (let retry = 0; retry < BACKEND_HEALTH_RETRIES; retry += 1) await elapse(BACKEND_HEALTH_RETRY_DELAY_MS)
  await settle()
}

const asleepHeading = () => screen.queryByRole('heading', { level: 1, name: ASLEEP_TITLE })

/** Every live status on the page, in order: the asleep screen's pill, then the gate's own region. */
const statusLines = () => screen.getAllByRole('status').map((status) => status.textContent)

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
  Object.assign(look, { play: true, narrow: false, widthUnknown: false })
  health.up = false
  mounts.page = 0
})

describe('the 404 (app/not-found.tsx)', () => {
  it("under Play at >= 1024 px: Auto lost, 'This page wandered off' and the way back to /chat", async () => {
    render(<NotFound />)
    // Play's lost screen loads through its lazy slot (P260-FX-23, M17).
    expect(await screen.findByRole('heading', { level: 1, name: LOST_TITLE })).toBeInTheDocument()
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

  it('under Play while the width is not known: nothing yet, neither the lost screen nor the plain page (M69)', async () => {
    look.widthUnknown = true
    const { container } = render(<NotFound />)
    // Let a lazy slot that had been asked for arrive: none was.
    await act(async () => {
      await Promise.resolve()
    })
    expect(container).toBeEmptyDOMElement()
    expect(screen.queryByTestId('auto')).not.toBeInTheDocument()
  })

  it("keeps Next's own 404 title for every style, so the tab and screen readers say it (M68)", () => {
    expect(metadata.title).toBe(NOT_FOUND_TITLE)
    expect(NOT_FOUND_TITLE).toBe('404: This page could not be found.')
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
    await settle()
    expect(healthCalls()).toBe(1 + BACKEND_HEALTH_RETRIES)
    expect(api.request).toHaveBeenCalledWith('/health', { signal: expect.any(AbortSignal) })
    expect(asleepHeading()).toBeInTheDocument()
    expect(screen.queryByText('The page')).not.toBeInTheDocument()
    expect(screen.queryByRole('navigation', { name: 'Main' })).not.toBeInTheDocument()
    expect(screen.getByTestId('auto')).toHaveAttribute('data-state', 'sleeping')
    expect(statusLines()).toEqual([NOT_CONNECTED, ASLEEP_TITLE])
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
    await settle()
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
    await settle()
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
    await settle()
    expect(healthCalls()).toBe(2 + BACKEND_HEALTH_RETRIES)
    expect(screen.getByText('The page')).toBeInTheDocument()
    expect(asleepHeading()).not.toBeInTheDocument()
  })

  it("a recheck's retries finish before the next automatic recheck asks", () => {
    expect(BACKEND_HEALTH_RETRIES * BACKEND_HEALTH_RETRY_DELAY_MS).toBeLessThan(ASLEEP_RECHECK_MS)
  })

  it('a frozen /health (it never answers) shows the asleep screen once every ask has timed out (M43)', async () => {
    renderFrame({ prime: freezeHealth })
    await elapse(0)
    const frozenFor =
      BACKEND_HEALTH_TIMEOUT_MS * (BACKEND_HEALTH_RETRIES + 1) + BACKEND_HEALTH_RETRY_DELAY_MS * BACKEND_HEALTH_RETRIES
    await elapse(frozenFor - 1)
    // The last ask has not given up yet: the page stays.
    expect(asleepHeading()).not.toBeInTheDocument()
    expect(screen.getByText('The page')).toBeInTheDocument()
    await elapse(1)
    await settle()
    expect(asleepHeading()).toBeInTheDocument()
    expect(healthCalls()).toBe(1 + BACKEND_HEALTH_RETRIES)
    expect(screen.queryByText('The page')).not.toBeInTheDocument()
  })

  it('mounted again while /health is down (browser Back), it opens on the asleep screen and the page never mounts (M89)', async () => {
    const earlier = renderFrame()
    await elapse(0)
    await throughRetries()
    expect(asleepHeading()).toBeInTheDocument()
    earlier.unmount()
    mounts.page = 0
    renderFrame({ client: earlier.client })
    expect(asleepHeading()).toBeInTheDocument()
    expect(mounts.page).toBe(0)
    // The new frame asks again on mount; still down, the page still never mounts.
    await elapse(0)
    await throughRetries()
    expect(asleepHeading()).toBeInTheDocument()
    expect(mounts.page).toBe(0)
  })

  it('Check again says Checking… while /health is asked and retried, then counts the full period again (M94)', async () => {
    await renderAsleepFrame()
    fireEvent.click(screen.getByRole('button', { name: CHECK_AGAIN }))
    await elapse(0)
    await settle()
    expect(screen.getByText(CHECKING)).toBeInTheDocument()
    expect(screen.queryByText(/^Checking again in/)).not.toBeInTheDocument()
    await throughRetries()
    expect(asleepHeading()).toBeInTheDocument()
    expect(screen.getByText('Checking again in 10 seconds.')).toBeInTheDocument()
    expect(countdownLine(ASLEEP_RECHECK_MS / SECOND_MS)).toBe('Checking again in 10 seconds.')
  })

  it('offline while /health fails: says so, with neither the hosted line nor the start command (M83)', async () => {
    await renderAsleepFrame()
    goOffline()
    connectivityEvent('offline')
    expect(screen.getByText(OFFLINE_LINE)).toBeInTheDocument()
    expect(screen.queryByText(HOSTED_LINE)).not.toBeInTheDocument()
    expect(screen.queryByText(START_COMMAND)).not.toBeInTheDocument()
    expect(asleepHeading()).toBeInTheDocument()
  })

  it("moves focus with the swap and says so: the asleep heading, then the frame's main once it answers (M21)", async () => {
    health.up = true
    renderFrame()
    await elapse(0)
    expect(statusLines()).toEqual([''])
    health.up = false
    await elapse(BACKEND_HEALTH_POLL_MS)
    await throughRetries()
    expect(asleepHeading()).toHaveFocus()
    expect(statusLines()).toEqual([NOT_CONNECTED, ASLEEP_TITLE])

    health.up = true
    fireEvent.click(screen.getByRole('button', { name: CHECK_AGAIN }))
    await elapse(0)
    await settle()
    expect(screen.getByText('The page')).toBeInTheDocument()
    expect(screen.getByRole('main')).toHaveFocus()
    expect(document.activeElement).not.toBe(document.body)
    expect(statusLines()).toEqual([AWAKE_LINE])
    expect(AWAKE_LINE).toBe('Automatos is back')
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

  it('mounted while the connection is down, Auto is asleep on the first render (M89)', () => {
    const renders: ReturnType<typeof useBackendMascot>[] = []
    const { result } = renderHook(() => {
      const auto = useBackendMascot('down')
      renders.push(auto)
      return auto
    })
    expect(renders[0]).toMatchObject({ mascot: 'sleeping', asleep: true })
    expect(result.current).toMatchObject({ mascot: 'sleeping', asleep: true })
  })
})

describe('the connection (connectionOf)', () => {
  it('stays down while a query that never answered is asked again, until the next answer (M89)', () => {
    const never = { isError: false, isSuccess: false, dataUpdatedAt: 0 }
    expect(connectionOf({ ...never, errorUpdatedAt: 0 })).toBe('checking')
    expect(connectionOf({ ...never, isError: true, errorUpdatedAt: 5 })).toBe('down')
    // react-query reads it as loading again while it asks: its last answer, a failure, still decides.
    expect(connectionOf({ ...never, errorUpdatedAt: 5 })).toBe('down')
    expect(connectionOf({ isError: false, isSuccess: true, errorUpdatedAt: 5, dataUpdatedAt: 9 })).toBe('connected')
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
    expect(START_COMMAND).toBe('make up')
    expect(screen.getByText(START_HINT)).toBeInTheDocument()
    expect(START_HINT).toBe('No make? docker compose up -d')
    expect(screen.getByText('automatos-ai')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: COPY_LABEL }))
    expect(await screen.findByRole('button', { name: COPIED_LABEL })).toBeInTheDocument()
    expect(await navigator.clipboard.readText()).toBe(START_COMMAND)
  })

  it('holds the count while a check is in flight, and counts the full period again once it ends (M94)', () => {
    vi.useFakeTimers()
    const onCheck = vi.fn()
    const { rerender } = render(asleepScreen({ onCheck, checking: true, recheckSeconds: 3 }))
    expect(screen.getByText(CHECKING)).toBeInTheDocument()
    tick(5 * SECOND_MS)
    expect(onCheck).not.toHaveBeenCalled()
    expect(screen.getByText(CHECKING)).toBeInTheDocument()
    rerender(asleepScreen({ onCheck, checking: false, recheckSeconds: 3 }))
    expect(screen.getByText('Checking again in 3 seconds.')).toBeInTheDocument()
    tick(SECOND_MS)
    expect(screen.getByText('Checking again in 2 seconds.')).toBeInTheDocument()
  })

  it('offline in the local edition: no start command, until the browser is back online (M83)', () => {
    const onLine = goOffline()
    renderAsleep({ local: true })
    expect(screen.getByText(OFFLINE_LINE)).toBeInTheDocument()
    expect(screen.queryByText(START_COMMAND)).not.toBeInTheDocument()
    expect(screen.queryByText(HOSTED_LINE)).not.toBeInTheDocument()
    onLine.mockReturnValue(true)
    connectivityEvent('online')
    expect(screen.getByText(START_COMMAND)).toBeInTheDocument()
    expect(screen.queryByText(OFFLINE_LINE)).not.toBeInTheDocument()
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
