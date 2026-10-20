/**
 * PRD-260 P260-FX-27 (M88) — one GET /health query, one poll. react-query runs
 * a timer per polling observer, each in its own phase, so every
 * useBackendHealth that polled sent its own request each period (two every
 * 30 s on Play's chat), and Settings → Session mode asked again under a key of
 * its own. Now only the caller that asks for `poll` (Play's AsleepGate) runs
 * the timer, every other caller reads the shared answer, and Session mode
 * reads the same query: it polls only where Play's frame doesn't.
 * The apiClient is a stand-in and the timers are fake; roles and text only.
 */
import React from 'react'
import { act, cleanup, render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ThemeEntry } from '@/lib/theme-host/registry-types'

const api = vi.hoisted(() => ({ request: vi.fn(), getBaseUrl: () => 'http://localhost:8000' }))
vi.mock('@/lib/api-client', () => ({ apiClient: api }))

// PRD-261 A-02b: the tab reads the frame's poll where the theme's layout renders and its entry
// has the asleep slot (the fixture theme's does, PRD-261 A-19b); `theme` stands in for the active entry.
const layout = vi.hoisted(() => ({
  current: 'fallback' as 'pending' | 'theme' | 'fallback',
  theme: null as ThemeEntry | null,
}))
vi.mock('@/hooks/use-studio-theme', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/hooks/use-studio-theme')>()),
  useThemeLayout: () => layout.current,
  useActiveTheme: () => layout.theme,
}))

import { SessionModeTab } from '@/components/settings/SessionModeTab'
import { BACKEND_HEALTH_POLL_MS, useBackendHealth } from '@/hooks/use-backend-health'
import { FIXTURE_THEME_ENTRY } from '@/lib/theme-host/__tests__/fixture-theme-entry'

/** How long after the poller the reader mounts: well inside one period, out of phase with it. */
const READER_LATE_MS = 5_000
const SESSION_MODE_OFF = 'Session mode is off on this instance.'

/** Play's frame: its AsleepGate owns the poll. */
function FramePoll() {
  useBackendHealth({ poll: true })
  return null
}

/** Play's pill, chat or shelf: reads the frame's answers. */
function Reader() {
  const { connection } = useBackendHealth()
  return <p role="status">{connection}</p>
}

const newClient = () => new QueryClient({ defaultOptions: { queries: { retry: false } } })

function renderWith(client: QueryClient, ui: React.ReactElement) {
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}

const healthCalls = () => api.request.mock.calls.filter(([endpoint]) => endpoint === '/health').length

/** Let `ms` of fake time pass, with the promises and React updates it starts. */
const elapse = (ms: number) =>
  act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })

beforeEach(() => {
  vi.useFakeTimers()
  api.request.mockImplementation((endpoint: string) =>
    endpoint === '/health'
      ? Promise.resolve({ status: 'healthy', edition: 'local', cli_runtime_enabled: false })
      : Promise.reject(new Error(`unexpected call to ${endpoint}`)),
  )
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  api.request.mockReset()
  layout.current = 'fallback'
  layout.theme = null
})

describe('useBackendHealth — one poll owner, every other caller reads', () => {
  it('a poller mounted at t=0 and a reader mounted 5 s later send one /health per period, not two', async () => {
    const client = newClient()
    renderWith(client, <FramePoll />)
    await elapse(0)
    expect(healthCalls()).toBe(1)
    await elapse(READER_LATE_MS)
    renderWith(client, <Reader />)
    await elapse(0)
    // The answer is fresh until the poll asks again: the reader reads it, it doesn't ask.
    expect(healthCalls()).toBe(1)
    expect(screen.getByRole('status')).toHaveTextContent('connected')
    await elapse(2 * BACKEND_HEALTH_POLL_MS)
    expect(healthCalls()).toBe(1 + 2)
  })

  it('a reader alone asks once, on mount, and never polls', async () => {
    renderWith(newClient(), <Reader />)
    await elapse(0)
    expect(healthCalls()).toBe(1)
    expect(screen.getByRole('status')).toHaveTextContent('connected')
    await elapse(2 * BACKEND_HEALTH_POLL_MS)
    expect(healthCalls()).toBe(1)
  })
})

describe('Settings → Session mode reads the same /health query', () => {
  it("shares one request and one answer with Play's connection", async () => {
    renderWith(
      newClient(),
      <>
        <Reader />
        <SessionModeTab />
      </>,
    )
    await elapse(0)
    await elapse(1)
    expect(healthCalls()).toBe(1)
    expect(screen.getByRole('status')).toHaveTextContent('connected')
    expect(screen.getByText(SESSION_MODE_OFF)).toBeInTheDocument()
  })

  it("inside a theme's frame with the asleep slot it reads the frame's poll: one /health per period, mounted 5 s apart", async () => {
    layout.current = 'theme'
    layout.theme = FIXTURE_THEME_ENTRY
    const client = newClient()
    renderWith(client, <FramePoll />)
    await elapse(READER_LATE_MS)
    renderWith(client, <SessionModeTab />)
    await elapse(2 * BACKEND_HEALTH_POLL_MS)
    expect(healthCalls()).toBe(1 + 2)
    expect(screen.getByText(SESSION_MODE_OFF)).toBeInTheDocument()
  })

  it("under a theme at its width whose entry has no asleep slot, the tab still polls /health itself", async () => {
    layout.current = 'theme'
    layout.theme = { ...FIXTURE_THEME_ENTRY, slots: FIXTURE_THEME_ENTRY.slots.filter((slot) => slot !== 'asleep@1') }
    renderWith(newClient(), <SessionModeTab />)
    await elapse(0)
    expect(healthCalls()).toBe(1)
    await elapse(BACKEND_HEALTH_POLL_MS)
    expect(healthCalls()).toBe(2)
  })

  it('outside it (Classic, Studio, Play below its width) the tab polls /health itself, as before', async () => {
    renderWith(newClient(), <SessionModeTab />)
    await elapse(0)
    expect(healthCalls()).toBe(1)
    await elapse(BACKEND_HEALTH_POLL_MS)
    expect(healthCalls()).toBe(2)
  })
})
