/**
 * PRD-261 A-19e2 — health down → the asleep slot, on the fixture theme
 * (lib/theme-host/__tests__/fixture-theme-entry.ts, A-19b). The fixture's entry lists asleep@1: its
 * frame draws the asleep screen itself, in place of the page, and owns the page's one GET /health
 * poll (so Settings → Session mode reads it there, hooks/__tests__/use-backend-health.test.tsx). The
 * frame here is written as a theme writes it, against '@automatos/theme-sdk' only: useHealth({ poll:
 * true }), its asleep@1 screen while the connection is down, the page otherwise. What it reads is the
 * host's: MainLayout's frame slot hands it the page, and the SDK's useHealth is the host's one /health
 * query (hooks/use-backend-health.ts) over a stood-in backend client, on fake timers.
 *
 * Down only once /health has failed on the first ask and every retry (P260-RVW-1), so a blip never
 * unmounts the page; a frozen /health is down once every ask has timed out (M43); `check` asks again
 * at once and `checking` holds while it and its retries are in flight (M94); an answer mounts the page
 * again; a frame mounted while it is down opens on the asleep screen and its page never mounts (M89).
 *
 * Replaces, in the core test ledger (plan §2.2.8), the core assertions of
 * components/play/edges/__tests__/edges.test.tsx ("Play's frame while Automatos is asleep (GET
 * /health)" and "the connection (connectionOf)"). Play's own asleep screen (its countdown, the start
 * command, offline, the focus move) and its 404 move to the kit (B-05f).
 */
import React, { useEffect, useState, type ComponentType } from 'react'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ request: vi.fn(), getBaseUrl: () => 'http://localhost:8000' }))
vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('next/navigation', async (importOriginal) => ({
  ...(await importOriginal<typeof import('next/navigation')>()),
  usePathname: () => '/command-center',
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
}))
// The fixture theme at its width, on a desktop: MainLayout draws the theme's frame.
vi.mock('@/hooks/use-mobile', () => ({ useIsTabletOrBelow: () => false, useIsMobile: () => false }))
vi.mock('@/hooks/use-studio-theme', async () => {
  const { FIXTURE_THEME_ENTRY } = await import('@/lib/theme-host/__tests__/fixture-theme-entry')
  return { useIsStudio: () => true, useActiveTheme: () => FIXTURE_THEME_ENTRY, useThemeLayout: () => 'theme' }
})
// The core's chrome, the frame slot's fallback, never drawn here: its parts have their own tests.
vi.mock('@/components/layout/sidebar', () => ({ Sidebar: () => null }))
vi.mock('@/components/layout/mobile-sidebar', () => ({ MobileSidebar: () => null }))
vi.mock('@/components/layout/header', () => ({ Header: () => null }))
vi.mock('@/components/layout/studio-sidebar', () => ({ StudioSidebar: () => null }))
vi.mock('@/components/layout/studio-header', () => ({ StudioHeader: () => null, StudioHeaderUtilities: () => null }))
vi.mock('@/components/chatbot/chat-widget', () => ({ AutoWidget: () => null }))

import { useHealth } from '@automatos/theme-sdk'
import { MainLayout } from '@/components/layout/main-layout'
import { ThemeHostContext } from '@/components/theme-host/theme-host-context'
import {
  BACKEND_HEALTH_POLL_MS,
  BACKEND_HEALTH_RETRIES,
  BACKEND_HEALTH_RETRY_DELAY_MS,
  BACKEND_HEALTH_TIMEOUT_MS,
  connectionOf,
} from '@/hooks/use-backend-health'
import {
  FIXTURE_THEME_REGISTRATIONS,
  FIXTURE_THEME_SLOTS,
  FIXTURE_THEME_TEST_IDS,
  fixtureThemeHost,
} from '@/lib/theme-host/__tests__/fixture-theme-entry'
import type { SlotId } from '@/lib/theme-host/registry-types'
import type { FrameProps } from '@/lib/theme-host/types'

const ASLEEP_SLOT: SlotId = 'asleep@1'
const HEALTH_PATH = '/health'
const PAGE_TEXT = 'The page'
const DRAFT_LABEL = 'Draft'
const TYPED = 'Half a message, not sent yet'
const CHECK_AGAIN = 'Check again'
const ids = FIXTURE_THEME_TEST_IDS

/** Whether GET /health answers, and how often the page has mounted. */
const health = vi.hoisted(() => ({ up: false }))
const mounts = vi.hoisted(() => ({ page: 0 }))

/** The fixture theme's asleep screen, the component it registered as asleep@1. */
const FixtureAsleep = FIXTURE_THEME_REGISTRATIONS.slots.get(ASLEEP_SLOT) as ComponentType

/** The fixture theme's frame: the page's one /health poller, its asleep screen in place of the page while down. */
function AsleepGateFrame({ children }: FrameProps) {
  const { connection, checking, check } = useHealth({ poll: true })
  return (
    <section data-testid={ids.frame} data-connection={connection} data-checking={String(checking)}>
      {connection === 'down' ? (
        <>
          <FixtureAsleep />
          <button type="button" onClick={check}>{CHECK_AGAIN}</button>
        </>
      ) : (
        <main>{children}</main>
      )}
    </section>
  )
}

const HOST = fixtureThemeHost('active', { frame: AsleepGateFrame })

/** The page inside the frame, counting its mounts; its draft lives only while it stays mounted. */
function Page() {
  const [draft, setDraft] = useState('')
  useEffect(() => {
    mounts.page += 1
  }, [])
  return (
    <>
      <p>{PAGE_TEXT}</p>
      <input aria-label={DRAFT_LABEL} value={draft} onChange={(event) => setDraft(event.target.value)} />
    </>
  )
}

function primeHealth() {
  api.request.mockImplementation((endpoint: string) => {
    if (endpoint !== HEALTH_PATH) return Promise.reject(new Error(`unexpected call to ${endpoint}`))
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

const healthCalls = () => api.request.mock.calls.filter(([endpoint]) => endpoint === HEALTH_PATH).length
const newClient = () => new QueryClient({ defaultOptions: { queries: { retry: false } } })

/** MainLayout under the fixture theme; pass an earlier frame's client to mount over its cache, as browser Back does. */
function renderFrame({ prime = primeHealth, client = newClient() } = {}) {
  prime()
  const { unmount } = render(
    <QueryClientProvider client={client}>
      <ThemeHostContext.Provider value={HOST}>
        <MainLayout>
          <Page />
        </MainLayout>
      </ThemeHostContext.Provider>
    </QueryClientProvider>,
  )
  return { client, unmount }
}

/** Let `ms` of fake time pass, with the promises and React updates it starts. */
const elapse = (ms: number) =>
  act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })

/** Let react-query hand a settled ask to React: its notify is a setTimeout(0), which fires a fake ms later. */
const settle = () => elapse(1)

/** Let every retry of the ask in flight run, each failing or answering as `health.up` says, and settle. */
async function throughRetries() {
  for (let retry = 0; retry < BACKEND_HEALTH_RETRIES; retry += 1) await elapse(BACKEND_HEALTH_RETRY_DELAY_MS)
  await settle()
}

const asleep = () => screen.queryByTestId(ids.asleep)
const frame = () => screen.getByTestId(ids.frame)

async function renderAsleepFrame() {
  const rendered = renderFrame()
  await elapse(0)
  await throughRetries()
  expect(asleep()).toBeInTheDocument()
  return rendered
}

beforeEach(() => {
  vi.useFakeTimers()
  vi.spyOn(console, 'error').mockImplementation(() => undefined)
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.restoreAllMocks()
  api.request.mockReset()
  health.up = false
  mounts.page = 0
})

describe('the fixture theme declares asleep@1', () => {
  it('lists the slot in its entry and registers its asleep screen', () => {
    expect(FIXTURE_THEME_SLOTS).toContain(ASLEEP_SLOT)
    expect(FixtureAsleep).toEqual(expect.any(Function))
  })
})

describe("the theme's frame while /health is down (the SDK's useHealth, the host's one query)", () => {
  it('shows the asleep screen in place of the page only once /health has failed on the first ask and every retry', async () => {
    renderFrame()
    await elapse(0)
    for (let retry = 1; retry <= BACKEND_HEALTH_RETRIES; retry += 1) {
      // A failed ask is not the answer yet: the page stays while /health is asked again.
      expect(asleep()).not.toBeInTheDocument()
      expect(screen.getByText(PAGE_TEXT)).toBeInTheDocument()
      await elapse(BACKEND_HEALTH_RETRY_DELAY_MS)
    }
    await settle()
    expect(healthCalls()).toBe(1 + BACKEND_HEALTH_RETRIES)
    expect(api.request).toHaveBeenCalledWith(HEALTH_PATH, { signal: expect.any(AbortSignal) })
    expect(asleep()).toBeInTheDocument()
    expect(frame()).toHaveAttribute('data-connection', 'down')
    expect(screen.queryByText(PAGE_TEXT)).not.toBeInTheDocument()
  })

  it('one failed poll whose retry answers never shows it, and the page keeps what was typed', async () => {
    health.up = true
    renderFrame()
    await elapse(0)
    fireEvent.change(screen.getByRole('textbox', { name: DRAFT_LABEL }), { target: { value: TYPED } })
    health.up = false
    await elapse(BACKEND_HEALTH_POLL_MS)
    expect(healthCalls()).toBe(2)
    expect(asleep()).not.toBeInTheDocument()
    health.up = true
    await elapse(BACKEND_HEALTH_RETRY_DELAY_MS)
    expect(healthCalls()).toBe(3)
    expect(asleep()).not.toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: DRAFT_LABEL })).toHaveValue(TYPED)
    expect(mounts.page).toBe(1)
  })

  it('after an answer, a poll that fails on every retry shows it, and only then', async () => {
    health.up = true
    renderFrame()
    await elapse(0)
    health.up = false
    await elapse(BACKEND_HEALTH_POLL_MS)
    for (let retry = 1; retry <= BACKEND_HEALTH_RETRIES; retry += 1) {
      expect(asleep()).not.toBeInTheDocument()
      await elapse(BACKEND_HEALTH_RETRY_DELAY_MS)
    }
    await settle()
    expect(healthCalls()).toBe(2 + BACKEND_HEALTH_RETRIES)
    expect(asleep()).toBeInTheDocument()
  })

  it('a frozen /health (it never answers) is down once every ask has timed out (M43)', async () => {
    renderFrame({ prime: freezeHealth })
    await elapse(0)
    const frozenFor = BACKEND_HEALTH_TIMEOUT_MS * (BACKEND_HEALTH_RETRIES + 1) + BACKEND_HEALTH_RETRY_DELAY_MS * BACKEND_HEALTH_RETRIES
    await elapse(frozenFor - 1)
    // The last ask has not given up yet: the page stays.
    expect(asleep()).not.toBeInTheDocument()
    expect(screen.getByText(PAGE_TEXT)).toBeInTheDocument()
    await elapse(1)
    await settle()
    expect(asleep()).toBeInTheDocument()
    expect(healthCalls()).toBe(1 + BACKEND_HEALTH_RETRIES)
  })

  it('check asks again at once; checking holds while it and its retries are in flight, and the page stays away (M94)', async () => {
    await renderAsleepFrame()
    const before = healthCalls()
    const mountsWhileAsleep = mounts.page
    fireEvent.click(screen.getByRole('button', { name: CHECK_AGAIN }))
    await elapse(0)
    await settle()
    expect(healthCalls()).toBe(before + 1)
    expect(frame()).toHaveAttribute('data-checking', 'true')
    expect(asleep()).toBeInTheDocument()
    await throughRetries()
    expect(healthCalls()).toBe(before + 1 + BACKEND_HEALTH_RETRIES)
    expect(frame()).toHaveAttribute('data-checking', 'false')
    expect(asleep()).toBeInTheDocument()
    expect(mounts.page).toBe(mountsWhileAsleep)
  })

  it('mounts the page again once /health answers', async () => {
    await renderAsleepFrame()
    const mountsWhileAsleep = mounts.page
    health.up = true
    fireEvent.click(screen.getByRole('button', { name: CHECK_AGAIN }))
    await elapse(0)
    await settle()
    expect(screen.getByText(PAGE_TEXT)).toBeInTheDocument()
    expect(mounts.page).toBe(mountsWhileAsleep + 1)
    expect(asleep()).not.toBeInTheDocument()
    expect(frame()).toHaveAttribute('data-connection', 'connected')
  })

  it('mounted again while /health is down (browser Back), it opens on the asleep screen and the page never mounts (M89)', async () => {
    const earlier = await renderAsleepFrame()
    earlier.unmount()
    mounts.page = 0
    renderFrame({ client: earlier.client })
    expect(asleep()).toBeInTheDocument()
    expect(mounts.page).toBe(0)
    // The new frame asks again on mount; still down, the page still never mounts.
    await elapse(0)
    await throughRetries()
    expect(asleep()).toBeInTheDocument()
    expect(mounts.page).toBe(0)
  })

  it('while /health answers: the page in the frame, one /health a period from the frame’s poll', async () => {
    health.up = true
    renderFrame()
    expect(screen.getByText(PAGE_TEXT)).toBeInTheDocument()
    await elapse(0)
    expect(healthCalls()).toBe(1)
    await elapse(BACKEND_HEALTH_POLL_MS)
    expect(healthCalls()).toBe(2)
    expect(asleep()).not.toBeInTheDocument()
    expect(mounts.page).toBe(1)
  })

  it('a frozen /health reads as down inside one poll period: every ask timed out, with the waits between them', () => {
    const frozenFor = BACKEND_HEALTH_TIMEOUT_MS * (BACKEND_HEALTH_RETRIES + 1) + BACKEND_HEALTH_RETRY_DELAY_MS * BACKEND_HEALTH_RETRIES
    expect(frozenFor).toBeLessThan(BACKEND_HEALTH_POLL_MS)
  })
})

describe('the connection the swap reads (connectionOf)', () => {
  it('stays down while a query that never answered is asked again, until the next answer (M89)', () => {
    const never = { isError: false, isSuccess: false, dataUpdatedAt: 0 }
    expect(connectionOf({ ...never, errorUpdatedAt: 0 })).toBe('checking')
    expect(connectionOf({ ...never, isError: true, errorUpdatedAt: 5 })).toBe('down')
    // react-query reads it as loading again while it asks: its last answer, a failure, still decides.
    expect(connectionOf({ ...never, errorUpdatedAt: 5 })).toBe('down')
    expect(connectionOf({ isError: false, isSuccess: true, errorUpdatedAt: 5, dataUpdatedAt: 9 })).toBe('connected')
  })
})
