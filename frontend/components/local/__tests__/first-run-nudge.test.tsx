/**
 * PRD-233 S3 — the local-edition first-run nudge.
 *
 * Pure/mocked: apiClient, the edition flag and next/link are stubbed; the
 * real react-query runs under a test QueryClient. Contract under test:
 *   empty keys (BYOK + platform)  ⇒ shown, linking to Settings
 *   any key                        ⇒ hidden
 *   dismissed (localStorage)       ⇒ hidden, and nothing is fetched
 *   click dismiss                  ⇒ hidden + remembered
 *   saas                           ⇒ hidden, nothing fetched
 *   request failure                ⇒ hidden (never nag on a guess)
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import React from 'react'

const { getMock, edition } = vi.hoisted(() => ({
  getMock: vi.fn(),
  edition: { isLocal: true },
}))

vi.mock('@/lib/api-client', () => ({
  apiClient: { get: (path: string) => getMock(path) },
}))
vi.mock('@/lib/auth-edition', () => ({
  get isLocal() {
    return edition.isLocal
  },
  get isSaaS() {
    return !edition.isLocal
  },
}))
vi.mock('next/link', () => ({
  default: ({ href, children, ...rest }: Record<string, unknown>) => (
    <a href={href as string} {...(rest as Record<string, string>)}>
      {children as React.ReactNode}
    </a>
  ),
}))

import {
  FirstRunNudge,
  FIRST_RUN_NUDGE_STORAGE_KEY,
  FIRST_RUN_NUDGE_SETTINGS_HREF,
  hasAnyLlmKey,
} from '../first-run-nudge'

const NO_PLATFORM_KEYS = { platform_keys: { openai: { configured: false }, anthropic: { configured: false } } }
const A_PLATFORM_KEY = { platform_keys: { openai: { configured: false }, openrouter: { configured: true } } }
const A_BYOK_KEY = [{ id: 1, provider: 'openrouter', is_active: true }]

function respond(keys: unknown, platform: unknown) {
  getMock.mockImplementation((path: string) => {
    if (path === '/api/keys') return Promise.resolve(keys)
    if (path === '/api/keys/platform-status') return Promise.resolve(platform)
    return Promise.reject(new Error(`unexpected path ${path}`))
  })
}

function renderNudge() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <FirstRunNudge />
    </QueryClientProvider>,
  )
}

async function settled() {
  await waitFor(() => expect(getMock).toHaveBeenCalledTimes(2))
  // let the resolved queries render
  await waitFor(() => expect(getMock).toHaveBeenCalledWith('/api/keys/platform-status'))
}

describe('FirstRunNudge (PRD-233 S3)', () => {
  beforeEach(() => {
    getMock.mockReset()
    edition.isLocal = true
    localStorage.clear()
  })
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('shows when the local workspace has no LLM key at all, linking to Settings', async () => {
    respond([], NO_PLATFORM_KEYS)
    renderNudge()

    const nudge = await screen.findByTestId('first-run-nudge')
    expect(nudge).toHaveTextContent('Add an LLM key to bring Auto to life')
    expect(screen.getByTestId('first-run-nudge-link')).toHaveAttribute('href', FIRST_RUN_NUDGE_SETTINGS_HREF)
    expect(getMock).toHaveBeenCalledWith('/api/keys')
    expect(getMock).toHaveBeenCalledWith('/api/keys/platform-status')
  })

  it('stays hidden when a BYOK key exists', async () => {
    respond(A_BYOK_KEY, NO_PLATFORM_KEYS)
    renderNudge()
    await settled()
    await waitFor(() => expect(screen.queryByTestId('first-run-nudge')).toBeNull())
  })

  it('stays hidden when a platform-level key is configured', async () => {
    respond([], A_PLATFORM_KEY)
    renderNudge()
    await settled()
    await waitFor(() => expect(screen.queryByTestId('first-run-nudge')).toBeNull())
  })

  it('stays hidden and fetches nothing once dismissed in localStorage', async () => {
    localStorage.setItem(FIRST_RUN_NUDGE_STORAGE_KEY, '1')
    respond([], NO_PLATFORM_KEYS)
    renderNudge()
    // give any (wrong) fetch a tick to fire
    await new Promise((r) => setTimeout(r, 20))
    expect(screen.queryByTestId('first-run-nudge')).toBeNull()
    expect(getMock).not.toHaveBeenCalled()
  })

  it('dismiss hides the banner and remembers it', async () => {
    respond([], NO_PLATFORM_KEYS)
    renderNudge()
    await screen.findByTestId('first-run-nudge')

    fireEvent.click(screen.getByTestId('first-run-nudge-dismiss'))

    expect(screen.queryByTestId('first-run-nudge')).toBeNull()
    expect(localStorage.getItem(FIRST_RUN_NUDGE_STORAGE_KEY)).toBe('1')
  })

  it('never renders or fetches in saas', async () => {
    edition.isLocal = false
    respond([], NO_PLATFORM_KEYS)
    renderNudge()
    await new Promise((r) => setTimeout(r, 20))
    expect(screen.queryByTestId('first-run-nudge')).toBeNull()
    expect(getMock).not.toHaveBeenCalled()
  })

  it('stays hidden when the key lookup fails (no nagging on a guess)', async () => {
    getMock.mockImplementation(() => Promise.reject(new Error('backend down')))
    renderNudge()
    await waitFor(() => expect(getMock).toHaveBeenCalled())
    await new Promise((r) => setTimeout(r, 20))
    expect(screen.queryByTestId('first-run-nudge')).toBeNull()
  })

  it('hasAnyLlmKey reads BYOK rows or a configured platform provider', () => {
    expect(hasAnyLlmKey(undefined, undefined)).toBe(false)
    expect(hasAnyLlmKey([], NO_PLATFORM_KEYS)).toBe(false)
    expect(hasAnyLlmKey(A_BYOK_KEY, NO_PLATFORM_KEYS)).toBe(true)
    expect(hasAnyLlmKey([], A_PLATFORM_KEY)).toBe(true)
  })
})
