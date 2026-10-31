/**
 * PRD-268 §6.11.3 (US-W05) — Settings → Orchestrator's "What Auto depends on": the server's items from
 * GET /api/readiness, in its order; this browser's microphone and speech rows, from the browser support
 * useVoice().available is made of, after them and never from the server; Check again posting
 * POST /api/readiness/check; "Checked a moment ago" when the server limits the check; no calendar row with the
 * calendar not connected (§12-Q20: no calendar in v1); and the same section under every theme, since /settings is
 * host-owned.
 */
import { fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { readFileSync } from 'fs'
import path from 'path'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const requestMock = vi.hoisted(() => vi.fn())
const postMock = vi.hoisted(() => vi.fn())
const voice = vi.hoisted(() => ({ browser: { read: true, quick: true, capture: true } }))
const style = vi.hoisted(() => ({ current: null as { isStudio: boolean; activeTheme: unknown } | null }))

vi.mock('@/lib/api-client', () => ({
  apiClient: {
    request: (...args: unknown[]) => requestMock(...args),
    post: (...args: unknown[]) => postMock(...args),
  },
}))
vi.mock('@/lib/voice/host/use-voice-availability', () => ({ useBrowserVoiceSupport: () => voice.browser }))
vi.mock('@/contexts/ui-style-context', () => ({ useUiStyleOptional: () => style.current }))
vi.mock('next/link', () => ({
  default: ({ href, children, ...rest }: { href: string; children: ReactNode; className?: string }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}))

import { CHECKED_A_MOMENT_AGO, CHECK_AGAIN, READINESS_TITLE } from '@/components/proactive/readiness-list'
import {
  BROWSER_DETAILS,
  WhatAutoDependsOn,
  browserReadinessItems,
} from '@/components/settings/what-auto-depends-on'
import type { ReadinessItem } from '@/lib/proactive/types'
import { HOST_OWNED_ROUTES } from '@/lib/theme-host/package-manifest'
import type { BrowserVoiceSupport } from '@/lib/voice/host/use-voice-availability'

// The server's items as services/readiness serves them (probes.ITEMS): no calendar, no microphone, no speech.
const CORE: ReadinessItem = {
  id: 'core',
  label: 'Automatos server',
  detail: 'The server and its database answer.',
  state: 'ready',
  needed_by: ['brief', 'voice', 'push', 'check_in'],
}
const MODEL: ReadinessItem = {
  id: 'model_route',
  label: "Auto's model",
  detail: "Auto's model has no key to run with.",
  state: 'needs_setup',
  fix: { label: 'Add a key', href: '/settings?tab=api-keys' },
  needed_by: ['brief', 'check_in'],
}
const SEARCH: ReadinessItem = { id: 'web_search', label: 'Web search', detail: "Didn't answer in time.", state: 'not_checked', needed_by: ['voice'] }
const PUSH: ReadinessItem = { id: 'push', label: 'Phone notifications', detail: 'Notifications can reach your phone.', state: 'ready', needed_by: ['push'] }

const CASES: [string, BrowserVoiceSupport, string, string][] = [
  ['the browser is not read yet', { read: false, quick: false, capture: false }, 'not_checked', 'not_checked'],
  ['it can capture and recognise speech', { read: true, quick: true, capture: true }, 'ready', 'ready'],
  ['it can capture but has no speech recognition', { read: true, quick: false, capture: true }, 'ready', 'needs_setup'],
  ['it recognises speech but cannot capture (an http page)', { read: true, quick: true, capture: false }, 'needs_setup', 'ready'],
  ['it can do neither', { read: true, quick: false, capture: false }, 'needs_setup', 'needs_setup'],
]

function serverReadiness(items: ReadinessItem[] = [CORE, MODEL, SEARCH, PUSH], checkedAt = '2026-10-30T08:00:00Z') {
  return { v: 1, checked_at: checkedAt, items }
}

function renderSection() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <WhatAutoDependsOn />
    </QueryClientProvider>,
  )
}

const rowIds = (container: HTMLElement) =>
  [...container.querySelectorAll('li[data-readiness-id]')].map((row) => row.getAttribute('data-readiness-id'))

beforeEach(() => {
  requestMock.mockReset()
  postMock.mockReset()
  voice.browser = { read: true, quick: true, capture: true }
  style.current = null
})

describe("What Auto depends on: the server's items", () => {
  it('reads GET /api/readiness and draws its items in its order, each with its glyph, word and fix', async () => {
    requestMock.mockResolvedValueOnce(serverReadiness())
    const { container } = renderSection()
    await screen.findByText('Automatos server')
    expect(requestMock).toHaveBeenCalledWith('/api/readiness')
    expect(screen.getByRole('region', { name: READINESS_TITLE })).toBeInTheDocument()
    expect(rowIds(container).slice(0, 4)).toEqual(['core', 'model_route', 'web_search', 'push'])
    expect(screen.getByRole('listitem', { name: 'Automatos server: ready.' })).toBeInTheDocument()
    expect(screen.getByRole('listitem', { name: "Auto's model: needs setup. Add a key." })).toBeInTheDocument()
    expect(screen.getByRole('listitem', { name: 'Web search: not checked.' })).toHaveAccessibleDescription("Didn't answer in time.")
    expect(screen.getByRole('link', { name: 'Add a key' })).toHaveAttribute('href', '/settings?tab=api-keys')
  })
})

describe("What Auto depends on: the browser's rows", () => {
  it("adds this browser's microphone and speech after the server's items, though the server sent neither", async () => {
    requestMock.mockResolvedValueOnce(serverReadiness())
    const { container } = renderSection()
    await screen.findByText('Automatos server')
    expect(rowIds(container)).toEqual(['core', 'model_route', 'web_search', 'push', 'microphone', 'speech'])
    expect(screen.getByRole('listitem', { name: 'Microphone: ready.' })).toHaveAccessibleDescription(BROWSER_DETAILS.microphoneReady)
    expect(screen.getByRole('listitem', { name: 'Speech: ready.' })).toHaveAccessibleDescription(BROWSER_DETAILS.speechReady)
  })

  it('draws a browser that can do neither as needing setup, and says why', async () => {
    requestMock.mockResolvedValueOnce(serverReadiness())
    voice.browser = { read: true, quick: false, capture: false }
    renderSection()
    await screen.findByText('Automatos server')
    expect(screen.getByRole('listitem', { name: 'Speech: needs setup.' })).toHaveAccessibleDescription(BROWSER_DETAILS.speechMissing)
    expect(screen.getByRole('listitem', { name: 'Microphone: needs setup.' })).toHaveAccessibleDescription(BROWSER_DETAILS.microphoneMissing)
  })

  it.each(CASES)('reads the rows when %s', (_, browser, microphone, speech) => {
    const rows = browserReadinessItems(browser)
    expect(rows.map((row) => [row.id, row.state])).toEqual([
      ['microphone', microphone],
      ['speech', speech],
    ])
    rows.forEach((row) => {
      expect(row.needed_by).toEqual(['voice'])
      expect(row.fix).toBeUndefined()
    })
  })
})

describe("What Auto depends on: this browser's support, as read", () => {
  const speech = window as unknown as Record<string, unknown>

  afterEach(() => {
    delete speech.webkitSpeechRecognition
    delete speech.speechSynthesis
    Reflect.deleteProperty(navigator, 'mediaDevices')
  })

  async function readSupport(): Promise<BrowserVoiceSupport[]> {
    const { useBrowserVoiceSupport } = await vi.importActual<typeof import('@/lib/voice/host/use-voice-availability')>(
      '@/lib/voice/host/use-voice-availability',
    )
    const seen: BrowserVoiceSupport[] = []
    renderHook(() => {
      const support = useBrowserVoiceSupport()
      seen.push(support)
      return support
    })
    return seen
  }

  it('says nothing before mount, then reads a browser with neither speech nor capture', async () => {
    const seen = await readSupport()
    expect(seen[0]).toEqual({ read: false, quick: false, capture: false })
    expect(seen[seen.length - 1]).toEqual({ read: true, quick: false, capture: false })
    expect(requestMock).not.toHaveBeenCalled()
  })

  it('reads speech recognition and synthesis, and microphone capture, when the browser has them', async () => {
    speech.webkitSpeechRecognition = function Recognition() {}
    speech.speechSynthesis = {}
    Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia: vi.fn() } })
    const seen = await readSupport()
    expect(seen[seen.length - 1]).toEqual({ read: true, quick: true, capture: true })
  })
})

describe('What Auto depends on: Check again', () => {
  it("posts POST /api/readiness/check and shows its answer, keeping the browser's rows", async () => {
    requestMock.mockResolvedValueOnce(serverReadiness())
    const fixed = { ...MODEL, state: 'ready' as const, detail: "Auto's model is chosen and has a key to run with.", fix: null }
    postMock.mockResolvedValueOnce(serverReadiness([CORE, fixed, SEARCH, PUSH], '2026-10-30T08:05:00Z'))
    renderSection()
    await screen.findByText('Automatos server')
    fireEvent.click(screen.getByRole('button', { name: CHECK_AGAIN }))
    await waitFor(() => expect(postMock).toHaveBeenCalledWith('/api/readiness/check'))
    expect(await screen.findByRole('listitem', { name: "Auto's model: ready." })).toBeInTheDocument()
    expect(screen.getByRole('listitem', { name: 'Microphone: ready.' })).toBeInTheDocument()
    expect(postMock).toHaveBeenCalledTimes(1)
  })

  it("says 'Checked a moment ago' when the server limits the check, and keeps the items shown", async () => {
    requestMock.mockResolvedValueOnce(serverReadiness())
    const detail = 'Checked a moment ago. Try again in a minute.'
    postMock.mockRejectedValueOnce(Object.assign(new Error(detail), { status: 429, body: { code: 'checked_recently', detail } }))
    renderSection()
    await screen.findByText('Automatos server')
    fireEvent.click(screen.getByRole('button', { name: CHECK_AGAIN }))
    expect(await screen.findByText(CHECKED_A_MOMENT_AGO)).toHaveAttribute('role', 'status')
    expect(screen.queryByRole('alert')).toBeNull()
    expect(screen.getByRole('listitem', { name: "Auto's model: needs setup. Add a key." })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: CHECK_AGAIN })).toBeEnabled()
  })
})

describe('What Auto depends on: no calendar (§12-Q20)', () => {
  it("draws no calendar row, and no 'Couldn't check' row for it, with the calendar not connected", async () => {
    requestMock.mockResolvedValueOnce(serverReadiness())
    const { container } = renderSection()
    await screen.findByText('Automatos server')
    expect(container.querySelector('li[data-readiness-id="calendar"]')).toBeNull()
    expect(screen.queryByText(/calendar/i)).toBeNull()
    expect(screen.queryByText(/couldn.t check/i)).toBeNull()
  })

  it.each(CASES)('adds only the microphone and speech when %s', (_, browser) => {
    expect(browserReadinessItems(browser).map((row) => row.id)).toEqual(['microphone', 'speech'])
  })
})

// A theme is drawn on Studio or on Classic (ThemeBase): under it, the page renders its base's look (useIsStudio).
const THEMES = [
  { id: 'play', base: 'studio' },
  { id: 'hud', base: 'studio' },
  { id: 'paper', base: 'classic' },
]

/** The section as drawn under a style, once the server's items are in; React's generated ids left out. */
async function drawnUnder(current: { isStudio: boolean; activeTheme: unknown }): Promise<string> {
  style.current = current
  requestMock.mockResolvedValueOnce(serverReadiness())
  const { container, unmount } = renderSection()
  await screen.findByText('Automatos server')
  const html = container.innerHTML.replace(/:r[0-9a-z]+:/g, ':id:')
  unmount()
  return html
}

describe('What Auto depends on: the host draws it under every theme', () => {
  it('sits on a host-owned route that no theme can replace', () => {
    expect(HOST_OWNED_ROUTES).toContain('/settings')
  })

  it('is drawn under each theme exactly as under the look the theme is drawn on', async () => {
    const looks = { studio: await drawnUnder({ isStudio: true, activeTheme: null }), classic: await drawnUnder({ isStudio: false, activeTheme: null }) }
    for (const theme of THEMES) {
      expect(await drawnUnder({ isStudio: theme.base === 'studio', activeTheme: theme })).toBe(looks[theme.base as 'studio' | 'classic'])
    }
  })

  it('draws the same rows, in the same order and words, in Studio and in Classic', async () => {
    const rowsOf = (html: string) => {
      const host = document.createElement('div')
      host.innerHTML = html
      return [...host.querySelectorAll('li[data-readiness-id]')].map((row) => [row.getAttribute('data-readiness-id'), row.getAttribute('aria-label')])
    }
    const studio = rowsOf(await drawnUnder({ isStudio: true, activeTheme: null }))
    expect(rowsOf(await drawnUnder({ isStudio: false, activeTheme: null }))).toEqual(studio)
    expect(studio).toHaveLength(6)
  })

  it('reads nothing from a theme: no theme host, no theme SDK, no slot', () => {
    const source = readFileSync(path.resolve(__dirname, '..', 'what-auto-depends-on.tsx'), 'utf8')
    expect(source).not.toMatch(/@\/lib\/theme-(?:host|sdk)|@\/components\/theme-host|useActiveTheme/)
  })
})
