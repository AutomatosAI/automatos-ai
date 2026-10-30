/**
 * PRD-268 §6.4, K11 (US-W02) — BriefOnOpen opens the day's brief (POST /api/proactive/brief/open) once per local date
 * when a signed-in workspace loads: one call per date across reloads of a tab, a new date calls again, no workspace
 * no call, a refusal logged once and never asked again in that load, once per load with storage blocked, and the
 * answer opens nothing by itself (no navigation, no sound, nothing drawn). A `brief_off` refusal (the workspace's
 * proactive mode is off, P268W2-RVW-3) logs nothing; any other code logs as before. It is mounted once in
 * components/providers.tsx, inside WorkspaceProvider.
 */
import { readFileSync } from 'fs'
import path from 'path'
import React from 'react'
import { act, render, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from 'vitest'

const postMock = vi.hoisted(() => vi.fn())
const workspace = vi.hoisted(() => ({ id: 'ws-1' as string | null }))
const router = vi.hoisted(() => ({ push: vi.fn(), replace: vi.fn() }))

vi.mock('@/lib/api-client', () => ({
  apiClient: {
    post: (...args: unknown[]) => postMock(...args),
    request: vi.fn(),
  },
}))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspaceOptional: () => (workspace.id === null ? null : { workspace: { id: workspace.id } }),
}))
vi.mock('next/navigation', () => ({ useRouter: () => router }))

import { BriefOnOpen } from '@/components/proactive/brief-on-open'
import { rememberedBrief } from '@/lib/proactive/brief-day'
import { HEX_COLOUR, MAX_COMPONENT_LINES, ORANGE_CLASS, componentSpans, proactiveSource } from './source-rules'

const OPEN_PATH = '/api/proactive/brief/open'
const CHAT = '5b7f2c1e-8d4a-4f6b-9c3e-1a2b3c4d5e6f'
const MESSAGE = '9e8d7c6b-5a4f-4e3d-8c2b-1a0f9e8d7c6b'
const POSTED = { chat_id: CHAT, message_id: MESSAGE, posted: true }
const OCT_30 = new Date(2026, 9, 30, 9, 0, 0)
const OCT_31 = new Date(2026, 9, 31, 8, 0, 0)
const TODAY = '2026-10-30'

/**
 * A load of the tab: a fresh query client, as a reload gives (localStorage is kept, as a reload keeps it). `again`
 * renders the same load once more, after the test has changed the workspace.
 */
function load() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  const tree = () => (
    <QueryClientProvider client={client}>
      <BriefOnOpen />
    </QueryClientProvider>
  )
  const view = render(tree())
  return { ...view, again: () => view.rerender(tree()) }
}

/** Lets an open that was asked for reach the api client (the mutation posts after a few microtasks). */
const settle = () => act(async () => new Promise<void>((resolve) => setTimeout(resolve, 0)))

/** The bodies each open posted. */
const opened = () => postMock.mock.calls.filter(([url]) => url === OPEN_PATH).map(([, body]) => body)

/** Each warning (or error) BriefOnOpen logged: every line it logs starts with its prefix. */
const briefWarnings = (warn: MockInstance<typeof console.warn>) => warn.mock.calls.filter(([prefix]) => prefix === '[brief]')

const refusal = (status: number, code = 'open_out_of_window') =>
  Object.assign(new Error('Refused'), { status, body: { code, detail: 'No.' } })

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['Date'] })
  vi.setSystemTime(OCT_30)
  postMock.mockReset()
  postMock.mockResolvedValue(POSTED)
  router.push.mockReset()
  router.replace.mockReset()
  workspace.id = 'ws-1'
  window.localStorage.clear()
})

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('BriefOnOpen: once per local date', () => {
  it("opens the day's brief when a workspace loads, with the browser's date, zone and language, and remembers where it is", async () => {
    load()
    await waitFor(() => expect(opened()).toHaveLength(1))
    expect(opened()[0]).toEqual({ local_date: TODAY, timezone: expect.any(String), locale: navigator.language })
    await waitFor(() => expect(rememberedBrief('ws-1', TODAY)).toEqual({ chat_id: CHAT, message_id: MESSAGE }))
  })

  it('makes one call per local date across reloads of a tab', async () => {
    const first = load()
    await waitFor(() => expect(rememberedBrief('ws-1', TODAY)).not.toBeNull())
    first.unmount()
    load().unmount()
    load()
    await settle()
    expect(opened()).toHaveLength(1)
  })

  it('calls again on a new date', async () => {
    const first = load()
    await waitFor(() => expect(rememberedBrief('ws-1', TODAY)).not.toBeNull())
    first.unmount()
    vi.setSystemTime(OCT_31)
    load()
    await waitFor(() => expect(opened()).toHaveLength(2))
    expect(opened()[1]).toMatchObject({ local_date: '2026-10-31' })
  })

  it('asks the next load again after a pending answer, which names no message', async () => {
    postMock.mockResolvedValueOnce({ posted: false, pending: true })
    const first = load()
    await waitFor(() => expect(opened()).toHaveLength(1))
    await settle()
    expect(rememberedBrief('ws-1', TODAY)).toBeNull()
    first.unmount()
    load()
    await waitFor(() => expect(opened()).toHaveLength(2))
    await waitFor(() => expect(rememberedBrief('ws-1', TODAY)).not.toBeNull())
  })

  it('opens each workspace once: a switch asks for the new one, a switch back asks nothing', async () => {
    const view = load()
    await waitFor(() => expect(opened()).toHaveLength(1))
    workspace.id = 'ws-2'
    view.again()
    await waitFor(() => expect(opened()).toHaveLength(2))
    workspace.id = 'ws-1'
    view.again()
    await settle()
    expect(opened()).toHaveLength(2)
  })
})

describe('BriefOnOpen: no workspace, no call', () => {
  it('asks nothing without a signed-in workspace, then opens once one loads', async () => {
    workspace.id = null
    const view = load()
    view.again()
    await settle()
    expect(postMock).not.toHaveBeenCalled()
    workspace.id = 'ws-1'
    view.again()
    await waitFor(() => expect(opened()).toHaveLength(1))
  })
})

describe('BriefOnOpen: refusals and blocked storage', () => {
  it('logs a 4xx once and never asks for that day again in that load', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    postMock.mockRejectedValueOnce(refusal(422))
    const view = load()
    await waitFor(() => expect(briefWarnings(warn)).toHaveLength(1))
    expect(briefWarnings(warn)[0]).toEqual(['[brief]', expect.any(String), 422, 'open_out_of_window'])
    expect(rememberedBrief('ws-1', TODAY)).toBeNull()
    workspace.id = 'ws-2'
    view.again()
    await waitFor(() => expect(opened()).toHaveLength(2))
    workspace.id = 'ws-1'
    view.again()
    await settle()
    expect(opened()).toHaveLength(2)
    expect(briefWarnings(warn)).toHaveLength(1)
  })

  it("still logs an open's refusal once when the workspace switched while it was in flight", async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    let refuseFirst: (error: Error) => void = () => undefined
    postMock.mockImplementationOnce(() => new Promise((_, reject) => { refuseFirst = reject }))
    const view = load()
    await waitFor(() => expect(opened()).toHaveLength(1))
    workspace.id = 'ws-2'
    view.again()
    await waitFor(() => expect(rememberedBrief('ws-2', TODAY)).not.toBeNull())
    await act(async () => refuseFirst(refusal(403)))
    await settle()
    expect(briefWarnings(warn)).toEqual([['[brief]', expect.any(String), 403, 'open_out_of_window']])
    expect(opened()).toHaveLength(2)
  })

  it("logs nothing for brief_off (the workspace's proactive mode is off) and never asks for that day again in that load", async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const error = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    postMock.mockRejectedValueOnce(refusal(404, 'brief_off'))
    const view = load()
    await waitFor(() => expect(opened()).toHaveLength(1))
    await settle()
    expect(rememberedBrief('ws-1', TODAY)).toBeNull()
    workspace.id = 'ws-2'
    view.again()
    await waitFor(() => expect(opened()).toHaveLength(2))
    workspace.id = 'ws-1'
    view.again()
    await settle()
    expect(opened()).toHaveLength(2)
    expect(briefWarnings(warn)).toEqual([])
    expect(briefWarnings(error)).toEqual([])
  })

  it('still logs a 404 with any other code', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    postMock.mockRejectedValueOnce(refusal(404, 'open_no_workspace'))
    load()
    await waitFor(() => expect(briefWarnings(warn)).toHaveLength(1))
    expect(briefWarnings(warn)[0]).toEqual(['[brief]', expect.any(String), 404, 'open_no_workspace'])
  })

  it('with storage blocked, still calls once per load', async () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('blocked')
    })
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('blocked')
    })
    const first = load()
    await waitFor(() => expect(opened()).toHaveLength(1))
    workspace.id = 'ws-2'
    first.again()
    await waitFor(() => expect(opened()).toHaveLength(2))
    workspace.id = 'ws-1'
    first.again()
    await settle()
    expect(opened()).toHaveLength(2)
    first.unmount()
    load()
    await waitFor(() => expect(opened()).toHaveLength(3))
  })
})

describe('BriefOnOpen: the answer opens nothing by itself', () => {
  it('draws nothing, navigates nowhere and plays no sound', async () => {
    const play = vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined)
    const speak = vi.fn()
    vi.stubGlobal('speechSynthesis', { speak, cancel: vi.fn() })
    const before = window.location.href
    const view = load()
    await waitFor(() => expect(rememberedBrief('ws-1', TODAY)).not.toBeNull())
    await settle()
    expect(view.container).toBeEmptyDOMElement()
    expect(router.push).not.toHaveBeenCalled()
    expect(router.replace).not.toHaveBeenCalled()
    expect(window.location.href).toBe(before)
    expect(play).not.toHaveBeenCalled()
    expect(speak).not.toHaveBeenCalled()
    expect(postMock.mock.calls.map(([url]) => url)).toEqual([OPEN_PATH])
  })
})

describe('the source rules', () => {
  const frontend = (file: string) => readFileSync(path.resolve(__dirname, '../../..', file), 'utf8')

  it('is mounted once in components/providers.tsx, inside WorkspaceProvider', () => {
    const providers = frontend('components/providers.tsx')
    expect(providers.match(/<BriefOnOpen \/>/g)).toHaveLength(1)
    const mounted = providers.indexOf('<BriefOnOpen />')
    expect(mounted).toBeGreaterThan(providers.indexOf('<WorkspaceProvider>'))
    expect(mounted).toBeLessThan(providers.indexOf('</WorkspaceProvider>'))
    expect(providers).toContain("from './proactive/brief-on-open'")
  })

  it('keeps to tokens, short components and the apiClient', () => {
    const source = proactiveSource('brief-on-open.tsx')
    expect(source).not.toMatch(HEX_COLOUR)
    expect(source).not.toMatch(ORANGE_CLASS)
    expect(source).not.toMatch(/\bfetch\(/)
    componentSpans(source).forEach((span) => expect(span.lines, span.name).toBeLessThanOrEqual(MAX_COMPONENT_LINES))
  })
})
