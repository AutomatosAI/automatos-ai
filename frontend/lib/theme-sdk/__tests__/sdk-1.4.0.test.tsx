/**
 * PRD-268 Wave K (SDK 1.4.0) — what 1.4.0 adds to the proactive and voice areas reaches a theme through
 * '@automatos/theme-sdk':
 *
 * - useRateBrief posts the brief's thumbs to the existing digest feedback route (POST /api/activity/digest/feedback)
 *   with the body that route takes, `{state_hash, rating}`, 1 for up and -1 for down, through the host's own hook; a
 *   theme's input that isn't a state_hash and a rating of up or down throws before anything is sent. Given the brief's
 *   state_hash, its `rating` is the person's own rating of it (GET /api/activity/digest/feedback), in the one cache
 *   entry cards.BriefCard's thumbs read: a press in either shows in both. Without one, nothing is read.
 * - cards.BriefCard takes `thumbs`: false draws no thumbs beside a theme's own; a value that isn't true or false throws.
 * - BriefVoice hands the host's Listen the theme's onSpeakingRow, still takes no text, and is placed with no props as
 *   in 1.3.0; a value that isn't a function throws, naming the prop. The row itself is the host's
 *   (components/voice/__tests__/brief-voice-row.test.tsx).
 */
import { act, cleanup, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const postMock = vi.hoisted(() => vi.fn())
const requestMock = vi.hoisted(() => vi.fn())
const hostListen = vi.hoisted(() => ({ props: [] as Record<string, unknown>[] }))
vi.mock('@/lib/api-client', () => ({
  apiClient: { post: (...args: unknown[]) => postMock(...args), request: (...args: unknown[]) => requestMock(...args) },
}))
vi.mock('@/hooks/use-brief', () => ({ useBrief: () => ({ data: { spoken: 'Morning, Sam. 1. Lisbon flights.' } }) }))
vi.mock('@/components/voice/brief-voice', () => ({
  BriefVoice: (props: Record<string, unknown>) => {
    hostListen.props.push(props)
    return null
  },
}))

import * as sdk from '@automatos/theme-sdk'
import { digestFeedbackQueryKeys } from '@/hooks/use-digest-api'

import * as proactiveArea from '../proactive'
import * as voiceArea from '../voice'

const FEEDBACK_ROUTE = '/api/activity/digest/feedback'
const STATE_HASH = '9f2c41d0a7b3'
const SPOKEN = 'Morning, Sam. 1. Lisbon flights.'
const RATE_BRIEF_KEYS = ['error', 'isRunning', 'rating', 'reset', 'result', 'run', 'runAsync']
const RATING_READ = `${FEEDBACK_ROUTE}?state_hash=${STATE_HASH}`
const BRIEF = {
  v: 1, date: '2026-10-30', greeting: 'Morning, Sam.', prose: 'Two things need you today.', spoken: SPOKEN,
  sections: { needs_you: [], suggestions: [], today: [], since: [], unchecked: [] },
  counts: { needs_you: 0, suggestions: 0, today: 0, since: 0 },
  labels: { sections: {}, more: {}, empty: 'Nothing needs you.' },
  state_hash: STATE_HASH, generated_at: '2026-10-30T07:00:00Z',
} as const satisfies sdk.Brief

function newClient() {
  return new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
}

function wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={newClient()}>{children}</QueryClientProvider>
}

/** Waits until the person's rating has been read into `client`'s cache, so a press never races the read. */
async function ratingRead(client: QueryClient) {
  await waitFor(() => expect(client.getQueryState(digestFeedbackQueryKeys.rating(STATE_HASH))?.status).toBe('success'))
}

/** The person's remembered rating, as the read answers it: 1, -1 or null. */
function rememberedRating(rating: 1 | -1 | null) {
  requestMock.mockImplementation(async (path: string) => {
    if (path === RATING_READ) return { state_hash: STATE_HASH, rating }
    throw new Error(`unexpected request ${path}`)
  })
}

beforeEach(() => {
  postMock.mockReset()
  postMock.mockResolvedValue({ ok: true })
  requestMock.mockReset()
  rememberedRating(null)
  hostListen.props.length = 0
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('SDK 1.4.0 — proactive.useRateBrief', () => {
  it('is the proactive area’s own hook', () => {
    expect(sdk.useRateBrief).toBe(proactiveArea.useRateBrief)
  })

  it('posts thumbs up as rating 1 with the brief’s state_hash, and answers a frozen copy', async () => {
    const { result } = renderHook(() => sdk.useRateBrief(), { wrapper })
    expect(Object.keys(result.current).sort()).toEqual(RATE_BRIEF_KEYS)
    expect(Object.isFrozen(result.current)).toBe(true)
    let answer: sdk.BriefRated | undefined
    await act(async () => {
      answer = await result.current.runAsync({ state_hash: STATE_HASH, rating: 'up' })
    })
    expect(postMock).toHaveBeenCalledTimes(1)
    expect(postMock).toHaveBeenCalledWith(FEEDBACK_ROUTE, { state_hash: STATE_HASH, rating: 1 })
    expect(answer).toEqual({ ok: true })
    expect(Object.isFrozen(answer)).toBe(true)
  })

  it('posts thumbs down as rating -1', async () => {
    const { result } = renderHook(() => sdk.useRateBrief(), { wrapper })
    await act(async () => {
      await result.current.runAsync({ state_hash: STATE_HASH, rating: 'down' })
    })
    expect(postMock).toHaveBeenCalledWith(FEEDBACK_ROUTE, { state_hash: STATE_HASH, rating: -1 })
  })

  it('reports a refused post as its error', async () => {
    postMock.mockRejectedValue(new Error('HTTP 503'))
    const { result } = renderHook(() => sdk.useRateBrief(), { wrapper })
    await act(async () => {
      await expect(result.current.runAsync({ state_hash: STATE_HASH, rating: 'up' })).rejects.toThrow('HTTP 503')
    })
    expect(result.current.error).toBe('HTTP 503')
  })

  it('throws a TypeError, with nothing sent, for no state_hash or a rating that is neither up nor down', async () => {
    const { result } = renderHook(() => sdk.useRateBrief(), { wrapper })
    const refused = [
      { state_hash: '', rating: 'up' },
      { state_hash: '   ', rating: 'down' },
      { state_hash: 42, rating: 'up' },
      { state_hash: STATE_HASH, rating: 1 },
      { state_hash: STATE_HASH, rating: 'sideways' },
      { state_hash: STATE_HASH },
      null,
    ] as unknown as sdk.RateBriefInput[]
    for (const input of refused) {
      expect(() => result.current.run(input)).toThrow(/^Automatos theme SDK: useRateBrief needs /)
      await expect(result.current.runAsync(input)).rejects.toThrow(TypeError)
    }
    expect(postMock).not.toHaveBeenCalled()
  })
})

describe('SDK 1.4.0 — useRateBrief’s rating, shared with cards.BriefCard', () => {
  it('reads nothing without a brief, or with an empty state_hash while it loads: its rating is null', () => {
    for (const stateHash of [undefined, null, '', '  ']) {
      const { result } = renderHook(() => sdk.useRateBrief(stateHash), { wrapper })
      expect(result.current.rating).toBeNull()
    }
    expect(requestMock).not.toHaveBeenCalled()
  })

  it('is the person’s remembered rating of the brief it is given', async () => {
    rememberedRating(-1)
    const { result } = renderHook(() => sdk.useRateBrief(STATE_HASH), { wrapper })
    await waitFor(() => expect(result.current.rating).toBe('down'))
    expect(requestMock).toHaveBeenCalledWith(RATING_READ)
    expect(Object.isFrozen(result.current)).toBe(true)
  })

  it('shows its own press once the server has it', async () => {
    const client = newClient()
    const { result } = renderHook(() => sdk.useRateBrief(STATE_HASH), {
      wrapper: ({ children }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>,
    })
    await ratingRead(client)
    expect(result.current.rating).toBeNull()
    await act(async () => {
      await result.current.runAsync({ state_hash: STATE_HASH, rating: 'up' })
    })
    expect(result.current.rating).toBe('up')
  })

  it('shares one cache entry with BriefCard’s thumbs: a press in either shows in both', async () => {
    const seen = { current: null as sdk.RateBriefAction | null }
    function Both() {
      seen.current = sdk.useRateBrief(STATE_HASH)
      return <sdk.BriefCard brief={BRIEF} listen={false} />
    }
    const first = newClient()
    render(<QueryClientProvider client={first}><Both /></QueryClientProvider>)
    await ratingRead(first)
    fireEvent.click(screen.getByRole('button', { name: 'Not helpful' }))
    await waitFor(() => expect(seen.current?.rating).toBe('down'))
    expect(postMock).toHaveBeenCalledWith(FEEDBACK_ROUTE, { state_hash: STATE_HASH, rating: -1 })
    cleanup()

    const second = newClient()
    render(<QueryClientProvider client={second}><Both /></QueryClientProvider>)
    await ratingRead(second)
    expect(seen.current?.rating).toBeNull()
    await act(async () => {
      await seen.current?.runAsync({ state_hash: STATE_HASH, rating: 'up' })
    })
    expect(screen.getByRole('button', { name: 'Helpful' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: 'Not helpful' })).toBeDisabled()
  })

  it('throws a TypeError for a state_hash that isn’t text, with nothing read', () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    for (const stateHash of [42, {}, true]) {
      expect(() => renderHook(() => sdk.useRateBrief(stateHash as never), { wrapper })).toThrow(
        /^Automatos theme SDK: useRateBrief needs the brief's state_hash to be some text/,
      )
    }
    expect(requestMock).not.toHaveBeenCalled()
  })
})

describe('SDK 1.4.0 — cards.BriefCard thumbs', () => {
  it('draws the thumbs by default, and none with thumbs={false}', async () => {
    const { unmount } = render(<sdk.BriefCard brief={BRIEF} listen={false} />, { wrapper })
    expect(screen.getByRole('group', { name: 'Rate this brief' })).toBeInTheDocument()
    await waitFor(() => expect(requestMock).toHaveBeenCalledWith(RATING_READ))
    unmount()
    requestMock.mockClear()
    render(<sdk.BriefCard brief={BRIEF} listen={false} thumbs={false} />, { wrapper })
    expect(screen.getByRole('article', { name: BRIEF.greeting })).toBeInTheDocument()
    expect(screen.queryByRole('group', { name: 'Rate this brief' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Helpful' })).toBeNull()
    expect(requestMock).not.toHaveBeenCalled()
  })

  it('throws, naming the prop, when thumbs is not true or false', () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    expect(() => render(<sdk.BriefCard brief={BRIEF} thumbs={'no' as never} />, { wrapper })).toThrow(
      /^Automatos theme SDK: BriefCard needs thumbs to be true or false/,
    )
  })
})

describe('SDK 1.4.0 — voice.BriefVoice', () => {
  it('is the voice area’s own component', () => {
    expect(sdk.BriefVoice).toBe(voiceArea.BriefVoice)
  })

  it('is placed with no props, as in 1.3.0: the host’s Listen reads the brief the host holds', () => {
    render(<sdk.BriefVoice />)
    expect(hostListen.props).toEqual([{ spoken: SPOKEN, onSpeakingRow: undefined }])
  })

  it('hands the host’s Listen the theme’s onSpeakingRow, and no text of the theme’s', () => {
    const onSpeakingRow = vi.fn()
    render(<sdk.BriefVoice {...({ onSpeakingRow, spoken: 'Say this instead.' } as sdk.BriefVoiceProps)} />)
    expect(hostListen.props).toHaveLength(1)
    expect(hostListen.props[0].onSpeakingRow).toBe(onSpeakingRow)
    expect(hostListen.props[0].spoken).toBe(SPOKEN)
  })

  it('throws, naming the prop, when onSpeakingRow is not a function', () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    expect(() => render(<sdk.BriefVoice onSpeakingRow={'row' as never} />)).toThrow(
      /^Automatos theme SDK: BriefVoice needs onSpeakingRow to be a function/,
    )
    expect(hostListen.props).toEqual([])
  })
})
