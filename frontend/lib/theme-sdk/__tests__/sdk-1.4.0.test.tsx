/**
 * PRD-268 Wave K (SDK 1.4.0) — what 1.4.0 adds to the proactive and voice areas reaches a theme through
 * '@automatos/theme-sdk':
 *
 * - useRateBrief posts the brief's thumbs to the existing digest feedback route (POST /api/activity/digest/feedback)
 *   with the body that route takes, `{state_hash, rating}`, 1 for up and -1 for down, through the host's own hook; a
 *   theme's input that isn't a state_hash and a rating of up or down throws before anything is sent.
 * - BriefVoice hands the host's Listen the theme's onSpeakingRow, still takes no text, and is placed with no props as
 *   in 1.3.0; a value that isn't a function throws, naming the prop. The row itself is the host's
 *   (components/voice/__tests__/brief-voice-row.test.tsx).
 */
import { act, cleanup, render, renderHook } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const postMock = vi.hoisted(() => vi.fn())
const hostListen = vi.hoisted(() => ({ props: [] as Record<string, unknown>[] }))
vi.mock('@/lib/api-client', () => ({ apiClient: { post: (...args: unknown[]) => postMock(...args), request: vi.fn() } }))
vi.mock('@/hooks/use-brief', () => ({ useBrief: () => ({ data: { spoken: 'Morning, Sam. 1. Lisbon flights.' } }) }))
vi.mock('@/components/voice/brief-voice', () => ({
  BriefVoice: (props: Record<string, unknown>) => {
    hostListen.props.push(props)
    return null
  },
}))

import * as sdk from '@automatos/theme-sdk'

import * as proactiveArea from '../proactive'
import * as voiceArea from '../voice'

const FEEDBACK_ROUTE = '/api/activity/digest/feedback'
const STATE_HASH = '9f2c41d0a7b3'
const SPOKEN = 'Morning, Sam. 1. Lisbon flights.'
const ACTION_VIEW_KEYS = ['error', 'isRunning', 'reset', 'result', 'run', 'runAsync']

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

beforeEach(() => {
  postMock.mockReset()
  postMock.mockResolvedValue({ ok: true })
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
    expect(Object.keys(result.current).sort()).toEqual(ACTION_VIEW_KEYS)
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
