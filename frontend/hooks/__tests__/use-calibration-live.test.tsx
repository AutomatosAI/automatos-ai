/**
 * PRD-268 §6.11.5, K15, K17 (US-W06) — calibration's hooks go live on US-C22b's routes, through apiClient.
 *
 * useCalibration reads GET /api/auto/calibration (with the page's locale) and gives the Calibration@1 and `available`;
 * useAnswerCalibration posts a line's Reject (…/{id}/lines/{line_id}/reject) and Sounds right (…/{id}/accept), and
 * puts the record each answers into the shared view; the host's useCalibrationConsent posts the consent line's answer
 * ({consent, chat_id, locale}) and announces the chat the calibration was posted into. While the server answers
 * available false nothing is posted at all, and an id that isn't a calibration's or a line's never reaches a path.
 * Their exported types are as W2 left them (K17): the SDK's area wraps them unchanged.
 */
import React from 'react'
import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const requestMock = vi.hoisted(() => vi.fn())
const postMock = vi.hoisted(() => vi.fn())

vi.mock('@/lib/api-client', () => ({
  apiClient: {
    request: (...args: unknown[]) => requestMock(...args),
    post: (...args: unknown[]) => postMock(...args),
  },
}))

import {
  CALIBRATION_PATH,
  CALIBRATION_UNAVAILABLE,
  CHAT_CHANGED_EVENT,
  calibrationQueryKeys,
  useAnswerCalibration,
  useCalibration,
  useCalibrationConsent,
  useCalibrationView,
} from '@/hooks/use-calibration'
import { calibrationViewOf, type CalibrationView } from '@/lib/proactive/calibration'

const CAL_ID = '3f2a1b0c-9d8e-4f7a-b6c5-d4e3f2a1b0c9'
const CHAT = '5b7f2c1e-8d4a-4f6b-9c3e-1a2b3c4d5e6f'
const CONSENT_LINE = 'Auto reads the titles of your recent work to say what it knows.'
const READ_PATH = `${CALIBRATION_PATH}?locale=${encodeURIComponent(navigator.language)}`

const CALIBRATION = {
  v: 1,
  id: CAL_ID,
  lines: [
    { id: 'line-1', text: 'You run a bakery in Lisbon.', sources: [{ type: 'deliverable', id: 'd1', label: 'Menu' }] },
    { id: 'line-2', text: 'You send a newsletter on Fridays.', sources: [{ type: 'conversation', id: 'c1', label: 'Friday' }] },
  ],
  offers: ['Draft the newsletter', 'Plan the week', 'Price the menu'],
  empty: false,
}

function view(overrides: Record<string, unknown> = {}) {
  return {
    available: true, offer: false, consent_line: CONSENT_LINE, status: 'open', calibration: CALIBRATION,
    rejected_lines: [], accepted_lines: [], ...overrides,
  }
}

const NOT_OFFERED = { available: false, offer: false, consent_line: CONSENT_LINE, status: null, calibration: null, rejected_lines: [], accepted_lines: [] }

function setup() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  const wrapper = ({ children }: { children: React.ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
  return { client, wrapper }
}

const shown = (client: QueryClient) => client.getQueryData<CalibrationView>(calibrationQueryKeys.mine)

/** The server's view before the answer, then after it (the view is asked again however an answer ends). */
function server(before: unknown, after: unknown) {
  requestMock.mockResolvedValueOnce(before).mockResolvedValue(after)
}

beforeEach(() => {
  requestMock.mockReset()
  postMock.mockReset()
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('useCalibration — GET /api/auto/calibration', () => {
  it("reads the route with the page's locale and gives the Calibration@1 and available", async () => {
    requestMock.mockResolvedValue(view())
    const { wrapper } = setup()
    const { result } = renderHook(() => useCalibration(), { wrapper })

    expect(result.current.isLoading).toBe(true)
    await waitFor(() => expect(result.current.available).toBe(true))
    expect(requestMock).toHaveBeenCalledWith(READ_PATH)
    expect(result.current.data?.id).toBe(CAL_ID)
    expect(result.current.data?.lines.map((line) => line.id)).toEqual(['line-1', 'line-2'])
    expect(result.current).toEqual({ data: result.current.data, available: true, isLoading: false })
    expect(postMock).not.toHaveBeenCalled()
  })

  it('shares one view with the host: the offer, the consent line and what was answered', async () => {
    requestMock.mockResolvedValue(view({ offer: true, status: null, calibration: null }))
    const { wrapper } = setup()
    const { result } = renderHook(() => ({ read: useCalibration(), whole: useCalibrationView() }), { wrapper })

    await waitFor(() => expect(result.current.whole.data?.offer).toBe(true))
    expect(result.current.whole.data?.consent_line).toBe(CONSENT_LINE)
    expect(result.current.read).toEqual({ data: null, available: true, isLoading: false })
    expect(requestMock).toHaveBeenCalledTimes(1)
  })

  it("reads a body that isn't a view as not available, and leaves out a line carrying a percentage", () => {
    expect(calibrationViewOf(null).available).toBe(false)
    expect(calibrationViewOf({ available: 'yes' }).available).toBe(false)
    const read = calibrationViewOf(view({
      calibration: { ...CALIBRATION, lines: [...CALIBRATION.lines, { id: 'line-3', text: 'You are 90% a baker.', sources: [] }] },
      rejected_lines: ['line-1', '../x'],
    }))
    expect(read.calibration?.lines.map((line) => line.id)).toEqual(['line-1', 'line-2'])
    expect(read.rejected_lines).toEqual(['line-1'])
    expect(calibrationViewOf(view({ calibration: { ...CALIBRATION, id: 'cal-1' } })).calibration).toBeNull()
  })
})

describe('useAnswerCalibration — Reject and Sounds right', () => {
  it('posts a line’s Reject to its route and puts the record it answers into the view', async () => {
    server(view(), view({ rejected_lines: ['line-1'] }))
    postMock.mockResolvedValue({ ...view(), rejected_lines: ['line-1'], not_stored: [], note: null })
    const { client, wrapper } = setup()
    const { result } = renderHook(() => useAnswerCalibration(), { wrapper })
    await waitFor(() => expect(result.current.available).toBe(true))

    await act(() => result.current.mutate({ calibration_id: CAL_ID, answer: 'reject', line_id: 'line-1' }))
    expect(postMock).toHaveBeenCalledTimes(1)
    expect(postMock).toHaveBeenCalledWith(`${CALIBRATION_PATH}/${CAL_ID}/lines/line-1/reject`, {})
    expect(shown(client)?.rejected_lines).toEqual(['line-1'])
    await waitFor(() => expect(requestMock).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(shown(client)?.rejected_lines).toEqual(['line-1']))
  })

  it('posts Sounds right to …/accept; the record says answered, and a line kept out of memory brings its note', async () => {
    server(view(), view({ status: 'answered', accepted_lines: ['line-1'] }))
    const note = 'Not kept in memory: it looks like it holds something private.'
    postMock.mockResolvedValue({ ...view(), status: 'answered', accepted_lines: ['line-1'], not_stored: ['line-2'], note })
    const { client, wrapper } = setup()
    const { result } = renderHook(() => useAnswerCalibration(), { wrapper })
    await waitFor(() => expect(result.current.available).toBe(true))

    await act(() => result.current.mutate({ calibration_id: CAL_ID, answer: 'accept' }))
    expect(postMock).toHaveBeenCalledWith(`${CALIBRATION_PATH}/${CAL_ID}/accept`, {})
    expect(shown(client)).toMatchObject({ status: 'answered', accepted_lines: ['line-1'], note })
    // The read after it has no note of its own: the answer's stays.
    await waitFor(() => expect(requestMock).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(shown(client)).toMatchObject({ status: 'answered', note }))
  })

  it("sends nothing for an id that isn't a calibration's or a line's", async () => {
    requestMock.mockResolvedValue(view())
    const { wrapper } = setup()
    const { result } = renderHook(() => useAnswerCalibration(), { wrapper })
    await waitFor(() => expect(result.current.available).toBe(true))

    await expect(result.current.mutate({ calibration_id: '../../admin', answer: 'accept' })).rejects.toThrow()
    await expect(result.current.mutate({ calibration_id: CAL_ID, answer: 'reject', line_id: 'a/b' })).rejects.toThrow()
    expect(postMock).not.toHaveBeenCalled()
  })
})

describe('useCalibrationConsent — the consent line', () => {
  it('posts Go ahead with the chat id and the locale, stops offering, and announces the chat it was posted into', async () => {
    server(view({ offer: true, status: null, calibration: null }), view())
    postMock.mockResolvedValue({ status: 'open', calibration: CALIBRATION, chat_id: CHAT, message_id: CAL_ID })
    const announced = vi.fn()
    window.addEventListener(CHAT_CHANGED_EVENT, announced)
    const { client, wrapper } = setup()
    const { result } = renderHook(() => ({ whole: useCalibrationView(), consent: useCalibrationConsent() }), { wrapper })
    await waitFor(() => expect(result.current.whole.data?.offer).toBe(true))

    await act(() => result.current.consent.mutateAsync({ consent: true, chatId: CHAT }))
    window.removeEventListener(CHAT_CHANGED_EVENT, announced)
    expect(postMock).toHaveBeenCalledTimes(1)
    expect(postMock).toHaveBeenCalledWith(CALIBRATION_PATH, { consent: true, chat_id: CHAT, locale: navigator.language })
    expect(shown(client)).toMatchObject({ offer: false, status: 'open' })
    expect(shown(client)?.calibration?.id).toBe(CAL_ID)
    expect(announced).toHaveBeenCalledTimes(1)
    expect((announced.mock.calls[0][0] as CustomEvent).detail).toEqual({ chat_id: CHAT })
  })

  it("posts Not now as consent false, leaving out a chat id that isn't a chat's", async () => {
    server(view({ offer: true, status: null, calibration: null }), view({ status: 'declined', calibration: null }))
    postMock.mockResolvedValue({ status: 'declined', calibration: null, chat_id: null, message_id: null })
    const { client, wrapper } = setup()
    const { result } = renderHook(() => ({ whole: useCalibrationView(), consent: useCalibrationConsent() }), { wrapper })
    await waitFor(() => expect(result.current.whole.data?.offer).toBe(true))

    await act(() => result.current.consent.mutateAsync({ consent: false, chatId: 'new-chat' }))
    expect(postMock).toHaveBeenCalledWith(CALIBRATION_PATH, { consent: false, locale: navigator.language })
    expect(shown(client)).toMatchObject({ offer: false, status: 'declined' })
  })

  it('takes the record a 409 already_offered carries, and stops offering', async () => {
    server(view({ offer: true, status: null, calibration: null }), view({ status: 'declined', calibration: null }))
    const refused = Object.assign(new Error('Auto has offered you this already.'), {
      status: 409,
      body: { code: 'already_offered', detail: 'Auto has offered you this already.', record: { status: 'declined', calibration: null } },
    })
    postMock.mockRejectedValue(refused)
    const { client, wrapper } = setup()
    const { result } = renderHook(() => ({ whole: useCalibrationView(), consent: useCalibrationConsent() }), { wrapper })
    await waitFor(() => expect(result.current.whole.data?.offer).toBe(true))

    await act(() => expect(result.current.consent.mutateAsync({ consent: true, chatId: CHAT })).rejects.toBe(refused))
    expect(shown(client)).toMatchObject({ offer: false, status: 'declined' })
  })
})

describe('while the server answers available false', () => {
  it('offers nothing and makes no POST: Reject, Sounds right and the consent are refused without a request', async () => {
    requestMock.mockResolvedValue(NOT_OFFERED)
    const { wrapper } = setup()
    const { result } = renderHook(
      () => ({ read: useCalibration(), answer: useAnswerCalibration(), consent: useCalibrationConsent() }),
      { wrapper },
    )
    await waitFor(() => expect(result.current.read.isLoading).toBe(false))

    expect(result.current.read).toEqual({ data: null, available: false, isLoading: false })
    expect(result.current.answer.available).toBe(false)
    await expect(result.current.answer.mutate({ calibration_id: CAL_ID, answer: 'accept' })).rejects.toThrow(CALIBRATION_UNAVAILABLE)
    await expect(
      result.current.answer.mutate({ calibration_id: CAL_ID, answer: 'reject', line_id: 'line-1' }),
    ).rejects.toThrow(CALIBRATION_UNAVAILABLE)
    await act(() => expect(result.current.consent.mutateAsync({ consent: true, chatId: CHAT })).rejects.toThrow(CALIBRATION_UNAVAILABLE))
    expect(postMock).not.toHaveBeenCalled()
    expect(requestMock).toHaveBeenCalled()
    expect(requestMock.mock.calls.every(([path]) => path === READ_PATH)).toBe(true)
  })

  it('reads a failed GET as not available, and still posts nothing', async () => {
    requestMock.mockRejectedValue(Object.assign(new Error('HTTP 503'), { status: 503 }))
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const { wrapper } = setup()
    const { result } = renderHook(() => ({ read: useCalibration(), answer: useAnswerCalibration() }), { wrapper })
    await waitFor(() => expect(result.current.read.isLoading).toBe(false))

    expect(result.current.read.available).toBe(false)
    await expect(result.current.answer.mutate({ calibration_id: CAL_ID, answer: 'accept' })).rejects.toThrow(CALIBRATION_UNAVAILABLE)
    expect(postMock).not.toHaveBeenCalled()
  })
})
